"""W3C WoT の Thing Description を読んで、LLM (Claude) が家電とアヒルを道具として使う。

uv run python app/llm_agent.py "居間の様子を見てきて"     # 1 回だけ頼む
uv run python app/llm_agent.py                            # 対話 (空行で終わり)
uv run python app/llm_agent.py --offline "..."            # LLM を使わず簡単な規則で動かす
uv run python app/llm_agent.py --list-tools               # TD から作ったツールを見る

TD → LLM のツールは機械的に作る (ツールを手で書かない):
  TD の action        → ツール <thing>__<action> (input の JSON Schema をそのまま使う)
  TD の書ける property → ツール <thing>__set__<property>
  読める property      → ツール read_property (thing/property を選ぶ)
ツールの実行も TD の forms に書かれた href と method のとおりに HTTP で行う (app/wot_server.py が受ける)。
ANTHROPIC_API_KEY が無いとき、または anthropic が入っていないときは --offline と同じ規則で動く。
"""
import argparse
import importlib.util
import json
import os
import re
import sys

import requests

DEFAULT_MODEL = "claude-sonnet-5-5"
SYSTEM = ("あなたはのび太君の家の見守りロボット (二足歩行のアヒル型ロボット) と家電を操作するアシスタントです。"
          "ツールは W3C WoT の Thing Description から作られています。頼まれたことをツールで実行し、結果を日本語で短く報告してください。"
          "ロボットの移動や確認には時間がかかります。部屋や機器が分からないときは推測せず聞き返してください。"
          "見守りカメラの値 (persons) やロボットの確認結果で、横になっている人 (lying) がいれば、転倒の可能性として伝えてください。")


def safe(name):
    """ツール名に使えない文字 (Claude のツール名は ^[a-zA-Z0-9_-]{1,64}$) を置き換える。"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


def form_of(aff, op):
    for f in aff.get("forms", []):
        ops = f.get("op", [])
        if (op in ops if isinstance(ops, list) else op == ops) and f["href"].startswith("http"):
            return f
    return None


class WoTTools:
    """TD の一覧から LLM のツール定義を作り、ツールの呼び出しを HTTP に直す。"""

    def __init__(self, directory):
        self.tds = {td["id"].split(":")[-1]: td for td in requests.get(directory, timeout=10).json()}
        self.tools, self.calls = [], {}
        readable = []
        for tid, td in self.tds.items():
            for name, a in td.get("actions", {}).items():
                tool = safe(f"{tid}__{name}")
                self.tools.append({"name": tool, "description": f"[{td['title']}] {a.get('title', name)}: "
                                                                f"{a.get('description', '')}",
                                   "input_schema": a.get("input") or {"type": "object", "properties": {}}})
                self.calls[tool] = (form_of(a, "invokeaction"), "POST", None)
            for name, p in td.get("properties", {}).items():
                if form_of(p, "readproperty"):
                    readable.append(f"{tid}/{name}")
                f = form_of(p, "writeproperty")
                if f and not p.get("readOnly"):
                    tool = safe(f"{tid}__set__{name}")
                    schema = {k: v for k, v in p.items() if k in ("type", "enum", "minimum", "maximum")}
                    self.tools.append({"name": tool, "description": f"[{td['title']}] {p.get('title', name)} を設定する",
                                       "input_schema": {"type": "object", "properties": {"value": schema},
                                                        "required": ["value"]}})
                    self.calls[tool] = (f, "PUT", "value")
        self.tools.append({
            "name": "read_property",
            "description": "機器やロボットの状態を読む (robot/state: アヒルの位置といる部屋、cam-*/persons: 見守りカメラが見た人、"
                           "motion-*/motion: 人感センサー、lock-*/locked: 鍵)",
            "input_schema": {"type": "object", "required": ["target"],
                             "properties": {"target": {"type": "string", "enum": readable}}}})

    def call(self, tool, args):
        if tool == "read_property":
            tid, name = args["target"].split("/", 1)
            f = form_of(self.tds[tid]["properties"][name], "readproperty")
            r = requests.get(f["href"], timeout=10)
        else:
            f, method, key = self.calls[tool]
            body = args[key] if key else args
            r = requests.request(f.get("htv:methodName", method), f["href"], json=body, timeout=900)
        try:
            return r.json()
        except ValueError:
            return r.text


def run_llm(tools, request, model, history):
    import anthropic
    client = anthropic.Anthropic()
    history.append({"role": "user", "content": request})
    while True:
        res = client.messages.create(model=model, max_tokens=1024, system=SYSTEM, tools=tools.tools,
                                     messages=history)
        history.append({"role": "assistant", "content": res.content})
        uses = [b for b in res.content if b.type == "tool_use"]
        for b in res.content:
            if b.type == "text" and b.text.strip():
                print(f"LLM: {b.text}")
        if not uses:
            return
        results = []
        for u in uses:
            print(f"  → {u.name} {json.dumps(u.input, ensure_ascii=False)}")
            out = tools.call(u.name, u.input)
            print(f"  ← {json.dumps(out, ensure_ascii=False)[:300]}")
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out, ensure_ascii=False)})
        history.append({"role": "user", "content": results})


def plan_offline(tools, request):
    """LLM が使えないときの簡単な規則。TD の部屋の説明 (id=名前) と機器の名前で言葉を探す。"""
    robot = tools.tds.get("robot", {})
    desc = robot.get("actions", {}).get("goToRoom", {}).get("input", {}).get("properties", {}) \
        .get("room", {}).get("description", "")
    rooms = {}
    for pair in desc.split(":", 1)[-1].split(","):
        if "=" in pair:
            rid, name = pair.strip().split("=", 1)
            rooms[name.split(" ")[0].split("(")[0]] = rid
    room = next((rid for name, rid in sorted(rooms.items(), key=lambda kv: -len(kv[0])) if name in request), None)
    if re.search(r"(戻|帰)", request):
        return [("robot__returnToDock", {})]
    if re.search(r"(止ま|とま|停止|待って)", request):
        return [("robot__startPause", {})]
    if re.search(r"(再開|続け)", request):
        return [("robot__stopPause", {})]
    if re.search(r"(どこ|位置)", request):
        return [("read_property", {"target": "robot/state"})]
    if re.search(r"(照明|電気|ライト)", request) and room:
        on = not re.search(r"(消|オフ)", request)
        for tool in tools.calls:
            if tool.startswith(f"light-{room}__set__"):
                return [(tool, {"value": on})]
    if re.search(r"(鍵|カギ)", request):
        lock = not re.search(r"(開け|あけ)", request)
        return [("lock-genkan__set__locked", {"value": lock})]
    if room and re.search(r"(見|確|調べ|様子)", request):
        return [("robot__checkRoom", {"room": room})]
    if room:
        return [("robot__goToRoom", {"room": room})]
    m = re.search(r"[「\"](.+?)[」\"]", request)
    if m:
        return [("robot__speak", {"text": m.group(1)})]
    return []


def run_offline(tools, request):
    plan = plan_offline(tools, request)
    if not plan:
        print("規則: 分かりませんでした (例: 「居間を見てきて」「戻って」「玄関の照明を点けて」「玄関の鍵を閉めて」)")
        return
    for tool, args in plan:
        print(f"  規則 → {tool} {json.dumps(args, ensure_ascii=False)}")
        print(f"  ← {json.dumps(tools.call(tool, args), ensure_ascii=False)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="*")
    ap.add_argument("--things", default="http://127.0.0.1:8080/things")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--list-tools", action="store_true")
    a = ap.parse_args()
    tools = WoTTools(a.things)
    if a.list_tools:
        print(json.dumps(tools.tools, ensure_ascii=False, indent=1))
        return
    offline = a.offline or not os.environ.get("ANTHROPIC_API_KEY") or importlib.util.find_spec("anthropic") is None
    print(f"ツール {len(tools.tools)} 個 ({len(tools.tds)} 台の TD から) / {'簡単な規則 (LLM なし)' if offline else a.model}")
    history = []

    def handle(text):
        if offline:
            run_offline(tools, text)
        else:
            run_llm(tools, text, a.model, history)

    if a.request:
        handle(" ".join(a.request))
        return
    while True:
        text = input("> ").strip()
        if not text:
            return
        handle(text)


if __name__ == "__main__":
    sys.exit(main())
