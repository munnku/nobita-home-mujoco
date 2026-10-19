"""W3C WoT の Thing Description の一覧から LLM の道具 (ツール) を機械的に作り、道具の呼び出しを HTTP に直す。

LLM が Claude API でも (app/llm_agent.py)、Claude Code の MCP でも (app/wot_mcp.py)、
このセッションのコマンドでも (app/wot_cli.py) 同じ道具を使う。
  TD の action        → ツール <thing>__<action> (input の JSON Schema をそのまま使う)
  TD の書ける property → ツール <thing>__set__<property>
  読める property      → ツール read_property (thing/property を選ぶ)
ツールの実行は TD の forms に書かれた href と method のとおりに HTTP で行う (app/wot_server.py が受ける)。
"""
import re

import requests

THINGS = "http://127.0.0.1:8080/things"
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
