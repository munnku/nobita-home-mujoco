"""かくれんぼ: 子どもが「かくれんぼしよう」と言ったら、アヒルが鬼になって 10 数え、隠れられそうな所を考えて探す。

- 始まり: 見守りカメラのマイクかアヒルのマイクで「かくれんぼ」が聞こえたら (おるすばんモード)
- ずるをしない: 遊びの間は見守りカメラの人の位置を使わない (アヒルの頭のカメラだけで探す)
- 隠れられそうな所 (hiding_spots): 家の地図から、
  1. 背の高い家具 (押入・冷蔵庫・本棚・ソファ…) の脇で、子どもが入れる広さがあり
  2. 部屋の入口あたり (いつもの場所) から、アヒルの低い目では家具に隠れて見えない所
  を選び、家具の種類ごとの見込み (PRIOR、「子どもは押入やソファの陰に隠れやすい」) を掛けて点数にする。
  カメラの無い部屋 (トイレ・脱衣所・浴室) も候補にする
- 探し方: 点数 ÷ (1 + 歩く距離/3 m) の高い順に、その所を見通せる立ち位置 (FloorMap.viewpoint) へ行って探す
  (detectObject)。見つけたら「みーつけた！」
- HIDE_LLM=claude-code / api なら、候補の一覧を LLM にも並べ替えさせる (最終テストと動画用)
"""
import json
import math
import os

from home.vda5050 import action

from .base import LABEL, M, Job, Rule, result_json

TALL = 0.7             # これより高い家具の陰に隠れられる [m]
CHILD_R = 0.18         # 子どもが入れる広さ (家具・壁からの距離) [m]
BEHIND = 0.32          # 家具の辺から外へこれだけ離れた所を隠れ場所とする [m]
PRIOR = {"oshiire": 0.9, "sofa": 0.8, "dining_table": 0.7, "fridge": 0.6, "bookshelf": 0.6, "armchair_n": 0.6,
         "armchair_s": 0.6, "cupboard": 0.5, "washer": 0.5, "kitchen": 0.4, "butsudan": 0.3, "tv_stand": 0.4}
ROOM_PRIOR = {"toilet": 0.5, "senmen": 0.5, "yokushitsu": 0.4}
NOT_SEEN = 1.0         # 入口から見えない所の重み (見える所は SEEN)
SEEN = 0.25
COUNT_TO = 10


def hiding_spots(fm, layout, spots):
    """候補 [{"name", "x", "y", "room", "score", "why"}] を点数の高い順で返す。
    fm: FloorMap (人のいない家)、spots: 部屋 → いつもの場所 (入口あたり) [m]。"""
    out = []
    mod = layout["module"]
    for f in layout["furniture"]:
        if f["level"] != "1F" or f.get("z0", 0) > 0.3:
            continue
        x0, x1, y0, y1 = (v * mod for v in f["rect"])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if f["name"] == "dining_table":
            sides = [("の下", cx, cy)]                       # テーブルの下にもぐる
        elif f["height"] >= TALL:
            sides = [("の東", x1 + BEHIND, cy), ("の西", x0 - BEHIND, cy), ("の北", cx, y1 + BEHIND), ("の南", cx, y0 - BEHIND)]
        else:
            continue
        for side, x, y in sides:
            c = fm.to_cell(x, y)
            if not (0 <= c[0] < fm.w and 0 <= c[1] < fm.h) or not fm.floor[c]:
                continue
            if f["name"] != "dining_table" and fm.dist[c] < CHILD_R:
                continue
            room = fm.room_of(x, y)
            if room is None:
                continue
            seen = True
            if room in spots:
                vi, vj = fm.to_cell(*spots[room])
                seen = fm._sight(vi, vj, c[0], c[1], ignore_near=0.15)
            score = PRIOR.get(f["name"], 0.3) * (SEEN if seen else NOT_SEEN)
            out.append({"name": f"{f['name']}{side}", "x": round(x, 2), "y": round(y, 2), "room": room,
                        "score": round(score, 2), "why": f"{f['name']}{side} ({'入口から見える' if seen else '入口から見えない'})"})
    for room, p in ROOM_PRIOR.items():
        r = next(r for r in layout["rooms"] if r["name"] == room)
        x0, x1, y0, y1 = (v * mod for v in r["rect"])
        out.append({"name": room, "x": round((x0 + x1) / 2, 2), "y": round((y0 + y1) / 2, 2), "room": room,
                    "score": p, "why": f"{LABEL.get(room, room)} (カメラの無い部屋)"})
    return sorted(out, key=lambda s: -s["score"])


class HideSeek(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.state = "idle"
        self.heard = []
        c.m.subscribe("homie/+/audio/heard", lambda t, p: self.heard.append(p) if isinstance(p, dict) else None)
        c.m.subscribe("home/voice", self._on_voice)       # アヒルのそばの声 (アヒルのマイク)
        self.spots = None

    def _on_voice(self, t, v):
        if not isinstance(v, dict) or "x" not in v:
            return
        pos = self.c.robot.get("agvPosition", {})
        if pos and math.hypot(v["x"] - pos.get("x", 1e9), v["y"] - pos.get("y", 1e9)) < 4.0:
            self.heard.append(v)

    def step(self):
        c = self.c
        if c.mode != "alone":
            self.heard.clear()
            return
        while self.heard:
            v = self.heard.pop(0)
            if self.state == "idle" and "かくれんぼ" in v.get("text", ""):
                self._start(v)

    def _start(self, v):
        c = self.c
        self.state = "counting"
        c.game = True
        c.log(f"「{v['text']}」→ かくれんぼを始める (アヒルが鬼。見守りカメラは使わない)", kind="hide_start")
        c.cancel_jobs("patrol")
        job = Job("hideseek", 1)
        job.say("いいよ！アヒルが鬼ね。10 かぞえるよ。", intent="hide_ok")
        for k in range(1, COUNT_TO + 1, 3):
            job.say("、".join(["いーち", "にー", "さーん", "しー", "ごー", "ろーく", "なーな", "はーち", "きゅーう", "じゅう"][k - 1:k + 2]))
        job.say("もういいかい？", intent="hide_ready").listen(4.0)
        job.then(self._plan)
        c.add_job(job)

    def _plan(self, j):
        c = self.c
        if self.spots is None:
            from .predict import DistanceFields, Predict
            pred = next((r for r in c.rules if isinstance(r, Predict) and r.df is not None), None)
            df = pred.df if pred else DistanceFields(os.path.join(c.root, "houses", "nobita"), [])
            spots = {r: (x * M, y * M) for r, (x, y) in c.home.get("everyday_spots", {}).items()}
            spots.setdefault("dk", spots.get("dk_table"))
            self.spots = hiding_spots(df.map, df.lay, spots)
            pos = c.robot.get("agvPosition", {})
            here = (pos.get("x", 0.0), pos.get("y", 0.0))
            # アヒルがその部屋の中で近づけない所は、確かめられないので外す (隠れ場所はもともと見通しが悪いので、
            # 見通せるかどうかでは外さない。近くまで行って首を振って探す)
            ok, skip = [], []
            for s in self.spots:
                r = next(r for r in df.lay["rooms"] if r["name"] == s["room"])
                vp = df.map.viewpoint((s["x"], s["y"]), here, r_min=0.4, r_max=1.5, need_sight=False,
                                      area=[v * M for v in r["rect"]])
                ok.append(s)                     # 外さない (外すと見通しの悪い良い隠れ場所まで消えた)。記録だけ残す
                if not (vp or s["room"] in ROOM_PRIOR):
                    skip.append(s)
            self.spots = ok
            if skip:
                c.log("アヒルが近づきにくい所 (遠くから探す): " + ", ".join(s["name"] for s in skip), kind="hide_skip")
        pos = c.robot.get("agvPosition", {})
        here = (pos.get("x", 0.0), pos.get("y", 0.0))
        order = sorted(self.spots, key=lambda s: -s["score"] / (1 + math.hypot(s["x"] - here[0], s["y"] - here[1]) / 3))
        order = self._llm_order(order) or order
        c.log("隠れていそうな所 (この順に探す): " + ", ".join(f"{s['name']} {s['score']}" for s in order[:6]),
              kind="hide_plan", spots=[s["name"] for s in order])
        self.state = "searching"
        j.order, j.tried = order, 0
        self._next(j)

    def _next(self, j):
        c = self.c
        if j.tried >= len(j.order):
            c.log("見つけられなかった → 降参", kind="hide_giveup")
            j.insert(*Job("tmp", 0).say("こうさん！どこにいるの？出てきてー！").then(self._end).steps)
            return
        s = j.order[j.tried]
        j.tried += 1
        nxt = Job("tmp", 0)
        nxt.go(s["x"], s["y"], 2.0, [action("detectObject", "x-detect", objectType="person", targetX=s["x"],
                                            targetY=s["y"], duration=5.0)])
        nxt.then(lambda j, s=s: self._looked(j, s))
        j.insert(*nxt.steps)

    def _looked(self, j, s):
        c = self.c
        res = result_json(j.last[0] if j.last else None)
        if res.get("found"):
            c.log(f"{s['name']}で見つけた ({j.tried} か所目)", kind="hide_found", spot=s["name"], tries=j.tried,
                  n_spots=len(j.order))
            # 隠れて伏せていた所は「転倒」として扱わない (アヒルが元気なのを見た)
            from .fall import Fall
            for r in c.rules:
                if isinstance(r, Fall):
                    r.handled.append({"x": res.get("x", s["x"]), "y": res.get("y", s["y"]), "time": c.time})
            j.insert(*Job("tmp", 0).say(f"のび太くん、みーつけた！{LABEL.get(s['room'], s['room'])}にいたね。",
                                         intent="hide_found").then(self._end).steps)
        else:
            c.log(f"{s['name']}にはいない", kind="hide_miss", spot=s["name"])
            self._next(j)

    def _end(self, j):
        self.state = "idle"
        self.c.game = False

    def _llm_order(self, order):
        """HIDE_LLM があれば、候補を LLM に並べ替えさせる (失敗したら None)。"""
        llm = os.environ.get("HIDE_LLM", "")
        if llm not in ("claude-code", "api"):
            return None
        from home.judge import ask_llm
        prompt = ("10 歳の子どもが家でかくれんぼをしています。次の隠れ場所の候補を、子どもが隠れていそうな順に並べてください。\n"
                  + json.dumps([{"name": s["name"], "why": s["why"]} for s in order], ensure_ascii=False)
                  + '\n次の JSON だけを返してください: {"order": ["name", ...]}')
        try:
            res = ask_llm(llm, prompt)
            names = res.get("order", [])
            by = {s["name"]: s for s in order}
            new = [by[n] for n in names if n in by]
            return new + [s for s in order if s["name"] not in names]
        except Exception as e:  # noqa: BLE001
            self.c.log(f"LLM の並べ替えに失敗 ({e!r:.80})")
            return None
