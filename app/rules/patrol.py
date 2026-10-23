"""見回り: 床に落ちている小さい危ない物 (ボタン電池・硬貨・ビー玉・小さいブロック・ハサミ) を、アヒルの頭のカメラで探す。

天井の見守りカメラからは小さすぎて見えない (1〜2 cm の物は数画素) ので、低い所から近くで見るアヒルの役目。
- おるすばんモードで、アヒルの手が空いていて、前の見回りから every 秒たったら見回る (WoT の依頼 patrol でも)
- 部屋ごとの見る所 (home.yaml の patrol.points) を順に回り、首を下に向けてその場で 1 周しながら探す
  (VDA 5050 の detectObject、objectType=hazard。結果は物の種類と床の上の位置)
- 見つけた物は home/hazards (retain) に出し、保護者に知らせる。子どもが近く (CHILD_NEAR) にいれば、
  その部屋のスピーカーと、アヒルが「さわらないでね」と声をかける
- 片付け: 拾うロボット (Stretch 3、VDA 5050 の stretch-01) がいれば、見つけた順に order を出す
  (物の所で pick → 片付けの箱で drop → 待機場所へ戻る)。アヒルとぶつからないよう、見回りが終わってから始める。
  いなければ保護者に片付けを頼む
"""
import math

from home.vda5050 import STRETCH_SERIAL, action, make_order, topic

from .base import LABEL, M, Job, Rule, result_json

SAME_OBJECT = 0.2      # 同じ物とみなす距離 [m]
CHILD_NEAR = 1.5       # 子どもがこれより近くにいれば声をかける [m]
KIND_LABEL = {"button_battery": "ボタン電池", "coin": "硬貨", "marble": "ビー玉", "toy_block": "小さいブロック",
              "scissors": "ハサミ"}


class Patrol(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.cfg = c.home.get("patrol", {})
        self.last = -1e9 if self.cfg.get("at_start") else 0.0
        self.found = []           # [{"kind", "x", "y", "room", "time", "score", "state": found/picked}]
        self.requests = []
        c.m.subscribe("home/patrol/request", lambda t, p: self.requests.append(p))
        # 片付け係 (Stretch 3)
        self.st_state = {}
        self.has_stretch = False
        c.m.subscribe(topic("state", serial=STRETCH_SERIAL), lambda t, p: self.st_state.update(p))
        c.m.subscribe(topic("factsheet", serial=STRETCH_SERIAL), lambda t, p: setattr(self, "has_stretch", True))
        self.tidy = None          # 片付け中の {"obj", "stage": pick/drop, "aid"}
        self.n = 0

    def step(self):
        c = self.c
        if not self.cfg:
            return
        self._step_tidy()
        want = bool(self.requests)
        if c.mode == "alone" and c.time - self.last >= self.cfg.get("every", 300):
            want = want or (c.job is None and not c.jobs)
        if want and not c.busy("patrol"):
            self.requests.clear()
            self._start()

    def _start(self):
        c = self.c
        self.last = c.time
        pts = self.cfg["points"]
        c.log(f"見回りを始める ({len(pts)} か所、床の小さい物を探す)", kind="patrol_start")
        job = Job("patrol", 2)
        job.interruptible = True
        for p in pts:
            x, y = p[0] * M, p[1] * M
            job.go(x, y, 0.3, [action("detectObject", "x-detect", objectType="hazard", sweep="full",
                                      duration=float(self.cfg.get("sweep", 11.0)))])
            job.then(lambda j, p=p: self._result(j, p))
        job.then(self._done)
        c.add_job(job)

    def _result(self, j, p):
        c = self.c
        res = result_json(j.last[0] if j.last else None)
        for o in res.get("objects", []):
            same = next((f for f in self.found if f["kind"] == o["kind"]
                         and math.hypot(f["x"] - o["x"], f["y"] - o["y"]) < SAME_OBJECT), None)
            if same:
                same.update(score=max(same["score"], o["score"]))
                continue
            room = o.get("room") or "?"
            f = {**o, "room": room, "time": round(c.time, 1), "state": "found"}
            self.found.append(f)
            what, where = KIND_LABEL.get(o["kind"], o["kind"]), LABEL.get(room, room)
            c.log(f"{where}の床に{what}を見つけた ({o['x']:.2f}, {o['y']:.2f}、score {o['score']:.2f})",
                  kind="hazard_found", hazard=o["kind"], room=room, x=o["x"], y=o["y"])
            todo = "片付けロボットが片付けます。" if self.has_stretch else "小さい子が口に入れると危ないので、片付けてください。"
            c.m.publish("home/alerts", {"type": "small_object", "level": "info", "room": room, "object": o["kind"],
                                        "x": o["x"], "y": o["y"], "time": round(c.time, 1),
                                        "summary": f"{where}の床に{what}が落ちています。{todo}",
                                        "evidence": o.get("image")}, qos=1)
            child = self._child_near(o)
            if child:
                if child in ("ima", "dk", "rouka", "nobita"):        # スピーカーのある部屋
                    c.speaker(child, f"のび太くん、床の{what}はさわらないでね。", intent="warn")
                j.insert(*Job("tmp", 0).say(f"のび太くん、その{what}はさわらないでね。", intent="warn").steps)
        c.m.publish("home/hazards", [f for f in self.found if f["state"] == "found"], retain=True)

    # ---- 片付け (Stretch 3 に VDA 5050 の order を出す)
    def _st_action(self, aid):
        return next((s for s in self.st_state.get("actionStates", []) if s["actionId"] == aid), None)

    def _send(self, obj, stage):
        c = self.c
        self.n += 1
        oid = f"tidy-{self.n}"
        if stage == "pick":
            a = action("pick", f"{oid}-pick", x=obj["x"], y=obj["y"], objectType=obj["kind"])
            goal = (obj["x"], obj["y"])
        elif stage == "home":
            a = action("wait", f"{oid}-home", duration=0.1)
            d = c.home["stretch"]["dock"]
            goal = (d["x"] * M, d["y"] * M)
        else:
            a = action("drop", f"{oid}-drop")
            goal = tuple(v * M for v in c.home["stretch"]["box"]) if "box" in c.home["stretch"] else (obj["x"], obj["y"])
        pos = self.st_state.get("agvPosition", {})
        c.m.publish(topic("order", serial=STRETCH_SERIAL),
                    make_order(oid, 0, [(pos.get("x", 0.0), pos.get("y", 0.0), None, 0.3), (goal[0], goal[1], None, 0.5)],
                               [a], serial=STRETCH_SERIAL), qos=1)
        self.tidy = {"obj": obj, "stage": stage, "aid": a["actionId"], "t": c.time}

    def _step_tidy(self):
        c = self.c
        if not self.has_stretch or "agvPosition" not in self.st_state:
            return
        if self.tidy is None:
            todo = next((f for f in self.found if f["state"] == "found"), None)
            if todo and c.busy("patrol"):
                return          # アヒルが見回っている間は待つ (同じ部屋で 2 台がぶつからないように)
            if todo:
                todo["state"] = "tidying"
                what, where = KIND_LABEL.get(todo["kind"], todo["kind"]), LABEL.get(todo["room"], todo["room"])
                c.log(f"片付け係 (Stretch 3) に{where}の{what}を拾ってもらう", kind="tidy_start", hazard=todo["kind"])
                self._send(todo, "pick")
            return
        t = self.tidy
        st = self._st_action(t["aid"])
        if st is None or st["actionStatus"] not in ("FINISHED", "FAILED"):
            if c.time - t["t"] > 300:
                st = {"actionStatus": "FAILED", "resultDescription": "時間切れ"}
            else:
                return
        obj = t["obj"]
        what, where = KIND_LABEL.get(obj["kind"], obj["kind"]), LABEL.get(obj["room"], obj["room"])
        if t["stage"] == "home":
            self.tidy = None
            return
        if st["actionStatus"] == "FINISHED" and t["stage"] == "pick":
            c.log(f"Stretch が{what}を拾った → 片付けの箱へ", kind="tidy_picked", hazard=obj["kind"])
            self._send(obj, "drop")
        elif st["actionStatus"] == "FINISHED":
            obj["state"] = "picked"
            c.log(f"{where}の{what}を片付けた", kind="tidy_done", hazard=obj["kind"])
            c.m.publish("home/alerts", {"type": "small_object_removed", "level": "info", "room": obj["room"],
                                        "object": obj["kind"], "time": round(c.time, 1),
                                        "summary": f"{where}の床に落ちていた{what}を、片付けロボットが箱に片付けました。"}, qos=1)
            self._after(obj)
        else:
            obj["state"] = "failed"
            c.log(f"{what}を片付けられなかった ({st.get('resultDescription')}) → 保護者に頼む", kind="tidy_failed",
                  hazard=obj["kind"])
            c.m.publish("home/alerts", {"type": "small_object", "level": "warn", "room": obj["room"],
                                        "object": obj["kind"], "time": round(c.time, 1),
                                        "summary": f"{where}の床の{what}を片付けロボットが拾えませんでした。片付けてください。"}, qos=1)
            self._after(obj)
        c.m.publish("home/hazards", [f for f in self.found if f["state"] in ("found", "tidying")], retain=True)

    def _after(self, obj):
        """次の物があれば続ける。なければ待機場所へ戻る。"""
        if any(f["state"] == "found" for f in self.found):
            self.tidy = None
        else:
            self._send(obj, "home")

    def _child_near(self, o):
        """見守りカメラで、子どもが物の近くにいるか (いれば部屋の名前)。"""
        c = self.c
        for room, (t, persons) in c.persons.items():
            if c.time - t > 3.0:
                continue
            for p in persons:
                if math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < CHILD_NEAR:
                    return room
        return None

    def _done(self, j):
        c = self.c
        n = sum(1 for f in self.found if f["state"] == "found")
        c.log(f"見回りが終わった (床の小さい物: {n} 個)", kind="patrol_done", count=n)
        d = c.home["robot"]["dock"]
        j.go(d["x"] * M, d["y"] * M, 0.3)
