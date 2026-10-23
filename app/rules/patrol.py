"""見回り: 床に落ちている小さい危ない物 (ボタン電池・硬貨・ビー玉・小さいブロック・ハサミ) を、アヒルの頭のカメラで探す。

天井の見守りカメラからは小さすぎて見えない (1〜2 cm の物は数画素) ので、低い所から近くで見るアヒルの役目。
- おるすばんモードで、アヒルの手が空いていて、前の見回りから every 秒たったら見回る (WoT の依頼 patrol でも)
- 部屋ごとの見る所 (home.yaml の patrol.points) を順に回り、首を下に向けてその場で 1 周しながら探す
  (VDA 5050 の detectObject、objectType=hazard。結果は物の種類と床の上の位置)
- 見つけた物は home/hazards (retain) に出し、保護者に知らせる。子どもが近く (CHILD_NEAR) にいれば、
  その部屋のスピーカーと、アヒルが「さわらないでね」と声をかける
- 片付け: 拾うロボット (Stretch 3、VDA 5050 の stretch-01) がいれば頼む。いなければ保護者に片付けを頼む
"""
import math

from home.vda5050 import action

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

    def step(self):
        c = self.c
        if not self.cfg:
            return
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
            c.m.publish("home/alerts", {"type": "small_object", "level": "info", "room": room, "object": o["kind"],
                                        "x": o["x"], "y": o["y"], "time": round(c.time, 1),
                                        "summary": f"{where}の床に{what}が落ちています。小さい子が口に入れると危ないので、"
                                                   "片付けてください。", "evidence": o.get("image")}, qos=1)
            child = self._child_near(o)
            if child:
                if child in ("ima", "dk", "rouka", "nobita"):        # スピーカーのある部屋
                    c.speaker(child, f"のび太くん、床の{what}はさわらないでね。", intent="warn")
                j.insert(*Job("tmp", 0).say(f"のび太くん、その{what}はさわらないでね。", intent="warn").steps)
        c.m.publish("home/hazards", [f for f in self.found if f["state"] == "found"], retain=True)

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
