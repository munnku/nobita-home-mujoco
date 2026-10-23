"""玄関: 鍵が開いて扉が開いたら、アヒルが玄関を見に行き、人がいれば「おかえりなさい」。
扉が閉まったまま鍵が開いていれば、しばらくして施錠し、スピーカーで知らせる。"""
from home.vda5050 import action

from .base import M, Job, Rule, result_json


class Entrance(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.cfg = c.home["entrance"]
        self.e = {"state": "idle", "closed_since": None}

    def step(self):
        c, cfg, e = self.c, self.cfg, self.e
        d = cfg["door"]
        if d not in c.locked:
            return
        if c.door_open.get(d) and e["state"] == "idle":
            c.log("玄関の扉が開いた (開閉センサー) → アヒルが見に行く", kind="door_open", room="rouka")
            c.light("genkan")
            tg = cfg["look_target"]
            job = Job("entrance", 2).go(tg["x"] * M, tg["y"] * M, 2.0, [
                action("detectObject", "x-detect", objectType="person", targetX=tg["x"] * M, targetY=tg["y"] * M,
                       duration=8.0)])
            job.then(self._result)
            c.add_job(job)
            e["state"] = "greeting"
        if e["state"] != "idle" and not c.door_open.get(d) and not c.locked.get(d):
            e["closed_since"] = e["closed_since"] or c.time
            if c.time - e["closed_since"] >= cfg["auto_lock_after"]:
                c.homie_set(f"lock-{d}", "lock", "locked", "true")
                c.speaker("dk", "玄関の鍵を閉めました。")
                c.log(f"玄関が閉まってから {cfg['auto_lock_after']} 秒鍵が開いていた → 施錠した", kind="auto_lock")
                e.update(state="idle", closed_since=None)
        else:
            e["closed_since"] = None

    def _result(self, j):
        res = result_json(j.last[0] if j.last else None)
        if res.get("found"):
            self.c.log("玄関に人がいた → 迎える", kind="greet")
            j.say("おかえりなさい！手を洗ってね。")
        else:
            self.c.log("玄関に人が見えない", kind="no_one")
            j.say("どなたですか？")
