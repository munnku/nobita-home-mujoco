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
        c = self.c
        res = result_json(j.last[0] if j.last else None)
        if not res.get("found"):
            c.log("玄関に人が見えない", kind="no_one")
            j.say("どなたですか？")
            return
        # だれかはアヒルの目の YOLO (who: child / adult / stranger) で見分ける。見分けない重みのときは、
        # だれもいない家 (away) に鍵を開けて入ってきたのは子ども、家族がいるときは家の人とみなす
        who = res.get("who") or ("child" if c.mode == "away" else "adult")
        if who == "stranger":
            c.log("玄関に知らない大人がいる → 声をかけ、保護者に知らせる", kind="stranger", who=who)
            j.say("どなたですか？ご家族に知らせます。", intent="ask_who")
            c.speaker("rouka", "どなたですか？家の人に知らせました。")
            c.m.publish("home/alerts", {"type": "stranger", "level": "emergency", "room": "genkan", "time": round(c.time, 1),
                                        "summary": "玄関に知らない人が入ってきました。すぐに確かめてください。",
                                        "evidence": res.get("image")}, qos=1)
            return
        if who == "child":
            c.log("玄関に子どもがいた → 迎える", kind="greet", who=who)
            j.say(f"{self.cfg.get('child_name', 'のび太')}くん、おかえり！手を洗ってね。", intent="greet")
            d = c.home["robot"]["dock"]
            j.go(d["x"] * M, d["y"] * M, 0.3)       # 先に居間へ戻る (廊下で子どもの通り道をふさがない)
            if c.mode == "away":
                # 子どもが帰ってきたが、家の大人は出かけている: おるすばんモードにする
                c.m.publish("home/mode", "alone", retain=True)
                c.log("のび太が帰ってきた。ママは出かけているので、おるすばんモードにする", kind="alone_start")
        else:
            c.log("玄関に家の人がいた → 迎える", kind="greet", who=who)
            j.say("おかえりなさい！")
