"""起床: 起きる時刻に布団で横になっていれば、部屋の照明とスピーカーで起こし、アヒルが階段の下から呼ぶ。
しばらくして起きなければ、もう一度呼んでから、アヒルがママ (1 階で立っている人) のところへ知らせに行く。"""
from home.vda5050 import action

from .base import LABEL, M, Job, Rule, secs


class WakeUp(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.cfg = c.home["wake_up"]
        self.w = {"state": "idle"}

    def _asleep(self):
        c, cfg = self.c, self.cfg
        t, persons = c.persons.get(cfg["room"], (None, []))
        if t is None or c.time - t > 3.0:
            return None
        x0, x1, y0, y1 = (v * M for v in cfg["bed"])
        in_bed = [p for p in persons if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1 and p["posture"] == "lying"]
        standing = [p for p in persons if p["posture"] == "standing"]
        if in_bed:
            return True
        return False if standing else None

    def step(self):
        c, cfg, w = self.c, self.cfg, self.w
        if not c.clock:
            return
        now = secs(c.clock)
        if w["state"] == "idle" and now >= secs(cfg["time"]) and now < secs(cfg["time"]) + 3600:
            if self._asleep():
                c.log(f"{cfg['time'][:5]} になってものび太が布団で寝ている → 起こす", kind="wake_up", room=cfg["room"])
                c.light(cfg["room"])
                c.speaker(cfg["room"], "のび太くん、7時だよ。起きて！")
                cp = cfg["call_point"]
                c.add_job(Job("wake_up", 1).go(cp["x"] * M, cp["y"] * M, 0.25).say("のび太くーん、朝ですよー！"))
                w.update(state="called", t=c.time, tries=1)
            else:
                w["state"] = "done"
        elif w["state"] == "called":
            if self._asleep() is False:
                c.log("のび太が起きた", kind="woke_up", room=cfg["room"])
                c.speaker(cfg["room"], "おはよう！朝ごはんができているよ。")
                w["state"] = "done"
            elif c.time - w["t"] > cfg["retry_after"] and w["tries"] == 1:
                c.log("まだ寝ている → もう一度起こす", room=cfg["room"])
                c.speaker(cfg["room"], "のび太くん！遅刻するよ！")
                cp = cfg["call_point"]
                c.add_job(Job("wake_up", 1).go(cp["x"] * M, cp["y"] * M, 0.25).say("のび太くーん！遅刻しますよー！"))
                w.update(t=c.time, tries=2)
            elif c.time - w["t"] > cfg["retry_after"] and w["tries"] == 2:
                mama = self._someone_standing(exclude_room=cfg["room"])
                if mama:
                    room, p = mama
                    c.log(f"それでも起きない → {LABEL.get(room, room)}にいるママに知らせに行く", kind="tell_family")
                    c.add_job(Job("tell_mama", 1).go(p["x"], p["y"], 2.0, [
                        action("detectObject", "x-detect", objectType="person", targetX=p["x"], targetY=p["y"],
                               duration=6.0)]).say("のび太くんがまだ起きません。起こしてあげてください。"))
                else:
                    c.log("それでも起きない → 家族のスマホに知らせる", kind="tell_family")
                    c.m.publish("home/alerts", {"type": "not_awake", "time": round(c.time, 1)}, qos=1)
                w.update(state="told", t=c.time, tries=3)
        elif w["state"] == "told" and self._asleep() is False:
            c.log("のび太が起きた", kind="woke_up", room=cfg["room"])
            c.speaker(cfg["room"], "おはよう！")
            c.cancel_jobs("tell_mama")       # 知らせに行く途中なら取り消す
            w["state"] = "done"

    def _someone_standing(self, exclude_room=None):
        c = self.c
        best = None
        for room, (t, persons) in c.persons.items():
            if room == exclude_room or c.time - t > 3.0:
                continue
            for p in persons:
                if p["posture"] == "standing" and (best is None or room == "dk"):
                    best = (room, p)
        return best
