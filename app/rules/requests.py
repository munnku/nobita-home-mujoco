"""LLM (WoT) などからの依頼: goToRoom / checkRoom / returnToDock。
依頼は home/coordinator/requests に来て、結果を home/coordinator/results/<id> に返す。"""
from home.vda5050 import action

from .base import M, Job, Rule, result_json

ROOM_SPOTS = {"ima": (6.0, 3.0), "dk": (2.4, 4.7), "ousetsu": (10.6, 3.3), "rouka": (8.0, 5.0),
              "senmen": (7.0, 6.7), "genkan": (9.9, 5.0)}


class Requests(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.queue = []
        c.m.subscribe("home/coordinator/requests", lambda t, p: self.queue.append(p))

    def step(self):
        c = self.c
        while self.queue:
            r = self.queue.pop(0)
            rid, kind, args = r.get("id"), r.get("type"), r.get("args", {})

            def reply(j, rid=rid, kind=kind):
                out = {"ok": True, "request": kind}
                st = j.last[0] if getattr(j, "last", None) else None
                if kind == "checkRoom" and st:
                    out["robot_saw"] = result_json(st)
                    t, persons = c.persons.get(j.room, (None, []))
                    if t is not None and c.time - t < 5:
                        out["iot_camera_saw"] = [{"posture": p["posture"], "x": p["x"], "y": p["y"]} for p in persons]
                pos = c.robot.get("agvPosition", {})
                out["robot_position"] = {"x": pos.get("x"), "y": pos.get("y"), "room": c.robot.get("room")}
                c.m.publish(f"home/coordinator/results/{rid}", out, qos=1)

            if kind in ("goToRoom", "checkRoom"):
                room = args.get("room")
                if room not in ROOM_SPOTS:
                    c.m.publish(f"home/coordinator/results/{rid}", {"ok": False, "error": f"unknown room {room}"})
                    continue
                x, y = (v * M for v in ROOM_SPOTS[room])
                acts = [action("detectObject", "x-detect", objectType="person", sweep="full", duration=12.0)] \
                    if kind == "checkRoom" else []
                job = Job(kind, 3).go(x, y, 0.6, acts)
                job.interruptible = True
                job.room = room
                job.then(reply)
                c.add_job(job)
            elif kind == "returnToDock":
                d = c.home["robot"]["dock"]
                c.add_job(Job(kind, 3).go(d["x"] * M, d["y"] * M, 0.2).then(reply))
