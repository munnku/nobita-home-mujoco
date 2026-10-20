"""家の司令塔: IoT 機器 (Homie) の知らせを見て、家電を動かし、アヒルに VDA 5050 の order を出す。

決まりごと (rule) が仕事 (Job) を作り、アヒルは一度に 1 つの仕事をする (優先度の高いものから)。
- 転倒 (fall): 見守りカメラが「横になった人」を FALL_CONFIRM 秒以上見たら、その部屋の照明をつけ、
  アヒルが人を見通せる所へ行って確かめ (detectObject)、家族に知らせて声をかける。立っていれば誤報として取り消す
- 起床 (wake_up): 起きる時刻に布団で横になっていれば、部屋の照明とスピーカーで起こし、アヒルが階段の下から呼ぶ。
  しばらくして起きなければ、もう一度呼んでから、アヒルがママ (1 階で立っている人) のところへ知らせに行く
- 照明 (lights): 人感センサーが LIGHT_OFF_AFTER 秒続けて「人がいない」部屋の照明がついていれば消す (消し忘れ)。
  人が入ってきたら (人感センサーが反応したら) つける
- 玄関 (entrance): 鍵が開いて扉が開いたら、アヒルが玄関を見に行き、人がいれば「おかえりなさい」。
  扉が閉まったまま鍵が開いていれば、しばらくして施錠し、スピーカーで知らせる
時刻はシミュレーションの時刻 (home/time) と家の時計 (home/clock) を使う。
uv run python app/coordinator.py
"""
import json
import math
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from home.mqttc import Mqtt  # noqa: E402
from home.vda5050 import action, make_order, topic  # noqa: E402

HOUSE = os.path.join(ROOT, "houses", "nobita")
M = 0.91
FALL_CONFIRM = 3.0     # 横になった人をこれだけ続けて見たら疑う [s]
LIGHT_OFF_AFTER = 60.0  # 人のいない部屋の照明をこれだけ待って消す [s]
SAME_PLACE = 1.5       # 同じ人とみなす距離 [m] (2 台のカメラの推定は 1 m ほどずれることがある)
LABEL = {"dk": "DK", "ima": "居間", "ousetsu": "応接室", "rouka": "廊下", "nobita": "のび太の部屋", "genkan": "玄関",
         "senmen": "洗面所"}


def secs(clock):
    h, m, s = (int(v) for v in clock.split(":"))
    return h * 3600 + m * 60 + s


class Job:
    """アヒルの仕事: 手順 (行く・話す・関数) を順に行う。"""

    def __init__(self, name, priority):
        self.name, self.priority = name, priority
        self.steps = []
        self.k = 0
        self.sent = False
        self.results = {}

    def go(self, x, y, dev=0.3, actions=()):
        self.steps.append(("go", (x, y, dev, list(actions))))
        return self

    def say(self, text):
        self.steps.append(("say", text))
        return self

    def then(self, fn):
        self.steps.append(("fn", fn))
        return self


class Coordinator:
    def __init__(self):
        with open(os.path.join(HOUSE, "home.yaml"), encoding="utf-8") as f:
            self.home = yaml.safe_load(f)
        self.m = Mqtt("coordinator")
        self.time = 0.0
        self.clock = None
        self.robot = {}
        self.cam_room = {}           # device → room
        self.persons = {}            # room → (時刻, [検出])
        self.locked, self.door_open = {}, {}
        self.lying = []              # 転倒の候補 [{"x","y","room","first","last"}]
        self.handled = []
        self.jobs, self.job = [], None
        self.n = 0
        self.wake = {"state": "idle"}
        self.entrance = {"state": "idle", "closed_since": None}
        s = self.m.subscribe
        s("home/time", lambda t, p: setattr(self, "time", float(p)))
        s("home/clock", lambda t, p: setattr(self, "clock", str(p)))
        s("homie/+/camera/room", lambda t, p: self.cam_room.__setitem__(t.split("/")[1], p))
        s("homie/+/camera/persons", self._on_persons)
        s("homie/+/lock/locked", lambda t, p: self.locked.__setitem__(t.split("/")[1][5:], str(p).lower() == "true"))
        s("homie/+/contact/open", lambda t, p: self.door_open.__setitem__(t.split("/")[1][5:], str(p).lower() == "true"))
        s(topic("state"), lambda t, p: self.robot.update(p))
        self.motion, self.light_on, self.empty_since = {}, {}, {}
        s("homie/+/sensor/motion", lambda t, p: self.motion.__setitem__(t.split("/")[1][7:], str(p).lower() == "true"))
        s("homie/+/light/on", lambda t, p: self.light_on.__setitem__(t.split("/")[1][6:], str(p).lower() == "true"))
        self.requests = []
        s("home/coordinator/requests", lambda t, p: self.requests.append(p))

    # ---- 共通
    def log(self, text, **kw):
        print(f"[coord {self.time:6.1f}] {text}", flush=True)
        self.m.publish("home/log", {"time": round(self.time, 1), "clock": self.clock, "text": text, **kw})

    def homie_set(self, device, node, prop, value):
        self.m.publish(f"homie/{device}/{node}/{prop}/set", value)

    def light(self, room, on=True):
        self.homie_set(f"light-{room.replace('_', '-')}", "light", "on", "true" if on else "false")

    def speaker(self, room, text):
        self.homie_set(f"speaker-{room}", "speaker", "say", text)

    def add_job(self, job):
        self.jobs.append(job)
        self.jobs.sort(key=lambda j: j.priority)
        self.log(f"仕事を受けた: {job.name}")

    def cancel_jobs(self, name):
        """その名前の仕事をやめる (動いていればアヒルに cancelOrder)。"""
        self.jobs = [j for j in self.jobs if j.name != name]
        if self.job and self.job.name == name:
            self.n += 1
            self.m.publish(topic("instantActions"), {"headerId": self.n, "actions": [
                action("cancelOrder", f"cancel-{self.n}")]}, qos=1)
            self.log(f"アヒルの仕事を取り消した: {name}")
            self.job = None

    def _on_persons(self, t, persons):
        room = self.cam_room.get(t.split("/")[1], "?")
        persons = persons if isinstance(persons, list) else []
        self.persons[room] = (self.time, persons)
        for p in persons:
            if p.get("posture") != "lying" or room == self.home["wake_up"]["room"]:
                continue        # 寝室の布団で横になっているのは転倒ではない (起床の決まりで見る)
            where = p.get("room") or room     # 人がいる部屋 (カメラの部屋とは限らない。ふすま越しに見えることがある)
            for c in self.lying:
                if math.hypot(c["x"] - p["x"], c["y"] - p["y"]) < SAME_PLACE:
                    c.update(x=p["x"], y=p["y"], last=self.time)
                    break
            else:
                self.lying.append({"x": p["x"], "y": p["y"], "room": where, "first": self.time, "last": self.time})

    # ---- 毎回
    def step(self):
        self.rule_requests()
        self.rule_fall()
        self.rule_wake_up()
        self.rule_entrance()
        self.rule_lights()
        self.run_jobs()

    # ---- 仕事を進める
    def _action_state(self, aid):
        for s in self.robot.get("actionStates", []):
            if s["actionId"] == aid:
                return s
        return None

    def run_jobs(self):
        if "agvPosition" not in self.robot:
            return          # アヒルの位置がまだ届いていない (シミュレーションの起動中)。届くまで仕事を待たせる
        if self.job is None:
            if not self.jobs:
                return
            self.job = self.jobs.pop(0)
            self.log(f"アヒルの仕事を始める: {self.job.name}")
        j = self.job
        if j.k >= len(j.steps):
            self.log(f"アヒルの仕事が終わった: {j.name}")
            self.job = None
            return
        kind, arg = j.steps[j.k]
        if kind == "fn":
            arg(j)
            j.k += 1
            return
        if not j.sent:
            self.n += 1
            oid = f"{j.name}-{self.n}"
            end = action("wait", f"{oid}-end", duration=0.1)
            if kind == "go":
                x, y, dev, acts = arg
                # アクションの ID は order ごとに一意にする (VDA 5050)
                acts = [{**a, "actionId": f"{oid}-{a['actionId'].split('-', 1)[-1]}"} for a in acts]
                pos = self.robot.get("agvPosition", {})
                self.m.publish(topic("order"), make_order(oid, 0, [(pos.get("x", 0.0), pos.get("y", 0.0), None, 0.3),
                                                                   (x, y, None, dev)], acts + [end]), qos=1)
                j.watch = [a["actionId"] for a in acts] + [end["actionId"]]
            else:
                sp = action("speak", f"{oid}-speak", text=arg)
                self.m.publish(topic("instantActions"), {"headerId": self.n, "actions": [sp]}, qos=1)
                j.watch = [sp["actionId"]]
            j.sent, j.t_sent = True, self.time
            return
        states = [self._action_state(a) for a in j.watch]
        if all(s and s["actionStatus"] in ("FINISHED", "FAILED") for s in states):
            for a, s in zip(j.watch, states):
                j.results[a.rsplit("-", 1)[-1]] = s
            j.last = states
            j.k += 1
            j.sent = False
        elif self.time - j.t_sent > 300:
            self.log(f"時間切れ: {j.name}")
            self.job = None

    # ---- LLM (WoT) などからの依頼
    ROOM_SPOTS = {"ima": (6.0, 3.0), "dk": (2.4, 4.7), "ousetsu": (10.6, 3.3), "rouka": (8.0, 5.0),
                  "senmen": (7.0, 6.7), "genkan": (9.9, 5.0)}

    def rule_requests(self):
        while self.requests:
            r = self.requests.pop(0)
            rid, kind, args = r.get("id"), r.get("type"), r.get("args", {})

            def reply(j, rid=rid, kind=kind):
                out = {"ok": True, "request": kind}
                st = j.last[0] if getattr(j, "last", None) else None
                if kind == "checkRoom" and st:
                    try:
                        out["robot_saw"] = json.loads(st.get("resultDescription") or "{}")
                    except ValueError:
                        out["robot_saw"] = {}
                    room = j.room
                    t, persons = self.persons.get(room, (None, []))
                    if t is not None and self.time - t < 5:
                        out["iot_camera_saw"] = [{"posture": p["posture"], "x": p["x"], "y": p["y"]} for p in persons]
                pos = self.robot.get("agvPosition", {})
                out["robot_position"] = {"x": pos.get("x"), "y": pos.get("y"), "room": self.robot.get("room")}
                self.m.publish(f"home/coordinator/results/{rid}", out, qos=1)

            if kind in ("goToRoom", "checkRoom"):
                room = args.get("room")
                if room not in self.ROOM_SPOTS:
                    self.m.publish(f"home/coordinator/results/{rid}", {"ok": False, "error": f"unknown room {room}"})
                    continue
                x, y = (v * M for v in self.ROOM_SPOTS[room])
                acts = [action("detectObject", "x-detect", objectType="person", sweep="full", duration=12.0)]                     if kind == "checkRoom" else []
                job = Job(kind, 3).go(x, y, 0.6, acts)
                job.room = room
                job.then(reply)
                self.add_job(job)
            elif kind == "returnToDock":
                d = self.home["robot"]["dock"]
                self.add_job(Job(kind, 3).go(d["x"] * M, d["y"] * M, 0.2).then(reply))

    # ---- 転倒
    def rule_fall(self):
        self.lying = [c for c in self.lying if self.time - c["last"] < 3.0]
        busy = any(j.name.startswith("fall") for j in self.jobs + ([self.job] if self.job else []))
        if busy:
            return
        for c in self.lying:
            recent = any(math.hypot(h["x"] - c["x"], h["y"] - c["y"]) < SAME_PLACE and self.time - h["time"] < 600
                         for h in self.handled)
            if c["last"] - c["first"] >= FALL_CONFIRM and not recent:
                self.handled.append({"x": c["x"], "y": c["y"], "time": self.time})
                self.log(f"{LABEL.get(c['room'], c['room'])}のカメラ: 横になった人 ({c['x']:.1f}, {c['y']:.1f}) → 転倒の疑い",
                         kind="fall_suspected", room=c["room"], x=c["x"], y=c["y"])
                self.light(c["room"])
                job = Job("fall", 0).go(c["x"], c["y"], 2.0, [
                    action("detectObject", "x-detect", objectType="person", targetX=c["x"], targetY=c["y"],
                           duration=12.0)])
                job.then(lambda j, c=c: self._fall_result(j, c))
                self.add_job(job)
                break

    def _fall_result(self, j, c):
        st = j.last[0] if j.last else None
        try:
            res = json.loads(st.get("resultDescription") or "{}") if st else {}
        except ValueError:
            res = {}
        if res.get("found") and res.get("posture") == "lying":
            self.log(f"アヒルの目でも横になった人を確認 (score {res.get('score')}) → 家族に知らせる", kind="fall_confirmed")
            self.m.publish("home/alerts", {"type": "fall", "room": c["room"], "x": c["x"], "y": c["y"],
                                           "time": round(self.time, 1), "confirmedBy": "microduck-01",
                                           "elapsed": round(self.time - c["first"], 1)}, qos=1)
            j.say("大丈夫ですか？ご家族に知らせました。")
        elif res.get("found"):
            self.log(f"アヒルが見ると立っていた ({res.get('posture')}) → 誤報として取り消し", kind="false_alarm")
            j.say("お元気そうでよかったです。")
        else:
            self.log("アヒルは人を見つけられなかった → 家族に確認を頼む", kind="fall_unconfirmed")
            self.m.publish("home/alerts", {"type": "fall_unconfirmed", "room": c["room"], "x": c["x"], "y": c["y"],
                                           "time": round(self.time, 1)}, qos=1)
            j.say("どこにいますか？ご家族に知らせます。")

    # ---- 起床
    def _asleep(self):
        cfg = self.home["wake_up"]
        t, persons = self.persons.get(cfg["room"], (None, []))
        if t is None or self.time - t > 3.0:
            return None
        x0, x1, y0, y1 = (v * M for v in cfg["bed"])
        in_bed = [p for p in persons if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1 and p["posture"] == "lying"]
        standing = [p for p in persons if p["posture"] == "standing"]
        if in_bed:
            return True
        return False if standing else None

    def rule_wake_up(self):
        cfg = self.home["wake_up"]
        w = self.wake
        if not self.clock:
            return
        now = secs(self.clock)
        if w["state"] == "idle" and now >= secs(cfg["time"]) and now < secs(cfg["time"]) + 3600:
            if self._asleep():
                self.log(f"{cfg['time'][:5]} になってものび太が布団で寝ている → 起こす", kind="wake_up", room=cfg["room"])
                self.light(cfg["room"])
                self.speaker(cfg["room"], "のび太くん、7時だよ。起きて！")
                cp = cfg["call_point"]
                self.add_job(Job("wake_up", 1).go(cp["x"] * M, cp["y"] * M, 0.25).say("のび太くーん、朝ですよー！"))
                w.update(state="called", t=self.time, tries=1)
            else:
                w["state"] = "done"
        elif w["state"] == "called":
            if self._asleep() is False:
                self.log("のび太が起きた", kind="woke_up", room=cfg["room"])
                self.speaker(cfg["room"], "おはよう！朝ごはんができているよ。")
                w["state"] = "done"
            elif self.time - w["t"] > cfg["retry_after"] and w["tries"] == 1:
                self.log("まだ寝ている → もう一度起こす", room=cfg["room"])
                self.speaker(cfg["room"], "のび太くん！遅刻するよ！")
                cp = cfg["call_point"]
                self.add_job(Job("wake_up", 1).go(cp["x"] * M, cp["y"] * M, 0.25).say("のび太くーん！遅刻しますよー！"))
                w.update(t=self.time, tries=2)
            elif self.time - w["t"] > cfg["retry_after"] and w["tries"] == 2:
                mama = self._someone_standing(exclude_room=cfg["room"])
                if mama:
                    room, p = mama
                    self.log(f"それでも起きない → {LABEL.get(room, room)}にいるママに知らせに行く", kind="tell_family")
                    self.add_job(Job("tell_mama", 1).go(p["x"], p["y"], 2.0, [
                        action("detectObject", "x-detect", objectType="person", targetX=p["x"], targetY=p["y"],
                               duration=6.0)]).say("のび太くんがまだ起きません。起こしてあげてください。"))
                else:
                    self.log("それでも起きない → 家族のスマホに知らせる", kind="tell_family")
                    self.m.publish("home/alerts", {"type": "not_awake", "time": round(self.time, 1)}, qos=1)
                w.update(state="told", t=self.time, tries=3)
        elif w["state"] == "told" and self._asleep() is False:
            self.log("のび太が起きた", kind="woke_up", room=cfg["room"])
            self.speaker(cfg["room"], "おはよう！")
            self.cancel_jobs("tell_mama")       # 知らせに行く途中なら取り消す
            w["state"] = "done"

    def _someone_standing(self, exclude_room=None):
        best = None
        for room, (t, persons) in self.persons.items():
            if room == exclude_room or self.time - t > 3.0:
                continue
            for p in persons:
                if p["posture"] == "standing" and (best is None or room == "dk"):
                    best = (room, p)
        return best

    # ---- 照明の消し忘れ
    def rule_lights(self):
        for room, on in list(self.light_on.items()):
            occupied = self.motion.get(room)
            if occupied and not on:
                self.light(room, True)
                self.light_on[room] = True
                self.log(f"{LABEL.get(room, room)}に人が来た (人感センサー) → 照明をつけた", kind="light_on")
                continue
            if not on or occupied is None or occupied:
                self.empty_since.pop(room, None)
                continue
            since = self.empty_since.setdefault(room, self.time)
            if self.time - since >= LIGHT_OFF_AFTER:
                self.light(room, False)
                self.light_on[room] = False
                self.empty_since.pop(room, None)
                self.log(f"{LABEL.get(room, room)}に {LIGHT_OFF_AFTER:.0f} 秒だれもいない → 照明を消した", kind="light_off")

    # ---- 玄関
    def rule_entrance(self):
        cfg = self.home["entrance"]
        d = cfg["door"]
        e = self.entrance
        if d not in self.locked:
            return
        if self.door_open.get(d) and e["state"] == "idle":
            self.log("玄関の扉が開いた (開閉センサー) → アヒルが見に行く", kind="door_open", room="rouka")
            self.light("genkan")
            tg = cfg["look_target"]
            job = Job("entrance", 2).go(tg["x"] * M, tg["y"] * M, 2.0, [
                action("detectObject", "x-detect", objectType="person", targetX=tg["x"] * M, targetY=tg["y"] * M,
                       duration=8.0)])
            job.then(self._entrance_result)
            self.add_job(job)
            e["state"] = "greeting"
        if e["state"] != "idle" and not self.door_open.get(d) and not self.locked.get(d):
            e["closed_since"] = e["closed_since"] or self.time
            if self.time - e["closed_since"] >= cfg["auto_lock_after"]:
                self.homie_set(f"lock-{d}", "lock", "locked", "true")
                self.speaker("dk", "玄関の鍵を閉めました。")
                self.log(f"玄関が閉まってから {cfg['auto_lock_after']} 秒鍵が開いていた → 施錠した", kind="auto_lock")
                e.update(state="idle", closed_since=None)
        else:
            e["closed_since"] = None

    def _entrance_result(self, j):
        st = j.last[0] if j.last else None
        try:
            res = json.loads(st.get("resultDescription") or "{}") if st else {}
        except ValueError:
            res = {}
        if res.get("found"):
            self.log("玄関に人がいた → 迎える", kind="greet")
            j.say("おかえりなさい！手を洗ってね。")
        else:
            self.log("玄関に人が見えない", kind="no_one")
            j.say("どなたですか？")


def main():
    c = Coordinator()
    c.log("司令塔を起動")
    while True:
        c.step()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
