"""家の司令塔: IoT 機器 (Homie) の知らせを見て、家電を動かし、アヒルに VDA 5050 の order を出す。

決まりごと (app/rules/) が仕事 (Job) を作り、アヒルは一度に 1 つの仕事をする (優先度の高いものから)。
- 依頼 (requests.py): LLM (WoT) からの goToRoom / checkRoom / returnToDock
- 転倒 (fall.py): 見守りカメラが横になった人を見たら、アヒルが見通せる所へ行って確かめ、家族に知らせる
- 起床 (wake_up.py): 起きる時刻に布団で寝ていれば起こし、起きなければママに知らせに行く
- 玄関 (entrance.py): 扉が開いたら迎えに行き、閉め忘れた鍵は施錠する
- 照明 (lights.py): だれもいない部屋の照明を消し、人が来たらつける
- カメラの無い部屋 (blind_spot.py): 脱衣所・トイレに長くいれば、アヒルが扉の前で声をかけ、返事がなければ確かめる
- 見回り (patrol.py): 床に落ちている小さい危ない物をアヒルの頭のカメラで探し、知らせる
- かくれんぼ (hide_seek.py): 子どもに頼まれたらアヒルが鬼になり、隠れられそうな所を考えて探す
- 先読み (predict.py): おるすばん中の子どもが危ない場所へ向かうのを先読みし、警告してアヒルが先回りする
時刻はシミュレーションの時刻 (home/time) と家の時計 (home/clock) を使う。
uv run python app/coordinator.py
"""
import json
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.rules.base import M  # noqa: E402
from app.rules.blind_spot import BlindSpot  # noqa: E402
from app.rules.entrance import Entrance  # noqa: E402
from app.rules.fall import Fall  # noqa: E402
from app.rules.hide_seek import HideSeek  # noqa: E402
from app.rules.lights import Lights  # noqa: E402
from app.rules.patrol import Patrol  # noqa: E402
from app.rules.predict import Predict  # noqa: E402
from app.rules.requests import ROOM_SPOTS, Requests  # noqa: E402
from app.rules.wake_up import WakeUp  # noqa: E402
from home.mqttc import Mqtt  # noqa: E402
from home.vda5050 import action, make_order, topic  # noqa: E402

HOUSE = os.path.join(ROOT, "houses", "home")
JOB_TIMEOUT = 300.0    # 1 つの手順にこれ以上かかったら仕事をやめる [s]


class Coordinator:
    def __init__(self):
        with open(os.path.join(HOUSE, "home.yaml"), encoding="utf-8") as f:
            self.home = yaml.safe_load(f)
        with open(os.path.join(HOUSE, "layout.yaml"), encoding="utf-8") as f:
            self.levels = {r["name"]: r["level"] for r in yaml.safe_load(f)["rooms"]}
        self.root = ROOT
        self.mode = "family"         # おるすばんモードなら "alone" (home/mode)
        self.m = Mqtt("coordinator")
        self.time = 0.0
        self.clock = None
        self.robot = {}
        self.cam_room = {}           # device → room
        self.persons = {}            # room → (時刻, [検出])
        self.locked, self.door_open = {}, {}
        self.motion, self.light_on = {}, {}
        self.jobs, self.job = [], None
        self.game = False            # かくれんぼの最中 (隠れて横になるのは転倒ではない。hide_seek.py)
        self.attended = {}           # 部屋 → アヒルが人の様子を確かめて知らせた時刻 (決まりごとどうしで重ねて知らせない)
        self.n = 0
        s = self.m.subscribe
        s("home/time", lambda t, p: setattr(self, "time", float(p)))
        s("home/clock", lambda t, p: setattr(self, "clock", str(p)))
        s("home/mode", self._on_mode)
        s("homie/+/camera/room", lambda t, p: self.cam_room.__setitem__(t.split("/")[1], p))
        s("homie/+/camera/persons", self._on_persons)
        s("homie/+/lock/locked", lambda t, p: self.locked.__setitem__(t.split("/")[1][5:], str(p).lower() == "true"))
        s("homie/+/contact/open", lambda t, p: self.door_open.__setitem__(t.split("/")[1][5:], str(p).lower() == "true"))
        s("homie/+/sensor/motion", lambda t, p: self.motion.__setitem__(t.split("/")[1][7:], str(p).lower() == "true"))
        s("homie/+/light/on", lambda t, p: self.light_on.__setitem__(t.split("/")[1][6:], str(p).lower() == "true"))
        s(topic("state"), self._on_robot_state)
        # 決まりごと (この順に毎回見る)
        self.rules = [Requests(self), Fall(self), Predict(self), BlindSpot(self), Patrol(self), HideSeek(self), WakeUp(self), Entrance(self), Lights(self)]

    # ---- 共通
    def log(self, text, **kw):
        print(f"[coord {self.time:6.1f}] {text}", flush=True)
        self.m.publish("home/log", {"time": round(self.time, 1), "clock": self.clock, "text": text, **kw})

    def homie_set(self, device, node, prop, value):
        self.m.publish(f"homie/{device}/{node}/{prop}/set", value)

    def light(self, room, on=True):
        self.homie_set(f"light-{room.replace('_', '-')}", "light", "on", "true" if on else "false")

    def speaker(self, room, text, intent=None):
        self.homie_set(f"speaker-{room}", "speaker", "say",
                       json.dumps({"text": text, "intent": intent}, ensure_ascii=False) if intent else text)

    def level_of(self, room):
        return self.levels.get(room, "1F")

    def _on_mode(self, t, p):
        if str(p) != self.mode:
            self.mode = str(p)
            self.log("おるすばんモードになった" if self.mode == "alone" else f"モード: {self.mode}", kind="mode")

    def add_job(self, job):
        self.jobs.append(job)
        self.jobs.sort(key=lambda j: j.priority)
        self.log(f"仕事を受けた: {job.name}")
        # 見回りのような後回しにできる仕事は、急ぐ仕事 (転倒・先読み) が来たらやめる (あとでまた見回る)
        if self.job and getattr(self.job, "interruptible", False) and job.priority < self.job.priority:
            self.log(f"{self.job.name} を中断して {job.name} を先にする", kind="preempt")
            self.cancel_jobs(self.job.name)

    def room_spot(self, room):
        """アヒルがその部屋を見に行くときに立つ所 [m] (1 階の部屋だけ。行けない部屋は None)。"""
        s = ROOM_SPOTS.get(room)
        return (s[0] * M, s[1] * M) if s else None

    def busy(self, prefix):
        """その名前で始まる仕事が待っているか動いているか。"""
        return any(j.name.startswith(prefix) for j in self.jobs + ([self.job] if self.job else []))

    def cancel_jobs(self, name):
        """その名前の仕事をやめる (動いていればアヒルに cancelOrder)。"""
        self.jobs = [j for j in self.jobs if j.name != name]
        if self.job and self.job.name == name:
            self.n += 1
            self.m.publish(topic("instantActions"), {"headerId": self.n, "actions": [
                action("cancelOrder", f"cancel-{self.n}")]}, qos=1)
            self.log(f"アヒルの仕事を取り消した: {name}")
            self.job = None

    def _on_robot_state(self, t, p):
        self.robot.update(p)
        info = [i for i in p.get("information", []) if i.get("infoType") == "fallRecovery"]
        if info and info[0].get("infoLevel") == "WARNING" and not getattr(self, "_robot_down", False):
            self._robot_down = True
            self.log("アヒルが転んで起き上がれない → 保護者に知らせる", kind="robot_down")
            self.m.publish("home/alerts", {"type": "robot_down", "level": "warn", "time": round(self.time, 1),
                                           "summary": "見守りロボットが転んで起き上がれません。帰ったら起こしてください。"}, qos=1)
        elif not info and getattr(self, "_robot_down", False):
            self._robot_down = False
            self.log("アヒルが起き上がった", kind="robot_up")

    def _on_persons(self, t, persons):
        room = self.cam_room.get(t.split("/")[1], "?")
        persons = persons if isinstance(persons, list) else []
        self.persons[room] = (self.time, persons)
        for r in self.rules:
            r.on_persons(room, persons)

    # ---- 毎回
    def step(self):
        for r in self.rules:
            r.step()
        self.run_jobs()

    # ---- 仕事を進める
    def _action_state(self, aid):
        for s in self.robot.get("actionStates", []):
            if s["actionId"] == aid:
                return s
        return None

    def _instant(self, a):
        self.m.publish(topic("instantActions"), {"headerId": self.n, "actions": [a]}, qos=1)
        return [a["actionId"]]

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
        if kind == "until":
            pred, timeout = arg
            j.t_wait = getattr(j, "t_wait", None) or self.time
            if pred(j) or self.time - j.t_wait > timeout:
                j.t_wait = None
                j.k += 1
            return
        if not j.sent:
            self.n += 1
            oid = f"{j.name}-{self.n}"
            if kind == "go":
                x, y, dev, acts = arg
                end = action("wait", f"{oid}-end", duration=0.1)
                # アクションの ID は order ごとに一意にする (VDA 5050)
                acts = [{**a, "actionId": f"{oid}-{a['actionId'].split('-', 1)[-1]}"} for a in acts]
                pos = self.robot.get("agvPosition", {})
                self.m.publish(topic("order"), make_order(oid, 0, [(pos.get("x", 0.0), pos.get("y", 0.0), None, 0.3),
                                                                   (x, y, None, dev)], acts + [end]), qos=1)
                j.watch = [a["actionId"] for a in acts] + [end["actionId"]]
            elif kind == "say":
                text, intent = arg
                kw = {"intent": intent} if intent else {}
                j.watch = self._instant(action("speak", f"{oid}-speak", text=text, **kw))
            elif kind == "listen":
                j.watch = self._instant(action("listen", f"{oid}-listen", duration=float(arg)))
            elif kind == "look":
                x, y, dur = arg
                kw = {"targetX": x, "targetY": y} if x is not None else {}
                j.watch = self._instant(action("detectObject", f"{oid}-detect", objectType="person",
                                               duration=float(dur), **kw))
            j.sent, j.t_sent = True, self.time
            return
        states = [self._action_state(a) for a in j.watch]
        if all(s and s["actionStatus"] in ("FINISHED", "FAILED") for s in states):
            for a, s in zip(j.watch, states):
                j.results[a.rsplit("-", 1)[-1]] = s
            j.last = states
            j.k += 1
            j.sent = False
        elif self.time - j.t_sent > JOB_TIMEOUT:
            self.log(f"時間切れ: {j.name}")
            self.job = None


def main():
    c = Coordinator()
    c.log("司令塔を起動")
    while True:
        c.step()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
