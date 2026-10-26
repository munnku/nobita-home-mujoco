"""家の司令塔: IoT 機器 (Homie) の知らせを見て、家電を動かし、アヒルに VDA 5050 の order を出す。

ひとりで留守番する子どもを 4 つの働きで見守る。決まりごと (app/rules/) が気づいてタスク (Job) を作り、
タスク一覧 (app/tasks.py) が重さの順に進め、知らせ (app/notify.py) が保護者に段階的に知らせる。
- 予防:     先読み (predict.py: 危ない場所へ向かうのを先に止める)、見回り (patrol.py: 床の小物を見つけて片付ける)
- 検知:     転倒 (fall.py: 見守りカメラの YOLO)、声 (voice.py: 常時マイク → LLM)、カメラの無い部屋 (blind_spot.py)、
            玄関 (entrance.py: 鍵・扉 → アヒルが見る)、災害 (emergency.py: 煙センサー・地震速報・停電)
- 対応支援: 安否確認・手当て・落ち着かせる・逃げ方の対話 (talk.py → home/brain.py の LLM)、インターホンの応対 (visitor.py)、
            保護者からの依頼 (requests.py: WoT の goToRoom / checkRoom)
- 通知:     notify.py (重さ別、emergency は「見た」が無ければ 2 人目へ)
LLM を通すのは声と対話 (home/brain.py) だけ。YOLO・先読み・センサーの決まりは LLM を通さずにすぐ動く。
LLM には、呼ぶときだけ「いまの様子」(situation) を短い文で添える。
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

from app.notify import Notifier  # noqa: E402
from app.rules.base import LABEL, M  # noqa: E402
from app.rules.blind_spot import BlindSpot  # noqa: E402
from app.rules.emergency import Emergency  # noqa: E402
from app.rules.entrance import Entrance  # noqa: E402
from app.rules.fall import Fall  # noqa: E402
from app.rules.patrol import Patrol  # noqa: E402
from app.rules.predict import Predict  # noqa: E402
from app.rules.requests import ROOM_SPOTS, Requests  # noqa: E402
from app.rules.visitor import Visitor  # noqa: E402
from app.rules.voice import Voice  # noqa: E402
from app.tasks import Tasks  # noqa: E402
from home.mqttc import Mqtt  # noqa: E402
from home.vda5050 import topic  # noqa: E402

HOUSE = os.path.join(ROOT, "houses", "home")
MODE_TEXT = {"alone": "おるすばん中 (子どもがひとり)", "family": "家族がいる", "away": "だれもいない"}


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
        self.child_seen = None       # 見守りカメラで最後に見た子ども {"room","x","y","posture","time"}
        self.locked, self.door_open = {}, {}
        self.motion = {}
        self.attended = {}           # 部屋 → アヒルが人の様子を確かめて知らせた時刻 (決まりごとどうしで重ねて知らせない)
        self.distress = {}           # 部屋 → 痛がる声を聞いた時刻 (voice.py が LLM の判断で入れ、fall.py が使う)
        self.recent = []             # この数分の出来事 [(時刻, 文)] (situation に使う)
        self.n = 0
        self.tasks = Tasks(self)
        self.notifier = Notifier(self)
        s = self.m.subscribe
        s("home/time", lambda t, p: setattr(self, "time", float(p)))
        s("home/clock", lambda t, p: setattr(self, "clock", str(p)))
        s("home/mode", self._on_mode)
        s("homie/+/camera/room", lambda t, p: self.cam_room.__setitem__(t.split("/")[1], p))
        s("homie/+/camera/persons", self._on_persons)
        s("homie/+/lock/locked", lambda t, p: self.locked.__setitem__(t.split("/")[1][5:], str(p).lower() == "true"))
        s("homie/+/contact/open", lambda t, p: self.door_open.__setitem__(t.split("/")[1][5:], str(p).lower() == "true"))
        s("homie/+/sensor/motion", lambda t, p: self.motion.__setitem__(t.split("/")[1][7:], str(p).lower() == "true"))
        s(topic("state"), self._on_robot_state)
        # 決まりごと (この順に毎回見る)
        self.rules = [Requests(self), Fall(self), Predict(self), BlindSpot(self), Voice(self), Entrance(self),
                      Visitor(self), Emergency(self), Patrol(self)]
        self.fall = next(r for r in self.rules if isinstance(r, Fall))
        self.emergency = next(r for r in self.rules if isinstance(r, Emergency))

    # ---- 共通
    def log(self, text, **kw):
        print(f"[coord {self.time:6.1f}] {text}", flush=True)
        self.m.publish("home/log", {"time": round(self.time, 1), "clock": self.clock, "text": text, **kw})
        if kw.get("kind") and kw["kind"] not in ("task_added", "reply"):
            self.recent = [(t, x) for t, x in self.recent if self.time - t < 180] + [(self.time, text)]

    def notify(self, type, level, summary, room=None, category="通知", **kw):
        return self.notifier.send(type, level, summary, room=room, category=category, **kw)

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

    # ---- タスク (app/tasks.py。決まりごとはこの名前で頼む)
    def add_job(self, job):
        self.tasks.add(job)

    def busy(self, prefix):
        """その名前で始まる仕事が待っているか動いているか。"""
        return self.tasks.busy(prefix)

    def cancel_jobs(self, name):
        self.tasks.cancel(name)

    @property
    def job(self):
        return self.tasks.duck.job          # アヒルがいましている仕事

    @property
    def jobs(self):
        return self.tasks.duck.jobs

    def room_spot(self, room):
        """アヒルがその部屋を見に行くときに立つ所 [m] (1 階の部屋だけ。行けない部屋は None)。"""
        s = ROOM_SPOTS.get(room)
        return (s[0] * M, s[1] * M) if s else None

    # ---- いまの様子 (LLM に添える短い文。YOLO などの結果を言葉にしたもの)
    def child_room(self):
        if self.child_seen and self.time - self.child_seen["time"] < 30:
            return self.child_seen["room"]
        rooms = [r for r, on in self.motion.items() if on]
        return rooms[0] if len(rooms) == 1 else None

    def situation(self):
        parts = [f"時刻 {self.clock or '?'}", MODE_TEXT.get(self.mode, self.mode)]
        cs = self.child_seen
        if cs and self.time - cs["time"] < 120:
            pose = {"standing": "立っている", "lying": "横になっている"}.get(cs["posture"], cs["posture"])
            parts.append(f"子どもは{LABEL.get(cs['room'], cs['room'])}で{pose} ({self.time - cs['time']:.0f} 秒前に見守りカメラで確認)")
        else:
            rooms = [LABEL.get(r, r) for r, on in self.motion.items() if on]
            parts.append(f"子どもの姿はカメラに無い (人感センサー: {'・'.join(rooms) or '反応なし'})")
        j = self.job
        parts.append(f"アヒルの仕事: {j.name}" if j else "アヒルは待機中")
        if self.recent:
            parts.append("最近の出来事: " + " / ".join(f"{self.time - t:.0f} 秒前 {x[:40]}" for t, x in self.recent[-4:]))
        return "。".join(parts)

    def _on_robot_state(self, t, p):
        self.robot.update(p)
        info = [i for i in p.get("information", []) if i.get("infoType") == "fallRecovery"]
        if info and info[0].get("infoLevel") == "WARNING" and not getattr(self, "_robot_down", False):
            self._robot_down = True
            self.log("アヒルが転んで起き上がれない → 保護者に知らせる", kind="robot_down")
            self.notify("robot_down", "warn", "見守りロボットが転んで起き上がれません。帰ったら起こしてください。")
        elif not info and getattr(self, "_robot_down", False):
            self._robot_down = False
            self.log("アヒルが起き上がった", kind="robot_up")

    def _on_persons(self, t, persons):
        room = self.cam_room.get(t.split("/")[1], "?")
        persons = persons if isinstance(persons, list) else []
        self.persons[room] = (self.time, persons)
        for p in persons:
            if p.get("who") in (None, "child"):
                self.child_seen = {"room": p.get("room") or room, "x": p["x"], "y": p["y"],
                                   "posture": p.get("posture"), "time": self.time}
        for r in self.rules:
            r.on_persons(room, persons)

    # ---- 毎回
    def step(self):
        for r in self.rules:
            r.step()
        self.notifier.step()
        self.tasks.step()


def main():
    c = Coordinator()
    c.log("司令塔を起動")
    while True:
        c.step()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
