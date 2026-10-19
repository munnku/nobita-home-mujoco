"""家の司令塔: IoT 機器 (Homie) の知らせを見て、アヒルに VDA 5050 の order を出す。

いまの決まりごと:
- 見守りカメラが「横になった人」を FALL_CONFIRM 秒以上続けて見たら転倒の疑い →
  その部屋の照明をつけ、アヒルを人のそばへ行かせて (detectObject) 確かめ、声をかける (speak)。
  アヒルも横になった人を見たら home/alerts に転倒を知らせる。立っていれば誤報として取り消す。
- 時刻はシミュレーションの時刻 (home/time) を使う。
uv run python app/coordinator.py
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from home.mqttc import Mqtt  # noqa: E402
from home.vda5050 import action, make_order, topic  # noqa: E402

FALL_CONFIRM = 3.0     # 横になった人をこれだけ続けて見たら疑う [s]
SAME_PLACE = 1.0       # 同じ人とみなす距離 [m]
APPROACH_DEV = 0.75    # 人からこの距離まで近づく (VDA 5050 の allowedDeviationXY) [m]
ROOM_LABEL = {"dk": "DK", "ima": "居間", "ousetsu": "応接室", "rouka": "廊下", "nobita": "のび太の部屋"}


class Coordinator:
    def __init__(self):
        self.m = Mqtt("coordinator")
        self.time = 0.0
        self.robot = {}
        self.cams = {}            # device → room
        self.lying = []           # [{"x","y","room","first","last"}]
        self.incident = None
        self.order_no = 0
        self.m.subscribe("home/time", self._on_time)
        self.m.subscribe("homie/+/camera/room", self._on_cam_room)
        self.m.subscribe("homie/+/camera/persons", self._on_persons)
        self.m.subscribe(topic("state"), lambda t, p: self.robot.update(p))

    def log(self, text, **kw):
        print(f"[coord {self.time:6.1f}] {text}", flush=True)
        self.m.publish("home/log", {"time": round(self.time, 1), "text": text, **kw})

    def _on_time(self, t, p):
        self.time = float(p)

    def _on_cam_room(self, t, p):
        self.cams[t.split("/")[1]] = p

    def _on_persons(self, t, persons):
        dev = t.split("/")[1]
        room = self.cams.get(dev, "?")
        for p in persons if isinstance(persons, list) else []:
            if p.get("posture") != "lying":
                continue
            for c in self.lying:
                if math.hypot(c["x"] - p["x"], c["y"] - p["y"]) < SAME_PLACE:
                    c.update(x=p["x"], y=p["y"], last=self.time)
                    break
            else:
                self.lying.append({"x": p["x"], "y": p["y"], "room": room, "first": self.time, "last": self.time})

    # ---- 毎回
    def step(self):
        self.lying = [c for c in self.lying if self.time - c["last"] < 3.0]
        if self.incident is None:
            for c in self.lying:
                if c["last"] - c["first"] >= FALL_CONFIRM and not self._recently_handled(c):
                    self._start_incident(c)
                    break
        else:
            self._follow_incident()

    def _recently_handled(self, c):
        for h in getattr(self, "handled", []):
            if math.hypot(h["x"] - c["x"], h["y"] - c["y"]) < SAME_PLACE and self.time - h["time"] < 600:
                return True
        return False

    def _start_incident(self, c):
        label = ROOM_LABEL.get(c["room"], c["room"])
        self.log(f"{label}のカメラ: 横になった人 ({c['x']:.1f}, {c['y']:.1f}) → 転倒の疑い", kind="fall_suspected",
                 room=c["room"], x=c["x"], y=c["y"])
        if c["room"] in ROOM_LABEL:
            self.m.publish(f"homie/light-{c['room'].replace('_', '-')}/light/on/set", "true")
        pos = self.robot.get("agvPosition", {})
        self.order_no += 1
        oid = f"fall-{self.order_no}"
        detect_id, speak_id = f"{oid}-detect", f"{oid}-speak"
        order = make_order(oid, 0, [
            (pos.get("x", 0.0), pos.get("y", 0.0), None, 0.3),
            (c["x"], c["y"], None, APPROACH_DEV),
        ], actions_at_last=[
            action("detectObject", detect_id, objectType="person", targetX=c["x"], targetY=c["y"], duration=12.0),
        ])
        self.m.publish(topic("order"), order, qos=1)
        self.incident = {"order": oid, "detect": detect_id, "speak": speak_id, "spot": c, "phase": "approach",
                         "t0": self.time}
        self.log("アヒルに確認を頼んだ (VDA 5050 order: 人のそばへ行き detectObject)")

    def _action_state(self, action_id):
        for s in self.robot.get("actionStates", []):
            if s["actionId"] == action_id:
                return s
        return None

    def _follow_incident(self):
        inc = self.incident
        st = self._action_state(inc["detect"])
        if inc["phase"] == "approach" and st and st["actionStatus"] in ("FINISHED", "FAILED"):
            try:
                res = json.loads(st.get("resultDescription") or "{}")
            except ValueError:
                res = {}
            c = inc["spot"]
            if res.get("found") and res.get("posture") == "lying":
                self.log(f"アヒルの目でも横になった人を確認 (score {res.get('score')}) → 家族に知らせる",
                         kind="fall_confirmed")
                self.m.publish("home/alerts", {"type": "fall", "room": c["room"], "x": c["x"], "y": c["y"],
                                               "time": round(self.time, 1), "confirmedBy": "microduck-01",
                                               "elapsed": round(self.time - c["first"], 1)}, qos=1)
                text = "大丈夫ですか？ご家族に知らせました。"
            elif res.get("found"):
                self.log(f"アヒルが見ると立っていた ({res.get('posture')}) → 誤報として取り消し", kind="false_alarm")
                text = "お元気そうでよかったです。"
            else:
                self.log("アヒルは人を見つけられなかった → 家族に確認を頼む", kind="fall_unconfirmed")
                self.m.publish("home/alerts", {"type": "fall_unconfirmed", "room": c["room"], "x": c["x"],
                                               "y": c["y"], "time": round(self.time, 1)}, qos=1)
                text = "どこにいますか？ご家族に知らせます。"
            self.m.publish(topic("instantActions"), {"headerId": 0, "actions": [action("speak", inc["speak"], text=text)]})
            inc["phase"] = "speak"
        elif inc["phase"] == "approach" and self.time - inc["t0"] > 240:
            self.log("時間切れ: アヒルが着けなかった", kind="timeout")
            inc["phase"] = "done"
        elif inc["phase"] == "speak":
            s = self._action_state(inc["speak"])
            if s and s["actionStatus"] in ("FINISHED", "FAILED"):
                inc["phase"] = "done"
        if inc["phase"] == "done":
            self.handled = getattr(self, "handled", []) + [{"x": inc["spot"]["x"], "y": inc["spot"]["y"],
                                                            "time": self.time}]
            self.incident = None


def main():
    c = Coordinator()
    c.log("司令塔を起動")
    while True:
        c.step()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
