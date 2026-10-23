"""VDA 5050 v2.0 (https://github.com/VDA5050/VDA5050) のロボット側 (Agent) と司令塔側の補助 (make_order)。

トピック: uagv/v2/<manufacturer>/<serialNumber>/{order, instantActions, state, visualization, connection, factsheet}
使う範囲:
- order: nodes (nodePosition x, y, theta, allowedDeviationXY) と edges。ノードを順にたどり、着いたらノードの actions を順に行う
- actions: detectObject (VDA 5050 の定義済みアクション。objectType=person、頭のカメラと YOLO で人を探す)、
           speak (この家で足したアクション。話す内容を home/speech に出す。intent は問いの種類)、
           listen (この家で足したアクション。マイクで duration 秒聞き、聞こえた声を返す)、wait
- instantActions: cancelOrder, startPause, stopPause, stateRequest, factsheetRequest、
  speak / listen / detectObject (いまの場所で行う。blockingType=NONE の speak は移動を止めずにすぐ話す)
- detectObject で人を見つけたら、その絵を build/alerts/ に残し、結果に image として書く (privacy=true なら残さない)
- detectObject の objectType=hazard は、首を下に向けてその場で 1 周しながら、床の小さい物を探す (見回り)。
  結果は objects [{kind, x, y, score, room}]
- state: 1 秒ごとと変化したとき。agvPosition, driving, nodeStates, edgeStates, actionStates, errors など
アヒルが自分で経路を作る (地図の上の A*) ので、ノードは「行き先」だけでよい。
"""
import datetime
import json
import math
import os
import queue

import numpy as np

from .perception import CameraModel, covered, estimate_object, estimate_person, robot_box

VERSION = "2.0.0"
MANUFACTURER = "NobitaHome"
SERIAL = "microduck-01"
MAP_ID = "nobita-1F"
HAZARD_NECK = -0.8    # 床の物を探すときの首の角度 (下向き) [rad]
SAME_OBJECT = 0.2     # 同じ物とみなす距離 [m]
STUCK_FALLEN = 20.0   # 転んでこれだけ起き上がれなければ「助けが要る」と知らせる [s]
TARGET_BEARING = 0.35  # 足元の見えない枠を target の人とみなす向きのずれの上限 [rad]
EDGE = 4              # 絵の縁からこれだけ [px] の所にかかる枠は使わない
HAZARD_RANGE = 2.5    # 床の物はこれより近い推定だけ使う [m]
HAZARD_TURN = 0.8     # 床の物を探すときに回る指令 [rad/s]。歩きの方策は 0.65 より小さい回転の指令ではほとんど回らない
                      # (実際は約 0.5 rad/s。首を下げて速く回ると位置がずれるので、ずれは前後・左右の指令で戻す)
HAZARD_SETTLE = 1.5   # 見る所に着いてから、首を上げたまま止まって落ち着く時間 [s]
MIN_TURN = 0.7       # 歩きの方策が回り始める回転の指令の下限 [rad/s] (これより小さいと止まれとして扱われる)
HAZARD_REPORT = 0.55  # 1 周のうちの一番高い score がこれより低いか、1 回しか見えなかった物は知らせない (壁際・足元の見間違い)
HEAR_RADIUS = 4.0     # アヒルのマイクに声が届く距離 [m] (同じ部屋ならこれより遠くても届く)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERT_DIR = os.path.join(ROOT, "build", "alerts")


def topic(name, manufacturer=MANUFACTURER, serial=SERIAL):
    return f"uagv/v2/{manufacturer}/{serial}/{name}"


def turn_cmd(err, gain, limit, dead=0.0):
    """向きの誤差 err [rad] を減らす回転の指令。小さい指令では回らないので、MIN_TURN より小さくしない (dead 以内なら 0)。"""
    if abs(err) <= dead:
        return 0.0
    return math.copysign(min(limit, max(MIN_TURN, gain * abs(err))), err)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def params(action):
    return {p["key"]: p["value"] for p in action.get("actionParameters", [])}


def make_order(order_id, update_id, points, actions_at_last=(), header_id=0, serial=SERIAL):
    """司令塔側: [(x, y, theta|None, allowedDeviationXY), ...] から order を作る。最後のノードに actions を付ける。"""
    nodes, edges = [], []
    for k, (x, y, th, dev) in enumerate(points):
        pos = {"x": x, "y": y, "mapId": MAP_ID, "allowedDeviationXY": dev}
        if th is not None:
            pos["theta"] = th
        nodes.append({"nodeId": f"n{k}", "sequenceId": 2 * k, "released": True, "nodePosition": pos,
                      "actions": list(actions_at_last) if k == len(points) - 1 else []})
        if k > 0:
            edges.append({"edgeId": f"e{k}", "sequenceId": 2 * k - 1, "released": True,
                          "startNodeId": f"n{k-1}", "endNodeId": f"n{k}", "actions": []})
    return {"headerId": header_id, "timestamp": now_iso(), "version": VERSION, "manufacturer": MANUFACTURER,
            "serialNumber": serial, "orderId": order_id, "orderUpdateId": update_id,
            "nodes": nodes, "edges": edges}


def action(action_type, action_id, **kw):
    return {"actionType": action_type, "actionId": action_id, "blockingType": "HARD",
            "actionParameters": [{"key": k, "value": v} for k, v in kw.items()]}


class Agent:
    """ロボット側。シミュレーションのスレッドから tick() を毎回呼ぶ (MQTT の受信はキューで受け渡す)。"""

    def __init__(self, world, mqtt, detector, hazard_detector=None, serial=SERIAL):
        self.w = world
        self.m = mqtt
        self.det = detector
        self.hazard = hazard_detector
        self.serial = serial
        self.inbox = queue.Queue()
        self.header = 0
        self.order = None
        self.node_idx = 0
        self.action_idx = 0
        self.action_run = None      # 実行中のアクションの状態 (dict)
        self.action_states = []
        self.errors = []
        self.paused = False
        self.driving = False
        self.next_state_t = 0.0
        self.last_node = ("", 0)
        self.speech = None          # (話し終わる時刻, 文)
        self.m.subscribe(self._topic("order"), lambda t, p: self.inbox.put(("order", p)))
        self.m.subscribe(self._topic("instantActions"), lambda t, p: self.inbox.put(("instant", p)))
        self.voices = []            # 聞こえた声 (home/voice) [(シミュレーションの時刻, 声)]
        self.m.subscribe("home/voice", lambda t, p: self.inbox.put(("voice", p)) if isinstance(p, dict) else None)
        self.m.publish(self._topic("connection"), self._hdr({"connectionState": "ONLINE"}), retain=True, qos=1)
        self.publish_factsheet()

    def status_text(self):
        """動画に出す今の様子 (日本語)。"""
        if self.paused:
            return "一時停止"
        if self.driving:
            return "移動中"
        if self.action_run is not None:
            if self.action_run.get("p", {}).get("objectType") == "hazard":
                return "床を見回っている"
            return {"detectObject": "人を確かめている", "speak": "話している", "listen": "聞いている",
                    "wait": "待っている"}.get(
                self.action_run.get("type"), "作業中")
        return "待機"

    # ---- MQTT へ出すもの
    def _hdr(self, body):
        self.header += 1
        return {"headerId": self.header, "timestamp": now_iso(), "version": VERSION,
                "manufacturer": MANUFACTURER, "serialNumber": self.serial, **body}

    def publish_factsheet(self):
        self.m.publish(self._topic("factsheet"), self._hdr({
            "typeSpecification": {"seriesName": "Microduck", "seriesDescription":
                                  "二足歩行の小型ロボット (Pollen Robotics)。学習済みの方策で歩き、頭のカメラで見る",
                                  "agvKinematic": "DIFF", "agvClass": "CARRIER", "maxLoadMass": 0,
                                  "localizationTypes": ["NATURAL"], "navigationTypes": ["AUTONOMOUS"]},
            "physicalParameters": {"speedMin": 0.0, "speedMax": 0.3, "accelerationMax": 0.5,
                                   "decelerationMax": 0.5, "heightMin": 0.25, "heightMax": 0.25,
                                   "width": 0.16, "length": 0.12},
            "protocolLimits": {}, "protocolFeatures": {"optionalParameters": [], "agvActions": [
                {"actionType": "detectObject", "actionScopes": ["NODE", "INSTANT"],
                 "actionParameters": [{"key": "objectType", "valueDataType": "STRING",
                                       "description": "person (人) / hazard (床の小さい危ない物)"},
                                      {"key": "targetX", "valueDataType": "FLOAT", "isOptional": True},
                                      {"key": "targetY", "valueDataType": "FLOAT", "isOptional": True},
                                      {"key": "duration", "valueDataType": "FLOAT", "isOptional": True},
                                      {"key": "privacy", "valueDataType": "BOOLEAN", "isOptional": True}],
                 "resultDescription": "JSON: found, score, posture, x, y, image"},
                {"actionType": "speak", "actionScopes": ["NODE", "INSTANT"],
                 "actionParameters": [{"key": "text", "valueDataType": "STRING"},
                                      {"key": "intent", "valueDataType": "STRING", "isOptional": True}]},
                {"actionType": "listen", "actionScopes": ["NODE", "INSTANT"],
                 "actionParameters": [{"key": "duration", "valueDataType": "FLOAT", "isOptional": True}],
                 "resultDescription": "JSON: heard [{who, text, time}]"},
                {"actionType": "wait", "actionScopes": ["NODE"],
                 "actionParameters": [{"key": "duration", "valueDataType": "FLOAT"}]},
                {"actionType": "cancelOrder", "actionScopes": ["INSTANT"]},
                {"actionType": "startPause", "actionScopes": ["INSTANT"]},
                {"actionType": "stopPause", "actionScopes": ["INSTANT"]},
                {"actionType": "stateRequest", "actionScopes": ["INSTANT"]},
                {"actionType": "factsheetRequest", "actionScopes": ["INSTANT"]}]},
            "agvGeometry": {}, "loadSpecification": {}}), retain=True)

    def publish_state(self):
        x, y, th = self._pose()
        o = self.order
        nodes = o["nodes"][self.node_idx:] if o else []
        state = self._hdr({
            "orderId": o["orderId"] if o else "", "orderUpdateId": o["orderUpdateId"] if o else 0,
            "lastNodeId": self.last_node[0], "lastNodeSequenceId": self.last_node[1],
            "driving": self.driving, "paused": self.paused, "operatingMode": "AUTOMATIC",
            "newBaseRequest": False, "distanceSinceLastNode": 0.0,
            "nodeStates": [{"nodeId": n["nodeId"], "sequenceId": n["sequenceId"], "released": True}
                           for n in nodes],
            "edgeStates": [{"edgeId": e["edgeId"], "sequenceId": e["sequenceId"], "released": True}
                           for e in (o["edges"] if o else []) if e["sequenceId"] > self.last_node[1]],
            "agvPosition": {"x": round(x, 3), "y": round(y, 3), "theta": round(th, 3), "mapId": MAP_ID,
                            "positionInitialized": True, "localizationScore": 1.0},
            "velocity": dict(zip(("vx", "vy", "omega"), self._vel())),
            "batteryState": {"batteryCharge": 80.0, "charging": False},
            "actionStates": self.action_states[-20:],
            "errors": self.errors, "information": self._information(),
            "safetyState": {"eStop": "NONE", "fieldViolation": False},
            "room": self.w.room_of(x, y),
        })
        self.m.publish(self._topic("state"), state)
        self.m.publish(self._topic("visualization"), self._hdr({"agvPosition": state["agvPosition"],
                                                          "velocity": state["velocity"]}))

    # ---- 毎回
    def tick(self):
        while not self.inbox.empty():
            kind, msg = self.inbox.get()
            if kind == "order":
                self._on_order(msg)
            elif kind == "voice":
                self.voices.append((self.w.time, msg))
                self.voices = [v for v in self.voices if self.w.time - v[0] < 30.0]
            else:
                for a in msg.get("actions", []):
                    self._on_instant(a)
        if self.order and not self.paused:
            self._run_order()
        if self.speech and self.w.time >= self.speech[0]:
            self.speech = None
        if self.w.time >= self.next_state_t:
            self.next_state_t = self.w.time + 1.0
            self.publish_state()

    def _set_action_state(self, a, status, result=""):
        for s in self.action_states:
            if s["actionId"] == a["actionId"]:
                s["actionStatus"] = status
                s["resultDescription"] = result
                break
        else:
            self.action_states.append({"actionId": a["actionId"], "actionType": a["actionType"],
                                       "actionStatus": status, "resultDescription": result})
        self.next_state_t = 0.0      # すぐに state を出す

    def _on_order(self, o):
        if self.order and o.get("orderId") == self.order["orderId"]:
            if o.get("orderUpdateId", 0) <= self.order["orderUpdateId"]:
                return
            self.order["orderUpdateId"] = o["orderUpdateId"]
            self.order["nodes"] += [n for n in o["nodes"] if n["sequenceId"] > self.order["nodes"][-1]["sequenceId"]]
            self.order["edges"] += [e for e in o["edges"] if e["sequenceId"] > self.order["edges"][-1]["sequenceId"]] \
                if self.order["edges"] else o["edges"]
            return
        if self.order and self.node_idx < len(self.order["nodes"]):
            self.errors = [{"errorType": "orderError", "errorLevel": "WARNING",
                            "errorDescription": "busy: 前の order が終わっていない"}]
            self.next_state_t = 0.0
            return
        self.errors = []
        self.order = o
        self.node_idx = 0
        self.action_idx = 0
        self.action_run = None
        for n in o["nodes"]:
            for a in n.get("actions", []):
                self._set_action_state(a, "WAITING")
        self._stop()

    def _on_instant(self, a):
        t = a["actionType"]
        if t == "cancelOrder":
            if self.order:
                for n in self.order["nodes"][self.node_idx:]:
                    for na in n.get("actions", []):
                        self._set_action_state(na, "FAILED", "canceled")
            self.order = None
            self.node_idx = self.action_idx = 0     # このあとの即時のアクションが新しい order の始めから動くように
            self.action_run = None
            self._stop()
            self.w.duck.set_head()
            self.driving = False
            self._set_action_state(a, "FINISHED")
        elif t == "startPause":
            self.paused = True
            self._stop()
            self.driving = False
            self._set_action_state(a, "FINISHED")
        elif t == "stopPause":
            self.paused = False
            if self.order and self.node_idx < len(self.order["nodes"]):
                self.order["_navigating"] = False
            self._set_action_state(a, "FINISHED")
        elif t == "stateRequest":
            self.next_state_t = 0.0
            self._set_action_state(a, "FINISHED")
        elif t == "factsheetRequest":
            self.publish_factsheet()
            self._set_action_state(a, "FINISHED")
        elif t == "speak" and a.get("blockingType") == "NONE":
            self._speak(params(a))          # 移動を止めずにすぐ話す
            self._set_action_state(a, "FINISHED")
        elif t in ("speak", "detectObject", "listen"):
            # 即時のアクションとしても受け付ける (いまの場所で行う)
            if self.order is None:
                self.order = {"orderId": f"instant-{a['actionId']}", "orderUpdateId": 0, "nodes": [], "edges": []}
                self.node_idx = self.action_idx = 0
            x, y, _ = self._pose()
            self.order["nodes"].append({"nodeId": f"here-{a['actionId']}", "sequenceId": 10000 + self.header,
                                        "nodePosition": {"x": x, "y": y, "allowedDeviationXY": 10.0},
                                        "actions": [a]})
            self._set_action_state(a, "WAITING")

    def _run_order(self):
        o = self.order
        if self.node_idx >= len(o["nodes"]):
            return
        node = o["nodes"][self.node_idx]
        pos = node.get("nodePosition")
        if not node.get("_reached"):
            x, y, _ = self._pose()
            dev = pos.get("allowedDeviationXY", 0.1) if pos else 0.0
            if pos is None or (not self._needs_nav(node) and math.hypot(pos["x"] - x, pos["y"] - y) <= max(dev, 0.12)):
                node["_reached"] = True
                if o.get("_navigating"):
                    self._stop()        # 許される距離に入った: 経路の残りを歩き続けないよう止める (止めないと首を下げたまま歩いた)
            elif not o.get("_navigating"):
                ok = self._start_nav(node, pos, dev, x, y)
                o["_navigating"] = True
                self.driving = ok
                if not ok:
                    self._fail_order(f"no path to node {node['nodeId']}")
                return
            else:
                st = self._nav_status()
                if st == "arrived":
                    node["_reached"] = True
                elif st == "failed":
                    self._fail_order(f"could not reach node {node['nodeId']}")
                    return
                else:
                    return
            o["_navigating"] = False
            self.driving = False
            self.last_node = (node["nodeId"], node["sequenceId"])
            self.action_idx = 0
            self.next_state_t = 0.0
        # ノードのアクション
        self._node_actions(node)

    def _information(self):
        d = getattr(self.w, "duck", None)
        if d is not None and getattr(d, "recovering", False) and not isinstance(self, StretchAgent):
            stuck = d.time - getattr(d, "fall_time", d.time) > STUCK_FALLEN
            return [{"infoType": "fallRecovery", "infoLevel": "WARNING" if stuck else "INFO",
                     "infoDescription": "転んで起き上がれない (助けが要る)" if stuck else "転んだので起き上がっている"}]
        return []

    # ---- 体ごとに違うところ (アヒル。Stretch は StretchAgent で替える)
    def _topic(self, name):
        return topic(name, serial=self.serial)

    def _pose(self):
        return self.w.duck.pose()

    def _vel(self):
        c = self.w.duck.command
        return float(c[0]), 0.0, float(c[2])

    def _stop(self):
        self.w.stop()

    def _needs_nav(self, node):
        """近くにいても、ノードの位置へ動き直す必要があるか (Stretch の pick は腕の下に物が来る所へ行く)。"""
        return False

    def _nav_status(self):
        return self.w.nav_status()

    def _start_nav(self, node, pos, dev, x, y):
        """ノードへ向かい始める。経路があれば True。"""
        gx, gy, gth, tol = pos["x"], pos["y"], pos.get("theta"), max(dev, 0.12)
        look = self._look_target(node)
        if look is not None:
            # 人を見に行くときは、人そのものではなく人を見通せる立ち位置へ行き、人の方を向く
            r_max = max(1.3, min(dev, 2.0))
            vp = self.w.map.viewpoint(look, (x, y), r_min=0.8, r_max=r_max)
            if vp is None:      # 見通せる所が無ければ、同じ部屋の行ける所から首を振って探す
                room = self.w.room_of(*look)
                area = None
                if room:
                    rr = next(r for r in self.w.layout["rooms"] if r["name"] == room)["rect"]
                    area = [v * self.w.module for v in rr]
                vp = self.w.map.viewpoint(look, (x, y), r_min=0.5, r_max=3.0, need_sight=False, area=area)
            if vp is not None:
                gx, gy, tol = vp[0], vp[1], 0.12
                gth = math.atan2(look[1] - vp[1], look[0] - vp[0])
        return self.w.go_to(gx, gy, gth, tolerance=tol)

    def _node_actions(self, node):
        o = self.order
        acts = node.get("actions", [])
        if self.action_idx < len(acts):
            a = acts[self.action_idx]
            if self.action_run is None:
                self.action_run = self._start_action(a)
                self._set_action_state(a, "RUNNING")
            done, ok, result = self._step_action(a, self.action_run)
            if done:
                self._set_action_state(a, "FINISHED" if ok else "FAILED", result)
                self.action_run = None
                self.action_idx += 1
            return
        self.node_idx += 1
        if self.node_idx >= len(o["nodes"]):
            self.next_state_t = 0.0

    @staticmethod
    def _look_target(node):
        for a in node.get("actions", []):
            p = params(a)
            if a["actionType"] == "detectObject" and "targetX" in p:
                return float(p["targetX"]), float(p["targetY"])
        return None

    def _fail_order(self, why):
        print(f"[agent] order failed: {why}", flush=True)
        self.errors = [{"errorType": "navigationError", "errorLevel": "WARNING", "errorDescription": why}]
        for n in self.order["nodes"][self.node_idx:]:
            for a in n.get("actions", []):
                self._set_action_state(a, "FAILED", why)
        self.node_idx = len(self.order["nodes"])
        self.driving = False
        self._stop()

    # ---- アクション
    def _start_action(self, a):
        p = params(a)
        t = a["actionType"]
        run = {"t0": self.w.time, "p": p, "type": t, "id": a["actionId"]}
        if t == "speak":
            run["dur"] = self._speak(p)
        elif t == "listen":
            run.update(dur=float(p.get("duration", 8.0)), heard=[])
        elif t == "detectObject":
            run.update(best=None, next_look=0.0, phase="face", dur=float(p.get("duration", 10.0)))
        return run

    def _step_action(self, a, run):
        """(終わったか, 成功か, 結果の文字列) を返す。"""
        t = a["actionType"]
        el = self.w.time - run["t0"]
        if t in ("speak", "wait"):
            dur = run.get("dur", float(run["p"].get("duration", 1.0)))
            return el >= dur, True, ""
        if t == "listen":
            return self._step_listen(run, el)
        if t == "detectObject" and run["p"].get("objectType") == "hazard":
            return self._step_hazard(run, el)
        if t == "detectObject":
            return self._step_detect(run, el)
        return True, False, f"unknown action {t}"

    def _speak(self, p):
        """話す (home/speech に出す)。話し終わるまでの秒数を返す。"""
        text = str(p.get("text", ""))
        dur = 1.0 + 0.12 * len(text)
        self.speech = (self.w.time + dur, text)
        self.last_speak_t = self.w.time
        x, y, _ = self._pose()
        msg = {"speaker": self.serial, "text": text, "room": self.w.room_of(x, y), "x": round(x, 2), "y": round(y, 2),
               "level": "1F", "time": round(self.w.time, 1), "duration": round(dur, 1)}
        if p.get("intent"):
            msg["intent"] = p["intent"]
        self.m.publish("home/speech", msg)
        return dur

    def _step_listen(self, run, el):
        """マイクで聞く: 聞こえる声 (同じ部屋か HEAR_RADIUS 以内) を集める。声が聞こえて 1.5 秒続かなければ終わる。
        直前に話していれば、話し始めてからの声も入れる (話し終わる前に返事が来ることがある)。"""
        x, y, _ = self.w.duck.pose()
        room = self.w.room_of(x, y)
        since = min(run["t0"], getattr(self, "last_speak_t", run["t0"]))
        for t, v in self.voices:
            if t < since or any(h["time"] == v.get("time") and h["text"] == v.get("text") for h in run["heard"]):
                continue
            near = v.get("room") == room or (v.get("level", "1F") == "1F" and v.get("x") is not None
                                             and math.hypot(v["x"] - x, v["y"] - y) <= HEAR_RADIUS)
            if near:
                run["heard"].append({"who": v.get("who"), "text": v.get("text"), "time": v.get("time")})
                run["t_last"] = t
        quiet = run["heard"] and self.w.time - run["t_last"] > 1.5
        if el >= run["dur"] or quiet:
            return True, True, json.dumps({"heard": run["heard"]}, ensure_ascii=False)
        return False, False, ""

    def _save_evidence(self, run, a_id):
        """見つけたときの頭のカメラの絵を残す (保護者への知らせに添える)。privacy=true なら残さない。"""
        if run.get("img") is None or str(run["p"].get("privacy", "")).lower() in ("true", "1"):
            return None
        import cv2
        os.makedirs(ALERT_DIR, exist_ok=True)
        path = os.path.join(ALERT_DIR, f"{a_id}_{self.w.time:.0f}.jpg")
        b = run["best"].get("box")
        vis = np.ascontiguousarray(run["img"][..., ::-1])
        if b:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (40, 40, 230), 3)
        cv2.imwrite(path, vis)
        return os.path.relpath(path, ROOT).replace("\\", "/")

    def _step_hazard(self, run, el):
        """床の小さい物を探す: 首を下に向けて 1 秒待ち、ゆっくり 1 周回りながら 0.5 秒ごとに見る。"""
        if self.hazard is None:
            return True, False, json.dumps({"found": False, "objects": [], "error": "no hazard detector"})
        objs = run.setdefault("objects", [])
        # 歩いてきた勢いのまま止まって首を下げると前に転ぶ (見る所に着いた直後に転んでいた)。
        # まず首を上げたまま止まって落ち着き (HAZARD_SETTLE 秒)、止まったまま首を下げてから回る
        if el < HAZARD_SETTLE:
            self.w.duck.set_head()
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
            return False, False, ""
        self.w.duck.set_head(neck_pitch=HAZARD_NECK)
        if el < HAZARD_SETTLE + 1.0:
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
            return False, False, ""
        # 回りながら少しずつずれる (首を下げると前に出やすい) ので、始めの位置へ戻すように前後・左右の指令を足す
        px, py, th = self._pose()
        sx, sy = run.setdefault("start", (px, py))
        ex, ey = sx - px, sy - py
        bx, by = math.cos(th) * ex + math.sin(th) * ey, -math.sin(th) * ex + math.cos(th) * ey
        self.w.set_manual_cmd(max(-0.15, min(0.15, 1.5 * bx)), max(-0.15, min(0.15, 1.5 * by)), HAZARD_TURN)
        if self.w.time >= run.get("next_look", 0.0):
            run["next_look"] = self.w.time + 0.5
            img = self.w.render("head_camera", 640, 480)
            cam = CameraModel.from_mujoco(self.w.model, self.w.data, "head_camera", 640, 480)
            x0, y0, _ = self._pose()
            for b in self.hazard.detect([img])[0]:
                if b[3] >= 480 - EDGE:
                    continue        # 絵の下の縁にかかる枠 (物の一部しか写っていないか、アヒル自身の体) は使わない
                est = estimate_object(cam, b, 0.0)
                if os.environ.get("HAZARD_DEBUG"):
                    print(f"[hazard] t={self.w.time:.1f} duck=({x0:.2f},{y0:.2f}) box={[int(v) for v in b[:4]]} "
                          f"est={est and (est['kind'], est['x'], est['y'], round(est['score'], 2))}", flush=True)
                # 遠すぎる (地平線の近くの枠は床に落とすと大きくずれる) か、家の外に落ちた推定は使わない
                if est is None or math.hypot(est["x"] - x0, est["y"] - y0) > HAZARD_RANGE                         or self.w.room_of(est["x"], est["y"]) is None:
                    continue
                same = next((o for o in objs if o["kind"] == est["kind"]
                             and math.hypot(o["x"] - est["x"], o["y"] - est["y"]) < SAME_OBJECT), None)
                shot = (img, [int(v) for v in b[:4]])          # 一番 score の高かった絵を証拠に残す
                # 位置は見えた絵ごとの推定の重み付き平均 (近いほど・score が高いほど重い。遠い絵は 1 画素のずれが大きい)
                wt = est["score"] / max(0.3, math.hypot(est["x"] - x0, est["y"] - y0)) ** 2
                if same is None:
                    objs.append({**est, "room": self.w.room_of(est["x"], est["y"]), "n": 1, "shot": shot,
                                 "sw": wt, "sx": wt * est["x"], "sy": wt * est["y"]})
                else:
                    same["n"] += 1
                    same["sw"] += wt
                    same["sx"] += wt * est["x"]
                    same["sy"] += wt * est["y"]
                    same["x"], same["y"] = same["sx"] / same["sw"], same["sy"] / same["sw"]
                    if est["score"] > same["score"]:
                        same["score"], same["shot"] = est["score"], shot
        if el >= float(run["p"].get("duration", 11.0)):
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
            self.w.duck.set_head()
            found = []
            for k, o in enumerate(x for x in objs if x["score"] >= HAZARD_REPORT and x["n"] >= 2):
                img = None
                if str(run["p"].get("privacy", "")).lower() not in ("true", "1"):
                    import cv2
                    os.makedirs(ALERT_DIR, exist_ok=True)
                    im, box = o["shot"]
                    vis = np.ascontiguousarray(im[..., ::-1])
                    cv2.rectangle(vis, (box[0] - 4, box[1] - 4), (box[2] + 4, box[3] + 4), (40, 40, 230), 2)
                    path = os.path.join(ALERT_DIR, f"{run['id']}_{self.w.time:.0f}_{k}.jpg")
                    cv2.imwrite(path, vis)
                    img = os.path.relpath(path, ROOT).replace("\\", "/")
                found.append({**{a: v for a, v in o.items() if a not in ("n", "shot", "sw", "sx", "sy")},
                              "x": round(o["x"], 3), "y": round(o["y"], 3), "image": img})
            return True, True, json.dumps({"found": bool(found), "objects": found}, ensure_ascii=False)
        return False, False, ""

    def _seen_at_target(self, cam, b, p):
        """足元が家具 (ちゃぶ台など) に隠れて床に落とせない枠でも、target の方向に見えていれば、そこにいるとみなす
        (姿勢は学習した YOLO のクラス)。アヒルの目は低いので、立った子どもの脚がちゃぶ台に隠れると枠の下端が目より高くなる。"""
        d = cam.ray((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        x, y, _ = self.w.duck.pose()
        tx, ty = float(p["targetX"]), float(p["targetY"])
        off = (math.atan2(d[1], d[0]) - math.atan2(ty - y, tx - x) + math.pi) % (2 * math.pi) - math.pi
        if abs(off) > TARGET_BEARING:
            return None
        return {"x": tx, "y": ty, "posture": b[5], "height": None, "length": None, "score": round(float(b[4]), 2),
                "occluded": True}

    def _step_detect(self, run, el):
        """人を探す: target があればそちらを向き、見えなければ左右に首 (体) を振る。最も確かな検出を返す。"""
        p = run["p"]
        x, y, th = self.w.duck.pose()
        if run["phase"] == "face" and "targetX" in p:
            want = math.atan2(float(p["targetY"]) - y, float(p["targetX"]) - x)
            err = (want - th + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.2 and el < 6.0:              # (回転の指令は MIN_TURN 以上なので、細かくは合わせない)
                self.w.set_manual_cmd(0.0, 0.0, turn_cmd(err, 1.5, 1.0))
                return False, False, ""
            run["phase"], run["t_scan"], run["yaw0"] = "scan", self.w.time, th
        elif run["phase"] == "face":
            run["phase"], run["t_scan"], run["yaw0"] = "scan", self.w.time, th
        ts = self.w.time - run["t_scan"]
        # 見る (0.5 秒ごと)。頭のカメラで YOLO
        if self.w.time >= run["next_look"]:
            run["next_look"] = self.w.time + 0.5
            img = self.w.render("head_camera", 640, 480)
            boxes = self.det.detect([img])[0]
            cam = CameraModel.from_mujoco(self.w.model, self.w.data, "head_camera", 640, 480)
            st = getattr(self.w, "stretch", None)
            rbox = robot_box(cam, st.pose()) if st else None
            if os.environ.get("DETECT_DEBUG"):
                print(f"[detect] t={self.w.time:.1f} ts={ts:.1f} duck={np.round(self.w.duck.pose(), 2).tolist()} "
                      f"boxes={[[int(v) for v in bb[:4]] + [round(float(bb[4]), 2), bb[5] if len(bb) > 5 else None] for bb in boxes]}",
                      flush=True)
            for b in boxes:
                if rbox and covered(b, rbox):
                    continue        # 片付け係の Stretch を人と間違えない
                est = estimate_person(cam, b, 0.0)
                if est is None and "targetX" in p and len(b) > 5:
                    est = self._seen_at_target(cam, b, p)
                if est and (run["best"] is None or est["score"] > run["best"]["score"]):
                    run["best"] = {**est, "box": [int(v) for v in b[:4]]}
                    run["img"] = img
            if run["best"] and run["best"]["posture"] != "unknown" and ts > 1.0:
                self.w.set_manual_cmd(0.0, 0.0, 0.0)
                return True, True, json.dumps({"found": True, **run["best"], "image": self._save_evidence(run, run["id"])},
                                              ensure_ascii=False)
        # 部屋を見回す (sweep=full) ときはゆっくり 1 周回る。見つけるまでは止まらない
        if p.get("sweep") == "full":
            if run["best"] is None or run["best"]["posture"] == "unknown":
                self.w.set_manual_cmd(0.0, 0.0, MIN_TURN + 0.1)
            else:
                self.w.set_manual_cmd(0.0, 0.0, 0.0)
        # まず 1.5 秒は target の方を見たまま。それでも見えなければ左右に振る (±40°)
        elif run["best"] is None and ts > 1.5:
            phase = math.sin((ts - 1.5) * 0.8)
            want = run["yaw0"] + 0.7 * phase
            err = (want - th + math.pi) % (2 * math.pi) - math.pi
            self.w.set_manual_cmd(0.0, 0.0, turn_cmd(err, 2.0, 0.9, dead=0.1))
        else:
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
        if ts >= run["dur"]:
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
            res = {"found": run["best"] is not None, **(run["best"] or {})}
            if run["best"]:
                res["image"] = self._save_evidence(run, run["id"])
            return True, run["best"] is not None, json.dumps(res, ensure_ascii=False)
        return False, False, ""


# ---- 2 台目: Stretch 3 (片付け係)
STRETCH_SERIAL = "stretch-01"
OBJECT_WIDTH = {"button_battery": 0.020, "coin": 0.0235, "marble": 0.017, "toy_block": 0.032, "scissors": 0.05}


class StretchAgent(Agent):
    """Stretch 3 のロボット側。order のノードへ台車で行き、アクション pick (床の物を拾う)・drop (片付けの箱に入れる)。

    pick / drop のノードでは、ノードの位置 (物・箱) が腕の下に来る台車の位置と向きを選んで行く (World.stretch_pick_pose)。
    拾う動きは sim/stretch.py の pick_step / drop_step (計算で決めた動き)。物の位置は見つけたアヒルの推定 (x, y)。
    """

    def __init__(self, world, mqtt, serial=STRETCH_SERIAL):
        super().__init__(world, mqtt, detector=None, serial=serial)

    def status_text(self):
        if self.driving:
            return "移動中"
        if self.action_run is not None:
            return {"pick": "拾っている", "drop": "箱に入れている"}.get(self.action_run.get("type"), "作業中")
        return "待機"

    def publish_factsheet(self):
        self.m.publish(self._topic("factsheet"), self._hdr({
            "typeSpecification": {"seriesName": "Stretch 3", "seriesDescription":
                                  "台車と上下・伸縮する腕を持つ家庭用のロボット (Hello Robot)。床の小さい物を拾って片付ける",
                                  "agvKinematic": "DIFF", "agvClass": "CARRIER", "maxLoadMass": 0.5,
                                  "localizationTypes": ["NATURAL"], "navigationTypes": ["AUTONOMOUS"]},
            "physicalParameters": {"speedMin": 0.0, "speedMax": 0.25, "accelerationMax": 0.5,
                                   "decelerationMax": 0.5, "heightMin": 1.4, "heightMax": 1.4,
                                   "width": 0.34, "length": 0.33},
            "protocolLimits": {}, "protocolFeatures": {"optionalParameters": [], "agvActions": [
                {"actionType": "pick", "actionScopes": ["NODE"], "actionDescription": "床の小さい物を拾う",
                 "actionParameters": [{"key": "x", "valueDataType": "FLOAT"}, {"key": "y", "valueDataType": "FLOAT"},
                                      {"key": "objectType", "valueDataType": "STRING", "isOptional": True}],
                 "resultDescription": "JSON: held"},
                {"actionType": "drop", "actionScopes": ["NODE"], "actionDescription": "持っている物を片付けの箱に入れる",
                 "actionParameters": [], "resultDescription": "JSON: dropped"},
                {"actionType": "cancelOrder", "actionScopes": ["INSTANT"]},
                {"actionType": "stateRequest", "actionScopes": ["INSTANT"]},
                {"actionType": "factsheetRequest", "actionScopes": ["INSTANT"]}]},
            "agvGeometry": {}, "loadSpecification": {}}), retain=True)

    # ---- 体
    def _pose(self):
        return self.w.stretch.pose()

    def _vel(self):
        v, w = self.w.stretch._vw
        return float(v), 0.0, float(w)

    def _stop(self):
        self.w.stretch_stop()
        self.w.stretch.stop()

    def _work_target(self, node):
        for a in node.get("actions", []):
            p = params(a)
            if a["actionType"] == "pick":
                return float(p["x"]), float(p["y"])
            if a["actionType"] == "drop":
                return self.w.tidy_box_xyz()[:2]
        return None

    def _needs_nav(self, node):
        return self._work_target(node) is not None

    def _start_nav(self, node, pos, dev, x, y):
        target = self._work_target(node)
        if target is not None:
            pp = self.w.stretch_pick_pose(target)
            if pp is None:
                return False
            return self.w.stretch_go_to(*pp)
        return self.w.stretch_go_to(pos["x"], pos["y"], pos.get("theta"), tolerance=max(dev, 0.1))

    def _nav_status(self):
        return self.w.stretch_nav_status()

    # ---- アクション
    def _start_action(self, a):
        run = {"t0": self.w.time, "p": params(a), "type": a["actionType"], "id": a["actionId"]}
        if run["type"] in ("pick", "drop"):
            self.w.stretch.reset_task()
        return run

    def _step_action(self, a, run):
        t, p, st = run["type"], run["p"], self.w.stretch
        if t == "pick":
            r = st.pick_step((float(p["x"]), float(p["y"]), 0.0), yaw=float(p.get("yaw", 0.0)),
                             width=OBJECT_WIDTH.get(p.get("objectType"), 0.025))
        elif t == "drop":
            bx, by, bz = self.w.tidy_box_xyz()
            r = st.drop_step((bx, by, bz + 0.12))
        elif t == "wait":
            return self.w.time - run["t0"] >= float(p.get("duration", 1.0)), True, ""
        else:
            return True, False, f"unknown action {t}"
        if os.environ.get("STRETCH_DEBUG") and (st.phase != run.get("phase") or r != "running"):
            run["phase"] = st.phase
            print(f"[stretch] t={self.w.time:.1f} {t} {st.phase} {r} {st.reason} base={np.round(st.pose(), 3).tolist()} "
                  f"grasp={np.round(st.grasp_point(), 3).tolist()} target=({p.get('x')}, {p.get('y')}) "
                  f"joints={ {k: round(v, 3) for k, v in st.joints().items()} }", flush=True)
        if r == "running":
            return False, False, ""
        if r == "done":
            return True, True, json.dumps({"held": t == "pick", "dropped": t == "drop"}, ensure_ascii=False)
        return True, False, json.dumps({"error": st.reason}, ensure_ascii=False)
