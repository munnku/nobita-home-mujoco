"""VDA 5050 v2.0 (https://github.com/VDA5050/VDA5050) のロボット側 (Agent) と司令塔側の補助 (make_order)。

トピック: uagv/v2/<manufacturer>/<serialNumber>/{order, instantActions, state, visualization, connection, factsheet}
使う範囲:
- order: nodes (nodePosition x, y, theta, allowedDeviationXY) と edges。ノードを順にたどり、着いたらノードの actions を順に行う
- actions: detectObject (VDA 5050 の定義済みアクション。objectType=person、頭のカメラと YOLO で人を探す)、
           speak (この家で足したアクション。話す内容を home/speech に出す)、wait
- instantActions: cancelOrder, startPause, stopPause, stateRequest, factsheetRequest
- state: 1 秒ごとと変化したとき。agvPosition, driving, nodeStates, edgeStates, actionStates, errors など
アヒルが自分で経路を作る (地図の上の A*) ので、ノードは「行き先」だけでよい。
"""
import datetime
import json
import math
import queue

from .perception import CameraModel, estimate_person

VERSION = "2.0.0"
MANUFACTURER = "NobitaHome"
SERIAL = "microduck-01"
MAP_ID = "nobita-1F"


def topic(name, manufacturer=MANUFACTURER, serial=SERIAL):
    return f"uagv/v2/{manufacturer}/{serial}/{name}"


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def params(action):
    return {p["key"]: p["value"] for p in action.get("actionParameters", [])}


def make_order(order_id, update_id, points, actions_at_last=(), header_id=0):
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
            "serialNumber": SERIAL, "orderId": order_id, "orderUpdateId": update_id,
            "nodes": nodes, "edges": edges}


def action(action_type, action_id, **kw):
    return {"actionType": action_type, "actionId": action_id, "blockingType": "HARD",
            "actionParameters": [{"key": k, "value": v} for k, v in kw.items()]}


class Agent:
    """ロボット側。シミュレーションのスレッドから tick() を毎回呼ぶ (MQTT の受信はキューで受け渡す)。"""

    def __init__(self, world, mqtt, detector):
        self.w = world
        self.m = mqtt
        self.det = detector
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
        self.m.subscribe(topic("order"), lambda t, p: self.inbox.put(("order", p)))
        self.m.subscribe(topic("instantActions"), lambda t, p: self.inbox.put(("instant", p)))
        self.m.publish(topic("connection"), self._hdr({"connectionState": "ONLINE"}), retain=True, qos=1)
        self.publish_factsheet()

    def status_text(self):
        """動画に出す今の様子 (日本語)。"""
        if self.paused:
            return "一時停止"
        if self.driving:
            return "移動中"
        if self.action_run is not None:
            return {"detectObject": "人を確かめている", "speak": "話している", "wait": "待っている"}.get(
                self.action_run.get("type"), "作業中")
        return "待機"

    # ---- MQTT へ出すもの
    def _hdr(self, body):
        self.header += 1
        return {"headerId": self.header, "timestamp": now_iso(), "version": VERSION,
                "manufacturer": MANUFACTURER, "serialNumber": SERIAL, **body}

    def publish_factsheet(self):
        self.m.publish(topic("factsheet"), self._hdr({
            "typeSpecification": {"seriesName": "Microduck", "seriesDescription":
                                  "二足歩行の小型ロボット (Pollen Robotics)。学習済みの方策で歩き、頭のカメラで見る",
                                  "agvKinematic": "DIFF", "agvClass": "CARRIER", "maxLoadMass": 0,
                                  "localizationTypes": ["NATURAL"], "navigationTypes": ["AUTONOMOUS"]},
            "physicalParameters": {"speedMin": 0.0, "speedMax": 0.3, "accelerationMax": 0.5,
                                   "decelerationMax": 0.5, "heightMin": 0.25, "heightMax": 0.25,
                                   "width": 0.16, "length": 0.12},
            "protocolLimits": {}, "protocolFeatures": {"optionalParameters": [], "agvActions": [
                {"actionType": "detectObject", "actionScopes": ["NODE", "INSTANT"],
                 "actionParameters": [{"key": "objectType", "valueDataType": "STRING"},
                                      {"key": "targetX", "valueDataType": "FLOAT", "isOptional": True},
                                      {"key": "targetY", "valueDataType": "FLOAT", "isOptional": True},
                                      {"key": "duration", "valueDataType": "FLOAT", "isOptional": True}],
                 "resultDescription": "JSON: found, score, posture, x, y"},
                {"actionType": "speak", "actionScopes": ["NODE", "INSTANT"],
                 "actionParameters": [{"key": "text", "valueDataType": "STRING"}]},
                {"actionType": "wait", "actionScopes": ["NODE"],
                 "actionParameters": [{"key": "duration", "valueDataType": "FLOAT"}]},
                {"actionType": "cancelOrder", "actionScopes": ["INSTANT"]},
                {"actionType": "startPause", "actionScopes": ["INSTANT"]},
                {"actionType": "stopPause", "actionScopes": ["INSTANT"]},
                {"actionType": "stateRequest", "actionScopes": ["INSTANT"]},
                {"actionType": "factsheetRequest", "actionScopes": ["INSTANT"]}]},
            "agvGeometry": {}, "loadSpecification": {}}), retain=True)

    def publish_state(self):
        x, y, th = self.w.duck.pose()
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
            "velocity": {"vx": float(self.w.duck.command[0]), "vy": 0.0, "omega": float(self.w.duck.command[2])},
            "batteryState": {"batteryCharge": 80.0, "charging": False},
            "actionStates": self.action_states[-20:],
            "errors": self.errors, "information": [],
            "safetyState": {"eStop": "NONE", "fieldViolation": False},
            "room": self.w.room_of(x, y),
        })
        self.m.publish(topic("state"), state)
        self.m.publish(topic("visualization"), self._hdr({"agvPosition": state["agvPosition"],
                                                          "velocity": state["velocity"]}))

    # ---- 毎回
    def tick(self):
        while not self.inbox.empty():
            kind, msg = self.inbox.get()
            if kind == "order":
                self._on_order(msg)
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
        self.w.stop()

    def _on_instant(self, a):
        t = a["actionType"]
        if t == "cancelOrder":
            if self.order:
                for n in self.order["nodes"][self.node_idx:]:
                    for na in n.get("actions", []):
                        self._set_action_state(na, "FAILED", "canceled")
            self.order = None
            self.w.stop()
            self.driving = False
            self._set_action_state(a, "FINISHED")
        elif t == "startPause":
            self.paused = True
            self.w.stop()
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
        elif t in ("speak", "detectObject"):
            # 即時のアクションとしても受け付ける (いまの場所で行う)
            self.order = self.order or {"orderId": f"instant-{a['actionId']}", "orderUpdateId": 0,
                                        "nodes": [], "edges": []}
            x, y, _ = self.w.duck.pose()
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
            x, y, _ = self.w.duck.pose()
            dev = pos.get("allowedDeviationXY", 0.1) if pos else 0.0
            if pos is None or math.hypot(pos["x"] - x, pos["y"] - y) <= max(dev, 0.12):
                node["_reached"] = True
            elif not o.get("_navigating"):
                gx, gy, gth, tol = pos["x"], pos["y"], pos.get("theta"), max(dev, 0.12)
                look = self._look_target(node)
                if look is not None:
                    # 人を見に行くときは、人そのものではなく人を見通せる立ち位置へ行き、人の方を向く
                    vp = self.w.map.viewpoint(look, (x, y), r_min=0.8, r_max=max(1.3, min(dev, 2.0)))
                    if vp is not None:
                        gx, gy, tol = vp[0], vp[1], 0.12
                        gth = math.atan2(look[1] - vp[1], look[0] - vp[0])
                ok = self.w.go_to(gx, gy, gth, tolerance=tol)
                o["_navigating"] = True
                self.driving = ok
                if not ok:
                    self._fail_order(f"no path to node {node['nodeId']}")
                return
            else:
                st = self.w.nav_status()
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
        self.errors = [{"errorType": "navigationError", "errorLevel": "WARNING", "errorDescription": why}]
        for n in self.order["nodes"][self.node_idx:]:
            for a in n.get("actions", []):
                self._set_action_state(a, "FAILED", why)
        self.node_idx = len(self.order["nodes"])
        self.driving = False
        self.w.stop()

    # ---- アクション
    def _start_action(self, a):
        p = params(a)
        t = a["actionType"]
        run = {"t0": self.w.time, "p": p, "type": t}
        if t == "speak":
            text = str(p.get("text", ""))
            dur = 1.0 + 0.12 * len(text)
            self.speech = (self.w.time + dur, text)
            x, y, _ = self.w.duck.pose()
            self.m.publish("home/speech", {"speaker": SERIAL, "text": text, "room": self.w.room_of(x, y),
                                           "time": round(self.w.time, 1)})
            run["dur"] = dur
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
        if t == "detectObject":
            return self._step_detect(run, el)
        return True, False, f"unknown action {t}"

    def _step_detect(self, run, el):
        """人を探す: target があればそちらを向き、見えなければ左右に首 (体) を振る。最も確かな検出を返す。"""
        p = run["p"]
        x, y, th = self.w.duck.pose()
        if run["phase"] == "face" and "targetX" in p:
            want = math.atan2(float(p["targetY"]) - y, float(p["targetX"]) - x)
            err = (want - th + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.15 and el < 6.0:
                self.w.set_manual_cmd(0.0, 0.0, max(-1.0, min(1.0, 1.5 * err + math.copysign(0.3, err))))
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
            for b in boxes:
                est = estimate_person(cam, b, 0.0)
                if est and (run["best"] is None or est["score"] > run["best"]["score"]):
                    run["best"] = est
            if run["best"] and run["best"]["posture"] != "unknown" and ts > 1.0:
                self.w.set_manual_cmd(0.0, 0.0, 0.0)
                return True, True, json.dumps({"found": True, **run["best"]}, ensure_ascii=False)
        # 見えないうちは左右に振る (±40°)
        if run["best"] is None:
            phase = math.sin(ts * 0.8)
            want = run["yaw0"] + 0.7 * phase
            err = (want - th + math.pi) % (2 * math.pi) - math.pi
            self.w.set_manual_cmd(0.0, 0.0, max(-0.8, min(0.8, 2.0 * err)))
        else:
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
        if ts >= run["dur"]:
            self.w.set_manual_cmd(0.0, 0.0, 0.0)
            res = {"found": run["best"] is not None, **(run["best"] or {})}
            return True, run["best"] is not None, json.dumps(res, ensure_ascii=False)
        return False, False, ""
