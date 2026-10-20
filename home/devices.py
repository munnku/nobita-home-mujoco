"""家の IoT 機器 (シミュレーション) を Homie の機器として MQTT に出す。

- 見守りカメラ (cameras.yaml): MuJoCo のカメラで撮り、YOLO で人を見つけて床の上の位置と姿勢を出す
- 人感センサー: 部屋ごと。人役がその部屋にいれば motion = true (焦電センサーの代わり)
- 照明: 部屋ごと。on を /set で切り替えると MuJoCo の照明が点く・消える
- スマートロック: 玄関と勝手口。locked を /set で切り替える。扉の開閉センサー (open) も出す
MQTT からの書き込みは受信スレッドで来るので、キューにためてシミュレーションのスレッドで反映する。
"""
import os
import queue

import yaml

from .homie import HomieDevice
from .perception import CameraModel, estimate_person


def _dev_id(prefix, name):
    return f"{prefix}-{name.replace('_', '-')}"


class HomeDevices:
    def __init__(self, world, mqtt, detector, house_dir):
        self.w = world
        self.m = mqtt
        self.det = detector
        self.pending = queue.Queue()
        with open(os.path.join(house_dir, "cameras.yaml"), encoding="utf-8") as f:
            self.cam_cfg = yaml.safe_load(f)
        rooms = {r["name"]: r for r in world.layout["rooms"]}
        self.rooms = rooms
        self.next_cam_t = 0.0
        self.next_sensor_t = 0.0
        self.latest = {}          # カメラ名 → (時刻, 検出のリスト)、録画の重ね描き用
        self.latest_images = {}

        # 見守りカメラ
        self.cams = {}
        for c in self.cam_cfg["cameras"]:
            self.cams[c["name"]] = HomieDevice(mqtt, _dev_id("cam", c["room"]), f"見守りカメラ ({rooms[c['room']]['label']})", {
                "camera": {"name": "人の検出", "type": "camera", "properties": {
                    "persons": {"name": "見つけた人 (位置 m・姿勢)", "datatype": "string", "format": "json"},
                    "count": {"name": "人数", "datatype": "integer"},
                    "room": {"name": "部屋", "datatype": "string"},
                }}})
            self.cams[c["name"]].set("camera", "room", c["room"])
        # 人感センサー
        self.motion = {}
        for name, r in rooms.items():
            self.motion[name] = HomieDevice(mqtt, _dev_id("motion", name), f"人感センサー ({r['label']})", {
                "sensor": {"name": "人感", "type": "motion", "properties": {
                    "motion": {"name": "人がいる", "datatype": "boolean"}}}})
        # 照明
        self.lights = {}
        for name in world.layout.get("lights", []):
            self.lights[name] = HomieDevice(
                mqtt, _dev_id("light", name), f"照明 ({rooms[name]['label']})",
                {"light": {"name": "照明", "type": "light", "properties": {
                    "on": {"name": "点灯", "datatype": "boolean", "settable": True}}}},
                on_set=lambda n, p, v, room=name: self.pending.put(("light", room, v)))
            self.lights[name].set("light", "on", world.light_is_on(name))
        # スマートスピーカー (話す内容を home/speech に出す)
        self.speakers = {}
        for name in ("nobita", "dk", "ima"):
            self.speakers[name] = HomieDevice(
                mqtt, _dev_id("speaker", name), f"スマートスピーカー ({rooms[name]['label']})",
                {"speaker": {"name": "スピーカー", "type": "speaker", "properties": {
                    "say": {"name": "話す", "datatype": "string", "settable": True}}}},
                on_set=lambda n, p, v, room=name: self.pending.put(("say", room, v)))
        # スマートロックと扉の開閉センサー
        self.locks = {}
        self.locked = {}
        for name, d in world.layout["doors"].items():
            if not d.get("lock"):
                continue
            self.locked[name] = True
            self.locks[name] = HomieDevice(
                mqtt, _dev_id("lock", name), f"スマートロック ({name})",
                {"lock": {"name": "錠", "type": "lock", "properties": {
                    "locked": {"name": "施錠", "datatype": "boolean", "settable": True}}},
                 "contact": {"name": "開閉センサー", "type": "contact", "properties": {
                     "open": {"name": "開いている", "datatype": "boolean"}}}},
                on_set=lambda n, p, v, door=name: self.pending.put(("lock", door, v)))
            self.locks[name].set("lock", "locked", True)

    # ---- シミュレーションのスレッドから毎回呼ぶ
    def tick(self):
        while not self.pending.empty():
            kind, name, value = self.pending.get()
            if kind == "light":
                self.w.set_light(name, bool(value))
            elif kind == "say":
                self.m.publish("home/speech", {"speaker": f"speaker-{name}", "text": str(value), "room": name,
                                               "time": round(self.w.time, 1)})
                self.speakers[name].set("speaker", "say", str(value), force=True)
            elif kind == "lock":
                self.locked[name] = bool(value)
                if self.locked[name] and self.w.door_is_open(name):
                    self.w.set_door(name, False)       # 開いていれば閉めてから施錠 (オートロック)
        t = self.w.time
        if t >= self.next_sensor_t:
            self.next_sensor_t = t + 0.5
            self._sensors()
        if t >= self.next_cam_t:
            self.next_cam_t = t + 1.0 / self.cam_cfg.get("rate", 1.0)
            self._cameras()

    def try_open_door(self, name):
        """人役が扉を開ける (鍵がかかっていれば開かない)。"""
        if self.locked.get(name):
            return False
        self.w.set_door(name, True)
        return True

    def _sensors(self):
        occupied = {name: False for name in self.rooms}
        for st in self.w.person_state.values():
            if st:
                room = self.w.room_of(st["x"], st["y"], st["level"])
                if room:
                    occupied[room] = True
        for name, dev in self.motion.items():
            dev.set("sensor", "motion", occupied[name])
        for name, dev in self.lights.items():
            dev.set("light", "on", self.w.light_is_on(name))
        for name, dev in self.locks.items():
            dev.set("lock", "locked", self.locked[name])
            dev.set("contact", "open", bool(self.w.door_is_open(name)))

    def _cameras(self):
        cfg = self.cam_cfg
        names = [c["name"] for c in cfg["cameras"]]
        images = [self.w.render(n, cfg["width"], cfg["height"]) for n in names]
        boxes = self.det.detect(images)
        for c, img, bxs in zip(cfg["cameras"], images, boxes):
            cam = CameraModel.from_mujoco(self.w.model, self.w.data, c["name"], cfg["width"], cfg["height"])
            floor_z = self.w.layout["levels"][self.rooms[c["room"]]["level"]]["z"]
            persons = []
            for b in bxs:
                p = estimate_person(cam, b, floor_z)
                if p:
                    # 足元が床の高さの違う部屋 (玄関の土間) に落ちたら、その床の高さで計算し直す
                    r = self.rooms.get(self.w.room_of(p["x"], p["y"], self.rooms[c["room"]]["level"]) or "")
                    if r and r.get("z"):
                        p = estimate_person(cam, b, floor_z + r["z"]) or p
                    else:
                        # 推定を部屋の中 (壁から 0.25 m 内側) に収める。壁際の人は壁の向こうに推定されることがある。
                        # 推定が入っている部屋に収める (開いた扉やふすま越しに隣の部屋の人が見えることもあるので、
                        # カメラの部屋に押し込まない)。どの部屋にも入っていなければカメラの部屋に収める
                        home_room = r if r is not None else self.rooms[c["room"]]
                        x0, x1, y0, y1 = (v * self.w.module for v in home_room["rect"])
                        p["x"] = round(min(max(p["x"], x0 + 0.25), x1 - 0.25), 2)
                        p["y"] = round(min(max(p["y"], y0 + 0.25), y1 - 0.25), 2)
                    p["room"] = self.w.room_of(p["x"], p["y"], self.rooms[c["room"]]["level"]) or c["room"]
                    p["box"] = [int(v) for v in b[:4]]
                    persons.append(p)
            self.cams[c["name"]].set("camera", "persons", persons, force=True)
            self.cams[c["name"]].set("camera", "count", len(persons))
            self.latest[c["name"]] = (self.w.time, persons)
            self.latest_images[c["name"]] = img
