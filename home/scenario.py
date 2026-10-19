"""人役の時刻表 (houses/<家>/scenarios/*.yaml) をシミュレーションの時刻に合わせて実行する。

人は置物なので、イベントの時刻に指定の場所へ瞬間移動し、姿勢 (stand / lie) を変える。
walk: true のイベントは、1 階の中なら地図の経路に沿って歩く速さで動かす (立った姿のまま滑る)。
扉のイベント ({t, door, do: unlock / lock / open / close}) は家の人が鍵や扉を操作したことを表す。
いまの場面は MQTT の sim/scenario に出す (評価用。司令塔はこれを見ない)。
start_clock があれば家の時計 (home/clock, "HH:MM:SS") をシミュレーションの時刻に合わせて進める。
"""
import math
import os

import yaml

WALK_SPEED = 0.8     # 人役の歩く速さ [m/s]


class Scenario:
    def __init__(self, world, mqtt, house_dir, name, devices=None):
        with open(os.path.join(house_dir, "home.yaml"), encoding="utf-8") as f:
            self.home = yaml.safe_load(f)
        with open(os.path.join(house_dir, "scenarios", f"{name}.yaml"), encoding="utf-8") as f:
            self.sc = yaml.safe_load(f)
        self.w = world
        self.m = mqtt
        self.devices = devices
        h, mi, se = (int(v) for v in str(self.sc.get("start_clock", "12:00:00")).split(":"))
        self.clock0 = h * 3600 + mi * 60 + se
        self.index = {p["name"]: i for i, p in enumerate(self.home["people"])}
        self.events = sorted(self.sc["events"], key=lambda e: e["t"])
        self.k = 0
        self.walking = {}         # 歩いている人 → 経路と進んだ距離
        self.duration = self.sc.get("duration", 300)

    def tick(self):
        mod = self.w.module
        changed = False
        while self.k < len(self.events) and self.events[self.k]["t"] <= self.w.time:
            e = self.events[self.k]
            self.k += 1
            if "door" in e:
                self._door(e)
                continue
            i = self.index[e["who"]]
            place = self.home["places"].get(e["at"])
            self.walking.pop(i, None)          # 歩いている途中なら、新しいイベントを優先する
            cur = self.w.person_state.get(i)
            if (e.get("walk") and place is not None and cur and cur["level"] == "1F"
                    and place.get("level", "1F") == "1F" and cur["pose"] == "stand"):
                if self._start_walk(i, cur, place, e):
                    continue
            if place is None:
                self.w.place_person(i, None, None, update_map=False)
            else:
                self.w.place_person(i, place["x"] * mod, place["y"] * mod, math.radians(place["yaw"]),
                                    e.get("pose", "stand"), place.get("level", "1F"), update_map=False,
                                    dz=place.get("dz", 0.0))
            changed = True
            self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": e["who"], "at": e["at"],
                                            "pose": e.get("pose", "stand")})
            print(f"[scenario] t={self.w.time:.1f} {e['who']} -> {e['at']} ({e.get('pose', 'stand')})", flush=True)
        # 歩いている人を進める (経路に沿って WALK_SPEED で、進む向きを向く)
        for i, wk in list(self.walking.items()):
            wk["s"] += WALK_SPEED * 0.02
            x, y, yaw = self._along(wk["path"], wk["s"])
            if wk["s"] >= wk["len"]:
                gx, gy, gyaw = wk["goal"]
                self.w.place_person(i, gx, gy, gyaw, wk["pose"], "1F", update_map=False)
                del self.walking[i]
                changed = True
            else:
                self.w.place_person(i, x, y, yaw, "stand", "1F", update_map=False)
        if changed:
            self.w.map.update()

    def _start_walk(self, i, cur, place, e):
        mod = self.w.module
        goal = (place["x"] * mod, place["y"] * mod)
        path = self.w.map.plan((cur["x"], cur["y"]), goal)
        if not path:
            return False
        path = [(cur["x"], cur["y"])] + path[1:-1] + [goal]
        length = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
        self.walking[i] = {"path": path, "s": 0.0, "len": length, "pose": e.get("pose", "stand"),
                           "goal": (goal[0], goal[1], math.radians(place["yaw"]))}
        self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": e["who"], "at": e["at"],
                                        "pose": e.get("pose", "stand"), "walk": True})
        print(f"[scenario] t={self.w.time:.1f} {e['who']} walks to {e['at']} ({length:.1f} m)", flush=True)
        return True

    @staticmethod
    def _along(path, s):
        for a, b in zip(path[:-1], path[1:]):
            ln = math.dist(a, b)
            if s <= ln and ln > 0:
                k = s / ln
                return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, math.atan2(b[1] - a[1], b[0] - a[0])
            s -= ln
        a, b = path[-2], path[-1]
        return b[0], b[1], math.atan2(b[1] - a[1], b[0] - a[0])

    def _door(self, e):
        d, do = e["door"], e["do"]
        dev = self.devices
        if do == "unlock" and dev:
            dev.locked[d] = False
        elif do == "lock" and dev:
            dev.locked[d] = True
        elif do == "open":
            ok = dev.try_open_door(d) if dev else self.w.set_door(d, True) or True
            if not ok:
                print(f"[scenario] {d} is locked", flush=True)
        elif do == "close":
            self.w.set_door(d, False)
        self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "door": d, "do": do})
        print(f"[scenario] t={self.w.time:.1f} door {d}: {do}", flush=True)

    def clock(self):
        t = int(self.clock0 + self.w.time) % 86400
        return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"

    def done(self):
        return self.w.time >= self.duration
