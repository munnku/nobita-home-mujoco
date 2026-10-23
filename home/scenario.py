"""人役の時刻表 (houses/<家>/scenarios/*.yaml) をシミュレーションの時刻に合わせて実行する。

人は置物なので、イベントの時刻に指定の場所へ瞬間移動し、姿勢 (stand / lie) を変える。
walk: true のイベントは、1 階の中なら地図の経路に沿って歩く速さで動かす (立った姿のまま滑る)。
扉のイベント ({t, door, do: unlock / lock / open / close}) は家の人が鍵や扉を操作したことを表す。
いまの場面は MQTT の sim/scenario に出す (評価用。司令塔はこれを見ない)。
start_clock があれば家の時計 (home/clock, "HH:MM:SS") をシミュレーションの時刻に合わせて進める。
声: {t, who, say: "いたっ"} はその時刻に人役が声を出す。replies: はアヒルやスピーカーの問いかけへの返事 (home/voice.py)。
mode: alone ならおるすばんモード (home/mode を retain で出す。司令塔の子どもの見守りが働く)。
歩く速さは walk のイベントの speed [m/s] で変えられる。
警告への反応 (react): {人: {on: warn, ignore: 無視する回数, delay: 止まるまでの秒数, near: アヒルがこれより近づけば止まる}}。
  止まると、その人の unless_stopped: true のイベント (止められなければ行うこと) は行わない。
危ない場所 (home.yaml の hazards) に人が入ったら sim/scenario に hazard_entered を出す (評価用)。
"""
import math
import os

import yaml

from .voice import Voices

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
        self.voices = Voices(world, mqtt, self.sc, {i: n for n, i in self.index.items()}, on_intent=self._on_intent)
        self.react = self.sc.get("react") or {}
        self.warned = {}          # 人 → 警告を聞いた回数
        self.stopped = set()      # 警告で止まった人
        self.stop_at = {}         # 人 → 止まる時刻
        self.in_hazard = {}
        mqtt.publish("home/mode", self.sc.get("mode", "family"), retain=True)

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
            if e.get("unless_stopped") and i in self.stopped:
                continue
            if "say" in e:
                self.voices.say(i, e["say"])
                continue
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
        for i, t in list(self.stop_at.items()):
            if self.w.time >= t:
                self._stop(i)
        for i, wk in list(self.walking.items()):
            wk["s"] += wk["speed"] * 0.02
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
        self.voices.tick()
        self._near_robot()
        self._hazards()

    # ---- 警告への反応
    def _on_intent(self, i, intent):
        name = next(n for n, k in self.index.items() if k == i)
        r = self.react.get(name)
        if not r or intent != r.get("on", "warn") or i in self.stopped or i in self.stop_at:
            return
        self.warned[i] = self.warned.get(i, 0) + 1
        if self.warned[i] > int(r.get("ignore", 0)):
            self.stop_at.setdefault(i, self.w.time + float(r.get("delay", 1.0)))
        else:
            print(f"[scenario] t={self.w.time:.1f} {name} は警告を聞かずに進む ({self.warned[i]} 回目)", flush=True)

    def _near_robot(self):
        x, y, _ = self.w.duck.pose()
        for name, r in self.react.items():
            i = self.index[name]
            st = self.w.person_state.get(i)
            if r.get("near") and st and i not in self.stopped and i not in self.stop_at \
                    and math.hypot(st["x"] - x, st["y"] - y) < float(r["near"]):
                self.stop_at[i] = self.w.time + float(r.get("delay", 1.0))

    def _stop(self, i):
        name = next(n for n, k in self.index.items() if k == i)
        self.walking.pop(i, None)
        self.stop_at.pop(i, None)
        self.stopped.add(i)
        st = self.w.person_state[i]
        self.w.map.update()
        self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": name, "stopped": True,
                                        "x": round(st["x"], 2), "y": round(st["y"], 2)})
        print(f"[scenario] t={self.w.time:.1f} {name} が止まった", flush=True)

    def _hazards(self):
        mod = self.w.module
        for h in self.home.get("hazards", []):
            x0, x1, y0, y1 = (v * mod for v in h["rect"])
            for i, st in self.w.person_state.items():
                inside = bool(st) and st["level"] == "1F" and x0 <= st["x"] <= x1 and y0 <= st["y"] <= y1
                key = (h["name"], i)
                if inside and not self.in_hazard.get(key):
                    name = next(n for n, k in self.index.items() if k == i)
                    self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": name,
                                                    "hazard_entered": h["name"]})
                    print(f"[scenario] t={self.w.time:.1f} {name} が危ない場所 ({h['label']}) に入った", flush=True)
                self.in_hazard[key] = inside

    def _start_walk(self, i, cur, place, e):
        mod = self.w.module
        goal = (place["x"] * mod, place["y"] * mod)
        path = self.w.map.plan((cur["x"], cur["y"]), goal)
        if not path:
            # 歩く人自身の当たり判定が出発点をふさいでいることがある: いったん地図から外して探し、元に戻す
            self.w.place_person(i, None, None)
            path = self.w.map.plan((cur["x"], cur["y"]), goal)
            self.w.place_person(i, cur["x"], cur["y"], cur["yaw"], cur["pose"], cur["level"])
        if not path:
            return False
        path = [(cur["x"], cur["y"])] + path[1:-1] + [goal]
        length = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
        self.walking[i] = {"path": path, "s": 0.0, "len": length, "pose": e.get("pose", "stand"),
                           "speed": float(e.get("speed", WALK_SPEED)),
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
