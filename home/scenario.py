"""人役の時刻表 (houses/<家>/scenarios/*.yaml) をシミュレーションの時刻に合わせて実行する。

人は置物なので、イベントの時刻に指定の場所へ瞬間移動し、姿勢 (stand / lie) を変える。
いまの場面は MQTT の sim/scenario に出す (評価用。司令塔はこれを見ない)。
"""
import math
import os

import yaml


class Scenario:
    def __init__(self, world, mqtt, house_dir, name):
        with open(os.path.join(house_dir, "home.yaml"), encoding="utf-8") as f:
            self.home = yaml.safe_load(f)
        with open(os.path.join(house_dir, "scenarios", f"{name}.yaml"), encoding="utf-8") as f:
            self.sc = yaml.safe_load(f)
        self.w = world
        self.m = mqtt
        self.index = {p["name"]: i for i, p in enumerate(self.home["people"])}
        self.events = sorted(self.sc["events"], key=lambda e: e["t"])
        self.k = 0
        self.duration = self.sc.get("duration", 300)

    def tick(self):
        mod = self.w.module
        changed = False
        while self.k < len(self.events) and self.events[self.k]["t"] <= self.w.time:
            e = self.events[self.k]
            self.k += 1
            i = self.index[e["who"]]
            place = self.home["places"].get(e["at"])
            if place is None:
                self.w.place_person(i, None, None, update_map=False)
            else:
                self.w.place_person(i, place["x"] * mod, place["y"] * mod, math.radians(place["yaw"]),
                                    e.get("pose", "stand"), place.get("level", "1F"), update_map=False)
            changed = True
            self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": e["who"], "at": e["at"],
                                            "pose": e.get("pose", "stand")})
            print(f"[scenario] t={self.w.time:.1f} {e['who']} -> {e['at']} ({e.get('pose', 'stand')})", flush=True)
        if changed:
            self.w.map.update()

    def done(self):
        return self.w.time >= self.duration
