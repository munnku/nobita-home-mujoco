"""一続きの場面 (物語): 場面 (home/scenario.py) を「章」でつなぎ、前の出来事が終わったら次の章を始める。

場面のファイル (houses/<家>/scenarios/) は絶対時刻の時刻表なので、ロボットの動きにかかる時間がぶれると、
いくつもの場面を 1 本につなげられない。物語 (houses/<家>/stories/<名前>.yaml) は章の並びで書く:

    mode: away                      # 家にだれもいない (子どもが帰ってきたら司令塔がおるすばんモードにする)
    stretch: true
    duration: 2400                  # 長さの上限 [s]
    events: [...]                   # 始めのイベント (場面と同じ書き方)
    chapters:
      - title: "1. おかえり"         # 録画の画面に出す章の題 (無ければ出さない)
        start_after: {at: 0}        # 始まり: 時刻 at、または
        # start_after: {kind: greet, delay: 4}       司令塔のログ (home/log) の kind が出てから delay 秒後
        # start_after: {done: true, delay: 5}        前の章のイベントが全部済み、人が歩き終わってから delay 秒後
        # max_wait: 900                              前の章の始まりからこれだけ待っても条件がそろわなければ始める
        events: [...]               # 時刻は章の始まりからの秒。{t, objects: [{kind, at}]} で物を床に置ける。
                                    # {t, who, stand_up: true} は、まだ横になっていれば起き上がる
        replies: {...}              # この章の台本の返事 (章の始まりで入れ替える)
        react: {...}                # この章の警告への反応 (章の始まりで入れ替え、止まった記録も消す)
    end_after: {kind: hide_found, delay: 15}   # 最後の章のあと、この出来事から delay 秒で終わる

歩く・声・扉・物・警告への反応は場面 (Scenario) のものをそのまま使う。章が始まると sim/scenario に {chapter, title} を出す。
"""
import math
import os
import queue

import yaml

from .scenario import Scenario
from .voice import Voices


class Story(Scenario):
    def __init__(self, world, mqtt, house_dir, name, devices=None):
        # Scenario.__init__ は scenarios/ を読むので使わず、同じ属性をここで用意する
        with open(os.path.join(house_dir, "home.yaml"), encoding="utf-8") as f:
            self.home = yaml.safe_load(f)
        with open(os.path.join(house_dir, "stories", f"{name}.yaml"), encoding="utf-8") as f:
            self.sc = yaml.safe_load(f)
        self.w = world
        self.m = mqtt
        self.devices = devices
        h, mi, se = (int(v) for v in str(self.sc.get("start_clock", "15:00:00")).split(":"))
        self.clock0 = h * 3600 + mi * 60 + se
        self.index = {p["name"]: i for i, p in enumerate(self.home["people"])}
        self.events = sorted(self.sc.get("events") or [], key=lambda e: e["t"])
        self.k = 0
        self.walking = {}
        self.duration = self.sc.get("duration", 2400)
        self.voices = Voices(world, mqtt, self.sc, {i: n for n, i in self.index.items()}, on_intent=self._on_intent)
        self.react = self.sc.get("react") or {}
        self.warned, self.stopped, self.stop_at, self.in_hazard = {}, set(), {}, {}
        mqtt.publish("home/mode", self.sc.get("mode", "family"), retain=True)
        self.obj_used = {}
        self._place_objects()
        # 章
        self.chapters = self.sc.get("chapters") or []
        self.ch = -1                 # 始まった章の番号
        self.ch_start = 0.0
        self.ch_done_t = None        # いまの章のイベントが全部済み、人が歩き終わった時刻
        self.ch_end_k = 0            # いまの章の最後のイベントが events のどこまでか
        self.later = []              # 物を置くイベント (Scenario は扱わないので、ここで行う)
        self.seen = {}               # 司令塔のログの kind → [(時刻, 中身)]
        self.logs = queue.Queue()
        mqtt.subscribe("home/log", lambda t, p: self.logs.put(p) if isinstance(p, dict) else None)
        self.end_t = None
        self.on_chapter = None       # 章が始まったときに呼ぶ (録画の題)

    # ---- 毎ステップ
    def tick(self):
        while not self.logs.empty():
            p = self.logs.get()
            if p.get("kind"):
                self.seen.setdefault(p["kind"], []).append((self.w.time, p))
        self._chapters()
        for e in [e for e in self.later if e["t"] <= self.w.time]:
            self.later.remove(e)
            if "objects" in e:
                self._objects(e["objects"])
            else:
                self._stand_up(e["who"])
        super().tick()
        if self.ch >= 0 and self.ch_done_t is None and self.k >= self.ch_end_k and not self.walking \
                and not self.later and not self.voices.pending:
            self.ch_done_t = self.w.time
        end = self.sc.get("end_after")
        if end and self.end_t is None and self.ch == len(self.chapters) - 1:
            t = self._kind_time(end)
            if t is not None:
                self.end_t = t + float(end.get("delay", 0.0))
                print(f"[story] t={self.w.time:.1f} 終わりの条件がそろった ({end['kind']}) → {self.end_t:.1f} s に終わる",
                      flush=True)

    def finished(self):
        return self.end_t is not None and self.w.time >= self.end_t

    # ---- 章
    def _kind_time(self, cond):
        """いまの章の始まりより後に、cond の kind (count 回目、where の値が合うもの) が出た時刻。"""
        hits = [t for t, p in self.seen.get(cond["kind"], []) if t >= self.ch_start
                and all(p.get(k) == v for k, v in (cond.get("where") or {}).items())]
        n = int(cond.get("count", 1))
        return hits[n - 1] if len(hits) >= n else None

    def _chapters(self):
        n = self.ch + 1
        if n >= len(self.chapters):
            return
        cfg = self.chapters[n]
        cond = cfg.get("start_after") or {"at": 0}
        ready = None
        if "at" in cond:
            ready = float(cond["at"])
        elif cond.get("kind"):
            t = self._kind_time(cond)
            ready = None if t is None else t + float(cond.get("delay", 0.0))
        elif cond.get("done"):
            ready = None if self.ch_done_t is None else self.ch_done_t + float(cond.get("delay", 0.0))
        if ready is None and cfg.get("max_wait") and self.ch >= 0:
            if self.w.time - self.ch_start >= float(cfg["max_wait"]):
                print(f"[story] t={self.w.time:.1f} 章 {n} の条件がそろわないまま {cfg['max_wait']} 秒たった → 始める",
                      flush=True)
                ready = self.w.time
        if ready is not None and self.w.time >= ready:
            self._start(n, cfg)

    def _start(self, n, cfg):
        now = self.w.time
        self.ch, self.ch_start, self.ch_done_t = n, now, None
        new = []
        for e in cfg.get("events") or []:
            e = {**e, "t": now + float(e.get("t", 0.0))}
            (self.later if "objects" in e or "stand_up" in e else new).append(e)
        self.events = self.events[:self.k] + sorted(self.events[self.k:] + new, key=lambda e: e["t"])
        self.ch_end_k = len(self.events)
        if "replies" in cfg:
            self.voices.replies = cfg["replies"] or {}
            self.voices.asked = {}
        if "react" in cfg:
            self.react = cfg["react"] or {}
            self.warned, self.stopped, self.stop_at = {}, set(), {}
        title = cfg.get("title")
        self.m.publish("sim/scenario", {"time": round(now, 1), "chapter": n, "title": title})
        print(f"[story] t={now:.1f} 章 {n} を始める: {title or '(題なし)'}", flush=True)
        if title and self.on_chapter:
            self.on_chapter(title)

    def _stand_up(self, who):
        """{t, who, stand_up: true}: まだ横になっていれば、その場で起き上がる (アヒルが見つけられず問いかけが来なかったとき、
        横になったままだと次の章で歩けないので)。"""
        i = self.index[who]
        st = self.w.person_state.get(i)
        if st and st["pose"] == "lie":
            self.w.place_person(i, st["x"], st["y"], st["yaw"], "stand", st["level"])
            self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": who, "at": "here", "pose": "stand"})
            print(f"[story] t={self.w.time:.1f} {who} が自分で起き上がった", flush=True)

    # ---- 物を床に置く (章の途中で)
    def _objects(self, objs):
        for o in objs:
            k = self.obj_used.get(o["kind"], 0)
            self.obj_used[o["kind"]] = k + 1
            name = f"obj_{o['kind']}_{k}"
            x, y = (v * self.w.module for v in o["at"])
            self.w.objects.place(name, x, y, math.radians(o.get("yaw", 0.0)))
            self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "object": o["kind"], "x": round(x, 2),
                                            "y": round(y, 2), "room": self.w.room_of(x, y)})
            print(f"[story] t={self.w.time:.1f} {o['kind']} を床に置く ({x:.2f}, {y:.2f})", flush=True)

    def _place_objects(self):
        self._objects(self.sc.get("objects") or [])
