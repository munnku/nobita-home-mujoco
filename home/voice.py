"""人役の声: 場面の台本 (replies) で、アヒルやスピーカーの問いかけに返事をする。

- アヒルやスマートスピーカーが話すと home/speech に出る。問いの種類 (intent: ask_ok, ask_pain, ...) が付いていて、
  人役に聞こえる (同じ部屋か HEAR_RADIUS 以内、同じ階) なら、台本の返事を delay 秒後に home/voice に出す。
- 台本の書き方 (場面の yaml):
    replies:
      child:
        ask_ok: {say: "いてて…でもへーき", delay: 2.0}
        ask_pain: [{say: "ひざがちょっといたい", delay: 1.5, then: {pose: stand}}, null]   # 1 回目と 2 回目
        default: null                                                                   # 書いていない問いは無言
  null は無言。リストなら n 回目の問いに n 番目を使う (足りなければ最後のもの)。then: {pose} で返事のあとに姿勢を変える。
- 台本のイベントの {t, who, say: "いたっ"} は、その時刻に人役が自分から声を出す (Scenario から say() を呼ぶ)。
声は文字で表す (音声認識・音声合成の代わり)。home/voice を聞くのは、アヒルのマイク (VDA 5050 の listen) と
見守りカメラのマイク (Homie の audio ノード)。
"""
import math
import queue

HEAR_RADIUS = 4.0     # 声が届く距離 [m] (同じ部屋ならこれより遠くても届く)


class Voices:
    def __init__(self, world, mqtt, scenario_cfg, names, on_intent=None):
        self.w = world
        self.m = mqtt
        self.replies = scenario_cfg.get("replies") or {}
        self.names = names                  # 人の番号 → 名前
        self.asked = {}                     # (名前, intent) → 聞かれた回数
        self.pending = []                   # [(時刻, 名前, 返事)]
        self.inbox = queue.Queue()
        self.on_intent = on_intent          # 問いかけ・警告が聞こえたとき (人の番号, intent) を渡す (場面の反応)
        mqtt.subscribe("home/speech", lambda t, p: self.inbox.put(p) if isinstance(p, dict) else None)

    def _where(self, i):
        st = self.w.person_state.get(i)
        if not st:
            return None
        return st["x"], st["y"], st["level"], self.w.room_of(st["x"], st["y"], st["level"])

    def hears(self, i, speech):
        """人 i にその声が聞こえるか。"""
        where = self._where(i)
        if where is None:
            return False
        x, y, level, room = where
        if speech.get("room") and speech["room"] == room:
            return True
        if speech.get("x") is not None and speech.get("level", "1F") == level:
            return math.hypot(speech["x"] - x, speech["y"] - y) <= HEAR_RADIUS
        return False

    def tick(self):
        while not self.inbox.empty():
            sp = self.inbox.get()
            intent = sp.get("intent")
            if not intent:
                continue
            for i, name in self.names.items():
                if not self.hears(i, sp):
                    continue
                if self.on_intent:
                    self.on_intent(i, intent)
                table = self.replies.get(name)
                if table is None:
                    continue
                n = self.asked.get((name, intent), 0)
                self.asked[(name, intent)] = n + 1
                r = table.get(intent, table.get("default"))
                if isinstance(r, list):
                    r = r[min(n, len(r) - 1)] if r else None
                if r is None:
                    print(f"[voice] t={self.w.time:.1f} {name}: (無言) ← {intent}", flush=True)
                    continue
                # 返事は話し終わってから delay 秒後
                self.pending.append((self.w.time + float(sp.get("duration", 0.0)) + float(r.get("delay", 1.5)), i, r))
        for item in [p for p in self.pending if p[0] <= self.w.time]:
            self.pending.remove(item)
            _, i, r = item
            if r.get("say"):
                self.say(i, r["say"])
            then = r.get("then") or {}
            if then.get("pose"):
                st = self.w.person_state.get(i)
                if st:
                    self.w.place_person(i, st["x"], st["y"], st["yaw"], then["pose"], st["level"])
                    self.m.publish("sim/scenario", {"time": round(self.w.time, 1), "who": self.names[i],
                                                    "at": "here", "pose": then["pose"]})
                    print(f"[voice] t={self.w.time:.1f} {self.names[i]} -> {then['pose']}", flush=True)

    def say(self, i, text):
        where = self._where(i)
        if where is None:
            return
        x, y, level, room = where
        name = self.names[i]
        self.m.publish("home/voice", {"who": name, "text": text, "x": round(x, 2), "y": round(y, 2),
                                      "level": level, "room": room, "time": round(self.w.time, 1)})
        print(f"[voice] t={self.w.time:.1f} {name}「{text}」", flush=True)
