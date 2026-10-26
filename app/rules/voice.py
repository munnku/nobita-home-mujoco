"""声 (検知 → 対応支援 → 通知): 見守りカメラのマイク (常時聞いている) の声を LLM (home/brain.py の hear) で判断する。

- 子どもの声を、同じ部屋で続いた声は MERGE 秒まとめて 1 つにし、LLM に「行くべき声か」を聞く (別スレッド)。
  LLM には、いまの様子 (situation: 子どもの部屋と姿勢・最近の出来事) を添える。決まった言葉だけで決めないので、
  「いたっ…なんちゃって」「テレビのまねで、たすけてー」のような声は行かなくてよいと判断できる
- 行くとき (concern が go / urgent):
  - その部屋のスピーカーで、すぐ一言返す (say_now: 「だいじょうぶ？いま いくね。」)
  - けが・痛い声は転倒の手がかりにもする (c.distress)。横になった人が見えれば転倒の決まり (fall.py) が確かめる
  - そうでなければ、アヒルが子どもの所へ行き、姿を見て、対話する (talk.py → LLM: 何があったか聞き、
    けがなら手当て、こわいなら落ち着かせ、質問なら答える)。対話と様子から重さを決めて知らせる
  - 煙・火の声は災害の決まり (emergency.py) に回す
- アヒルが子どもと話している間 (転倒・死角・声・災害の仕事の最中) の声は、その対話の返事なので、ここでは扱わない
"""
import queue
import threading

from home import brain
from home.vda5050 import action

from .base import LABEL, Job, Rule, result_json
from .talk import Dialogue, judge_then

MERGE = 1.5            # 声が途切れてこれだけたったら、まとめた声を判断する [s]
FALL_WAIT = 3.0        # 痛がる声のあと、転倒の決まりが確かめに行くかをこれだけ待つ [s]
TALKING = ("fall", "blind", "voice", "disaster")
SPEAKERS = ("kodomo", "dk", "ima", "rouka")
WHAT = {"injury": "けがをし", "illness": "具合が悪くなり", "fear": "こわがってい", "help": "助けを求め",
        "talk": "アヒルに話しかけ", "disaster": "災害をこわがってい"}


class Voice(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.inbox = queue.Queue()     # 受信スレッドではためるだけ。step で見る
        self.buf = {}                  # 部屋 → {"text": [...], "who", "t0", "last"}
        self.done = queue.Queue()      # LLM の判断 (room, text, res)
        self.waiting = []              # 転倒の決まりを待っている声 [(時刻, room, text, res)]
        c.m.subscribe("homie/+/audio/heard",
                      lambda t, p: self.inbox.put((t, p)) if isinstance(p, dict) and p.get("text") else None)

    def step(self):
        c = self.c
        while not self.inbox.empty():
            t, v = self.inbox.get()
            if v.get("who") not in (None, "child") or any(c.busy(n) for n in TALKING):
                continue
            room = v.get("room") or c.cam_room.get(t.split("/")[1], "?")
            b = self.buf.setdefault(room, {"text": [], "t0": c.time})
            if not b["text"] or b["text"][-1] != v["text"]:
                b["text"].append(v["text"])
            b["last"] = c.time
        for room, b in list(self.buf.items()):
            if c.time - b["last"] >= MERGE:
                del self.buf[room]
                self._judge(room, " ".join(b["text"]))
        while not self.done.empty():
            self._handle(*self.done.get())
        for item in list(self.waiting):
            t, room, text, res = item
            if c.busy("fall"):
                self.waiting.remove(item)          # 転倒の決まりが確かめに行った
            elif c.time - t >= FALL_WAIT:
                self.waiting.remove(item)
                self._go(room, text, res)

    def _judge(self, room, text):
        c = self.c
        if brain.llm_mode() not in ("claude-code", "api"):
            self._handle(room, text, brain.hear(text, room))
            return
        situation = c.situation()
        threading.Thread(target=lambda: self.done.put((room, text, brain.hear(text, LABEL.get(room, room), situation))),
                         daemon=True).start()

    def _handle(self, room, text, res):
        c = self.c
        label = LABEL.get(room, room)
        c.log(f"{label}のマイク: 「{text}」→ {res['concern']} ({res['category']}、{res.get('by')}) {res.get('why') or ''}",
              kind="voice_heard", room=room, concern=res["concern"], category=res["category"], by=res.get("by"),
              heard=text)
        if res["concern"] not in ("go", "urgent") or any(c.busy(n) for n in TALKING):
            return
        if res.get("say_now") and room in SPEAKERS:
            c.speaker(room, res["say_now"])
        if res["category"] == "fire":
            c.emergency.start("fire", f"{label}で子どもが「{text}」と言った", room)
            return
        if res["category"] in ("injury", "help", "illness"):
            c.distress[room] = c.time          # 横になった人が見えれば、転倒の決まりがすぐ確かめる
            self.waiting.append((c.time, room, text, res))
            return
        self._go(room, text, res)

    def _go(self, room, text, res):
        """アヒルが子どもの所へ行き、姿を見て、対話する。"""
        c = self.c
        if any(c.busy(n) for n in TALKING):
            return
        c.distress.pop(room, None)
        cs = c.child_seen
        if cs and cs["room"] == room and c.time - cs["time"] < 10:
            x, y, kw = cs["x"], cs["y"], {"targetX": cs["x"], "targetY": cs["y"]}
        else:
            spot = c.room_spot(room)
            if spot is None:
                c.log(f"{LABEL.get(room, room)}にはアヒルが行けない → スピーカーで話す", kind="voice_far", room=room)
                return
            (x, y), kw = spot, {"sweep": "full"}
        cat = res["category"]
        c.log(f"アヒルが{LABEL.get(room, room)}の子どものところへ行く ({cat})", kind="voice_go", room=room, category=cat)
        job = Job("voice", 1 if res["concern"] == "urgent" or cat in ("injury", "help", "illness") else 2)
        job.interruptible = cat in ("talk", "fear", "disaster")
        job.go(x, y, 1.2, [action("detectObject", "x-detect", objectType="person", duration=10.0, **kw)])
        ev = {"who": "子ども", "room": room, "room_label": LABEL.get(room, room), "category": cat, "heard": text,
              "what": WHAT.get(cat, "声をかけ")}
        job.ev = ev
        job.t_heard = c.time

        def arrived(j):
            r = result_json(j.last[0] if j.last else None)
            if r.get("found") and r.get("posture") == "lying":
                # 横になっていた (カメラの死角で転んでいた): 転倒として確かめる
                c.log("声の所へ行くと横になっていた → 転倒として確かめる", kind="fall_suspected", room=room, x=x, y=y)
                c.fall.confirmed(j, {"x": x, "y": y, "room": room, "first": job.t_heard, "last": c.time})
                return
            ev.update(found=bool(r.get("found")), posture=r.get("posture"), image=r.get("image"))
            Dialogue(c, "voice", ev, self._talked).start(j)
        job.then(arrived)
        c.add_job(job)

    def _talked(self, j, dialogue, res):
        judge_then(self.c, j, "voice", {k: v for k, v in j.ev.items() if k != "image"}, self._report)

    def _report(self, j, v):
        c = self.c
        c.log(f"判断: {v['level']} ({v['reason']}) [{v['by']}]", kind="voice_judged", level=v["level"], by=v["by"],
              category=j.ev["category"])
        c.attended[j.ev["room"]] = c.time
        c.notify("voice_report", v["level"], v["message_to_parent"], room=j.ev["room"], category="通知",
                 reason=v["reason"], by=v["by"], heard=j.ev["heard"], replies=j.ev["replies"],
                 dialog=j.ev["dialog"], evidence=j.ev.get("image"))
        j.say({"info": "おうちの人にも知らせておくね。",
               "warn": "おうちの人に知らせたよ。",
               "emergency": "すぐにおうちの人に連絡したよ。そのまま待っていてね。"}[v["level"]])
