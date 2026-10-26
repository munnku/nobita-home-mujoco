"""災害 (検知 → 対応支援 → 通知): 煙センサー・緊急地震速報・停電の知らせを受けて、子どもに逃げ方・身の守り方を伝える。

- 火事 (煙センサー smoke-*、または子どもの「けむりが出てる」の声): 全部のスピーカーで逃げ方を伝え、保護者に emergency。
  アヒルが子どもの所へ行き、対話する (talk.py の kind fire → LLM: けががないか、どう逃げるか)
- 地震 (home/emergency の earthquake、緊急地震速報): すぐに全部のスピーカーで「机の下で頭を守って」。アヒルはゆれている間
  (shake 秒) は動かずに待ち (歩くと転ぶ・ぶつかる)、おさまったら子どもを確かめて対話する
- 停電 (home/emergency の blackout): 照明は消えたまま (家の機器が電池で動く見守りカメラ・スピーカー・アヒルだけになる)。
  アヒルが子どもの所へ行き「そこで待っていてね」と落ち着かせる
- 災害のタスクは重さ 0 で、いまのアヒルの仕事を止めて先にする。対話のあと、返事と様子から重さを決めて知らせる
  (起きたこと自体の重さが下限: 火事は emergency、地震は warn、停電は info)
"""
import queue

from home import brain
from home.vda5050 import action

from .base import LABEL, Job, Rule, result_json
from .talk import Dialogue, judge_then

SPEAKERS = ("kodomo", "dk", "ima", "rouka")
SAY_ALL = {"fire": "火事です！けむりを すわないように、ひくい しせいで、玄関から外に出て！",
           "earthquake": "地震です！つくえの下にもぐって、頭をまもって！",
           "blackout": "停電です。あぶないから、その場で うごかずに まっていてね。"}
FIRST = {"fire": ("emergency", "火事の疑いがあります ({why})。子どもに逃げ方を伝えました。すぐに確かめ、必要なら 119 に連絡してください。"),
         "earthquake": ("warn", "緊急地震速報がありました。子どもに机の下で頭を守るよう伝えました。ゆれのあとロボットが様子を確かめます。"),
         "blackout": ("info", "家が停電しました。ロボットが子どものそばへ行きます。")}
WHAT = {"fire": "火事の疑いがあり", "earthquake": "地震にあい", "blackout": "停電にあい"}


class Emergency(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.inbox = queue.Queue()
        self.active = {}          # 種類 → 始まった時刻
        c.m.subscribe("homie/+/smoke/detected",
                      lambda t, p: self.inbox.put(("fire", f"{t.split('/')[1]} が煙を検知", t.split("/")[1][6:], {}))
                      if str(p).lower() == "true" else None)
        c.m.subscribe("home/emergency", lambda t, p: self.inbox.put((p.get("type"), p.get("source", "知らせ"), None, p))
                      if isinstance(p, dict) else None)

    def step(self):
        while not self.inbox.empty():
            kind, why, room, p = self.inbox.get()
            if kind == "fire" and self.c.time < 1.0:
                continue            # 前の場面の煙センサーの値 (retain) が届いただけ
            if kind in SAY_ALL:
                self.start(kind, why, room, **p)

    def start(self, kind, why, room=None, **p):
        c = self.c
        if kind in self.active:
            return
        self.active[kind] = c.time
        c.log(f"{dict(fire='火事', earthquake='地震', blackout='停電')[kind]}を知った ({why}) → スピーカーで知らせ、アヒルが子どもの所へ",
              kind=f"{kind}_detected", room=room)
        for spk in SPEAKERS:
            c.speaker(spk, SAY_ALL[kind], intent="evacuate" if kind == "fire" else "warn")
        level, text = FIRST[kind]
        c.notify(kind, level, text.format(why=why), room=room, category="検知")
        job = Job("disaster", 0)
        if kind == "earthquake":
            shake = float(p.get("shake", 15.0))
            t0 = c.time
            job.until(lambda j: c.time - t0 >= shake, shake + 5.0)       # ゆれている間はその場で待つ
            job.then(lambda j: c.log("ゆれがおさまった → 子どもを確かめに行く", kind="quake_over"))
        job.then(lambda j: self._go(j, kind))
        c.add_job(job)

    def _go(self, j, kind):
        """子どもの居場所 (見守りカメラで最後に見た所、なければ人感センサーの部屋) へ行き、姿を見る。"""
        c = self.c
        cs = c.child_seen
        if cs and c.time - cs["time"] < 60 and c.room_spot(cs["room"]):
            x, y, kw, room = cs["x"], cs["y"], {"targetX": cs["x"], "targetY": cs["y"]}, cs["room"]
        else:
            room = c.child_room() or "ima"
            spot = c.room_spot(room) or c.room_spot("ima")
            (x, y), kw = spot, {"sweep": "full"}
        ev = {"who": "子ども", "room": room, "room_label": LABEL.get(room, room), "event": kind, "what": WHAT[kind]}
        j.ev = ev

        def arrived(j):
            r = result_json(j.last[0] if j.last else None)
            ev.update(found=bool(r.get("found")), posture=r.get("posture"), image=r.get("image"))
            Dialogue(c, kind, ev, self._talked).start(j)
        j.insert(*Job("tmp", 0).go(x, y, 1.2, [action("detectObject", "x-detect", objectType="person", duration=10.0,
                                                       **kw)]).then(arrived).steps)

    def _talked(self, j, dialogue, res):
        judge_then(self.c, j, j.ev["event"], {k: v for k, v in j.ev.items() if k != "image"}, self._report)

    def _report(self, j, v):
        c = self.c
        kind = j.ev["event"]
        c.log(f"判断: {v['level']} ({v['reason']}) [{v['by']}]", kind="disaster_judged", level=v["level"], event=kind,
              by=v["by"])
        c.notify("disaster_report", v["level"], v["message_to_parent"], room=j.ev["room"], category="通知", event=kind,
                 reason=v["reason"], by=v["by"], replies=j.ev["replies"], dialog=j.ev["dialog"],
                 evidence=j.ev.get("image"))
        j.say({"fire": "おうちの人に連絡したよ。外に出たら、おとなの人に「火事です」と言ってね。",
               "earthquake": "おうちの人に知らせたよ。また ゆれるかもしれないから、つくえの近くにいてね。",
               "blackout": "おうちの人に知らせたよ。電気がつくまで、いっしょに まっていようね。"}[kind])
        if brain.EVENT_FLOOR.get(kind) == "info":
            self.active.pop(kind, None)        # 停電は終わったらまた知らせられるように
