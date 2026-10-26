"""カメラを置けない所 (脱衣所・浴室・トイレ) の見守り。おるすばんモード (home/mode = alone) のときだけ働く。

- 人感センサーで、子どもがカメラの無い部屋にいると分かる。そこに max_stay 秒より長くいたら、アヒルが扉の前へ行き、
  声をかけて返事を聞く (返事は台本。home/voice.py)。返事の意味は対話 (talk.py → LLM) で分ける
- 返事があれば (平気・あいまいでも声が返れば) 「ゆっくりでいいよ」と言って戻る。また max_stay 秒たてば聞きに行く
- 返事がなければもう一度呼ぶ。それでもなければ、扉が開いていれば中に入って姿勢だけを確かめる (privacy=true:
  絵は残さない)。横になっている・見つからない・扉が閉まっていて入れないときは、保護者に emergency
- 痛い・助けての返事なら warn / emergency (home/brain.py の規則の判断を使う)
設定は home.yaml の blind_rooms (部屋 → 扉・アヒルが立つ所・中の見る所・max_stay)。
"""
from home.brain import judge_rules
from home.vda5050 import action

from .base import LABEL, M, Job, Rule, result_json
from .talk import Dialogue

ATTENDED = 300.0       # ほかの決まり (転倒) で確かめて知らせた部屋は、この秒数は聞きに行かない


class BlindSpot(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.cfg = c.home.get("blind_rooms", {})
        self.since = {}          # 部屋 → 人感センサーが続けて反応している始まりの時刻
        self.asked_at = {}       # 部屋 → 最後に聞きに行った時刻

    def step(self):
        c = self.c
        if c.mode != "alone":
            return
        for room, cfg in self.cfg.items():
            if c.motion.get(room):
                self.since.setdefault(room, c.time)
            else:
                self.since.pop(room, None)
                self.asked_at.pop(room, None)
                continue
            start = max(self.since[room], self.asked_at.get(room, -1e9))
            if c.time - c.attended.get(room, -1e9) < ATTENDED:
                continue
            if c.time - start >= cfg["max_stay"] and not c.busy("blind") and not c.busy("fall"):
                self.asked_at[room] = c.time
                self._go(room, cfg)

    def _go(self, room, cfg):
        c = self.c
        label = LABEL.get(room, room)
        c.log(f"{label}に {c.time - self.since[room]:.0f} 秒いる (人感センサー、カメラなし) → アヒルが扉の前で声をかける",
              kind="blind_check", room=room)
        c.add_job(self._job(room, cfg))

    def _job(self, room, cfg):
        x, y = (v * M for v in cfg["stand"])
        job = Job("blind", 2).go(x, y, 0.3)
        job.restart = lambda: self._job(room, cfg)       # 火事・地震で止められたら、初めから声をかけ直す
        job.room = room
        job.then(Dialogue(self.c, "blind", {"who": "子ども", "room": room, "room_label": LABEL.get(room, room)},
                          self._heard).start)
        return job

    def _heard(self, j, dialogue, res):
        """声をかけ終わった: 最後の返事の意味で、待つ・知らせる・中を確かめるを決める。"""
        c = self.c
        lab = dialogue.last_label()
        text = " ".join(t.get("child") or "" for t in dialogue.ev["dialog"]).strip()
        room, label = j.room, LABEL.get(j.room, j.room)
        nxt = Job("tmp", 0)
        if lab in ("ok", "unclear"):
            c.log(f"{label}から返事があった → 待つ", kind="blind_ok", room=room)
            nxt.say("わかった。ゆっくりでいいよ。")
            d = c.home["robot"]["dock"]
            nxt.go(d["x"] * M, d["y"] * M, 0.3)     # 扉の前から離れて待つ (出てきた子どもとぶつからない)
        elif lab in ("hurt", "help"):
            v = judge_rules({"who": "子ども", "room": room, "room_label": label, "replies": dialogue.ev["replies"],
                             "posture": None, "found": False})
            self._alert(room, v["level"], f"子どもが{label}で「{text}」と言っています。{v['message_to_parent']}")
            nxt.say("おうちの人に知らせたよ。")
        else:
            door = self.cfg[room].get("door")
            if door and not c.door_open.get(door, True):      # 扉の開閉センサー (分からなければ開いているとみなす)
                c.log(f"{label}から返事がなく、扉が閉まっていて入れない → 保護者に連絡", kind="blind_no_reply", room=room)
                self._alert(room, "emergency", f"子どもが{label}にいますが、呼んでも返事がありません。扉が閉まっていて"
                                               "ロボットは入れません。すぐに連絡してください。")
                nxt.say("おうちの人に連絡したよ。")
            else:
                c.log(f"{label}から返事がない → 中に入って姿勢だけを確かめる (絵は残さない)", kind="blind_enter", room=room)
                ix, iy = (v * M for v in self.cfg[room]["inside"])
                nxt.go(ix, iy, 0.3, [action("detectObject", "x-detect", objectType="person", sweep="full",
                                            duration=10.0, privacy=True)])
                nxt.then(self._inside)
        j.insert(*nxt.steps)

    def _inside(self, j):
        c = self.c
        room, label = j.room, LABEL.get(j.room, j.room)
        res = result_json(j.last[0] if j.last else None)
        if res.get("found") and res.get("posture") == "standing":
            c.log(f"{label}の中: 立っていた", kind="blind_standing", room=room)
            self._alert(room, "warn", f"子どもが{label}で呼んでも返事をしませんが、立っています。様子を確かめてください。")
            j.say("返事がなかったから心配したよ。")
        elif res.get("found"):
            c.log(f"{label}の中: 横になっている (score {res.get('score')}) → 保護者に連絡", kind="blind_lying", room=room)
            self._alert(room, "emergency", f"子どもが{label}で倒れていて、呼んでも返事がありません。すぐに連絡してください。")
            j.say("すぐにおうちの人に連絡したよ。")
        else:
            c.log(f"{label}の中に人が見えない → 保護者に連絡", kind="blind_not_found", room=room)
            self._alert(room, "emergency", f"子どもが{label}にいるはずですが、返事がなく、ロボットにも見えません。")
            j.say("どこにいるの？")

    def _alert(self, room, level, summary):
        self.c.attended[room] = self.c.time
        self.c.notify("no_reply", level, summary, room=room, category="通知", privacy=True)
