"""転倒 (検知 → 対応支援 → 通知): 見守りカメラが「横になった人」を FALL_CONFIRM 秒以上見たら、その部屋の照明をつけ、
アヒルが人を見通せる所へ行って確かめる (detectObject)。立っていれば誤報として取り消す。
- 声も手がかりにする: 常時マイクの声を LLM が「痛がっている」と判断した部屋 (voice.py が c.distress に入れる) で
  横になった人が見えれば、FALL_CONFIRM を待たずに確かめに行く。横になった人が見えない (カメラの死角) ときは、
  声の決まりがアヒルを向かわせ、着いて横になっていれば confirmed() からここの流れに入る
- アヒルの目でも横になっていれば、家族に「転倒」を知らせ、対話で安否を確かめ (talk.py → LLM。だいじょうぶ？ → いたいところは？
  → 手当ての仕方)、もう一度見て (起き上がったか)、返事と様子から知らせの重さ (info / warn / emergency) を決めて知らせる
"""
import math

from home.vda5050 import action

from .base import LABEL, M, Job, Rule, result_json
from .talk import Dialogue, judge_then

FALL_CONFIRM = 3.0     # 横になった人をこれだけ続けて見たら疑う [s]
VOICE_CONFIRM = 1.0    # 痛がる声が聞こえた部屋なら、これだけ見れば疑う [s]
VOICE_RECENT = 15.0    # 声をこの秒数のあいだ手がかりにする [s]
SAME_PLACE = 1.5       # 同じ人とみなす距離 [m] (2 台のカメラの推定は 1 m ほどずれることがある)
CHILD = "子ども"


class Fall(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.lying = []              # 転倒の候補 [{"x","y","room","first","last"}]
        self.handled = []
        bed = c.home.get("bed") or {}
        self.bed = (bed.get("room"), [v * M for v in bed.get("rect", [])])

    def on_persons(self, room, persons):
        c = self.c
        for p in persons:
            if p.get("posture") != "lying" or self._in_bed(room, p):
                continue        # 子ども部屋の布団で横になっているのは転倒ではない (寝ている)
            where = p.get("room") or room     # 人がいる部屋 (カメラの部屋とは限らない。ふすま越しに見えることがある)
            for cand in self.lying:
                if math.hypot(cand["x"] - p["x"], cand["y"] - p["y"]) < SAME_PLACE:
                    cand.update(x=p["x"], y=p["y"], last=c.time)
                    break
            else:
                self.lying.append({"x": p["x"], "y": p["y"], "room": where, "first": c.time, "last": c.time})

    def _in_bed(self, room, p):
        broom, r = self.bed
        return room == broom and len(r) == 4 and r[0] <= p["x"] <= r[1] and r[2] <= p["y"] <= r[3]

    def _heard_in(self, room):
        t = self.c.distress.get(room)
        return t is not None and self.c.time - t < VOICE_RECENT

    def step(self):
        c = self.c
        self.lying = [cand for cand in self.lying if c.time - cand["last"] < 3.0]
        if c.busy("fall") or c.busy("voice"):
            return
        for cand in self.lying:
            recent = any(math.hypot(h["x"] - cand["x"], h["y"] - cand["y"]) < SAME_PLACE and c.time - h["time"] < 600
                         for h in self.handled)
            need = VOICE_CONFIRM if self._heard_in(cand["room"]) else FALL_CONFIRM
            if cand["last"] - cand["first"] >= need and not recent:
                self.handled.append({"x": cand["x"], "y": cand["y"], "time": c.time})
                why = "横になった人と痛がる声" if need == VOICE_CONFIRM else "横になった人"
                c.log(f"{LABEL.get(cand['room'], cand['room'])}のカメラ: {why} ({cand['x']:.1f}, {cand['y']:.1f})"
                      " → 転倒の疑い", kind="fall_suspected", room=cand["room"], x=cand["x"], y=cand["y"])
                c.distress.pop(cand["room"], None)
                self._dispatch(cand)
                return

    def _dispatch(self, cand):
        self.c.light(cand["room"])
        self.c.add_job(self._job(cand))

    def _job(self, cand):
        c = self.c
        # カメラを置かない部屋 (脱衣所など) では、アヒルの目の絵も残さない
        privacy = cand["room"] in c.home.get("blind_rooms", {})
        if cand.get("sweep"):
            det = action("detectObject", "x-detect", objectType="person", sweep="full", duration=15.0, privacy=privacy)
        else:
            det = action("detectObject", "x-detect", objectType="person", targetX=cand["x"], targetY=cand["y"],
                         duration=12.0, privacy=privacy)
        job = Job("fall", 1).go(cand["x"], cand["y"], 2.0, [det])
        job.restart = lambda: self._job(cand)      # 火事・地震で止められたら、初めから確かめ直す
        job.then(lambda j, cand=cand: self._result(j, cand))
        return job

    def confirmed(self, j, cand):
        """ほかの仕事 (声を聞いて行った) で、アヒルの目が横になった人を見た: 転倒として確かめ・対話する。"""
        self.handled.append({"x": cand["x"], "y": cand["y"], "time": self.c.time})
        self._result(j, cand)

    def _result(self, j, cand):
        c = self.c
        res = result_json(j.last[0] if j.last else None)
        room_label = LABEL.get(cand["room"], cand["room"])
        if res.get("found") and res.get("posture") == "lying":
            c.log(f"アヒルの目でも横になった人を確認 (score {res.get('score')}) → 家族に知らせ、声をかける",
                  kind="fall_confirmed")
            c.attended[cand["room"]] = c.time
            c.notify("fall", "checking", f"{room_label}で転んだようです。いまロボットが様子を確かめています。",
                     room=cand["room"], category="検知", x=cand["x"], y=cand["y"], confirmedBy="microduck-01",
                     elapsed=round(c.time - cand["first"], 1), evidence=res.get("image"))
            ev = {"who": CHILD, "room": cand["room"], "room_label": room_label, "x": cand["x"], "y": cand["y"],
                  "first_seen": cand["first"], "image": res.get("image")}
            j.ev = ev
            Dialogue(c, "fall", ev, self._talked).start(j)
        elif res.get("found"):
            c.log(f"アヒルが見ると立っていた ({res.get('posture')}) → 誤報として取り消し", kind="false_alarm")
            j.say("お元気そうでよかったです。")
        else:
            c.log("アヒルは人を見つけられなかった → 家族に確認を頼む", kind="fall_unconfirmed")
            c.notify("fall_unconfirmed", "warn", f"{room_label}で転んだようですが、ロボットが見つけられません。確かめてください。",
                     room=cand["room"], category="検知", x=cand["x"], y=cand["y"])
            j.say("どこにいますか？ご家族に知らせます。")

    def _talked(self, j, dialogue, res):
        """対話が済んだ: もう一度見て (起き上がったか)、重さを決める。"""
        j.insert(*Job("tmp", 0).look(j.ev.get("x"), j.ev.get("y"), 6.0).then(self._judge).steps)

    def _judge(self, j):
        res = result_json(j.last[0] if j.last else None)
        j.ev.update(found=bool(res.get("found")), posture=res.get("posture"),
                    lying_seconds=round(self.c.time - j.ev["first_seen"], 1))
        judge_then(self.c, j, "fall", {k: v for k, v in j.ev.items() if k != "image"}, self._report)

    def _report(self, j, v):
        c = self.c
        r = v["rules"]
        llm = f" / LLM ({v['by']}): {v['llm']['level']}" if v.get("llm") else ""
        c.log(f"判断: {v['level']} ({v['reason']}) [規則: {r['level']}{llm}]", kind="fall_judged", level=v["level"],
              rules_level=r["level"], llm_level=(v.get("llm") or {}).get("level"), by=v["by"])
        c.notify("fall_report", v["level"], v["message_to_parent"], room=j.ev["room"], category="通知",
                 reason=v["reason"], by=v["by"], replies=j.ev["replies"], posture=j.ev.get("posture"),
                 evidence=j.ev.get("image"))
        say = {"info": "よかった。おうちの人にも知らせておくね。",
               "warn": "おうちの人に知らせたよ。じっとしていてね。",
               "emergency": "すぐにおうちの人に連絡したよ。そのまま動かないでね。"}[v["level"]]
        j.say(say)
