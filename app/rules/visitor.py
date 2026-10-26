"""インターホン (対応支援 → 通知): 玄関のインターホンが鳴ったら、家の人の代わりに LLM が応対する (talk.py の kind visitor)。

- 子どもには出させない。子どもがひとりでいることや留守番中であることは言わない (home/brain.py の決まりと、言葉の見張り)
- 用件を聞き、宅配なら置き配を頼み、それ以外はていねいに断る。相手が開けさせようとする・家族の在宅を聞き出そうとするなど
  怪しければ (visitor = suspicious)、保護者に warn を送り、子どものいる部屋のスピーカーで「出なくていいよ」と伝える
- アヒルは使わない (受け持ちは intercom)。ほかのタスク (見回り・転倒の確認) と同時に進む
"""
import queue

from .base import Job, Rule
from .talk import Dialogue

SPEAKERS = ("kodomo", "dk", "ima", "rouka")


class Visitor(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.rings = queue.Queue()
        c.m.subscribe("homie/intercom-genkan/intercom/ring",
                      lambda t, p: self.rings.put(p) if str(p).lower() == "true" else None)

    def step(self):
        c = self.c
        while not self.rings.empty():
            self.rings.get()
            if c.busy("visitor"):
                continue
            c.log("インターホンが鳴った → 家の人の代わりに応対する", kind="intercom_ring")
            job = Job("visitor", 1)
            job.lane = "intercom"
            job.then(Dialogue(c, "visitor", {"mode": c.mode}, self._done).start)
            c.add_job(job)

    def _done(self, j, dialogue, res):
        c = self.c
        kind = res.get("visitor") or "unknown"
        said = " / ".join(t.get("child") or "(無言)" for t in dialogue.ev["dialog"] if "child" in t)
        c.log(f"インターホンの応対が終わった: {kind} ({res.get('by')}) 相手「{said}」", kind="visitor_done", visitor=kind,
              by=res.get("by"))
        if kind == "suspicious":
            c.notify("visitor", "warn", f"インターホンに知らない人が来て「{said}」と言いました。家の者は出られないと断りました。",
                     category="通知", visitor=kind, dialog=dialogue.ev["dialog"])
            if c.mode == "alone":
                room = c.child_room()
                c.speaker(room if room in SPEAKERS else "ima",
                          "インターホンが なったけど、でなくて だいじょうぶだよ。ドアも あけないでね。", intent="warn")
        elif kind == "delivery":
            c.notify("visitor", "info", "宅配便が来ました。玄関の前に置いてもらうよう頼みました。", category="通知", visitor=kind)
        else:
            c.notify("visitor", "info", f"インターホンに「{said}」と来客がありました。家の者は出られないと伝えました。",
                     category="通知", visitor=kind)
