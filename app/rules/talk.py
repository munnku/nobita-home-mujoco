"""対話の共通部分: 話す → 聞く → 次の一言を考える (home/brain.py の talk) … を、仕事 (Job) の手順に 1 回分ずつ足していく。

使い方: job.then(Dialogue(c, "fall", ev, on_done).start)
  kind: fall / voice / blind / fire / earthquake / blackout / visitor (brain.KIND_GOAL)
  ev: 対話に添える事実 ({"room", "room_label", "who", ...})。やりとりは ev["dialog"]、返事の意味は ev["replies"] にたまる
  on_done(j, dialogue, res): 終わったときに呼ぶ (res は最後の talk の結果。res["reply_label"] が最後の返事の意味)
LLM を使うとき (BRAIN_LLM) は考える間 (数秒) 別スレッドで待つので、司令塔は止まらない。
最初の一言は規則の決まった問い (速い) で始め、声を聞いて来たとき (voice) だけ LLM が聞いた言葉に合わせて始める。
"""
import threading

from home import brain

from .base import Job, result_json

THINK_TIMEOUT = 90.0     # LLM の答えを待つ上限 [s] (過ぎたら規則の一言にする)


class Dialogue:
    def __init__(self, c, kind, ev, on_done):
        self.c, self.kind, self.ev, self.on_done = c, kind, ev, on_done
        ev.setdefault("dialog", [])
        ev.setdefault("replies", [])
        self.res = None

    def start(self, j):
        if self.kind == "voice":
            self._think(j)
        else:
            self.res = brain.talk_rules(self.kind, self.ev)
            self._apply(j)

    # ---- 考える (別スレッド) → 決まった一言を手順にする
    def _think(self, j):
        self.res = None
        if brain.llm_mode() not in ("claude-code", "api"):
            self.res = brain.talk(self.kind, self.ev)
            j.insert(*Job("tmp", 0).then(self._apply).steps)
            return
        situation = self.c.situation()

        def run():
            self.res = brain.talk(self.kind, self.ev, situation)
        threading.Thread(target=run, daemon=True).start()
        j.insert(*Job("tmp", 0).until(lambda j: self.res is not None, THINK_TIMEOUT).then(self._apply).steps)

    def _apply(self, j):
        c, ev = self.c, self.ev
        res = self.res or {**brain.talk_rules(self.kind, ev), "by": "rules (時間切れ)"}
        self.res = res
        d = ev["dialog"]
        if d and "child" in d[-1] and "label" not in d[-1]:
            d[-1]["label"] = res.get("reply_label")
            ev["replies"].append({"asked": d[-1]["robot"], "text": d[-1]["child"], "label": d[-1]["label"]})
            text = d[-1]["child"]
            c.log((f"返事: 「{text}」→ {d[-1]['label']}" if text else "返事がない") + f" ({res.get('by')})",
                  kind="reply", label=d[-1]["label"], by=res.get("by"))
        nxt = Job("tmp", 0)
        if res.get("done"):
            if res.get("say"):
                d.append({"robot": res["say"], "intent": res.get("intent", "close")})
                nxt.say(res["say"], res.get("intent") if res.get("intent") in ("instruct", "comfort") else None)
            nxt.then(lambda j: self.on_done(j, self, res))
        else:
            d.append({"robot": res["say"], "intent": res.get("intent")})
            nxt.say(res["say"], res.get("intent") or "ask_more").listen(float(res.get("listen") or 8.0))
            nxt.then(self._heard)
        j.insert(*nxt.steps)

    def _heard(self, j):
        heard = result_json(j.last[0] if j.last else None).get("heard") or []
        text = " ".join(h["text"] for h in heard if h.get("who"))
        self.ev["dialog"][-1]["child"] = text or None
        self._think(j)

    def last_label(self):
        labels = [t.get("label") for t in self.ev["dialog"] if "child" in t]
        return labels[-1] if labels else None


def judge_then(c, j, kind, ev, report):
    """知らせの重さを決め (LLM なら別スレッド)、report(j, verdict) を呼ぶ手順を足す。"""
    j.verdict = None
    public = {k: v for k, v in ev.items() if k != "first_seen"}
    if brain.llm_mode() not in ("claude-code", "api"):
        j.verdict = brain.judge(kind, public)
        j.insert(*Job("tmp", 0).then(lambda j: report(j, j.verdict)).steps)
        return
    situation = c.situation()

    def run():
        j.verdict = brain.judge(kind, public, situation)
    threading.Thread(target=run, daemon=True).start()

    def done(j):
        v = j.verdict or {**brain.judge(kind, public, llm=""), "by": "rules (時間切れ)"}
        report(j, v)
    j.insert(*Job("tmp", 0).until(lambda j: j.verdict is not None, 150.0).then(done).steps)
