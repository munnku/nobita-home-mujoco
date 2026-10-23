"""転倒: 見守りカメラが「横になった人」を FALL_CONFIRM 秒以上見たら、その部屋の照明をつけ、
アヒルが人を見通せる所へ行って確かめる (detectObject)。立っていれば誤報として取り消す。
- 声も手がかりにする: 見守りカメラのマイクが「いたっ」「うわーん」などを聞いた部屋で横になった人が見えれば、
  FALL_CONFIRM を待たずに確かめに行く。横になった人が見えない (カメラの死角) ときも、その部屋へ見に行く
- アヒルの目でも横になっていれば、家族に「転倒」を知らせ、声をかけて返事を聞き (だいじょうぶ？ → いたいところは？)、
  もう一度見て (起き上がったか)、返事と様子から知らせの重さ (info / warn / emergency) を決めて知らせる (home/judge.py)
"""
import math
import threading

from home.judge import classify_reply, judge, start_loading
from home.vda5050 import action

from .base import LABEL, Job, Rule, result_json

FALL_CONFIRM = 3.0     # 横になった人をこれだけ続けて見たら疑う [s]
GAME_CONFIRM = 240.0   # かくれんぼの最中は、隠れて伏せていることがあるので長く待つ (痛がる声がすればすぐ)
VOICE_CONFIRM = 1.0    # 痛がる声が聞こえた部屋なら、これだけ見れば疑う [s]
VOICE_RECENT = 15.0    # 声をこの秒数のあいだ手がかりにする [s]
SAME_PLACE = 1.5       # 同じ人とみなす距離 [m] (2 台のカメラの推定は 1 m ほどずれることがある)
LISTEN = 8.0           # 返事を待つ秒数
ASK = {"ask_ok": "だいじょうぶ？けがはない？", "ask_ok_again": "聞こえる？だいじょうぶ？",
       "ask_pain": "いたいところはある？"}
CHILD = "のび太"


class Fall(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.lying = []              # 転倒の候補 [{"x","y","room","first","last"}]
        self.handled = []
        self.distress = {}           # 部屋 → 痛がる声を聞いた時刻
        self.voices = []             # 見守りカメラのマイクが聞いた声 (受信スレッドではためるだけ。step で見る)
        c.m.subscribe("homie/+/audio/heard",
                      lambda t, p: self.voices.append((t, p)) if isinstance(p, dict) and p.get("text") else None)
        # 返事を分けるモデルは読み込みに数秒かかるので、先に裏で読んでおく
        start_loading()

    def _on_heard(self, t, voice):
        lab, _, _ = classify_reply(voice["text"], wait=False)
        if lab in ("hurt", "help"):
            room = voice.get("room") or self.c.cam_room.get(t.split("/")[1], "?")
            self.distress[room] = self.c.time
            self.c.log(f"{LABEL.get(room, room)}のマイク: 「{voice['text']}」(痛がる声)", kind="distress_voice", room=room)

    def on_persons(self, room, persons):
        c = self.c
        for p in persons:
            if p.get("posture") != "lying" or room == c.home["wake_up"]["room"]:
                continue        # 寝室の布団で横になっているのは転倒ではない (起床の決まりで見る)
            where = p.get("room") or room     # 人がいる部屋 (カメラの部屋とは限らない。ふすま越しに見えることがある)
            for cand in self.lying:
                if math.hypot(cand["x"] - p["x"], cand["y"] - p["y"]) < SAME_PLACE:
                    cand.update(x=p["x"], y=p["y"], last=c.time)
                    break
            else:
                self.lying.append({"x": p["x"], "y": p["y"], "room": where, "first": c.time, "last": c.time})

    def _heard_in(self, room):
        t = self.distress.get(room)
        return t is not None and self.c.time - t < VOICE_RECENT

    def step(self):
        c = self.c
        while self.voices:
            self._on_heard(*self.voices.pop(0))
        self.lying = [cand for cand in self.lying if c.time - cand["last"] < 3.0]
        if c.busy("fall"):
            return
        for cand in self.lying:
            recent = any(math.hypot(h["x"] - cand["x"], h["y"] - cand["y"]) < SAME_PLACE and c.time - h["time"] < 600
                         for h in self.handled)
            need = VOICE_CONFIRM if self._heard_in(cand["room"]) else (GAME_CONFIRM if c.game else FALL_CONFIRM)
            if cand["last"] - cand["first"] >= need and not recent:
                self.handled.append({"x": cand["x"], "y": cand["y"], "time": c.time})
                why = "横になった人と痛がる声" if need == VOICE_CONFIRM else "横になった人"
                c.log(f"{LABEL.get(cand['room'], cand['room'])}のカメラ: {why} ({cand['x']:.1f}, {cand['y']:.1f})"
                      " → 転倒の疑い", kind="fall_suspected", room=cand["room"], x=cand["x"], y=cand["y"])
                self._dispatch(cand)
                return
        # 声は聞こえたが、横になった人がカメラに映らない (死角): その部屋を見に行く
        for room, t in list(self.distress.items()):
            if not self._heard_in(room) or any(cand["room"] == room for cand in self.lying):
                continue
            del self.distress[room]
            spot = c.room_spot(room)
            if spot is None:
                continue
            c.log(f"{LABEL.get(room, room)}で痛がる声がしたが、カメラに横になった人が映らない → アヒルが見に行く",
                  kind="fall_suspected", room=room, x=spot[0], y=spot[1])
            self._dispatch({"x": spot[0], "y": spot[1], "room": room, "first": t, "last": t, "sweep": True})
            return

    def _dispatch(self, cand):
        c = self.c
        c.light(cand["room"])
        # カメラを置かない部屋 (脱衣所など) では、アヒルの目の絵も残さない
        privacy = cand["room"] in c.home.get("blind_rooms", {})
        if cand.get("sweep"):
            det = action("detectObject", "x-detect", objectType="person", sweep="full", duration=15.0, privacy=privacy)
        else:
            det = action("detectObject", "x-detect", objectType="person", targetX=cand["x"], targetY=cand["y"],
                         duration=12.0, privacy=privacy)
        job = Job("fall", 0).go(cand["x"], cand["y"], 2.0, [det])
        job.then(lambda j, cand=cand: self._result(j, cand))
        c.add_job(job)

    def _result(self, j, cand):
        c = self.c
        res = result_json(j.last[0] if j.last else None)
        if res.get("found") and res.get("posture") == "lying":
            c.log(f"アヒルの目でも横になった人を確認 (score {res.get('score')}) → 家族に知らせ、声をかける",
                  kind="fall_confirmed")
            c.attended[cand["room"]] = c.time
            c.m.publish("home/alerts", {"type": "fall", "level": "checking", "room": cand["room"],
                                        "x": cand["x"], "y": cand["y"], "time": round(c.time, 1),
                                        "confirmedBy": "microduck-01", "elapsed": round(c.time - cand["first"], 1),
                                        "summary": f"{LABEL.get(cand['room'], cand['room'])}で転んだようです。"
                                                   "いまロボットが様子を確かめています。",
                                        "evidence": res.get("image")}, qos=1)
            j.ev = {"who": CHILD, "room": cand["room"], "room_label": LABEL.get(cand["room"], cand["room"]),
                    "x": cand["x"], "y": cand["y"], "first_seen": cand["first"], "replies": [],
                    "image": res.get("image")}
            self._ask(j, "ask_ok")
            j.then(lambda j: self._heard(j, "ask_ok"))
        elif res.get("found"):
            c.log(f"アヒルが見ると立っていた ({res.get('posture')}) → 誤報として取り消し", kind="false_alarm")
            j.say("お元気そうでよかったです。")
        else:
            c.log("アヒルは人を見つけられなかった → 家族に確認を頼む", kind="fall_unconfirmed")
            c.m.publish("home/alerts", {"type": "fall_unconfirmed", "level": "warn", "room": cand["room"],
                                        "x": cand["x"], "y": cand["y"], "time": round(c.time, 1),
                                        "summary": f"{LABEL.get(cand['room'], cand['room'])}で転んだようですが、"
                                                   "ロボットが見つけられません。確かめてください。"}, qos=1)
            j.say("どこにいますか？ご家族に知らせます。")

    # ---- 声をかけて聞く
    @staticmethod
    def _ask(j, intent):
        j.say(ASK[intent], intent="ask_ok" if intent == "ask_ok_again" else intent).listen(LISTEN)

    def _heard(self, j, intent):
        """listen の結果を分けて記録し、次に何を聞くかを決める。"""
        heard = result_json(j.last[0] if j.last else None).get("heard") or []
        text = " ".join(h["text"] for h in heard if h.get("who"))
        lab, margin, _ = classify_reply(text)
        j.ev["replies"].append({"asked": ASK[intent], "text": text or None, "label": lab, "margin": margin})
        self.c.log(f"返事: 「{text}」→ {lab or '無言'}" if text else "返事がない", kind="reply", label=lab)
        nxt = Job("tmp", 0)
        if intent == "ask_ok" and lab in (None, "unclear"):
            self._ask(nxt, "ask_ok_again")
            nxt.then(lambda j: self._heard(j, "ask_ok_again"))
        elif intent in ("ask_ok", "ask_ok_again") and lab != "help":
            self._ask(nxt, "ask_pain")
            nxt.then(lambda j: self._heard(j, "ask_pain"))
        else:
            # もう一度見て (起き上がったか)、判断する
            nxt.look(j.ev.get("x"), j.ev.get("y"), 6.0)
            nxt.then(self._judge).until(lambda j: j.verdict is not None, 150.0).then(self._report)
        j.insert(*nxt.steps)

    def _judge(self, j):
        res = result_json(j.last[0] if j.last else None)
        j.ev.update(found=bool(res.get("found")), posture=res.get("posture"),
                    lying_seconds=round(self.c.time - j.ev["first_seen"], 1))
        j.verdict = None

        def run():
            j.verdict = judge({k: v for k, v in j.ev.items() if k != "first_seen"})
        threading.Thread(target=run, daemon=True).start()

    def _report(self, j):
        c = self.c
        v = j.verdict or {**judge({k: v for k, v in j.ev.items() if k != "first_seen"}, llm=""), "by": "rules (時間切れ)"}
        r = v["rules"]
        llm = f" / LLM ({v['by']}): {v['llm']['level']}" if v.get("llm") else ""
        c.log(f"判断: {v['level']} ({v['reason']}) [規則: {r['level']}{llm}]", kind="fall_judged", level=v["level"],
              rules_level=r["level"], llm_level=(v.get("llm") or {}).get("level"), by=v["by"])
        c.m.publish("home/alerts", {"type": "fall_report", "level": v["level"], "room": j.ev["room"],
                                    "time": round(c.time, 1), "summary": v["message_to_parent"], "reason": v["reason"],
                                    "by": v["by"], "replies": j.ev["replies"], "posture": j.ev.get("posture"),
                                    "evidence": j.ev.get("image")}, qos=1)
        say = {"info": "よかった。おうちの人にも知らせておくね。",
               "warn": "おうちの人に知らせたよ。じっとしていてね。",
               "emergency": "すぐにおうちの人に連絡したよ。そのまま動かないでね。"}[v["level"]]
        j.say(say)
