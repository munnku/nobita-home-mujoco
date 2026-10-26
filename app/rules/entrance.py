"""玄関 (検知 → 対応支援 → 通知): 鍵が開いた・扉が開いたら、アヒルが玄関を見に行き、だれか (YOLO の who) で動きを変える。

- 子ども (家にだれもいない away のとき): 帰ってきた → 「おかえり！」と迎え、おるすばんモードにする
- 子ども (おるすばん中): 中から開けた → 無断外出の心配。「外に出ちゃだめだよ」と言い、保護者に warn
- 家の大人: 「おかえりなさい」と迎え、おるすばんモードを終える (family)
- 知らない大人 (不審者): 「どなたですか？」と声をかけ、保護者に emergency (「見た」が無ければ 2 人目へ)。
  子どものいる部屋のスピーカーで「子ども部屋に行って、ドアをしめてね」と伝える
- おるすばん中に鍵や扉が開いたことは、だれか分かる前にも保護者に知らせる
- 扉が閉まったまま鍵が開いていれば、しばらくして施錠し、スピーカーで知らせる (鍵の閉め忘れ)
"""
from home.vda5050 import action

from .base import LABEL, M, Job, Rule, result_json

SAFE_ROOMS = ("kodomo", "dk", "ima")      # 不審者が来たとき、子どもに声をかけるスピーカー (玄関から離れた部屋)


class Entrance(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.cfg = c.home["entrance"]
        self.e = {"state": "idle", "closed_since": None}
        self.was_locked = None

    def step(self):
        c, cfg, e = self.c, self.cfg, self.e
        d = cfg["door"]
        if d not in c.locked:
            return
        unlocked_now = self.was_locked is True and not c.locked[d]
        self.was_locked = c.locked[d]
        if unlocked_now and c.mode == "alone":
            c.log("おるすばん中に玄関の鍵が開いた (スマートロック)", kind="unlocked", room="genkan")
            c.notify("door", "info", "玄関の鍵が開きました。ロボットがだれか確かめます。", room="genkan", category="検知")
        if c.door_open.get(d) and e["state"] == "idle":
            c.log("玄関の扉が開いた (開閉センサー) → アヒルが見に行く", kind="door_open", room="rouka")
            if c.mode == "alone" and not unlocked_now:
                c.notify("door", "info", "玄関の扉が開きました。ロボットがだれか確かめます。", room="genkan", category="検知")
            c.light("genkan")
            tg = cfg["look_target"]
            job = Job("entrance", 1).go(tg["x"] * M, tg["y"] * M, 2.0, [
                action("detectObject", "x-detect", objectType="person", targetX=tg["x"] * M, targetY=tg["y"] * M,
                       duration=8.0)])
            job.then(self._result)
            c.add_job(job)
            e["state"] = "greeting"
        if e["state"] != "idle" and not c.door_open.get(d) and not c.locked.get(d):
            e["closed_since"] = e["closed_since"] or c.time
            if c.time - e["closed_since"] >= cfg["auto_lock_after"]:
                c.homie_set(f"lock-{d}", "lock", "locked", "true")
                c.speaker("dk", "玄関の鍵を閉めました。")
                c.log(f"玄関が閉まってから {cfg['auto_lock_after']} 秒鍵が開いていた → 施錠した", kind="auto_lock")
                e.update(state="idle", closed_since=None)
        else:
            e["closed_since"] = None

    def _result(self, j):
        c = self.c
        res = result_json(j.last[0] if j.last else None)
        if not res.get("found"):
            c.log("玄関に人が見えない", kind="no_one")
            j.say("どなたですか？")
            if c.mode == "alone":
                c.notify("door", "warn", "玄関が開きましたが、ロボットにはだれも見えません。確かめてください。", room="genkan",
                         category="検知")
            return
        # だれかはアヒルの目の YOLO (who: child / adult / stranger) で見分ける。見分けない重みのときは、
        # だれもいない家 (away) に鍵を開けて入ってきたのは子ども、家族がいるときは家の人とみなす
        who = res.get("who") or ("child" if c.mode in ("away", "alone") else "adult")
        if who == "stranger":
            c.log("玄関に知らない大人がいる → 声をかけ、保護者に知らせる", kind="stranger", who=who)
            j.say("どなたですか？家の人を呼びました。", intent="ask_who")
            room = c.child_room()
            c.speaker(room if room in SAFE_ROOMS else "ima",
                      "知らない人が来たよ。子ども部屋に行って、ドアをしめて待っていてね。", intent="hide")
            c.notify("stranger", "emergency", "玄関に知らない人が入ってきました。すぐに確かめてください。", room="genkan",
                     category="通知", evidence=res.get("image"))
            return
        if who == "child" and c.mode == "alone" and "fire" in c.emergency.active:
            c.log("火事のときに子どもが玄関を開けた → そのまま外へ", kind="evacuating", who=who)
            j.say("そのまま外に出て、ちかくの おとなの人に「火事です」と言ってね！", intent="evacuate")
            return
        if who == "child" and c.mode == "alone":
            c.log("おるすばん中の子どもが玄関を開けた → 外に出ないよう声をかける", kind="outing", who=who)
            j.say("外に出ちゃだめだよ！ママが帰るまで、おうちで待っていようね。", intent="warn")
            c.notify("outing", "warn", "子どもが玄関の扉を開けました。ロボットが外に出ないよう声をかけています。",
                     room="genkan", category="通知", evidence=res.get("image"))
            return
        if who == "child":
            c.log("玄関に子どもがいた → 迎える", kind="greet", who=who)
            j.say("おかえり！手を洗ってね。", intent="greet")
            d = c.home["robot"]["dock"]
            j.go(d["x"] * M, d["y"] * M, 0.3)       # 先に居間へ戻る (廊下で子どもの通り道をふさがない)
            if c.mode == "away":
                # 子どもが帰ってきたが、家の大人は出かけている: おるすばんモードにする
                c.m.publish("home/mode", "alone", retain=True)
                c.log("子どもが帰ってきた。ママは出かけているので、おるすばんモードにする", kind="alone_start")
                c.notify("home", "info", "子どもが帰ってきました。おるすばんの見守りを始めます。", room="genkan")
            return
        c.log("玄関に家の人がいた → 迎える", kind="greet", who=who)
        j.say("おかえりなさい！")
        if c.mode == "alone":
            c.m.publish("home/mode", "family", retain=True)
            c.log(f"家の大人が帰ってきた → おるすばんモードを終える ({LABEL.get('genkan')})", kind="alone_end")
