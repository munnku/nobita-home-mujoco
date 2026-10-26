"""タスク一覧: 決まりごと (app/rules/) が作った仕事 (Job) を、重さの順にためて 1 つずつ進める。

- 受け持ち (lane) ごとに同時に動く: アヒル (duck、VDA 5050)・インターホン (intercom)。
  片付け係の Stretch は見回りの決まり (patrol.py) が動かし、ここには表示だけ出す (show)。
- 重さ (priority、小さいほど先):
  0 火事・地震・停電・返事が無い / 1 転倒・けが・先読みの警告・不審者・玄関 / 2 死角の声かけ・声の相談 /
  3 保護者からの依頼 (WoT) / 4 見回り
- 割り込み: 重い仕事が来たら、いまの仕事が interruptible (見回り・依頼・声の相談) か、来た仕事が重さ 0 なら止める。
  止めた仕事に restart (仕事を作り直す関数) があれば、作り直して待ちに戻す (見回りは自分で続きから回る)。
- 一覧は変わるたびに home/tasks (retain) に出す (録画の画面とアプリで見る)。
"""
import json

from home.vda5050 import action, make_order, topic

JOB_TIMEOUT = 300.0    # 1 つの手順にこれ以上かかったら仕事をやめる [s]
# 名前 → (一覧に出す名前, 分類)。分類は 予防・検知・対応支援・通知
LABELS = {"fall": ("転倒の確認と安否確認", "対応支援"), "danger": ("危ない所の手前へ先回り", "予防"),
          "blind": ("カメラの無い部屋の声かけ", "検知"), "patrol": ("床の小物の見回り", "予防"),
          "entrance": ("玄関の確認", "検知"), "voice": ("声を聞いて様子を見る", "対応支援"),
          "visitor": ("インターホンの応対", "対応支援"), "disaster": ("災害のときの声かけ", "対応支援"),
          "goToRoom": ("保護者の依頼: 部屋へ行く", "対応支援"), "checkRoom": ("保護者の依頼: 部屋を見る", "対応支援"),
          "returnToDock": ("待機場所へ戻る", "-")}


class Lane:
    """1 つの受け持ち (アヒル・インターホン) の待ち行列と、手順の進め方。"""

    def __init__(self, c, name):
        self.c, self.name = c, name
        self.jobs, self.job = [], None

    def ready(self):
        return True

    def step(self):
        c = self.c
        if not self.ready():
            return
        if self.job is None:
            if not self.jobs:
                return
            self.job = self.jobs.pop(0)
            c.log(f"仕事を始める: {self.job.name}" + ("" if self.name == "duck" else f" ({self.name})"))
            c.tasks.publish()
        j = self.job
        if j.k >= len(j.steps):
            c.log(f"仕事が終わった: {j.name}")
            self.job = None
            c.tasks.publish()
            return
        kind, arg = j.steps[j.k]
        if kind == "fn":
            arg(j)
            j.k += 1
            return
        if kind == "until":
            pred, timeout = arg
            j.t_wait = getattr(j, "t_wait", None) or c.time
            if pred(j) or c.time - j.t_wait > timeout:
                j.t_wait = None
                j.k += 1
            return
        if not j.sent:
            c.n += 1
            j.watch = self.send(j, kind, arg, f"{j.name}-{c.n}")
            j.sent, j.t_sent = True, c.time
            return
        states = self.states(j)
        if all(s and s["actionStatus"] in ("FINISHED", "FAILED") for s in states):
            for a, s in zip(j.watch, states):
                j.results[a.rsplit("-", 1)[-1]] = s
            j.last = states
            j.k += 1
            j.sent = False
        elif c.time - j.t_sent > JOB_TIMEOUT:
            c.log(f"時間切れ: {j.name}")
            self.job = None
            c.tasks.publish()

    def stop(self):
        """いまの仕事を止める (アヒルなら cancelOrder)。"""
        self.job = None

    def send(self, j, kind, arg, oid):
        raise NotImplementedError

    def states(self, j):
        raise NotImplementedError


class DuckLane(Lane):
    """アヒル: 手順を VDA 5050 の order / instantActions にして送り、state の actionStates で終わりを見る。"""

    def ready(self):
        return "agvPosition" in self.c.robot      # アヒルの位置が届くまで (シミュレーションの起動中) は待たせる

    def _instant(self, a):
        self.c.m.publish(topic("instantActions"), {"headerId": self.c.n, "actions": [a]}, qos=1)
        return [a["actionId"]]

    def send(self, j, kind, arg, oid):
        c = self.c
        if kind == "go":
            x, y, dev, acts = arg
            end = action("wait", f"{oid}-end", duration=0.1)
            # アクションの ID は order ごとに一意にする (VDA 5050)
            acts = [{**a, "actionId": f"{oid}-{a['actionId'].split('-', 1)[-1]}"} for a in acts]
            pos = c.robot.get("agvPosition", {})
            c.m.publish(topic("order"), make_order(oid, 0, [(pos.get("x", 0.0), pos.get("y", 0.0), None, 0.3),
                                                          (x, y, None, dev)], acts + [end]), qos=1)
            return [a["actionId"] for a in acts] + [end["actionId"]]
        if kind == "say":
            text, intent = arg
            return self._instant(action("speak", f"{oid}-speak", text=text, **({"intent": intent} if intent else {})))
        if kind == "listen":
            return self._instant(action("listen", f"{oid}-listen", duration=float(arg)))
        if kind == "look":
            x, y, dur = arg
            kw = {"targetX": x, "targetY": y} if x is not None else {}
            return self._instant(action("detectObject", f"{oid}-detect", objectType="person", duration=float(dur), **kw))
        raise ValueError(kind)

    def states(self, j):
        return [next((s for s in self.c.robot.get("actionStates", []) if s["actionId"] == a), None) for a in j.watch]

    def stop(self):
        c = self.c
        c.n += 1
        c.m.publish(topic("instantActions"), {"headerId": c.n, "actions": [action("cancelOrder", f"cancel-{c.n}")]},
                    qos=1)
        self.job = None


class IntercomLane(Lane):
    """インターホン: say は玄関の子機のスピーカー、listen は子機のマイク (homie/intercom-genkan/intercom/heard)。"""

    DEVICE = "intercom-genkan"

    def __init__(self, c, name):
        super().__init__(c, name)
        self.heard = []           # [(時刻, 声)]
        self.runs = {}            # actionId → 状態
        c.m.subscribe(f"homie/{self.DEVICE}/intercom/heard",
                      lambda t, p: self.heard.append((self.c.time, p)) if isinstance(p, dict) and p.get("text") else None)

    def send(self, j, kind, arg, oid):
        c = self.c
        if kind == "say":
            text, intent = arg
            c.homie_set(self.DEVICE, "intercom", "say", json.dumps({"text": text, "intent": intent}, ensure_ascii=False))
            self.runs[oid] = {"type": "say", "t0": c.time, "dur": 1.0 + 0.12 * len(text)}
        elif kind == "listen":
            self.runs[oid] = {"type": "listen", "t0": c.time, "dur": float(arg), "heard": []}
        else:
            raise ValueError(f"インターホンでは {kind} はできない")
        return [oid]

    def states(self, j):
        c = self.c
        out = []
        for oid in j.watch:
            r = self.runs[oid]
            el = c.time - r["t0"]
            if r["type"] == "say":
                done, desc = el >= r["dur"], ""
            else:
                r["heard"] = [{"who": v.get("who"), "text": v.get("text"), "time": t} for t, v in self.heard if t >= r["t0"]]
                quiet = r["heard"] and c.time - r["heard"][-1]["time"] > 1.5
                done, desc = el >= r["dur"] or quiet, json.dumps({"heard": r["heard"]}, ensure_ascii=False)
            out.append({"actionId": oid, "actionStatus": "FINISHED" if done else "RUNNING", "resultDescription": desc})
        return out


class Tasks:
    def __init__(self, c):
        self.c = c
        self.lanes = {"duck": DuckLane(c, "duck"), "intercom": IntercomLane(c, "intercom")}
        self.shown = {}           # 表示だけのタスク (Stretch の片付けなど) 名前 → 項目
        self._last = None

    @property
    def duck(self):
        return self.lanes["duck"]

    def add(self, job):
        c = self.c
        lane = self.lanes[getattr(job, "lane", "duck")]
        lane.jobs.append(job)
        lane.jobs.sort(key=lambda j: j.priority)
        c.log(f"タスクに足した: {job.name} (重さ {job.priority})", kind="task_added", task=job.name, priority=job.priority)
        cur = lane.job
        if cur and job.priority < cur.priority and (getattr(cur, "interruptible", False) or job.priority == 0):
            c.log(f"{cur.name} を中断して {job.name} を先にする", kind="preempt", task=cur.name, by=job.name)
            lane.stop()
            if callable(getattr(cur, "restart", None)):
                lane.jobs.append(cur.restart())
                lane.jobs.sort(key=lambda j: j.priority)
        self.publish()

    def busy(self, prefix):
        return any(j.name.startswith(prefix) for lane in self.lanes.values()
                   for j in lane.jobs + ([lane.job] if lane.job else []))

    def cancel(self, name):
        for lane in self.lanes.values():
            lane.jobs = [j for j in lane.jobs if j.name != name]
            if lane.job and lane.job.name == name:
                lane.stop()
                self.c.log(f"仕事を取り消した: {name}")
        self.publish()

    def show(self, name, label=None, category="予防", lane="stretch"):
        """表示だけのタスク (label が None なら消す)。"""
        if label is None:
            self.shown.pop(name, None)
        else:
            self.shown[name] = {"name": name, "label": label, "category": category, "lane": lane, "state": "running",
                                "priority": 4}
        self.publish()

    def step(self):
        for lane in self.lanes.values():
            lane.step()

    def items(self):
        out = []
        for lane in self.lanes.values():
            for j, state in ([(lane.job, "running")] if lane.job else []) + [(j, "waiting") for j in lane.jobs]:
                label, cat = LABELS.get(j.name, (j.name, "-"))
                out.append({"name": j.name, "label": getattr(j, "label", None) or label,
                            "category": getattr(j, "category", None) or cat, "priority": j.priority,
                            "lane": lane.name, "state": state})
        out += list(self.shown.values())
        return sorted(out, key=lambda t: (t["state"] != "running", t["priority"]))

    def publish(self):
        items = self.items()
        if items != self._last:
            self._last = items
            self.c.m.publish("home/tasks", items, retain=True)
