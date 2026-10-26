"""司令塔の決まりごと (rule) と、アヒルの仕事 (Job) の共通部分。

決まりごとは Rule を継いで step() を書く。司令塔 (Coordinator) の共有の状態 (時刻、見守りカメラの結果、
アヒルの state など) は self.c から読み、家電は self.c.light / speaker / homie_set、アヒルには self.c.add_job で頼む。
見守りカメラの結果を 1 回ずつ見たいときは on_persons(room, persons) を書く。
"""
import json
import math

M = 0.91
LABEL = {"dk": "DK", "ima": "居間", "ousetsu": "応接室", "rouka": "廊下", "kodomo": "子ども部屋", "genkan": "玄関",
         "senmen": "脱衣所", "toilet": "トイレ", "yokushitsu": "浴室"}


def secs(clock):
    h, m, s = (int(v) for v in clock.split(":"))
    return h * 3600 + m * 60 + s


def result_json(state):
    """VDA 5050 の actionState の resultDescription (JSON) を dict にする。"""
    try:
        return json.loads(state.get("resultDescription") or "{}") if state else {}
    except ValueError:
        return {}


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Job:
    """アヒルの仕事: 手順を順に行う。
    go: 行く (着いたら actions)、say: 話す、listen: 聞く、look: その場で人を探す、
    then: 関数を呼ぶ (関数の中で手順を足してよい)、until: 条件が真になるまで待つ。"""

    def __init__(self, name, priority):
        self.name, self.priority = name, priority
        self.steps = []
        self.k = 0
        self.sent = False
        self.results = {}

    def go(self, x, y, dev=0.3, actions=()):
        self.steps.append(("go", (x, y, dev, list(actions))))
        return self

    def say(self, text, intent=None):
        self.steps.append(("say", (text, intent)))
        return self

    def listen(self, duration=8.0):
        self.steps.append(("listen", duration))
        return self

    def look(self, x=None, y=None, duration=6.0):
        self.steps.append(("look", (x, y, duration)))
        return self

    def then(self, fn):
        self.steps.append(("fn", fn))
        return self

    def until(self, pred, timeout=60.0):
        self.steps.append(("until", (pred, timeout)))
        return self

    def insert(self, *steps):
        """いまの手順の次に手順を差し込む (then の関数の中から、答えに応じて手順を足すとき)。
        steps は別の Job で作った手順 (Job(...).say(...).steps)。"""
        self.steps[self.k + 1:self.k + 1] = list(steps)


class Rule:
    def __init__(self, c):
        self.c = c

    def step(self):
        pass

    def on_persons(self, room, persons):
        pass
