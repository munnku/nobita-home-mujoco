"""先読み: 子どもが危ない場所 (コンロの前・玄関の土間・浴室・階段) へ向かっているのを、着く前に見つけて止めに行く。

おるすばんモード (home/mode = alone) のときだけ働く (家族がいるときにママが台所に立っても警告しない)。
- 追跡: 見守りカメラの人の位置を、近い点を同じ人とみなしてつなぐ (2 台のカメラに映っても 1 人にまとめる)
- 行き先の推定: 行き先の候補 = 危ない場所 + ふつうの場所 (部屋)。候補ごとに家の地図の上の歩く距離 (距離の場) を前もって
  求めておき、0.5 秒ごとに「その候補への距離がどれだけ縮んだか」を証拠として足す (ベイズ更新、古い証拠は少しずつ忘れる)。
  まっすぐ近づいていく候補ほど確率が上がる (「人は目的地へ近道を選ぶ」という仮定。Baker らの逆計画の簡単な形)
- 先読みは 2 段:
  1. 備える: 危ない場所の確率が P_PREPARE 以上で、着くまでの時間 (歩く距離 ÷ 速さ) が ETA_PREPARE 秒以下なら、
     アヒルが危ない場所の手前 (子どもの経路の上) へ黙って向かう。アヒルは遅いので早めに動く (外れても歩くだけで済む)
  2. 警告する: 確率が P_WARN 以上で ETA_WARN 秒以下、または NEAR_WARN より近づいたら、その部屋のスピーカーですぐに
     警告し、アヒルも歩きながら声をかける
  備えたあとで確率が下がれば (テーブルへ行くだけだった、など) アヒルの仕事を取り消す
  確率には危なさの重み (home.yaml の severity) を掛けて比べる (外へ出る・お風呂は低い確率でも早めに動く)
  アヒルが手前に着いたら、警告の前でもやさしく声をかける
"""
import math
import os
import sys

import numpy as np

from home.vda5050 import action, topic

from .base import M, Job, Rule

P_PREPARE = 0.35       # 備える (アヒルが向かう) 確率
ETA_PREPARE = 15.0     # 備える: 着くまでの時間 [s]
P_WARN = 0.45          # 警告する確率 (危なさの重みを掛けたもの)
P_WARN_RAW = 0.4       # 警告するには、重みを掛ける前の確率もこれ以上 (声を出すのは確からしいときだけ)
ETA_WARN = 6.0         # 警告する: 着くまでの時間 [s]
NEAR_WARN = 0.8        # 確率にかかわらず、これより近づいたら警告する [m]
P_CLEAR = 0.15         # 備えたあと、確率がこれより下がって CLEAR_AFTER 秒たてば取り消す
CLEAR_AFTER = 4.0
BETA = 4.0             # 距離が 1 m 縮むごとに足す証拠の大きさ
FORGET = 0.85          # 1 秒ごとに古い証拠に掛ける値 (0.5 秒ごとに更新するので、その平方根を掛ける)
STEP = 0.5             # 行き先を推定する間隔 [s]
TRACK_GATE = 1.0       # 同じ人とみなす距離 [m]
ARRIVED = 0.3          # ふつうの行き先にこれより近ければ、そこに着いたとみなす [m]
STOP_GUARD = 0.9      # アヒルが立つ所: 危ない場所からこの距離の、子どもの経路の上 [m]
REPEAT_WARN = 6.0      # 歩きながら声をかける間隔 [s]
SPEAKERS = ("nobita", "dk", "ima", "rouka")


class DistanceFields:
    """家の 1 階の地図 (FloorMap、扉は鍵のかかる外の扉のほかは開いているとみなす) の上で、候補ごとの歩く距離の場。"""

    def __init__(self, house_dir, goals):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(root, "sim"))
        import mujoco
        from floormap import FloorMap
        from house import build_with_info, load
        lay = load(house_dir)
        self.lay = lay
        xml, door_open = build_with_info(house_dir)
        model = mujoco.MjSpec.from_string(xml).compile()
        data = mujoco.MjData(model)
        for name, d in lay["doors"].items():
            if not d.get("lock"):
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"door_{name}")
                data.qpos[model.jnt_qposadr[j]] = door_open[name]
        self.map = FloorMap(model, data, lay["rooms"], lay["module"])
        self.map.update()
        self.fields = {g["name"]: self._field(g["rect"]) for g in goals}

    def _field(self, rect):
        """rect [m] の中の通れるセルからの歩く距離 [m] (8 近傍の Dijkstra)。"""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        fm = self.map
        free = fm.free
        idx = -np.ones(free.shape, int)
        cells = np.argwhere(free)
        idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
        rows, cols, w = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = cells
            b = cells + [di, dj]
            ok = (b[:, 0] >= 0) & (b[:, 0] < free.shape[0]) & (b[:, 1] >= 0) & (b[:, 1] < free.shape[1])
            a, b = a[ok], b[ok]
            ok = free[b[:, 0], b[:, 1]]
            a, b = a[ok], b[ok]
            rows += list(idx[a[:, 0], a[:, 1]])
            cols += list(idx[b[:, 0], b[:, 1]])
            w += [math.hypot(di, dj) * 0.05] * len(a)
        g = coo_matrix((w, (rows, cols)), shape=(len(cells), len(cells))).tocsr()
        x0, x1, y0, y1 = rect
        src = [k for k, (i, j) in enumerate(cells) if x0 <= fm.to_xy(i, j)[0] <= x1 and y0 <= fm.to_xy(i, j)[1] <= y1]
        if not src:          # 範囲の中に通れるセルが無い (家具の上など) なら、真ん中に一番近い通れるセル
            c = fm.nearest_free((x0 + x1) / 2, (y0 + y1) / 2, max_r=1.5)
            src = [int(idx[c])] if c else []
        d = dijkstra(g, directed=False, indices=src, min_only=True)
        out = np.full(free.shape, np.inf)
        out[cells[:, 0], cells[:, 1]] = d
        return out

    def dist(self, name, x, y):
        fm = self.map
        c = fm.nearest_free(x, y, max_r=0.6)
        return float(self.fields[name][c]) if c else math.inf

    def toward(self, name, x, y, stop_at):
        """(x, y) から name へ距離の場を下って、残りが stop_at になる点 [m] (子どもの通り道の上)。"""
        fm = self.map
        c = fm.nearest_free(x, y, max_r=0.6)
        if c is None:
            return None
        f = self.fields[name]
        i, j = c
        for _ in range(2000):
            if f[i, j] <= stop_at:
                break
            best = min(((f[i + a, j + b], i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                        if 0 <= i + a < f.shape[0] and 0 <= j + b < f.shape[1]), key=lambda t: t[0])
            if best[0] >= f[i, j]:
                break
            _, i, j = best
        return fm.to_xy(i, j)


class Predict(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.hazards = {h["name"]: h for h in c.home.get("hazards", [])}
        goals = [{"name": n, "rect": [v * M for v in h["rect"]]} for n, h in self.hazards.items()]
        for room, (x, y) in c.home.get("everyday_spots", {}).items():
            goals.append({"name": f"room:{room}", "rect": [(x - 0.3) * M, (x + 0.3) * M, (y - 0.3) * M, (y + 0.3) * M]})
        self.goals = [g["name"] for g in goals]
        self.df = DistanceFields(os.path.join(c.root, "houses", "nobita"), goals) if self.hazards else None
        self.tracks = []         # [{"x","y","t","hist":[(t,x,y)], "ll":{goal: 対数尤度}, "v", "warned":{}}]
        self.next_t = 0.0
        self.active = {}         # 危ない場所 → 警告している間の状態

    # ---- 追跡
    def on_persons(self, room, persons):
        if self.df is None or self.c.mode != "alone":
            return
        c = self.c
        for p in persons:
            if p.get("posture") != "standing" or c.level_of(room) != "1F":
                continue
            best = min(self.tracks, key=lambda tr: math.hypot(tr["x"] - p["x"], tr["y"] - p["y"]), default=None)
            if best and math.hypot(best["x"] - p["x"], best["y"] - p["y"]) < TRACK_GATE:
                if best["t"] == c.time:          # 同じ時刻に別のカメラでも見えた: 平均する
                    best.update(x=(best["x"] + p["x"]) / 2, y=(best["y"] + p["y"]) / 2)
                else:
                    best.update(x=p["x"], y=p["y"], t=c.time)
                best["room"] = p.get("room") or room
            else:
                # 危ない場所の中で見つかり始めた人 (帰ってきて玄関の土間にいる子ども) は、一度そこから出るまで
                # その場所について警告しない (外から入ってきただけで「外に出ないで」と言わないように)
                inside = {n for n in self.hazards if self.df.dist(n, p["x"], p["y"]) <= NEAR_WARN}
                self.tracks.append({"x": p["x"], "y": p["y"], "t": c.time, "hist": [], "room": p.get("room") or room,
                                    "ll": {g: 0.0 for g in self.goals}, "v": 0.0, "came_from": inside})

    # ---- STEP 秒ごとに行き先を推定する
    def step(self):
        c = self.c
        if self.df is None or c.mode != "alone":
            return
        self.tracks = [tr for tr in self.tracks if c.time - tr["t"] < 3.0]
        self._keep_warning()
        if c.time < self.next_t:
            return
        self.next_t = c.time + STEP
        for tr in self.tracks:
            d = {g: self.df.dist(g, tr["x"], tr["y"]) for g in self.goals}
            if tr["hist"]:
                t0, d0 = tr["hist"][-1]
                dt = max(c.time - t0, 1e-3)
                gains = {g: (d0[g] - d[g]) if math.isfinite(d0[g]) and math.isfinite(d[g]) else 0.0 for g in self.goals}
                for g in self.goals:
                    tr["ll"][g] = FORGET ** dt * tr["ll"][g] + BETA * gains[g]
                tr["v"] = 0.5 * tr["v"] + 0.5 * max(gains.values()) / dt     # 一番近づいている候補への速さ
            tr["hist"] = (tr["hist"] + [(c.time, d)])[-10:]
            prob = self._posterior(tr["ll"])
            tr["prob"] = prob
            # ふつうの行き先 (トイレの扉の前など) に着いた人は、そこが行き先だった。危ない場所のすぐそばでなければ見送る
            # (着く直前は、着いた所への距離が縮まらなくなり、その先の危ない場所の確率が上がって見えるため)
            arrived = min((d[g] for g in self.goals if g.startswith("room:")), default=math.inf) <= ARRIVED
            for name, h in self.hazards.items():
                p, dist = prob[name], d[name]
                if name in tr.get("came_from", ()):
                    if dist > NEAR_WARN + 0.5:
                        tr["came_from"].discard(name)       # 出ていった。ここからはふつうに見る
                    continue
                risk = min(1.0, p * float(h.get("severity", 1.0)))
                eta = dist / tr["v"] if tr["v"] > 0.15 else math.inf
                a = self.active.get(name)
                high = risk >= P_WARN and p >= P_WARN_RAW and eta <= ETA_WARN and not arrived
                streak = tr.setdefault("streak", {})
                streak[name] = streak.get(name, 0) + 1 if high else 0
                # すぐそばまで来たら確率にかかわらず警告する。ただし近づいているときだけ (居間の真ん中に立っているだけで
                # 隣の廊下の階段について何度も警告していた)
                near = dist <= NEAR_WARN and tr["v"] > 0.1 and not arrived
                if streak[name] >= 2 or near:      # 声を出すのは 2 回続けて確からしいとき (外れを減らす)
                    self._warn(name, h, tr, p, eta, dist)
                elif risk >= P_PREPARE and eta <= ETA_PREPARE and not arrived:
                    self._prepare(name, h, tr, p, eta, dist)
                elif a and not high:
                    if risk >= P_CLEAR:
                        a["low_since"] = None
                    elif c.time - (a.get("low_since") or c.time) >= CLEAR_AFTER:
                        self._clear(name, p)
                    else:
                        a["low_since"] = a.get("low_since") or c.time

    @staticmethod
    def _posterior(ll):
        m = max(ll.values())
        e = {g: math.exp(v - m) for g, v in ll.items()}
        s = sum(e.values())
        return {g: v / s for g, v in e.items()}

    # ---- 備える・警告する
    def _describe(self, h, tr, p, eta, dist):
        top = sorted(tr["prob"].items(), key=lambda kv: -kv[1])[:3]
        return (f"のび太が{h['label']}へ向かっているかも (確率 {p:.2f}、あと {dist:.1f} m・"
                f"{'%.0f 秒' % eta if math.isfinite(eta) else '止まっている'}) 行き先の候補 "
                + ", ".join(f"{g.replace('room:', '')} {q:.2f}" for g, q in top))

    def _prepare(self, name, h, tr, p, eta, dist):
        """アヒルが危ない場所の手前へ向かう (まだ声はかけない)。"""
        c = self.c
        if c.game:
            return                          # かくれんぼの最中はアヒルは遊んでいる (走り回るのは遊び)
        if name in self.active or c.time - h.get("_last", -1e9) < 60.0:
            return
        for other, a in list(self.active.items()):
            if tr["prob"].get(other, 0.0) >= p:
                return                      # 別の危ない場所に備えている (そちらの方が確からしい)
            self._clear(other, tr["prob"].get(other, 0.0))
        h["_last"] = c.time
        c.log("先読み (備える): " + self._describe(h, tr, p, eta, dist) + " → アヒルが先回りする",
              kind="predict_prepare", hazard=name, p=round(p, 2), eta=round(eta, 1) if math.isfinite(eta) else None,
              dist=round(dist, 2), room=tr.get("room"))
        guard = self.df.toward(name, tr["x"], tr["y"], STOP_GUARD) or (tr["x"], tr["y"])
        job = Job("danger", 0).go(guard[0], guard[1], 0.4)
        job.then(lambda j, name=name: self._arrived(j, name))
        c.add_job(job)
        self.active[name] = {"t": c.time, "next": c.time + REPEAT_WARN, "track": tr, "min_dist": dist,
                             "say": h.get("say_walk", h["say"]), "warned": False}

    def _warn(self, name, h, tr, p, eta, dist):
        """スピーカーですぐに警告する (アヒルがまだなら向かわせる)。"""
        c = self.c
        a = self.active.get(name)
        if a and a.get("warned"):
            return
        if c.game:
            # かくれんぼの最中は、外へ出る・お風呂のような重い危なさだけスピーカーで警告する (アヒルは遊びを続ける)
            if float(h.get("severity", 1.0)) >= 1.5 and c.time - h.get("_last", -1e9) >= 60.0:
                h["_last"] = c.time
                c.log("先読み (遊びの最中の警告): " + self._describe(h, tr, p, eta, dist), kind="predict_hazard",
                      hazard=name, p=round(p, 2), dist=round(dist, 2))
                spk = tr.get("room") if tr.get("room") in SPEAKERS else h.get("speaker")
                if spk:
                    c.speaker(spk, h["say"], intent="warn")
            return
        if a is None:
            if c.time - h.get("_last", -1e9) < 60.0:
                return
            self._prepare(name, h, tr, p, eta, dist)
            a = self.active.get(name)
            if a is None:
                return                      # 別の危ない場所の方が確からしい
        a["warned"] = True
        c.log("先読み (警告): " + self._describe(h, tr, p, eta, dist), kind="predict_hazard", hazard=name,
              p=round(p, 2), eta=round(eta, 1) if math.isfinite(eta) else None, dist=round(dist, 2), room=tr.get("room"))
        spk = tr.get("room") if tr.get("room") in SPEAKERS else h.get("speaker")
        if spk:
            c.speaker(spk, h["say"], intent="warn")
        a["next"] = c.time + REPEAT_WARN

    def _clear(self, name, p):
        self.active.pop(name, None)
        self.c.log(f"先読みを取り消す: {self.hazards[name]['label']}へ向かう確率が下がった ({p:.2f})", kind="predict_cleared",
                   hazard=name)
        self.c.cancel_jobs("danger")

    def _arrived(self, j, name):
        """手前に着いた: 警告していれば声をかける。"""
        j.say(self.hazards[name].get("say_near", "ここはあぶないよ。こっちであそぼう。"), intent="warn")
        j.then(lambda j, name=name: self._done(name))

    def _keep_warning(self):
        """アヒルが向かっている間は、歩きながら声をかける。子どもが止まったら記録する。"""
        c = self.c
        for name, a in list(self.active.items()):
            tr = a["track"]
            dist = self.df.dist(name, tr["x"], tr["y"])
            if dist < a["min_dist"] - 0.05:
                a.update(min_dist=dist, t_closer=c.time)
            elif a.get("warned") and not a.get("stopped") and c.time - a.get("t_closer", a["t"]) > 3.0:
                a["stopped"] = True
                c.log(f"のび太が止まった ({self.hazards[name]['label']}まであと {dist:.1f} m)", kind="child_stopped",
                      hazard=name, dist=round(dist, 2))
            if a.get("warned") and c.time >= a["next"] and c.job is not None and c.job.name == "danger" and c.robot.get("driving"):
                a["next"] = c.time + REPEAT_WARN
                c.n += 1
                sp = action("speak", f"warn-{c.n}", text=a["say"], intent="warn")
                sp["blockingType"] = "NONE"       # 歩きながら話す
                c.m.publish(topic("instantActions"), {"headerId": c.n, "actions": [sp]}, qos=1)

    def _done(self, name):
        a = self.active.pop(name, None)
        if a and not a.get("stopped"):
            self.c.log(f"アヒルが{self.hazards[name]['label']}の手前に着いた", kind="guard_arrived", hazard=name)
