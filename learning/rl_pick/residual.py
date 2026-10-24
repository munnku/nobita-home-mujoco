"""計算で決めた拾い方 (sim/stretch.py の pick_step) に、学習した補正 (残差) を足して拾う。

指先が床に触れて閉じ始める所 (phase "close") から先 — 閉じる・挟む・持ち上げる・腕を縮める — の間、
0.02 s ごとに方策が次の補正を出す (行動 5 つ、どれも -1〜1):
  0: lift の目標 + 1 cm × a   (押しつけ・持ち上げの強さ。lift は約 400 N/m)
  1: arm の目標 + 1 cm × a    (横の寄せ)
  2: wrist_roll の目標 + 0.1 rad × a
  3: gripper の目標 + 3 mm × a (閉じる量)
  4: 指の力の上限 × exp(0.5 a) (1〜15 N)
行動がすべて 0 なら、今の拾い方と同じ動きになる。方策は numpy だけで動く (learning/rl_pick/policy_*.npz)。
"""
import math

import numpy as np

from sim.objects import KINDS

KIND_LIST = list(KINDS)
PHASES = ("close", "squeeze", "lift", "retract")
ACT_KEYS = ("lift", "arm", "wrist_roll", "gripper")
ACT_SCALE = np.array([0.01, 0.01, 0.1, 0.003])
N_ACT = 5
OBS_DIM = 3 + 3 + 2 + 3 + len(PHASES) + 1 + len(KIND_LIST) + 1 + 1 + N_ACT


class MLPPolicy:
    """学習した方策 (平均だけ使う)。npz: obs_mean, obs_std, w0, b0, w1, b1, w2, b2。"""

    def __init__(self, path=None, params=None):
        p = dict(np.load(path)) if path else params
        self.p = {k: np.asarray(v, np.float64) for k, v in p.items()}

    def __call__(self, obs):
        p = self.p
        x = np.clip((obs - p["obs_mean"]) / p["obs_std"], -10, 10)
        x = np.tanh(x @ p["w0"] + p["b0"])
        x = np.tanh(x @ p["w1"] + p["b1"])
        return x @ p["w2"] + p["b2"]


class ResidualPick:
    """Stretch 1 台の拾う動き。step() を pick_step の代わりに呼ぶ。"""

    def __init__(self, st, policy=None):
        self.st, self.policy = st, policy
        self.idx = [st.act[k] for k in ACT_KEYS]
        self.gi = st.act["gripper"]
        self.reset()

    def reset(self):
        self.st.reset_task()
        self.active = False
        self.last_a = np.zeros(N_ACT)

    def begin(self, target, kind, width):
        """閉じ始める所から補正を始める (pick_step が phase "close" に入った直後)。"""
        st, m, d = self.st, self.st.m, self.st.d
        self.active = True
        self.target, self.kind, self.width = np.asarray(target, float), kind, width
        j = st.joints()
        self.j0 = (j["lift"], j["arm"], float(d.ctrl[st.act["wrist_roll"]]))
        self.base_ctrl = d.ctrl[self.idx].copy()
        self.base_force = float(m.actuator_forcerange[self.gi, 1])
        self.last_a = np.zeros(N_ACT)

    def observe(self):
        st, m, d = self.st, self.st.m, self.st.d
        gc = st.grasp_point()
        rel = np.r_[st.to_base(self.target) - st.to_base(gc), self.target[2] - gc[2]]
        j = st.joints()
        left = right = False             # 左右の指が動く物に触れているか (触覚の代わり)
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if m.geom_bodyid[o] in st.free_bodies:
                    left |= g in st.left_geoms
                    right |= g in st.right_geoms
        ph = np.array([st.phase == p for p in PHASES], float)
        kind = np.array([self.kind == k for k in KIND_LIST], float)
        force = float(d.actuator_force[self.gi]) / 10.0
        return np.r_[rel * 20.0,
                     (j["lift"] - self.j0[0]) * 20.0, (j["arm"] - self.j0[1]) * 20.0, j["wrist_roll"] - self.j0[2],
                     j["gripper"] * 100.0, gc[2] * 10.0,
                     float(left), float(right), float(st.tip_touch()),
                     ph, min(st._elapsed(), 3.0),
                     kind, self.width * 20.0, force, self.last_a].astype(np.float64)

    def step(self, target, yaw, width, kind, action=None):
        """毎ステップ (制御の周期ごとに) 呼ぶ。pick_step と同じく 'running' / 'done' / 'failed' を返す。"""
        st, m, d = self.st, self.st.m, self.st.d
        if not self.active:
            r = st.pick_step(target, yaw=yaw, width=width)
            if r == "running" and st.phase == "close":
                self.begin(target, kind, width)
            return r
        if action is None:
            action = self.policy(self.observe()) if self.policy is not None else np.zeros(N_ACT)
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        # 計算で決めた拾い方の出力 (前のステップの補正を外したもの) に戻してから 1 ステップ進める
        d.ctrl[self.idx] = self.base_ctrl
        m.actuator_forcerange[self.gi] = [-self.base_force, self.base_force]
        r = st.pick_step(target, yaw=yaw, width=width)
        self.base_ctrl = d.ctrl[self.idx].copy()
        self.base_force = float(m.actuator_forcerange[self.gi, 1])
        if r == "running":
            ctrl = self.base_ctrl + ACT_SCALE * a[:4]
            for i, c in zip(self.idx, ctrl):
                lo, hi = m.actuator_ctrlrange[i]
                d.ctrl[i] = min(max(c, lo), hi)
            f = min(max(self.base_force * math.exp(0.5 * a[4]), 1.0), 15.0)
            m.actuator_forcerange[self.gi] = [-f, f]
        self.last_a = a
        return r
