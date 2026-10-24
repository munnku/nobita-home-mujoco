"""学習の場面: learning/eval_pick.py と同じ床 + 物 1 つ + Stretch (Euler・noslip 3・timestep 0.005)。

1 回ずつ台車の位置合わせから始めると遅いので、計算で決めた拾い方で「指先が床に触れて閉じ始める所」まで
進めた状態を先にたくさん作っておき (make_pool)、学習はそこから始める (PickEnv)。
乱数: 物の種類 5 つ・位置・向き・摩擦 0.5〜1.2・質量 ×0.7〜1.3・位置の見積もりの誤差 ±1〜2 cm・台車の止まる位置。
報酬: 物の持ち上がった高さ (6 cm まで) の増えた分 + 成功 (5 cm 以上上げて 2 秒保つ、eval_pick と同じ条件) で 1。
"""
import os
import pickle
import sys
import warnings

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "learning"))
import eval_pick as E  # noqa: E402
from rl_pick.residual import KIND_LIST, ResidualPick  # noqa: E402

POOL_DIR = os.path.join(ROOT, "build", "rl_pick")
ST_ATTRS = ("phase", "status", "reason", "_t0", "_grip", "_open", "_yaw_ref", "_hold", "_vw", "_cmd",
            "_last_t", "_dt", "held")
RP_ATTRS = ("target", "kind", "width", "j0", "base_ctrl", "base_force", "active")
HOLD = 2.0
T_MAX = 15.0


class KindSim:
    """物の種類 1 つぶんのモデル・Stretch・残差の拾い方。"""

    def __init__(self, kind):
        warnings.filterwarnings("ignore")
        self.kind = kind
        self.env = E.Env(kind, "noslip")
        self.m, self.d = self.env.m, self.env.d
        self.body = self.env.obj.body[self.env.name]
        self.mass0 = float(self.m.body_mass[self.body])
        self.inertia0 = self.m.body_inertia[self.body].copy()
        self.st = None

    def set_mass(self, f):
        self.m.body_mass[self.body] = self.mass0 * f
        self.m.body_inertia[self.body] = self.inertia0 * f

    def set_friction(self, mu):
        for g in self.env.geoms:
            self.m.geom_friction[g, 0] = mu

    def obj_z(self):
        return float(self.env.obj.pos(self.env.name)[2])


def make_states(kind, n, seed):
    """計算で決めた拾い方で、閉じ始める所までの状態を n 個作る。"""
    ks = KindSim(kind)
    rng = np.random.default_rng(seed)
    out = []
    width = E.grip_width(kind)
    while len(out) < n:
        mf = rng.uniform(0.7, 1.3)
        ks.set_mass(mf)
        st, true, est, est_yaw, info = ks.env.reset(rng, False, pos_noise=rng.uniform(0.01, 0.02))
        rp = ResidualPick(st)
        t0 = ks.d.time
        while ks.d.time - t0 < 30:
            r = rp.step(est, est_yaw, width, kind)
            if rp.active or r != "running":
                break
            for _ in range(E.CTRL_EVERY):
                mujoco.mj_step(ks.m, ks.d)
        if not rp.active:
            continue
        size = mujoco.mj_stateSize(ks.m, mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.zeros(size)
        mujoco.mj_getState(ks.m, ks.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        out.append(dict(kind=kind, state=state, mass=mf, mu=info["mu"], est=np.asarray(est), est_yaw=est_yaw,
                        z0=float(true[2]), force=float(ks.m.actuator_forcerange[st.act["gripper"], 1]),
                        st={k: getattr(st, k) for k in ST_ATTRS}, rp={k: getattr(rp, k) for k in RP_ATTRS}))
    return out


class PickEnv:
    """reset() → 観測、step(行動) → (観測, 報酬, 終わりか, info)。"""

    def __init__(self, pool, seed=0):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.sims = {}

    def _sim(self, kind):
        if kind not in self.sims:
            ks = KindSim(kind)
            from sim.stretch import Stretch
            ks.st = Stretch(ks.m, ks.d)
            ks.rp = ResidualPick(ks.st)
            self.sims[kind] = ks
        return self.sims[kind]

    def reset(self, entry=None):
        e = entry if entry is not None else self.pool[self.rng.integers(len(self.pool))]
        ks = self._sim(e["kind"])
        m, d = ks.m, ks.d
        ks.set_mass(e["mass"])
        ks.set_friction(e["mu"])
        mujoco.mj_setState(m, d, e["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        for k, v in e["st"].items():
            setattr(ks.st, k, v)
        for k, v in e["rp"].items():
            setattr(ks.rp, k, v)
        ks.rp.last_a = np.zeros_like(ks.rp.last_a)
        m.actuator_forcerange[ks.st.act["gripper"]] = [-e["force"], e["force"]]
        for eq in ks.st.welds.values():
            d.eq_active[eq] = 0
        mujoco.mj_forward(m, d)
        self.ks, self.e = ks, e
        self.t0 = d.time
        self.h = 0.0
        return ks.rp.observe()

    def _height(self):
        return min(max((self.ks.obj_z() - self.e["z0"]) / 0.06, 0.0), 1.0)

    def step(self, action):
        ks, e = self.ks, self.e
        m, d = ks.m, ks.d
        r = ks.rp.step(e["est"], e["est_yaw"], E.grip_width(e["kind"]), e["kind"], action=action)
        for _ in range(E.CTRL_EVERY):
            mujoco.mj_step(m, d)
        h = self._height()
        a = np.clip(action, -1, 1)
        reward = (h - self.h) - 0.001 * float(a @ a)
        self.h = h
        info = {}
        done = r != "running" or d.time - self.t0 > T_MAX
        if not np.all(np.isfinite(d.qpos)):
            done, r = True, "unstable"
        if done:
            ok = r == "done" and ks.obj_z() - e["z0"] >= 0.05
            if ok:                                   # eval_pick と同じく 2 秒そのまま持っていられるか
                for _ in range(int(HOLD / E.TIMESTEP)):
                    ks.st.update()
                    mujoco.mj_step(m, d)
                    if ks.obj_z() - e["z0"] < 0.05:
                        ok = False
                        break
            reward += 1.0 if ok else 0.0
            info = dict(success=ok, kind=e["kind"], status=r, reason=ks.st.reason)
            return None, reward, True, info
        return ks.rp.observe(), reward, False, info


def build_pool(n_per_kind, seed, jobs):
    from multiprocessing import Pool
    os.makedirs(POOL_DIR, exist_ok=True)
    args = [(k, n_per_kind // 4, seed * 100 + i * 10 + j) for i, k in enumerate(KIND_LIST) for j in range(4)]
    with Pool(jobs) as p:
        parts = p.starmap(make_states, args)
    pool = [x for part in parts for x in part]
    path = os.path.join(POOL_DIR, f"pool_s{seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pool, f)
    print("wrote", path, len(pool))
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="種類ごとの数")
    ap.add_argument("--seed", type=int, default=1)          # eval_pick は seed 0
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    build_pool(a.n, a.seed, a.jobs)
