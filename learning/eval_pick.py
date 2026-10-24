"""Stretch 3 が床の小さい物を拾えるかの評価 (sim/stretch.py の決め打ちの拾い方)。

床と物 1 つだけの場面。物の種類ごとに N 回、次を乱数で変える:
  - 物の位置: 台車から見て腕の届く帯 (横 0.12〜0.45 m、前後 ±0.15 m)。台車は arm_side_point で
    物の横に来たつもりの位置から ±3 cm / ±5° ずれて止まっている (ナビの誤差)
  - 物の向き (0〜180°)、摩擦係数 (0.5〜1.2)
  - ロボットに渡す位置の見積もりに ±1 cm、向きの見積もりに ±10° の誤差
成功 = 物が元の高さより 5 cm 以上上がり、そのまま 2 秒落とさない。weld の補助は既定で無し。

    PYTHONUTF8=1 .venv/Scripts/python.exe learning/eval_pick.py                 # 5 種 x 100 回、補助なし/あり
    PYTHONUTF8=1 .venv/Scripts/python.exe learning/eval_pick.py --kinds coin --n 10 --assist off --no-write
    ... --set GRIP_FORCE=5 --set PRESS=0.02     # sim/stretch.py の定数を変えて試す
    ... --opt house noslip --jobs 6             # 物理の設定ごとに、並列で
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import stretch as S  # noqa: E402
from sim.objects import KINDS, add_objects, Objects, label  # noqa: E402

TIMESTEP = 0.005             # 家のシミュレーションと同じ
CTRL_EVERY = 4               # pick_step を何ステップごとに呼ぶか (家では Duck.step が 4 ステップ = 0.02 s 進めるので 4)
OUT = os.path.join(ROOT, "docs", "learning", "pick_eval.md")


def grip_width(kind):
    """指で挟む向きの幅 (物の y 軸の向き。円柱・球は直径、箱は y の長さ)。"""
    shape, size = KINDS[kind][1], KINDS[kind][2]
    return 2 * size[1] if shape == "box" else 2 * size[0]


# 物理の設定。house = 家のシミュレーションと同じ (MuJoCo の既定: Euler, pyramidal, impratio 1)。
# stretch = Stretch の MJCF の <option> (attach では捨てられる)。
OPTIONS = {
    "house": dict(),
    "stretch": dict(integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST, cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
                    impratio=10.0, noslip_iterations=3),
    "elliptic": dict(cone=mujoco.mjtCone.mjCONE_ELLIPTIC, impratio=10.0),
    "noslip": dict(noslip_iterations=3),
}


def build(kind, weld_objects=True, opt="house"):
    """床 + 物 1 つ + Stretch。timestep 0.005 (家と同じ)。opt は OPTIONS の名前。"""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.05])
    names = add_objects(spec, {kind: 1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S.add_stretch(spec, weld_objects=names if weld_objects else (), grip_force=S.GRIP_FORCE,
                      tip_level=S.TIP_LEVEL)
        m = spec.compile()
    m.opt.timestep = TIMESTEP
    for k, v in OPTIONS[opt].items():
        setattr(m.opt, k, v)
    d = mujoco.MjData(m)
    return m, d, names[0]


class Env:
    def __init__(self, kind, opt="house"):
        self.kind = kind
        self.m, self.d, self.name = build(kind, opt=opt)
        self.obj = Objects(self.m, self.d, [self.name])
        b = self.obj.body[self.name]
        self.geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]

    def reset(self, rng, assist, pos_noise=0.01, yaw_noise=math.radians(10)):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        st = S.Stretch(m, d, weld_assist=assist)
        # 物の場所 (台車の座標で決めてから、台車を少しずらして置く)
        reach = rng.uniform(0.12, 0.45)
        fwd = rng.uniform(-0.15, 0.15)
        oyaw = rng.uniform(0, math.pi)
        mu = rng.uniform(0.5, 1.2)
        for g in self.geoms:
            m.geom_friction[g, 0] = mu
        ox, oy = 0.0, 0.0
        bx, by, byaw = st.arm_side_point((ox, oy), yaw=0.0, reach=reach)
        bx += fwd
        byaw = rng.uniform(-1, 1) * math.radians(5)
        c, s = math.cos(byaw), math.sin(byaw)
        # 台車の向きを変えても、物が腕の帯の同じ所 (reach, fwd) に来るように台車を回す
        rel = np.array([ox - bx, oy - by])
        bx, by = ox - (c * rel[0] - s * rel[1]), oy - (s * rel[0] + c * rel[1])
        bx += rng.uniform(-0.03, 0.03)
        by += rng.uniform(-0.03, 0.03)
        st.reset(bx, by, byaw)
        self.obj.place(self.name, ox, oy, oyaw)
        for _ in range(int(0.5 / TIMESTEP)):         # 物と台車を落ち着かせる
            st.update()
            mujoco.mj_step(m, d)
        true = self.obj.pos(self.name)
        est = true + np.array([rng.uniform(-pos_noise, pos_noise), rng.uniform(-pos_noise, pos_noise), 0.0])
        est_yaw = oyaw + rng.uniform(-yaw_noise, yaw_noise)
        return st, true, est, est_yaw, dict(reach=reach, fwd=fwd, yaw=oyaw, mu=mu)


def trial(env, rng, assist, t_max=45.0, hold=2.0, policy=None):
    """policy: 学習した補正 (learning/rl_pick/residual.py の MLPPolicy)。None なら計算で決めた拾い方だけ。"""
    st, true, est, est_yaw, info = env.reset(rng, assist)
    m, d = env.m, env.d
    z0 = true[2]
    t0 = d.time
    status = "running"
    rp = None
    if policy is not None:
        from rl_pick.residual import ResidualPick
        rp = ResidualPick(st, policy)
    while status == "running" and d.time - t0 < t_max:
        if rp is None:
            status = st.pick_step(est, yaw=est_yaw, width=grip_width(env.kind))
        else:
            status = rp.step(est, est_yaw, grip_width(env.kind), env.kind)
        for _ in range(CTRL_EVERY):
            mujoco.mj_step(m, d)
    t_pick = d.time - t0
    lifted = env.obj.pos(env.name)[2] - z0
    ok = status == "done" and lifted >= 0.05
    if ok:                                           # 2 秒そのまま持っていられるか
        for _ in range(int(hold / TIMESTEP)):
            st.update()
            mujoco.mj_step(m, d)
            if env.obj.pos(env.name)[2] - z0 < 0.05:
                ok = False
                break
    if not np.all(np.isfinite(d.qpos)) or np.abs(d.qvel).max() > 100:
        status, ok = "unstable", False
    reason = "" if ok else (st.reason or ("落とした" if status == "done" else status))
    return dict(ok=ok, status=status, reason=reason, t=t_pick, lifted=float(lifted),
                grip=st.joints()["gripper"], **info)


def apply_overrides(overrides):
    for kv in overrides:
        k, v = kv.split("=")
        setattr(S, k, type(getattr(S, k))(float(v)))


def run_one(job):
    """(物の種類, 物理の設定, 補助, 回数, seed, 上書き, verbose) → 集計。並列に動かせるよう 1 つで完結させる。"""
    kind, opt, assist, n, seed, overrides, verbose, ctrl_every, policy_path = job
    warnings.filterwarnings("ignore")
    apply_overrides(overrides)
    global CTRL_EVERY
    CTRL_EVERY = ctrl_every
    env = Env(kind, opt)
    policy = None
    if policy_path:
        from rl_pick.residual import MLPPolicy
        policy = MLPPolicy(policy_path)
    rng = np.random.default_rng(seed)
    res = []
    t = time.time()
    for i in range(n):
        r = trial(env, rng, assist, policy=policy)
        res.append(r)
        if verbose:
            print(f"  {kind} #{i:3d} {'OK ' if r['ok'] else 'NG '} {r['reason']:24s} reach {r['reach']:.2f} fwd {r['fwd']:+.2f} "
                  f"yaw {math.degrees(r['yaw']):5.0f} mu {r['mu']:.2f} lifted {r['lifted']:+.3f} grip {r['grip']:+.4f} t {r['t']:.1f}",
                  flush=True)
    ok = int(sum(r["ok"] for r in res))
    fails = {}
    for r in res:
        if not r["ok"]:
            fails[r["reason"]] = fails.get(r["reason"], 0) + 1
    t_ok = [r["t"] for r in res if r["ok"]]
    row = dict(ok=ok, n=n, fails=fails, t=float(np.mean(t_ok)) if t_ok else float("nan"))
    print(f"{kind:15s} opt={opt:8s} assist={'on ' if assist else 'off'} {'rl ' if policy_path else 'calc'} {ok:3d}/{n}  "
          f"({time.time() - t:.0f} s)  {fails}", flush=True)
    return kind, opt, assist, policy_path, row


def column(opt, assist, policy_path=None):
    return f"{OPT_LABEL[opt]}・weld 補助{'あり' if assist else 'なし'}" + ("・学習した補正" if policy_path else "")


OPT_LABEL = {"house": "家の既定", "noslip": "noslip 3", "stretch": "Stretch の option", "elliptic": "elliptic"}


def write_md(results, n, seed, overrides, ctrl_every=CTRL_EVERY, out=OUT,
             title="Stretch 3 で床の小さい物を拾う — 決め打ちの拾い方の成功率", extra=()):
    cols = list(results)
    kinds = list(next(iter(results.values())))
    lines = [f"# {title}", "", *extra,
             f"`learning/eval_pick.py` で作成 (種類ごとに {n} 回、seed {seed}、timestep {TIMESTEP}、"
             f"制御の周期 {ctrl_every * TIMESTEP:.3f} s)。"
             " 考察と注意は [pick_notes.md](pick_notes.md)。", "",
             "成功 = 物が元の高さから 5 cm 以上上がり、そのまま 2 秒落とさない。乱数で変えたもの: 物の位置 (腕の帯 横 0.12〜0.45 m・前後 ±0.15 m)、"
             "向き、摩擦 0.5〜1.2、台車の止まる位置 ±3 cm・±5°、ロボットに渡す位置の見積もり ±1 cm・向き ±10°。", "",
             "物理の設定: **家の既定** = 今の家のシミュレーション (MuJoCo の既定: Euler・pyramidal・impratio 1・noslip なし)。"
             "**noslip 3** = それに `noslip_iterations=3` だけ足したもの (Microduck の歩き方は変わらないことを確かめた)。", ""]
    if overrides:
        lines += [f"定数の上書き: `{' '.join(overrides)}`", ""]
    head = "| 物 | 大きさ | " + " | ".join(cols) + " |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for k in kinds:
        size, shape = KINDS[k][2], KINDS[k][1]
        if shape == "cylinder":
            dims = f"直径 {size[0] * 2000:.1f} mm・厚さ {size[1] * 2000:.1f} mm"
        elif shape == "sphere":
            dims = f"直径 {size[0] * 2000:.0f} mm"
        else:
            dims = " x ".join(f"{v * 2000:.0f}" for v in size) + " mm"
        cells = [f"{results[c][k]['ok']}/{results[c][k]['n']} ({100 * results[c][k]['ok'] / results[c][k]['n']:.0f}%)" for c in cols]
        lines.append(f"| {label(k)} (`{k}`) | {dims} | " + " | ".join(cells) + " |")
    lines += ["", "## 失敗の内訳と、成功したときの平均時間", ""]
    for c in cols:
        lines.append(f"- **{c}**")
        for k in kinds:
            r = results[c][k]
            fails = ", ".join(f"{why} {cnt}" for why, cnt in sorted(r["fails"].items(), key=lambda x: -x[1])) or "なし"
            t = f"{r['t']:.1f} s" if r["ok"] else "—"
            lines.append(f"  - {label(k)}: 平均 {t}、失敗: {fails}")
    lines.append("")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assist", choices=["off", "on", "both"], default="both")
    ap.add_argument("--opt", nargs="*", choices=list(OPTIONS), default=["house", "noslip"], help="物理の設定")
    ap.add_argument("--set", action="append", default=[], help="sim/stretch.py の定数を上書き NAME=VALUE")
    ap.add_argument("--jobs", type=int, default=1, help="並列のプロセス数")
    ap.add_argument("--ctrl-every", type=int, default=CTRL_EVERY, help="pick_step を何ステップごとに呼ぶか (4 = 0.02 s、家と同じ)")
    ap.add_argument("--controller", choices=["calc", "rl", "both"], default="calc",
                    help="calc = 計算で決めた拾い方、rl = それに学習した補正を足したもの (learning/rl_pick)")
    ap.add_argument("--policy", default=os.path.join(ROOT, "learning", "rl_pick", "policy.npz"))
    ap.add_argument("--out", default=OUT, help="書き出す表 (.md。同じ名前の .json も)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    apply_overrides(a.set)
    modes = {"off": [False], "on": [True], "both": [False, True]}[a.assist]
    pols = {"calc": [None], "rl": [a.policy], "both": [None, a.policy]}[a.controller]
    jobs = [(k, o, m, a.n, a.seed, a.set, a.verbose, a.ctrl_every, p)
            for o in a.opt for m in modes for p in pols for k in a.kinds]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            out = pool.map(run_one, jobs)
    else:
        out = [run_one(j) for j in jobs]
    results = {}
    for o in a.opt:
        for m in modes:
            for p in pols:
                results[column(o, m, p)] = {k: row for k, oo, mm, pp, row in out if oo == o and mm == m and pp == p}
    if not a.no_write:
        with open(a.out.replace(".md", ".json"), "w", encoding="utf-8") as f:
            json.dump(dict(n=a.n, seed=a.seed, overrides=a.set, ctrl_every=a.ctrl_every, controller=a.controller,
                           policy=os.path.relpath(a.policy, ROOT) if a.controller != "calc" else None, results=results),
                      f, ensure_ascii=False, indent=1)
        extra = ()
        if a.controller != "calc":
            extra = (f"学習した補正: `{os.path.relpath(a.policy, ROOT)}` (learning/rl_pick、詳しくは "
                     "[pick_rl.md](pick_rl.md))。", "")
        write_md(results, a.n, a.seed, a.set, a.ctrl_every, out=a.out,
                 title="Stretch 3 で床の小さい物を拾う — " + ("計算で決めた拾い方と学習した補正の比較" if a.controller != "calc"
                                                         else "決め打ちの拾い方の成功率"), extra=extra)


if __name__ == "__main__":
    main()
