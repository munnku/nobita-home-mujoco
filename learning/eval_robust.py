"""散らかった居間 (敷物・座布団・おもちゃ) と押しに、歩き方の方策がどれだけ耐えるかを測る。

居間 (ima) に敷物・座布団・おもちゃをこのスクリプトの中だけで (MjSpec に) 置き、
居間を横切る 4 本の道を pure pursuit (sim/navigator.py) で歩かせる。条件は 4 つ:
  flat (何も置かない) / clutter (散らかした居間) / flat+push / clutter+push
push: 3〜5 秒ごとに胴体を横から 0.1 s 押す (xfrc_applied、速さが PUSH_DV 変わるくらいの力積)。
倒れたら (高さ 7 cm 未満か 60° 以上傾く) 指令を 0 にして 10 秒待ち、立ち直れば続きを歩く。

uv run python learning/eval_robust.py [--policy velstand --policy models/duck_walk.onnx ...] [--seeds 3]
                                      [--recovery own|velstand]
--recovery own (既定): 方策だけで測る (sim/duck.py の「倒れたら velstand」を止める)。velstand: アプリと同じ
出力: 標準出力の表、build/learning/robust_eval[_velstand].json、docs/learning/robust_eval[_velstand].md
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import world as world_mod  # noqa: E402
from navigator import REACH_GOAL, Navigator  # noqa: E402

M = 0.91
DEFAULT_POLICIES = ["velstand", "models/duck_walk.onnx", "models/duck_robust.onnx", "models/duck_robust2.onnx"]
# 前進の指令の上限と下限 (sim/world.py の POLICY_SPEED と同じ。duck_robust* は duck_walk から温めたので同じ)
SPEED = {"velstand": (0.3, 0.0), "duck_walk": (0.4, 0.32), "duck_robust": (0.4, 0.32),
         "duck_robust2": (0.4, 0.32)}
# 居間を横切る道 (半間のマス目)。ちゃぶ台 (6.45〜7.45, 1.7〜2.7) と床の間・押入れ (x < 5) を避ける
COURSES = {
    "south_E": [(5.3, 0.75), (8.0, 0.75)],
    "north_W": [(8.0, 3.35), (5.3, 3.35)],
    "west_N": [(5.6, 0.6), (5.6, 3.4)],
    "loop": [(8.0, 1.2), (7.9, 3.2), (6.0, 3.1), (5.9, 1.3)],
}
CONDITIONS = ["flat", "clutter", "flat+push", "clutter+push"]
TIME_LIMIT = 70.0       # 1 本の道の制限時間 [s] (velstand は 0.1 m/s ほどなので長め)
GETUP_WAIT = 10.0       # 倒れてから立ち直るのを待つ時間 [s]
PUSH_DV = 0.35          # 押しで変わる胴体の速さ [m/s] (学習の push は ±0.3 m/s)
PUSH_DUR = 0.1
PUSH_EVERY = (3.0, 5.0)

# 座布団の柔らかさ: learning/rl/home_clutter_env_cfg.py の version 2 と同じ (約 2 cm 沈む)。
# 柔らかいと薄い箱は突き抜けるので、箱を床の下 ZABUTON_DEPTH まで伸ばす
ZABUTON_SOLREF = (0.35, 1.5)
ZABUTON_SOLIMP = (0.3, 0.95, 0.06, 0.5, 2.0)
ZABUTON_DEPTH = 0.15


def _box(body, name, pos, size, yaw=0.0, rgba=(0.7, 0.3, 0.3, 1), **kw):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                  quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], rgba=rgba, **kw)


def add_clutter(spec, seed=7):
    """居間に敷物 2 枚 (+重ね 1 枚・めくれ)、座布団 3 枚、おもちゃ 28 個を置く。位置は seed で決まる。"""
    wb = spec.worldbody
    u = lambda v: v * M  # noqa: E731
    # 敷物 (厚さ 10 mm / 6 mm + 重ねた 8 mm)。端のめくれ (+12 mm の帯)
    _box(wb, "rug_a", [u(6.6), u(0.8), 0.005], [0.80, 0.40, 0.005], yaw=0.05, rgba=(0.72, 0.25, 0.25, 1))
    _box(wb, "rug_a_fold", [u(6.6) + 0.78, u(0.8) + 0.04, 0.011], [0.04, 0.40, 0.011], yaw=0.05,
         rgba=(0.62, 0.2, 0.2, 1))
    _box(wb, "rug_b", [u(5.6), u(2.2), 0.003], [0.45, 0.65, 0.003], yaw=-0.1, rgba=(0.85, 0.75, 0.45, 1))
    _box(wb, "rug_b2", [u(5.75), u(2.7), 0.010], [0.35, 0.30, 0.004], yaw=0.4, rgba=(0.3, 0.55, 0.35, 1))
    # 座布団 (下段 = 縁、上段 = ふくらみ)。柔らかい接触
    soft = dict(solref=ZABUTON_SOLREF, solimp=ZABUTON_SOLIMP, rgba=(0.25, 0.35, 0.65, 1))
    for i, (x, y, h, yaw) in enumerate([(6.2, 3.35, 0.045, 0.2), (7.55, 3.3, 0.04, -0.3), (5.6, 1.35, 0.035, 0.7)]):
        for part, top, half_xy in (("base", h / 2, (0.27, 0.29)), ("top", h, (0.22, 0.24))):
            hz = (top + ZABUTON_DEPTH) / 2
            _box(wb, f"zabuton{i}_{part}", [u(x), u(y), top - hz], [*half_xy, hz], yaw=yaw, **soft)
    # おもちゃ: 積み木 (箱)・立った円柱・寝た円柱。道の近くに散らす
    rng = np.random.default_rng(seed)
    starts = [c[0] for c in COURSES.values()]
    lanes = [p for c in COURSES.values() for p in c]
    n = 0
    while n < 28:
        a, b = lanes[rng.integers(len(lanes))], lanes[rng.integers(len(lanes))]
        k = rng.uniform()
        x = u(a[0] + (b[0] - a[0]) * k) + rng.normal(0, 0.12)
        y = u(a[1] + (b[1] - a[1]) * k) + rng.normal(0, 0.12)
        if not (u(5.05) < x < u(8.25) and u(0.1) < y < u(3.85)):
            continue
        if u(6.4) < x < u(7.5) and u(1.65) < y < u(2.75):    # ちゃぶ台の下
            continue
        if min(math.hypot(x - u(s[0]), y - u(s[1])) for s in starts) < 0.3:
            continue
        h = rng.uniform(0.010, 0.025)
        yaw = rng.uniform(0, 2 * math.pi)
        kind = rng.uniform()
        if kind < 0.45:
            _box(wb, f"toy{n}", [x, y, h / 2], [rng.uniform(0.01, 0.035), rng.uniform(0.01, 0.035), h / 2],
                 yaw=yaw, rgba=(0.95, 0.75, 0.1, 1))
        elif kind < 0.75:
            wb.add_geom(name=f"toy{n}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, h / 2],
                        size=[rng.uniform(0.01, 0.03), h / 2, 0], rgba=(0.2, 0.7, 0.3, 1))
        else:
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
                               np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]))
            wb.add_geom(name=f"toy{n}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, h / 2], quat=q,
                        size=[h / 2, rng.uniform(0.03, 0.07), 0], rgba=(0.2, 0.4, 0.9, 1))
        n += 1
    return spec


def make_world(policy, clutter, recovery="own"):
    """家を作るときだけ compose を差し替えて、居間を散らかす (houses/ と sim/ は変えない)。
    recovery="own": 方策だけで歩き・起き上がる (sim/duck.py の velstand への切り替えを止める)。
    recovery="velstand": アプリと同じく、倒れたら velstand で起き上がる。"""
    orig = world_mod.compose

    def compose_with_clutter(*a, **kw):
        spec, door_open = orig(*a, **kw)
        return add_clutter(spec), door_open

    if clutter:
        world_mod.compose = compose_with_clutter
    try:
        if policy.endswith(".onnx") and not os.path.isabs(policy):
            policy = os.path.join(ROOT, policy)
        w = world_mod.World(duck_pose=(5.0 * M, 3.0 * M, 0.0), n_persons=0, policy=policy)
        if recovery == "own" and hasattr(w.duck, "recovery"):
            w.duck.recovery = None
        return w
    finally:
        world_mod.compose = orig


def duck_mass(w):
    m = w.model
    root = m.body_rootid[w.duck.trunk]
    return float(sum(m.body_mass[b] for b in range(m.nbody) if m.body_rootid[b] == root))


def is_fallen(w):
    d, qa = w.data, w.duck.qa
    _, qx, qy, _ = d.qpos[qa + 3:qa + 7]
    up_z = 1 - 2 * (qx * qx + qy * qy)        # 胴体の上向きの z 成分
    return d.qpos[qa + 2] < 0.07 or up_z < 0.5


def is_upright(w):
    d, qa = w.data, w.duck.qa
    _, qx, qy, _ = d.qpos[qa + 3:qa + 7]
    return d.qpos[qa + 2] > 0.09 and 1 - 2 * (qx * qx + qy * qy) > 0.85


def run_course(w, pts, push, rng, speed, mass):
    """1 本の道を歩く。結果: reached, falls, getups, time, pushes"""
    path = [(x * M, y * M) for x, y in pts]
    (x0, y0), (x1, y1) = path[0], path[1]
    mujoco.mj_resetData(w.model, w.data)
    w.duck.reset(x0, y0, math.atan2(y1 - y0, x1 - x0) + rng.normal(0, 0.05))  # 扉は w.step() が戻す
    w.nav = Navigator(*speed)
    w.stop()
    for _ in range(50):      # 立ち姿勢になじませる
        w.step()
    w.nav = Navigator(*speed)
    w.nav.set_path(path)
    goal = path[-1]
    t, falls, getups, pushes = 0.0, 0, 0, 0
    next_push = rng.uniform(*PUSH_EVERY)
    push_end, force = -1.0, np.zeros(3)
    trunk = w.duck.trunk
    reached = False
    fall_at = []
    end = "timeout"
    while t < TIME_LIMIT:
        if push and t >= next_push:
            ang = rng.uniform(0, 2 * math.pi)
            force = mass * PUSH_DV / PUSH_DUR * np.array([math.cos(ang), math.sin(ang), 0.0])
            push_end, next_push = t + PUSH_DUR, t + rng.uniform(*PUSH_EVERY)
            pushes += 1
        w.data.xfrc_applied[trunk, :3] = force if t < push_end else 0.0
        w.step()
        t += 0.02
        x, y, _ = w.duck.pose()
        if math.hypot(goal[0] - x, goal[1] - y) < REACH_GOAL + 0.03:
            reached, end = True, "reached"
            break
        if w.nav.done and not w.nav.arrived:     # stuck
            end = "stuck"
            break
        if is_fallen(w):
            falls += 1
            fall_at.append((round(x, 2), round(y, 2), nearest_obstacle(w, x, y)))
            w.nav.stop()
            w.set_manual_cmd(0.0, 0.0, 0.0)
            w.data.xfrc_applied[trunk, :3] = 0.0
            ok_since, t_fall = None, t
            while t - t_fall < GETUP_WAIT:
                w.step()
                t += 0.02
                if is_upright(w):
                    ok_since = ok_since if ok_since is not None else t
                    if t - ok_since > 1.0:
                        break
                else:
                    ok_since = None
            if ok_since is None or t - ok_since <= 1.0:
                end = "no_getup"                  # 立ち直れなかった
                break
            getups += 1
            # 続き: いまの位置から、まだ通っていない点を通ってゴールへ
            x, y, _ = w.duck.pose()
            k = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y))
            rest = path[min(k + 1, len(path) - 1):]
            w.nav = Navigator(*speed)
            w.nav.set_path([(x, y)] + rest)
            next_push = t + rng.uniform(*PUSH_EVERY)
    w.data.xfrc_applied[trunk, :3] = 0.0
    return {"reached": reached, "end": end, "falls": falls, "getups": getups, "time": round(t, 1), "pushes": pushes,
            "fall_at": fall_at}


def nearest_obstacle(w, x, y, within=0.25):
    """倒れた場所に一番近い、置いた物 (rug / zabuton / toy) の名前。なければ "-"。"""
    best, name = within, "-"
    for g in range(w.model.ngeom):
        n = mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.startswith(("rug", "zabuton", "toy")):
            gx, gy = w.data.geom_xpos[g][:2]
            r = math.hypot(*w.model.geom_size[g][:2]) if n.startswith(("rug", "zabuton")) else 0.0
            d = max(0.0, math.hypot(gx - x, gy - y) - r)
            if d < best:
                best, name = d, n.split("_")[0] if n.startswith("zabuton") else n
    return name


def snapshot(w, path):
    """居間を斜め上から撮る (レポート用)。"""
    r = mujoco.Renderer(w.model, 480, 800)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [6.6 * M, 1.9 * M, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 4.2, 90, -78   # ほぼ真上から、北 (+y) が上・東が右
    mujoco.mj_forward(w.model, w.data)
    r.update_scene(w.data, cam)
    img = r.render()
    from PIL import Image
    Image.fromarray(img).save(path)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--recovery", choices=["own", "velstand"], default="own",
                    help="own: 方策だけ / velstand: 倒れたら velstand で起き上がる (アプリと同じ)")
    a = ap.parse_args()
    conds = a.conditions.split(",")
    policies = a.policy or DEFAULT_POLICIES
    out_dir = os.path.join(ROOT, "build", "learning")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for pol in policies:
        key = os.path.splitext(os.path.basename(pol))[0]
        speed = SPEED.get(key, (0.3, 0.0))
        worlds = {}
        for cond in conds:
            clutter = cond.startswith("clutter")
            if clutter not in worlds:
                worlds[clutter] = make_world(pol, clutter, a.recovery)
                if clutter and key == os.path.splitext(os.path.basename(policies[0]))[0]:
                    try:
                        snapshot(worlds[clutter], os.path.join(ROOT, "docs", "learning", "robust_scene.png"))
                    except Exception as e:  # 描けなくても測定は続ける
                        print("snapshot skipped:", e)
            w = worlds[clutter]
            mass = duck_mass(w)
            runs = []
            t0 = time.time()
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 + seed)
                for cname, pts in COURSES.items():
                    r = run_course(w, pts, cond.endswith("push"), rng, speed, mass)
                    r.update(course=cname, seed=seed)
                    runs.append(r)
            n = len(runs)
            falls = sum(r["falls"] for r in runs)
            getups = sum(r["getups"] for r in runs)
            summ = {"runs": n, "reach": sum(r["reached"] for r in runs) / n, "falls": falls,
                    "runs_with_fall": sum(r["falls"] > 0 for r in runs), "getups": getups,
                    "getup_rate": (getups / falls) if falls else None,
                    "mean_time_reached": round(float(np.mean([r["time"] for r in runs if r["reached"]] or [float("nan")])), 1),
                    "pushes": sum(r["pushes"] for r in runs),
                    "stuck": sum(r["end"] == "stuck" for r in runs),
                    "no_getup": sum(r["end"] == "no_getup" for r in runs)}
            results.setdefault(key, {})[cond] = {"summary": summ, "runs": runs}
            gr = "-" if summ["getup_rate"] is None else f"{getups}/{falls}"
            print(f"{key:14s} {cond:13s} reach {summ['reach']*100:5.1f}%  falls {falls:3d}  "
                  f"get-up {gr:6s}  t={summ['mean_time_reached']}s  ({time.time()-t0:.0f}s wall)", flush=True)
    suffix = "" if a.recovery == "own" else "_" + a.recovery
    with open(os.path.join(out_dir, f"robust_eval{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    # 表 (Markdown)
    lines = ["| 方策 | 条件 | 到着率 | 転倒 (回) | 転倒した道 | 起き上がり | 止まった道 (stuck / 起きられず) | 到着までの平均 [s] |",
             "|---|---|---|---|---|---|---|---|"]
    for key, rc in results.items():
        for cond, r in rc.items():
            s = r["summary"]
            gr = "-" if s["getup_rate"] is None else f"{s['getups']}/{s['falls']}"
            lines.append(f"| {key} | {cond} | {s['reach']*100:.0f}% ({round(s['reach']*s['runs'])}/{s['runs']}) | "
                         f"{s['falls']} | {s['runs_with_fall']}/{s['runs']} | {gr} | {s['stuck']} / {s['no_getup']} | "
                         f"{s['mean_time_reached']} |")
    table = "\n".join(lines)
    print("\n" + table)
    rec = "方策だけ (velstand への切り替えなし)" if a.recovery == "own" else "倒れたら velstand で起き上がる (sim/duck.py、アプリと同じ)"
    with open(os.path.join(ROOT, "docs", "learning", f"robust_eval{suffix}.md"), "w", encoding="utf-8") as f:
        f.write("# 散らかった居間と押しへの強さ (learning/eval_robust.py)\n\n"
                f"道 {len(COURSES)} 本 × seed {a.seeds}。push: {PUSH_EVERY[0]:.0f}〜{PUSH_EVERY[1]:.0f} 秒ごとに "
                f"Δv={PUSH_DV} m/s (0.1 s)。倒れたら {GETUP_WAIT:.0f} 秒待って立ち直れば続ける。起き上がり: {rec}。\n\n"
                + table + "\n")


if __name__ == "__main__":
    main()
