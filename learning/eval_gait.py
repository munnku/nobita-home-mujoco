"""歩き方の方策 (ONNX) が速度指令にどれだけついてくるかを、この家のシミュレーションで測る。

居間の広い所で、前進 (vx) と旋回 (wz) の指令を 6 秒ずつ与え、実際の速さ・回る速さ・転倒を記録する。
uv run python learning/eval_gait.py --policy velstand [--policy third_party/policies/my.onnx ...]
出力: 標準出力と build/learning/gait_<方策>.json
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from world import World  # noqa: E402

M = 0.91
CMDS = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.0, 0.5), (0.0, 1.0), (0.2, 0.5)]
HOLD = 6.0


def run(policy):
    out = []
    for vx, wz in CMDS:
        w = World(duck_pose=(5.0 * M, 3.0 * M, -math.pi / 2 if vx > 0 else 0.0), n_persons=0, policy=policy)
        for _ in range(50):
            w.duck.step()
        x0, y0, th0 = w.duck.pose()
        prev, turned, fell = th0, 0.0, False
        w.duck.set_cmd(vx, 0.0, wz)
        steps = int(HOLD / 0.02)
        for _ in range(steps):
            w.duck.step()
            _, _, th = w.duck.pose()
            turned += (th - prev + math.pi) % (2 * math.pi) - math.pi
            prev = th
            if w.duck.fallen():
                fell = True
                break
        x1, y1, _ = w.duck.pose()
        r = {"cmd_vx": vx, "cmd_wz": wz, "vx": round(math.hypot(x1 - x0, y1 - y0) / HOLD, 3),
             "wz": round(turned / HOLD, 3), "fell": fell}
        out.append(r)
        print(f"  cmd vx={vx:.1f} wz={wz:.1f} → 実際 {r['vx']:.3f} m/s, {r['wz']:.2f} rad/s {'転倒' if fell else ''}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "build", "learning"), exist_ok=True)
    for p in a.policy or ["velstand", "alpha_walking"]:
        print(f"== {p}")
        res = run(p)
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(ROOT, "build", "learning", f"gait_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
