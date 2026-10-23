"""家の床のタスクの学習曲線を tensorboard のログから取り出して、JSON と PNG にする。

コンテナ pai_mujoco の中で (tensorboard と matplotlib がある microduck_rl の venv で) 実行する:
  ~/ext/microduck_rl/.venv/bin/python /home/ubuntu/pai_ws/learning/rl/plot_curve.py [run_dir ...]
既定は最新の *_nobita_robust と、比べるための 2026-10-19_01-45-26_nobita (平らな床で学習した歩き)。
出力: docs/learning/rl_<1 つ目の run の nobita_ より後>_curve.{json,png} (rl_robust_curve、rl_robust2_curve)
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.expanduser("~/ext/microduck_rl/logs/rsl_rl/velocity")
TAGS = {
    "reward": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
    "fell": "Episode_Termination/fell_over",
    "terrain": "Curriculum/terrain_levels/mean",
    "rug": "Curriculum/terrain_levels/rug",
    "zabuton": "Curriculum/terrain_levels/zabuton",
    "clutter": "Curriculum/terrain_levels/clutter",
    "grid": "Curriculum/terrain_levels/grid",
    "flat": "Curriculum/terrain_levels/flat",
    "err_xy": "Metrics/twist/error_vel_xy",
}


def load(run_dir):
    ev = EventAccumulator(glob.glob(os.path.join(run_dir, "events*"))[0], size_guidance={"scalars": 0})
    ev.Reload()
    have = set(ev.Tags()["scalars"])
    rows = {}
    for key, tag in TAGS.items():
        if tag in have:
            for s in ev.Scalars(tag):
                rows.setdefault(s.step, {"it": s.step})[key] = round(s.value, 4)
    return [rows[k] for k in sorted(rows)]


def main():
    runs = sys.argv[1:] or [sorted(glob.glob(os.path.join(LOGS, "*_nobita_robust")))[-1],
                            os.path.join(LOGS, "2026-10-19_01-45-26_nobita")]
    data = {os.path.basename(r): load(r) for r in runs}
    out = os.path.join(ROOT, "docs", "learning")
    stem = "rl_" + os.path.basename(runs[0]).split("nobita_")[-1] + "_curve"   # rl_robust_curve, rl_robust2_curve
    with open(os.path.join(out, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    robust = data[os.path.basename(runs[0])]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, rows in data.items():
        it = [r["it"] for r in rows]
        ax[0].plot(it, [r.get("reward") for r in rows], label=name)
        ax[1].plot(it, [r.get("fell") for r in rows], label=name)
    ax[0].set_title("mean reward")
    ax[1].set_title("fell_over terminations / iter")
    it = [r["it"] for r in robust]
    for key in ("flat", "rug", "zabuton", "clutter", "grid"):
        ax[2].plot(it, [r.get(key) for r in robust], label=key)
    ax[2].set_title("terrain level (0-9) by floor type")
    for a in ax:
        a.set_xlabel("iteration")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, stem + ".png"), dpi=110)
    last = robust[-1]
    print("last:", last)


if __name__ == "__main__":
    main()
