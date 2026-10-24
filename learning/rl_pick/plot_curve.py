"""学習曲線を docs/learning/pick_rl_curve.png に描く。

左: 学習中の成功率 (行動に探索の乱数を足し、位置の見積もりの誤差 ±1〜2 cm・質量 ×0.7〜1.3 の場面。評価とは条件が違う)。
右: 学習に使わない場面 600 個 (build/rl_pick/pool_s2.pkl) での、乱数を足さない方策の成功率 (2 回目の学習だけ)。
    0 回目 (補正ほぼ 0) が計算で決めた拾い方の成功率。

    PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/plot_curve.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = {"run 1 (std 0.37)": os.path.join(ROOT, "build", "rl_pick", "run1_log.json"),
        "run 2 (std 0.14)": os.path.join(ROOT, "build", "rl_pick", "run2_log.json")}
OUT = os.path.join(ROOT, "docs", "learning", "pick_rl_curve.png")
KINDS = ["button_battery", "coin", "marble", "toy_block", "scissors"]


def smooth(x, k=20):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = w.mean()
    return out


def main():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    run2 = None
    for name, path in LOGS.items():
        if not os.path.exists(path):
            continue
        rows = json.load(open(path, encoding="utf-8"))
        steps = np.array([r["steps"] for r in rows]) / 1e6
        ax[0].plot(steps, smooth([r["success"] if r["success"] is not None else np.nan for r in rows]), label=name)
        if "run 2" in name:
            run2 = rows
    ax[0].set(xlabel="steps [M]", ylabel="success rate", ylim=(0, 1),
              title="training (with exploration noise, harder randomization)")
    ax[0].legend(fontsize=8)
    if run2:
        ev = [r for r in run2 if "det_success" in r]
        steps = np.array([r["steps"] for r in ev]) / 1e6
        for k, c in zip(KINDS, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
            y = [r["det_by_kind"].get(k, np.nan) for r in ev]
            ax[1].plot(steps, y, color=c, label=k)
            ax[1].axhline(y[0], color=c, ls=":", lw=1)
        ax[1].plot(steps, [r["det_success"] for r in ev], color="k", lw=2, label="all")
        ax[1].axhline(ev[0]["det_success"], color="k", ls=":", lw=1)
        best = max(ev, key=lambda r: r["det_success"])
        ax[1].axvline(best["steps"] / 1e6, color="gray", ls="--", lw=1)
    ax[1].set(xlabel="steps [M]", ylabel="success rate", ylim=(0, 1),
              title="held-out 600 states, no noise (run 2; dotted = scripted)")
    ax[1].legend(fontsize=7, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
