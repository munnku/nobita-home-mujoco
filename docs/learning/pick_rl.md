# Stretch 3 の拾う動きを強化学習で直せるか (2026-10-24)

結論: **学習した補正は、計算で決めた拾い方を上回らなかった。** 電池とビー玉は少し良くなった (+5・+4 点、100 回の差としては誤差の範囲) が、硬貨は悪くなった (−13 点)。家の場面は今のまま (計算で決めた拾い方 + weld の補助) を使うのがよい。

表 (100 回ずつ、[pick_rl_eval.md](pick_rl_eval.md)。条件は [pick_eval.md](pick_eval.md) と同じ: seed 0、noslip 3、制御 0.02 s、位置の見積もり ±1 cm):

| 物 | 計算 | 計算 + 学習した補正 | 計算・weld 補助 | 計算 + 学習・weld 補助 |
|---|---|---|---|---|
| ボタン電池 | 77 | 82 | 96 | 97 |
| 硬貨 | 55 | **42** | 95 | **87** |
| ビー玉 | 92 | 96 | 100 | 100 |
| 小さいブロック | 75 | 75 | 99 | 99 |
| ハサミ | 0 | 0 | 69 | 66 |
| 計 (500 回中) | 299 | 295 | 459 | 449 |

![学習曲線](pick_rl_curve.png)

## 何を学習したか

- **どこで**: CPU の MuJoCo (評価と同じ物理: Euler・noslip 3・timestep 0.005)。mjlab (MuJoCo Warp) は `noslip_iterations` を使うと `NotImplementedError` (コンテナの mujoco_warp で確認) で、noslip なしでは計算で決めた拾い方でもほぼ 0% なので、学習と評価の物理が合わず使わなかった。stable-baselines3 は `.venv` に無く、入れると共有の pyproject が変わるので、PPO は torch で書いた (`learning/rl_pick/train.py`、環境は numpy だけのプロセス 8 本)。
- **形**: 計算で決めた拾い方への補正 (残差)。台車の位置合わせ・下ろすまでは今のまま。指先が床に触れて閉じ始めてから、閉じる・挟む・持ち上げる・腕を縮めるまでの間、0.02 s ごとに lift・arm・wrist_roll・gripper の目標の補正と、指の力の上限の倍率を出す (`learning/rl_pick/residual.py`)。補正 0 なら今の拾い方と同じ。
- **観測 (28)**: 物の推定位置と掴む点の差、lift・arm・roll の動いた量、指の開き、掴む点の高さ、左右の指が物に触れているか・指先が何かに触れているか、段階 (閉じる/挟む/上げる/縮める) と経過時間、物の種類、幅、指の力、前の行動。
- **乱数**: 種類 5 つ・位置・向き・摩擦 0.5〜1.2・質量 ×0.7〜1.3・位置の見積もりの誤差 ±1〜2 cm・台車の止まる位置。閉じ始める所の状態を 6000 個先に作り (`learning/rl_pick/env.py`、seed 1)、そこから始める。
- **報酬**: 物の持ち上がった高さ (6 cm まで) の増えた分 + 成功 (5 cm 以上上げて 2 秒保つ、評価と同じ条件) で 1。
- **2 回学習した**: 1 回目は探索の乱数 std 0.37 で 820 万ステップ (37 分)。乱数ありの成功率は 0.35 → 0.5 に上がったが、乱数なしで試すと今の拾い方と同じ (0.44)。2 回目は std 0.14 で 490 万ステップ (25 分)、学習に使わない場面 600 個 (seed 2) で乱数なしの成功率を 25 回ごとに測り、一番良いもの (290 万ステップ、0.468。今の拾い方は 0.438) を残した。その後は硬貨と電池が下がった。表はこの方策 (`learning/rl_pick/policy.npz`)。

## なぜ良くならなかったか

失敗のほとんどは「持ち上げる途中で落とした」で、数 g の物を柔らかい接触で挟む物理の限界 (指先のパッドの向き、力を強めると物が押し出される) による ([pick_notes.md](pick_notes.md))。指の目標や力を少し変える補正では越えられず、ある種類で効く補正が別の種類を悪くした (方策は種類を観測に持っているが、分けて覚えきれなかった)。ハサミは補助なしでは学習でも 0。

## 使うかどうか

使わないことを勧める。家の場面は今のまま (計算で決めた拾い方 + weld 補助)。使うなら `ResidualPick(stretch, MLPPolicy("learning/rl_pick/policy.npz")).step(target, yaw, width, kind)` を `pick_step` の代わりに呼べば動く (numpy だけ)。ただし硬貨が補助ありでも 95 → 87% に下がる。

## 再現

```
PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/env.py --n 1200 --seed 1        # 学習の場面 6000 個 (約 5 分)
PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/env.py --n 120 --seed 2         # 学習に使わない場面 600 個
PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/train.py --iters 600 --log-std -2.0 --log build/rl_pick/run2_log.json
PYTHONUTF8=1 .venv/Scripts/python.exe learning/eval_pick.py --opt noslip --assist both --controller both --jobs 6 --out docs/learning/pick_rl_eval.md
PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/plot_curve.py
```
学習の記録と場面は `build/rl_pick/` (git に入らない)。
