# 散らかった居間と押しへの強さ (learning/eval_robust.py)

道 4 本 × seed 3。push: 3〜5 秒ごとに Δv=0.35 m/s (0.1 s)。倒れたら 10 秒待って立ち直れば続ける。起き上がり: 倒れたら velstand で起き上がる (sim/duck.py、アプリと同じ)。

| 方策 | 条件 | 到着率 | 転倒 (回) | 転倒した道 | 起き上がり | 止まった道 (stuck / 起きられず) | 到着までの平均 [s] |
|---|---|---|---|---|---|---|---|
| velstand | flat | 100% (12/12) | 0 | 0/12 | - | 0 / 0 | 32.5 |
| velstand | clutter | 17% (2/12) | 35 | 9/12 | 35/35 | 6 / 0 | 49.4 |
| velstand | flat+push | 100% (12/12) | 6 | 6/12 | 6/6 | 0 / 0 | 33.2 |
| velstand | clutter+push | 83% (10/12) | 35 | 10/12 | 34/35 | 1 / 1 | 41.6 |
| nobita_walk | flat | 100% (12/12) | 0 | 0/12 | - | 0 / 0 | 13.5 |
| nobita_walk | clutter | 67% (8/12) | 44 | 11/12 | 43/44 | 2 / 1 | 30.0 |
| nobita_walk | flat+push | 100% (12/12) | 7 | 6/12 | 7/7 | 0 / 0 | 15.2 |
| nobita_walk | clutter+push | 67% (8/12) | 44 | 11/12 | 42/44 | 0 / 2 | 30.2 |
| nobita_robust | flat | 100% (12/12) | 0 | 0/12 | - | 0 / 0 | 15.5 |
| nobita_robust | clutter | 33% (4/12) | 29 | 11/12 | 29/29 | 8 / 0 | 32.8 |
| nobita_robust | flat+push | 100% (12/12) | 4 | 2/12 | 4/4 | 0 / 0 | 17.6 |
| nobita_robust | clutter+push | 75% (9/12) | 41 | 11/12 | 38/41 | 0 / 3 | 38.9 |
| nobita_robust2 | flat | 100% (12/12) | 0 | 0/12 | - | 0 / 0 | 19.0 |
| nobita_robust2 | clutter | 25% (3/12) | 23 | 11/12 | 21/23 | 7 / 2 | 36.5 |
| nobita_robust2 | flat+push | 92% (11/12) | 9 | 8/12 | 9/9 | 0 / 0 | 23.2 |
| nobita_robust2 | clutter+push | 50% (6/12) | 31 | 11/12 | 25/31 | 0 / 6 | 38.5 |
