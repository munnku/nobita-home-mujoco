# Stretch 3 で床の小さい物を拾う — 作り方・注意・考察

成功率の表は [pick_eval.md](pick_eval.md) (`learning/eval_pick.py` が作る)。コードは `sim/stretch.py`。

## 使い方 (家に入れるとき)

```python
from sim.objects import add_objects
from sim.stretch import add_stretch, Stretch

names = add_objects(spec, {"coin": 1, "marble": 1})           # 先に物を足す
add_stretch(spec, prefix="st_", pos=(x, y, 0), yaw=yaw,
            weld_objects=names)                                # 補助用の weld (無効) を物ごとに作る
model = spec.compile(); model.opt.timestep = 0.005
model.opt.noslip_iterations = 3                                # 補助なしで拾うなら必要 (下の「物理の設定」)
st = Stretch(model, data, prefix="st_", weld_assist=False)     # 動画で確実に拾わせたいなら True
st.reset(x, y, yaw)                                            # 台車を置き、関節を home の姿勢に入れる
```

毎ステップ (または制御の周期ごと、家なら `Duck.step` の 0.02 s ごと) に次のどれかを呼ぶ。時間は `data.time` で測る。

| 関数 | 説明 |
|---|---|
| `drive(v, w)` / `stop()` / `update()` | 台車の前進 v [m/s]・左回り w [rad/s]。加速度を 0.5 m/s²・1.5 rad/s² に抑えて `update()` で近づける (段の指令を車輪に入れると台車が浮いて回ったり倒れたりする。元の MJCF でも)。最高 0.3 m/s |
| `pose()` → `(x, y, yaw)` | 台車の位置と向き |
| `set_targets(lift=, arm=, wrist_yaw=, wrist_pitch=, wrist_roll=, gripper=, head_pan=, head_tilt=)` / `joints()` / `targets()` | 関節の目標と今の値。`home()` で運ぶ姿勢 |
| `arm_side_point(target_xy, yaw=None, reach=0.25)` → `(x, y, yaw)` | 物を腕の真下 (腕を `reach` 伸ばした所) に来させる台車の位置。腕は台車の右 (-y) に出るので、物は台車の右 0.15〜0.6 m。家では壁に当たらない yaw をいくつか試す。台車の外形は約 0.34 m 角 (`BASE_RADIUS = 0.17`) |
| `reach_of(target)` → `(腕の伸び, 前後のずれ)` | 今の台車から届くか。腕 0.02〜0.50 m、前後は 1 m まで自分で動いて合わせる |
| `pick_step(target_xyz, yaw=0, width=0.025)` → `'running' / 'done' / 'failed'` | 拾う。`reason` に失敗の理由。`width` は指で挟む向きの物の幅 (下の表) |
| `drop_step(box_xyz, height=0.10)` → 同上 | 箱の口の中心 `box_xyz` の上 `height` で指を開いて落とす |
| `holding()` / `held_body()` | 物を持っているか (左右の指が同じ動く物に触れている、か weld が効いている) |
| `reset_task()` | 次の pick / drop の前に呼ぶ (状態機械を初めに戻す) |

`pick_step` の手順: 腕を上げ手首を下へ → 台車を前後に動かし (向きを保ちながら) 掴む点を物の前後に合わせる → 腕の伸びで横を合わせる → 指を「物の幅 + 2 cm」に開き、手首を回して物の短い向きで挟むようにする → 指先が床に触れるまで下げる (台車は止めておき、横のずれは腕で直し続ける) → 床に約 1.6 N で押しつけながら 3 cm/s で閉じる → 左右の指が物に触れたら 4 N で挟む → 6 cm 上げる → 腕を縮め運ぶ高さへ。

物の種類ごとの `width` (`learning/eval_pick.py` の `grip_width`): ボタン電池 0.020、硬貨 0.0235、ビー玉 0.017、小さいブロック 0.016 (短い辺)、ハサミ 0.050 (刃の幅)。`yaw` は細長い物 (ブロック・ハサミ) だけ効く。

VDA 5050 の `pick` / `drop` アクションは、`pick_step` / `drop_step` が `running` の間 RUNNING、`done` で FINISHED、`failed` で FAILED (`reason` を resultDescription に) と対応させればよい。

## 家に入れるときに sim/duck.py で直す必要があること (このセッションでは触っていない)

- `apply_bam(spec, ...)` が **spec の全アクチュエータ** を BAM のモーターに替え、関節の damping も 0 にする。Stretch のアクチュエータ (名前が `st_` で始まる) を除くこと。除かないと Stretch の腕・車輪がトルク 0 のモーターになり動かない。
- `Duck.__init__` の `qpos_idx` / `qvel_idx` / `last_action` が `range(model.nu)` を使う。Stretch の 10 個が足されると観測の形が合わず止まる。Microduck の 14 個に絞ること (Stretch は後につなぐので先頭 14 個)。
- 上の 2 つを試験の中で継ぎはぎして、この家の居間 (台車 (5.0, 3.0)、ビー玉 (5.08, 2.62)) で `Duck.step` ごとに `pick_step` を呼び、weld 補助なしで拾えることを確かめた (19.4 s)。アヒルと同時に歩かせる確認はしていない。

## わかったこと

### timestep 0.005 と Euler (前に見えた qvel ~17 の正体)
- 元の MJCF は `implicitfast` 用。車輪の速度サーボ (kv=100, gear 3 → 関節で 900 N·m·s) と指の位置サーボの kv=124 は、Euler では陽に積分され、0.005 では不安定 (台車が勝手に回り、指先のゴムの関節が qvel 140 まで暴れる)。
- `add_stretch(euler_safe=True)` (既定) で、その速度の項を関節の damping に移した。Euler は関節の damping を陰に扱うので安定し、力の式は同じ。implicitfast と並べて同じ動きになることを確かめた。
- 速度の項を移すと、元の車輪の forcerange ±100 が駆動の項だけを切り、指令が 1 で頭打ちになる (台車が 1.6 cm/s しか出ない)。車輪の力の上限は外し、指令の範囲を車輪の関節で ±6 rad/s (0.3 m/s) にした (元は gear のため ±2 rad/s)。
- `MjSpec.attach` は子の `<option>` を捨てて親の値を使う (警告が出る)。アクチュエータ・tendon・equality・キーフレーム `st_home` / `st_stow` (ctrl 付き) は prefix 付きで残る。

### 指先のパッドの形
- 指を閉じると指先は弧を描いて上がる。大きく開いてから閉じると、パッドが 2 cm 以上浮いて物の上を通り過ぎる → 物の幅 + 2 cm に開いてから下ろす。
- 元のモデルではゴムのパッドが 12° 内に傾いた V 字で、内側の下の角が床から約 5 mm 浮く。硬貨 (1.6 mm)・ボタン電池 (3.2 mm) は横から挟めず上から押すだけになる。パッドの受動関節 `rubber_*_y` のばねの中立を 0.21 rad にすると、下を向けて閉じたときに左右のパッドが平行になる (`TIP_LEVEL`、`add_stretch(tip_level=0)` で元のまま)。これで電池・硬貨が 0〜3/8 → 4〜6/8 に上がった。
- 指が空で閉じ切ると、パッドが反対の指に当たって gripper -0.0115 で止まる。これを「掴めた」と見ないよう、掴めたかは「左右の指が同じ動く物 (free joint) に触れているか」で見る。

### 数 g の物と柔らかい接触
- MuJoCo の接触は柔らかく、めり込みは「力 ÷ 質量」で大きくなる。2〜5 g の物を元の指 (力の上限なし、kp 4000) で挟むと、物が指や床にめり込み、押し出される。指の力の上限を 4 N にした (6〜10 N より良く、15 N では全部ほぼ 0)。持ち上げたあと力を上げても良くならなかった。
- 床から離すときに速く上げる (0.2 m/s) 方が、ゆっくり上げる (0.05 m/s) より落としにくかった。

### 物理の設定 (一番効く)
- **家の既定 (Euler・pyramidal・impratio 1・noslip なし) では、補助なしの成功率はほぼ 0** (ブロック 13%、他 0%)。挟んで少し持ち上げても、柔らかい摩擦で物がじわじわ滑り落ちる。
- `noslip_iterations=3` を足すと: ビー玉 92%、ボタン電池 77%、ブロック 75%、硬貨 55%、ハサミ 0% (各 100 回)。Microduck の歩き方 (velstand、前進 0.2・0.3 m/s、旋回 1 rad/s) は noslip 3 でも同じだった (転ばず、速さもほぼ同じ。計算は少し重くなる)。
- Stretch の MJCF の option (implicitfast・elliptic・impratio 10・noslip 3) や elliptic + impratio 10 では、床の上でパッドの摩擦が強すぎて指が閉じ切らなかった。
- weld 補助ありは家の既定でも 84〜100%。補助は「左右の指が物に触れた」ときだけ効くので、残る失敗は挟む前に物が抜けたもの (ハサミ 16〜28%)。

### 制御の周期
- 家では `Duck.step` が 4 ステップ (0.02 s) ずつ進めるので、`pick_step` も 0.02 s ごとになる。初めの版は 0.005 ごとでしか合わず、0.02 s では腕を伸ばした台車の揺れを台車で追いかけて大きくしていた。下ろす間と挟む間は台車をその場に止める制御 (`_hold_pose`) にし、横のずれは腕で直すようにした。表は 0.02 s の周期 (`--ctrl-every 4`、既定) の結果。0.005 s ごとでもほぼ同じ (各 20 回の比較)。
- 腕を横に伸ばした台車は、車輪を止めていても少しずつ回る (車輪が速度サーボで位置を保たないため)。

### ハサミ (0%)
- 刃 (170 x 50 x 8 mm、50 g) の幅 50 mm を挟む。持ち上げる途中で落とすか、挟む前に抜ける。力を 8〜30 N にしても、開き・押しつけを変えても 0/8 のままで、原因は詰め切れていない (重さ 60 g に対して 4 N では弱く、強くすると押し出される)。動画では weld 補助ありにするのがよい (69〜84%)。

## 学習 (PPO) をしなかった理由

補助なしの成功率が 90% を下回る種類 (硬貨・電池・ブロック・ハサミ) があるので、手順 3 の条件の片方は満たす。ただし 2026-10-23 14:04 の時点で GPU は `pai_mujoco` の歩き方の学習 (Mjlab-Home-Clutter-Backlash-MicroDuck、11.6 GB 使用、あと 10 分ほど) が使っていたので始めていない。CPU で stable-baselines3 を使う案も、`.venv` に stable-baselines3 / gymnasium が入っておらず、入れると共有の `pyproject.toml` / `uv.lock` を変えることになるのでやめた。

やるなら: 台車は固定、観測 = 物の位置の見積もり (±1 cm) + 腕・手首・指の関節、行動 = lift / arm / wrist_pitch / wrist_roll / gripper の目標の差分、報酬 = 物の高さ + 両指の接触 − 物の床へのめり込み、`learning/eval_pick.py` の `Env.reset` をそのまま乱数の場面に使う。ただし失敗の多くは数 g の物と柔らかい接触の性質 (押し出される・滑る) なので、学習がそれを乗り越えるより、シミュレーションの穴 (めり込み) を使う方策になるおそれがある。mjlab (MuJoCo Warp) で noslip が使えるかも確かめていない。
