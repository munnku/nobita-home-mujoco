"""Hello Robot Stretch 3 (片付け係)。床の小さい危ない物を拾って箱に入れる。

モデルは MuJoCo Menagerie の hello_robot_stretch_3 (Apache-2.0)。add_stretch() で家の MjSpec につなぎ、
Stretch(model, data) で動かす。物理を進めるのは呼ぶ側 (mj_step)。pick_step / drop_step / update は
毎ステップか制御の周期ごと (家では Duck.step の 0.02 s ごと) に呼ぶ。時間は data.time で測るのでどちらでもよい。

座標: 台車の前が +x、腕は右 (-y) に伸びる。link_grasp_center が指の間の掴む点。

家に入れるときの注意 (成功率と理由は docs/learning/pick_notes.md):
  - 物理の設定に noslip_iterations=3 を足すこと。MuJoCo の既定 (pyramidal・noslip なし) では数 g の物が
    指の間でじわじわ滑り、補助なしではほぼ拾えない。Microduck の歩き方は noslip 3 でも変わらなかった。
  - sim/duck.py は今、全アクチュエータを Microduck のものとみなしている (apply_bam が全部を BAM のモーターに替え、
    観測も range(model.nu))。Stretch (名前が st_) を除くようにしないと、Stretch の腕と車輪が動かない。
  - このクラスは model の値も書き換える (指の力の上限 actuator_forcerange、weld の相対姿勢 eq_data)。

timestep 0.005 と家の既定の積分器 (Euler) について:
  元の MJCF は implicitfast 用。車輪の速度サーボ (kv=100, gear 3) と指の位置サーボ (kv=124) の速度の項は
  Euler だと陽に積分され、0.005 では台車が勝手に回り指が暴れる (以前の「qvel ~17」の正体)。
  euler_safe=True (既定) ではその速度の項を関節の damping に移す。Euler は関節の damping を陰に扱うので安定し、
  力の式は同じ (違いは forcerange で切られるのが駆動の項だけになる点)。implicitfast でも同じ動きになる。
  MjSpec.attach は子の <option> を捨てて親の値を使う (積分器・cone・impratio・noslip)。
"""
import math
import os

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRETCH_XML = os.path.join(ROOT, "third_party", "mujoco_menagerie", "hello_robot_stretch_3", "stretch.xml")

WHEEL_R = 0.0508            # 車輪の半径 [m]
WHEEL_SEP = 0.3407          # 左右の車輪の間隔 [m]
WHEEL_MAX = 6.0             # 車輪の速度の上限 [rad/s]
BASE_RADIUS = 0.17          # 台車の外形 (約 0.34 m 角) の半分。経路計画で壁から離す量の目安
BASE_ACC = 0.5              # 台車の加速度の上限 [m/s^2]
BASE_ANG_ACC = 1.5          # 回る加速度の上限 [rad/s^2]

# 拾う姿勢
PITCH_DOWN = -1.55          # 手首を下に向ける (下限は -1.57)
LIFT_HOVER = 0.30           # 位置合わせ中の高さ (指先が床から 15 cm ほど)
LIFT_CARRY = 0.45           # 物を持って動くときの高さ
GRIP_OPEN = 0.02            # 物を持たないときの指の開き (パッドの下の端の間 約 12 cm)
GRIP_MARGIN = 0.02          # 下ろす前の指の開き = 物の幅 + これ (位置の誤差 ±1 cm を吸収)
GRIP_RATE = 0.008           # 閉じる速さ (gripper の指令/s、パッドの下の端で約 3 cm/s)
GRIP_CLOSE = -0.02          # 閉じる指令 (何も無ければ指どうしが当たる -0.0115 まで閉じる)
GRIP_FORCE = 4.0            # 指で挟む力の上限 [N] (床の上で挟むとき。強いと物が床へ押し込まれて抜ける)
GRIP_HOLD_FORCE = 6.0       # 床から離れたあとの力の上限 [N] (強くしても良くならなかった。V 字のパッドが物を押し出す)
LIFT_RATE = 0.2             # 持ち上げ・腕を縮めるときの目標の速さ [m/s] (急に動かすと小さい物が抜ける)
ARM_RATE = 0.08
GRIP_EMPTY = -0.0105        # 指がここより閉じたら「何も掴んでいない」
PRESS = 0.004               # 指先を床に押しつける量 (lift は約 400 N/m なので約 1.6 N)
REACH_MIN, REACH_MAX = 0.02, 0.50   # 腕の伸び (arm の指令) の範囲
REACH_DEFAULT = 0.25        # arm_side_point() で台車を置くときの腕の伸び
SMALL_WIDTH = 0.025         # 幅が分からないときの物の幅
TIP_LEVEL = 0.21            # 指先のゴムの受動関節 (rubber_*_y) のばねの中立 [rad]。元は 0 (パッドが 12° 内に傾いた V 字)
DROP_TOL = 0.03             # 箱に落とすときの位置合わせの誤差 [m]
ALIGN_MAX = 1.0             # 前後の位置合わせで台車が自分で動く距離の上限 [m] (それより遠ければナビで近づける)


def _quat_z(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def add_stretch(spec, prefix="st_", pos=(0.0, 0.0, 0.0), yaw=0.0, euler_safe=True, weld_objects=(),
                grip_force=GRIP_FORCE, tip_level=TIP_LEVEL):
    """Stretch を spec につなぐ。名前は全部 prefix 付き (アクチュエータ・キーフレーム home/stow も残る)。

    weld_objects: 掴んだときに溶接で補助する物の body 名 (add_objects の名前)。
        物ごとに無効の weld (<prefix>hold_<名前>) を作り、Stretch(weld_assist=True) のときだけ使う。
    grip_force: 指の位置サーボの力の上限 [N] (元は上限なしで kp=4000)。数 g の物を数十 N で挟むと
        柔らかい接触のせいで物が指や床にめり込むので抑える。None で元のまま。
    tip_level: 指先のパッドの傾き (受動関節のばねの中立)。元の 0 だとパッドが V 字で、内側の下の角が
        床から約 5 mm 浮き、硬貨 (1.6 mm)・ボタン電池 (3.2 mm) を横から挟めない。0.21 でパッドが平行になる。
    """
    robot = mujoco.MjSpec.from_file(STRETCH_XML)
    for side in ("left", "right"):     # 下を向けて閉じたとき、左右のパッドが平行で底が床と平らになる角度
        robot.joint(f"rubber_{side}_y").springref = tip_level
    if grip_force is not None:
        a = robot.actuator("gripper")
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-grip_force, grip_force]
    if euler_safe:
        for a in robot.actuators:
            j = robot.joint(a.target) if a.trntype == mujoco.mjtTrn.mjTRN_JOINT else None
            if a.name.endswith("wheel_vel"):          # 速度サーボ: 力 = g*kv*u - g^2*kv*qdot
                j.damping[0] += a.gainprm[0] * a.gear[0] ** 2
                a.biasprm[2] = 0.0
                # 元の forcerange ±100 は (kv*u - kv*g*qdot) を切るもの。駆動の項だけを切ると u≈1 で頭打ちに
                # なるので外す。指令は車輪の関節で ±6 rad/s (0.3 m/s、実機の最高速) まで (元は ±2 rad/s)
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
                a.ctrlrange = [-WHEEL_MAX * a.gear[0], WHEEL_MAX * a.gear[0]]
            elif a.name == "gripper":                 # 位置サーボの kv
                j.damping[0] += -a.biasprm[2]
                a.biasprm[2] = 0.0
    frame = spec.worldbody.add_frame(pos=list(pos), quat=_quat_z(yaw))
    spec.attach(robot, frame=frame, prefix=prefix)
    for name in weld_objects:
        eq = spec.add_equality(name=f"{prefix}hold_{name}", type=mujoco.mjtEq.mjEQ_WELD,
                               objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"{prefix}link_grasp_center", name2=name)
        eq.active = False
    return spec


class Stretch:
    """コンパイルしたモデルの上の Stretch 1 台。pick_step / drop_step は状態を持つので 1 つずつ使う。"""

    JOINTS = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper", "head_pan", "head_tilt")

    def __init__(self, model, data, prefix="st_", weld_assist=False):
        self.m, self.d, self.p = model, data, prefix
        m = model
        self.weld_assist = weld_assist
        self.act = {}
        for i in range(m.nu):
            n = m.actuator(i).name
            if n.startswith(prefix):
                self.act[n[len(prefix):]] = i
        self.base = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.gc = self._id(mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center")
        jq = lambda n: int(m.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, n)])
        self.qa_base = int(m.jnt_qposadr[m.body_jntadr[self.base]])
        self.va_base = int(m.jnt_dofadr[m.body_jntadr[self.base]])
        self.qa = {"lift": jq("joint_lift"), "wrist_yaw": jq("joint_wrist_yaw"), "wrist_pitch": jq("joint_wrist_pitch"),
                   "wrist_roll": jq("joint_wrist_roll"), "gripper": jq("joint_gripper_slide"),
                   "head_pan": jq("joint_head_pan"), "head_tilt": jq("joint_head_tilt")}
        self.qa_arm = [jq(f"joint_arm_l{i}") for i in range(4)]
        # ロボットの body と指先の geom
        self.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(prefix)}
        def side(s):
            bs = {self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in (f"rubber_tip_{s}", f"link_gripper_finger_{s}")}
            return {g for g in range(m.ngeom) if m.geom_bodyid[g] in bs and m.geom_contype[g] | m.geom_conaffinity[g]}
        self.left_geoms, self.right_geoms = side("left"), side("right")
        self.tip_geoms = self.left_geoms | self.right_geoms
        self.free_bodies = {b for b in range(m.nbody) if b not in self.bodies and m.body_jntnum[b] > 0
                            and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}
        self.qa_fingers = [jq(f"joint_gripper_finger_{s}_open") for s in ("left", "right")]
        self.welds = {}
        for e in range(m.neq):
            n = m.equality(e).name
            if n and n.startswith(prefix + "hold_"):
                self.welds[n[len(prefix) + 5:]] = e
        self.held = None                # weld で持っている物の名前
        self._cmd = self._vw = (0.0, 0.0)
        self._last_t = None
        self._dt = 0.0
        self._calibrate()
        self.reset_task()

    def _id(self, objtype, name):
        i = mujoco.mj_name2id(self.m, objtype, self.p + name)
        if i < 0:
            raise KeyError(self.p + name)
        return i

    # ---- 台車 ----
    def drive(self, v, w):
        """前進 v [m/s]、左回り w [rad/s] の目標。加速度を BASE_ACC / BASE_ANG_ACC で抑えて update() で近づける。

        段の指令をそのまま車輪に入れると、台車が浮いて回ったり倒れたりする (元の MJCF でも)。
        update() を毎ステップ (か制御の周期ごとに) 呼ぶこと。pick_step / drop_step は中で呼ぶ。
        """
        self._cmd = (float(v), float(w))
        self.update()

    def update(self):
        t = self.d.time
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        v = self._vw[0] + float(np.clip(self._cmd[0] - self._vw[0], -BASE_ACC * dt, BASE_ACC * dt))
        w = self._vw[1] + float(np.clip(self._cmd[1] - self._vw[1], -BASE_ANG_ACC * dt, BASE_ANG_ACC * dt))
        self._vw = (v, w)
        if dt > 0:                  # 同じステップで 2 回呼ばれても制御の周期を覚えておく (_ramp・指を閉じる速さに使う)
            self._dt = dt
        self._wheels(v, w)

    def _wheels(self, v, w):
        wl = (v - w * WHEEL_SEP / 2) / WHEEL_R
        wr = (v + w * WHEEL_SEP / 2) / WHEEL_R
        s = max(1.0, abs(wl) / WHEEL_MAX, abs(wr) / WHEEL_MAX)
        for name, wv in (("left_wheel_vel", wl), ("right_wheel_vel", wr)):
            i = self.act[name]      # 速度サーボの指令は gear 倍 (車輪の関節で wv rad/s)
            self.d.ctrl[i] = float(np.clip(wv / s * self._wheel_sign * self.m.actuator_gear[i, 0],
                                           *self.m.actuator_ctrlrange[i]))

    def stop(self):
        self.drive(0.0, 0.0)

    def halt(self):
        """加速度の制限なしにすぐ止める (場面の初期化用)。"""
        self._cmd = self._vw = (0.0, 0.0)
        self._wheels(0.0, 0.0)

    def reset(self, x, y, yaw):
        """台車を (x, y, yaw) に置き、関節を home の姿勢にそのまま入れる (立ち上がりの揺れなし)。"""
        self.set_pose(x, y, yaw)
        self.home()
        t = self.targets()
        q = self.d.qpos
        for k in ("lift", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper", "head_pan", "head_tilt"):
            q[self.qa[k]] = t[k]
        for a in self.qa_arm:
            q[a] = t["arm"] / 4
        for side in ("left", "right"):
            j = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"rubber_{side}_y")
            q[self.m.jnt_qposadr[j]] = self.m.qpos_spring[self.m.jnt_qposadr[j]]
        self.halt()
        self.reset_task()
        mujoco.mj_forward(self.m, self.d)

    def pose(self):
        """台車の (x, y, yaw)。"""
        R = self.d.xmat[self.base].reshape(3, 3)
        x, y = self.d.xpos[self.base][:2]
        return float(x), float(y), math.atan2(R[1, 0], R[0, 0])

    def set_pose(self, x, y, yaw):
        """台車を瞬間移動する (場面の初期化用)。"""
        q = self.qa_base
        self.d.qpos[q:q + 3] = [x, y, self.d.qpos[q + 2]]
        self.d.qpos[q + 3:q + 7] = _quat_z(yaw)
        self.d.qvel[self.va_base:self.va_base + 6] = 0

    def base_speed(self):
        return float(np.linalg.norm(self.d.qvel[self.va_base:self.va_base + 2]))

    def to_base(self, p):
        """世界座標の点 → 台車の座標 (x 前, y 左)。"""
        x, y, yaw = self.pose()
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    # ---- 関節 ----
    def joints(self):
        q = self.d.qpos
        out = {k: float(q[a]) for k, a in self.qa.items()}
        out["arm"] = float(sum(q[a] for a in self.qa_arm))
        return out

    def set_targets(self, **targets):
        """lift, arm, wrist_yaw, wrist_pitch, wrist_roll, gripper, head_pan, head_tilt の目標 (範囲で切る)。"""
        for k, v in targets.items():
            if v is None:
                continue
            i = self.act[k]
            lo, hi = self.m.actuator_ctrlrange[i]
            self.d.ctrl[i] = float(np.clip(v, lo, hi))

    def targets(self):
        return {k: float(self.d.ctrl[self.act[k]]) for k in self.JOINTS}

    def settled(self, keys=("lift", "arm", "wrist_pitch", "wrist_roll", "wrist_yaw"), tol=0.01):
        j, t = self.joints(), self.targets()
        return all(abs(j[k] - t[k]) < (tol if k in ("lift", "arm") else 5 * tol) for k in keys)

    def home(self):
        self.set_targets(lift=LIFT_CARRY, arm=0.0, wrist_yaw=0.0, wrist_pitch=PITCH_DOWN, wrist_roll=0.0,
                         gripper=0.0, head_pan=0.0, head_tilt=-0.6)
        self.stop()

    def grasp_point(self):
        return np.array(self.d.xpos[self.gc])

    # ---- 掴む点の位置 (順運動学) ----
    def _calibrate(self):
        """拾う姿勢での掴む点の台車座標、指の閉じる向き、車輪の向きを、別の MjData で順運動学から求める。"""
        m = self.m
        d = mujoco.MjData(m)
        q = self.qa_base
        d.qpos[q:q + 7] = [0, 0, 0, 1, 0, 0, 0]

        def fk(arm, roll=0.0, lift=LIFT_HOVER):
            d.qpos[self.qa["lift"]] = lift
            for a in self.qa_arm:
                d.qpos[a] = arm / 4
            d.qpos[self.qa["wrist_pitch"]] = PITCH_DOWN
            d.qpos[self.qa["wrist_roll"]] = roll
            d.qpos[self.qa["wrist_yaw"]] = 0.0
            mujoco.mj_kinematics(m, d)
            return np.array(d.xpos[self.gc]), np.array(d.xmat[self.gc]).reshape(3, 3)

        p0, _ = fk(0.0)
        p1, _ = fk(0.4)
        self.gc_x = float(p0[0])                          # 掴む点の前後の位置 (腕を伸ばしても変わらない)
        self.gc_y0 = float(p0[1])                         # arm=0 での横の位置
        self.gc_dy = float((p1[1] - p0[1]) / 0.4)         # arm 1 m あたりの横の動き (約 -1)
        self.gc_z_per_lift = float(p0[2] - LIFT_HOVER)    # 掴む点の高さ = lift + これ
        # 指の閉じる向き = 掴む点の座標のどれかの軸。roll を変えて水平面での向きの変わり方を見る
        tl = self._id(mujoco.mjtObj.mjOBJ_BODY, "rubber_tip_left")
        tr = self._id(mujoco.mjtObj.mjOBJ_BODY, "rubber_tip_right")

        def close_dir(roll):
            fk(0.2, roll)
            v = d.xpos[tl] - d.xpos[tr]
            return math.atan2(v[1], v[0])
        self.close_angle0 = close_dir(0.0)
        self.close_slope = _wrap(close_dir(0.5) - self.close_angle0) / 0.5      # 約 -1
        # 指の開き → パッドの下の端 (下から 4 mm) の間の幅。ほぼ直線 (q=0 で約 41 mm、q 1 mm で約 4 mm)
        box = mujoco.mjtGeom.mjGEOM_BOX
        gl = [g for g in self.left_geoms if m.geom_bodyid[g] == tl and m.geom_type[g] == box][0]
        gr = [g for g in self.right_geoms if m.geom_bodyid[g] == tr and m.geom_type[g] == box][0]

        def bottom_gap(q):
            fk(0.2)
            d.qpos[self.qa["gripper"]] = q
            for a in self.qa_fingers:
                d.qpos[a] = 10 * q          # 指の関節 = 10 x gripper (MJCF の equality)
            mujoco.mj_kinematics(m, d)
            u = d.xpos[tl] - d.xpos[tr]
            u[2] = 0
            u /= np.linalg.norm(u)

            def corners(g):
                c, R, sz = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
                return [c + R @ (np.array([a, b, e]) * sz) for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]
            L, Rr = corners(gl), corners(gr)
            z = min(p[2] for p in L + Rr) + 0.004
            return min(p @ u for p in L if p[2] <= z) - max(p @ u for p in Rr if p[2] <= z)
        g0, g1 = bottom_gap(0.0), bottom_gap(0.01)
        self.gap0, self.gap_per_q = g0, (g1 - g0) / 0.01
        # 車輪の向き: 両輪を + に回したとき台車が前 (+x) に進むか。車輪の関節軸 (z) の向きで決まる
        jl = self._id(mujoco.mjtObj.mjOBJ_JOINT, "joint_left_wheel")
        axis = d.xmat[m.jnt_bodyid[jl]].reshape(3, 3) @ m.jnt_axis[jl]
        # 車輪が軸 a まわりに + に回ると、接地点 (下) の速度は a x (0,0,-r)、台車は逆向きに進む
        ground = np.cross(axis, [0, 0, -WHEEL_R])
        self._wheel_sign = 1.0 if -ground[0] > 0 else -1.0

    def grip_for(self, width):
        """パッドの下の端の間を width にする gripper の指令。"""
        return float(np.clip((width - self.gap0) / self.gap_per_q, GRIP_EMPTY, 0.04))

    def roll_for(self, obj_yaw):
        """物の短い向き (物の y 軸 = yaw + 90°) で指を閉じる wrist_roll。指の向きは 180° で同じなので ±90° に収める。"""
        want = obj_yaw + math.pi / 2
        x, y, yaw = self.pose()
        r = _wrap(want - (yaw + self.close_angle0)) / self.close_slope
        r = (r + math.pi / 2) % math.pi - math.pi / 2
        return r

    def arm_side_point(self, target, yaw=None, reach=REACH_DEFAULT):
        """target (x, y) を、腕を reach 伸ばしたときの掴む点の真下に来させる台車の (x, y, yaw)。

        yaw を与えなければ今の向きのまま。家の中では壁にぶつからない yaw をいくつか試して選ぶ。
        """
        if yaw is None:
            yaw = self.pose()[2]
        bx, by = self.gc_x, self.gc_y0 + self.gc_dy * reach
        c, s = math.cos(yaw), math.sin(yaw)
        return (float(target[0] - (c * bx - s * by)), float(target[1] - (s * bx + c * by)), float(yaw))

    def reach_of(self, target):
        """target が今の台車からどれだけ腕を伸ばせば届くか (横方向)、と前後のずれ。"""
        b = self.to_base(target)
        return (b[1] - self.gc_y0) / self.gc_dy, b[0] - self.gc_x

    # ---- 指先の接触 ----
    def tip_touch(self):
        """指先が、ロボット以外の何か (床・物) に触れているか。"""
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.tip_geoms and m.geom_bodyid[g2] not in self.bodies:
                return True
            if g2 in self.tip_geoms and m.geom_bodyid[g1] not in self.bodies:
                return True
        return False

    def held_body(self):
        """左右の指 (パッドか指そのもの) の両方が触れている、ロボット以外の動く物 (free joint) の body。無ければ None。

        実機なら指の開き (と電流) で分かること。床や家具 (動かない) は数えない。
        """
        m, d = self.m, self.d
        left, right = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                b = int(m.geom_bodyid[o])
                if b in self.free_bodies:
                    if g in self.left_geoms:
                        left.add(b)
                    elif g in self.right_geoms:
                        right.add(b)
        both = left & right
        return min(both) if both else None

    def holding(self):
        """物を持っているか (weld で補助しているときは weld が効いていれば持っている)。"""
        if self.held is not None:
            return True
        return self.held_body() is not None and self.joints()["gripper"] > GRIP_EMPTY

    # ---- weld で補助 ----
    def _weld_on(self):
        if not self.weld_assist or not self.welds:
            return
        b = self.held_body()
        best = self.m.body(b).name if b is not None else None
        if best not in self.welds:
            return
        e = self.welds[best]
        b1, b2 = self.gc, mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, best)
        # 今の相対姿勢 (body1 から見た body2) のまま固める
        R1 = self.d.xmat[b1].reshape(3, 3)
        relpos = R1.T @ (self.d.xpos[b2] - self.d.xpos[b1])
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, self.d.xquat[b1])
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, q1inv, self.d.xquat[b2])
        self.m.eq_data[e, 0:3] = 0.0              # anchor (body2 の原点)
        self.m.eq_data[e, 3:6] = relpos
        self.m.eq_data[e, 6:10] = relq
        self.m.eq_data[e, 10] = 1.0
        self.d.eq_active[e] = 1
        self.held = best

    def _weld_off(self):
        if self.held is not None:
            self.d.eq_active[self.welds[self.held]] = 0
            self.held = None

    # ---- 拾う・置く (状態機械) ----
    def reset_task(self):
        self.phase = None
        self.status = "idle"
        self.reason = ""
        self._t0 = 0.0
        self._yaw_ref = None
        self._hold = None

    def _go(self, phase):
        self.phase = phase
        self._t0 = self.d.time
        self._yaw_ref = None
        if phase in ("align", "descend"):
            self._hold = None

    def _elapsed(self):
        return self.d.time - self._t0

    def _ramp(self, **goals):
        """関節の目標を LIFT_RATE / ARM_RATE の速さで goals に近づける。全部着いたら True。"""
        done = True
        for k, goal in goals.items():
            i = self.act[k]
            rate = {"lift": LIFT_RATE, "arm": ARM_RATE}[k] * self._dt
            c = float(self.d.ctrl[i])
            self.set_targets(**{k: c + float(np.clip(goal - c, -rate, rate))})
            done &= abs(goal - float(self.d.ctrl[i])) < 1e-6 or abs(goal - c) <= rate
        return done

    def set_grip_force(self, f):
        """指の力の上限 [N] を変える (このロボットのアクチュエータだけ)。"""
        self.m.actuator_forcerange[self.act["gripper"]] = [-f, f]

    def _fail(self, why):
        self.stop()
        # 腕を縮めて上げておく (伸ばしたまま走ると、腕が低い家具 (ちゃぶ台) に引っかかる)。持っている物は持ったまま
        self.set_targets(arm=0.0, lift=max(LIFT_CARRY, self.joints()["lift"]))
        self.status = "failed"
        self.reason = why
        return self.status

    def _align_base(self, target, tol=0.003):
        """台車を前後に動かして、掴む点の前後の位置を target に合わせる (誤差 tol [m] まで)。合えば True。"""
        yaw = self.pose()[2]
        if self._yaw_ref is None:           # 腕が横に出ていると左右の車輪の荷重が違い、まっすぐ進まないので向きを保つ
            self._yaw_ref = yaw
        ex = float(self.to_base(target)[0] - self.to_base(self.grasp_point())[0])
        if abs(ex) < tol and self.base_speed() < 0.005:
            self.stop()
            return True
        self.drive(float(np.clip(1.5 * ex, -0.12, 0.12)), float(np.clip(1.0 * _wrap(self._yaw_ref - yaw), -0.15, 0.15)))
        return False

    def _hold_pose(self):
        """台車を今の位置・向きに止めておく。車輪は速度サーボなので、指令 0 でも腕の力などで少しずつ回ってしまう。"""
        x, y, yaw = self.pose()
        if self._hold is None:
            self._hold = (x, y, yaw)
        hx, hy, hyaw = self._hold
        ex = math.cos(yaw) * (hx - x) + math.sin(yaw) * (hy - y)
        self.drive(float(np.clip(2.0 * ex, -0.03, 0.03)), float(np.clip(2.0 * _wrap(hyaw - yaw), -0.1, 0.1)))

    def _align_arm(self, target, tol=0.003):
        """腕の伸びで、掴む点の横の位置を target に合わせる (誤差 tol [m] まで)。合えば True。"""
        b, g = self.to_base(target), self.to_base(self.grasp_point())
        ey = float(b[1] - g[1]) / self.gc_dy
        arm = self.joints()["arm"] + ey
        if not REACH_MIN <= arm <= REACH_MAX:
            return None
        self.set_targets(arm=arm)
        return abs(ey) < tol and self.settled(("arm",), tol=max(tol, 0.004))

    def pick_step(self, target_xyz, yaw=0.0, width=SMALL_WIDTH):
        """毎ステップ呼ぶ。'running' / 'done' (掴んで持ち上げ腕を縮めた) / 'failed' (self.reason に理由)。

        target_xyz: 物の位置の見積もり (世界座標)。yaw: 物の向きの見積もり (分からなければ 0)。
        width: 指で挟む向きの物の幅 [m] (物の種類から。細長い物は短い方)。
        台車は target の横 (右側) に、腕が届く範囲 (REACH_MIN..REACH_MAX) に置いておくこと (arm_side_point)。
        手順: 腕を上げ手首を下へ → 台車を前後に動かし腕を伸ばして真上へ → 指を物の幅+3 cm に開く →
              指先が床に触れるまで下げる → 床に押しつけながらゆっくり閉じる → 5 cm 上げる → 腕を縮める。
        """
        self.update()
        if self.status in ("done", "failed"):
            return self.status
        target = np.asarray(target_xyz, float)
        if self.phase is None:
            self.status = "running"
            reach, ex = self.reach_of(target)
            if not REACH_MIN <= reach <= REACH_MAX or abs(ex) > ALIGN_MAX:
                return self._fail(f"届かない (腕 {reach:.2f} m, 前後 {ex:+.2f} m)")
            self._weld_off()
            self.set_grip_force(GRIP_FORCE)
            self._open = self.grip_for(width + GRIP_MARGIN)
            self.set_targets(lift=max(LIFT_HOVER, self.joints()["lift"]), wrist_yaw=0.0, wrist_pitch=PITCH_DOWN,
                             gripper=self._open)
            self._go("raise")
        ph = self.phase
        if ph == "raise":            # 腕を上げ、手首を下に向ける (台車が動く前に)
            if self.settled(("lift", "wrist_pitch", "wrist_yaw"), tol=0.02) or self._elapsed() > 4:
                self.set_targets(wrist_roll=self.roll_for(yaw))
                self._go("align")
        elif ph == "align":          # 台車の前後 → 腕の伸び
            ok_base = self._align_base(target)
            ok_arm = self._align_arm(target) if ok_base else False
            if ok_arm is None:
                return self._fail("腕の伸びの範囲を超えた")
            if ok_base and ok_arm and self.settled(("wrist_roll",), tol=0.02):
                self._go("descend")
            elif self._elapsed() > 15:
                return self._fail("位置合わせが終わらない")
        elif ph == "descend":        # 指先が床か物に触れるまで下げる。台車はその場に止め、横のずれは腕で直し続ける
            self._hold_pose()           # (台車で掴む点を追いかけると、腕を伸ばした台車の揺れを大きくしてしまう)
            self._align_arm(target)
            if self.tip_touch():
                self._hold = None
                self.set_targets(lift=self.joints()["lift"] - PRESS)
                self._grip = self._open
                self._go("close")
            elif self._elapsed() > 6:
                return self._fail("床に届かない")
            else:                    # 物の 5 cm 上までは速く、そこからはゆっくり
                above = self.grasp_point()[2] - target[2] - 0.05
                self.set_targets(lift=self.joints()["lift"] - (min(0.05, above) if above > 0.01 else 0.008))
        elif ph == "close":          # 閉じると指先は弧を描いて上がるので、lift で押し続けて床に沿わせる
            self._hold_pose()
            self.set_targets(lift=self.joints()["lift"] - PRESS)
            if self.holding():
                self.set_targets(gripper=GRIP_CLOSE)              # 力の上限 GRIP_FORCE で挟む
                self._go("squeeze")
            elif self.joints()["gripper"] < GRIP_EMPTY:
                self.set_targets(lift=LIFT_HOVER, gripper=GRIP_OPEN)
                return self._fail("つかめなかった (指が閉じ切った)")
            elif self._elapsed() > 6:
                self.set_targets(lift=LIFT_HOVER, gripper=GRIP_OPEN)
                return self._fail("閉じ終わらない")
            else:
                self._grip -= GRIP_RATE * self._dt
                self.set_targets(gripper=max(self._grip, GRIP_CLOSE))
        elif ph == "squeeze":
            self._hold_pose()
            if self._elapsed() > 0.4:
                if not self.holding():
                    self.set_targets(lift=LIFT_HOVER, gripper=GRIP_OPEN)
                    return self._fail("つかめなかった (挟むと抜けた)")
                self._weld_on()
                self._lift_goal = self.joints()["lift"] + 0.06
                self.set_targets(lift=self.joints()["lift"])           # 押しつけをやめる
                self._go("lift")
        elif ph == "lift":           # 6 cm 上げる
            self._hold_pose()
            arrived = self._ramp(lift=self._lift_goal)
            if self._elapsed() > 0.4:
                self.set_grip_force(GRIP_HOLD_FORCE)
            if arrived and self.settled(("lift",), tol=0.02):
                if not self.holding():
                    return self._fail("持ち上げる途中で落とした")
                self._go("retract")
            elif self._elapsed() > 5:
                return self._fail("持ち上がらない")
        elif ph == "retract":
            arrived = self._ramp(arm=0.0, lift=LIFT_CARRY)
            if not self.holding():
                return self._fail("腕を縮める途中で落とした")
            if arrived and self.settled(("arm", "lift"), tol=0.02) or self._elapsed() > 12:
                self.status = "done"
        return self.status

    def drop_step(self, box_xyz, height=0.10):
        """毎ステップ呼ぶ。box_xyz (箱の口の中心) の上 height で指を開いて落とす。'running' / 'done' / 'failed'。"""
        self.update()
        if self.status in ("done", "failed"):
            return self.status
        target = np.asarray(box_xyz, float)
        if self.phase is None:
            self.status = "running"
            reach, ex = self.reach_of(target)
            if not REACH_MIN <= reach <= REACH_MAX or abs(ex) > ALIGN_MAX:
                return self._fail(f"箱に届かない (腕 {reach:.2f} m, 前後 {ex:+.2f} m)")
            lift = target[2] + height + 0.05 - self.gc_z_per_lift
            self.set_targets(lift=max(lift, LIFT_HOVER), wrist_pitch=PITCH_DOWN)
            self._go("raise")
        ph = self.phase
        if ph == "raise":
            if self.settled(("lift",), tol=0.02) or self._elapsed() > 5:
                self._go("align")
        elif ph == "align":          # 箱の口 (30 × 25 cm) に落とすだけなので、拾うときほど細かく合わせない
            ok_base = self._align_base(target, tol=DROP_TOL)
            ok_arm = self._align_arm(target, tol=DROP_TOL) if ok_base else False
            if ok_arm is None:
                return self._fail("腕の伸びの範囲を超えた")
            if ok_base and ok_arm:
                self.set_targets(lift=target[2] + height - self.gc_z_per_lift)
                self._go("lower")
            elif self._elapsed() > 15:
                return self._fail("位置合わせが終わらない")
        elif ph == "lower":
            if self.settled(("lift",), tol=0.01) or self._elapsed() > 3:
                self._weld_off()
                self.set_grip_force(GRIP_FORCE)
                self.set_targets(gripper=GRIP_OPEN)
                self._go("release")
        elif ph == "release":
            if self._elapsed() > 1.0:
                self.set_targets(arm=0.0, lift=LIFT_CARRY, gripper=0.0)
                self._go("retract")
        elif ph == "retract":
            if self.settled(("arm",), tol=0.01) or self._elapsed() > 6:
                self.status = "done"
        return self.status
