"""Microduck を学習済み方策 (ONNX) で歩かせる。

観測・行動の決まりは pollen-robotics/microduck_rl の scripts/infer_policy.py (Apache-2.0) に合わせた:
  観測 61 次元 = 角速度 (IMU, 3) + 重力の向き (胴体座標, 3) + 関節角 - 既定姿勢 (14) + 関節速度 (14)
               + 前回の行動 (14) + 指令 (13: 速度 vx, vy, wz / 頭の姿勢 4 / 胴体の姿勢 6)
  行動 14 次元 → 関節の目標角 = 既定姿勢 + 行動。50 Hz (物理 5 ms × 4)
サーボは BAM (Rhoban/bam) の XL330 モデルで動かす (方策の学習と同じ)。
"""
import math
import os

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL = os.environ.get("MICRODUCK_RL", os.path.join(ROOT, "third_party", "microduck_rl"))
POLICY_DIR = os.environ.get("MICRODUCK_POLICIES", os.path.join(ROOT, "third_party", "policies"))
DUCK_XML = os.path.join(RL, "src", "mjlab_microduck", "robot", "microduck",
                        "robot_groundcontact_backlash.xml")

TIMESTEP = 0.005
DECIMATION = 4
# 既定姿勢 (infer_policy.py の DEFAULT_POSE = STAND2)。関節の順はアクチュエータの順
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,   # 左脚
                         0.3491, 0.3491, 0.0, 0.0,                  # 首・頭
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530],     # 右脚
                        dtype=np.float32)
# BAM の設定 (infer_policy.py と同じ値)
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_bam_model(vin=BAM_VIN):
    from bam.model import load_model
    m = load_model(motor_name="xl330", model="m6")
    m.actuator.kp = BAM_KP_FW
    m.actuator.vin = vin
    m.actuator.max_current = None
    return m


def apply_bam(spec, bam_model):
    """位置制御のアクチュエータを BAM が書くトルクのモーターに替える (infer_policy.load_mujoco_with_bam と同じ)。"""
    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    joints = {j.name: j for j in spec.joints}
    names = []
    for act in spec.actuators:
        if act.target.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        j = joints.get(act.target)
        if j is not None:
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = BAM_STIFF_SOLREF_FRICTION
            j.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
    return names


def _rotate_inverse(q, v):
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


class Duck:
    """家のシーン (MjSpec、アヒルをつないだもの) から作る。model と data を持ち、step() で 0.02 s 進める。"""

    def __init__(self, spec, policy="velstand"):
        from bam.mujoco import MujocoController
        bam_model = load_bam_model()
        names = apply_bam(spec, bam_model)
        self.model = spec.compile()
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.bam = MujocoController(bam_model, names, self.model, self.data,
                                    vin_drop_gain=BAM_VIN_DROP_GAIN, vin_min=BAM_VIN_MIN)
        path = policy if policy.endswith(".onnx") else os.path.join(POLICY_DIR, policy + ".onnx")
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        m = self.model
        self.qpos_idx = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.qvel_idx = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qa = int(m.jnt_qposadr[fj])
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = int(m.sensor_adr[gyro])
        self.last_action = np.zeros(m.nu, np.float32)
        self.command = np.zeros(13, np.float32)
        self.time = 0.0

    def reset(self, x, y, yaw):
        d, qa = self.data, self.qa
        d.qpos[qa:qa + 3] = [x, y, 0.125]
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[self.qpos_idx] = DEFAULT_POSE
        d.qvel[:] = 0
        self.bam.reset(d.qpos)
        self.bam.q_target[:] = DEFAULT_POSE
        self.last_action[:] = 0
        mujoco.mj_forward(self.model, d)

    def set_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        """速度指令 (胴体の向きで前 vx・左 vy [m/s]、左回り wz [rad/s])。"""
        self.command[0:3] = (vx, vy, wz)

    def set_head(self, neck_pitch=0.0, head_pitch=0.0, head_yaw=0.0, head_roll=0.0):
        """頭の向きの指令 (既定姿勢からの差 [rad])。人を見上げるときなどに使う。"""
        self.command[3:7] = (neck_pitch, head_pitch, head_yaw, head_roll)

    def observe(self):
        d = self.data
        quat = d.xquat[self.trunk].astype(np.float32)
        return np.concatenate([
            d.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32),
            _rotate_inverse(quat, np.array([0.0, 0.0, -1.0], np.float32)),
            d.qpos[self.qpos_idx].astype(np.float32) - DEFAULT_POSE,
            d.qvel[self.qvel_idx].astype(np.float32),
            self.last_action,
            self.command,
        ]).astype(np.float32)

    def step(self):
        action = self.session.run([self.out_name], {self.in_name: self.observe()[None]})[0][0]
        self.last_action = action.astype(np.float32)
        self.bam.q_target[:] = DEFAULT_POSE + self.last_action
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        self.time += TIMESTEP * DECIMATION

    def pose(self):
        q = self.data.qpos[self.qa:self.qa + 7]
        w, x, y, z = q[3:7]
        return float(q[0]), float(q[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def fallen(self):
        return self.data.qpos[self.qa + 2] < 0.07
