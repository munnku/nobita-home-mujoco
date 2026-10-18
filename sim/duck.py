"""Microduck を学習済み方策 (ONNX) で歩かせる。

pollen-robotics/microduck_rl の scripts/infer_policy.py の PolicyInference (観測の組み立てと方策の実行) と、
BAM (サーボのモデル) の読み込み手順をそのまま使う。ここで足すのは:
- 家のシーン (MjSpec) にアヒルをつないだあとで BAM 用にアクチュエータを書き換える
- 速度指令 (vx, vy, wz) を受けて 50 Hz で方策を回す step()
- 位置と向き (pose) の取り出し
"""
import math
import os
import sys

import mujoco
import numpy as np

RL = os.path.expanduser(os.environ.get("MICRODUCK_RL", "~/ext/microduck_rl"))
sys.path.insert(0, os.path.join(RL, "scripts"))
import infer_policy as ip  # noqa: E402

POLICY_DIR = os.path.expanduser(os.environ.get("MICRODUCK_POLICIES", "~/ext/policies"))
DECIMATION = 4          # 物理 5 ms × 4 = 方策 50 Hz
TIMESTEP = 0.005


def apply_bam(spec, bam_model):
    """infer_policy.load_mujoco_with_bam と同じ書き換えを、ファイルではなく MjSpec に対して行う。"""
    kt, R = bam_model.kt.value, bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R
    names = []
    joints = {j.name: j for j in spec.joints}
    for act in spec.actuators:
        tgt = act.target
        if tgt.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        j = joints.get(tgt)
        if j is not None:
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = ip.BAM_STIFF_SOLREF_FRICTION
            j.solimp_friction = ip.BAM_STIFF_SOLIMP_FRICTION
    return names


class Duck:
    def __init__(self, spec, policy="alpha_walking", vin=7.4):
        from bam.mujoco import MujocoController
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, vin, ip.BAM_MAX_CURRENT)
        names = apply_bam(spec, bam_model)
        self.model = spec.compile()
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.bam = MujocoController(bam_model, names, self.model, self.data,
                                    vin_drop_gain=0.1, vin_min=ip.BAM_VIN_MIN)
        devnull = open(os.devnull, "w")
        stdout, sys.stdout = sys.stdout, devnull   # PolicyInference は起動時に多くを print する
        try:
            self.pol = ip.PolicyInference(
                self.model, self.data, bam_ctrl=self.bam,
                walking_onnx_path=os.path.join(POLICY_DIR, policy + ".onnx"),
                use_projected_gravity=True, new_cmd_obs=True)
        finally:
            sys.stdout = stdout
        fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qa = self.model.jnt_qposadr[fj]
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.cmd = (0.0, 0.0, 0.0)

    def reset(self, x, y, yaw):
        d, qa = self.data, self.qa
        d.qpos[qa:qa + 3] = [x, y, 0.125]
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, q in enumerate(self.pol.joint_qpos_indices):
            d.qpos[q] = self.pol.default_pose[i]
        d.qvel[:] = 0
        self.bam.reset(d.qpos)
        self.pol.set_position_targets(self.pol.default_pose)
        mujoco.mj_forward(self.model, d)

    def set_cmd(self, vx, vy, wz):
        if (vx, vy, wz) == self.cmd:
            return
        self.cmd = (vx, vy, wz)
        self.pol.vel_cmd = np.array([vx, vy, wz], dtype=np.float32)
        self.pol._update_policy_session()
        self.pol._update_command()

    def step(self):
        """方策を 1 回 (0.02 s 分) 進める。"""
        self.pol.apply_action(self.pol.infer())
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)

    def pose(self):
        q = self.data.qpos[self.qa:self.qa + 7]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), yaw

    def fallen(self):
        return self.data.qpos[self.qa + 2] < 0.07
