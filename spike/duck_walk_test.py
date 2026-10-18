"""Microduck の公開方策 (velstand.onnx) を画面なしで動かし、速度指令どおりに歩けるかを確かめる。

uv run --project ~/ext/microduck_rl python spike/duck_walk_test.py [scene.xml]
"""
import os
import sys
import time

import mujoco
import numpy as np

RL = os.path.expanduser("~/ext/microduck_rl")
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)
import infer_policy as ip  # noqa: E402

scene = sys.argv[1] if len(sys.argv) > 1 else ip.MICRODUCK_XML
policy_path = os.path.expanduser("~/ext/policies/" + os.environ.get("POLICY", "velstand") + ".onnx")

bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
model, data, bam, _ = ip.load_mujoco_with_bam(scene, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
pol = ip.PolicyInference(model, data, bam_ctrl=bam, walking_onnx_path=policy_path,
                         use_projected_gravity=True, new_cmd_obs=True)
fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
qa = model.jnt_qposadr[fj]
data.qpos[qa:qa + 3] = [0, 0, 0.125]
data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
for i, q in enumerate(pol.joint_qpos_indices):
    data.qpos[q] = pol.default_pose[i]
bam.reset(data.qpos)
pol.set_position_targets(pol.default_pose)
mujoco.mj_forward(model, data)


def yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# (秒, vx, vy, wz): 立つ → 前進 → その場で回る → 前進しながら曲がる → 止まる
plan = [(2, 0, 0, 0), (8, 0.25, 0, 0), (4, 0, 0, 1.0), (6, 0.2, 0, 0.5), (3, 0, 0, 0)]
t0 = time.time()
sim_t = 0.0
for dur, vx, vy, wz in plan:
    pol.set_vel_cmd(vx, vy, wz)
    p0 = data.qpos[qa:qa + 3].copy()
    y0 = yaw(data.qpos[qa + 3:qa + 7])
    for _ in range(int(dur * 50)):
        pol.apply_action(pol.infer())
        for _ in range(4):
            bam.update()
            mujoco.mj_step(model, data)
        sim_t += 0.02
    p1 = data.qpos[qa:qa + 3]
    y1 = yaw(data.qpos[qa + 3:qa + 7])
    print(f"cmd=({vx},{vy},{wz}) {dur}s: moved={np.linalg.norm(p1[:2]-p0[:2]):.2f} m "
          f"dyaw={np.degrees((y1-y0+np.pi)%(2*np.pi)-np.pi):.0f} deg height={p1[2]:.3f}")
print(f"sim {sim_t:.0f}s in wall {time.time()-t0:.1f}s")
