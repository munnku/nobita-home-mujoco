"""新しい家を描いて確かめる: 1 階・2 階の見下ろしと、IoT カメラ 5 台の絵 → build/house/"""
import os
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from house import build_with_info  # noqa: E402

xml, door_open = build_with_info(os.path.join(ROOT, "houses", "nobita"))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for name, q in door_open.items():
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"door_{name}")
    state = "open" if name not in ("genkan", "katteguchi", "toilet") else "closed"
    data.qpos[model.jnt_qposadr[j]] = q if state == "open" else 0.0
mujoco.mj_forward(model, data)
print(f"nbody={model.nbody} ngeom={model.ngeom} ncam={model.ncam} nlight={model.nlight}")
out = os.path.join(ROOT, "build", "house")
os.makedirs(out, exist_ok=True)
r = mujoco.Renderer(model, 900, 1200)
opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
for name, groups, lookat, dist, az, el in [
        ("top_1F", (0, 1, 2), (5.5, 3.6, 0), 12.5, 90, -89),
        ("top_2F", (0, 1, 2, 4), (5.5, 3.6, 2.7), 12.5, 90, -89),
        ("bird_SE", (0, 1, 2, 4, 5), (5.5, 3.6, 1.0), 16, 235, -30),
        ("cut_SE", (0, 1, 2), (5.5, 3.6, 0.0), 13, 240, -45)]:
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    r.update_scene(data, cam, opt)
    Image.fromarray(r.render()).save(os.path.join(out, f"{name}.png"))
r2 = mujoco.Renderer(model, 480, 640)
opt.geomgroup[:] = 1
opt.geomgroup[3] = 0
for i in range(model.ncam):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    r2.update_scene(data, n, opt)
    Image.fromarray(r2.render()).save(os.path.join(out, f"{n}.png"))
print("saved", out)
# 扉の確認 (近くから)
opt.geomgroup[:] = 0
for g in (0, 1, 2):
    opt.geomgroup[g] = 1
for name, lookat, az, el in [("door_genkan", (11.5, 5.0, 0.8), 0, -20),
                             ("door_katte", (0.5, 5.6, 0.8), 90, -20),
                             ("door_inner", (8.5, 3.6, 0.6), 90, -25)]:
    cam.lookat[:] = (lookat[0] * 0.91, lookat[1] * 0.91, lookat[2])
    cam.distance, cam.azimuth, cam.elevation = 3.0, az, el
    r.update_scene(data, cam, opt)
    Image.fromarray(r.render()).save(os.path.join(out, f"{name}.png"))
