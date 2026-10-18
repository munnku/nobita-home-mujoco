"""家 + アヒル + 人を作り、アヒルの頭のカメラと IoT カメラで描いて YOLO が人を見つけるか確かめる。"""
import os
import sys
import time

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from scene import compose  # noqa: E402

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # Docker の中は画面なしで描く
out = os.path.join(ROOT, "build", "spike")
os.makedirs(out, exist_ok=True)

spec = compose(os.path.join(ROOT, "houses", "nobita"), duck=(2.4, 4.0, 2.3),
               persons=[(1.0, 5.6, -0.6, "stand"), (0.5, 3.65, 0.0, "lie")])
model = spec.compile()
data = mujoco.MjData(model)
fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
qa = model.jnt_qposadr[fj]
data.qpos[qa + 2] = 0.125
mujoco.mj_forward(model, data)

from ultralytics import YOLO  # noqa: E402
yolo = YOLO(os.path.join(ROOT, "third_party", "yolov8s.pt"))
r = mujoco.Renderer(model, 480, 640)
for cam in ["head_camera", "cam_ima", "cam_ousetsu", "cam_dk", "cam_rouka"]:
    t = time.time()
    r.update_scene(data, cam)
    img = r.render()
    ms = (time.time() - t) * 1000
    res = yolo(np.ascontiguousarray(img[..., ::-1]), verbose=False)[0]  # YOLO は numpy を BGR とみなす
    dets = [(yolo.names[int(b.cls)], round(float(b.conf), 2), [int(v) for v in b.xyxy[0]])
            for b in res.boxes if float(b.conf) > 0.25]
    Image.fromarray(res.plot()[..., ::-1]).save(os.path.join(out, f"{cam}.png"))
    print(f"{cam}: render {ms:.0f} ms, dets={dets}")
# 全体を上から
r2 = mujoco.Renderer(model, 720, 960)
cam = mujoco.MjvCamera()
cam.lookat[:] = [3.6, 5.4, 0]
cam.distance, cam.azimuth, cam.elevation = 13.0, 90, -70
r2.update_scene(data, cam)
Image.fromarray(r2.render()).save(os.path.join(out, "overview.png"))
print("saved to", out)
# アヒルの後ろから (カメラの向きの確認用)
cam.lookat[:] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")]
cam.distance, cam.azimuth, cam.elevation = 0.8, np.degrees(2.3) + 180, -20
r2.update_scene(data, cam)
Image.fromarray(r2.render()).save(os.path.join(out, "chase.png"))
t = time.time()
for _ in range(10):
    r.update_scene(data, "cam_ima"); r.render()
print(f"steady render cam_ima: {(time.time()-t)*100:.0f} ms")
