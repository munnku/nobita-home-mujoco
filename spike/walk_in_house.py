"""のび太君の家の居間で、アヒルを経由点に向けて歩かせ、頭のカメラで YOLO が人を見つけるか確かめる。

経由点の追従は「向きの誤差に比例して回り、向きが合っていれば前に進む」だけの簡単な制御。
結果: build/spike/walk.mp4 (左: 後ろからの視点、右: 頭のカメラ + YOLO)、標準出力に到達時間と検出。
"""
import math
import os
import sys
import time

import cv2
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
os.environ.setdefault("MUJOCO_GL", "egl")
from scene import compose  # noqa: E402
from duck import Duck  # noqa: E402
from ultralytics import YOLO  # noqa: E402

PERSON = (1.0, 5.6)
spec = compose(os.path.join(ROOT, "houses", "nobita"), duck=(0, 0, 0),
               persons=[(PERSON[0], PERSON[1], -0.6, "stand"), (0.5, 3.65, 0.0, "lie")])
duck = Duck(spec, policy=os.environ.get("POLICY", "alpha_walking"))
duck.reset(2.9, 3.4, math.pi / 2)
m, d = duck.model, duck.data
yolo = YOLO("yolov8s.pt")
head = mujoco.Renderer(m, 240, 320)
chase = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
out = os.path.join(ROOT, "build", "spike")
vw = cv2.VideoWriter(os.path.join(out, "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640 + 320, 480))

waypoints = [(2.9, 5.8), (2.0, 5.9)]
wi, t, t0 = 0, 0.0, time.time()
log = []
while t < 90.0 and wi <= len(waypoints):
    x, y, yaw = duck.pose()
    if wi < len(waypoints):
        gx, gy = waypoints[wi]
        dist = math.hypot(gx - x, gy - y)
        if dist < 0.15:
            print(f"t={t:.1f}s reached waypoint {wi} {waypoints[wi]}")
            wi += 1
            continue
        err = math.atan2(gy - y, gx - x) - yaw
    else:
        # 最後は人の方を向いて止まる
        err = math.atan2(PERSON[1] - y, PERSON[0] - x) - yaw
        dist = 0.0
    err = (err + math.pi) % (2 * math.pi) - math.pi
    wz = max(-1.0, min(1.0, 2.0 * err))
    vx = 0.0 if abs(err) > 0.5 or dist == 0.0 else min(0.3, 0.3 * dist + 0.1)
    if wi >= len(waypoints) and abs(err) < 0.1:
        duck.set_cmd(0.0, 0.0, 0.0)
        for _ in range(50):
            duck.step()
        wi += 1
    duck.set_cmd(vx, 0.0, wz)
    duck.step()
    t += 0.02
    if duck.fallen():
        print(f"t={t:.1f}s FELL at {duck.pose()}")
        break
    if round(t / 0.02) % 10 == 0:   # 5 fps で描く
        head.update_scene(d, "head_camera")
        himg = head.render()
        res = yolo(np.ascontiguousarray(himg[..., ::-1]), verbose=False)[0]
        persons = [round(float(b.conf), 2) for b in res.boxes if yolo.names[int(b.cls)] == "person"]
        log.append((round(t, 1), persons))
        cam.lookat[:] = d.xpos[duck.trunk]
        cam.distance, cam.azimuth, cam.elevation = 1.2, math.degrees(yaw), -25
        chase.update_scene(d, cam)
        frame = np.zeros((480, 960, 3), np.uint8)
        frame[:, :640] = chase.render()[..., ::-1]
        frame[:240, 640:] = res.plot()
        cv2.putText(frame, f"t={t:.1f}s cmd=({vx:.2f},{wz:.2f}) person={persons}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(frame)
vw.release()
x, y, yaw = duck.pose()
print(f"end t={t:.1f}s pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) wall={time.time()-t0:.0f}s")
seen = [e for e in log if e[1]]
print(f"person seen in {len(seen)}/{len(log)} head frames; last: {log[-5:]}")
