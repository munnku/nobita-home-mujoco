"""地図と A* でアヒルを部屋から部屋へ歩かせる。居間 → DK → 応接室 → 洗面所。

結果: build/nav/map_*.png (地図と経路)、build/nav/nav.mp4 (見下ろし + 後ろから)、標準出力に各区間の時間。
"""
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from world import World  # noqa: E402

M = 0.91
out = os.path.join(ROOT, "build", "nav")
os.makedirs(out, exist_ok=True)
w = World(duck_pose=(5.6 * M, 3.2 * M, -math.pi / 2), n_persons=1)
w.place_person(0, 10.0 * M, 2.2 * M, yaw=math.pi, pose="stand")   # 応接室のソファの前に人
goals = [("dk", 2.4 * M, 4.7 * M), ("ousetsu", 10.6 * M, 3.3 * M), ("senmen", 7.0 * M, 6.7 * M)]
top = mujoco.MjvCamera()
top.lookat[:] = [5.5, 3.6, 0]
top.distance, top.azimuth, top.elevation = 11.5, 90, -89
vw = cv2.VideoWriter(os.path.join(out, "nav.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960 + 480, 720))
t0 = time.time()
for name, gx, gy in goals:
    ok = w.go_to(gx, gy)
    img = w.map.image(w.last_path if ok else None, marks=[(gx, gy)])
    cv2.imwrite(os.path.join(out, f"map_{name}.png"), cv2.resize(img[..., ::-1], None, fx=3, fy=3,
                                                                 interpolation=cv2.INTER_NEAREST))
    if not ok:
        print(f"{name}: no path")
        continue
    start = w.time
    while w.nav_status() == "moving" and w.time - start < 240:
        w.step()
        if w.duck.fallen():
            break
        if round(w.time / 0.02) % 10 == 0:
            frame = np.zeros((720, 1440, 3), np.uint8)
            frame[:, :960] = w.render(top, 960, 720, groups=(0, 1, 2))[..., ::-1]
            frame[:360, 960:] = w.render(w.chase_camera(), 480, 360)[..., ::-1]
            frame[360:, 960:] = w.render("head_camera", 480, 360)[..., ::-1]
            cv2.putText(frame, f"t={w.time:.0f}s -> {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 2)
            vw.write(frame)
    x, y, yaw = w.duck.pose()
    print(f"{name}: {"FELL" if w.duck.fallen() else w.nav_status()} retries={w.retries} "
          f"in {w.time-start:.1f}s, pose=({x:.2f},{y:.2f}) goal=({gx:.2f},{gy:.2f}) room={w.room_of(x, y)} "
          f"path_pts={len(w.last_path)}")
    if w.duck.fallen():
        break
vw.release()
print(f"wall {time.time()-t0:.0f}s for sim {w.time:.0f}s")
