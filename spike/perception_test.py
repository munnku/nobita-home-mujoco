"""IoT カメラ (と頭のカメラ) で人の位置と姿勢を推定し、正解と比べる。"""
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, ROOT)
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
from home.perception import CameraModel, PersonDetector, estimate_person  # noqa: E402
from world import World  # noqa: E402

M = 0.91
w = World(duck_pose=(5.6 * M, 3.0 * M, 0.0), n_persons=2)
det = PersonDetector()
out = os.path.join(ROOT, "build", "perception")
os.makedirs(out, exist_ok=True)
cases = [
    ("ima", [(0, 6.0 * M, 3.0 * M, 0.5, "stand"), (1, 7.6 * M, 1.0 * M, 2.5, "lie")], "cam_ima"),
    ("ousetsu", [(0, 11.0 * M, 3.3 * M, -2.0, "stand"), (1, 10.2 * M, 0.5 * M, 0.0, "lie")], "cam_ousetsu"),
    ("dk", [(0, 1.3 * M, 2.0 * M, 0.0, "stand"), (1, 3.0 * M, 4.6 * M, 1.6, "lie")], "cam_dk"),
    ("rouka", [(0, 8.0 * M, 5.0 * M, 3.1, "stand"), (1, 9.5 * M, 4.6 * M, 3.1, "lie")], "cam_rouka"),
]
for name, persons, camname in cases:
    for i, x, y, yaw, pose in persons:
        w.place_person(i, x, y, yaw, pose, update_map=False)
    mujoco.mj_forward(w.model, w.data)
    img = w.render(camname)
    t = time.time()
    boxes = det.detect([img])[0]
    ms = (time.time() - t) * 1000
    cam = CameraModel.from_mujoco(w.model, w.data, camname, 640, 480)
    print(f"--- {name} ({camname}) yolo {ms:.0f} ms")
    for i, x, y, yaw, pose in persons:
        # 正解: 立っていれば足元、横になっていれば体の真ん中
        cx, cy = (x, y) if pose == "stand" else (x + 0.9 * math.cos(yaw), y + 0.9 * math.sin(yaw))
        print(f"  truth person{i}: ({cx:.2f},{cy:.2f}) {pose}")
    vis = img[..., ::-1].copy()
    for b in boxes:
        est = estimate_person(cam, b)
        print(f"  est: {est}  box={[int(v) for v in b[:4]]}")
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        if est:
            cv2.putText(vis, f"{est['posture']} h={est['height']}", (int(b[0]), int(b[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, f"{name}.png"), vis)
