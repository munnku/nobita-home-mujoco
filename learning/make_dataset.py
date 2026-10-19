"""YOLO の追加学習用のデータを MuJoCo で作る (正解の枠は自動で付く)。

1 枚ごとに: 人 2 人を家の中のランダムな場所・向き・姿勢 (立つ / 横になる) に置き、照明をランダムに点け消しし、
見守りカメラ 5 台とアヒルの頭のカメラ (ランダムな場所・向き) で撮る。
正解の枠: セグメンテーションの描画 (画素ごとにどの geom か) から人ごとの見えている画素を集めて外接する箱。
見えている画素が少なすぎる人 (MIN_PIXELS) は「難しすぎる」として、その絵ごと捨てずに枠を付けない (ignore しない)。

クラス: 0 = person_standing (立っている人), 1 = person_lying (横になっている人)
出力: build/datasets/nobita/{images,labels}/{train,val,test}/*.{png,txt} と data.yaml (Ultralytics の形式)
uv run python learning/make_dataset.py --scenes 1500 --seed 0
"""
import argparse
import math
import os
import random
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from world import World  # noqa: E402

W, H = 640, 480
MIN_PIXELS = 120
CLASSES = ["person_standing", "person_lying"]
# 人を置く部屋 (見守りカメラのある部屋と、カメラの無い廊下の奥など)
ROOMS = ["ima", "ousetsu", "dk", "rouka", "nobita"]


def inside_furniture(lay, level, x, y, margin):
    m = lay["module"]
    for f in lay["furniture"]:
        if f["level"] != level or f.get("z0", 0) > 1.0:
            continue
        x0, x1, y0, y1 = (v * m for v in f["rect"])
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
            return True
    st = lay.get("stairs")
    if level == "1F" and st:
        if 5.0 * m - margin <= x <= 7.0 * m + margin and 4.0 * m - margin <= y <= 5.0 * m + margin:
            return True
    return False


def sample_person(rng, w, room):
    """部屋の中で家具と重ならない (x, y, yaw, pose, level) を選ぶ。"""
    lay = w.layout
    r = next(r for r in lay["rooms"] if r["name"] == room)
    m = lay["module"]
    x0, x1, y0, y1 = (v * m for v in r["rect"])
    level = r["level"]
    pose = "lie" if rng.random() < 0.55 else "stand"     # 姿勢を先に決める (場所が見つかりやすい立ち姿に偏らないように)
    for _ in range(400):
        x, y = rng.uniform(x0 + 0.3, x1 - 0.3), rng.uniform(y0 + 0.3, y1 - 0.3)
        yaw = rng.uniform(-math.pi, math.pi)
        pts = [(x, y)]
        if pose == "lie":
            pts += [(x + k * math.cos(yaw), y + k * math.sin(yaw)) for k in (0.6, 1.2, 1.75)]
        if all(x0 + 0.15 <= px <= x1 - 0.15 and y0 + 0.15 <= py <= y1 - 0.15
               and not inside_furniture(lay, level, px, py, 0.15) for px, py in pts):
            return x, y, yaw, pose, level
    return None


def boxes_from_seg(seg, model, person_bodies):
    """seg: (H, W, 2) [objid, objtype]。人 (body id → 人の番号) ごとの見えている画素の外接箱。"""
    out = {}
    geom_mask = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = np.where(geom_mask, seg[..., 0], -1)
    for b, k in person_bodies.items():
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        mask = np.isin(ids, geoms)
        n = int(mask.sum())
        if n < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(mask)
        out[k] = (xs.min(), ys.min(), xs.max(), ys.max(), n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "datasets", "nobita"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    w = World(n_persons=2)
    m = w.model
    person_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"person{i}"): i for i in range(2)}
    rgb = mujoco.Renderer(m, H, W)
    seg = mujoco.Renderer(m, H, W)
    seg.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    opt.geomgroup[3] = 0
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) != "head_camera"]
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)
    counts = {"images": 0, "standing": 0, "lying": 0}
    meta = {"train": {}, "val": {}, "test": {}}    # 絵ごとのカメラの位置・向きと人の正解 (評価に使う)
    for s in range(a.scenes):
        split = "train" if s % 10 < 8 else ("val" if s % 10 == 8 else "test")
        rooms = rng.sample(ROOMS, 2) if rng.random() < 0.6 else [rng.choice(ROOMS)] * 2
        poses = {}
        for i, room in enumerate(rooms):
            p = sample_person(rng, w, room)
            if p is None:
                w.place_person(i, None, None, update_map=False)
                continue
            x, y, yaw, pose, level = p
            w.place_person(i, x, y, yaw, pose, level, update_map=False)
            poses[i] = pose
        for room in w.layout.get("lights", []):
            w.set_light(room, rng.random() < 0.7)
        # アヒルを人の近くの床に置き、だいたい人の方へ向ける (頭のカメラの絵を作るため)
        target = next((st for st in w.person_state.values() if st and st["level"] == "1F"), None)
        if target is not None:
            for _ in range(50):
                d = rng.uniform(0.6, 3.0)
                ang = rng.uniform(-math.pi, math.pi)
                dx, dy = target["x"] + d * math.cos(ang), target["y"] + d * math.sin(ang)
                if w.room_of(dx, dy) and not inside_furniture(w.layout, "1F", dx, dy, 0.15):
                    yaw = math.atan2(target["y"] - dy, target["x"] - dx) + rng.uniform(-0.5, 0.5)
                    w.duck.reset(dx, dy, yaw)
                    break
        mujoco.mj_forward(m, w.data)
        views = cams + (["head_camera"] if target is not None else [])
        for cam in views:
            seg.update_scene(w.data, cam, opt)
            boxes = boxes_from_seg(seg.render(), m, person_bodies)
            if not boxes and rng.random() < 0.8:
                continue            # 人の写っていない絵は 2 割だけ残す (背景の例)
            rgb.update_scene(w.data, cam, opt)
            img = rgb.render()
            name = f"s{a.seed}_{s:05d}_{cam}"
            Image.fromarray(img).save(os.path.join(a.out, "images", split, name + ".jpg"), quality=92)
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            level = "2F" if cam == "cam_nobita" else "1F"
            meta[split][name] = {
                "camera": cam, "pos": w.data.cam_xpos[cid].round(4).tolist(),
                "R": w.data.cam_xmat[cid].round(5).tolist(), "fovy": float(m.cam_fovy[cid]),
                "floor_z": w.layout["levels"][level]["z"],
                "persons": {str(k): {**w.person_state[k], "box": [int(v) for v in boxes[k][:4]],
                                     "pixels": int(boxes[k][4])} for k in boxes}}
            with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
                for k, (x0, y0, x1, y1, _) in boxes.items():
                    cls = 1 if poses.get(k) == "lie" else 0
                    counts["lying" if cls else "standing"] += 1
                    f.write(f"{cls} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0+1)/W:.6f} {(y1-y0+1)/H:.6f}\n")
            counts["images"] += 1
        if s % 100 == 0:
            print(f"scene {s}/{a.scenes} {counts}", flush=True)
    import json
    for split, d in meta.items():
        with open(os.path.join(a.out, f"meta_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f)
    with open(os.path.join(a.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {a.out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                f"names:\n  0: {CLASSES[0]}\n  1: {CLASSES[1]}\n")
    print("done", counts)


if __name__ == "__main__":
    main()
