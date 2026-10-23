"""床に落ちている小さい危ない物を、アヒルの頭のカメラで見つける YOLO のデータを MuJoCo で作る (正解の枠は自動)。

1 枚ごとに: 1 階のランダムな部屋の床にアヒルを置き、首を下に向け (見回りのときの姿勢)、カメラに入る床の上に
小さい物 (sim/objects.py: ボタン電池・硬貨・ビー玉・小さいブロック・ハサミ) を 0〜6 個置いて撮る。
照明・物の向き・首の角度・アヒルの向きもランダム。物の無い絵 (床だけ) も混ぜる (誤検出を減らす)。
正解の枠: セグメンテーションの描画から物ごとの見えている画素の外接箱 (MIN_PIXELS 未満は付けない)。

クラス: sim/objects.py の CLASSES の順
出力: build/datasets/hazard/{images,labels}/{train,val,test}/ と data.yaml
uv run python learning/make_dataset_hazard.py --scenes 4000 --seed 0
"""
import argparse
import json
import math
import os
import random
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from duck import DEFAULT_POSE  # noqa: E402
from make_dataset import inside_furniture  # noqa: E402
from objects import CLASSES, Objects  # noqa: E402
from world import World  # noqa: E402

W, H = 640, 480
MIN_PIXELS = 12
ROOMS = ["ima", "ousetsu", "dk", "rouka", "senmen", "toilet", "yokushitsu"]
PER_KIND = 2          # 種類ごとに用意する個数 (1 枚に同じ種類が 2 個まで)
NECK = 5              # 関節の順で首 (neck_pitch) の番号


def free_spot(rng, w, room, margin):
    lay = w.layout
    r = next(r for r in lay["rooms"] if r["name"] == room)
    x0, x1, y0, y1 = (v * lay["module"] for v in r["rect"])
    for _ in range(200):
        x, y = rng.uniform(x0 + margin, x1 - margin), rng.uniform(y0 + margin, y1 - margin)
        if not inside_furniture(lay, "1F", x, y, margin):
            return x, y
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "datasets", "hazard"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    w = World(n_persons=0, objects={k: PER_KIND for k in CLASSES})
    m, d = w.model, w.data
    obj = w.objects
    geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == obj.body[n]] for n in obj.names}
    rgb = mujoco.Renderer(m, H, W)
    seg = mujoco.Renderer(m, H, W)
    seg.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    opt.geomgroup[3] = 0
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)
    counts = {"images": 0, **{c: 0 for c in CLASSES}}
    meta = {"train": {}, "val": {}, "test": {}}
    qa = w.duck.qpos_idx
    for s in range(a.scenes):
        split = "train" if s % 10 < 8 else ("val" if s % 10 == 8 else "test")
        room = rng.choice(ROOMS)
        p = free_spot(rng, w, room, 0.2)
        if p is None:
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        w.duck.reset(p[0], p[1], yaw)
        q = DEFAULT_POSE.copy()
        q[NECK] += rng.uniform(-1.0, -0.4)               # 首を下に向ける (見回りの姿勢)
        d.qpos[qa] = q
        for n in obj.names:
            obj.hide(n)
        # カメラの前の床 (0.25〜1.8 m、左右 ±35°) に物を置く。0 個の絵も 1 割
        n_obj = 0 if rng.random() < 0.1 else rng.randint(1, 6)
        names = rng.sample(obj.names, n_obj)
        lay = w.layout
        for n in names:
            for _ in range(30):
                r_, ang = rng.uniform(0.25, 1.8), yaw + rng.uniform(-0.6, 0.6)
                x, y = p[0] + r_ * math.cos(ang), p[1] + r_ * math.sin(ang)
                if w.room_of(x, y) and not inside_furniture(lay, "1F", x, y, 0.03):
                    obj.place(n, x, y, rng.uniform(-math.pi, math.pi), floor_z=0.0)
                    break
        for lr in lay.get("lights", []):
            w.set_light(lr, rng.random() < 0.7)
        mujoco.mj_forward(m, d)
        seg.update_scene(d, "head_camera", opt)
        sg = seg.render()
        ids = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1)
        boxes = []
        for n in names:
            mask = np.isin(ids, geoms[n])
            if int(mask.sum()) < MIN_PIXELS:
                continue
            ys, xs = np.nonzero(mask)
            boxes.append((CLASSES.index(Objects.kind(n)), xs.min(), ys.min(), xs.max(), ys.max(), n))
        rgb.update_scene(d, "head_camera", opt)
        img = rgb.render()
        name = f"h{a.seed}_{s:05d}"
        Image.fromarray(img).save(os.path.join(a.out, "images", split, name + ".jpg"), quality=92)
        with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
            for c, x0, y0, x1, y1, _ in boxes:
                counts[CLASSES[c]] += 1
                f.write(f"{c} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0+1)/W:.6f} {(y1-y0+1)/H:.6f}\n")
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        meta[split][name] = {"pos": d.cam_xpos[cid].round(4).tolist(), "R": d.cam_xmat[cid].round(5).tolist(),
                             "fovy": float(m.cam_fovy[cid]), "room": room,
                             "objects": [{"kind": CLASSES[c], "box": [int(v) for v in (x0, y0, x1, y1)],
                                          "xy": [round(v, 3) for v in obj.pos(n)[:2]]}
                                         for c, x0, y0, x1, y1, n in boxes]}
        counts["images"] += 1
        if s % 200 == 0:
            print(f"scene {s}/{a.scenes} {counts}", flush=True)
    for split, dd in meta.items():
        with open(os.path.join(a.out, f"meta_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(dd, f)
    with open(os.path.join(a.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {a.out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
    print("done", counts)


if __name__ == "__main__":
    main()
