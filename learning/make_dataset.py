"""YOLO の追加学習用のデータを MuJoCo で作る (正解の枠は自動で付く)。

1 枚ごとに: 3 種類の人形 (子ども・家の大人・知らない大人、sim/scene.py の LOOKS) から 1〜3 人を選んで
家の中のランダムな場所・向き・姿勢 (立つ / 横になる) に置き、照明をランダムに点け消しし、
見守りカメラ 5 台とアヒルの頭のカメラ (ランダムな場所・向き) で撮る。
正解の枠: セグメンテーションの描画 (画素ごとにどの geom か) から人ごとの見えている画素を集めて外接する箱。
見えている画素が少なすぎる人 (MIN_PIXELS) は「難しすぎる」として、その絵ごと捨てずに枠を付けない (ignore しない)。

クラス (見た目 × 姿勢): 0 = child_standing, 1 = child_lying, 2 = adult_standing, 3 = adult_lying,
4 = stranger_standing, 5 = stranger_lying
出力: build/datasets/nobita3/{images,labels}/{train,val,test}/*.{jpg,txt} と data.yaml (Ultralytics の形式)
uv run python learning/make_dataset.py --scenes 2400 --seed 0
(以前の 2 クラス (person_standing / person_lying) のデータは build/datasets/nobita。models/yolo_nobita.pt はそれで学習した)
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
from scene import LOOKS  # noqa: E402
from world import World  # noqa: E402

W, H = 640, 480
MIN_PIXELS = 120
PERSON_LOOKS = ["child", "adult", "stranger"]       # 人 i の見た目 (1 人ずつ)
CLASSES = [f"{look}_{p}" for look in PERSON_LOOKS for p in ("standing", "lying")]
N_PEOPLE = ([1, 2, 3], [0.35, 0.45, 0.2])           # 1 枚に置く人数とその割合
# 人を置く部屋 (見守りカメラのある部屋と、カメラの無い廊下の奥、玄関の土間など)
ROOMS = ["ima", "ousetsu", "dk", "rouka", "nobita", "genkan"]


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


def sample_person(rng, w, room, scale=1.0):
    """部屋の中で家具と重ならない (x, y, yaw, pose, level, dz) を選ぶ。scale: 人形の大きさ (子どもは小さい)。
    dz: 部屋の床の高さ (玄関の土間は -0.2 m)。"""
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
            pts += [(x + k * scale * math.cos(yaw), y + k * scale * math.sin(yaw)) for k in (0.6, 1.2, 1.75)]
        if all(x0 + 0.15 <= px <= x1 - 0.15 and y0 + 0.15 <= py <= y1 - 0.15
               and not inside_furniture(lay, level, px, py, 0.15) for px, py in pts):
            return x, y, yaw, pose, level, r.get("z", 0.0)
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


def render_seg(r):
    """mujoco.Renderer のセグメンテーションの描画 (H, W, 2) [objid, objtype]。Renderer.render() と同じだが、
    半透明の窓ガラスの縁で色が混ざって geom の数より大きい番号になった画素は背景 (-1) にする
    (Renderer.render() はその画素で IndexError になる)。"""
    sc = r._scene
    flags = sc.flags.copy()
    sc.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
    sc.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
    img = np.empty((r.height, r.width, 3), np.uint8)
    if r._gl_context:
        r._gl_context.make_current()
    mujoco.mjr_render(r._rect, sc, r._mjr_context)
    mujoco.mjr_readPixels(img, None, r._rect, r._mjr_context)
    np.copyto(sc.flags, flags)
    ids = img[..., 0].astype(np.int64) + img[..., 1].astype(np.int64) * 256 + img[..., 2].astype(np.int64) * 65536
    lut = np.full((sc.ngeom + 2, 2), -1, np.int32)
    for g in sc.geoms[:sc.ngeom]:
        if g.segid != -1:
            lut[g.segid + 1] = (g.objid, g.objtype)
    out = lut[np.where(ids <= sc.ngeom, ids, sc.ngeom + 1)]
    return out[::-1] if r._gl_context else out     # Renderer.render() と同じく上下を直す


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "datasets", "nobita3"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    w = World(looks=PERSON_LOOKS)
    m = w.model
    n = len(PERSON_LOOKS)
    person_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"person{i}"): i for i in range(n)}
    low_rooms = {r["name"] for r in w.layout["rooms"] if r.get("z")}     # 床の低い部屋 (玄関の土間) にはアヒルを置かない
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
    counts = {"images": 0, **{c: 0 for c in CLASSES}}
    meta = {"train": {}, "val": {}, "test": {}}    # 絵ごとのカメラの位置・向きと人の正解 (評価に使う)
    for s in range(a.scenes):
        split = "train" if s % 10 < 8 else ("val" if s % 10 == 8 else "test")
        k = rng.choices(*N_PEOPLE)[0]
        who = set(rng.sample(range(n), k))
        rooms = rng.sample(ROOMS, n) if rng.random() < 0.6 else [rng.choice(ROOMS)] * n
        poses = {}
        for i, room in enumerate(rooms):
            p = sample_person(rng, w, room, LOOKS[PERSON_LOOKS[i]]["scale"]) if i in who else None
            if p is None:
                w.place_person(i, None, None, update_map=False)
                continue
            x, y, yaw, pose, level, dz = p
            w.place_person(i, x, y, yaw, pose, level, update_map=False, dz=dz)
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
                if w.room_of(dx, dy) not in low_rooms | {None} and not inside_furniture(w.layout, "1F", dx, dy, 0.15):
                    yaw = math.atan2(target["y"] - dy, target["x"] - dx) + rng.uniform(-0.5, 0.5)
                    w.duck.reset(dx, dy, yaw)
                    break
        mujoco.mj_forward(m, w.data)
        views = cams + (["head_camera"] if target is not None else [])
        for cam in views:
            seg.update_scene(w.data, cam, opt)
            boxes = {k: b for k, b in boxes_from_seg(render_seg(seg), m, person_bodies).items() if k in poses}
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
                "persons": {str(k): {**w.person_state[k], "look": PERSON_LOOKS[k],
                                     "box": [int(v) for v in boxes[k][:4]],
                                     "pixels": int(boxes[k][4])} for k in boxes}}
            with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
                for k, (x0, y0, x1, y1, _) in boxes.items():
                    cls = 2 * k + (1 if poses[k] == "lie" else 0)
                    counts[CLASSES[cls]] += 1
                    f.write(f"{cls} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0+1)/W:.6f} {(y1-y0+1)/H:.6f}\n")
            counts["images"] += 1
        if s % 100 == 0:
            print(f"scene {s}/{a.scenes} {counts}", flush=True)
    import json
    for split, d in meta.items():
        with open(os.path.join(a.out, f"meta_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f)
    with open(os.path.join(a.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {a.out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print("done", counts)


if __name__ == "__main__":
    main()
