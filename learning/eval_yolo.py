"""学習前と学習後の YOLO を、同じテスト画像 (学習に使っていない場面) で比べる。

A: 公開の YOLOv8s (COCO の person) + 幾何の姿勢判定 (home/perception.py の estimate_person)
B: このデータで追加学習した YOLOv8s (クラスで姿勢: 0 = 立っている、1 = 横になっている)
正解の人ごとに: 見つけたか (IoU >= 0.5 の枠があるか)、姿勢が合っているか。誤検出 (正解にない枠) も数える。
カメラの種類 (天井の見守りカメラ / アヒルの頭のカメラ) と姿勢ごとに集計する。

uv run python learning/eval_yolo.py --model build/learning/yolo_nobita/weights/best.pt
出力: build/learning/eval.json と eval.md
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.perception import CameraModel, estimate_person  # noqa: E402

CONF = 0.4
IOU = 0.5


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def predict(model, images, finetuned, metas):
    """絵ごとに [(box, posture)] を返す。"""
    res = model([np.ascontiguousarray(im[..., ::-1]) for im in images], verbose=False, conf=CONF)
    out = []
    for r, meta in zip(res, metas):
        preds = []
        for b in r.boxes:
            cls = int(b.cls)
            box = [float(v) for v in b.xyxy[0]]
            if finetuned:
                preds.append((box, "lying" if cls == 1 else "standing"))
            elif model.names[cls] == "person":
                cam = CameraModel(meta["pos"], meta["R"], meta["fovy"], 640, 480)
                est = estimate_person(cam, (*box, float(b.conf)), meta["floor_z"])
                preds.append((box, est["posture"] if est else "unknown"))
        out.append(preds)
    return out


def evaluate(model, finetuned, data_dir, metas, batch=16):
    from collections import defaultdict
    stats = defaultdict(lambda: {"n": 0, "found": 0, "posture_ok": 0})
    fp = defaultdict(int)
    names = sorted(metas)
    for k in range(0, len(names), batch):
        chunk = names[k:k + batch]
        imgs = [np.asarray(Image.open(os.path.join(data_dir, "images", "test", n + ".jpg")).convert("RGB"))
                for n in chunk]
        preds = predict(model, imgs, finetuned, [metas[n] for n in chunk])
        for n, pr in zip(chunk, preds):
            meta = metas[n]
            kind = "head" if meta["camera"] == "head_camera" else "iot"
            used = set()
            for p in meta["persons"].values():
                gt_post = "lying" if p["pose"] == "lie" else "standing"
                key = (kind, gt_post)
                stats[key]["n"] += 1
                best, bj = 0.0, -1
                for j, (box, _) in enumerate(pr):
                    if j in used:
                        continue
                    v = iou(p["box"], box)
                    if v > best:
                        best, bj = v, j
                if best >= IOU:
                    used.add(bj)
                    stats[key]["found"] += 1
                    if pr[bj][1] == gt_post:
                        stats[key]["posture_ok"] += 1
            fp[kind] += len(pr) - len(used)
    return {f"{k[0]}/{k[1]}": v for k, v in stats.items()}, dict(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "build", "datasets", "nobita"))
    ap.add_argument("--model", default=os.path.join(ROOT, "build", "learning", "yolo_nobita", "weights", "best.pt"))
    ap.add_argument("--baseline", default=os.path.join(ROOT, "third_party", "yolov8s.pt"))
    a = ap.parse_args()
    from ultralytics import YOLO
    with open(os.path.join(a.data, "meta_test.json"), encoding="utf-8") as f:
        metas = json.load(f)
    results = {}
    for label, path, ft in (("A: 公開 YOLOv8s + 幾何の姿勢判定", a.baseline, False),
                            ("B: 追加学習した YOLOv8s (2 クラス)", a.model, True)):
        if not os.path.exists(path):
            print("skip", path)
            continue
        stats, fp = evaluate(YOLO(path), ft, a.data, metas)
        results[label] = {"stats": stats, "false_positives": fp}
        print(label, json.dumps(results[label], ensure_ascii=False))
    out = os.path.join(ROOT, "build", "learning")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    lines = [f"テスト画像 {len(metas)} 枚 (学習に使っていない場面)。見つけた = IoU {IOU} 以上、確信度 {CONF} 以上\n",
             "| 方法 | カメラ | 姿勢 | 人数 | 見つけた | 姿勢も正しい |", "|---|---|---|---|---|---|"]
    for label, r in results.items():
        for key in sorted(r["stats"]):
            s = r["stats"][key]
            cam, post = key.split("/")
            lines.append(f"| {label} | {'見守りカメラ' if cam == 'iot' else 'アヒルの頭'} | "
                         f"{'横' if post == 'lying' else '立'} | {s['n']} | {s['found']/s['n']:.1%} | "
                         f"{s['posture_ok']/s['n']:.1%} |")
        lines.append(f"| {label} | 誤検出 | | | {r['false_positives']} | |")
    with open(os.path.join(out, "eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
