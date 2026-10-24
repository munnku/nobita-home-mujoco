"""学習前と学習後の YOLO を、同じテスト画像 (学習に使っていない場面) で比べる。

A: 公開の YOLOv8s (COCO の person) + 幾何の姿勢判定 (home/perception.py の estimate_person)
B: 2 クラスで追加学習した YOLOv8s (models/yolo_nobita.pt、クラスで姿勢)
C: 6 クラスで追加学習した YOLOv8s (models/yolo_nobita_who.pt、クラスで見た目 child / adult / stranger と姿勢)
正解の人ごとに: 見つけたか (IoU >= 0.5 の枠があるか)、姿勢が合っているか、(C は) だれかが合っているか。
誤検出 (正解にない枠) も数える。カメラの種類 (天井の見守りカメラ / アヒルの頭のカメラ)・姿勢・見た目ごとに集計し、
C の見分けの混同行列 (正解の見た目 × 答えた見た目、見逃しも) を出す。

uv run python learning/eval_yolo.py
出力: docs/learning/eval_who.{json,md}、who_confusion.png、who_results.png
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.perception import CameraModel, estimate_person, parse_class  # noqa: E402

CONF = 0.4
IOU = 0.5
LOOKS = ["child", "adult", "stranger"]
LOOK_JA = {"child": "子ども", "adult": "家の大人", "stranger": "不審者"}
CAM_JA = {"iot": "見守りカメラ", "head": "アヒルの頭"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def predict(model, images, metas):
    """絵ごとに [(box, posture, who)] を返す。追加学習したモデルはクラスから、公開のモデルは幾何で姿勢を決める。"""
    res = model([np.ascontiguousarray(im[..., ::-1]) for im in images], verbose=False, conf=CONF)
    out = []
    for r, meta in zip(res, metas):
        preds = []
        for b in r.boxes:
            kind = parse_class(model.names[int(b.cls)])
            if kind is None:
                continue
            box = [float(v) for v in b.xyxy[0]]
            posture, who = kind
            if posture is None:
                cam = CameraModel(meta["pos"], meta["R"], meta["fovy"], 640, 480)
                est = estimate_person(cam, (*box, float(b.conf)), meta["floor_z"])
                posture = est["posture"] if est else "unknown"
            preds.append((box, posture, who))
        out.append(preds)
    return out


def evaluate(model, data_dir, metas, batch=16):
    stats = defaultdict(lambda: {"n": 0, "found": 0, "posture_ok": 0, "who_ok": 0})
    conf = {k: np.zeros((3, 4), int) for k in ("all", "iot", "head")}   # 行: 正解の見た目、列: 答え (+ 見逃し)
    fp = defaultdict(int)
    names = sorted(metas)
    for k in range(0, len(names), batch):
        chunk = names[k:k + batch]
        imgs = [np.asarray(Image.open(os.path.join(data_dir, "images", "test", n + ".jpg")).convert("RGB"))
                for n in chunk]
        preds = predict(model, imgs, [metas[n] for n in chunk])
        for n, pr in zip(chunk, preds):
            meta = metas[n]
            kind = "head" if meta["camera"] == "head_camera" else "iot"
            used = set()
            for p in meta["persons"].values():
                gt_post = "lying" if p["pose"] == "lie" else "standing"
                look = p.get("look", "adult")
                best, bj = 0.0, -1
                for j, (box, _, _) in enumerate(pr):
                    if j in used:
                        continue
                    v = iou(p["box"], box)
                    if v > best:
                        best, bj = v, j
                found = best >= IOU
                if found:
                    used.add(bj)
                # 見逃しは 4 列目。見つけたが見た目を答えない (2 クラスのモデル) ときは数えない
                col = 3 if not found else (LOOKS.index(pr[bj][2]) if pr[bj][2] in LOOKS else None)
                if col is not None:
                    for c in ("all", kind):
                        conf[c][LOOKS.index(look), col] += 1
                for key in ((kind, gt_post, "all"), (kind, gt_post, look), ("all", "all", look), ("all", "all", "all")):
                    s = stats[key]
                    s["n"] += 1
                    if found:
                        s["found"] += 1
                        s["posture_ok"] += pr[bj][1] == gt_post
                        s["who_ok"] += pr[bj][2] == look
            fp[kind] += len(pr) - len(used)
    return {"/".join(k): v for k, v in stats.items()}, dict(fp), {k: v.tolist() for k, v in conf.items()}


def pct(s, key):
    return f"{s[key] / s['n']:.1%}" if s["n"] else "-"


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    who = [(label, r) for label, r in results.items() if r["who"]]
    if who:
        label, r = who[-1]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, cam in zip(axes, ("iot", "head")):
            m = np.array(r["confusion"][cam], float)
            rate = m / np.maximum(m.sum(1, keepdims=True), 1)
            ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(4):
                    ax.text(j, i, f"{rate[i, j]:.0%}\n({int(m[i, j])})", ha="center", va="center", fontsize=9,
                            color="white" if rate[i, j] > 0.6 else "black")
            ax.set_xticks(range(4), [LOOK_JA[x] for x in LOOKS] + ["見逃し"])
            ax.set_yticks(range(3), [LOOK_JA[x] for x in LOOKS])
            ax.set_xlabel("YOLO の答え")
            ax.set_ylabel("正解")
            ax.set_title(CAM_JA[cam])
        fig.suptitle(f"見分けの混同行列 ({label})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "who_confusion.png"), dpi=120)
        plt.close(fig)
    # 見た目ごとの「見つけて姿勢も正しい」割合を方法ごとに並べる
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    width = 0.8 / max(1, len(results))
    for ax, cam in zip(axes, ("iot", "head")):
        for k, (label, r) in enumerate(results.items()):
            vals = []
            for look in LOOKS:
                ss = [r["stats"].get(f"{cam}/{p}/{look}") for p in ("standing", "lying")]
                n = sum(s["n"] for s in ss if s)
                vals.append(sum(s["posture_ok"] for s in ss if s) / n if n else 0)
            ax.bar(np.arange(3) + (k - (len(results) - 1) / 2) * width, vals, width, label=label)
        ax.set_xticks(range(3), [LOOK_JA[x] for x in LOOKS])
        ax.set_title(f"{CAM_JA[cam]}: 見つけて姿勢も正しい割合")
        ax.set_ylim(0, 1)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "who_results.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "build", "datasets", "nobita3"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "yolo_nobita_who.pt"))
    ap.add_argument("--posture-model", default=os.path.join(ROOT, "models", "yolo_nobita.pt"))
    ap.add_argument("--baseline", default=os.path.join(ROOT, "third_party", "yolov8s.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "learning"))
    a = ap.parse_args()
    from ultralytics import YOLO
    with open(os.path.join(a.data, "meta_test.json"), encoding="utf-8") as f:
        metas = json.load(f)
    results = {}
    for label, path, who in (("A: 公開 YOLOv8s + 幾何の姿勢判定", a.baseline, False),
                             ("B: 追加学習 2 クラス (yolo_nobita)", a.posture_model, False),
                             ("C: 追加学習 6 クラス (yolo_nobita_who)", a.model, True)):
        if not os.path.exists(path):
            print("skip", path)
            continue
        stats, fp, conf = evaluate(YOLO(path), a.data, metas)
        results[label] = {"weights": os.path.relpath(path, ROOT).replace("\\", "/"), "who": who,
                          "stats": stats, "false_positives": fp, "confusion": conf}
        print(label, json.dumps(stats["all/all/all"]), fp, flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "eval_who.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    n_people = next(iter(results.values()))["stats"]["all/all/all"]["n"] if results else 0
    lines = [f"テスト画像 {len(metas)} 枚・写っている人 {n_people} 人 (学習に使っていない場面、3 種類の人形)。"
             f"見つけた = IoU {IOU} 以上、確信度 {CONF} 以上\n",
             "| 方法 | カメラ | 姿勢 | 見た目 | 人数 | 見つけた | 姿勢も正しい | だれかも正しい |",
             "|---|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        st = r["stats"]
        for cam in ("iot", "head"):
            for post in ("standing", "lying"):
                for look in ["all"] + LOOKS:
                    s = st.get(f"{cam}/{post}/{look}")
                    if not s:
                        continue
                    lines.append(f"| {label} | {CAM_JA[cam]} | {'立' if post == 'standing' else '横'} | "
                                 f"{'全員' if look == 'all' else LOOK_JA[look]} | {s['n']} | {pct(s, 'found')} | "
                                 f"{pct(s, 'posture_ok')} | {pct(s, 'who_ok') if r['who'] else '-'} |")
        s = st["all/all/all"]
        lines.append(f"| {label} | **全体** | | | {s['n']} | {pct(s, 'found')} | {pct(s, 'posture_ok')} | "
                     f"{pct(s, 'who_ok') if r['who'] else '-'} |")
        fp = r["false_positives"]
        lines.append(f"| {label} | 誤検出 (正解に無い枠の数) | | | | 見守りカメラ {fp.get('iot', 0)}・"
                     f"アヒルの頭 {fp.get('head', 0)} | | |")
    for label, r in results.items():
        if not r["who"]:
            continue
        for cam in ("all", "iot", "head"):
            m = r["confusion"][cam]
            lines += ["", f"見分けの混同行列 ({label}、{'全カメラ' if cam == 'all' else CAM_JA[cam]}。"
                      "行 = 正解、列 = YOLO の答え)\n",
                      "| 正解 \\ 答え | 子ども | 家の大人 | 不審者 | 見逃し |", "|---|---|---|---|---|"]
            lines += [f"| {LOOK_JA[LOOKS[i]]} | " + " | ".join(str(v) for v in m[i]) + " |" for i in range(3)]
    with open(os.path.join(a.out, "eval_who.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    plot(results, a.out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
