"""MuJoCo で作ったデータで YOLOv8s を追加学習する。

既定は 6 クラス (見た目 child / adult / stranger × 姿勢 standing / lying、learning/make_dataset.py の build/datasets/people3)。
以前の 2 クラス (立っている人 / 横になっている人) は --data build/datasets/people/data.yaml --name yolo_person。
公開の YOLOv8s (COCO で学習済み) から始める転移学習。RTX 3060 (12GB) で batch 16、640 px。
uv run python learning/train_yolo.py --epochs 40
出力: build/learning/<name>/ (weights/best.pt、results.csv、学習曲線の図)。best.pt を models/<name>.pt に写す
"""
import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "build", "datasets", "people3", "data.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolo_person_who")
    ap.add_argument("--workers", type=int, default=2, help="絵を読む子プロセスの数 (RAM が足りなければ 1)")
    ap.add_argument("--resume", action="store_true", help="途中で止まった学習を weights/last.pt から続ける")
    a = ap.parse_args()
    from ultralytics import YOLO
    if a.resume:
        model = YOLO(os.path.join(ROOT, "build", "learning", a.name, "weights", "last.pt"))
        model.train(resume=True, workers=a.workers)
    else:
        model = YOLO(os.path.join(ROOT, "third_party", "yolov8s.pt"))
        model.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, device=0, workers=a.workers,
                    project=os.path.join(ROOT, "build", "learning"), name=a.name, exist_ok=True,
                    patience=10, seed=0, plots=True,
                    # 左右反転だけ使い、上下反転はしない (天井のカメラの絵が上下逆になることはない)
                    fliplr=0.5, flipud=0.0, mosaic=1.0, hsv_v=0.4)
    metrics = model.val(split="test")
    print("test mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
    best = os.path.join(ROOT, "build", "learning", a.name, "weights", "best.pt")
    shutil.copy(best, os.path.join(ROOT, "models", a.name + ".pt"))
    print("copied", best, "->", os.path.join("models", a.name + ".pt"))


if __name__ == "__main__":
    main()
