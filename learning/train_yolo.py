"""MuJoCo で作ったデータで YOLOv8s を追加学習する (立っている人 / 横になっている人 の 2 クラス)。

公開の YOLOv8s (COCO で学習済み) から始める転移学習。RTX 3060 (12GB) で batch 16、640 px。
uv run python learning/train_yolo.py --epochs 40
出力: build/learning/yolo_nobita/ (weights/best.pt、results.csv、学習曲線の図)
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "build", "datasets", "nobita", "data.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolo_nobita")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "third_party", "yolov8s.pt"))
    model.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, device=0, workers=2,
                project=os.path.join(ROOT, "build", "learning"), name=a.name, exist_ok=True,
                patience=10, seed=0, plots=True,
                # 左右反転だけ使い、上下反転はしない (天井のカメラの絵が上下逆になることはない)
                fliplr=0.5, flipud=0.0, mosaic=1.0, hsv_v=0.4)
    metrics = model.val(split="test")
    print("test mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
