"""カメラの絵から人を見つけ (YOLOv8)、床の上の位置と姿勢を推定する。

- 位置: 人の枠の下端の中央を通る視線と床 (z = 床の高さ) の交点 = 足元 (立っている人)
- 姿勢: 枠の上端の視線も床に落とし、下端の点との距離 L を見る。横になった人なら L は身長くらい
  (LYING_LENGTH 以下)、立っている人なら上端の視線は頭の上を通ってずっと遠くの床に落ちる (または床に届かない)。
  ただしカメラの真下近くに立つ人は縮んで見えて長さが短くなるので、頭の高さ (STAND_HEIGHT 以上) なら立っているとする。
  アヒルの頭のような低いカメラでは、体の上端の高さ (LYING_HEIGHT) で決める。
  上端が絵の外で切れていれば判定しない (unknown)。頭の高さ (足元の鉛直の面での高さ) は参考に出す
- 横になっている人の位置: 枠の下端と上端の床への投影の中点 (体の真ん中)
カメラの位置・向き・画角は MuJoCo のモデルから読む (事前の較正は不要という前提)。
"""
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHTS = os.path.join(ROOT, "third_party", "yolov8s.pt")              # 公開の YOLOv8s (COCO)
FINETUNED_WEIGHTS = os.path.join(ROOT, "models", "yolo_nobita.pt")          # この家で追加学習したもの (2 クラス)
POSTURE_CLASSES = {"person_standing": "standing", "person_lying": "lying"}
LYING_LENGTH = 2.6     # 枠の上端と下端の床への投影がこれより近ければ横になっている [m] (高いカメラ)
LYING_HEIGHT = 0.8     # 体の上端がこれより低ければ横になっている [m] (低いカメラ)
HIGH_CAMERA = 1.5      # これより高いカメラは天井のカメラとして扱う [m]
STAND_HEIGHT = 1.5     # 天井のカメラで、足元の鉛直面での頭の高さがこれ以上なら立っている [m]
MIN_SCORE = 0.4


class PersonDetector:
    """weights=None なら、追加学習したモデル (models/yolo_nobita.pt) があればそれを、なければ公開の YOLOv8s を使う。
    追加学習したモデルはクラスで姿勢を返す (枠の 6 番目の値)。公開のモデルでは姿勢は幾何で決める。"""

    def __init__(self, weights=None):
        from ultralytics import YOLO
        if weights is None:
            weights = os.environ.get("YOLO_WEIGHTS") or (
                FINETUNED_WEIGHTS if os.path.exists(FINETUNED_WEIGHTS) else YOLO_WEIGHTS)
        self.weights = weights
        self.yolo = YOLO(weights)
        self.classes = [k for k, v in self.yolo.names.items() if v == "person" or v in POSTURE_CLASSES]
        print(f"[perception] YOLO: {os.path.basename(weights)} classes={[self.yolo.names[k] for k in self.classes]}",
              flush=True)

    def detect(self, images):
        """images: RGB の絵のリスト。絵ごとに [(x1, y1, x2, y2, score, 姿勢 or None), ...] を返す。"""
        if not images:
            return []
        bgr = [np.ascontiguousarray(im[..., ::-1]) for im in images]   # YOLO は numpy を BGR とみなす
        results = self.yolo(bgr, verbose=False, classes=self.classes, conf=MIN_SCORE)
        out = []
        for r in results:
            out.append([tuple(float(v) for v in b.xyxy[0]) + (float(b.conf),
                                                               POSTURE_CLASSES.get(self.yolo.names[int(b.cls)]))
                        for b in r.boxes])
        return out


class CameraModel:
    """ピンホールカメラ。pos: 位置、R: 回転 (列が カメラの x=右, y=上, z=後ろ)、fovy [deg]、絵の大きさ。"""

    def __init__(self, pos, R, fovy, width, height):
        self.pos = np.asarray(pos, float)
        self.R = np.asarray(R, float).reshape(3, 3)
        self.w, self.h = width, height
        self.f = (height / 2) / math.tan(math.radians(fovy) / 2)

    @classmethod
    def from_mujoco(cls, model, data, name, width, height):
        import mujoco
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return cls(data.cam_xpos[cid].copy(), data.cam_xmat[cid].copy(), model.cam_fovy[cid], width, height)

    def ray(self, u, v):
        """絵の画素 (u 右, v 下) を通る視線の向き (世界座標、長さ 1)。"""
        d_cam = np.array([(u - self.w / 2) / self.f, -(v - self.h / 2) / self.f, -1.0])
        d = self.R @ d_cam
        return d / np.linalg.norm(d)

    def floor_point(self, u, v, floor_z):
        d = self.ray(u, v)
        if d[2] > -1e-6:
            return None
        t = (floor_z - self.pos[2]) / d[2]
        return self.pos + t * d

    def height_at(self, u, v, foot):
        """視線 (u, v) が、足元 foot を通りカメラの方を向いた鉛直の面と交わる高さ。"""
        d = self.ray(u, v)
        n = np.array([foot[0] - self.pos[0], foot[1] - self.pos[1], 0.0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            return None
        n /= nn
        denom = d @ n
        if denom < 1e-6:
            return None
        t = ((foot - self.pos) @ n) / denom
        return float((self.pos + t * d)[2])

    def project(self, p):
        """世界の点を絵の画素へ (カメラの後ろなら None)。"""
        q = self.R.T @ (np.asarray(p, float) - self.pos)
        if q[2] > -1e-6:
            return None
        return self.w / 2 + self.f * q[0] / -q[2], self.h / 2 - self.f * q[1] / -q[2]


def estimate_person(cam, box, floor_z=0.0, margin=4):
    """枠 (x1, y1, x2, y2, score) から {'x','y','posture','height','score'} を推定する (床に届かなければ None)。"""
    x1, y1, x2, y2, score = box[:5]
    learned = box[5] if len(box) > 5 else None    # 追加学習したモデルが返した姿勢
    uc = (x1 + x2) / 2
    foot = cam.floor_point(uc, y2, floor_z)
    if foot is None:
        return None
    top_cut = y1 <= margin
    height = None if top_cut else cam.height_at(uc, y1, foot)
    if height is not None:
        height -= floor_z
    posture = "unknown"
    length = None
    x, y = float(foot[0]), float(foot[1])
    if not top_cut and cam.pos[2] - floor_z < HIGH_CAMERA:
        # 低いカメラ (アヒルの頭): 体の上端の高さで決める (横になった人は 0.5 m もない)
        if height is not None:
            posture = "lying" if height < LYING_HEIGHT else "standing"
    elif not top_cut:
        far = cam.floor_point(uc, y1, floor_z)
        length = None if far is None else float(np.hypot(*(far[:2] - foot[:2])))
        # 長さが短くても、頭の高さが立った人の高さなら立っている (カメラの真下近くの人は縮んで見える)
        if length is not None and length < LYING_LENGTH and (height is None or height < STAND_HEIGHT):
            posture = "lying"
            x, y = float((foot[0] + far[0]) / 2), float((foot[1] + far[1]) / 2)
        else:
            posture = "standing"
    if learned:
        # 姿勢はモデルのクラスで決める。横なら体の真ん中 (枠の上下端の床投影の中点) を位置にする
        if learned != posture and learned == "lying" and not top_cut and cam.pos[2] - floor_z >= HIGH_CAMERA:
            far = cam.floor_point(uc, y1, floor_z)
            if far is not None and np.hypot(*(far[:2] - foot[:2])) < 3.0:
                x, y = float((foot[0] + far[0]) / 2), float((foot[1] + far[1]) / 2)
        elif learned == "standing":
            x, y = float(foot[0]), float(foot[1])
        posture = learned
    return {"x": round(x, 2), "y": round(y, 2), "posture": posture,
            "height": None if height is None else round(height, 2),
            "length": None if length is None else round(length, 2), "score": round(score, 2)}


# ---- 床に落ちている小さい物 (アヒルの頭のカメラ、learning/make_dataset_hazard.py で学習)
HAZARD_WEIGHTS = os.path.join(ROOT, "models", "yolo_hazard.pt")
HAZARD_MIN_SCORE = 0.35


class HazardDetector:
    """床の小さい危ない物 (ボタン電池・硬貨・ビー玉・小さいブロック・ハサミ) を見つける。重みが無ければ使えない (None を返す)。"""

    @classmethod
    def load(cls, weights=None):
        weights = weights or os.environ.get("HAZARD_WEIGHTS") or HAZARD_WEIGHTS
        if not os.path.exists(weights):
            print(f"[perception] 床の物の YOLO が無い ({weights})。見回りでは物を見つけられない", flush=True)
            return None
        return cls(weights)

    def __init__(self, weights):
        from ultralytics import YOLO
        self.yolo = YOLO(weights)
        print(f"[perception] 床の物の YOLO: {os.path.basename(weights)} classes={list(self.yolo.names.values())}",
              flush=True)

    def detect(self, images):
        """絵ごとに [(x1, y1, x2, y2, score, 種類), ...]。"""
        if not images:
            return []
        bgr = [np.ascontiguousarray(im[..., ::-1]) for im in images]
        out = []
        for r in self.yolo(bgr, verbose=False, conf=HAZARD_MIN_SCORE):
            out.append([tuple(float(v) for v in b.xyxy[0]) + (float(b.conf), self.yolo.names[int(b.cls)])
                        for b in r.boxes])
        return out


def estimate_object(cam, box, floor_z=0.0):
    """床の物の枠から床の上の位置 {'x','y','kind','score'} (枠の真ん中の視線と床の交点。物は低いので床とみなす)。"""
    x1, y1, x2, y2, score, kind = box[:6]
    p = cam.floor_point((x1 + x2) / 2, (y1 + y2) / 2, floor_z)
    if p is None:
        return None
    return {"kind": kind, "x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "score": round(score, 2)}
