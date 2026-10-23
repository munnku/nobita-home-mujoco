"""床に落ちている小さい危ない物 (誤飲・けがのもと)。家の MjSpec に足し、場面で置く。

どれも当たり判定のある自由な body (freejoint)。初めは家の外 (南の庭) に置いておき、place() で床に置く。
大きさは実物に近くする (ボタン電池 CR2032 は直径 20 mm、10 円玉は 23.5 mm、ビー玉は 17 mm)。
YOLO のクラスはこの順 (learning/make_dataset_hazard.py)。
"""
import math

import mujoco
import numpy as np

KINDS = {
    #  名前            ラベル          形          大きさ [m]                      色 (rgba)                質量 [kg]
    "button_battery": ("ボタン電池", "cylinder", (0.010, 0.0016),             (0.80, 0.80, 0.83, 1.0), 0.003),
    "coin":           ("硬貨",       "cylinder", (0.01175, 0.0008),           (0.72, 0.45, 0.25, 1.0), 0.0045),
    "marble":         ("ビー玉",     "sphere",   (0.0085,),                   (0.35, 0.65, 0.95, 0.75), 0.005),
    "toy_block":      ("小さいブロック", "box",  (0.016, 0.008, 0.0057),     (0.90, 0.15, 0.15, 1.0), 0.002),
    "scissors":       ("ハサミ",     "box",      (0.085, 0.025, 0.004),       (0.95, 0.75, 0.10, 1.0), 0.05),
}
CLASSES = list(KINDS)
HIDE_Y = -2.2          # 片付けた物を置いておく所 (家の南の庭の上)
HIDE_Z = -0.43


def label(kind):
    return KINDS[kind][0]


def add_objects(spec, counts):
    """counts: {種類: 個数}。body の名前は obj_<種類>_<番号>。"""
    names = []
    k = 0
    for kind, n in counts.items():
        lab, shape, size, rgba, mass = KINDS[kind]
        for i in range(n):
            name = f"obj_{kind}_{i}"
            b = spec.worldbody.add_body(name=name, pos=[-1.0 - 0.1 * k, HIDE_Y, HIDE_Z])
            b.add_freejoint(name=f"{name}_free")
            g = b.add_geom(name=f"{name}_geom", mass=mass, rgba=list(rgba), friction=[0.9, 0.01, 0.001],
                           condim=4, group=1)
            g.type = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                      "box": mujoco.mjtGeom.mjGEOM_BOX}[shape]
            g.size = list(size) + [0.0] * (3 - len(size))
            if kind == "scissors":       # 持ち手の輪 (黒)
                h = b.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.022, 0.004, 0],
                               pos=[-0.08, 0, 0], rgba=[0.1, 0.1, 0.1, 1], mass=0.01, group=1)
                h.contype = h.conaffinity = 1
            names.append(name)
            k += 1
    return names


class Objects:
    """コンパイルしたモデルの上で物を置く・片付ける。"""

    def __init__(self, model, data, names):
        self.m, self.d = model, data
        self.names = names
        self.body = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
        self.qadr = {n: model.jnt_qposadr[model.body_jntadr[b]] for n, b in self.body.items()}
        self.vadr = {n: model.jnt_dofadr[model.body_jntadr[b]] for n, b in self.body.items()}
        self.state = {n: None for n in names}      # None (片付いている) か (x, y)

    @staticmethod
    def kind(name):
        return name[4:].rsplit("_", 1)[0]

    def place(self, name, x, y, yaw=0.0, floor_z=0.0):
        kind = self.kind(name)
        shape, size = KINDS[kind][1], KINDS[kind][2]
        h = size[0] if shape == "sphere" else (size[1] if shape == "cylinder" else size[2])
        q = self.qadr[name]
        self.d.qpos[q:q + 3] = [x, y, floor_z + h + 0.001]
        self.d.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.d.qvel[self.vadr[name]:self.vadr[name] + 6] = 0
        self.state[name] = (x, y)

    def hide(self, name):
        k = self.names.index(name)
        q = self.qadr[name]
        self.d.qpos[q:q + 3] = [-1.0 - 0.1 * k, HIDE_Y, HIDE_Z]
        self.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        self.d.qvel[self.vadr[name]:self.vadr[name] + 6] = 0
        self.state[name] = None

    def pos(self, name):
        return np.array(self.d.xpos[self.body[name]])

    def on_floor(self):
        return {n: tuple(self.pos(n)[:2]) for n, s in self.state.items() if s is not None}
