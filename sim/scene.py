"""家 (sim/house.py) に Microduck と人をつないだ MjSpec を作る。

人は mocap の body (物理で動かず、位置と向きを書けば瞬間移動する置物)。立つ・横になる・座るの姿勢は
体ごと回して表す (メッシュは立った姿勢の 1 つだけ)。当たり判定は体の軸に沿ったカプセル 1 本。
"""
import json
import math
import os

import mujoco

from duck import DUCK_XML
from house import build_with_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON_DIR = os.path.join(ROOT, "assets", "person")
EYE_FOVY = 75.0                    # アヒルの頭のカメラを広角にする (元は 45°)
EYE_TILT_UP = math.radians(10)


def quat_z(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def person_pose(x, y, yaw, pose="stand", floor_z=0.0):
    """人の body の (pos, quat)。(x, y) は足元、yaw は顔の向き (横になると頭の向き)。"""
    if pose == "lie":
        # 足を (x, y) に置き、体を y 軸まわりに +90° 倒す (頭が yaw の向き)。床から体の厚みの半分浮かす
        q = quat_mul(quat_z(yaw), [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
        return [x, y, floor_z + 0.13], q
    return [x, y, floor_z], quat_z(yaw)


def add_person_assets(spec):
    with open(os.path.join(PERSON_DIR, "person.json"), encoding="utf-8") as f:
        info = json.load(f)
    parts = []
    for i, p in enumerate(info["parts"]):
        if "Low-Poly" in p["name"]:      # 目の周りの小さな部品 (透明用) は使わない
            continue
        mesh = spec.add_mesh(name=f"person_m{i}", file=os.path.join(PERSON_DIR, p["obj"]))
        mat = None
        if p["png"]:
            tex = spec.add_texture(name=f"person_t{i}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   file=os.path.join(PERSON_DIR, p["png"]))
            mat = spec.add_material(name=f"person_mat{i}")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        parts.append((mesh.name, mat.name if mat else None))
    return parts


def add_person(spec, parts, name, x=0.0, y=-5.0, yaw=0.0, pose="stand"):
    pos, quat = person_pose(x, y, yaw, pose)
    body = spec.worldbody.add_body(name=name, pos=pos, quat=quat, mocap=True)
    for k, (mesh, mat) in enumerate(parts):
        kw = {"material": mat} if mat else {}
        body.add_geom(name=f"{name}_g{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                      contype=0, conaffinity=0, group=1, **kw)
    # 当たり判定: 足元から頭までのカプセル (立てば縦、横になれば横)
    body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.18, 0.7, 0],
                  pos=[0, -0.05, 0.9], rgba=[0, 0, 0, 0], group=3)


def add_duck(spec, x=0.0, y=0.0, yaw=0.0):
    """Microduck をつなぎ、頭のカメラ head_camera を MuJoCo の向き (-z が視線、y が上) に直す。

    元の MJCF のカメラは Onshape のフレームのまま (立った姿勢で後ろ上を向き、画像が 90° 回っている)。
    その向きから見た「前・上・右」への回転 (軸 (1,-1,0) まわりに 180°) をかけ、少し上向きにする。
    """
    robot = mujoco.MjSpec.from_file(DUCK_XML)
    cam = robot.camera("head_camera")
    q_rel = [0.0, math.sqrt(0.5), -math.sqrt(0.5), 0.0]
    q_tilt = [math.cos(EYE_TILT_UP / 2), math.sin(EYE_TILT_UP / 2), 0.0, 0.0]
    cam.quat = quat_mul(quat_mul(list(cam.quat), q_rel), q_tilt)
    cam.fovy = EYE_FOVY
    frame = spec.worldbody.add_frame(pos=[x, y, 0.0], quat=quat_z(yaw))
    spec.attach(robot, frame=frame, prefix="")


def compose(house_dir, duck=True, n_persons=0):
    """(MjSpec, 扉の開いた値) を返す。人は家の外 (y=-5) に置いておき、場面に合わせて動かす。"""
    xml, door_open = build_with_info(house_dir)
    spec = mujoco.MjSpec.from_string(xml)
    if duck:
        add_duck(spec)
    if n_persons:
        parts = add_person_assets(spec)
        for i in range(n_persons):
            add_person(spec, parts, f"person{i}", x=-1.5 - i, y=-2.5)
    return spec, door_open
