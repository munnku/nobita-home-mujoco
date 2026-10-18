"""家 (sim/house.py) に Microduck と人をつないで 1 つの MJCF を書き出す。

python sim/scene.py <家のフォルダ> <出力 xml> [--duck x,y,yaw] [--person x,y,yaw,pose ...]
pose は stand (立つ) か lie (横になる)。人は動かない置物 (mocap なしの固定 body)。
"""
import argparse
import json
import math
import os

import mujoco

from house import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from duck import DUCK_XML  # noqa: E402
PERSON_DIR = os.path.join(ROOT, "assets", "person")


def quat_z(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def add_person(spec, name, x, y, yaw, pose="stand"):
    with open(os.path.join(PERSON_DIR, "person.json"), encoding="utf-8") as f:
        info = json.load(f)
    q = quat_z(yaw)
    z = 0.0
    if pose == "lie":
        # 足を原点に、体を水平に倒す (y 軸まわりに +90°: 頭が +x 側)、床から少し浮かせる
        q = quat_mul(q, [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
        z = 0.13
    body = spec.worldbody.add_body(name=name, pos=[x, y, z], quat=q)
    for i, p in enumerate(info["parts"]):
        if p["faces"] < 200 and "Low-Poly" in p["name"]:
            continue
        mesh = spec.add_mesh(name=f"{name}_m{i}", file=os.path.join(PERSON_DIR, p["obj"]))
        kw = {}
        if p["png"]:
            tex = spec.add_texture(name=f"{name}_t{i}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   file=os.path.join(PERSON_DIR, p["png"]))
            mat = spec.add_material(name=f"{name}_mat{i}")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            kw["material"] = mat.name
        body.add_geom(name=f"{name}_g{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                      contype=0, conaffinity=0, group=1, **kw)
    # ぶつかり判定は円柱 1 本 (ロボットが人を通り抜けないように)
    if pose == "lie":
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.15, 0.8, 0],
                      pos=[0.95, 0, 0], quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0],
                      rgba=[0, 0, 0, 0], group=3)
    else:
        body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.22, 0.9, 0],
                      pos=[0, -0.05, 0.9], rgba=[0, 0, 0, 0], group=3)


EYE_FOVY = 75.0          # 広角のカメラにする (元は 45°)
EYE_TILT_UP = math.radians(10)


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


def compose(house_dir, duck=(0.0, 0.0, 0.0), persons=(), ceiling=False):
    spec = mujoco.MjSpec.from_string(build(house_dir, ceiling=ceiling))
    spec.compiler.meshdir = ""
    if duck is not None:
        add_duck(spec, *duck)
    for i, p in enumerate(persons):
        add_person(spec, f"person{i}", *p)
    return spec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("house")
    ap.add_argument("out")
    ap.add_argument("--duck", default="0,0,0")
    ap.add_argument("--no-duck", action="store_true")
    ap.add_argument("--person", action="append", default=[])
    ap.add_argument("--ceiling", action="store_true")
    a = ap.parse_args()
    duck = None if a.no_duck else tuple(float(v) for v in a.duck.split(","))
    persons = []
    for s in a.person:
        v = s.split(",")
        persons.append((float(v[0]), float(v[1]), float(v[2]), v[3] if len(v) > 3 else "stand"))
    spec = compose(a.house, duck, persons, a.ceiling)
    model = spec.compile()
    xml = spec.to_xml()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {a.out}: nbody={model.nbody} ngeom={model.ngeom} ncam={model.ncam} nu={model.nu}")
