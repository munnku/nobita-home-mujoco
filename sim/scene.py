"""家 (sim/house.py) に Microduck と人をつないだ MjSpec を作る。

人は mocap の body (物理で動かず、位置と向きを書けば瞬間移動する置物)。立つ・横になる・座るの姿勢は
体ごと回して表す (メッシュは立った姿勢の 1 つだけ)。当たり判定は体の軸に沿ったカプセル 1 本。
見た目 (look) は 3 種類 (LOOKS): 家の大人 (adult)、子ども (child、大人の形を縮める)、知らない大人 (stranger、黒っぽい服に帽子とマスク)。
"""
import json
import math
import os

import mujoco
import numpy as np

from duck import DUCK_XML
from house import build_with_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON_DIR = os.path.join(ROOT, "assets", "person")
EYE_FOVY = 75.0                    # アヒルの頭のカメラを広角にする (元は 45°)
EYE_TILT_UP = math.radians(10)
# 人の見た目。scale は大人の人形 (Gazebo の person_standing、1.9 m) に掛ける大きさ。
# shirt / pants は服の色 (テクスチャは使わず色だけ。元の T シャツ・ジーンズのテクスチャは暗く、色を掛けても明るくならない)。
# shorts: ズボンの膝から下を肌の色にする (半ズボン。体のメッシュは服の下の脚が無いので、ズボンを切ると脚が消える)
LOOKS = {
    "adult": {"scale": 1.0, "shirt": [1.0, 0.55, 0.6, 1], "pants": [0.4, 0.24, 0.16, 1]},        # ママ: 桃色のシャツ・茶色
    "child": {"scale": 0.68, "shirt": [1.0, 0.86, 0.12, 1], "pants": [0.09, 0.12, 0.36, 1],      # のび太: 黄色のシャツ・紺の半ズボン
              "shorts": True},
    "stranger": {"scale": 1.0, "shirt": [0.16, 0.16, 0.18, 1], "pants": [0.07, 0.07, 0.09, 1],   # 不審者: 黒っぽい服・帽子・マスク
                 "hat": True, "mask": True},
}
LIE_Z = 0.13          # 横になったとき床から浮かす高さ (大人の体の厚みの半分) [m]。見た目の scale を掛ける
KNEE_Z = 0.62         # 半ズボンの裾の高さ (大人の形で) [m]
SKIN = [0.86, 0.68, 0.56, 1]


def quat_z(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def person_pose(x, y, yaw, pose="stand", floor_z=0.0, scale=1.0):
    """人の body の (pos, quat)。(x, y) は足元、yaw は顔の向き (横になると頭の向き)。scale は見た目の大きさ。"""
    if pose == "lie":
        # 足を (x, y) に置き、体を y 軸まわりに +90° 倒す (頭が yaw の向き)。床から体の厚みの半分浮かす
        q = quat_mul(quat_z(yaw), [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
        return [x, y, floor_z + LIE_Z * scale], q
    return [x, y, floor_z], quat_z(yaw)


def _split_obj(path, z):
    """obj を高さ z で上下に分ける。[(頂点, 三角形) (z より上), (頂点, 三角形) (下)]。"""
    verts, faces = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    verts, faces = np.array(verts), np.array(faces)
    upper = verts[faces][..., 2].mean(axis=1) > z
    out = []
    for sel in (faces[upper], faces[~upper]):
        used, f = np.unique(sel, return_inverse=True)
        out.append((verts[used], f.reshape(-1, 3)))
    return out


def add_person_assets(spec, looks=("adult",)):
    """見た目ごとの部品 {look: [(mesh, material), ...]} を足す (テクスチャは見た目どうしで共有)。"""
    with open(os.path.join(PERSON_DIR, "person.json"), encoding="utf-8") as f:
        info = json.load(f)
    textures = {}
    out = {}
    for look in sorted(set(looks)):
        L = LOOKS[look]
        s = L["scale"]
        parts = []
        for i, p in enumerate(info["parts"]):
            if "Low-Poly" in p["name"]:      # 目の周りの小さな部品 (透明用) は使わない
                continue
            role = "shirt" if "tshirt" in p["name"] else ("pants" if "jeans" in p["name"] else None)
            obj = os.path.join(PERSON_DIR, p["obj"])
            if role == "pants" and L.get("shorts"):
                pieces = [(f"{i}{k}", v, fc, c) for k, ((v, fc), c) in
                          enumerate(zip(_split_obj(obj, KNEE_Z), (L["pants"], SKIN)))]
            else:
                pieces = [(f"{i}", None, None, L[role] if role else None)]
            for key, v, fc, rgba in pieces:
                if v is None:
                    mesh = spec.add_mesh(name=f"person_{look}_m{key}", file=obj)
                else:
                    mesh = spec.add_mesh(name=f"person_{look}_m{key}", uservert=v.ravel().tolist(),
                                         userface=fc.ravel().tolist())
                mesh.scale = [s, s, s]
                mat = spec.add_material(name=f"person_{look}_mat{key}")
                if rgba:
                    mat.rgba = rgba
                elif p["png"]:
                    if i not in textures:
                        textures[i] = spec.add_texture(name=f"person_t{i}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                                       file=os.path.join(PERSON_DIR, p["png"])).name
                    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = textures[i]
                parts.append((mesh.name, mat.name))
        out[look] = parts
    return out


def add_person(spec, parts, name, x=0.0, y=-5.0, yaw=0.0, pose="stand", look="adult"):
    """parts: add_person_assets の look の部品。メッシュは顔が -y を向いている (頭の真ん中は y≈-0.04)。"""
    L = LOOKS[look]
    s = L["scale"]
    pos, quat = person_pose(x, y, yaw, pose, scale=s)
    body = spec.worldbody.add_body(name=name, pos=pos, quat=quat, mocap=True)
    for k, (mesh, mat) in enumerate(parts):
        body.add_geom(name=f"{name}_g{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material=mat,
                      contype=0, conaffinity=0, group=1)
    vis = {"contype": 0, "conaffinity": 0, "group": 1}
    if L.get("hat"):        # つばのある黒い帽子 (円柱 2 つ。つばは顔の側へ出す)
        body.add_geom(name=f"{name}_hat", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.115 * s, 0.05 * s, 0],
                      pos=[0, -0.035 * s, 1.875 * s], rgba=[0.05, 0.05, 0.06, 1], **vis)
        body.add_geom(name=f"{name}_brim", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.15 * s, 0.006 * s, 0],
                      pos=[0, -0.08 * s, 1.83 * s], rgba=[0.05, 0.05, 0.06, 1], **vis)
    if L.get("mask"):       # 口と鼻をおおう白いマスク (箱)
        body.add_geom(name=f"{name}_mask", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.07 * s, 0.03 * s, 0.045 * s],
                      pos=[0, -0.15 * s, 1.715 * s], rgba=[0.93, 0.93, 0.93, 1], **vis)
    # 当たり判定: 足元から頭までのカプセル (立てば縦、横になれば横)
    body.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.18 * s, 0.7 * s, 0],
                  pos=[0, -0.05 * s, 0.9 * s], rgba=[0, 0, 0, 0], group=3)


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


def compose(house_dir, duck=True, n_persons=0, looks=None):
    """(MjSpec, 扉の開いた値) を返す。人は家の外 (y=-2.5) に置いておき、場面に合わせて動かす。
    looks: 人 i の見た目 (LOOKS のキー)。無ければみな大人。"""
    xml, door_open = build_with_info(house_dir)
    spec = mujoco.MjSpec.from_string(xml)
    if duck:
        add_duck(spec)
    if n_persons:
        looks = list(looks or []) + ["adult"] * (n_persons - len(looks or []))
        parts = add_person_assets(spec, looks[:n_persons])
        for i in range(n_persons):
            add_person(spec, parts[looks[i]], f"person{i}", x=-1.5 - i, y=-2.5, look=looks[i])
    return spec, door_open
