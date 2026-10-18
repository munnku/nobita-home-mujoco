"""Gazebo の person_standing (MakeHuman 製の DAE) を MuJoCo で読める OBJ + PNG に分ける。

MuJoCo の mesh は 1 つの材質 (テクスチャ 1 枚) しか持てないので、DAE の材質ごとに OBJ を書き出す。
使い方: python3 tools/convert_person.py <person_standing のフォルダ> <出力フォルダ>
出力: <出力>/person_*.obj, *.png, person.json (部品名 → obj, png, 色)
"""
import json
import os
import shutil
import sys

import numpy as np
import trimesh

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
scene = trimesh.load(os.path.join(src, "meshes", "standing.dae"), force="scene")
parts = []
for i, (name, geom) in enumerate(scene.geometry.items()):
    # シーングラフの変換 (DAE の up_axis など) を頂点に焼き込む
    nodes = scene.graph.geometry_nodes.get(name, [])
    mesh = geom.copy()
    if nodes:
        mat, _ = scene.graph[nodes[0]]
        mesh.apply_transform(mat)
    tex = None
    rgba = [0.8, 0.7, 0.6, 1.0]
    vis = mesh.visual
    if isinstance(vis, trimesh.visual.TextureVisuals) and vis.material is not None:
        img = getattr(vis.material, "image", None)
        if img is None and hasattr(vis.material, "baseColorTexture"):
            img = vis.material.baseColorTexture
        if img is not None:
            tex = f"person_{i}.png"
            img.convert("RGB").save(os.path.join(out, tex))
        diffuse = getattr(vis.material, "diffuse", None)
        if diffuse is not None:
            rgba = (np.asarray(diffuse[:4]) / 255.0).round(3).tolist()
    obj = f"person_{i}.obj"
    mesh.export(os.path.join(out, obj), include_texture=False)
    parts.append({"name": name, "obj": obj, "png": tex, "rgba": rgba,
                  "faces": int(len(mesh.faces)), "zmin": float(mesh.bounds[0][2]),
                  "zmax": float(mesh.bounds[1][2])})
    print(parts[-1])
lo = np.min([scene.geometry[g].bounds[0] for g in scene.geometry], axis=0)
print("scene bounds", scene.bounds)
with open(os.path.join(out, "person.json"), "w") as f:
    json.dump({"parts": parts, "bounds": np.asarray(scene.bounds).tolist()}, f, indent=1)
