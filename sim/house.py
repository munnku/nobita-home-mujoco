"""間取り (layout.yaml) から MuJoCo の家 (MJCF) を作る。

壁・床・家具は直方体。扉の開口は壁の gaps で抜く。IoT カメラは cameras.yaml から固定カメラとして置く。
ロボットと人はあとから MjSpec でつなぐ (sim/scene.py)。
"""
import math
import os

import yaml

COLORS = {
    "wood": "0.62 0.45 0.28 1", "tatami": "0.72 0.74 0.45 1", "tile": "0.82 0.82 0.80 1",
    "paper": "0.93 0.91 0.84 1", "dark": "0.25 0.18 0.12 1", "sofa": "0.45 0.30 0.30 1",
    "white": "0.95 0.95 0.95 1", "wall": "0.93 0.90 0.83 1", "site": "0.45 0.55 0.35 1",
}


def _box(name, xmin, xmax, ymin, ymax, zmin, zmax, rgba, extra=""):
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    sx, sy, sz = (xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2
    return (f'<geom name="{name}" type="box" pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
            f'size="{sx:.4f} {sy:.4f} {sz:.4f}" rgba="{rgba}" {extra}/>')


def wall_pieces(seg, gaps, t):
    """線分の壁を、開口 (gaps) を抜いた直方体の [xmin, xmax, ymin, ymax] の列にする。"""
    x1, y1, x2, y2 = seg
    horizontal = abs(y1 - y2) < 1e-6
    a, b = (min(x1, x2), max(x1, x2)) if horizontal else (min(y1, y2), max(y1, y2))
    cuts = [a] + [v for g in sorted(gaps or []) for v in g] + [b]
    out = []
    for s, e in zip(cuts[0::2], cuts[1::2]):
        if e - s < 1e-3:
            continue
        if horizontal:
            out.append((s - t / 2, e + t / 2, y1 - t / 2, y1 + t / 2))
        else:
            out.append((x1 - t / 2, x1 + t / 2, s - t / 2, e + t / 2))
    return out


def euler_to_xyaxes(roll, pitch, yaw):
    """ROS 流 (x 前・z 上、pitch>0 で下向き) のカメラ姿勢を MuJoCo カメラ (-z が視線) の xyaxes にする。"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = (cp * cy, cp * sy, -sp)
    left = (-sy, cy, 0.0)
    up = (fwd[1] * left[2] - fwd[2] * left[1], fwd[2] * left[0] - fwd[0] * left[2],
          fwd[0] * left[1] - fwd[1] * left[0])
    up = tuple(-u for u in up) if up[2] < 0 else up
    right = (-left[0], -left[1], -left[2])
    return " ".join(f"{v:.4f}" for v in right + up)


def build(house_dir, ceiling=False):
    with open(os.path.join(house_dir, "layout.yaml")) as f:
        lay = yaml.safe_load(f)
    cams = {}
    cam_file = os.path.join(house_dir, "cameras.yaml")
    if os.path.exists(cam_file):
        with open(cam_file) as f:
            cams = yaml.safe_load(f)
    t, h = lay["wall_thickness"], lay["wall_height"]
    g = []
    sx0, sx1, sy0, sy1 = lay["site"]["rect"]
    g.append(_box("site_ground", sx0, sx1, sy0, sy1, -0.02, 0.0, COLORS["site"]))
    for fl in lay["floors"]:
        x0, x1, y0, y1 = fl["rect"]
        g.append(_box(f"floor_{fl['name']}", x0, x1, y0, y1, 0.0, 0.01,
                      "1 1 1 1", f'material="mat_{fl["material"]}"'))
    for i, w in enumerate(lay["walls"]):
        for j, (x0, x1, y0, y1) in enumerate(wall_pieces(w["seg"], w.get("gaps"), t)):
            g.append(_box(f"wall_{i}_{j}", x0, x1, y0, y1, 0.0, h, COLORS["wall"]))
    for fu in lay["furniture"]:
        x0, x1, y0, y1 = fu["rect"]
        g.append(_box(fu["name"], x0, x1, y0, y1, 0.01, fu["height"], COLORS[fu["color"]]))
    if ceiling:
        g.append(_box("ceiling", 0, 7.28, 0, 10.92, h, h + 0.02, COLORS["wall"], 'group="1"'))
    lights = []
    for fl in lay["floors"]:
        x0, x1, y0, y1 = fl["rect"]
        lights.append(f'<light name="light_{fl["name"]}" pos="{(x0+x1)/2:.2f} {(y0+y1)/2:.2f} {h-0.1}" '
                      f'dir="0 0 -1" diffuse="0.35 0.35 0.33" attenuation="1 0 0.02" castshadow="false"/>')
    cam_xml = []
    for c in cams.get("cameras", []):
        x, y, z, r, p, yw = c["pose"]
        fovy = math.degrees(2 * math.atan(math.tan(cams["horizontal_fov"] / 2)
                                          * cams["image_height"] / cams["image_width"]))
        cam_xml.append(f'<camera name="{c["name"]}" pos="{x} {y} {z}" '
                       f'xyaxes="{euler_to_xyaxes(r, p, yw)}" fovy="{fovy:.1f}"/>')
    return f"""<mujoco model="{lay['name']}">
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.4 0.4 0.4" specular="0 0 0"/>
    <quality shadowsize="0"/><global offwidth="1920" offheight="1080"/></visual>
  <asset>
    <texture name="tx_tatami" type="2d" builtin="checker" rgb1="0.74 0.76 0.47" rgb2="0.70 0.72 0.43" width="64" height="128" mark="edge" markrgb="0.35 0.30 0.20"/>
    <material name="mat_tatami" texture="tx_tatami" texrepeat="4 2"/>
    <texture name="tx_wood" type="2d" builtin="flat" rgb1="0.62 0.45 0.28" width="8" height="8"/>
    <material name="mat_wood" rgba="0.62 0.45 0.28 1"/>
    <texture name="tx_tile" type="2d" builtin="checker" rgb1="0.85 0.85 0.83" rgb2="0.78 0.78 0.76" width="64" height="64"/>
    <material name="mat_tile" texture="tx_tile" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 10" dir="0.3 0.2 -1" diffuse="0.4 0.4 0.4" castshadow="false"/>
    {chr(10).join('    ' + s for s in lights)}
{chr(10).join('    ' + s for s in g)}
{chr(10).join('    ' + s for s in cam_xml)}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1]))
