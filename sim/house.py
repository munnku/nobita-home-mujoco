"""間取り (houses/<家>/layout.yaml) から MuJoCo の家 (MJCF の文字列) を作る。

layout.yaml の平面の座標はマス目 (半間 0.91 m) の単位、高さは m。ここで m に直す。
- 床: 1 階は地面から床までの箱 (土間の段差もそのまま形になる)、2 階は厚さ 0.3 m の床板
- 壁: 線分を開口 (扉) と窓で切り分けた箱。扉の上は欠き込み (2.0 m から上は壁)、窓はガラス
- 扉: 開き戸は hinge、引き戸は slide の関節を持つ body (家の IoT が qpos を書いて開け閉めする)
- 階段・家具・天井・屋根・外構
見せ方の group: 0 = 1 階 (と当たり判定)、4 = 2 階、5 = 天井と屋根。見下ろしの絵では 4・5 を隠す。
"""
import math
import os

import yaml

GROUP_1F, GROUP_2F, GROUP_CEIL = 0, 4, 5
DOOR_HEAD = 2.0          # 扉の上端 [m]
WIN_LOW, WIN_HIGH = 0.9, 2.0
COLORS = {
    "wood": "0.62 0.45 0.28 1", "tatami": "0.74 0.76 0.47 1", "tile": "0.85 0.85 0.83 1",
    "vinyl": "0.88 0.86 0.80 1", "concrete": "0.6 0.6 0.58 1", "grass": "0.40 0.55 0.30 1",
    "paper": "0.93 0.91 0.84 1", "dark": "0.25 0.18 0.12 1", "sofa": "0.55 0.25 0.22 1",
    "white": "0.95 0.95 0.95 1", "black": "0.05 0.05 0.05 1", "wall": "0.93 0.90 0.83 1",
    "glass": "0.6 0.8 0.9 0.25", "door": "0.70 0.55 0.38 1", "fusuma": "0.92 0.90 0.80 1",
    "ceiling": "0.97 0.96 0.93 1", "roof": "0.35 0.30 0.30 1",
}
TEXTURED = ("tatami", "tile", "wood", "grass")


def load(house_dir):
    with open(os.path.join(house_dir, "layout.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


class Builder:
    def __init__(self, lay):
        self.lay = lay
        self.m = lay["module"]
        self.t = lay["wall_thickness"]
        self.geoms = []       # worldbody の geom の XML
        self.bodies = []      # 扉の body の XML
        self.door_open = {}   # 扉の名前 → 開いたときの関節の値 (閉じると 0)

    def level(self, name):
        lv = self.lay["levels"][name]
        return lv["z"], lv["height"]

    @staticmethod
    def group(level):
        return GROUP_2F if level == "2F" else GROUP_1F

    def box(self, name, x0, x1, y0, y1, z0, z1, color, group=0, extra=""):
        """[x0, x1, y0, y1] はマス、z は m の直方体を足す。"""
        k = self.m
        x0, x1, y0, y1 = x0 * k, x1 * k, y0 * k, y1 * k
        if x1 - x0 < 1e-4 or y1 - y0 < 1e-4 or z1 - z0 < 1e-4:
            return
        mat = f'material="mat_{color}" ' if color in TEXTURED else ""
        rgba = "1 1 1 1" if color in TEXTURED else COLORS.get(color, color)
        self.geoms.append(
            f'<geom name="{name}" type="box" pos="{(x0+x1)/2:.4f} {(y0+y1)/2:.4f} {(z0+z1)/2:.4f}" '
            f'size="{(x1-x0)/2:.4f} {(y1-y0)/2:.4f} {(z1-z0)/2:.4f}" rgba="{rgba}" {mat}group="{group}" {extra}/>')

    # ---- 床
    def floors(self):
        gz = self.lay["ground_z"]
        for r in self.lay["rooms"]:
            z, _ = self.level(r["level"])
            top = z + r.get("z", 0.0)
            x0, x1, y0, y1 = r["rect"]
            bottom = gz if r["level"] == "1F" else z - 0.3
            self.box(f"floor_{r['name']}", x0, x1, y0, y1, bottom, top, r["material"],
                     self.group(r["level"]))
        for o in self.lay.get("outside", []):
            x0, x1, y0, y1 = o["rect"]
            self.box(f"outside_{o['name']}", x0, x1, y0, y1, gz - 0.05, o["z"], o["material"])

    # ---- 壁と扉
    def walls(self):
        t = self.t / self.m           # 壁の厚さ (マス)
        for i, w in enumerate(self.lay["walls"]):
            z, h = self.level(w["level"])
            top = z + w.get("height", h + (0.3 if w["level"] == "1F" else 0.0))
            g = self.group(w["level"])
            x1, y1, x2, y2 = w["seg"]
            horiz = abs(y1 - y2) < 1e-6
            a, b = (min(x1, x2), max(x1, x2)) if horiz else (min(y1, y2), max(y1, y2))
            cuts = [(o["at"][0], o["at"][1], "opening", o) for o in w.get("openings", [])]
            cuts += [(win[0], win[1], "window", None) for win in w.get("windows", [])]
            cuts.sort(key=lambda c: c[0])
            spans, pos = [], a
            for s, e, kind, o in cuts:
                if s > pos:
                    spans.append((pos, s, "wall", None))
                spans.append((s, e, kind, o))
                pos = e
            if pos < b:
                spans.append((pos, b, "wall", None))

            def piece(name, s, e, z0, z1, color="wall", extra=""):
                # 線分の端の壁は厚さの半分だけ延ばして角を埋める
                ss = s - t / 2 if abs(s - a) < 1e-6 else s
                ee = e + t / 2 if abs(e - b) < 1e-6 else e
                if horiz:
                    self.box(name, ss, ee, y1 - t / 2, y1 + t / 2, z0, z1, color, g, extra)
                else:
                    self.box(name, x1 - t / 2, x1 + t / 2, ss, ee, z0, z1, color, g, extra)

            for j, (s, e, kind, o) in enumerate(spans):
                nm = f"wall_{i}_{j}"
                if kind == "wall":
                    piece(nm, s, e, z, top)
                elif kind == "window":
                    piece(nm + "_lo", s, e, z, z + WIN_LOW)
                    piece(nm + "_glass", s, e, z + WIN_LOW, z + WIN_HIGH, "glass",
                          'contype="0" conaffinity="0"')
                    piece(nm + "_hi", s, e, z + WIN_HIGH, top)
                else:
                    if top > z + DOOR_HEAD + 0.01:
                        piece(nm + "_head", s, e, z + DOOR_HEAD, top)
                    if o.get("door"):
                        self.door(o["door"], horiz, x1, y1, s, e, z, g, room=(s - a, b - e))

    def door(self, name, horiz, x1, y1, s, e, z, group, room=(1e9, 1e9)):
        """room: 開口の前後に残っている壁の長さ (マス)。引き戸はこれが長い側へ引き込む。"""
        d = self.lay["doors"][name]
        k = self.m
        width = (e - s) * k
        th = 0.04
        dirx, diry = (1.0, 0.0) if horiz else (0.0, 1.0)     # 線分の向き (座標が増える向き)
        if horiz:
            p_from, p_to = (s * k, y1 * k), (e * k, y1 * k)
        else:
            p_from, p_to = (x1 * k, s * k), (x1 * k, e * k)
        if d["type"] == "hinge":
            at_from = d.get("hinge_at", "from") == "from"
            hx, hy = p_from if at_from else p_to
            ux, uy = (dirx, diry) if at_from else (-dirx, -diry)  # 軸から扉の先への向き
            yaw = math.atan2(uy, ux)
            sign = d.get("swing", 1) * (1 if at_from else -1)
            ang = math.radians(d.get("open", 90))
            rng = f"{min(0, sign * ang):.3f} {max(0, sign * ang):.3f}"
            self.bodies.append(
                f'<body name="door_{name}" pos="{hx:.4f} {hy:.4f} {z:.3f}" euler="0 0 {yaw:.4f}">'
                f'<joint name="door_{name}" type="hinge" axis="0 0 1" range="{rng}" damping="200" frictionloss="20"/>'
                f'<geom name="door_{name}_leaf" type="box" pos="{width/2:.4f} 0 {DOOR_HEAD/2+0.01:.3f}" '
                f'size="{width/2-0.01:.4f} {th/2} {DOOR_HEAD/2-0.01:.3f}" rgba="{COLORS["door"]}" group="{group}" mass="10"/>'
                f'<geom name="door_{name}_knob" type="sphere" pos="{width-0.08:.3f} {sign*0.04:.3f} 0.95" size="0.025" '
                f'rgba="0.8 0.7 0.3 1" contype="0" conaffinity="0" group="{group}"/></body>')
            self.door_open[name] = sign * ang
        else:
            # 引き戸: 壁の片側に寄せて置き、壁が長く残っている側へ引き込む (はみ出さない分だけ開く)
            cx, cy = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
            off = self.t / 2 + th / 2 + 0.005
            yaw = math.atan2(diry, dirx)
            sgn = -1.0 if room[0] >= room[1] else 1.0
            travel = min(width * 0.98, max(room) * k - self.t / 2)
            rng = f"{-travel:.3f} 0" if sgn < 0 else f"0 {travel:.3f}"
            color = COLORS["fusuma"] if name == "ima" else COLORS["door"]
            self.bodies.append(
                f'<body name="door_{name}" pos="{cx - diry*off:.4f} {cy + dirx*off:.4f} {z:.3f}" euler="0 0 {yaw:.4f}">'
                f'<joint name="door_{name}" type="slide" axis="1 0 0" range="{rng}" damping="200" frictionloss="20"/>'
                f'<geom name="door_{name}_leaf" type="box" pos="0 0 {DOOR_HEAD/2+0.01:.3f}" '
                f'size="{width/2:.4f} {th/2} {DOOR_HEAD/2-0.01:.3f}" rgba="{color}" group="{group}" mass="5"/></body>')
            self.door_open[name] = sgn * travel

    # ---- 階段
    def stairs(self):
        st = self.lay.get("stairs")
        if not st:
            return
        runs = st["runs"]
        rise = st["total_rise"] / sum(r["steps"] for r in runs)
        k = 0
        for r in runs:
            a0, a1 = r["across"]
            for i in range(r["steps"]):
                s = r["from"] + (r["to"] - r["from"]) * i / r["steps"]
                e = r["from"] + (r["to"] - r["from"]) * (i + 1) / r["steps"]
                lo, hi = min(s, e), max(s, e)
                top = (k + 1) * rise
                bottom = 0.0 if r.get("solid") else top - 0.2
                if r["axis"] == "x":
                    self.box(f"stair_{k}", lo, hi, a0, a1, bottom, top, "wood")
                else:
                    self.box(f"stair_{k}", a0, a1, lo, hi, bottom, top, "wood")
                k += 1

    # ---- 家具・天井・屋根・照明
    def furniture(self):
        for f in self.lay["furniture"]:
            z, _ = self.level(f["level"])
            x0, x1, y0, y1 = f["rect"]
            z0 = z + f.get("z0", 0.0)
            self.box(f["name"], x0, x1, y0, y1, z0, z0 + f["height"], f["color"], self.group(f["level"]))

    def ceilings(self):
        z1, h1 = self.level("1F")
        z2, h2 = self.level("2F")
        for i, (x0, x1, y0, y1) in enumerate(self.lay["ceilings"]["1F"]):
            self.box(f"ceiling1_{i}", x0, x1, y0, y1, z1 + h1, z1 + h1 + 0.3, "ceiling", GROUP_CEIL)
        for i, (x0, x1, y0, y1) in enumerate(self.lay["ceilings"]["2F"]):
            self.box(f"roof_{i}", x0 - 0.3, x1 + 0.3, y0 - 0.3, y1 + 0.3, z2 + h2, z2 + h2 + 0.2, "roof",
                     GROUP_CEIL)

    def lights(self):
        rooms = {r["name"]: r for r in self.lay["rooms"]}
        out = []
        for name in self.lay.get("lights", []):
            r = rooms[name]
            z, h = self.level(r["level"])
            x0, x1, y0, y1 = r["rect"]
            out.append(f'<light name="light_{name}" pos="{(x0+x1)/2*self.m:.2f} {(y0+y1)/2*self.m:.2f} '
                       f'{z+h-0.15:.2f}" dir="0 0 -1" diffuse="0.45 0.44 0.40" specular="0 0 0" '
                       f'attenuation="1 0.1 0.05" cutoff="80" exponent="1" castshadow="false"/>')
        return out


def cameras(house_dir, m):
    """cameras.yaml (pos と look_at、x y はマス・z は m) を MuJoCo の固定カメラにする。"""
    path = os.path.join(house_dir, "cameras.yaml")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = []
    for c in cfg["cameras"]:
        px, py, pz = c["pos"][0] * m, c["pos"][1] * m, c["pos"][2]
        tx, ty, tz = c["look_at"][0] * m, c["look_at"][1] * m, c["look_at"][2]
        fx, fy, fz = tx - px, ty - py, tz - pz
        n = math.sqrt(fx * fx + fy * fy + fz * fz)
        fx, fy, fz = fx / n, fy / n, fz / n
        rn = math.hypot(fx, fy)
        rx, ry = fy / rn, -fx / rn                         # 右 = 前 × 上 (z)
        ux, uy, uz = ry * fz, -rx * fz, rx * fy - ry * fx  # 上 = 右 × 前
        out.append(f'<camera name="{c["name"]}" pos="{px:.3f} {py:.3f} {pz:.3f}" '
                   f'xyaxes="{rx:.4f} {ry:.4f} 0 {ux:.4f} {uy:.4f} {uz:.4f}" fovy="{cfg.get("fovy", 70)}"/>')
    return out


def build_with_info(house_dir):
    """(MJCF の文字列, 扉の開いた値の辞書) を返す。"""
    lay = load(house_dir)
    b = Builder(lay)
    b.floors()
    b.walls()
    b.stairs()
    b.furniture()
    b.ceilings()
    lights = b.lights()
    cams = cameras(house_dir, lay["module"])
    nl = "\n    "
    xml = f"""<mujoco model="{lay['name']}">
  <compiler angle="radian"/>
  <visual><headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <quality shadowsize="0"/><global offwidth="1920" offheight="1080"/></visual>
  <asset>
    <texture name="tx_tatami" type="2d" builtin="checker" rgb1="0.76 0.78 0.50" rgb2="0.72 0.74 0.46" width="64" height="128" mark="edge" markrgb="0.40 0.35 0.22"/>
    <material name="mat_tatami" texture="tx_tatami" texrepeat="1.1 0.55" texuniform="true"/>
    <texture name="tx_wood" type="2d" builtin="gradient" rgb1="0.66 0.49 0.31" rgb2="0.58 0.42 0.26" width="32" height="256"/>
    <material name="mat_wood" texture="tx_wood" texrepeat="6 1" texuniform="true"/>
    <texture name="tx_tile" type="2d" builtin="checker" rgb1="0.86 0.86 0.84" rgb2="0.80 0.80 0.78" width="64" height="64"/>
    <material name="mat_tile" texture="tx_tile" texrepeat="3 3" texuniform="true"/>
    <texture name="tx_grass" type="2d" builtin="flat" rgb1="0.40 0.55 0.30" mark="random" random="0.2" markrgb="0.3 0.45 0.22" width="256" height="256"/>
    <material name="mat_grass" texture="tx_grass" texrepeat="4 4" texuniform="true"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.95 0.97 1.0" width="256" height="1536"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 20" dir="0.3 0.5 -1" diffuse="0.45 0.45 0.42" specular="0 0 0" castshadow="false"/>
    {nl.join(lights)}
    {nl.join(b.geoms)}
    {nl.join(b.bodies)}
    {nl.join(cams)}
  </worldbody>
</mujoco>
"""
    return xml, b.door_open


def build(house_dir, **_):
    return build_with_info(house_dir)[0]


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1]))
