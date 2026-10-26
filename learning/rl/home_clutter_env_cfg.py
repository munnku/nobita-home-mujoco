"""この家の床で歩く Microduck の方策 (Mjlab-Home-Clutter[2]-Backlash-MicroDuck)。

microduck_rl の速度追従タスク (rough=True) をもとに、地形を「家の床」に入れ替える。
- flat     : 何もない床 (フローリング・畳)
- rug      : 薄い敷物 (5〜15 mm)。何枚か重なり、端がめくれている
- zabuton  : 座布団。柔らかい接触 (solref/solimp)
- clutter  : 床に散らかったおもちゃ (10〜25 mm の箱・円柱)
- grid     : ゆるい凸凹 (BoxRandomGrid を低くしたもの)
足裏の摩擦は 0.4〜1.2 (板の間〜畳) に広げる。押し (push_robot) は元のまま。

学習済みの歩きから温めて始めるので、元のタスクの段階的な curriculum は最後の段に固定する
(velstand と同じやり方)。地形の curriculum (terrain_levels) だけは段階を上がっていく。

version 1 (duck_robust, 2026-10-23): 座布団 30〜60 mm、接触はほぼ硬い (4 mm しか沈まない)。
  mjlab の terrain_levels は「4 m 進めば上へ」で、0.34 m/s のアヒルには届かず全員が 0 段に落ちた。
version 2 (duck_robust2): 座布団 30〜45 mm、約 2 cm 沈む (床の下まである背の高い箱にして突き抜けない)。
  地形の段は「倒れずに 1 回 (20 s) を終えたら上へ、倒れたら下へ」(terrain_levels_survival)。

このファイルは PAI1-mujoco/learning/rl/ に置き、install.sh が
~/ext/microduck_rl/src/mjlab_microduck/tasks/ に写して tasks/__init__.py に登録する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import make_plane

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final

FOOT_FRICTION_RANGE = (0.4, 1.2)  # 板の間 (つるつる) 〜 畳 (よく止まる)
FLOOR_FRICTION = 0.4  # MuJoCo は 2 つの geom の大きい方の摩擦を使う → 床を低くして足の DR を効かせる
RUG_FRICTION = 0.8
ZABUTON_FRICTION = 0.7
TOY_FRICTION = 0.5

# 座布団 v1: 時定数 0.1 s、impedance の幅 3 cm。測ると 4 mm しか沈まない (ほぼ段)
ZABUTON_SOLREF = (0.1, 1.0)
ZABUTON_SOLIMP = (0.5, 0.95, 0.03, 0.5, 2.0)
# 座布団 v2: 時定数 0.35 s・減衰比 1.5・幅 6 cm。重さによらず止まって約 2 cm、踏み込むと 3 cm 余り沈む
# (MuJoCo で箱を落として測った。時定数 0.4 s 以上だと厚さ 4 cm の箱は突き抜ける → 床の下まで伸ばす)
ZABUTON2_SOLREF = (0.35, 1.5)
ZABUTON2_SOLIMP = (0.3, 0.95, 0.06, 0.5, 2.0)
ZABUTON2_DEPTH = 0.15  # 座布団の箱を床の下へ伸ばす長さ [m]
# 他の床の geom は元の rough と同じく少しだけ柔らかくする (箱の角で NaN が出ないように)
TERRAIN_SOLREF = (0.04, 1.0)
TERRAIN_SOLIMP = (0.85, 0.95, 0.001, 0.5, 2.0)

SPAWN_CLEAR = 0.5  # タイルの中心 (出現位置) から半径 0.5 m は何も置かない
WARM_START = True


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _random_xy(rng, size, margin: float, clear: float) -> tuple[float, float]:
    """タイルの中で、中心から clear 以上離れた点。"""
    cx, cy = size[0] / 2, size[1] / 2
    while True:
        x = rng.uniform(margin, size[0] - margin)
        y = rng.uniform(margin, size[1] - margin)
        if math.hypot(x - cx, y - cy) > clear:
            return x, y


def _floor(body, size) -> TerrainGeometry:
    g = make_plane(body, size, 0.0, center_zero=False)[0]
    return TerrainGeometry(geom=g, color=(0.62, 0.52, 0.40, 1.0))


@dataclass(kw_only=True)
class RugTerrainCfg(SubTerrainCfg):
    """薄い敷物を何枚か (重なりあり)。厚さは difficulty で 5→15 mm、端のめくれも足す。"""

    thickness_range: tuple[float, float] = (0.005, 0.015)
    num_rugs: tuple[int, int] = (6, 14)

    def function(self, difficulty, spec, rng) -> TerrainOutput:
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        t_max = _lerp(*self.thickness_range, difficulty)
        n = int(rng.integers(*self.num_rugs))
        for _ in range(n):
            x, y = _random_xy(rng, self.size, 0.8, 0.0)
            sx, sy = rng.uniform(0.6, 2.0), rng.uniform(0.4, 1.5)
            t = rng.uniform(self.thickness_range[0], t_max)
            yaw = rng.uniform(0, math.pi)
            # 重なった敷物は上に乗る (下の敷物の厚さぶん少し持ち上げる)
            z = t / 2 + rng.choice([0.0, 0.0, rng.uniform(0.0, t_max)])
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, t / 2),
                              pos=(x, y, z), quat=_yaw_quat(yaw), friction=(RUG_FRICTION, 0.005, 0.0001))
            geoms.append(TerrainGeometry(geom=g, color=(0.70, 0.25, 0.25, 1.0)))
            # 端のめくれ (細い帯が少し高い)
            if rng.uniform() < 0.5:
                c, s = math.cos(yaw), math.sin(yaw)
                ex, ey = x + c * sx / 2, y + s * sx / 2
                w = rng.uniform(0.03, 0.08)
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(w / 2, sy / 2, t),
                                  pos=(ex, ey, t), quat=_yaw_quat(yaw), friction=(RUG_FRICTION, 0.005, 0.0001))
                geoms.append(TerrainGeometry(geom=g, color=(0.60, 0.20, 0.20, 1.0)))
        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, geometries=geoms)


@dataclass(kw_only=True)
class ZabutonTerrainCfg(SubTerrainCfg):
    """座布団 (約 55×59 cm)。高さは difficulty で 30→60 mm。縁は丸いので 2 段の箱で作る。"""

    height_range: tuple[float, float] = (0.03, 0.06)
    num_cushions: tuple[int, int] = (8, 16)
    solref: tuple[float, float] = ZABUTON_SOLREF
    solimp: tuple[float, ...] = ZABUTON_SOLIMP
    depth: float = 0.0  # 箱の下を床の下へ伸ばす長さ (柔らかいと突き抜けるので)

    def function(self, difficulty, spec, rng) -> TerrainOutput:
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        h_max = _lerp(*self.height_range, difficulty)
        n = int(rng.integers(*self.num_cushions))
        for _ in range(n):
            x, y = _random_xy(rng, self.size, 0.6, SPAWN_CLEAR + 0.4)
            h = rng.uniform(self.height_range[0], h_max)
            yaw = rng.uniform(0, math.pi)
            sx, sy = rng.uniform(0.50, 0.60), rng.uniform(0.52, 0.62)
            for frac, inset in ((0.5, 0.0), (1.0, 0.05)):  # 下段 (縁) と上段 (ふくらみ)
                hh = h * frac + self.depth
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                  size=(sx / 2 - inset, sy / 2 - inset, hh / 2),
                                  pos=(x, y, h * frac - hh / 2),
                                  quat=_yaw_quat(yaw), friction=(ZABUTON_FRICTION, 0.005, 0.0001),
                                  solref=self.solref, solimp=self.solimp)
                geoms.append(TerrainGeometry(geom=g, color=(0.25, 0.35, 0.65, 1.0)))
        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, geometries=geoms)


@dataclass(kw_only=True)
class ClutterTerrainCfg(SubTerrainCfg):
    """散らかったおもちゃ: 小さな箱 (積み木)・立った円柱 (ブロック)・寝た円柱 (ペン・クレヨン)。
    数と高さの上限は difficulty で増える (高さ 10→25 mm)。"""

    height_range: tuple[float, float] = (0.010, 0.025)
    count_range: tuple[int, int] = (60, 200)

    def function(self, difficulty, spec, rng) -> TerrainOutput:
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        h_max = _lerp(*self.height_range, difficulty)
        n = int(_lerp(*self.count_range, difficulty))
        fr = (TOY_FRICTION, 0.005, 0.0001)
        for _ in range(n):
            x, y = _random_xy(rng, self.size, 0.3, SPAWN_CLEAR)
            h = rng.uniform(self.height_range[0], h_max)
            yaw = rng.uniform(0, 2 * math.pi)
            kind = rng.uniform()
            if kind < 0.45:
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                  size=(rng.uniform(0.01, 0.04), rng.uniform(0.01, 0.04), h / 2),
                                  pos=(x, y, h / 2), quat=_yaw_quat(yaw), friction=fr)
                col = (0.9, 0.7, 0.1, 1.0)
            elif kind < 0.75:
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                  size=(rng.uniform(0.01, 0.03), h / 2, 0.0), pos=(x, y, h / 2), friction=fr)
                col = (0.2, 0.7, 0.3, 1.0)
            else:
                # 寝た円柱: 軸を水平に (x 軸まわりに 90°) してから yaw
                r = h / 2
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, np.array(_yaw_quat(yaw)), np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]))
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                  size=(r, rng.uniform(0.025, 0.075), 0.0), pos=(x, y, r), quat=q, friction=fr)
                col = (0.2, 0.4, 0.9, 1.0)
            geoms.append(TerrainGeometry(geom=g, color=col))
        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, geometries=geoms)


HOME_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=5,
    curriculum=True,  # 列 = 床の種類、行 = 難しさ。terrain_levels で上の行へ進む
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "rug": RugTerrainCfg(proportion=0.20),
        "zabuton": ZabutonTerrainCfg(proportion=0.15),
        "clutter": ClutterTerrainCfg(proportion=0.25),
        # 0.45 m 角のマス、±6 mm (元の rough は ±10 mm)
        "grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.006), platform_width=1.5,
        ),
    },
    add_lights=False,
)

# version 2: 座布団を低く柔らかく (30〜45 mm、約 2 cm 沈む)。ほかは v1 と同じ
HOME_TERRAINS_CFG2 = replace(
    HOME_TERRAINS_CFG,
    sub_terrains={
        **HOME_TERRAINS_CFG.sub_terrains,
        "zabuton": ZabutonTerrainCfg(proportion=0.15, height_range=(0.03, 0.045), solref=ZABUTON2_SOLREF,
                                     solimp=ZABUTON2_SOLIMP, depth=ZABUTON2_DEPTH),
    },
)


def terrain_levels_survival(env, env_ids, command_name=None, asset_cfg=None):
    """地形の段: 倒れずに 1 回を終えた (time out) ら 1 段上、倒れた (terminated) ら 1 段下。

    mjlab の terrain_levels_vel は「タイルの半分 (4 m) 進めば上」なので、0.3 m/s ほどで指令の向きも
    変わるアヒルには届かない (version 1 で全員 0 段に落ちた)。指令 0 の env も上がるが、
    散らかった床で立ち続けるのも練習になる。"""
    del command_name, asset_cfg
    terrain = env.scene.terrain
    move_up = env.termination_manager.time_outs[env_ids]
    move_down = env.termination_manager.terminated[env_ids] & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    result = {"mean": levels.mean(), "max": levels.max()}
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if terrain.terrain_origins.shape[1] == len(names):
        for i, name in enumerate(names):
            mask = terrain.terrain_types == i
            if mask.any():
                result[name] = levels[mask].mean()
    return result


def _house_floor_contacts(spec: mujoco.MjSpec) -> None:
    """床の geom の接触を整える。座布団 (solref を自分で決めたもの) はそのまま、
    それ以外は元の rough と同じく少し柔らかく。摩擦を決めていない床は FLOOR_FRICTION に。"""
    n_soft = n_floor = 0
    for geom in spec.body("terrain").geoms:
        if geom.solref[0] >= ZABUTON_SOLREF[0] - 1e-9:  # 座布団 (v1 0.1 s、v2 0.35 s)
            n_soft += 1
            continue
        geom.solref = TERRAIN_SOLREF
        geom.solimp = TERRAIN_SOLIMP
        if abs(geom.friction[0] - 1.0) < 1e-9:  # MuJoCo の既定値のまま = 板の間
            geom.friction[0] = FLOOR_FRICTION
            n_floor += 1
    print(f"[home terrain] {n_soft} soft zabuton geoms, {n_floor} floor geoms (friction {FLOOR_FRICTION})")


def make_home_clutter_env_cfg(play: bool = False, version: int = 1):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)
    if WARM_START and not play:
        _collapse_curricula_to_final(cfg)

    base = HOME_TERRAINS_CFG2 if version >= 2 else HOME_TERRAINS_CFG
    gen = replace(base, sub_terrains=dict(base.sub_terrains))
    if version >= 2 and "terrain_levels" in cfg.curriculum:
        cfg.curriculum["terrain_levels"].func = terrain_levels_survival
    if play:
        gen.curriculum = False
        gen.num_rows = 5
        gen.num_cols = 5
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.spec_fn = _house_floor_contacts

    cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION_RANGE
    return cfg


HomeClutterRlCfg = replace(MicroduckRlCfg, run_name="duck_robust")
HomeClutter2RlCfg = replace(MicroduckRlCfg, run_name="duck_robust2")
