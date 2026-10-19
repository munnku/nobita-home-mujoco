"""シミュレーションの世界: 家 + アヒル + 人 + 扉 + 照明 + 地図 をまとめて持つ。

IoT 機器やロボットの制御はこのクラスを通して世界を読み書きする (MuJoCo を直接触らない)。
- step(): アヒルの方策を 1 回 (0.02 s) 進める
- set_door / door_is_open: 扉の開け閉め (地図も作り直す)
- place_person: 人を置く・姿勢を変える (立つ / 横になる)
- set_light: 部屋の照明
- render(camera): カメラの絵 (RGB)
"""
import math
import os

import mujoco
import numpy as np

from duck import Duck
from floormap import FloorMap
from house import load as load_layout
from navigator import Navigator
from scene import compose, person_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HOUSE = os.path.join(ROOT, "houses", "nobita")
LIGHT_ON = np.array([0.45, 0.44, 0.40])
MAX_RETRIES = 3       # 進まなくなったときに経路を作り直す回数


class World:
    def __init__(self, house_dir=DEFAULT_HOUSE, duck_pose=(0.0, 0.0, 0.0), n_persons=2,
                 policy="velstand"):
        self.layout = load_layout(house_dir)
        self.module = self.layout["module"]
        spec, self.door_open_q = compose(house_dir, duck=True, n_persons=n_persons)
        self.duck = Duck(spec, policy=policy)
        self.model, self.data = self.duck.model, self.duck.data
        m = self.model
        self.n_persons = n_persons
        self.person_state = {}
        self.door_state = {}
        for name, d in self.layout["doors"].items():
            self.set_door(name, d.get("state", "closed") == "open", update_map=False)
        self.duck.reset(*duck_pose)
        duck_root = m.body_rootid[self.duck.trunk]
        ignore = [b for b in range(m.nbody) if m.body_rootid[b] == duck_root]
        self.map = FloorMap(m, self.data, self.layout["rooms"], self.module, ignore_bodies=ignore)
        self.nav = Navigator()
        self.goal, self.retries, self.last_path = None, 0, None
        self.manual_cmd = (0.0, 0.0, 0.0)
        self.time = 0.0
        self._renderers = {}

    # ---- 時間
    def step(self):
        self._hold_doors()
        if not self.nav.done:
            vx, wz = self.nav.update(*self.duck.pose())
            self.duck.set_cmd(vx, 0.0, wz)
        else:
            self.duck.set_cmd(*self.manual_cmd)
            if self.nav.stuck and self.goal is not None and self.retries < MAX_RETRIES:
                # 進まなくなった: いまの位置から経路を作り直す
                self.retries += 1
                if not self._plan(*self.goal):
                    self.goal = None
        self.duck.step()
        self.time += 0.02

    # ---- 移動
    def go_to(self, x, y, yaw=None, tolerance=None):
        """(x, y) [m] へ向かう経路を作ってアヒルに渡す。経路がなければ False。
        tolerance: (x, y) からこの距離に入れば着いたとする (人のそばなど、その点には立てないとき)。"""
        self.goal, self.retries = (x, y, yaw, tolerance), 0
        return self._plan(x, y, yaw, tolerance)

    def stop(self):
        self.goal = None
        self.nav.stop()
        self.manual_cmd = (0.0, 0.0, 0.0)

    def set_manual_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        """経路をたどっていないときの速度指令 (その場で向きを変えるときなど)。"""
        self.manual_cmd = (vx, vy, wz)

    def _plan(self, x, y, yaw=None, tolerance=None):
        px, py, _ = self.duck.pose()
        path = self.map.plan((px, py), (x, y))
        if path is None:
            return False
        self.nav.set_path(path, goal_yaw=yaw, target=(x, y), tolerance=tolerance)
        self.last_path = path
        return True

    def nav_status(self):
        """'moving' / 'arrived' / 'failed' / 'idle'"""
        if not self.nav.done:
            return "moving"
        if self.nav.arrived:
            return "arrived"
        if self.nav.stuck:
            return "failed" if self.retries >= MAX_RETRIES else "moving"
        return "idle"

    # ---- 扉
    def _joint_adr(self, name):
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"door_{name}")
        return self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]

    def set_door(self, name, is_open, update_map=True):
        qa, da = self._joint_adr(name)
        self.data.qpos[qa] = self.door_open_q[name] if is_open else 0.0
        self.data.qvel[da] = 0.0
        self.door_state[name] = bool(is_open)
        if update_map and hasattr(self, "map"):
            self.map.update()

    def _hold_doors(self):
        """扉は開け閉めしたときだけ動く (人やアヒルに押されて動かないよう、毎回決めた角度に戻す)。"""
        for name, is_open in self.door_state.items():
            qa, da = self._joint_adr(name)
            self.data.qpos[qa] = self.door_open_q[name] if is_open else 0.0
            self.data.qvel[da] = 0.0

    def door_is_open(self, name):
        qa, _ = self._joint_adr(name)
        return bool(abs(self.data.qpos[qa]) > 0.1 * abs(self.door_open_q[name]))

    # ---- 人
    def place_person(self, i, x, y, yaw=0.0, pose="stand", level="1F", update_map=True, dz=0.0):
        """人 i を (x, y) [m] に置く。pose: stand / lie。家の外へ出すときは x=None。"""
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"person{i}")
        mid = self.model.body_mocapid[b]
        if x is None:
            pos, quat = person_pose(-1.5 - i, -2.5, 0.0, "stand", floor_z=-0.45)
        else:
            pos, quat = person_pose(x, y, yaw, pose, self.layout["levels"][level]["z"] + dz)
        self.data.mocap_pos[mid] = pos
        self.data.mocap_quat[mid] = quat
        # 見せ方の group: 1 階の人は 1、2 階の人は 2 階と同じ 4 (1 階の見下ろしで浮いて見えないように)
        grp = 4 if level == "2F" else 1
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == b and self.model.geom_group[g] in (1, 4):
                self.model.geom_group[g] = grp
        self.person_state[i] = None if x is None else {"x": x, "y": y, "yaw": yaw, "pose": pose,
                                                       "level": level}
        if update_map and hasattr(self, "map"):
            self.map.update()

    # ---- 照明
    def set_light(self, room, on):
        lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, f"light_{room}")
        if lid >= 0:
            self.model.light_diffuse[lid] = LIGHT_ON if on else 0.0
            self.model.light_active[lid] = 1 if on else 0

    def light_is_on(self, room):
        lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, f"light_{room}")
        return lid >= 0 and bool(self.model.light_active[lid])

    # ---- 部屋
    def room_of(self, x, y, level="1F"):
        for r in self.layout["rooms"]:
            if r["level"] != level:
                continue
            x0, x1, y0, y1 = (v * self.module for v in r["rect"])
            if x0 <= x <= x1 and y0 <= y <= y1:
                return r["name"]
        return None

    def room_center(self, name):
        r = next(r for r in self.layout["rooms"] if r["name"] == name)
        x0, x1, y0, y1 = (v * self.module for v in r["rect"])
        return (x0 + x1) / 2, (y0 + y1) / 2

    # ---- 描画
    def renderer(self, w, h):
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, h, w)
        return self._renderers[key]

    def render(self, camera, w=640, h=480, groups=(0, 1, 2, 4, 5)):
        """camera: カメラの名前か MjvCamera。groups: 見せる geom の group (既定は天井・2 階込みの全部)。"""
        r = self.renderer(w, h)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in groups:
            opt.geomgroup[g] = 1
        r.update_scene(self.data, camera, opt)
        return r.render()

    def chase_camera(self, distance=1.4, elevation=-25):
        cam = mujoco.MjvCamera()
        x, y, yaw = self.duck.pose()
        cam.lookat[:] = [x, y, 0.15]
        cam.distance, cam.azimuth, cam.elevation = distance, math.degrees(yaw), elevation
        return cam
