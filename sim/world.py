"""シミュレーションの世界: 家 + アヒル + 人 + 扉 + 照明 + 地図 をまとめて持つ。

IoT 機器やロボットの制御はこのクラスを通して世界を読み書きする (MuJoCo を直接触らない)。
- step(): アヒルの方策を 1 回 (0.02 s) 進める
- set_door / door_is_open: 扉の開け閉め (地図も作り直す)
- place_person: 人を置く・姿勢を変える (立つ / 横になる)
- set_light: 部屋の照明
- render(camera): カメラの絵 (RGB)
- objects: 床に落ちている小さい物 (sim/objects.py)。objects={種類: 個数} で用意し、self.objects で置く・片付ける
- stretch: 片付け係の Stretch 3 (sim/stretch.py)。stretch=(x, y, yaw) で置く。台車は自分の地図 (大きさに合わせて
  広げたもの) の A* と pure pursuit で動く (stretch_go_to)。拾ったものを入れる箱 (tidy_box) も置く。
  2 台のロボットは互いを地図に入れない (動いているロボットどうしはよけない)
"""
import math
import os

import mujoco
import numpy as np

from duck import Duck
from floormap import FloorMap
from house import load as load_layout
from navigator import Navigator
from objects import Objects, add_objects
from stretch import BASE_RADIUS, Stretch, add_stretch
from scene import compose, person_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HOUSE = os.path.join(ROOT, "houses", "nobita")
LIGHT_ON = np.array([0.45, 0.44, 0.40])
# 歩き方の方策ごとの前進の指令の上限と下限 (learning/eval_gait.py で測った)
#   velstand (公開): 指令の 1/3 ほどの速さ。nobita_walk (この家で強化学習): 0.4 で 0.34 m/s、0.2 以下では足踏み
POLICY_SPEED = {"velstand": (0.3, 0.0), "alpha_walking": (0.3, 0.0), "nobita_walk": (0.4, 0.32),
                "nobita_walk2": (0.4, 0.3)}
DUCK_MIN_WZ = 0.7     # その場で回る指令の下限 [rad/s] (歩きの方策は小さい回転の指令を「止まれ」として扱う)
BACK_VX, BACK_TIME = -0.4, 1.5   # 進まなくなったら後ろへ下がる指令 [m/s] と時間 [s] (-0.3 より遅い指令では下がらない)
MAX_RETRIES = 3      # 進まなくなったときに経路を作り直す回数
STRETCH_RADIUS = BASE_RADIUS + 0.07     # Stretch の地図で壁・家具から離す距離 [m]
STRETCH_KEEP_OUT = 0.28                 # Stretch が床の物から離れて通る距離 [m] (台車の角まで約 0.24 m)
STRETCH_SPEED = 0.25                    # Stretch の前進の上限 [m/s]
TIDY_BOX = (1.15, 5.5, 0.30, 0.25, 0.12)  # 片付けの箱 (x, y [マス]、幅・奥行き・高さ [m])。DK の冷蔵庫の東


class World:
    def __init__(self, house_dir=DEFAULT_HOUSE, duck_pose=(0.0, 0.0, 0.0), n_persons=2,
                 policy=None, objects=None, stretch=None):
        # 既定は、この家で強化学習した歩き方 (models/nobita_walk.onnx) があればそれ、なければ公開の velstand
        default = "nobita_walk" if os.path.exists(os.path.join(ROOT, "models", "nobita_walk.onnx")) else "velstand"
        policy = policy or os.environ.get("DUCK_POLICY", default)
        if policy.startswith("nobita_walk"):
            policy = os.path.join(ROOT, "models", policy + ".onnx")
        self.layout = load_layout(house_dir)
        self.module = self.layout["module"]
        spec, self.door_open_q = compose(house_dir, duck=True, n_persons=n_persons)
        obj_names = add_objects(spec, objects) if objects else []
        if stretch:
            add_stretch(spec, "st_", (stretch[0], stretch[1], 0.0), stretch[2], weld_objects=obj_names)
            self._add_tidy_box(spec, self.module)
        self.duck = Duck(spec, policy=policy)
        self.model, self.data = self.duck.model, self.duck.data
        self.objects = Objects(self.model, self.data, obj_names) if obj_names else None
        self.stretch = None
        if stretch:
            # 数 g の物を指で挟むには滑りの補正 (noslip) が要る (docs/learning/pick_notes.md。アヒルの歩き方は変わらない)
            self.model.opt.noslip_iterations = 3
            self.stretch = Stretch(self.model, self.data, "st_",
                                   weld_assist=os.environ.get("STRETCH_WELD_ASSIST", "1") == "1")
            self.stretch.reset(*stretch)
        m = self.model
        self.n_persons = n_persons
        self.person_state = {}
        self.door_state = {}
        for name, d in self.layout["doors"].items():
            self.set_door(name, d.get("state", "closed") == "open", update_map=False)
        self.duck.reset(*duck_pose)
        duck_root = m.body_rootid[self.duck.trunk]
        ignore = [b for b in range(m.nbody) if m.body_rootid[b] == duck_root]
        if self.stretch:
            ignore += list(self.stretch.bodies)
        self.map = FloorMap(m, self.data, self.layout["rooms"], self.module, ignore_bodies=ignore)
        if self.stretch:
            self.st_map = FloorMap(m, self.data, self.layout["rooms"], self.module, ignore_bodies=ignore,
                                   radius=STRETCH_RADIUS)
            self.st_nav = Navigator(STRETCH_SPEED, 0.0)
            self.st_driving = False
        key = os.path.splitext(os.path.basename(policy))[0]
        self.nav = Navigator(*POLICY_SPEED.get(key, (0.3, 0.0)), min_wz=DUCK_MIN_WZ)
        self.policy_name = key
        self.goal, self.retries, self.last_path = None, 0, None
        self._back_until = None   # 進まなくなって後ろへ下がっている間の終わりの時刻
        self.manual_cmd = (0.0, 0.0, 0.0)
        self.time = 0.0
        self._renderers = {}

    # ---- 時間
    def step(self):
        self._hold_doors()
        if not self.nav.done:
            vx, wz = self.nav.update(*self.duck.pose())
            self.duck.set_cmd(vx, 0.0, wz)
        elif self.nav.stuck and self.goal is not None and self.retries < MAX_RETRIES:
            # 進まなくなった (壁の角に引っかかるなど。pure pursuit は少し先を狙うので角を内側から回る):
            # 少し後ろへ下がってから、いまの位置から経路を作り直す
            if self._back_until is None:
                self._back_until = self.time + BACK_TIME
            if self.time < self._back_until:
                self.duck.set_cmd(BACK_VX, 0.0, 0.0)
            else:
                self._back_until = None
                self.retries += 1
                if not self._plan(*self.goal):
                    self.goal = None
        else:
            self.duck.set_cmd(*self.manual_cmd)
        if self.stretch:
            if not self.st_nav.done:
                self.stretch.drive(*self.st_nav.update(*self.stretch.pose()))
                self.st_driving = True
            elif self.st_driving:
                self.stretch.drive(0.0, 0.0)
                self.st_driving = False
            self.stretch.update()
        if self.objects:
            self._unstick_objects()
        self.duck.step()
        self.time += 0.02

    def _unstick_objects(self):
        """床に置いた物の真ん中が床より下に沈んでいたら、床の上に戻す (シミュレーションの現象の補正。片付けの箱の中は除く)。"""
        bx, by, bw, bd, _ = TIDY_BOX
        bx, by = bx * self.module, by * self.module
        for n, s in self.objects.state.items():
            if s is None:
                continue
            p = self.objects.pos(n)
            if -0.05 < p[2] < -0.001 and not (abs(p[0] - bx) < bw / 2 + 0.05 and abs(p[1] - by) < bd / 2 + 0.05):
                self.objects.unstick(n)

    # ---- 移動
    def go_to(self, x, y, yaw=None, tolerance=None):
        """(x, y) [m] へ向かう経路を作ってアヒルに渡す。経路がなければ False。
        tolerance: (x, y) からこの距離に入れば着いたとする (人のそばなど、その点には立てないとき)。"""
        self.goal, self.retries, self._back_until = (x, y, yaw, tolerance), 0, None
        return self._plan(x, y, yaw, tolerance)

    def stop(self):
        self.goal, self._back_until = None, None
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

    # ---- Stretch (片付け係)
    @staticmethod
    def _add_tidy_box(spec, mod):
        x, y, w, d, h = TIDY_BOX
        b = spec.worldbody.add_body(name="tidy_box", pos=[x * mod, y * mod, 0.0])
        t = 0.01
        tf = 0.03       # 底は厚く (薄い物を落とすと 1 ステップで 1 cm ほど進み、1 cm の底はすり抜ける)
        for name, pos, size in (("floor", [0, 0, tf / 2], [w / 2, d / 2, tf / 2]),
                                ("n", [0, d / 2, h / 2], [w / 2, t / 2, h / 2]), ("s", [0, -d / 2, h / 2], [w / 2, t / 2, h / 2]),
                                ("e", [w / 2, 0, h / 2], [t / 2, d / 2, h / 2]), ("w", [-w / 2, 0, h / 2], [t / 2, d / 2, h / 2])):
            b.add_geom(name=f"tidy_box_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                       rgba=[0.25, 0.45, 0.75, 1], group=1)

    def tidy_box_xyz(self):
        x, y, *_ = TIDY_BOX
        return (x * self.module, y * self.module, 0.0)

    def stretch_go_to(self, x, y, yaw=None, tolerance=0.08):
        """Stretch の台車を (x, y) へ (着いたら yaw を向く)。経路がなければ False。
        床に落ちている物の周り (STRETCH_KEEP_OUT) は通らない。車輪が薄い物 (硬貨・電池) を踏むと、物が床の中に
        めり込んで抜けなくなり (軽い物の接触が振動し続ける MuJoCo の現象)、拾えなくなる。避けると目的地に
        届かないときだけ、ふつうの経路にする。"""
        px, py, _ = self.stretch.pose()
        fm = self.st_map
        free = fm.free
        if self.objects:
            ii, jj = np.meshgrid(np.arange(fm.w), np.arange(fm.h), indexing="ij")
            cx, cy = fm.to_xy(ii, jj)
            keep = free.copy()
            for n in self.objects.names:
                p = self.objects.pos(n)
                if -0.05 < p[2] < 0.03:                  # 床の上 (持っている物・隠してある物は除く)
                    keep &= (cx - p[0]) ** 2 + (cy - p[1]) ** 2 > STRETCH_KEEP_OUT ** 2
            fm.free = keep
        try:
            path = fm.plan((px, py), (x, y))
        finally:
            fm.free = free
        if path is None or math.hypot(path[-1][0] - x, path[-1][1] - y) > 0.15:
            path = fm.plan((px, py), (x, y))
        if path is None:
            return False
        self.st_nav.set_path(path, goal_yaw=yaw, target=(x, y), tolerance=tolerance)
        return True

    def stretch_stop(self):
        self.st_nav.stop()

    def stretch_nav_status(self):
        n = self.st_nav
        return "moving" if not n.done else ("arrived" if n.arrived else ("failed" if n.stuck else "idle"))

    def stretch_pick_pose(self, target):
        """target (x, y) を拾うための台車の (x, y, yaw)。腕の下に来て、台車が地図の通れる所に収まる向きを選ぶ。"""
        best = None
        sx, sy, _ = self.stretch.pose()
        for k in range(16):
            yaw = -math.pi + k * math.pi / 8
            x, y, yaw = self.stretch.arm_side_point(target, yaw)
            c = self.st_map.to_cell(x, y)
            if not (0 <= c[0] < self.st_map.w and 0 <= c[1] < self.st_map.h) or not self.st_map.free[c]:
                continue
            cost = math.hypot(x - sx, y - sy)
            if best is None or cost < best[0]:
                best = (cost, (x, y, yaw))
        return best[1] if best else None

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
