"""1 階の通れる場所の地図 (グリッド) を MuJoCo のモデルから作り、A* で経路を探す。

- 通れる床: 上面が 1 階の床の高さ (z≈0) の床の geom (玄関の土間や外は段差なので入らない)
- 障害物: アヒルの高さ (床から 0.01〜0.35 m) にかかる当たり判定のある geom (壁・家具・閉じた扉・人)。
  扉の開け閉めや人の置き場所が変わったら作り直す (update)
- アヒルの半径 + 余裕だけ障害物を膨らませ、壁から離れた道を好むように距離に応じた費用を足す
LiDAR を使わず、間取りから作った地図をそのまま使う (家の地図は事前に分かっている前提)。
"""
import heapq
import math

import mujoco
import numpy as np

RES = 0.05            # 1 マス 5 cm
ROBOT_RADIUS = 0.15   # アヒルの足元の半径 + 余裕
CLEARANCE = 0.30      # これより壁に近いと費用が増える
BAND = (0.01, 0.35)   # アヒルがぶつかる高さ
SIGHT_CLEARANCE = 0.15  # 人を見る線から家具までの距離がこれより近いと見通しが悪いとする


def _geom_corners(model, data, g):
    """箱・円柱などの geom の床への投影を、4 隅の点で返す (箱以外は外接する箱)。"""
    t = model.geom_type[g]
    size = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        hx, hy = size[0], size[1]
    elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
        hx = hy = size[0]
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        hx = hy = size[0] + size[1]
    else:
        return None
    R = data.geom_xmat[g].reshape(3, 3)
    c = data.geom_xpos[g]
    pts = [c + R @ np.array([sx * hx, sy * hy, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts)[:, :2]


def _zrange(model, data, g):
    t = model.geom_type[g]
    size = model.geom_size[g]
    R = data.geom_xmat[g].reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        hz = abs(R[2, 0]) * size[0] + abs(R[2, 1]) * size[1] + abs(R[2, 2]) * size[2]
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        hz = abs(R[2, 2]) * size[1] + size[0]
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        hz = abs(R[2, 2]) * size[1] + size[0] * math.hypot(R[2, 0], R[2, 1])
    else:
        hz = size[0]
    z = data.geom_xpos[g][2]
    return z - hz, z + hz


class FloorMap:
    def __init__(self, model, data, rooms, module, ignore_bodies=()):
        """rooms: layout.yaml の rooms (1 階のものだけ使う)。ignore_bodies: アヒルなど地図に入れない body の id。"""
        self.model, self.data = model, data
        self.rooms = [r for r in rooms if r["level"] == "1F"]
        self.module = module
        self.ignore = set(ignore_bodies)
        xs = [r["rect"][0] for r in self.rooms] + [r["rect"][1] for r in self.rooms]
        ys = [r["rect"][2] for r in self.rooms] + [r["rect"][3] for r in self.rooms]
        self.x0, self.y0 = min(xs) * module - 0.2, min(ys) * module - 0.2
        self.w = int((max(xs) * module + 0.2 - self.x0) / RES) + 1
        self.h = int((max(ys) * module + 0.2 - self.y0) / RES) + 1
        self.update()

    # ---- 座標の変換
    def to_cell(self, x, y):
        return int((x - self.x0) / RES), int((y - self.y0) / RES)

    def to_xy(self, i, j):
        return self.x0 + (i + 0.5) * RES, self.y0 + (j + 0.5) * RES

    def _fill(self, grid, pts, grow=0.0):
        lo = pts.min(axis=0) - grow
        hi = pts.max(axis=0) + grow
        i0, j0 = self.to_cell(*lo)
        i1, j1 = self.to_cell(*hi)
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, self.w - 1), min(j1, self.h - 1)
        if i1 < i0 or j1 < j0:
            return
        # 回った箱 (開いた扉) もあるので、4 辺の内側にあるセルを塗る
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px = self.x0 + (ii + 0.5) * RES
        py = self.y0 + (jj + 0.5) * RES
        inside = np.ones_like(px, dtype=bool)
        quad = pts[[0, 1, 3, 2]]          # 4 隅を一周する順に並べる
        ccw = self._ccw(quad)
        tol = grow
        for k in range(4):
            ax, ay = quad[k]
            bx, by = quad[(k + 1) % 4]
            ln = math.hypot(bx - ax, by - ay) or 1.0
            sd = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / ln * (1 if ccw else -1)
            inside &= sd >= -tol - 1e-9
        grid[i0:i1 + 1, j0:j1 + 1][inside] = True

    @staticmethod
    def _ccw(q):
        s = 0.0
        for k in range(4):
            s += q[k][0] * q[(k + 1) % 4][1] - q[(k + 1) % 4][0] * q[k][1]
        return s > 0

    def update(self):
        """床・障害物から地図を作り直す (扉の開閉や人の移動の後に呼ぶ)。"""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        floor = np.zeros((self.w, self.h), bool)
        obst = np.zeros((self.w, self.h), bool)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.ignore:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            pts = _geom_corners(m, d, g)
            if pts is None:
                continue
            zlo, zhi = _zrange(m, d, g)
            if name.startswith("floor_") and abs(zhi) < 0.01:
                self._fill(floor, pts)
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            if name.startswith("floor_") or name.startswith("outside_"):
                continue
            if zhi > BAND[0] and zlo < BAND[1]:
                self._fill(obst, pts, grow=RES * 0.75)  # 薄い物 (扉の板) がセルの間をすり抜けないように
        self.floor = floor
        self.obstacle = obst | ~floor
        # 障害物からの距離 (セル) を求めて、膨らませと費用に使う
        self.dist = self._distance(self.obstacle) * RES
        self.free = self.dist > ROBOT_RADIUS

    @staticmethod
    def _distance(obst):
        """障害物からのユークリッド距離 (セル単位)。scipy の距離変換を使う。"""
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~obst)

    # ---- 部屋
    def room_of(self, x, y):
        for r in self.rooms:
            x0, x1, y0, y1 = (v * self.module for v in r["rect"])
            if x0 <= x <= x1 and y0 <= y <= y1:
                return r["name"]
        return None

    def nearest_free(self, x, y, max_r=1.0):
        i, j = self.to_cell(x, y)
        best, bd = None, 1e9
        n = int(max_r / RES)
        for di in range(-n, n + 1):
            for dj in range(-n, n + 1):
                a, b = i + di, j + dj
                if 0 <= a < self.w and 0 <= b < self.h and self.free[a, b]:
                    dd = di * di + dj * dj
                    if dd < bd:
                        best, bd = (a, b), dd
        return best

    # ---- 経路
    def plan(self, start, goal):
        """start, goal: (x, y) [m]。通れる経路の点列 [m] (間引いたもの) か None を返す。"""
        s = self.nearest_free(*start, max_r=0.5)
        g = self.nearest_free(*goal)
        if s is None or g is None:
            return None
        walk = self.reach(start)
        if not np.isfinite(walk[g]):
            # 目的地のそばへは行けない (人や家具でふさがれている): 歩いて行ける所のうち目的地に一番近い所へ
            gi, gj = self.to_cell(*goal)
            ii, jj = np.nonzero(np.isfinite(walk))
            if len(ii) == 0:
                return None
            k = int(np.argmin((ii - gi) ** 2 + (jj - gj) ** 2))
            g = (int(ii[k]), int(jj[k]))
        cost_extra = np.clip(CLEARANCE - self.dist, 0, None) / CLEARANCE * 4.0
        openq = [(0.0, s)]
        came = {s: None}
        gscore = {s: 0.0}
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for di, dj, c in steps:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.w and 0 <= nb[1] < self.h) or not self.free[nb]:
                    continue
                ng = gscore[cur] + c * (1.0 + cost_extra[nb])
                if ng < gscore.get(nb, 1e18):
                    gscore[nb] = ng
                    came[nb] = cur
                    h = math.hypot(nb[0] - g[0], nb[1] - g[1])
                    heapq.heappush(openq, (ng + h, nb))
        if g not in came:
            return None
        cells = []
        cur = g
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells.reverse()
        return [self.to_xy(*c) for c in self._shortcut(cells)]

    def _line_free(self, a, b, margin=ROBOT_RADIUS + 0.02):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for k in range(n + 1):
            i = round(a[0] + (b[0] - a[0]) * k / n)
            j = round(a[1] + (b[1] - a[1]) * k / n)
            if self.dist[i, j] < margin:
                return False
        return True

    def _shortcut(self, cells):
        """見通せる点は飛ばして、曲がり角だけの経路にする。"""
        out = [cells[0]]
        k = 0
        while k < len(cells) - 1:
            nxt = k + 1
            for m in range(len(cells) - 1, k, -1):
                if self._line_free(cells[k], cells[m]):
                    nxt = m
                    break
            out.append(cells[nxt])
            k = nxt
        return out

    def reach(self, start):
        """start から通れるセルをたどった歩く距離 [m] の地図 (行けないセルは inf)。"""
        dist = np.full((self.w, self.h), np.inf)
        s = self.nearest_free(*start, max_r=0.5)
        if s is None:
            return dist
        dist[s] = 0.0
        heap = [(0.0, s)]
        steps = [(1, 0, RES), (-1, 0, RES), (0, 1, RES), (0, -1, RES),
                 (1, 1, RES * 1.414), (1, -1, RES * 1.414), (-1, 1, RES * 1.414), (-1, -1, RES * 1.414)]
        while heap:
            d, (i, j) = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, c in steps:
                a, b = i + di, j + dj
                if 0 <= a < self.w and 0 <= b < self.h and self.free[a, b] and d + c < dist[a, b]:
                    dist[a, b] = d + c
                    heapq.heappush(heap, (d + c, (a, b)))
        return dist

    def viewpoint(self, target, start, r_min=0.8, r_max=1.3, ignore_near=0.6, need_sight=True, area=None):
        """target [m] を見る立ち位置を選ぶ。start から歩いて行けるセルのうち、target から r_min〜r_max で、
        target までの線分 (target の近く ignore_near は人自身なので除く) の近くに家具が無く (need_sight)、
        歩く距離が短いもの。area: [x0, x1, y0, y1] [m] を渡すとその中 (target の部屋) だけから選ぶ。"""
        walk = self.reach(start)
        ti, tj = self.to_cell(*target)
        best, best_cost = None, 1e18
        n = int(r_max / RES) + 1
        for di in range(-n, n + 1):
            for dj in range(-n, n + 1):
                r = math.hypot(di, dj) * RES
                i, j = ti + di, tj + dj
                if not (r_min <= r <= r_max and 0 <= i < self.w and 0 <= j < self.h and np.isfinite(walk[i, j])):
                    continue
                x, y = self.to_xy(i, j)
                if area and not (area[0] <= x <= area[1] and area[2] <= y <= area[3]):
                    continue
                if need_sight and not self._sight(i, j, ti, tj, ignore_near):
                    continue
                cost = walk[i, j] + 2.0 * abs(r - (r_min + r_max) / 2)
                if cost < best_cost:
                    best, best_cost = (x, y), cost
        return best

    def _sight(self, i0, j0, i1, j1, ignore_near):
        n = int(max(abs(i1 - i0), abs(j1 - j0))) + 1
        stop = int(ignore_near / RES)
        for k in range(n + 1):
            i = round(i0 + (i1 - i0) * k / n)
            j = round(j0 + (j1 - j0) * k / n)
            if math.hypot(i - i1, j - j1) <= stop:
                return True
            if self.dist[i, j] < SIGHT_CLEARANCE:      # 見る線の近くに家具などがあれば見通しが悪い
                return False
        return True

    def image(self, path=None, marks=()):
        """確認用の絵 (北が上)。白 = 通れる、灰 = 壁の近く、黒 = 障害物。"""
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[...] = 40
        near = (~self.obstacle) & (~self.free)
        img[near.T] = (150, 150, 150)
        img[self.free.T] = (255, 255, 255)
        img = img[::-1].copy()

        def px(x, y):
            i, j = self.to_cell(x, y)
            return i, self.h - 1 - j
        if path:
            import cv2
            for a, b in zip(path[:-1], path[1:]):
                cv2.line(img, px(*a), px(*b), (220, 40, 40), 2)
            for p in path:
                cv2.circle(img, px(*p), 3, (40, 40, 220), -1)
        for x, y in marks:
            import cv2
            cv2.circle(img, px(x, y), 5, (40, 160, 40), -1)
        return img
