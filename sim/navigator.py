"""経路 (点の列) をアヒルの速度指令 (vx, wz) に変える (pure pursuit)。

アヒルは指令どおりに進まない (速度は指令の半分ほど、向きもずれる) ので、毎回いまの位置を経路に投影し、
経路に沿って LOOKAHEAD 先の点を狙う。向きが大きくずれていればその場で回り、合っていれば進みながら直す。
進まなくなったら (stuck) 呼び出し側が経路を作り直す。
"""
import math

MAX_VX = 0.3          # 指令の上限 [m/s] (方策の学習の範囲 ±0.4 の内側)
MAX_WZ = 1.0          # [rad/s]
TURN_IN_PLACE = 0.7   # これより向きがずれていたら止まって回る [rad]
LOOKAHEAD = 0.30      # 経路のこれだけ先を狙う [m]
REACH_GOAL = 0.12     # 着いたとみなす距離 [m]
STUCK_TIME = 12.0     # この間に 0.1 m も進まず 0.5 rad も回らなければ stuck


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Navigator:
    def __init__(self):
        self.stop()

    def set_path(self, path, goal_yaw=None):
        """path: [(x, y), ...] (最初の点は今いる所)。goal_yaw: 着いたあとに向く向き (なければ None)。"""
        self.path = [tuple(p) for p in path]
        self.s = 0.0                  # 経路に沿って進んだ距離 (戻らない)
        self.seg_len = [math.dist(a, b) for a, b in zip(self.path[:-1], self.path[1:])]
        self.goal_yaw = goal_yaw
        self.done = False
        self.arrived = False
        self.stuck = False
        self._progress = (0.0, None)  # (最後に進んだときの時刻, 位置)
        self.t = 0.0

    def stop(self):
        self.path, self.seg_len, self.s = [], [], 0.0
        self.goal_yaw, self.done, self.arrived, self.stuck = None, True, False, False
        self._progress, self.t = (0.0, None), 0.0

    def _point_at(self, s):
        for (a, b), ln in zip(zip(self.path[:-1], self.path[1:]), self.seg_len):
            if s <= ln:
                k = s / ln if ln > 0 else 0.0
                return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
            s -= ln
        return self.path[-1]

    def _project(self, x, y):
        """経路上でいまの位置に一番近い点までの距離 s (いまの s より前だけ探す)。"""
        best_s, best_d, acc = self.s, 1e9, 0.0
        for (a, b), ln in zip(zip(self.path[:-1], self.path[1:]), self.seg_len):
            if ln > 0:
                k = max(0.0, min(1.0, ((x - a[0]) * (b[0] - a[0]) + (y - a[1]) * (b[1] - a[1])) / ln ** 2))
                px, py = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
                d = math.hypot(x - px, y - py)
                s = acc + k * ln
                if s >= self.s - 0.05 and s <= self.s + 1.0 and d < best_d:
                    best_s, best_d = s, d
            acc += ln
        return max(self.s, best_s)

    def update(self, x, y, yaw, dt=0.02):
        """(vx, wz) を返す。着いたら done = arrived = True、進まなくなったら stuck = True。"""
        self.t += dt
        if self.done:
            return 0.0, 0.0
        if self.path and not self.arrived:
            gx, gy = self.path[-1]
            if math.hypot(gx - x, gy - y) < REACH_GOAL:
                self.arrived = True
            else:
                t0, p0 = self._progress
                if p0 is None or math.hypot(x - p0[0], y - p0[1]) > 0.1 or abs(wrap(yaw - p0[2])) > 0.5:
                    self._progress = (self.t, (x, y, yaw))
                elif self.t - t0 > STUCK_TIME:
                    self.stuck = True
                    self.done = True
                    return 0.0, 0.0
                self.s = self._project(x, y)
                tx, ty = self._point_at(self.s + LOOKAHEAD)
                if math.hypot(gx - x, gy - y) < LOOKAHEAD:
                    tx, ty = gx, gy
                err = wrap(math.atan2(ty - y, tx - x) - yaw)
                if abs(err) > TURN_IN_PLACE:
                    return 0.0, math.copysign(MAX_WZ, err)
                dist = math.hypot(gx - x, gy - y)
                vx = min(MAX_VX, 0.12 + 0.5 * dist) * max(0.0, math.cos(err))
                wz = max(-MAX_WZ, min(MAX_WZ, 2.0 * err))
                return vx, wz
        # 着いた: 向きを合わせる
        if self.goal_yaw is None:
            self.done = True
            return 0.0, 0.0
        err = wrap(self.goal_yaw - yaw)
        if abs(err) < 0.12:
            self.done = True
            return 0.0, 0.0
        return 0.0, max(-MAX_WZ, min(MAX_WZ, 1.5 * err + math.copysign(0.3, err)))
