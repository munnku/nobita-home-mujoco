"""照明: 人感センサーが LIGHT_OFF_AFTER 秒続けて「人がいない」部屋の照明がついていれば消す (消し忘れ)。
人が入ってきたら (人感センサーが反応したら) つける。"""
from .base import LABEL, Rule

LIGHT_OFF_AFTER = 60.0  # 人のいない部屋の照明をこれだけ待って消す [s]


class Lights(Rule):
    def __init__(self, c):
        super().__init__(c)
        self.empty_since = {}

    def step(self):
        c = self.c
        for room, on in list(c.light_on.items()):
            occupied = c.motion.get(room)
            if occupied and not on:
                c.light(room, True)
                c.light_on[room] = True
                c.log(f"{LABEL.get(room, room)}に人が来た (人感センサー) → 照明をつけた", kind="light_on")
                continue
            if not on or occupied is None or occupied:
                self.empty_since.pop(room, None)
                continue
            since = self.empty_since.setdefault(room, c.time)
            if c.time - since >= LIGHT_OFF_AFTER:
                c.light(room, False)
                c.light_on[room] = False
                self.empty_since.pop(room, None)
                c.log(f"{LABEL.get(room, room)}に {LIGHT_OFF_AFTER:.0f} 秒だれもいない → 照明を消した", kind="light_off")
