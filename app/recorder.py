"""動画に残す: 見下ろし (1 階) + アヒルの後ろから + 頭のカメラ + 状況の文字 を 1 枚にして MP4 に書く。"""
import os
import sys

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]


def jp_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Recorder:
    W, H = 1280, 720

    def __init__(self, world, path, fps=10, speedup=1.0):
        self.w = world
        self.fps = fps
        self.period = 1.0 / fps * speedup      # シミュレーションの時間で何秒ごとに 1 コマ
        self.next_t = 0.0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.W, self.H))
        self.top = mujoco.MjvCamera()
        self.top.lookat[:] = [5.46, 3.64, 0]        # 1 階 (東西 10.9 m・南北 7.3 m) の真ん中
        self.top.distance, self.top.azimuth, self.top.elevation = 12.4, 90, -89   # 東西が全部入る距離
        self.font = jp_font(20)
        self.font_small = jp_font(16)
        self.font_title = jp_font(44)
        self.title, self.title_until = "", 0.0    # 物語の章の題 (始まってからしばらくは大きく出す)
        self.lines = []            # 状況の文 (新しいものが下)
        self.head_boxes = []
        self.focus_room, self.focus_until = None, 0.0

    def chapter(self, title, seconds=5.0):
        """物語の章が始まった: 題を見下ろしの絵の真ん中に大きく出し、そのあとは左下に小さく残す。"""
        self.title, self.title_until = title, self.w.time + seconds

    def focus(self, room, seconds=25.0):
        """しばらくその部屋の見守りカメラを出す (司令塔が部屋のことを知らせたとき)。"""
        self.focus_room, self.focus_until = room, self.w.time + seconds

    def log(self, text):
        line = f"[{self.w.time:4.0f}s] "
        for ch in text:                       # 右の欄の幅 (全角 26 字ほど) で折り返す
            if len(line) >= 30:
                self.lines.append(line)
                line = "        "
            line += ch
        self.lines.append(line)
        self.lines = self.lines[-9:]

    def tick(self, devices=None, agent=None):
        if self.w.time < self.next_t:
            return
        self.next_t += self.period      # (時刻 + 間隔 にすると 0.02 s 刻みに丸められ、動画が実時間の 1.2 倍速になった)
        frame = np.zeros((self.H, self.W, 3), np.uint8)
        frame[:, :800] = self.w.render(self.top, 800, 720, groups=(0, 1, 2))
        x, y, _ = self.w.duck.pose()
        room = self.w.room_of(x, y)
        show = self.focus_room if self.w.time < self.focus_until else room
        cam_name = self._room_camera(devices, show) or self._room_camera(devices, room)
        if cam_name:
            # アヒルがいる部屋の見守りカメラ (YOLO の枠と推定した姿勢を重ねる)
            src = devices.latest_images.get(cam_name)
            panel = cv2.resize(src, (480, 270)) if src is not None else np.zeros((270, 480, 3), np.uint8)
            for p in devices.latest.get(cam_name, (0, []))[1]:
                x1, y1, x2, y2 = (int(v * 0.75) for v in p["box"])
                color = (255, 60, 60) if p["posture"] == "lying" else (60, 200, 60)
                cv2.rectangle(panel, (x1, int(y1 * 270 / 360)), (x2, int(y2 * 270 / 360)), color, 2)
            frame[:270, 800:] = panel
        else:
            frame[:270, 800:] = self.w.render(self.w.chase_camera(distance=1.8, elevation=-50), 480, 270)
        frame[270:540, 800:] = self.w.render("head_camera", 480, 270)
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.rectangle([800, 540, 1280, 720], fill=(25, 25, 30))
        status = agent.status_text() if agent else ""
        d.text((10, 8), f"t = {self.w.time:5.1f} s   アヒル: {room or '-'}  {status}", font=self.font, fill=(0, 0, 0))
        if self.title:
            d.text((10, 690), self.title, font=self.font, fill=(0, 0, 0))
            if self.w.time < self.title_until:
                tw = d.textlength(self.title, font=self.font_title)
                d.rectangle([400 - tw / 2 - 24, 300, 400 + tw / 2 + 24, 380], fill=(255, 255, 255),
                            outline=(40, 40, 40), width=3)
                d.text((400 - tw / 2, 312), self.title, font=self.font_title, fill=(20, 20, 20))
        d.text((808, 4), f"見守りカメラ ({cam_name})" if cam_name else "アヒルの上から", font=self.font_small,
               fill=(255, 255, 255))
        d.text((808, 274), "頭のカメラ", font=self.font_small, fill=(255, 255, 255))
        if agent and agent.speech:
            d.rectangle([810, 500, 1270, 536], fill=(255, 255, 220))
            d.text((818, 506), f"アヒル「{agent.speech[1]}」", font=self.font_small, fill=(0, 0, 0))
        for k, line in enumerate(self.lines):
            d.text((808, 546 + k * 19), line, font=self.font_small, fill=(230, 230, 230))
        self.vw.write(np.asarray(img)[..., ::-1])

    @staticmethod
    def _room_camera(devices, room):
        if devices is None or room is None:
            return None
        for c in devices.cam_cfg["cameras"]:
            if c["room"] == room:
                return c["name"]
        return None

    def close(self):
        self.vw.release()
