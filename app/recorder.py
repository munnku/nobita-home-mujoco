"""動画に残す: 見下ろし (1 階) + アヒルの後ろから + 頭のカメラ + 状況の文字 を 1 枚にして MP4 に書く。
左上 (庭の上) にタスク一覧 (home/tasks)、見下ろしの下に保護者のスマホへの最新の知らせ (home/alerts) を重ねる。"""
import math
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
        self.lines = []            # 状況の文 (新しいものが下)
        self.head_boxes = []
        self.focus_room, self.focus_until = None, 0.0
        self.tasks = []             # タスク一覧 (home/tasks)
        self.alert = None           # (時刻, 知らせ) 保護者のスマホへの最新の知らせ

    def focus(self, room, seconds=25.0):
        """しばらくその部屋の見守りカメラを出す (司令塔が部屋のことを知らせたとき)。"""
        self.focus_room, self.focus_until = room, self.w.time + seconds

    QUIET = ("仕事を始める", "仕事が終わった", "タスクに足した")    # タスク一覧の欄に出ているので記録の欄には出さない

    def log(self, text):
        if text.startswith(self.QUIET):
            return
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
        st = self.w.stretch
        arm = st is not None and st.status == "running"      # (終わったあとも phase は残るので status で見る)
        if arm:
            # Stretch が腕で拾う・置く間は、上から小さくしか見えないので後ろから近くで映す
            cam_name = None
            sx, sy, syaw = st.pose()
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [sx, sy, 0.3]
            cam.distance, cam.azimuth, cam.elevation = 2.0, math.degrees(syaw), -55
            frame[:270, 800:] = self.w.render(cam, 480, 270)
        elif cam_name:
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
        label = (f"見守りカメラ ({cam_name})" if cam_name else
                 "片付け係 (Stretch 3)" if arm else "アヒルの上から")
        d.text((808, 4), label, font=self.font_small, fill=(255, 255, 255))
        d.text((808, 274), "頭のカメラ", font=self.font_small, fill=(255, 255, 255))
        self._overlay(img, d)
        if agent and agent.speech:
            # 吹き出し: 26 字ごとに折り返し、下 (頭のカメラの下端) から上へ伸ばす (最大 3 行)
            text = f"アヒル「{agent.speech[1]}」"
            rows = [text[k:k + 26] for k in range(0, len(text), 26)][:3]
            top = 536 - 6 - 20 * len(rows)
            d.rectangle([810, top, 1270, 536], fill=(255, 255, 220))
            for k, row in enumerate(rows):
                d.text((818, top + 4 + 20 * k), row, font=self.font_small, fill=(0, 0, 0))
        for k, line in enumerate(self.lines):
            d.text((808, 546 + k * 19), line, font=self.font_small, fill=(230, 230, 230))
        self.vw.write(np.asarray(img)[..., ::-1])

    CAT = {"予防": (90, 170, 255), "検知": (255, 200, 80), "対応支援": (120, 220, 120), "通知": (255, 140, 140)}
    LEVEL = {"info": (230, 230, 230), "checking": (230, 230, 230), "warn": (255, 210, 60), "emergency": (255, 90, 90)}

    def _overlay(self, img, d):
        """タスク一覧と保護者への知らせを重ねる。"""
        box = Image.new("RGBA", img.size, (0, 0, 0, 0))
        b = ImageDraw.Draw(box)
        items = self.tasks[:5]
        if items:
            h = 26 + 20 * len(items)
            b.rectangle([4, 36, 236, 36 + h], fill=(20, 20, 30, 200))
        al = self.alert if self.alert and self.w.time - self.alert[0] < 25 else None
        if al:
            b.rectangle([0, 692, 800, 720], fill=(20, 20, 30, 220))
        img.paste(Image.alpha_composite(img.convert("RGBA"), box).convert("RGB"))
        if items:
            d.text((10, 39), "タスク一覧 (重い順)", font=self.font_small, fill=(255, 255, 255))
            for k, t in enumerate(items):
                mark = "▶" if t.get("state") == "running" else "・"
                label = str(t.get("label", t.get("name")))[:11]
                d.text((10, 61 + 20 * k), f"{mark}{label}", font=self.font_small,
                       fill=self.CAT.get(t.get("category"), (220, 220, 220)))
        if al:
            a = al[1]
            who = "2人目の家族" if a.get("to") == "family2" else "保護者"
            d.text((8, 696), f"{who}のスマホ [{a.get('level')}] {a.get('summary', '')}"[:46], font=self.font_small,
                   fill=self.LEVEL.get(a.get("level"), (230, 230, 230)))

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
