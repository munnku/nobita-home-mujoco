"""物語の録画を、場面ごとのファイルに切り分ける (動画の編集用)。

uv run python app/cut_clips.py --story orusuban_afternoon --video build/videos/orusuban_afternoon_v3.mp4 [--run build/runs/....json]
切り方は houses/nobita/stories/<物語>_clips.yaml。印 (司令塔のログの kind・扉の操作) の時刻は run_scenario.py の結果
(--run、無ければその物語のいちばん新しいもの) から取る。録画はシミュレーションの時刻と同じ時間軸 (0 秒から 15 fps)。
結果: build/videos/<物語>_clips/NN_<名前>_x<倍速>.mp4 と 構成.md (各ファイルの長さ・勧める倍速・倍速後の合計)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = re.compile(r"^(?P<name>[^#+-]+?)(?:#(?P<n>\d+))?(?P<off>[+-]\d+(?:\.\d+)?)?$")


def marks(events):
    """印の名前 → 起きた時刻の並び。"""
    out = {}
    for e in events:
        p = e["payload"]
        if not isinstance(p, dict):
            continue
        if e["topic"] == "home/log" and p.get("kind"):
            out.setdefault(p["kind"], []).append(p["time"])
        elif e["topic"] == "sim/scenario" and "door" in p:
            out.setdefault(f"door:{p['door']}:{p['do']}", []).append(p["time"])
    return out


def resolve(mark, found):
    m = MARK.match(mark)
    if not m:
        raise ValueError(f"印の書き方がおかしい: {mark}")
    times = found.get(m["name"], [])
    n = int(m["n"] or 1)
    if len(times) < n:
        raise ValueError(f"印 {m['name']} の {n} 回目が録画の結果に無い")
    return times[n - 1] + float(m["off"] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="orusuban_afternoon")
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--house", default=os.path.join(ROOT, "houses", "nobita"))
    ap.add_argument("--dry", action="store_true", help="切らずに長さだけ出す")
    a = ap.parse_args()
    run = a.run or max(glob.glob(os.path.join(ROOT, "build", "runs", f"{a.story}_*.json")), key=os.path.getmtime)
    with open(run, encoding="utf-8") as f:
        found = marks(json.load(f)["events"])
    with open(os.path.join(a.house, "stories", f"{a.story}_clips.yaml"), encoding="utf-8") as f:
        clips = yaml.safe_load(f)["clips"]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg が見つからない")
    out_dir = os.path.join(ROOT, "build", "videos", f"{a.story}_clips")
    os.makedirs(out_dir, exist_ok=True)
    for old in [] if a.dry else glob.glob(os.path.join(out_dir, "*.mp4")):     # 前に切ったものは消して切り直す
        os.remove(old)
    rows, total, raw = [], 0.0, 0.0
    for k, c in enumerate(clips, 1):
        t0, t1 = resolve(c["from"], found), resolve(c["to"], found)
        name = f"{k:02d}_{c['name']}_x{c['speed']:g}.mp4"
        if not a.dry:
            # 切り口をコマ単位で合わせるため、コピーではなく H.264 で書き直す (編集ソフトでも開きやすい)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", a.video,
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                            os.path.join(out_dir, name)], check=True)
        after = (t1 - t0) / c["speed"]
        total, raw = total + after, raw + t1 - t0
        rows.append(f"| {name} | {t0:.1f}–{t1:.1f} s | {t1 - t0:.1f} s | ×{c['speed']:g} | {after:.1f} s | {c.get('note', '')} |")
        print(f"{name}  {t0:6.1f}-{t1:6.1f}s  {t1 - t0:5.1f}s ×{c['speed']:g} → {after:5.1f}s", flush=True)
    with open(os.path.join(out_dir, "構成.md"), "w", encoding="utf-8") as f:
        f.write(f"# {a.story} の切り分け\n\n録画: {os.path.basename(a.video)} / 印: {os.path.basename(run)}\n\n")
        f.write("| ファイル | 録画の中の時刻 | 長さ | 勧める倍速 | 倍速後 | 映っていること |\n|---|---|---|---|---|---|\n")
        f.write("\n".join(rows))
        f.write(f"\n\n切り出した長さの合計 {raw:.0f} s → 勧める倍速を掛けると **{total:.0f} s** (3 分 = 180 s)。"
                "ファイルの間は捨てたところ (動きの無い待ち)。\n")
    print(f"合計 {raw:.0f}s → 倍速後 {total:.0f}s  ({out_dir})")


if __name__ == "__main__":
    main()
