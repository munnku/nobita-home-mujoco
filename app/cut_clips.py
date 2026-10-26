"""物語の録画を、場面ごとのファイルに切り分ける (動画の編集用)。

uv run python app/cut_clips.py --story orusuban_afternoon --video build/videos/orusuban_afternoon_v3.mp4 [--run build/runs/....json]
切り方は houses/home/stories/<物語>_clips.yaml。印 (司令塔のログの kind・扉の操作) の時刻は run_scenario.py の結果
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


SQUEEZE_GAP = 4.0     # squeeze: 声も発話もない間がこれより長ければ詰める [s]
SQUEEZE_KEEP = 1.0    # 詰めるとき、間の前後に残す長さ [s]


def talk_spans(events):
    """だれかが話している間 [(始め, 終わり)] (アヒル・スピーカー・インターホンの発話と、人役の声)。"""
    out = []
    for e in events:
        p = e["payload"]
        if not isinstance(p, dict) or "time" not in p or not p.get("text"):
            continue
        if e["topic"] == "home/speech":
            out.append((p["time"], p["time"] + float(p.get("duration", 2.0))))
        elif e["topic"] == "home/voice":
            out.append((p["time"], p["time"] + 1.0 + 0.15 * len(p["text"])))
    return sorted(out)


def squeeze(t0, t1, spans):
    """[t0, t1] のうち、話していない長い間 (LLM が考えている間) を詰めて、残す区間 [(始め, 終わり)] を返す。"""
    keep, cur = [], t0
    busy = [(max(a, t0), min(b, t1)) for a, b in spans if b > t0 and a < t1]
    gaps, last = [], t0
    for a, b in busy:
        if a - last > SQUEEZE_GAP:
            gaps.append((last, a))
        last = max(last, b)
    if t1 - last > SQUEEZE_GAP:
        gaps.append((last, t1))
    for g0, g1 in gaps:
        keep.append((cur, g0 + SQUEEZE_KEEP))
        cur = g1 - SQUEEZE_KEEP
    keep.append((cur, t1))
    return [(a, b) for a, b in keep if b - a > 0.05] or [(t0, t1)]


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
    ap.add_argument("--house", default=os.path.join(ROOT, "houses", "home"))
    ap.add_argument("--dry", action="store_true", help="切らずに長さだけ出す")
    a = ap.parse_args()
    run = a.run or max(glob.glob(os.path.join(ROOT, "build", "runs", f"{a.story}_*.json")), key=os.path.getmtime)
    with open(run, encoding="utf-8") as f:
        events = json.load(f)["events"]
    found, spans = marks(events), talk_spans(events)
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
        # squeeze: true の場面は、LLM が考えている間 (話していない長い間) を詰める
        segs = squeeze(t0, t1, spans) if c.get("squeeze") else [(t0, t1)]
        kept = sum(b - a for a, b in segs)
        if not a.dry:
            # 切り口をコマ単位で合わせるため、コピーではなく H.264 で書き直す (編集ソフトでも開きやすい)
            parts = "".join(f"[0:v]trim=start={x - t0:.2f}:end={y - t0:.2f},setpts=PTS-STARTPTS[v{i}];"
                            for i, (x, y) in enumerate(segs))
            graph = parts + "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[out]"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", a.video,
                            "-filter_complex", graph, "-map", "[out]",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                            os.path.join(out_dir, name)], check=True)
        after = kept / c["speed"]
        if kept < t1 - t0 - 0.5:
            c["note"] = (c.get("note", "") + f" (LLM が考える間 {t1 - t0 - kept:.0f} s を詰めた)").strip()
        total, raw = total + after, raw + kept
        rows.append(f"| {name} | {t0:.1f}–{t1:.1f} s | {kept:.1f} s | ×{c['speed']:g} | {after:.1f} s | {c.get('note', '')} |")
        print(f"{name}  {t0:6.1f}-{t1:6.1f}s  {kept:5.1f}s ×{c['speed']:g} → {after:5.1f}s", flush=True)
    with open(os.path.join(out_dir, "構成.md"), "w", encoding="utf-8") as f:
        f.write(f"# {a.story} の切り分け\n\n録画: {os.path.basename(a.video)} / 印: {os.path.basename(run)}\n\n")
        f.write("| ファイル | 録画の中の時刻 | 長さ | 勧める倍速 | 倍速後 | 映っていること |\n|---|---|---|---|---|---|\n")
        f.write("\n".join(rows))
        f.write(f"\n\n切り出した長さの合計 {raw:.0f} s → 勧める倍速を掛けると **{total:.0f} s** (3 分 = 180 s)。"
                "ファイルの間は捨てたところ (動きの無い待ち)。\n"
                "対話の場面は、Claude Pro (claude -p) が考える間 (1 回 10 秒ほど、話していない間) を詰めてある。\n")
    print(f"合計 {raw:.0f}s → 倍速後 {total:.0f}s  ({out_dir})")


if __name__ == "__main__":
    main()
