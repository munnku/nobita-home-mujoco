"""場面を 1 回通しで動かして結果をまとめる: ブローカー (なければ起動) + 司令塔 + シミュレーション。

uv run python app/run_scenario.py --scenario fall [--speed 0] [--record build/videos/fall.mp4] [--viewer]
結果: 標準出力と build/runs/<場面>_<時刻>.json (出来事の時刻、知らせ、アヒルの位置など)
"""
import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.mqttc import Mqtt  # noqa: E402

PY = sys.executable


def port_open(port=1883):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="fall")
    ap.add_argument("--story", default=None, help="一続きの物語 (houses/home/stories/<名前>.yaml)")
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--record", default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()

    procs = []
    if not port_open():
        procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "broker.py")]))
        for _ in range(50):
            if port_open():
                break
            time.sleep(0.1)
    events = []
    m = Mqtt("run-scenario")
    for pat in ("home/alerts", "home/log", "home/speech", "home/voice", "sim/scenario"):
        m.subscribe(pat, lambda t, p: events.append({"topic": t, "payload": p}))
    truth, track = {}, []
    def on_truth(t, p):
        truth.update(p)
        track.append({"time": p.get("time"), "robot": p.get("robot"), "fallen": p.get("fallen"),
                      "recovering": p.get("recovering"), "duck_z": p.get("duck_z"), "duck_cmd": p.get("duck_cmd"),
                      "nav": p.get("nav"), "stretch": p.get("stretch")})
    m.subscribe("sim/truth", on_truth)
    status = {}
    m.subscribe("sim/status", lambda t, p: status.update(s=p))
    m.publish("sim/status", "starting", retain=True)
    coord = subprocess.Popen([PY, os.path.join(ROOT, "app", "coordinator.py")])
    procs.append(coord)
    time.sleep(1.0)
    cmd = [PY, os.path.join(ROOT, "app", "sim_server.py"), "--speed", str(a.speed)]
    cmd += ["--story", a.story] if a.story else ["--scenario", a.scenario]
    if a.record:
        cmd += ["--record", a.record]
    if a.viewer:
        cmd.append("--viewer")
    if a.duration:
        cmd += ["--duration", str(a.duration)]
    sim = subprocess.Popen(cmd)
    sim.wait()
    time.sleep(1.0)
    coord.terminate()
    # まとめ
    out_dir = os.path.join(ROOT, "build", "runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = a.story or a.scenario
    path = os.path.join(out_dir, f"{name}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scenario": name, "events": events, "final_truth": truth, "robot_track": track}, f, ensure_ascii=False, indent=1)
    print("==== 結果 ====")
    for e in events:
        p = e["payload"]
        if e["topic"] == "home/log":
            print(f"  {p.get('time', 0):6.1f}s  {p.get('text')}")
        elif e["topic"] == "home/alerts":
            print(f"  {p.get('time', 0):6.1f}s  [知らせ] {p}")
        elif e["topic"] == "home/voice":
            print(f"  {p.get('time', 0):6.1f}s  [{p.get('who')}]「{p.get('text')}」")
        elif e["topic"] == "home/speech":
            who = "アヒル" if p.get("speaker") == "microduck-01" else p.get("speaker")
            print(f"  {p.get('time', 0):6.1f}s  [{who}] {p.get('text')}")
        elif e["topic"] == "sim/scenario" and "chapter" in p:
            print(f"  {p.get('time', 0):6.1f}s  ==== 章 {p['chapter']}: {p.get('title') or ''}")
        elif e["topic"] == "sim/scenario" and "door" in p:
            print(f"  {p.get('time', 0):6.1f}s  [人役] 扉 {p.get('door')}: {p.get('do')}")
        elif e["topic"] == "sim/scenario":
            print(f"  {p.get('time', 0):6.1f}s  [人役] {p.get('who')} → {p.get('at')} ({p.get('pose')})")
    print("saved", path)
    for p in procs:
        if p.poll() is None and p is not coord:
            p.terminate()
    m.close()


if __name__ == "__main__":
    main()
