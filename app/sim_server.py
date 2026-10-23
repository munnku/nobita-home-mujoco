"""シミュレーションのサーバー: 家 (MuJoCo) + IoT 機器 + アヒル (VDA 5050) + 人役の時刻表 を MQTT につなぐ。

uv run python app/sim_server.py --scenario fall [--speed 1.0] [--viewer] [--record build/videos/fall.mp4]
--speed: 実時間の何倍で進めるか (0 = できるだけ速く)
MQTT: homie/... (家の機器)、uagv/v2/NobitaHome/microduck-01/... (アヒル)、home/time (シミュレーションの時刻)、
      home/speech (アヒルの発話)、sim/scenario・sim/truth (評価用の正解。司令塔は使わない)
"""
import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import yaml  # noqa: E402

from home.devices import HomeDevices  # noqa: E402
from home.mqttc import Mqtt  # noqa: E402
from home.perception import HazardDetector, PersonDetector  # noqa: E402
from home.scenario import Scenario  # noqa: E402
from home.vda5050 import Agent, StretchAgent  # noqa: E402
from objects import CLASSES as OBJECT_KINDS  # noqa: E402
from world import DEFAULT_HOUSE, World  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default=DEFAULT_HOUSE)
    ap.add_argument("--scenario", default="fall")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()

    with open(os.path.join(a.house, "home.yaml"), encoding="utf-8") as f:
        home = yaml.safe_load(f)
    mod = 0.91
    dock = home["robot"]["dock"]
    with open(os.path.join(a.house, "scenarios", f"{a.scenario}.yaml"), encoding="utf-8") as f:
        sc = yaml.safe_load(f)
    st = home["stretch"]["dock"] if sc.get("stretch") else None      # 片付け係の Stretch 3 は使う場面だけ置く (重い)
    w = World(a.house, duck_pose=(dock["x"] * mod, dock["y"] * mod, math.radians(dock["yaw"])),
              n_persons=len(home["people"]), objects={k: 2 for k in OBJECT_KINDS},
              stretch=(st["x"] * mod, st["y"] * mod, math.radians(st["yaw"])) if st else None)
    det = PersonDetector()
    hazard = HazardDetector.load()
    mq = Mqtt("sim-server", will=("sim/status", "offline"))
    devices = HomeDevices(w, mq, det, a.house)
    agent = Agent(w, mq, det, hazard)
    st_agent = StretchAgent(w, mq) if w.stretch else None
    scen = Scenario(w, mq, a.house, a.scenario, devices)
    duration = a.duration or scen.duration
    rec = None
    if a.record:
        from app.recorder import Recorder
        rec = Recorder(w, a.record)
        def on_log(t, p):
            if isinstance(p, dict):
                rec.log(p.get("text", ""))
                if p.get("room"):
                    rec.focus(p["room"])
            else:
                rec.log(str(p))
        mq.subscribe("home/log", on_log)
        mq.subscribe("home/speech", lambda t, p: rec.log(f"[{p.get('speaker')}]「{p.get('text')}」")
                     if isinstance(p, dict) and p.get("speaker") != "microduck-01" else None)
        mq.subscribe("home/voice", lambda t, p: rec.log(f"[{p.get('who')}]「{p.get('text')}」")
                     if isinstance(p, dict) else None)
    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(w.model, w.data)
    control = {}
    mq.subscribe("sim/control", lambda t, p: control.update(cmd=p))      # "stop" で止める (録画を閉じてから終わる)
    mq.publish("sim/status", "running", retain=True)
    print(f"[sim] scenario={a.scenario} duration={duration}s speed={a.speed}", flush=True)
    t_wall0 = time.time()
    next_pub = 0.0
    while w.time < duration and control.get("cmd") != "stop":
        scen.tick()
        devices.tick()
        agent.tick()
        if st_agent:
            st_agent.tick()
        w.step()
        if rec:
            rec.tick(devices, agent)
        if w.time >= next_pub:
            next_pub = w.time + 1.0
            mq.publish("home/time", round(w.time, 1))
            mq.publish("home/clock", scen.clock(), retain=True)
            mq.publish("sim/truth", {"time": round(w.time, 1), "persons": w.person_state,
                                     "objects": {n: [round(v, 2) for v in xy] for n, xy in w.objects.on_floor().items()},
                                     "robot": dict(zip(("x", "y", "theta"), (round(v, 2) for v in w.duck.pose()))),
                                     "fallen": bool(w.duck.fallen()), "recovering": bool(w.duck.recovering),
                                     "duck_z": round(float(w.data.qpos[w.duck.qa + 2]), 3),
                                     "stretch": (dict(zip(("x", "y", "theta"), (round(v, 2) for v in w.stretch.pose())))
                                                 if w.stretch else None)})
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
        if a.speed > 0:
            lag = w.time / a.speed - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(lag)
    mq.publish("sim/status", "finished", retain=True)
    if rec:
        rec.close()
    print(f"[sim] done: sim {w.time:.0f}s in wall {time.time() - t_wall0:.0f}s", flush=True)
    time.sleep(0.5)
    mq.close()


if __name__ == "__main__":
    main()
