"""家をずっと動かしておく: ブローカー + シミュレーション + 司令塔 + WoT サーバー。頼みごとを試すとき用。

uv run python app/start_home.py [--scenario daytime] [--speed 1] [--viewer] [--record build/videos/x.mp4]
止めるとき: Ctrl+C か、別の端末で `uv run python app/start_home.py --stop`
(止めるときはシミュレーションに stop を送り、録画を閉じてから終わる)
"""
import argparse
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def stop():
    from home.mqttc import Mqtt
    m = Mqtt("start-home-stop")
    m.publish("sim/control", "stop", qos=1)
    m.publish("home/control", "stop", qos=1)
    time.sleep(1)
    m.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="daytime")
    ap.add_argument("--speed", default="1")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()
    if a.stop:
        stop()
        return
    procs = []
    if not port_open(1883):
        procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "broker.py")]))
        for _ in range(50):
            if port_open(1883):
                break
            time.sleep(0.1)
    from home.mqttc import Mqtt
    m = Mqtt("start-home")
    ctl = {}
    m.subscribe("home/control", lambda t, p: ctl.update(cmd=p))
    procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "coordinator.py")]))
    cmd = [PY, os.path.join(ROOT, "app", "sim_server.py"), "--scenario", a.scenario, "--speed", a.speed]
    if a.viewer:
        cmd.append("--viewer")
    if a.record:
        cmd += ["--record", a.record]
    sim = subprocess.Popen(cmd)
    procs.append(sim)
    if not port_open(8080):
        procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "wot_server.py")]))
    print("[start_home] 起動しました。WoT: http://127.0.0.1:8080/things  (止める: Ctrl+C または --stop)", flush=True)
    try:
        while sim.poll() is None and ctl.get("cmd") != "stop":
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        m.publish("sim/control", "stop", qos=1)
        try:
            sim.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        for p in procs:
            if p.poll() is None:
                p.terminate()
        print("[start_home] 止めました", flush=True)


if __name__ == "__main__":
    main()
