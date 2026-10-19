"""WoT と LLM を場面と一緒に動かして確かめる: ブローカー + シミュレーション (daytime) + 司令塔 + WoT サーバーを起動し、
llm_agent に頼みごとを順に渡す。uv run python app/demo_wot.py ["頼みごと" ...] [--offline]
"""
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REQUESTS = ["応接室の様子を見てきて", "居間の照明を消して", "玄関の鍵を閉めて", "アヒルはどこにいる？", "戻って"]


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    procs = []
    if not port_open(1883):
        procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "broker.py")]))
        time.sleep(2)
    procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "coordinator.py")]))
    procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "sim_server.py"), "--scenario", "daytime",
                                   "--speed", "2", "--record", os.path.join(ROOT, "build", "videos", "wot.mp4")]))
    procs.append(subprocess.Popen([PY, os.path.join(ROOT, "app", "wot_server.py")]))
    for _ in range(120):
        if port_open(8080):
            break
        time.sleep(0.5)
    time.sleep(15)      # 機器の記述が揃うのを待つ
    try:
        for req in args or REQUESTS:
            print(f"\n=== 頼みごと: {req}", flush=True)
            subprocess.run([PY, os.path.join(ROOT, "app", "llm_agent.py"), *flags, req])
    finally:
        sys.path.insert(0, ROOT)
        from home.mqttc import Mqtt
        m = Mqtt("demo-wot")
        m.publish("sim/control", "stop", qos=1)
        time.sleep(5)
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
