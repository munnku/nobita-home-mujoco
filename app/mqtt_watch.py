"""MQTT に流れているメッセージを覗く (仕組みを理解するため)。

uv run python app/mqtt_watch.py                    # 主なもの (sim/truth と state の毎秒の値は除く)
uv run python app/mqtt_watch.py "homie/+/camera/#"  # 見守りカメラだけ
uv run python app/mqtt_watch.py "uagv/#" --all      # アヒル (VDA 5050) のすべて
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.mqttc import Mqtt  # noqa: E402

NOISY = ("sim/truth", "home/time", "home/clock", "/state", "/visualization")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    pattern = args[0] if args else "#"
    m = Mqtt(f"watch-{os.getpid()}")

    def show(topic, payload):
        if not show_all and any(topic.endswith(n) or topic == n for n in NOISY):
            return
        text = str(payload)
        print(f"{time.strftime('%H:%M:%S')}  {topic}  {text[:160]}{'…' if len(text) > 160 else ''}", flush=True)

    m.subscribe(pattern, show)
    print(f"watching {pattern} (Ctrl+C で終わる)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        m.close()


if __name__ == "__main__":
    main()
