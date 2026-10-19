"""MQTT (paho-mqtt) の薄い包み。JSON で送受信し、受信は topic のパターンごとのコールバックへ渡す。"""
import json
import threading

import paho.mqtt.client as mqtt

HOST = "127.0.0.1"
PORT = 1883


class Mqtt:
    def __init__(self, client_id, host=HOST, port=PORT, will=None):
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.handlers = []          # (pattern, callback(topic, payload))
        self.lock = threading.Lock()
        if will:
            topic, payload = will
            self.c.will_set(topic, self._enc(payload), qos=1, retain=True)
        self.c.on_message = self._on_message
        self.c.on_connect = self._on_connect
        self.c.connect(host, port, keepalive=30)
        self.c.loop_start()

    @staticmethod
    def _enc(payload):
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False)
        if isinstance(payload, bool):
            return "true" if payload else "false"
        return str(payload)

    def publish(self, topic, payload, retain=False, qos=0):
        self.c.publish(topic, self._enc(payload), qos=qos, retain=retain)

    def subscribe(self, pattern, callback):
        """callback(topic, payload)。payload は JSON なら dict/list、そうでなければ文字列。"""
        with self.lock:
            self.handlers.append((pattern, callback))
        self.c.subscribe(pattern, qos=1)

    def _on_connect(self, client, userdata, flags, reason, props):
        with self.lock:
            for pattern, _ in self.handlers:
                client.subscribe(pattern, qos=1)

    def _on_message(self, client, userdata, msg):
        raw = msg.payload.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw
        with self.lock:
            handlers = list(self.handlers)
        for pattern, cb in handlers:
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    cb(msg.topic, payload)
                except Exception as e:  # noqa: BLE001 (受信スレッドを止めない)
                    print(f"[mqtt] handler error on {msg.topic}: {e!r}")

    def close(self):
        self.c.loop_stop()
        self.c.disconnect()
