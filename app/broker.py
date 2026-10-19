"""MQTT ブローカー (amqtt、Python だけで動く)。127.0.0.1:1883。

uv run python app/broker.py
"""
import asyncio
import logging

from amqtt.broker import Broker

CONFIG = {
    "listeners": {"default": {"type": "tcp", "bind": "127.0.0.1:1883", "max_connections": 50}},
    "sys_interval": 0,
    "auth": {"allow-anonymous": True},
    "topic-check": {"enabled": False},
}


async def main():
    broker = Broker(CONFIG)
    await broker.start()
    print("MQTT broker on 127.0.0.1:1883", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
