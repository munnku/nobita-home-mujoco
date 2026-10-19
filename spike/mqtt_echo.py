import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from home.mqttc import Mqtt
got = []
a = Mqtt("a"); b = Mqtt("b")
b.subscribe("test/#", lambda t, p: got.append((t, p)))
time.sleep(0.5)
a.publish("test/x", {"hello": "世界"}); a.publish("test/y", True, retain=True)
time.sleep(0.5)
print(got)
