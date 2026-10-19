"""WoT の Thing Directory と HTTP の窓口 (FastAPI)。家電とアヒルの TD を出し、HTTP の読み書きを MQTT に直す。

GET  /things                         TD の一覧 (W3C TD 1.1、公式の JSON Schema で検証済み)
GET  /things/{id}                    TD 1 つ
GET  /things/{id}/properties/{name}  いまの値
PUT  /things/{id}/properties/{name}  書く (Homie の /set へ)
POST /things/robot/actions/{name}    アヒルのアクション (話す・一時停止は VDA 5050 の instantActions、
                                     部屋へ行く・確かめる・戻るは司令塔への依頼。終わるまで待って結果を返す)
uv run python app/wot_server.py  (http://127.0.0.1:8080)
"""
import json
import os
import sys
import time
import uuid

import uvicorn
from fastapi import Body, FastAPI, HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.mqttc import Mqtt  # noqa: E402
from home.vda5050 import action, topic  # noqa: E402
from home.wot import HomieCatalog, all_tds  # noqa: E402

BASE = "http://127.0.0.1:8080"
ROOMS = {"ima": "居間", "dk": "DK (台所と食堂)", "ousetsu": "応接室", "rouka": "廊下", "senmen": "洗面所",
         "genkan": "玄関 (の手前)"}
SCHEMA = os.path.join(ROOT, "wot_schemas", "td-json-schema-validation.json")

app = FastAPI(title="のび太君の家 WoT")
mq = Mqtt(f"wot-server-{uuid.uuid4().hex[:6]}")
cat = HomieCatalog(mq)
results = {}
mq.subscribe("home/coordinator/results/+", lambda t, p: results.__setitem__(t.rsplit("/", 1)[-1], p))
_validator = None


def validator():
    global _validator
    if _validator is None:
        import jsonschema
        with open(SCHEMA, encoding="utf-8") as f:
            _validator = jsonschema.Draft7Validator(json.load(f))
    return _validator


def tds():
    return all_tds(cat, ROOMS, BASE)


def find(tid):
    for td in tds():
        if td["id"].split(":")[-1] == tid:
            return td
    raise HTTPException(404, f"no thing {tid}")


@app.get("/things")
def list_things():
    return tds()


@app.get("/things/validate")
def validate():
    """すべての TD を W3C の TD 1.1 の JSON Schema で検証する。"""
    out = {}
    for td in tds():
        errs = [e.message for e in validator().iter_errors(td)]
        out[td["id"]] = errs or "ok"
    return out


@app.get("/things/{tid}")
def get_thing(tid: str):
    return find(tid)


@app.get("/things/{tid}/properties/{name}")
def read_property(tid: str, name: str):
    if tid == "robot" and name == "state":
        s = cat.robot_state
        return {k: s.get(k) for k in ("agvPosition", "room", "driving", "paused", "orderId", "actionStates",
                                      "errors") if k in s}
    p = find(tid)["properties"].get(name)
    if not p:
        raise HTTPException(404)
    return {"value": cat.get(p["homie:topic"])}


@app.put("/things/{tid}/properties/{name}")
def write_property(tid: str, name: str, value=Body(...)):
    p = find(tid)["properties"].get(name)
    if not p or p.get("readOnly"):
        raise HTTPException(400, "not writable")
    mq.publish(p["homie:topic"] + "/set", value)
    return {"ok": True, "value": value}


@app.post("/things/robot/actions/{name}")
def robot_action(name: str, args: dict = Body(default={})):
    if name in ("speak", "startPause", "stopPause"):
        aid = f"wot-{uuid.uuid4().hex[:8]}"
        mq.publish(topic("instantActions"), {"headerId": 0, "actions": [action(name, aid, **args)]}, qos=1)
        return {"ok": True, "actionId": aid}
    if name not in ("goToRoom", "checkRoom", "returnToDock"):
        raise HTTPException(404)
    rid = uuid.uuid4().hex[:8]
    mq.publish("home/coordinator/requests", {"id": rid, "type": name, "args": args}, qos=1)
    t0 = time.time()
    while time.time() - t0 < 600:
        if rid in results:
            return results.pop(rid)
        time.sleep(0.2)
    raise HTTPException(504, "timeout")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
