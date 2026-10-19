"""家電とロボットの W3C WoT Thing Description (TD 1.1) を、MQTT に出ている標準の記述から作る (手で書かない)。

- 家電: Homie 4.0 の機器の記述 ($name, $nodes, $properties, $datatype, $settable) → TD の properties
- ロボット: VDA 5050 の factsheet (agvActions) → TD の actions (話す、一時停止など、その場でできるもの)
           + 司令塔に頼む仕事 (部屋へ行く・部屋を確かめる・戻る)。VDA 5050 ではロボットに order を出すのは
             司令塔 (master control) 1 つなので、LLM からの移動の頼みは司令塔を通す
フォーム: HTTP (app/wot_server.py) と MQTT (WoT Binding Templates の MQTT。mqv:topic)
"""
import threading

TD_CONTEXT = ["https://www.w3.org/2022/wot/td/v1.1",
              {"mqv": "http://www.example.org/mqtt-binding#",
               "htv": "http://www.w3.org/2011/http#",
               "vda5050": "https://github.com/VDA5050/VDA5050#",
               "homie": "https://homieiot.github.io/specification/#"}]
NOSEC = {"securityDefinitions": {"nosec_sc": {"scheme": "nosec"}}, "security": "nosec_sc"}
HOMIE_TYPES = {"boolean": "boolean", "integer": "integer", "float": "number", "string": "string",
               "enum": "string", "color": "string", "datetime": "string", "duration": "string"}
VDA_TYPES = {"BOOL": "boolean", "NUMBER": "number", "INTEGER": "integer", "FLOAT": "number",
             "STRING": "string", "OBJECT": "object", "ARRAY": "array"}
ROBOT_INSTANT = {"speak", "startPause", "stopPause"}       # LLM に見せるその場のアクション
BROKER = "mqtt://127.0.0.1:1883"


class HomieCatalog:
    """homie/# を購読して、機器の記述といまの値を集める。"""

    def __init__(self, mqtt):
        self.lock = threading.Lock()
        self.attrs = {}      # topic → 値
        self.factsheet = None
        self.robot_state = {}
        mqtt.subscribe("homie/#", self._on)
        mqtt.subscribe("uagv/v2/+/+/factsheet", lambda t, p: setattr(self, "factsheet", p))
        mqtt.subscribe("uagv/v2/+/+/state", lambda t, p: self.robot_state.update(p))

    def _on(self, t, p):
        with self.lock:
            self.attrs[t] = p

    def get(self, topic, default=None):
        with self.lock:
            return self.attrs.get(topic, default)

    def devices(self):
        with self.lock:
            return sorted({t.split("/")[1] for t in self.attrs if t.endswith("/$homie")})


def _http(base, tid, kind, name, op, method):
    return {"href": f"{base}/things/{tid}/{kind}/{name}", "contentType": "application/json", "op": op,
            "htv:methodName": method}


def homie_td(cat, dev, base):
    """Homie の機器 1 台の TD。"""
    nodes = str(cat.get(f"homie/{dev}/$nodes", "")).split(",")
    props = {}
    for node in filter(None, nodes):
        for prop in filter(None, str(cat.get(f"homie/{dev}/{node}/$properties", "")).split(",")):
            pb = f"homie/{dev}/{node}/{prop}"
            dt = str(cat.get(f"{pb}/$datatype", "string"))
            settable = str(cat.get(f"{pb}/$settable", "false")).lower() == "true"
            name = prop if prop not in props else f"{node}_{prop}"
            forms = [_http(base, dev, "properties", name, ["readproperty"], "GET"),
                     {"href": BROKER, "mqv:topic": pb, "op": ["readproperty", "observeproperty"],
                      "contentType": "text/plain"}]
            if settable:
                forms.append(_http(base, dev, "properties", name, ["writeproperty"], "PUT"))
                forms.append({"href": BROKER, "mqv:topic": f"{pb}/set", "op": ["writeproperty"],
                              "contentType": "text/plain"})
            p = {"title": str(cat.get(f"{pb}/$name", prop)), "type": HOMIE_TYPES.get(dt, "string"),
                 "readOnly": not settable, "observable": True, "homie:topic": pb, "forms": forms}
            if cat.get(f"{pb}/$format") == "json":
                p["type"] = "string"
                p["contentMediaType"] = "application/json"
            props[name] = p
    return {"@context": TD_CONTEXT, "id": f"urn:nobita-home:{dev}", "title": str(cat.get(f"homie/{dev}/$name", dev)),
            "@type": "homie:Device", **NOSEC, "properties": props, "actions": {}, "events": {}}


def robot_td(cat, rooms, base, tid="robot"):
    """VDA 5050 の factsheet と部屋の一覧から、アヒルの TD を作る。"""
    fs = cat.factsheet or {}
    ts = fs.get("typeSpecification", {})
    room_schema = {"type": "string", "enum": list(rooms),
                   "description": "部屋: " + ", ".join(f"{k}={v}" for k, v in rooms.items())}
    actions = {}
    for a in fs.get("protocolFeatures", {}).get("agvActions", []):
        name = a["actionType"]
        if name not in ROBOT_INSTANT:
            continue
        params = a.get("actionParameters", []) or []
        act = {"title": name, "description": {"speak": "いる場所で話す", "startPause": "一時停止する",
                                               "stopPause": "一時停止をやめる"}.get(name, name),
               "vda5050:actionType": name, "forms": [_http(base, tid, "actions", name, "invokeaction", "POST")]}
        if params:
            act["input"] = {"type": "object", "required": [p["key"] for p in params if not p.get("isOptional")],
                            "properties": {p["key"]: {"type": VDA_TYPES.get(p.get("valueDataType"), "string")}
                                           for p in params}}
        actions[name] = act
    job = "司令塔を通して VDA 5050 の order で行う。終わるまで待って結果を返す (数十秒〜数分かかる)"
    actions["goToRoom"] = {"title": "部屋へ行く", "description": f"アヒルが部屋へ歩いて行く。{job}",
                           "input": {"type": "object", "required": ["room"], "properties": {"room": room_schema}},
                           "forms": [_http(base, tid, "actions", "goToRoom", "invokeaction", "POST")]}
    actions["checkRoom"] = {"title": "部屋を確かめる",
                            "description": f"アヒルが部屋へ行き、頭のカメラで人を探して様子 (いるか、立っているか横になっているか) を返す。{job}",
                            "input": {"type": "object", "required": ["room"], "properties": {"room": room_schema}},
                            "output": {"type": "object"},
                            "forms": [_http(base, tid, "actions", "checkRoom", "invokeaction", "POST")]}
    actions["returnToDock"] = {"title": "待機場所へ戻る", "description": f"アヒルが居間の待機場所へ戻る。{job}",
                               "forms": [_http(base, tid, "actions", "returnToDock", "invokeaction", "POST")]}
    props = {"state": {"title": "アヒルの状態 (位置・いる部屋・動いているか・最近のアクション)", "type": "object",
                       "readOnly": True, "observable": True,
                       "forms": [_http(base, tid, "properties", "state", ["readproperty"], "GET"),
                                 {"href": BROKER, "mqv:topic": "uagv/v2/NobitaHome/microduck-01/state",
                                  "op": ["readproperty", "observeproperty"], "contentType": "application/json"}]}}
    return {"@context": TD_CONTEXT, "id": f"urn:nobita-home:{tid}",
            "title": f"見守りロボット ({ts.get('seriesName', 'Microduck')})",
            "description": ts.get("seriesDescription", ""), **NOSEC, "properties": props, "actions": actions,
            "events": {}}


def all_tds(cat, rooms, base):
    tds = [robot_td(cat, rooms, base)]
    for dev in cat.devices():
        tds.append(homie_td(cat, dev, base))
    return tds
