"""Homie 4.0 (https://homieiot.github.io/specification/) の機器を MQTT に出す。

homie/<device>/$homie, $name, $state, $nodes
homie/<device>/<node>/$name, $type, $properties
homie/<device>/<node>/<property>          値 (retain)
homie/<device>/<node>/<property>/$name, $datatype, $settable, $unit, $format
homie/<device>/<node>/<property>/set      書き込み (settable のとき)
"""

PREFIX = "homie"


class HomieDevice:
    def __init__(self, mqtt, device_id, name, nodes, on_set=None):
        """nodes: {node_id: {"name":..., "type":..., "properties": {prop_id: {"name", "datatype",
        "settable", "unit", "format"}}}}。on_set(node, prop, value) は /set を受けたとき呼ばれる。"""
        self.m = mqtt
        self.id = device_id
        self.name = name
        self.nodes = nodes
        self.on_set = on_set
        self.values = {}
        base = f"{PREFIX}/{device_id}"
        self.m.publish(f"{base}/$state", "init", retain=True)
        self.m.publish(f"{base}/$homie", "4.0", retain=True)
        self.m.publish(f"{base}/$name", name, retain=True)
        self.m.publish(f"{base}/$nodes", ",".join(nodes), retain=True)
        for nid, node in nodes.items():
            self.m.publish(f"{base}/{nid}/$name", node["name"], retain=True)
            self.m.publish(f"{base}/{nid}/$type", node.get("type", ""), retain=True)
            self.m.publish(f"{base}/{nid}/$properties", ",".join(node["properties"]), retain=True)
            for pid, p in node["properties"].items():
                pb = f"{base}/{nid}/{pid}"
                self.m.publish(f"{pb}/$name", p["name"], retain=True)
                self.m.publish(f"{pb}/$datatype", p["datatype"], retain=True)
                if p.get("settable"):
                    self.m.publish(f"{pb}/$settable", "true", retain=True)
                    self.m.subscribe(f"{pb}/set", self._make_setter(nid, pid, p["datatype"]))
                if p.get("unit"):
                    self.m.publish(f"{pb}/$unit", p["unit"], retain=True)
                if p.get("format"):
                    self.m.publish(f"{pb}/$format", p["format"], retain=True)
        self.m.publish(f"{base}/$state", "ready", retain=True)

    def _make_setter(self, nid, pid, datatype):
        def cb(topic, payload):
            v = payload
            if datatype == "boolean":
                v = str(payload).lower() in ("true", "1", "on")
            elif datatype in ("integer", "float") and isinstance(payload, str):
                v = float(payload)
            if self.on_set:
                self.on_set(nid, pid, v)
        return cb

    def set(self, node, prop, value, force=False):
        """値を出す (変わったときだけ。force=True なら毎回)。"""
        key = (node, prop)
        if not force and self.values.get(key) == value:
            return
        self.values[key] = value
        self.m.publish(f"{PREFIX}/{self.id}/{node}/{prop}", value, retain=True)
