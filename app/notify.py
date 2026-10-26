"""知らせ: 保護者のスマホ (home/alerts) への知らせを 1 か所から出し、重いものは段階的に広げる。

- send(type, level, summary, ...): id と分類を付けて home/alerts に出す (to: parent)
- emergency は ESCALATE_AFTER 秒のあいだに保護者のアプリの「見た」(home/alerts/ack、{id} か {} で全部) が来なければ、
  2 人目の家族 (to: family2) にも知らせ、「119 への連絡」を勧める。子どものいる部屋のスピーカーで 119 のかけ方も話す
  (火事・けが・返事がないとき)。
- 119 へ自動で通報はしない (誤報のときの影響が大きいので、人が決める)。
"""
ESCALATE_AFTER = 60.0
TEACH_119 = ("fire", "fall_report", "voice_report", "no_reply", "disaster_report")
SPEAKERS = ("kodomo", "dk", "ima", "rouka")


class Notifier:
    def __init__(self, c):
        self.c = c
        self.n = 0
        self.pending = {}         # id → 知らせ (emergency で「見た」を待っているもの)
        self.acked = set()
        c.m.subscribe("home/alerts/ack", lambda t, p: self._ack(p))

    def _ack(self, p):
        aid = p.get("id") if isinstance(p, dict) else None
        if aid:
            self.acked.add(aid)
        else:
            self.acked.update(self.pending)

    def send(self, type, level, summary, room=None, category="通知", **kw):
        c = self.c
        self.n += 1
        aid = f"alert-{self.n}"
        msg = {"id": aid, "type": type, "level": level, "room": room, "time": round(c.time, 1), "summary": summary,
               "to": "parent", "category": category, **kw}
        c.m.publish("home/alerts", msg, qos=1)
        c.log(f"保護者に知らせた ({level}): {summary}", kind="notify", level=level, type=type, alert=aid)
        if level == "emergency":
            self.pending[aid] = msg
        return aid

    def step(self):
        c = self.c
        for aid, msg in list(self.pending.items()):
            if aid in self.acked:
                del self.pending[aid]
                c.log("保護者が知らせを見た", kind="ack", alert=aid)
                continue
            if c.time - msg["time"] < ESCALATE_AFTER:
                continue
            del self.pending[aid]
            c.log(f"保護者が {ESCALATE_AFTER:.0f} 秒たっても知らせを見ない → 2 人目の家族に知らせる", kind="escalate", alert=aid)
            c.m.publish("home/alerts", {**msg, "id": f"{aid}-2", "to": "family2", "time": round(c.time, 1),
                                        "summary": msg["summary"] + " 保護者と連絡がつきません。必要なら 119 に連絡してください。"},
                        qos=1)
            if msg["type"] in TEACH_119:
                room = c.child_room()
                spk = room if room in SPEAKERS else "rouka"
                c.speaker(spk, "おうちの人に まだ れんらくが つかないよ。ぐあいが わるかったら、でんわで 1、1、9 に かけて、"
                               "「たすけて」と おうちの ばしょを いってね。", intent="instruct")
                c.log("子どもに 119 のかけ方を話した", kind="teach_119", room=spk)
