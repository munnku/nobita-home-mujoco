"""場面を順に通して、合格の目安を満たすか確かめる (PLAN.md の確認手順)。

uv run python app/check_scenarios.py [--only fall fall_ok | --only all] [--repeat 3]
各場面の結果 (PASS / FAIL と理由、かかった時間) を表にして、build/runs/check_<時刻>.md に残す。
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def kinds(events):
    return [(e["payload"].get("time", 0), e["payload"].get("kind")) for e in events
            if e["topic"] == "home/log" and isinstance(e["payload"], dict) and e["payload"].get("kind")]


def first(ks, kind):
    return next((t for t, k in ks if k == kind), None)


def check_fall(events):
    fall_t = next((e["payload"]["time"] for e in events if e["topic"] == "sim/scenario"
                   and e["payload"].get("pose") == "lie"), None)
    alert = next((e["payload"] for e in events if e["topic"] == "home/alerts"
                  and e["payload"]["type"] in ("fall", "fall_unconfirmed")), None)
    if fall_t is None or alert is None:
        return False, "転倒の知らせが無い"
    dt = alert["time"] - fall_t
    how = "アヒルが確認" if alert["type"] == "fall" else "アヒルは見えず、カメラの知らせのまま"
    return dt < 90, f"倒れてから {dt:.1f} 秒で知らせ ({how})"


def check_homecoming(events):
    ks = kinds(events)
    t_open, t_greet, t_lock = first(ks, "door_open"), first(ks, "greet"), first(ks, "auto_lock")
    false_fall = first(ks, "fall_suspected")
    ok = None not in (t_open, t_greet, t_lock) and false_fall is None
    return ok, f"扉 {t_open}s / 迎える {t_greet}s / 施錠 {t_lock}s / 転倒の誤報 {'あり' if false_fall else 'なし'}"


def check_report(want):
    """転んだ子どもに声をかけ、返事と様子から決めた知らせの重さが want か。"""
    def check(events):
        ok, why = check_fall(events)
        rep = next((e["payload"] for e in events if e["topic"] == "home/alerts"
                    and e["payload"].get("type") == "fall_report"), None)
        if rep is None:
            return False, why + " / 判断の知らせが無い"
        replies = " → ".join(r.get("text") or "(無言)" for r in rep.get("replies", []))
        return ok and at_least(rep["level"], want), (f"{why} / 判断 {rep['level']} (期待 {want}, {rep.get('by')}) "
                                            f"{rep.get('reason')} / 返事: {replies}")
    return check


def check_danger(hazard):
    """危ない場所 (hazard) へ向かう子どもを、入る前に先読みして (備える・警告する) 止められたか。"""
    def check(events):
        logs = [e["payload"] for e in events if e["topic"] == "home/log" and isinstance(e["payload"], dict)]
        prep = next((p for p in logs if p.get("kind") == "predict_prepare" and p.get("hazard") == hazard), None)
        warn = next((p for p in logs if p.get("kind") == "predict_hazard" and p.get("hazard") == hazard), None)
        stop = next((e["payload"] for e in events if e["topic"] == "sim/scenario"
                     and e["payload"].get("stopped")), None)
        entered = [e["payload"] for e in events if e["topic"] == "sim/scenario"
                   and e["payload"].get("hazard_entered") == hazard]
        start = next((e["payload"]["time"] for e in events if e["topic"] == "sim/scenario"
                      and e["payload"].get("walk")), None)
        ok = (warn or prep) is not None and not entered and stop is not None
        def t(p):
            return f"{p['time']:.1f}s (確率 {p.get('p')}、あと {p.get('dist')} m)" if p else "なし"
        return ok, (f"歩き始め {start}s / 備える {t(prep)} / 警告 {t(warn)} / "
                    f"止まった {stop['time'] if stop else 'なし'}s / 危ない場所に入った {'あり' if entered else 'なし'}")
    return check


def check_blind(want_level, want_kind):
    """カメラの無い部屋: アヒルが声をかけに行き、want_kind (blind_ok / blind_lying など) になって、
    知らせが want_level (None なら知らせなし) か。"""
    def check(events):
        ks = kinds(events)
        t_check, t_kind = first(ks, "blind_check"), first(ks, want_kind)
        alerts = [e["payload"] for e in events if e["topic"] == "home/alerts" and e["payload"].get("type") == "no_reply"]
        level = alerts[0]["level"] if alerts else None
        ok = t_check is not None and t_kind is not None and level == want_level
        return ok, (f"声をかけに行く {t_check}s / {want_kind} {t_kind}s / 知らせ {level} (期待 {want_level})"
                    + (f" 「{alerts[0]['summary']}」" if alerts else ""))
    return check


def check_patrol(events):
    """見回り: 床に置いた物のうち、いくつ見つけたか (位置の誤差も)。"""
    placed = [e["payload"] for e in events if e["topic"] == "sim/scenario" and "object" in e["payload"]]
    found = [e["payload"] for e in events if e["topic"] == "home/log" and isinstance(e["payload"], dict)
             and e["payload"].get("kind") == "hazard_found"]
    hits, errs = 0, []
    used = []
    for o in placed:
        # 同じ種類の物がいくつもある (ビー玉 3 個) ので、まだ組にしていない一番近い物と組にする
        f = min((f for f in found if f.get("hazard") == o["object"] and f not in used), default=None,
                key=lambda f: (f["x"] - o["x"]) ** 2 + (f["y"] - o["y"]) ** 2)
        if f:
            used.append(f)
            hits += 1
            errs.append(((f["x"] - o["x"]) ** 2 + (f["y"] - o["y"]) ** 2) ** 0.5)
    wrong = len(found) - hits
    err = f"位置の誤差 平均 {sum(errs) / len(errs):.2f} m" if errs else ""
    kinds = lambda k: [e["payload"].get("hazard") for e in events if e["topic"] == "home/log"  # noqa: E731
                       and isinstance(e["payload"], dict) and e["payload"].get("kind") == k]
    done, failed = kinds("tidy_done"), kinds("tidy_failed")
    tidy = f"片付けた {len(done)} 個 / 片付けられなかった {len(failed)} 個"
    ok = hits >= len(placed) - 1 and wrong == 0 and len(done) >= hits - 1
    return ok, f"置いた {len(placed)} 個のうち {hits} 個を見つけた / 誤り {wrong} 個 / {err} / {tidy}"


LEVELS = ("info", "warn", "emergency")
FORBID = re.compile(r"(ひとり|一人|留守|るすばん|子ども|子供|お子|開けます|あけます)")   # インターホンで言ってはいけない言葉


def logs_of(events):
    return [e["payload"] for e in events if e["topic"] == "home/log" and isinstance(e["payload"], dict)]


def alerts_of(events, type=None):
    return [e["payload"] for e in events if e["topic"] == "home/alerts" and isinstance(e["payload"], dict)
            and (type is None or e["payload"].get("type") == type)]


def at_least(level, want):
    """LLM で回すと重い方に倒すことがあるので、期待以上なら合格 (規則だけなら期待どおりになる)。"""
    return level in LEVELS and LEVELS.index(level) >= LEVELS.index(want)


def check_voice(want, say=None):
    """常時マイクの声: LLM (なければ規則) が行くべき声と判断し、アヒルが行って対話し、want 以上で知らせたか。"""
    def check(events):
        heard = next((p for p in logs_of(events) if p.get("kind") == "voice_heard"
                      and p.get("concern") in ("go", "urgent")), None)
        rep = next(iter(alerts_of(events, "voice_report")), None)
        talk = [e["payload"]["text"] for e in events if e["topic"] == "home/speech"
                and e["payload"].get("speaker") == "microduck-01"]
        said = say is None or any(say in t for t in talk)
        ok = heard is not None and rep is not None and at_least(rep["level"], want) and said
        return ok, (f"声 {heard['time'] if heard else 'なし'}s ({heard.get('category') if heard else '-'}, "
                    f"{heard.get('by') if heard else '-'}) / 知らせ {rep['level'] if rep else 'なし'} (期待 {want} 以上) / "
                    f"アヒル: {' → '.join(talk)[:160]}")
    return check


def check_visitor(want):
    """インターホン: 応対が want (delivery / suspicious) になり、子どもがひとりだと分かる言葉を言わなかったか。"""
    def check(events):
        done = next((p for p in logs_of(events) if p.get("kind") == "visitor_done"), None)
        said = [e["payload"]["text"] for e in events if e["topic"] == "home/speech"
                and e["payload"].get("speaker") == "intercom-genkan"]
        bad = [t for t in said if FORBID.search(t)]
        child = [e for e in events if e["topic"] == "home/speech" and "でなくて" in e["payload"].get("text", "")]
        ok = done is not None and done.get("visitor") == want and not bad and (want != "suspicious" or child)
        return ok, (f"応対 {done.get('visitor') if done else 'なし'} (期待 {want}, {done.get('by') if done else '-'}) / "
                    f"インターホン: {' → '.join(said)[:160]} / 言ってはいけない言葉 {len(bad)} / 子どもへの声 {len(child)}")
    return check


def check_door(kind, level, escalate=False):
    """玄関: アヒルが見分けて kind (stranger / outing / alone_end) になり、level 以上で知らせたか。"""
    def check(events):
        hit = next((p for p in logs_of(events) if p.get("kind") == kind), None)
        alerts = [a for a in alerts_of(events) if a.get("to") == "parent" and a.get("type") in ("stranger", "outing", "door")]
        top = max((a["level"] for a in alerts if a["level"] in LEVELS), key=LEVELS.index, default=None)
        esc = any(a.get("to") == "family2" for a in alerts_of(events))
        ok = hit is not None and top is not None and at_least(top, level) and (esc or not escalate)
        return ok, (f"{kind} {hit['time'] if hit else 'なし'}s / 知らせの重さ {top} (期待 {level}) / "
                    f"2 人目へ {'あり' if esc else 'なし'}")
    return check


def check_disaster(event, want, escalate=False):
    """災害: 気づいてスピーカーで知らせ、アヒルが子どもと話し、want 以上で知らせたか (火事は 2 人目と 119 の案内も)。"""
    def check(events):
        ks = kinds(events)
        t0 = first(ks, f"{event}_detected")
        rep = next(iter(alerts_of(events, "disaster_report")), None)
        esc = any(a.get("to") == "family2" for a in alerts_of(events))
        t119 = first(ks, "teach_119")
        ok = t0 is not None and rep is not None and at_least(rep["level"], want) and (not escalate or (esc and t119))
        extra = f" / ゆれのあと {first(ks, 'quake_over')}s" if event == "earthquake" else ""
        return ok, (f"気づく {t0}s / 知らせ {rep['level'] if rep else 'なし'} (期待 {want} 以上, {rep.get('by') if rep else '-'})"
                    f" / 2 人目へ {'あり' if esc else 'なし'} / 119 の案内 {t119}s{extra}")
    return check


def check_multi(events):
    """同時に起きる: 見回りを中断して転倒を先にし、その間にインターホンも応対し、見回りを続きから回ったか。"""
    ks = kinds(events)
    pre = next((p for p in logs_of(events) if p.get("kind") == "preempt" and p.get("task") == "patrol"), None)
    judged = first(ks, "fall_judged")
    visitor = first(ks, "visitor_done")
    resume = next((t for t, k in ks if k == "patrol_resume" and judged and t > judged), None)
    overlap = visitor is not None and judged is not None and visitor < judged
    ok = pre is not None and judged is not None and overlap and resume is not None
    return ok, (f"見回りを中断 {pre['time'] if pre else 'なし'}s / 転倒の判断 {judged}s / インターホン {visitor}s "
                f"(転倒の最中 {'はい' if overlap else 'いいえ'}) / 見回りの続き {resume}s")


def check_afternoon(events):
    """物語 orusuban_afternoon: 章ごとの印が全部そろったか (時刻つきで並べる)。"""
    logs = logs_of(events)
    alerts = alerts_of(events)
    first_ = lambda pred: next((p for p in logs if pred(p)), None)  # noqa: E731
    marks = [("迎える", first_(lambda p: p.get("kind") == "greet" and p.get("who") in (None, "child"))),
             ("おるすばん", first_(lambda p: p.get("kind") == "alone_start")),
             ("転倒の判断", next((a for a in alerts if a.get("type") == "fall_report"), None)),
             ("やけど", next((a for a in alerts if a.get("type") == "voice_report"), None)),
             ("コンロで止まった", first_(lambda p: p.get("kind") == "child_stopped")),
             ("インターホン", first_(lambda p: p.get("kind") == "visitor_done")),
             ("トイレの返事", first_(lambda p: p.get("kind") == "blind_ok")),
             ("片付け", first_(lambda p: p.get("kind") == "tidy_finished" and p.get("count", 0) >= 2)),
             ("地震", next((a for a in alerts if a.get("type") == "disaster_report"), None))]
    chapters = [e["payload"] for e in events if e["topic"] == "sim/scenario" and "chapter" in e["payload"]]
    why = " / ".join(f"{n} {p['time']}s" if p else f"{n} なし" for n, p in marks)
    return all(p for _, p in marks), f"章 {len(chapters)} 個 / " + why


STORIES = {"orusuban_afternoon"}      # houses/home/stories/ の物語 (run_scenario.py --story で回す)

CHECKS = {"hazard_patrol": check_patrol,
          "senmen_silent": check_blind("emergency", "blind_lying"), "toilet_long": check_blind(None, "blind_ok"),
          "danger_stove": check_danger("stove"), "danger_genkan": check_danger("genkan"),
          "danger_bath": check_danger("bath"),
          "fall": check_fall, "homecoming": check_homecoming,
          "fall_ousetsu": check_fall, "fall_dk": check_fall,
          "fall_ok": check_report("info"), "fall_hurt": check_report("warn"), "fall_silent": check_report("emergency"),
          "voice_burn": check_voice("warn", "ひやし"), "voice_scared": check_voice("info"),
          "intercom_delivery": check_visitor("delivery"), "intercom_suspicious": check_visitor("suspicious"),
          "stranger_door": check_door("stranger", "emergency", escalate=True), "outing": check_door("outing", "warn"),
          "mama_home": check_door("alone_end", "info"),
          "fire_kitchen": check_disaster("fire", "emergency", escalate=True),
          "earthquake": check_disaster("earthquake", "warn"), "blackout": check_disaster("blackout", "info"),
          "multi": check_multi, "orusuban_afternoon": check_afternoon}
ALL = [n for n in CHECKS if n not in STORIES]      # all で回す場面 (物語は長いので別に回す)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["fall", "homecoming"], help="場面の名前 (all で全部)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--weights", default=None, help="YOLO の重み (既定は models/yolo_person.pt があればそれ)")
    ap.add_argument("--policy", default=None, help="アヒルの歩き方 (velstand / duck_walk)")
    ap.add_argument("--speed", default="2", help="実時間の何倍で進めるか (0 = できるだけ速く。司令塔は実時間で動くので、"
                                               "0 だと司令塔の反応の遅れがシミュレーションの時間で大きく見える)")
    a = ap.parse_args()
    if a.policy:
        os.environ["DUCK_POLICY"] = a.policy
    if a.weights:
        os.environ["YOLO_WEIGHTS"] = os.path.abspath(a.weights)
    rows = []
    for name in (ALL if a.only == ["all"] else a.only):
        for k in range(a.repeat):
            before = set(glob.glob(os.path.join(ROOT, "build", "runs", f"{name}_*.json")))
            os.makedirs(os.path.join(ROOT, "build", "runs"), exist_ok=True)
            log = os.path.join(ROOT, "build", "runs", f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
            with open(log, "w", encoding="utf-8") as lf:     # 司令塔・シミュレーションの出力 (うまくいかなかったときに見る)
                subprocess.run([PY, os.path.join(ROOT, "app", "run_scenario.py"),
                                "--story" if name in STORIES else "--scenario", name, "--speed", a.speed],
                               stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
            new = sorted(set(glob.glob(os.path.join(ROOT, "build", "runs", f"{name}_*.json"))) - before)
            if not new:
                rows.append((name, k + 1, False, "実行できなかった"))
                continue
            with open(new[-1], encoding="utf-8") as f:
                events = json.load(f)["events"]
            ok, why = CHECKS[name](events)
            rows.append((name, k + 1, ok, why))
            print(f"{name} #{k+1}: {'PASS' if ok else 'FAIL'}  {why}", flush=True)
    lines = [f"YOLO: {os.environ.get('YOLO_WEIGHTS', '既定')} / 歩き方: {os.environ.get('DUCK_POLICY', '既定 (duck_walk)')}", "", "| 場面 | 回 | 結果 | 内容 |", "|---|---|---|---|"]
    lines += [f"| {n} | {k} | {'PASS' if ok else 'FAIL'} | {why} |" for n, k, ok, why in rows]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT, "build", "runs", f"check_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("ALL PASS" if all(r[2] for r in rows) else "SOME FAILED")


if __name__ == "__main__":
    main()
