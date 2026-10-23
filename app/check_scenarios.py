"""場面を順に通して、合格の目安を満たすか確かめる (PLAN.md の確認手順)。

uv run python app/check_scenarios.py [--only fall morning] [--repeat 3]
各場面の結果 (PASS / FAIL と理由、かかった時間) を表にして、build/runs/check_<時刻>.md に残す。
"""
import argparse
import datetime
import glob
import json
import os
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


def check_morning(events):
    ks = kinds(events)
    t_wake, t_tell, t_woke = first(ks, "wake_up"), first(ks, "tell_family"), first(ks, "woke_up")
    late = [e for e in events if e["topic"] == "home/speech" and "まだ起きません" in e["payload"].get("text", "")
            and t_woke is not None and e["payload"]["time"] > t_woke]
    ok = None not in (t_wake, t_tell, t_woke) and not late
    return ok, f"起こす {t_wake}s / ママに知らせる {t_tell}s / 起きた {t_woke}s / 起きた後の誤った知らせ {len(late)} 回"


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
        return ok and rep["level"] == want, (f"{why} / 判断 {rep['level']} (期待 {want}, {rep.get('by')}) "
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


def check_hide(events):
    """かくれんぼ: 見つけたか、何か所目か (でたらめな順なら平均 (候補の数 + 1) / 2 か所目)、転倒の誤報がないか。"""
    logs = [e["payload"] for e in events if e["topic"] == "home/log" and isinstance(e["payload"], dict)]
    found = next((p for p in logs if p.get("kind") == "hide_found"), None)
    false_fall = [e for e in events if e["topic"] == "home/alerts" and e["payload"].get("type", "").startswith("fall")]
    if found is None:
        return False, "見つけられなかった"
    n = found.get("n_spots", 0)
    return not false_fall, (f"{found['time']}s に {found['spot']} で見つけた ({found['tries']} か所目 / 候補 {n}、"
                            f"でたらめな順なら平均 {(n + 1) / 2:.1f} か所目) / 転倒の誤報 {len(false_fall)} 回")


def check_patrol(events):
    """見回り: 床に置いた物のうち、いくつ見つけたか (位置の誤差も)。"""
    placed = [e["payload"] for e in events if e["topic"] == "sim/scenario" and "object" in e["payload"]]
    found = [e["payload"] for e in events if e["topic"] == "home/log" and isinstance(e["payload"], dict)
             and e["payload"].get("kind") == "hazard_found"]
    hits, errs = 0, []
    for o in placed:
        f = next((f for f in found if f.get("hazard") == o["object"]), None)
        if f:
            hits += 1
            errs.append(((f["x"] - o["x"]) ** 2 + (f["y"] - o["y"]) ** 2) ** 0.5)
    wrong = len(found) - hits
    err = f"位置の誤差 平均 {sum(errs) / len(errs):.2f} m" if errs else ""
    return hits >= len(placed) - 1 and wrong == 0, f"置いた {len(placed)} 個のうち {hits} 個を見つけた / 誤り {wrong} 個 / {err}"


CHECKS = {"hide_sofa": check_hide, "hide_table": check_hide, "hide_cupboard": check_hide, "hazard_patrol": check_patrol,
          "senmen_silent": check_blind("emergency", "blind_lying"), "toilet_long": check_blind(None, "blind_ok"),
          "danger_stove": check_danger("stove"), "danger_genkan": check_danger("genkan"),
          "danger_bath": check_danger("bath"),
          "fall": check_fall, "morning": check_morning, "homecoming": check_homecoming,
          "fall_ousetsu": check_fall, "fall_dk": check_fall,
          "fall_ok": check_report("info"), "fall_hurt": check_report("warn"), "fall_silent": check_report("emergency")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["fall", "morning", "homecoming"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--weights", default=None, help="YOLO の重み (既定は models/yolo_nobita.pt があればそれ)")
    ap.add_argument("--policy", default=None, help="アヒルの歩き方 (velstand / nobita_walk)")
    ap.add_argument("--speed", default="2", help="実時間の何倍で進めるか (0 = できるだけ速く。司令塔は実時間で動くので、"
                                               "0 だと司令塔の反応の遅れがシミュレーションの時間で大きく見える)")
    a = ap.parse_args()
    if a.policy:
        os.environ["DUCK_POLICY"] = a.policy
    if a.weights:
        os.environ["YOLO_WEIGHTS"] = os.path.abspath(a.weights)
    rows = []
    for name in a.only:
        for k in range(a.repeat):
            before = set(glob.glob(os.path.join(ROOT, "build", "runs", f"{name}_*.json")))
            log = os.path.join(ROOT, "build", "runs", f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
            with open(log, "w", encoding="utf-8") as lf:     # 司令塔・シミュレーションの出力 (うまくいかなかったときに見る)
                subprocess.run([PY, os.path.join(ROOT, "app", "run_scenario.py"), "--scenario", name, "--speed", a.speed],
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
    lines = [f"YOLO: {os.environ.get('YOLO_WEIGHTS', '既定')} / 歩き方: {os.environ.get('DUCK_POLICY', '既定 (nobita_walk)')}", "", "| 場面 | 回 | 結果 | 内容 |", "|---|---|---|---|"]
    lines += [f"| {n} | {k} | {'PASS' if ok else 'FAIL'} | {why} |" for n, k, ok, why in rows]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT, "build", "runs", f"check_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("ALL PASS" if all(r[2] for r in rows) else "SOME FAILED")


if __name__ == "__main__":
    main()
