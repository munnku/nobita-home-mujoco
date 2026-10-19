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


CHECKS = {"fall": check_fall, "morning": check_morning, "homecoming": check_homecoming,
          "fall_ousetsu": check_fall, "fall_dk": check_fall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["fall", "morning", "homecoming"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--weights", default=None, help="YOLO の重み (既定は models/yolo_nobita.pt があればそれ)")
    ap.add_argument("--policy", default=None, help="アヒルの歩き方 (velstand / nobita_walk)")
    a = ap.parse_args()
    if a.policy:
        os.environ["DUCK_POLICY"] = a.policy
    if a.weights:
        os.environ["YOLO_WEIGHTS"] = os.path.abspath(a.weights)
    rows = []
    for name in a.only:
        for k in range(a.repeat):
            before = set(glob.glob(os.path.join(ROOT, "build", "runs", f"{name}_*.json")))
            subprocess.run([PY, os.path.join(ROOT, "app", "run_scenario.py"), "--scenario", name, "--speed", "0"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            new = sorted(set(glob.glob(os.path.join(ROOT, "build", "runs", f"{name}_*.json"))) - before)
            if not new:
                rows.append((name, k + 1, False, "実行できなかった"))
                continue
            with open(new[-1], encoding="utf-8") as f:
                events = json.load(f)["events"]
            ok, why = CHECKS[name](events)
            rows.append((name, k + 1, ok, why))
            print(f"{name} #{k+1}: {'PASS' if ok else 'FAIL'}  {why}", flush=True)
    lines = [f"YOLO: {os.environ.get('YOLO_WEIGHTS', '既定')} / 歩き方: {os.environ.get('DUCK_POLICY', 'velstand')}", "", "| 場面 | 回 | 結果 | 内容 |", "|---|---|---|---|"]
    lines += [f"| {n} | {k} | {'PASS' if ok else 'FAIL'} | {why} |" for n, k, ok, why in rows]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT, "build", "runs", f"check_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("ALL PASS" if all(r[2] for r in rows) else "SOME FAILED")


if __name__ == "__main__":
    main()
