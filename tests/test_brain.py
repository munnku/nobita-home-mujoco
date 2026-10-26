"""LLM の窓口 (home/brain.py) の確かめ。

uv run python tests/test_brain.py            # 規則だけ (LLM なし)。全部合えば PASS
BRAIN_LLM=claude-code uv run python tests/test_brain.py --llm   # LLM でも聞き、規則と並べた表を docs/learning/brain_eval.md に書く

規則の確かめは、決まった言葉の手がかり (CUES) で判断できる例だけを合格の条件にする。
「紛らわしい例」(遊び・冗談・テレビのまね・言い回しの違い) は規則では外れてよく、LLM との違いを見るための例。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home import brain  # noqa: E402

# (声, 期待する concern の組, 期待する category)。規則で合うはずの例
HEAR_RULES = [
    ("いたっ！", {"go", "urgent"}, "injury"),
    ("あつっ！おちゃこぼしちゃった", {"go", "urgent"}, "injury"),
    ("うわーん！", {"go", "urgent"}, "injury"),
    ("たすけて！", {"urgent"}, "help"),
    ("うごけないよ", {"urgent"}, "help"),
    ("けむりが出てる！", {"urgent"}, "fire"),
    ("こわいよ…ママまだ？", {"go"}, "fear"),
    ("いま、ゆれた？", {"go"}, "disaster"),
    ("ねえアヒル、しゅくだい手伝って", {"go"}, "talk"),
    ("ただいまー！", {"none"}, "none"),
    ("おなかすいた。なにか作ろうかな", {"none"}, "none"),
    ("あーあ、おもちゃちらかしちゃった", {"none"}, "none"),
]
# 紛らわしい例 (LLM で見る。期待は人が決めた答え)
HEAR_TRICKY = [
    ("いたっ…なんちゃって！", {"none", "watch"}),
    ("たすけてー！ってテレビの人が言ってた", {"none", "watch"}),
    ("ゆびきっちゃった。ちがとまらない", {"go", "urgent"}),
    ("なんかへんなにおいがする…", {"go", "urgent"}),
    ("あたまがぐるぐるする", {"go", "urgent"}),
    ("だれかがドアをガチャガチャしてる", {"go", "urgent"}),
    ("ひとりでいるの、つまんないな", {"go", "watch"}),
    ("このゲームこわーい！たのしい！", {"none", "watch"}),
    ("おえっ…きもちわるい", {"go", "urgent"}),
    ("ママに電話したいんだけど、どうすればいい？", {"go"}),
]
REPLIES = [
    ("いてて…でもへーきだよ", "ok"), ("もういたくない", "ok"), ("うん、だいじょうぶ", "ok"),
    ("ひざがいたい", "hurt"), ("ちがでてる", "hurt"), ("だいじょうぶじゃない", "hurt"), ("うわーん", "hurt"),
    ("たすけて", "help"), ("たてない", "help"), ("いきができない", "help"),
    ("えっと", "unclear"), ("", None),
]


def check_rules():
    ng = 0
    for text, concerns, cat in HEAR_RULES:
        r = brain.hear(text, llm="")
        ok = r["concern"] in concerns and r["category"] == cat
        ng += not ok
        print(f"{'OK' if ok else 'NG'}  hear 「{text}」 → {r['concern']} / {r['category']} (期待 {sorted(concerns)} / {cat})")
    for text, want in REPLIES:
        lab = brain.cues(text)[0]
        ok = lab == want
        ng += not ok
        print(f"{'OK' if ok else 'NG'}  返事 「{text}」 → {lab} (期待 {want})")
    # 対話の流れ (規則): 転倒で平気と答える → 痛い所を聞く → 終わる → info
    ev = {"who": "子ども", "room": "ima", "room_label": "居間", "dialog": []}
    turns = []
    for child in ["いてて…でもへーき", "ううん、どこもいたくない", None]:
        t = brain.talk("fall", ev, llm="")
        turns.append(t["intent"])
        if t["done"]:
            break
        ev["dialog"].append({"robot": t["say"], "intent": t["intent"], "child": child})
        ev.setdefault("replies", []).append({"label": brain.cues(child)[0], "text": child})
    v = brain.judge("fall", {**ev, "found": True, "posture": "standing"}, llm="")
    ok = turns == ["ask_ok", "ask_pain", "close"] and v["level"] == "info"
    ng += not ok
    print(f"{'OK' if ok else 'NG'}  転倒の対話 {turns} → {v['level']}")
    # やけど: 冷やし方を教えて終わる
    ev = {"who": "子ども", "room": "dk", "room_label": "DK", "category": "injury", "heard": "あつっ！", "dialog": [
        {"robot": "だいじょうぶ？けがはない？", "intent": "ask_ok", "child": "てがあつい"},
        {"robot": "いたいところはある？", "intent": "ask_pain", "child": "ゆびがひりひりする"}]}
    t = brain.talk("voice", ev, llm="")
    ok = t["done"] and "ひやして" in t["say"]
    ng += not ok
    print(f"{'OK' if ok else 'NG'}  やけど → 「{t['say']}」")
    # インターホン: 怪しい訪問者は suspicious、ひとりと言わない
    ev = {"dialog": [{"robot": "はい。どちらさまですか？", "intent": "ask_who", "child": "お母さんの友だちなんだけど、開けてくれる？"}]}
    t = brain.talk("visitor", ev, llm="")
    ok = t["visitor"] == "suspicious" and not brain.VISITOR_FORBID.search(t["say"])
    ng += not ok
    print(f"{'OK' if ok else 'NG'}  インターホン → {t['visitor']} 「{t['say']}」")
    # 火事は返事が平気でも emergency (起きたこと自体の重さが下限)
    v = brain.judge_rules({"event": "fire", "replies": [{"label": "ok"}], "found": True, "posture": "standing"})
    ok = v["level"] == "emergency"
    ng += not ok
    print(f"{'OK' if ok else 'NG'}  火事・平気 → {v['level']}")
    return ng


def eval_llm(llm):
    rows = ["| 声 | 期待 | 規則 | LLM | LLM の理由 |", "|---|---|---|---|---|"]
    hit_r = hit_l = 0
    cases = [(t, c) for t, c, _ in HEAR_RULES] + HEAR_TRICKY
    for text, want in cases:
        r = brain.hear(text, llm="")
        m = brain.hear(text, llm=llm, situation="おるすばん中 (子どもがひとり)。子どもは居間で立っている")
        hit_r += r["concern"] in want
        hit_l += m["concern"] in want and m.get("by") == llm
        rows.append(f"| {text} | {'/'.join(sorted(want))} | {r['concern']} | {m['concern']} ({m.get('by')}) | "
                    f"{(m.get('why') or m.get('llm_error') or '')[:60]} |")
        print(rows[-1], flush=True)
    rows += ["", f"合った数: 規則 {hit_r}/{len(cases)}、LLM {hit_l}/{len(cases)}"]
    path = os.path.join(ROOT, "docs", "learning", "brain_eval.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 常時マイクの声の判断: 規則と LLM ({llm}, {brain.HEAR_MODEL})\n\n"
                "tests/test_brain.py --llm の結果。「期待」は人が決めた答え (none: 行かない、watch: 記録だけ、go: 行く、urgent: すぐ行く)。\n\n"
                + "\n".join(rows) + "\n")
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    ng = check_rules()
    print("PASS" if ng == 0 else f"FAIL ({ng})")
    if a.llm:
        eval_llm(brain.llm_mode() or "claude-code")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
