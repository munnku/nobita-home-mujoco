"""返事の分け方と重さの判断を確かめる。例文 (home/judge.py の EXAMPLES) に無い言い回しで試す。

uv run python tests/test_judge.py [--llm claude-code]
返事の分類の正しさ (9 割以上で合格) と、重さの判断の表を出す。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.judge import classify_reply, judge, judge_rules  # noqa: E402

REPLIES = [
    ("いてて…でもへいきだよ", "ok"), ("うん、だいじょぶ", "ok"), ("ころんじゃった。でもなんともないよ", "ok"),
    ("ちょっとびっくりした。いたくない", "ok"), ("へっちゃら！", "ok"), ("だいじょうぶ、ひとりで立てるよ", "ok"),
    ("ぜんぜん平気", "ok"), ("もうなおったよ", "ok"),
    ("ひざすりむいてちょっといたい", "hurt"), ("あたまぶつけた…いたい", "hurt"), ("だいじょうぶじゃないかも", "hurt"),
    ("てくびがいたいよう", "hurt"), ("あしひねったみたい", "hurt"), ("ちがでちゃった", "hurt"), ("うぅ、いたいよぉ", "hurt"),
    ("へいきじゃない…", "hurt"), ("せなかがずきずきする", "hurt"),
    ("たすけて…", "help"), ("あしがうごかない", "help"), ("ママよんで、はやく", "help"), ("いきがくるしい", "help"),
    ("たてないよ、たすけて", "help"), ("めがまわる", "help"), ("だれかたすけて", "help"), ("おきあがれない", "help"),
    ("えー？", "unclear"), ("んー", "unclear"), ("なにが？", "unclear"), ("……わかんない", "unclear"), ("あのね", "unclear"),
]

CASES = [  # (返事, 姿勢, 横の秒数, 見つかった, 期待する重さ)
    (["へーき"], "standing", 10, True, "info"),
    (["ころんだだけ"], "lying", 20, True, "warn"),
    (["いたい…"], "standing", 15, True, "warn"),
    (["いたい…", "もうへいき"], "standing", 25, True, "info"),
    (["たすけて"], "lying", 20, True, "emergency"),
    ([None, None], "lying", 40, True, "emergency"),
    ([None, None], "standing", 40, True, "warn"),
    (["うーん"], "standing", 10, True, "info"),
    ([None], None, 30, False, "emergency"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="", help="claude-code / api (重さの判断を LLM でも行い、規則と比べる)")
    a = ap.parse_args()
    ok = 0
    for text, want in REPLIES:
        lab, margin, sc = classify_reply(text)
        hit = lab == want
        ok += hit
        print(f"{'OK ' if hit else 'NG '} {text:24s} → {lab:8s} (期待 {want}, 差 {margin}) {sc}")
    acc = ok / len(REPLIES)
    print(f"返事の分類: {ok}/{len(REPLIES)} = {acc:.0%}")
    print()
    bad = 0
    for texts, posture, lying_s, found, want in CASES:
        replies = [{"text": t, "label": classify_reply(t)[0] if t else None} for t in texts]
        ev = {"who": "のび太", "room": "ima", "room_label": "居間", "replies": replies, "posture": posture,
              "lying_seconds": lying_s, "found": found}
        res = judge(ev, llm=a.llm) if a.llm else judge_rules(ev)
        r = res.get("rules", res)
        bad += r["level"] != want
        llm = f" / LLM: {res['level']} ({res.get('reason')})" if a.llm else ""
        print(f"{'OK ' if r['level'] == want else 'NG '} {texts} {posture} {lying_s}s → 規則: {r['level']} "
              f"(期待 {want}, {r['reason']}){llm}{' / LLM 失敗 ' + res['llm_error'] if res.get('llm_error') else ''}")
    print("PASS" if acc >= 0.9 and bad == 0 else "FAIL")


if __name__ == "__main__":
    main()
