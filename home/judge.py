"""子どもの様子の判断: 返事の意味と、アヒルの目で見た様子を合わせて、保護者への知らせの重さを決める。

返事の意味 (classify_reply):
  返事を「平気 ok・痛い hurt・助けて help・あいまい unclear」に分ける (無言は None)。決まった言葉だけに頼らないよう、
  手元で動く多言語の文の埋め込み (intfloat/multilingual-e5-small) で、分類ごとの例文との近さを見る。
  「だいじょうぶじゃない」のような否定や、「いたい」「たすけて」のようなはっきりした言葉は、手がかりとして点を足す。
  いちばん近い分類と 2 番目の差が小さければ「あいまい」とする (もう一度聞く)。
重さ (judge_rules):
  返事 (無言・助けて・痛い・平気・あいまい) と、絵 (起き上がったか・横のままの時間・見つかったか) を点数にして合わせ、
  info (様子だけ知らせる) / warn (早めに確かめてほしい) / emergency (すぐ連絡) を選ぶ。迷う所は重い方に倒す。
LLM の判断 (judge, JUDGE_LLM):
  claude-code: Claude Code (Pro などのプラン) を `claude -p` で呼ぶ (API キー不要)
  api:         Claude API (ANTHROPIC_API_KEY、料金がかかる)
  無し:        judge_rules だけ
  LLM が失敗したら judge_rules の結果を使う。どちらの結果も返す (違いを見られるように)。
"""
import json
import os
import re
import shutil
import subprocess
import threading

LEVELS = ("info", "warn", "emergency")
EMBED_MODEL = os.environ.get("JUDGE_EMBED_MODEL", "intfloat/multilingual-e5-small")
API_MODEL = os.environ.get("JUDGE_API_MODEL", "claude-sonnet-5-5")
UNCLEAR_MARGIN = 0.012   # いちばん近い分類と 2 番目の近さの差がこれより小さければ「あいまい」

EXAMPLES = {
    "ok": ["だいじょうぶ", "へーき", "平気だよ", "うん、なんともない", "ちょっところんだだけ", "もう痛くない",
           "すりむいただけ、平気", "びっくりしただけ", "大丈夫、自分で立てる", "うん、げんき", "なんでもないよ",
           "ちょっといたかったけど、もうへいき"],
    "hurt": ["いたい", "あしがいたい", "ひざがいたいよ", "頭をぶつけた", "うでがいたい", "ちがでてる",
             "いたくてなきそう", "だいじょうぶじゃない", "あんまり平気じゃない", "ずきずきする", "うーん、いたい…",
             "おなかがいたい"],
    "help": ["たすけて", "うごけない", "たてない", "ママをよんで", "くるしい", "いきができない", "めまいがする",
             "たすけて、いたくてうごけない", "だれかきて", "あたまがぐるぐるする", "立ち上がれない", "きもちわるい"],
    "unclear": ["うん…", "えっと", "なに？", "わかんない", "うーん", "ん？", "べつに", "しらない"],
}
# はっきりした言葉は埋め込みの近さに点を足す (埋め込みは否定を見落としやすい)
CUES = [
    (re.compile(r"(大丈夫|だいじょうぶ|だいじょぶ|平気|へいき|へーき)(じゃ|では|で)(な|ね)"), "hurt", 0.05),
    # 「〜できない」(動けない・動かない・立てない・起き上がれない・起きられない) と、助けを呼ぶ言葉
    (re.compile(r"(助けて|たすけて|苦しい|くるしい|息ができ|いきができ|"
                r"(うご|動)(け|か)ない|(たて|立て|たちあがれ|立ち上がれ|おきあがれ|起き上がれ|おきられ|起きられ)ない)"),
     "help", 0.05),
    (re.compile(r"(痛|いた[いくっ]|いてて|ずきずき|血|ちが)(?!(く|くは)?ない)"), "hurt", 0.04),
    (re.compile(r"(痛く|いたく)(は)?ない"), "ok", 0.05),                        # 「いたくない」は平気
    (re.compile(r"(うわーん|わーん|えーん|うえーん|ぐすん)"), "hurt", 0.05),      # 泣き声
    (re.compile(r"(大丈夫|だいじょうぶ|だいじょぶ|平気|へいき|へーき|へっちゃら|なんともない|元気|げんき)"
                r"(?!(じゃ|では|で)(な|ね))"), "ok", 0.03),
]


class ReplyClassifier:
    """返事の文を分類する。モデルは初めて使うときに読み込む (数秒)。"""

    def __init__(self, model=EMBED_MODEL):
        self.model_name = model
        self.model = None

    def _load(self):
        if self.model is False:
            raise RuntimeError("unavailable")
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.labels, texts = [], []
            for lab, exs in EXAMPLES.items():
                for e in exs:
                    self.labels.append(lab)
                    texts.append(f"query: {e}")
            self.ex_emb = self.model.encode(texts, normalize_embeddings=True)

    def scores(self, text):
        """分類ごとの点 (例文との近さの上位 2 つの平均 + 手がかりの点)。
        埋め込みのモデルが使えなければ手がかりの点だけ (どれにも当たらなければ「あいまい」)。"""
        import numpy as np
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            if self.model is not False:
                print(f"[judge] 埋め込みのモデルが使えない ({e!r})。言葉の手がかりだけで分ける", flush=True)
            self.model = False
        out = {lab: 0.0 for lab in EXAMPLES}
        out["unclear"] = 0.01
        if self.model is not False:
            v = self.model.encode([f"query: {text}"], normalize_embeddings=True)[0]
            sims = self.ex_emb @ v
            for lab in EXAMPLES:
                s = np.sort(sims[[i for i, x in enumerate(self.labels) if x == lab]])[::-1]
                out[lab] = float(s[:2].mean())
        for pat, lab, bonus in CUES:
            if pat.search(text):
                out[lab] += bonus
        return out

    def classify(self, text):
        """(分類, 差, 点) を返す。text が空 (無言) なら (None, 0, {})。"""
        if not text or not text.strip():
            return None, 0.0, {}
        sc = self.scores(text)
        # 「いてて…でもへーき」のように逆接 (でも・けど) があれば、あとの方を重く見る
        parts = re.split(r"でも|けど|けれど|だけど", text)
        if len(parts) > 1 and parts[-1].strip(" 、。…！!？?"):
            last = self.scores(parts[-1])
            sc = {k: 0.4 * sc[k] + 0.6 * last[k] for k in sc}
        ranked = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
        (lab, s1), (_, s2) = ranked[0], ranked[1]
        margin = s1 - s2
        if lab != "unclear" and margin < UNCLEAR_MARGIN:
            lab = "unclear"
        return lab, round(margin, 3), {k: round(v, 3) for k, v in sc.items()}


_clf = None
_lock = threading.Lock()


def classify_reply(text):
    """(分類, 差, 点)。司令塔の MQTT の受信スレッドからも呼ぶので、1 つずつ行う。"""
    global _clf
    with _lock:
        _clf = _clf or ReplyClassifier()
        return _clf.classify(text)


# ---- 重さの判断
REPLY_POINTS = {None: 3, "help": 3, "hurt": 2, "unclear": 1, "ok": 0}
REPLY_TEXT = {None: "返事がない", "help": "助けを求めている", "hurt": "痛がっている", "unclear": "返事がはっきりしない",
              "ok": "平気だと言っている"}


def judge_rules(ev):
    """ev: {"replies": [{"label", "text"}], "posture": standing/lying/unknown/None, "lying_seconds", "found",
    "room", "who"}。返す: {"level", "reason", "message_to_parent", "points"}"""
    labels = [r.get("label") for r in ev.get("replies", [])]
    said = [x for x in labels if x is not None]
    if "help" in labels:
        reply = "help"                  # 一度でも助けを求めたら、そのあとの返事より重く見る
    elif said:
        reply = said[-1]                # 言い直したら最後の返事を見る (「いたい…」→「もうへいき」)
        if reply == "unclear" and "hurt" in said:
            reply = "hurt"
    else:
        reply = None
    posture = ev.get("posture")
    lying_s = float(ev.get("lying_seconds") or 0.0)
    if not ev.get("found"):
        img, img_text = 2, "アヒルの目で見つけられない"
    elif posture == "standing":
        img, img_text = 0, "起き上がった"
    elif posture == "lying":
        img, img_text = (2 if lying_s >= 60 else 1), f"横になったまま ({lying_s:.0f} 秒)"
    else:
        img, img_text = 1, "姿勢が分からない"
    rp = REPLY_POINTS[reply]
    if reply == "help" or (reply is None and img >= 1):
        level = "emergency"             # 助けを求めた、または返事がなく起き上がっていない
    elif rp <= 1 and img == 0:
        level = "info"                  # 起き上がっていて、平気そう (返事があいまいでも)
    else:
        level = "warn"                  # 痛がっている・平気と言うが起き上がらない など。迷う所は重い方に倒す
    who = ev.get("who", "お子さん")
    room = ev.get("room_label") or ev.get("room") or ""
    reason = f"{REPLY_TEXT[reply]}・{img_text}"
    msg = {"info": f"{who}が{room}で転びましたが、{REPLY_TEXT[reply]}、{img_text}。様子を見守っています。",
           "warn": f"{who}が{room}で転び、{REPLY_TEXT[reply]}・{img_text}。早めに確かめてください。",
           "emergency": f"{who}が{room}で倒れています。{REPLY_TEXT[reply]}・{img_text}。すぐに連絡してください。"}[level]
    return {"level": level, "reason": reason, "message_to_parent": msg, "points": {"reply": rp, "image": img},
            "reply": reply}


# ---- LLM
PROMPT = """あなたは家庭の見守りシステムの判断役です。ひとりで留守番中の子どもが転んだ疑いがあり、ロボットが確かめました。
次の観察から、保護者への知らせの重さを 1 つ選んでください。
- info: 心配なさそう。様子だけ知らせる
- warn: けがの可能性がある。早めに確かめてほしい
- emergency: すぐに連絡・駆けつけが必要
迷うときは重い方を選んでください。

観察 (JSON):
{evidence}

次の JSON だけを返してください (説明の文は付けない):
{{"level": "info|warn|emergency", "reason": "判断の理由 (日本語 1 文)", "message_to_parent": "保護者のスマホに送る文 (日本語 1〜2 文)"}}"""


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group(0)) if m else {}
    if out.get("level") not in LEVELS:
        raise ValueError(f"bad level in {text!r}")
    return out


def _claude_code(prompt, timeout):
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        raise RuntimeError("claude (Claude Code) が見つからない")
    r = subprocess.run([exe, "-p", prompt, "--output-format", "json"], capture_output=True, text=True,
                       encoding="utf-8", timeout=timeout)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError(f"claude -p failed ({r.returncode}): {(r.stderr or r.stdout)[-300:]}") from None
    if out.get("is_error") or r.returncode != 0:
        raise RuntimeError(f"claude -p failed: {out.get('result', '')[:300]}")   # ログインが切れた、など
    return _parse(out.get("result", ""))


def _api(prompt, timeout):
    import anthropic
    msg = anthropic.Anthropic(timeout=timeout).messages.create(
        model=API_MODEL, max_tokens=400, messages=[{"role": "user", "content": prompt}])
    return _parse("".join(b.text for b in msg.content if b.type == "text"))


def judge(ev, llm=None, timeout=90.0):
    """規則の判断と (JUDGE_LLM があれば) LLM の判断。返す: {"level", "reason", "message_to_parent", "by",
    "rules": 規則の結果, "llm": LLM の結果 or None, "llm_error"}"""
    rules = judge_rules(ev)
    llm = llm if llm is not None else os.environ.get("JUDGE_LLM", "")
    out = {**rules, "by": "rules", "rules": rules, "llm": None}
    if llm in ("claude-code", "api"):
        public = {k: v for k, v in ev.items() if k != "image"}
        prompt = PROMPT.format(evidence=json.dumps(public, ensure_ascii=False, indent=1))
        try:
            res = (_claude_code if llm == "claude-code" else _api)(prompt, timeout)
            out.update(level=res["level"], reason=res.get("reason", ""),
                       message_to_parent=res.get("message_to_parent", rules["message_to_parent"]), by=llm, llm=res)
        except Exception as e:  # noqa: BLE001 (LLM が駄目でも規則の判断で知らせる)
            out["llm_error"] = repr(e)[:300]
    return out
