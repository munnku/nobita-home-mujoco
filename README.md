# おるすばんアヒル — IoT と歩くロボットと LLM が、ひとりで留守番する子どもを見守り・守る家 (MuJoCo 版)

Physical AI 応用編 1 最終課題。2 階建ての日本家屋を間取り図から MuJoCo で作り、ママが出かけている間、
子どもがひとりで留守番する家を、家の IoT 機器と二足歩行の小型ロボット (Pollen Robotics の Microduck、以下アヒル)、
床の物を拾うロボット (Hello Robot の Stretch 3)、LLM (Claude) が手分けして見守る。詳しい構成・確認手順・直したことは [PLAN.md](PLAN.md)。

## できること: 予防・検知・対応支援・通知

| リスク | 予防 | 検知 | 対応支援 | 通知 |
|---|---|---|---|---|
| 転倒・けが・急病 | 見回りで床のビー玉を見つけ、Stretch 3 が片付ける | 見守りカメラの YOLO (横になった人)、常時マイクの声 → LLM | アヒルが行って目で確かめ、LLM が安否を聞き、手当ての仕方を話す | 返事と様子から info / warn / emergency |
| 転落・やけど・お風呂 | 階段・コンロ・浴室へ向かうのを先読みして、着く前に止める | 「あつっ」などの声 → LLM | 「すぐに水で 10 分冷やしてね」 | 〃 |
| カメラの死角 (トイレ・脱衣所) | — | 人感センサーで長くいる | 扉の前で声をかけ、返事がなければ中を確かめる (絵は残さない) | 返事がなければ emergency |
| 無断外出 | 玄関へ向かうのを先読みして止める | 玄関の扉が開く → アヒルが見る (YOLO で子ども) | 「外に出ちゃだめだよ」 | warn |
| 不審者・訪問者 | インターホンで LLM が家の人の代わりに応対 (子どもがひとりとは言わない) | 鍵・扉が開く → アヒルが見分ける (子ども・家の大人・知らない大人) | 不審者には「どなたですか」、子どもには「子ども部屋へ」 | emergency |
| 火事・地震・停電 | — | 煙センサー・緊急地震速報・停電 | スピーカーで逃げ方・身の守り方、ゆれがおさまったらアヒルが行って話す | 火事は emergency |
| こわい・助けて | — | 常時マイクの声 → LLM | アヒルが行って落ち着かせ、質問に答える | info / warn |

- **知らせは段階的に**: emergency は保護者のスマホへ。60 秒「見た」が無ければ 2 人目の家族にも知らせ、119 への連絡を勧める。
  子どもには 119 のかけ方を話す (自動で通報はしない)
- **いくつも同時に起きたら**: タスク一覧にためて重い順に進める (火事・地震 > 転倒・けが・不審者 > 死角・相談 > 依頼 > 見回り)。
  アヒルとインターホンは別々に同時に動く。見回りは中断して、あとで続きから回る。一覧は録画の左上に出る

## 仕組み

```
センサー (Homie / MQTT)   見守りカメラ + YOLO・マイク・人感・鍵と扉・煙・インターホン・停電と地震速報
        │
気づく (app/rules/)       LLM を通さない: 転倒 (YOLO)・先読み・死角・玄関・見回り・災害
                          LLM を通す   : 常時マイクの声 (home/brain.py の hear)
        │ タスク
タスク一覧 (app/tasks.py)  重い順。アヒル (VDA 5050) とインターホンは別々に同時に動く
        │
動く・話す                 アヒル・Stretch 3・スピーカー・インターホン。何を話すかは LLM (home/brain.py の talk)
        │
知らせる (app/notify.py)   保護者のスマホ (home/alerts)。「見た」が無ければ 2 人目へ
```
- **LLM の使い方**: LLM を呼ぶのは `home/brain.py` だけ。声の判断・対話の次の一言・知らせの重さの 3 つ。
  - 呼ぶときにだけ、司令塔が YOLO やセンサーの結果を短い文 (いまの様子) にして添える。常に流さないので、呼ぶ回数は増えない。
  - LLM ができるのは「話す・重さを決める・文を書く」だけ。鍵や扉は動かせない (訪問者の言葉で操られないように)。
  - 知らせの重さは LLM と規則の重い方にする。LLM が使えないときは、言葉の手がかりと決まった流れで動く。
- **家**: 間取り図を半間のマス目で読み取った `houses/home/layout.yaml` から MJCF を自動生成 (1 階・2 階、扉、窓、階段)
- **ロボット**: Microduck (この家で強化学習した歩き方 + 転んだら公開の方策で起き上がる)、Stretch 3 (MuJoCo Menagerie)
- **ロボットへの指示**: VDA 5050 v2.0 (order / instantActions / state / factsheet)。アヒルと Stretch の 2 台
- **保護者からの依頼**: 家電とロボットの W3C WoT Thing Description から道具を自動で作り、Claude が使う (「居間を見てきて」)
- **学習**: MuJoCo で正解付きの画像を自動で作り、YOLOv8 を追加学習 (人: 子ども・家の大人・知らない大人 × 立つ・横になる、床の物)。
  歩き方の強化学習 (mjlab、平らな床 / 散らかった床と押し)

## 動かし方

```bash
uv sync
uv run python scripts/setup_third_party.py
uv run python app/run_scenario.py --scenario fall_ok --speed 0 --record build/videos/fall_ok.mp4
```

- 1 行目で Python の環境を作る（Windows / Linux）。
- 2 行目で Microduck のモデルと学習済み方策を取ってくる。
- 3 行目で場面を 1 回通す。場面は `houses/home/scenarios/` にある。
  - 転倒: fall_ok / fall_hurt / fall_silent
  - 先読み: danger_stove / danger_genkan / danger_bath
  - 死角: senmen_silent / toilet_long
  - 声: voice_burn / voice_scared
  - 玄関: homecoming / mama_home / stranger_door / outing
  - インターホン: intercom_delivery / intercom_suspicious
  - 災害: fire_kitchen / earthquake / blackout
  - 見回り: hazard_patrol
  - 同時に起きる: multi
- 一続きの物語 (動画用) を通す: `uv run python app/run_scenario.py --story orusuban_afternoon --speed 1`
- 全部の場面をまとめて確かめる: `uv run python app/check_scenarios.py --only all`
- LLM を使う (使わなければ規則で動く):
  - `BRAIN_LLM=claude-code`: Claude Code の Pro などのプラン
  - `BRAIN_LLM=api`: Claude API。`ANTHROPIC_API_KEY` が要る
  - 声の判断を規則と比べる: `uv run python tests/test_brain.py --llm`

保護者からの頼みごと (LLM) を試すときは、`uv run python app/start_home.py` で家を動かしたまま、次のどれかを使う。
- Claude API：`uv run python app/llm_agent.py "居間の様子を見てきて"`（`ANTHROPIC_API_KEY` が要る）
- Claude Code：このフォルダで開くと、MCP サーバー（`.mcp.json`）から家電とアヒルの道具が見える

## 使っている OSS とデータ
- [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0)
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (コード Apache-2.0、ロボットのメッシュ CC BY-NC-SA 4.0。このリポジトリには含めない)
- [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (学習済み方策、Apache-2.0)
- [Rhoban/bam](https://github.com/Rhoban/bam) (サーボのモデル)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の Hello Robot Stretch 3 (Apache-2.0。このリポジトリには含めない)
- Claude (Anthropic)。Claude Code の `claude -p` か Claude API で、声の判断・対話・知らせの重さに使う
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [amqtt](https://github.com/Yakifo/amqtt) (MQTT ブローカー)、[paho-mqtt](https://github.com/eclipse/paho.mqtt.python)
- 人のモデル: Gazebo の person_standing (MakeHuman 製)
- W3C WoT TD 1.1 の JSON Schema (`wot_schemas/`)
