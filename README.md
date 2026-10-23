# おるすばんアヒル — IoT と歩くロボットが、ひとりで留守番する子どもを見守り・守り・遊ぶ家 (MuJoCo 版)

Physical AI 応用編 1 最終課題。のび太君の家 (野比家) を間取り図から MuJoCo で作り、ママが出かけている間、
のび太がひとりで留守番する家を、家の IoT 機器と二足歩行の小型ロボット (Pollen Robotics の Microduck、以下アヒル)、
床の物を拾うロボット (Hello Robot の Stretch 3)、LLM が手分けして見守る。詳しい構成・確認手順・直したことは [PLAN.md](PLAN.md)。

**できること (おるすばんモード)**
1. **転んだら見に行き、話しかけて判断する**: 見守りカメラの YOLO が「横になった人」を 3 秒見るか、「いたっ」などの声を聞いたら、
   アヒルが見通せる所へ行って「だいじょうぶ？」と聞く。返事の意味 (文の埋め込みで分類) と、起き上がったか (アヒルの目) を合わせて
   info / warn / emergency を決め、ママのスマホ (home/alerts) に知らせる。判断は規則か LLM (Claude Code / Claude API)
2. **危ない所へ向かうのを先読みして止める**: コンロ・玄関・浴室などへの残りの道のりの減り方から行き先を推定し (ベイズ更新)、
   着く前にスピーカーで注意し、アヒルが歩きながら声をかけて間に入る
3. **カメラを置けない所を見に行く**: 脱衣所・トイレに長くいたら、扉の前で声をかけ、返事がなければ中を確かめる (絵は残さない)
4. **床の小さい危ない物を見回りで見つけ、片付ける**: アヒルが首を下げて部屋を回り、ボタン電池・硬貨・ビー玉などを YOLO で見つける。
   Stretch 3 が拾って箱に入れる
5. **かくれんぼ**: アヒルが鬼になり、家の地図から「隠れられそうな所」を考えて探す (見守りカメラは使わない)

**仕組み**
- **家**: 間取り図を半間のマス目で読み取った `houses/nobita/layout.yaml` から MJCF を自動生成 (1 階・2 階、扉、窓、階段)
- **ロボット**: Microduck (この家で強化学習した歩き方 + 転んだら公開の方策で起き上がる)、Stretch 3 (MuJoCo Menagerie)
- **IoT**: 見守りカメラ (YOLO で人の位置と姿勢、マイク)、人感センサー、扉のセンサー、照明、スマートロック、スピーカー。Homie 4.0 で MQTT に出す
- **ロボットへの指示**: VDA 5050 v2.0 (order / instantActions / state / factsheet)。アヒルと Stretch の 2 台
- **LLM**: 家電とロボットの W3C WoT Thing Description から道具を自動で作り、Claude が使う。転倒の判断・かくれんぼの探す順にも使える
- **学習**: MuJoCo で正解付きの画像を自動で作り、YOLOv8 を追加学習 (人の姿勢 2 クラス、床の物 5 クラス)。
  歩き方の強化学習 (mjlab、平らな床 / 散らかった床と押し)

## 動かし方

```bash
uv sync
uv run python scripts/setup_third_party.py
uv run python app/run_scenario.py --scenario fall --speed 0 --record build/videos/fall.mp4
```

- 1 行目で Python の環境を作る（Windows / Linux）。
- 2 行目で Microduck のモデルと学習済み方策を取ってくる。
- 3 行目で場面を 1 回通す（場面は `houses/nobita/scenarios/`。おるすばんの場面は fall_ok / fall_hurt / fall_silent /
  danger_stove / danger_genkan / danger_bath / senmen_silent / toilet_long / hazard_patrol / hide_sofa / hide_table / hide_cupboard）。
- 全部の場面をまとめて確かめる: `uv run python app/check_scenarios.py --only <場面 ...>`
- 転倒の判断に LLM を使う: `JUDGE_LLM=claude-code` (Claude Code の Pro などのプラン) か `JUDGE_LLM=api` (Claude API)。
  かくれんぼの探す順は `HIDE_LLM` で同じように選ぶ

頼みごと (LLM) を試すときは、`uv run python app/start_home.py` で家を動かしたまま、次のどれかを使う。
- Claude API：`uv run python app/llm_agent.py "居間の様子を見てきて"`（`ANTHROPIC_API_KEY` が要る）
- Claude Code：このフォルダで開くと、MCP サーバー（`.mcp.json`）から家電とアヒルの道具が見える

## 使っている OSS とデータ
- [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0)
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (コード Apache-2.0、ロボットのメッシュ CC BY-NC-SA 4.0。このリポジトリには含めない)
- [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (学習済み方策、Apache-2.0)
- [Rhoban/bam](https://github.com/Rhoban/bam) (サーボのモデル)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の Hello Robot Stretch 3 (Apache-2.0。このリポジトリには含めない)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (Apache-2.0) と [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (MIT)。子どもの返事の分類
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [amqtt](https://github.com/Yakifo/amqtt) (MQTT ブローカー)、[paho-mqtt](https://github.com/eclipse/paho.mqtt.python)
- 人のモデル: Gazebo の person_standing (MakeHuman 製)
- 間取り: LIFULL HOME'S「ドラえもん・野比家の間取り」(https://www.homes.co.jp/cont/living/living_00324/)
- W3C WoT TD 1.1 の JSON Schema (`wot_schemas/`)
