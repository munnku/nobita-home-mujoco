# のび太君の家を IoT とロボットで見守る (MuJoCo 版)

Physical AI 応用編 1 最終課題。のび太君の家 (野比家) を間取り図から MuJoCo で作り、家の IoT 機器と二足歩行の小型ロボット
(Pollen Robotics の Microduck)、LLM が手分けして家族を見守る。詳しい構成・確認手順・直したことは [PLAN.md](PLAN.md)。

- **家**: 間取り図を半間のマス目で読み取った `houses/nobita/layout.yaml` から MJCF を自動生成 (1 階・2 階、扉、窓、階段)
- **ロボット**: Microduck。歩き方は公開の学習済み方策 (強化学習、ONNX) をそのまま使い、地図の上の A* と pure pursuit で歩かせる
- **IoT**: 見守りカメラ (YOLO で人の位置と姿勢)、人感センサー、照明、スマートロック、スピーカー。Homie 4.0 で MQTT に出す
- **ロボットへの指示**: VDA 5050 v2.0 (order / instantActions / state / factsheet)
- **LLM**: 家電とロボットの W3C WoT Thing Description から道具を自動で作り、Claude が使う (キーが無ければ簡単な規則)
- **学習**: MuJoCo で正解付きの画像を自動で作り、YOLOv8 を「立っている人 / 横になっている人」で追加学習

## 動かし方

```bash
uv sync
uv run python scripts/setup_third_party.py
uv run python app/run_scenario.py --scenario fall --speed 0 --record build/videos/fall.mp4
```

- 1 行目で Python の環境を作る（Windows / Linux）。
- 2 行目で Microduck のモデルと学習済み方策を取ってくる。
- 3 行目で場面を 1 回通す（fall / morning / homecoming）。

頼みごと (LLM) を試すときは、`uv run python app/start_home.py` で家を動かしたまま、次のどれかを使う。
- Claude API：`uv run python app/llm_agent.py "居間の様子を見てきて"`（`ANTHROPIC_API_KEY` が要る）
- Claude Code：このフォルダで開くと、MCP サーバー（`.mcp.json`）から家電とアヒルの道具が見える

## 使っている OSS とデータ
- [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0)
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (コード Apache-2.0、ロボットのメッシュ CC BY-NC-SA 4.0。このリポジトリには含めない)
- [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (学習済み方策、Apache-2.0)
- [Rhoban/bam](https://github.com/Rhoban/bam) (サーボのモデル)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [amqtt](https://github.com/Yakifo/amqtt) (MQTT ブローカー)、[paho-mqtt](https://github.com/eclipse/paho.mqtt.python)
- 人のモデル: Gazebo の person_standing (MakeHuman 製)
- 間取り: LIFULL HOME'S「ドラえもん・野比家の間取り」(https://www.homes.co.jp/cont/living/living_00324/)
- W3C WoT TD 1.1 の JSON Schema (`wot_schemas/`)
