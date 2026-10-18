# のび太君の家を IoT とロボットで見守る (MuJoCo 版)

Physical AI 応用編 1 最終課題の作り直し。Gazebo 版 (別リポジトリ・別ブランチ) を参考に、MuJoCo で新しく作る。

- 家: 野比家 1 階の間取り (`houses/nobita/layout.yaml`) から MJCF を自動生成 (`sim/house.py`)
- ロボット: Pollen Robotics の Microduck (二足歩行)。歩き方は公開の学習済み方策 (ONNX) を使う (`sim/duck.py`)
- 人: MakeHuman 製の人のメッシュ (Gazebo の person_standing) を置物として置く (`tools/convert_person.py`)

## 使っている OSS
- [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0)
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (コード Apache-2.0、ロボットのメッシュ CC BY-NC-SA 4.0。このリポジトリには含めない)
- [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (学習済み方策、Apache-2.0)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)

## 試作 (spike/)
- `duck_walk_test.py`: 方策が速度指令で歩くか (画面なし)
- `render_yolo.py`: 家・人・カメラを描いて YOLO で人を検出
- `walk_in_house.py`: 居間で経由点まで歩き、頭のカメラで人を探す → `build/spike/walk.mp4`
