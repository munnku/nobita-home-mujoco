#!/usr/bin/env bash
# コンテナ pai_mujoco の中で、家の床のタスクを学習する。
#   bash learning/rl/train.sh <envs> <iters> [task] [warm-start run]
#   v1 (nobita_robust) : bash learning/rl/train.sh 2048 1500
#   v2 (nobita_robust2): bash learning/rl/train.sh 2048 1500 Mjlab-Nobita-Clutter2-Backlash-MicroDuck 2026-10-23_02-55-54_nobita_robust
#   smoke              : bash learning/rl/train.sh 64 5 [task] [run]
# 学習済みの歩き (既定は 2026-10-19_01-45-26_nobita) の model_1499.pt から温めて始める。RTX 3060 で 1500 iter 約 2 h。
set -euo pipefail
ENVS="${1:-4096}"
ITERS="${2:-1500}"
TASK="${3:-Mjlab-Nobita-Clutter-Backlash-MicroDuck}"
RUN="${4:-2026-10-19_01-45-26_nobita}"
cd "$HOME/ext/microduck_rl"
export MUJOCO_GL=egl
# MICRODUCK_WARM_START=1: 読み込んだ重みはそのまま、反復数と curriculum の数は 0 から (tasks/mdp.py Patch 5)
MICRODUCK_WARM_START=1 .venv/bin/train "$TASK" \
  --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITERS" --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$RUN" --agent.load-checkpoint model_1499.pt
