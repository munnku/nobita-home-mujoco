#!/usr/bin/env bash
# コンテナ pai_mujoco の中で実行する: learning/rl の env cfg を microduck_rl に写して登録する。
#   MSYS_NO_PATHCONV=1 docker exec -u ubuntu pai_mujoco bash /home/ubuntu/pai_ws/learning/rl/install.sh
# 何度実行してもよい (tasks/__init__.py の末尾の登録の塊を、印から下ごと書き直す)。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RL="${MICRODUCK_RL:-$HOME/ext/microduck_rl}"
TASKS="$RL/src/mjlab_microduck/tasks"

cp "$HERE/nobita_clutter_env_cfg.py" "$TASKS/nobita_clutter_env_cfg.py"

MARK="# --- PAI1-mujoco: nobita tasks (learning/rl/install.sh) ---"
# 前に足した塊を消す (印の行から末尾まで)
"$RL/.venv/bin/python" - "$TASKS/__init__.py" "$MARK" <<'EOF'
import sys
path, mark = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()
if mark in src:
    src = src[:src.index(mark)].rstrip("\n") + "\n"
    open(path, "w", encoding="utf-8").write(src)
EOF
cat >> "$TASKS/__init__.py" <<EOF

$MARK
from .nobita_clutter_env_cfg import make_nobita_clutter_env_cfg, NobitaClutterRlCfg, NobitaClutter2RlCfg

for _task_id, _version, _rl_cfg in (
    ("Mjlab-Nobita-Clutter-Backlash-MicroDuck", 1, NobitaClutterRlCfg),
    ("Mjlab-Nobita-Clutter2-Backlash-MicroDuck", 2, NobitaClutter2RlCfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(make_nobita_clutter_env_cfg(version=_version), _BL_WALK),
        play_env_cfg=make_backlash_variant(make_nobita_clutter_env_cfg(play=True, version=_version), _BL_WALK),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
EOF
echo "registered Mjlab-Nobita-Clutter-Backlash-MicroDuck, Mjlab-Nobita-Clutter2-Backlash-MicroDuck"
