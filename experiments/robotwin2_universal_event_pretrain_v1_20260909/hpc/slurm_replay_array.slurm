#!/bin/bash
#SBATCH --job-name=uev_replay
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-59%20
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/replay_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/replay_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
ROBOTWIN=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
OFFICIAL=$ROOT/data/official/robotwin2_universal_event_replay/dataset
TASKS=(move_can_pot place_container_plate place_can_basket place_empty_cup put_object_cabinet stack_blocks_two hanging_mug open_laptop open_microwave press_stapler click_bell handover_block)
BODIES=(aloha-agilex arx-x5 franka piper ur5)
ID=${SLURM_ARRAY_TASK_ID}
TASK_INDEX=$((ID / 5))
BODY_INDEX=$((ID % 5))
TASK=${TASKS[$TASK_INDEX]}
BODY=${BODIES[$BODY_INDEX]}
ARCHIVE=$OFFICIAL/$TASK/${BODY}_clean_50.zip
case "$TASK:$BODY" in
  place_can_basket:piper|put_object_cabinet:piper|put_object_cabinet:ur5|hanging_mug:piper|handover_block:franka|handover_block:ur5)
    echo "OFFICIAL_PAIR_UNAVAILABLE task=$TASK body=$BODY"
    exit 0
    ;;
esac
# Reuse the already verified four-task cache where applicable.
if [[ ! -f "$ARCHIVE" ]]; then
  LEGACY=$ROOT/data/official/robotwin2_four_task_actor/dataset/$TASK/${BODY}_clean_50.zip
  [[ -f "$LEGACY" ]] && ARCHIVE=$LEGACY
fi
if [[ ! -f "$ARCHIVE" ]]; then
  echo "missing required official archive: $ARCHIVE" >&2
  exit 2
fi
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP"
export ASSETS_PATH="$ROBOTWIN"
export PYTHONUNBUFFERED=1
mkdir -p "$EXP/logs" "$ROOT/data/robotwin2_universal_event_v1"
cd "$ROBOTWIN"
python -u "$EXP/collect_robotwin.py" \
  --robotwin-root "$ROBOTWIN" --task "$TASK" --body "$BODY" \
  --condition clean --seed-start 0 --episodes 50 \
  --scene-backend headless-physics --replay-archive "$ARCHIVE" \
  --replay-index-start 0 --output "$ROOT/data/robotwin2_universal_event_v1"
