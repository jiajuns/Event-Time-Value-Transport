#!/bin/bash
#SBATCH --job-name=uev_collect
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-119%20
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/collect_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/collect_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
ROBOTWIN=$ROOT/runtime/etsf_stage0/RoboTwin_v14_dual_gpu_20260831
TASKS=(move_can_pot place_container_plate place_can_basket place_empty_cup put_object_cabinet stack_blocks_two hanging_mug open_laptop open_microwave press_stapler click_bell handover_block)
BODIES=(aloha-agilex arx-x5 franka piper ur5)
CONDITIONS=(clean randomized)
ID=${SLURM_ARRAY_TASK_ID}
TASK_INDEX=$((ID / 10))
BODY_INDEX=$(((ID / 2) % 5))
CONDITION_INDEX=$((ID % 2))
TASK=${TASKS[$TASK_INDEX]}
BODY=${BODIES[$BODY_INDEX]}
CONDITION=${CONDITIONS[$CONDITION_INDEX]}
SEED_START=$((2026100000 + ID * 100))
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
  --condition "$CONDITION" --seed-start "$SEED_START" --episodes 30 \
  --scene-backend headless-physics \
  --output "$ROOT/data/robotwin2_universal_event_v1"
