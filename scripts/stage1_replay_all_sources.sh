#!/usr/bin/env bash
set -euo pipefail

stage0=/home/user/etsf_stage0
stage1=/home/user/etsf_stage1
python_env=/home/user/anaconda3/envs/ETSF_RoboTwin/bin
tasks=(adjust_bottle handover_block move_can_pot place_container_plate beat_block_hammer lift_pot)
bodies=(aloha-agilex ARX-X5)

mkdir -p "$stage1/logs"
for task in "${tasks[@]}"; do
  for body in "${bodies[@]}"; do
    env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES=0 \
      PATH="$python_env:$PATH" \
      python "$stage1/stage1_replay_source_objects.py" \
        --repo "$stage0/RoboTwin" \
        --source "$stage1/source_data/$task/$body" \
        --task "$task" \
        --embodiment "$body" \
        --output "$stage1/source_object_poses/$task/$body" \
        --start 0 --end 50 2>&1 | tee -a "$stage1/logs/source_replay.log"
  done
done
