#!/usr/bin/env bash
set -euo pipefail

root=/home/user/etsf_stage0
repo="$root/RoboTwin"
output="$root/matched_speed_gate_v2/raw.csv"
log="$root/matched_speed_gate_v2/run.log"
python_env=/home/user/anaconda3/envs/ETSF_RoboTwin/bin

tasks=(
  adjust_bottle
  beat_block_hammer
  handover_block
  lift_pot
  move_can_pot
  place_container_plate
)
bodies=(aloha-agilex piper ARX-X5 ur5-wsg)

mkdir -p "$root/matched_speed_gate_v2"
cd "$repo"

for task in "${tasks[@]}"; do
  for body in "${bodies[@]}"; do
    env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES=0 \
      PATH="$python_env:$PATH" \
      python ../measure_matched_robotwin_horizons.py \
        --repo "$repo" \
        --tasks "$task" \
        --embodiments "$body" \
        --seed-start 0 \
        --seed-end 10 \
        --output "$output" 2>&1 | tee -a "$log"
  done
done
