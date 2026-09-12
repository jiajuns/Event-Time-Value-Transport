#!/bin/bash
#SBATCH --job-name=fruit_cfc3_prepare
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/fruit_cfc3_prepare_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/fruit_cfc3_prepare_%j.err
set -euo pipefail

unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
ROOT=/data/user/leviccdong/EKSF
RUN=$ROOT/experiments/cfc_awr_three_camera_fruit_pot_10000_20260911
OLD=$ROOT/experiments/rl_fruit_event_v4_cfc_awr_10000_20260909
TRAIN_ROOT=$ROOT/code/smolvla_train
RAW=$ROOT/data/rl_fruit_20260909/raw
OUT=$ROOT/data/rl_fruit_20260909/actor_three_camera_lerobot_v3
PY=$ROOT/env_smolvla_plate062/bin/python

export PYTHONPATH=$OLD/vendor:$TRAIN_ROOT/lerobot/src:$OLD/code:$RUN/code
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH=$ROOT/env_smolvla_plate062/bin:/usr/bin:/bin
export HF_HUB_CACHE=$TRAIN_ROOT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=12

test ! -e "$OUT"
cd "$TRAIN_ROOT/lerobot"
"$PY" "$RUN/code/convert_s1_hdf5_to_lerobot.py" \
  --data-dir "$RAW" \
  --root "$OUT" \
  --repo-id astribot/rl_fruit_20260909_three_camera \
  --task "put the fruit to the plate" \
  --fps 30 \
  --glob 'rl_fruit_episode_*.hdf5' \
  --cameras head,left,right
