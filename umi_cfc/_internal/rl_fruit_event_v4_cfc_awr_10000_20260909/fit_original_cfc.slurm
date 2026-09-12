#!/bin/bash
#SBATCH --job-name=rlfruit_cfc_fit
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/rlfruit_cfc_fit_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/rlfruit_cfc_fit_%j.err
set -euo pipefail
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
ROOT=/data/user/leviccdong/EKSF
RUN=$ROOT/experiments/rl_fruit_event_v4_cfc_awr_10000_20260909
TRAIN_ROOT=$ROOT/code/smolvla_train
DATA=$ROOT/data/rl_fruit_20260909/observer_head_lerobot_v3
CHECKPOINT=$RUN/weights/original_cfc_selected_observer.pt
PY=$ROOT/env_smolvla_plate062/bin/python
export PYTHONPATH=$RUN/vendor:$TRAIN_ROOT/lerobot/src:$RUN/code
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH=$ROOT/env_smolvla_plate062/bin:/usr/bin:/bin
export HF_HUB_CACHE=$TRAIN_ROOT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true OMP_NUM_THREADS=8
test ! -e "$RUN/original_cfc/features.npz"
test ! -e "$RUN/scores_original_cfc"
cd "$TRAIN_ROOT/lerobot"
"$PY" -u "$RUN/code/extract_original_cfc_features.py" \
  --data "$DATA" --checkpoint "$CHECKPOINT" --output "$RUN/original_cfc/features.npz" --device cuda
"$PY" -u "$RUN/code/fit_original_cfc_awr.py" \
  --features "$RUN/original_cfc/features.npz" --labels "$RUN/alignment/terminal_labels.json" \
  --output "$RUN/scores_original_cfc" --steps 1500 --device cuda
