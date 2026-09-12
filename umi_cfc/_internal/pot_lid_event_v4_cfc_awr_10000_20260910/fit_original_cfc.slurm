#!/bin/bash
#SBATCH --job-name=potrl_cfcfit
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/potrl_cfcfit_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/potrl_cfcfit_%j.err
set -euo pipefail
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
ROOT=/data/user/leviccdong/EKSF
RUN=$ROOT/experiments/pot_lid_event_v4_cfc_awr_10000_20260910
DATA=$ROOT/data/pick_up_pot_20260909/pick_up_pot_20260909_three_camera_lerobot_v3
PY=$ROOT/env_smolvla_plate062/bin/python
export PYTHONPATH=$RUN/vendor:$RUN/code:$RUN/code/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
test ! -e "$RUN/original_cfc_features.npz"; test ! -e "$RUN/scores_original_cfc"
"$PY" -u "$RUN/code/extract_original_cfc_pot.py" --data "$DATA" \
  --checkpoint "$RUN/weights/original_cfc_selected_observer.pt" --alignment "$RUN/alignment" \
  --output "$RUN/original_cfc_features.npz" --device cuda
"$PY" -u "$RUN/code/fit_original_cfc_pot_awr.py" --features "$RUN/original_cfc_features.npz" \
  --labels "$RUN/alignment/terminal_labels.json" --output "$RUN/scores_original_cfc" --steps 1500 --device cuda
