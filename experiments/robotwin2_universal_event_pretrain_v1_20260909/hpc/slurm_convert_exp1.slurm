#!/bin/bash
#SBATCH --job-name=uev_convert_exp1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/convert_exp1_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/convert_exp1_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP"
mkdir -p "$EXP/logs" "$ROOT/data/robotwin2_universal_event_v1"
python -u "$EXP/convert_exp1_failures.py" \
  --source "$ROOT/data/exp1_rollouts" \
  --output "$ROOT/data/robotwin2_universal_event_v1"
