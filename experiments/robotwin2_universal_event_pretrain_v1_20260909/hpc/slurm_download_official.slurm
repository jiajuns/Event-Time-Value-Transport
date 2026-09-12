#!/bin/bash
#SBATCH --job-name=uev_download
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/download_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/download_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$EXP/logs" "$ROOT/data/official/robotwin2_universal_event_replay"
python -u "$EXP/download_official.py" \
  --output "$ROOT/data/official/robotwin2_universal_event_replay"
