#!/bin/bash
#SBATCH --job-name=uev_audit
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/audit_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/audit_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP"
python -u "$EXP/audit_dataset.py" \
  --data "$ROOT/data/robotwin2_universal_event_v1" \
  --output "$EXP/results/dataset_audit_${SLURM_JOB_ID}.json" \
  --minimum-clean-per-pair 40 --require-full-matrix
