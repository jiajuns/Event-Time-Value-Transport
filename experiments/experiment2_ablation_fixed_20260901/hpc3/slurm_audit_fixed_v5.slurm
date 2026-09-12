#!/bin/bash
#SBATCH --job-name=e2fix_audit
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/audit_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/audit_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/experiment2_ablation_fixed_20260901
EXP1=$ROOT/experiments/current/experiment1_critic_horizontal_20260901
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP/scripts"

python -u "$EXP/scripts/audit_experiment2_integrity_v5.py" \
  --data-root "$ROOT/data/exp1_rollouts" \
  --stats "$EXP1/results/train/source_zscore.npz" \
  --train-output "$EXP/results/train_v5" \
  --abc-train-output "$EXP/results/train_v8" \
  --results "$EXP/results/final_v8/integrity_and_cfc_use_audit.json" \
  --device cuda
