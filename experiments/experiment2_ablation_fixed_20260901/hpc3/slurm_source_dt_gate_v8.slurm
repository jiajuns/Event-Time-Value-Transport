#!/bin/bash
#SBATCH --job-name=e2src_dt_gate
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/source_dt_gate_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/source_dt_gate_%j.err

set -euo pipefail
: "${SEED:?SEED is required}"
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/experiment2_ablation_fixed_20260901
EXP1=$ROOT/experiments/current/experiment1_critic_horizontal_20260901
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP/scripts"

python -u "$EXP/scripts/audit_source_lobo_dt_gate_v8.py" \
  --data-root "$ROOT/data/exp1_rollouts" \
  --stats "$EXP1/results/train/source_zscore.npz" \
  --seed-dir "$EXP/results/train_v8/e2_cfc_event_fixed/seed${SEED}" \
  --output "$EXP/results/source_lobo_dt_gate_v8_seed${SEED}.json" \
  --device cuda
