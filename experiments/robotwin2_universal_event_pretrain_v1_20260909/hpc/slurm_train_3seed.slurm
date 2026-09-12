#!/bin/bash
#SBATCH --job-name=uev_pretrain
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/train_%A_%a.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/robotwin2_universal_event_pretrain_v1_20260909/logs/train_%A_%a.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/robotwin2_universal_event_pretrain_v1_20260909
SEEDS=(20260909 20260910 20260911)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=$EXP/results/seed_${SEED}
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP"
mkdir -p "$EXP/logs" "$EXP/results"
python -u "$EXP/train.py" --data "$ROOT/data/robotwin2_universal_event_v1" \
  --output "$OUT" --v4-checkpoint "$EXP/weights/event_v4_selected_observer.pt" \
  --steps 100000 --eval-every 1000 --save-every 5000 --validation-batches 100 \
  --batch-size 64 --workers 8 --history 16 --horizon 30 --stride 3 --seed "$SEED"
