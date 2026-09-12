#!/bin/bash
#SBATCH --job-name=e2fix_train
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/train_%x_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/train_%x_%j.err

set -euo pipefail
: "${METHOD:?METHOD is required}"
: "${SEED:?SEED is required}"

ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/experiment2_ablation_fixed_20260901
EXP1=$ROOT/experiments/current/experiment1_critic_horizontal_20260901
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP/scripts"

mkdir -p "$EXP/logs" "$EXP/results/train_v8"
echo "[ENV] host=$(hostname) method=$METHOD seed=$SEED cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import importlib.metadata,torch,h5py; print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(0),'ncps',importlib.metadata.version('ncps'))"

python -u "$EXP/scripts/train_experiment2_fixed_v2.py" \
  --method "$METHOD" \
  --seed "$SEED" \
  --data-root "$ROOT/data/exp1_rollouts" \
  --stats "$EXP1/results/train/source_zscore.npz" \
  --output "$EXP/results/train_v8" \
  --device cuda \
  --steps 3000 \
  --pretrain-steps 2000 \
  --batch-size 64 \
  --lr 0.002 \
  --lobo-snapshot-every 100 \
  --formal-snapshot-every 1

echo "[DONE] method=$METHOD seed=$SEED"
