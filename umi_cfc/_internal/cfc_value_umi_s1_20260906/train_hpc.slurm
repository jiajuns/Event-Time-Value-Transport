#!/bin/bash
#SBATCH --job-name=huanggua050000_cfcawr
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/huanggua050000_cfcawr_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/huanggua050000_cfcawr_%j.err
set -euo pipefail
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
EKSF_ROOT=/data/user/leviccdong/EKSF
CFC_RUN=$EKSF_ROOT/experiments/cfc_value_umi_s1_20260906
TRAIN_ROOT=$EKSF_ROOT/code/smolvla_train
export PYTHONPATH=$TRAIN_ROOT/lerobot/src:$CFC_RUN/code/_internal
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH=$EKSF_ROOT/env_smolvla_plate062/bin:/usr/bin:/bin
export HF_HUB_CACHE=$TRAIN_ROOT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true OMP_NUM_THREADS=8
CFC_PY=$EKSF_ROOT/env_smolvla_plate062/bin/python
CFC_SOURCE=$EKSF_ROOT/outputs/smolvla_put_plate_100_40000_20260904/checkpoints/050000/pretrained_model
CFC_DATA=$EKSF_ROOT/data/rl_huanggua_20260906/lerobot_v3
CFC_STEPS=${CFC_AWR_STEPS:-10000}
if [[ "$CFC_STEPS" == 5 ]]; then
  CFC_OUT=$EKSF_ROOT/outputs/smolvla_huanggua050000_cfcawr_smoke5_${SLURM_JOB_ID}
  CFC_SAVE=5
elif [[ "$CFC_STEPS" == 10000 ]]; then
  CFC_OUT=$EKSF_ROOT/outputs/smolvla_huanggua050000_cfcawr_10000_20260906
  CFC_SAVE=500
else
  echo 'Only smoke5 or authorized10000 supported' >&2
  exit 2
fi
test ! -e "$CFC_OUT"
cd "$TRAIN_ROOT/lerobot"
"$CFC_PY" "$CFC_RUN/code/_internal/cfc_value_umi_s1_20260906/preflight.py" \
  --source "$CFC_SOURCE" --data "$CFC_DATA" --scores "$CFC_RUN/critic" \
  --observer "$CFC_RUN/pretrain/selected_observer.pt" --labels "$CFC_RUN/terminal_labels.json" \
  --repair "$CFC_RUN/normalization_repair.json" --output "$CFC_RUN/preflight_${SLURM_JOB_ID}.json"
echo "INIT_FROM=$CFC_SOURCE FRESH_OPTIMIZER=true ADDITIONAL_STEPS=$CFC_STEPS OUTPUT=$CFC_OUT"
"$CFC_PY" -u "$CFC_RUN/code/_internal/finetune_huanggua/train_huanggua.py" \
  --normalization-repair "$CFC_RUN/normalization_repair.json" \
  --event-sidecar "$CFC_RUN/critic/event_weights.parquet" \
  --dataset.repo_id=astribot/rl_huanggua_20260906 --dataset.root="$CFC_DATA" \
  --dataset.video_backend=pyav --dataloader_multiprocessing_context=fork \
  --policy.type=smolvla --policy.pretrained_path="$CFC_SOURCE" --policy.push_to_hub=false \
  --resume=false --use_policy_training_preset=false \
  --optimizer.type=adamw --optimizer.lr=0.000001 --optimizer.weight_decay=0.0000000001 \
  --optimizer.grad_clip_norm=10 --optimizer.betas='[0.9,0.95]' --optimizer.eps=0.00000001 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=100 \
  --scheduler.num_decay_steps=10000 --scheduler.peak_lr=0.000001 --scheduler.decay_lr=0.00000025 \
  --output_dir="$CFC_OUT" --job_name=smolvla_huanggua050000_crossfit_cfcawr \
  --batch_size=32 --num_workers=8 --steps="$CFC_STEPS" --save_freq="$CFC_SAVE" --log_freq=5 \
  --env_eval_freq=0 --eval_steps=0 --seed=20260906 --wandb.enable=false
