#!/bin/bash
#SBATCH --job-name=cfc3_actor
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/cfc3_actor_%x_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/cfc3_actor_%x_%j.err
set -euo pipefail

unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
ROOT=/data/user/leviccdong/EKSF
RUN=$ROOT/experiments/cfc_awr_three_camera_fruit_pot_10000_20260911
TRAIN_ROOT=$ROOT/code/smolvla_train
FRUIT_RUN=$ROOT/experiments/rl_fruit_event_v4_cfc_awr_10000_20260909
POT_RUN=$ROOT/experiments/pot_lid_event_v4_cfc_awr_10000_20260910
PY=$ROOT/env_smolvla_plate062/bin/python
TASK_KIND=${CFC3_TASK:?CFC3_TASK must be fruit or pot}
STEPS=${CFC3_STEPS:?CFC3_STEPS must be 5 or 10000}

case "$TASK_KIND" in
  fruit)
    TASK_TEXT="put the fruit to the plate"
    SOURCE=$ROOT/outputs/smolvla_put_fruit_plate_20260908_three_camera_50000_seed1000/checkpoints/050000/pretrained_model
    SOURCE_SHA=12aad179d4a9f4a648a5a1ba2fe6ab583146828a1999d77a30f686bfc7198243
    ACTOR_DATA=$ROOT/data/rl_fruit_20260909/actor_three_camera_lerobot_v3
    REFERENCE_DATA=$ROOT/data/rl_fruit_20260909/actor_right_wrist_lerobot_v3
    SCORES=$FRUIT_RUN/scores_original_cfc
    OBSERVER=$FRUIT_RUN/weights/original_cfc_selected_observer.pt
    OBSERVER_SHA=0b9638af3dd39a39d2f6fe61c6f671c08176e4316f424c1233d4a59f86bb8b89
    EPISODES=52
    FRAMES=9649
    REPO=astribot/rl_fruit_20260909_three_camera
    FORMAL=$ROOT/outputs/smolvla_put_fruit_plate050000_three_camera_cfcawr_10000_20260911
    SEED=20260911
    ;;
  pot)
    TASK_TEXT="pick up the pot lid and place it beside the pot"
    SOURCE=$ROOT/outputs/smolvla_pick_up_pot_20260909_three_camera_70000_seed1000/checkpoints/050000/pretrained_model
    SOURCE_SHA=2160bd127ec03fde74d95be795b40a47099fff6316d4d9b09fbcf806bf834291
    ACTOR_DATA=$ROOT/data/pick_up_pot_20260909/pick_up_pot_20260909_three_camera_lerobot_v3
    REFERENCE_DATA=$ROOT/data/pick_up_pot_20260909/pick_up_pot_20260909_right_wrist_lerobot_v3
    SCORES=$POT_RUN/scores_original_cfc
    OBSERVER=$POT_RUN/weights/original_cfc_selected_observer.pt
    OBSERVER_SHA=0b9638af3dd39a39d2f6fe61c6f671c08176e4316f424c1233d4a59f86bb8b89
    EPISODES=100
    FRAMES=24754
    REPO=astribot/pick_up_pot_20260909_three_camera
    FORMAL=$ROOT/outputs/smolvla_pick_up_pot050000_three_camera_cfcawr_10000_20260911
    SEED=20260912
    ;;
  *) echo "unsupported CFC3_TASK=$TASK_KIND" >&2; exit 2 ;;
esac

if [[ "$STEPS" == 5 ]]; then
  OUT=$ROOT/outputs/smolvla_${TASK_KIND}050000_three_camera_cfcawr_smoke5_${SLURM_JOB_ID}
  SAVE=5
elif [[ "$STEPS" == 10000 ]]; then
  OUT=$FORMAL
  SAVE=1000
else
  echo "CFC3_STEPS must be 5 or 10000" >&2
  exit 2
fi

export PYTHONPATH=$FRUIT_RUN/vendor:$TRAIN_ROOT/lerobot/src:$FRUIT_RUN/code:$RUN/code
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH=$ROOT/env_smolvla_plate062/bin:/usr/bin:/bin
export HF_HUB_CACHE=$TRAIN_ROOT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true OMP_NUM_THREADS=8

test ! -e "$OUT"
PREFLIGHT=$RUN/preflight_${TASK_KIND}_${STEPS}_${SLURM_JOB_ID}.json
RECEIPT=$RUN/receipt_${TASK_KIND}_${STEPS}_${SLURM_JOB_ID}.json
cd "$TRAIN_ROOT/lerobot"

"$PY" "$RUN/code/preflight_three_camera_cfc_awr.py" \
  --task "$TASK_TEXT" --source "$SOURCE" --expected-source-sha "$SOURCE_SHA" \
  --actor-data "$ACTOR_DATA" --reference-data "$REFERENCE_DATA" \
  --scores "$SCORES" --observer "$OBSERVER" --expected-observer-sha "$OBSERVER_SHA" \
  --expected-episodes "$EPISODES" --expected-frames "$FRAMES" --output "$PREFLIGHT"

echo "TASK=$TASK_KIND METHOD=original_cfc_awr CAMERAS=head,left_wrist,right_wrist SOURCE=$SOURCE STEPS=$STEPS OUT=$OUT"
"$PY" -u "$RUN/code/train_weighted_preserve_source.py" \
  --source "$SOURCE" --expected-source-sha "$SOURCE_SHA" \
  --event-sidecar "$SCORES/event_weights.parquet" \
  --dataset.repo_id="$REPO" --dataset.root="$ACTOR_DATA" --dataset.video_backend=pyav \
  --dataloader_multiprocessing_context=fork \
  --policy.type=smolvla --policy.pretrained_path="$SOURCE" --policy.push_to_hub=false \
  --resume=false --use_policy_training_preset=false \
  --optimizer.type=adamw --optimizer.lr=0.000001 --optimizer.weight_decay=0.0000000001 \
  --optimizer.grad_clip_norm=10 --optimizer.betas='[0.9,0.95]' --optimizer.eps=0.00000001 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=100 \
  --scheduler.num_decay_steps=10000 --scheduler.peak_lr=0.000001 --scheduler.decay_lr=0.00000025 \
  --output_dir="$OUT" --job_name="${TASK_KIND}_three_camera_original_cfc_awr" \
  --batch_size=32 --num_workers=8 --steps="$STEPS" --save_freq="$SAVE" --log_freq=5 \
  --env_eval_freq=0 --eval_steps=0 --seed="$SEED" --wandb.enable=false

"$PY" "$RUN/code/verify_three_camera_actor.py" \
  --task "$TASK_TEXT" --source "$SOURCE" --expected-source-sha "$SOURCE_SHA" \
  --actor-output "$OUT" --scores "$SCORES" --preflight "$PREFLIGHT" \
  --steps "$STEPS" --save-freq "$SAVE" --receipt "$RECEIPT"
