#!/usr/bin/env bash
set -euo pipefail

# Remote RTX-4090 factual pretraining launcher.  The actor is not loaded: all
# inputs are frozen OpenVLA hidden states already stored in the rollout HDF5s.
python_bin=${PYTHON_BIN:-/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python}
code_root=${CODE_ROOT:-/home/user/etsf_event_world_model_code_20260827}
data_root=${DATA_ROOT:-/home/user/etsf_openvla_rollouts_move_can_pot_20260826}
split_manifest=${SPLIT_MANIFEST:-/home/user/etsf_openvla_shadow_trained_move_can_pot_20260826/split_manifest.json}
shadow_checkpoint=${SHADOW_CHECKPOINT:-/home/user/etsf_openvla_shadow_trained_move_can_pot_20260826/openvla_etsf_shadow_selected.pt}
event_spec=${EVENT_SPEC:-/home/user/etsf_stage2_run_20260825/event_spec.json}
output_root=${OUTPUT_ROOT:-/home/user/etsf_openvla_structured_event_world_model_move_can_pot_sealed_schema3_20260827}
steps=${STEPS:-5000}
batch_size=${BATCH_SIZE:-64}
early_stopping_patience=${EARLY_STOPPING_PATIENCE:-1000}
device=${DEVICE:-cuda}
amp=${AMP:-bf16}
read -r -a seeds <<< "${SEEDS:-20260827 20260828 20260829}"

if [[ ${#seeds[@]} -ne 3 ]]; then
  echo "ERROR formal ensemble requires exactly three seeds" >&2
  exit 2
fi
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]] || [[ -n "${seen_seeds[$seed]:-}" ]]; then
    echo "ERROR seeds must be unique non-negative integers: ${seeds[*]}" >&2
    exit 2
  fi
  seen_seeds[$seed]=1
done

for required in \
  "$python_bin" \
  "$code_root/scripts/train_openvla_etsf_event_world_model.py" \
  "$code_root/scripts/verify_openvla_etsf_factual_run.py" \
  "$data_root/manifest.json" \
  "$split_manifest" \
  "$shadow_checkpoint" \
  "$event_spec"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR required formal-training input is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_root"
# Never reuse/overwrite the legacy all-episode cache filename.  This path is
# reserved for schema-3 train+validation-only transitions.
cache_path="$output_root/query_transitions_schema3_train_validation_only.pt"
verifier="$code_root/scripts/verify_openvla_etsf_factual_run.py"

verify_artifact() {
  local mode=$1
  local artifact=$2
  local seed=$3
  "$python_bin" "$verifier" \
    --mode "$mode" \
    --artifact "$artifact" \
    --seed "$seed" \
    --requested-steps "$steps" \
    --data "$data_root" \
    --split-manifest "$split_manifest" \
    --event-spec "$event_spec" \
    --cache "$cache_path"
}

for seed in "${seeds[@]}"; do
  seed_output="$output_root/seed_${seed}"
  summary_path="$seed_output/training_summary.json"
  latest_path="$seed_output/event_world_model_latest.pt"
  if [[ -f "$summary_path" ]] && \
     verify_artifact complete "$summary_path" "$seed" >/dev/null; then
    echo "SKIP_COMPLETE seed=$seed output=$seed_output"
    continue
  fi

  resume_args=()
  if [[ -f "$latest_path" ]]; then
    if ! verify_artifact resume "$latest_path" "$seed"; then
      echo "ERROR incompatible partial run; use a new OUTPUT_ROOT to preserve old files: $seed_output" >&2
      exit 3
    fi
    resume_args=(--resume "$latest_path")
  elif [[ -d "$seed_output" ]] && find "$seed_output" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR non-empty seed output has no verifiable resume checkpoint: $seed_output" >&2
    echo "Use a new OUTPUT_ROOT; existing files will not be overwritten." >&2
    exit 3
  fi

  echo "START_SEED seed=$seed output=$seed_output"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
    "$python_bin" "$code_root/scripts/train_openvla_etsf_event_world_model.py" \
      --data "$data_root" \
      --split-manifest "$split_manifest" \
      --shadow-checkpoint "$shadow_checkpoint" \
      --event-spec "$event_spec" \
      --event-mode structured \
      --cache "$cache_path" \
      --output "$seed_output" \
      --device "$device" \
      --amp "$amp" \
      --steps "$steps" \
      --batch-size "$batch_size" \
      --learning-rate 3e-4 \
      --eval-every 100 \
      --save-every 100 \
      --early-stopping-patience "$early_stopping_patience" \
      --num-workers 4 \
      --seed "$seed" \
      "${resume_args[@]}"
  verify_artifact complete "$summary_path" "$seed"
  echo "COMPLETE_SEED seed=$seed output=$seed_output"
done

# A success marker is printed only after all three artifacts pass the same
# current data/split/event/cache/checkpoint verification, including skipped runs.
for seed in "${seeds[@]}"; do
  verify_artifact complete "$output_root/seed_${seed}/training_summary.json" "$seed" >/dev/null
done
echo "ENSEMBLE_TRAINING_COMPLETE output=$output_root seeds=${seeds[*]}"
