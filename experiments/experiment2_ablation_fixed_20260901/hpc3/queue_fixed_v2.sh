#!/bin/bash
set -euo pipefail

ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/experiment2_ablation_fixed_20260901
mkdir -p "$EXP/logs" "$EXP/results/train_v8" "$EXP/results/final_v8"

SEEDS=(20260901 20260902 20260903 20260904 20260905)
TRAIN_JOBS=()
GATE_JOBS=()

for method in e2_base_gru_fixed e2_gru_event_fixed; do
  for seed in "${SEEDS[@]}"; do
    checkpoint="$EXP/results/train_v5/$method/seed$seed/formal/step_3000.pt"
    if [[ ! -s "$checkpoint" ]]; then
      echo "[REFUSE] missing audited A/A+B checkpoint: $checkpoint" >&2
      exit 1
    fi
  done
done

for seed in "${SEEDS[@]}"; do
  train_job=$(sbatch --parsable \
    --job-name="e2_cfc_v8_${seed}" \
    --export=ALL,METHOD=e2_cfc_event_fixed,SEED="$seed" \
    "$EXP/hpc3/slurm_train_fixed_v2.slurm")
  TRAIN_JOBS+=("$train_job")

  gate_job=$(sbatch --parsable \
    --dependency="afterok:$train_job" \
    --export=ALL,SEED="$seed" \
    "$EXP/hpc3/slurm_source_dt_gate_v8.slurm")
  GATE_JOBS+=("$gate_job")
  echo "[QUEUE] seed=$seed train_job=$train_job source_dt_gate_job=$gate_job"
done

gate_dependency=$(IFS=:; echo "${GATE_JOBS[*]}")
printf '%s\n' "${TRAIN_JOBS[@]}" > "$EXP/results/train_v8_job_ids.txt"
printf '%s\n' "${GATE_JOBS[@]}" > "$EXP/results/source_dt_gate_v8_job_ids.txt"

eval_job=$(sbatch --parsable \
  --dependency="afterok:$gate_dependency" \
  "$EXP/hpc3/slurm_eval_fixed_v2.slurm")
audit_job=$(sbatch --parsable \
  --dependency="afterok:$gate_dependency" \
  "$EXP/hpc3/slurm_audit_fixed_v5.slurm")
printf '%s\n' "$eval_job" > "$EXP/results/eval_v8_job_id.txt"
printf '%s\n' "$audit_job" > "$EXP/results/audit_v8_job_id.txt"
echo "[QUEUE] eval_job=$eval_job audit_job=$audit_job dependency=afterok:$gate_dependency"
