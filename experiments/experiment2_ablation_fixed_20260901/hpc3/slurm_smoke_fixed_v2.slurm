#!/bin/bash
#SBATCH --job-name=e2fix_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/smoke_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/logs/smoke_%j.err

set -euo pipefail
ROOT=/data/user/leviccdong/EKSF
EXP=$ROOT/experiments/current/experiment2_ablation_fixed_20260901
EXP1=$ROOT/experiments/current/experiment1_critic_horizontal_20260901
source /share/anaconda3/bin/activate
conda activate "$ROOT/env_etsf"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EXP/scripts"
export E2FIX_ROOT="$ROOT"
export E2FIX_EXP="$EXP"
export E2FIX_EXP1="$EXP1"
export E2FIX_SMOKE="/tmp/leviccdong_e2fix_smoke_${SLURM_JOB_ID}"
mkdir -p "$E2FIX_SMOKE"

python - <<'PY'
import os
from pathlib import Path
import numpy as np
import torch
from experiment2_fixed_core_v2 import (
    build_boundary_dataset, collect_rollouts, file_sha256,
    logical_data_manifest_sha256, validate_frozen_protocol,
)
from train_experiment2_fixed_v2 import (
    FROZEN_DATA_MANIFEST_SHA256, FROZEN_STATS_SHA256, train_once,
)

root = Path(os.environ["E2FIX_ROOT"])
exp1 = Path(os.environ["E2FIX_EXP1"])
data_root = root / "data/exp1_rollouts"
stats_path = exp1 / "results/train/source_zscore.npz"
assert logical_data_manifest_sha256(data_root) == FROZEN_DATA_MANIFEST_SHA256
assert file_sha256(stats_path) == FROZEN_STATS_SHA256
rollouts = collect_rollouts(data_root)
protocol = validate_frozen_protocol(rollouts)
stats = np.load(stats_path)
dataset = build_boundary_dataset(rollouts, stats["mean"], stats["std"])
source_ids = [i for i, rollout in enumerate(rollouts)
              if rollout.body in {"aloha-agilex", "arx-x5"}]
source_indices = dataset.indices_for_rollouts(source_ids)
positive_dts = dataset.arrays["event_dt"][dataset.arrays["event_dt"] > 0]
assert len(np.unique(np.round(positive_dts, 6))) > 10
print("[DATA]", protocol, "boundary_examples", dataset.size,
      "source_examples", len(source_indices),
      "event_dt_median", float(np.median(positive_dts)))
for method in ("e2_base_gru_fixed", "e2_gru_event_fixed", "e2_cfc_event_fixed"):
    checkpoint, trace, info = train_once(
        method=method, seed=20260901, dataset=dataset,
        train_indices=source_indices, device=torch.device("cuda"),
        steps=3, pretrain_steps=2, batch_size=32, base_lr=2e-3,
        output_dir=Path(os.environ["E2FIX_SMOKE"]) / method,
        snapshot_every=1, save_every_step=False,
    )
    assert all(np.isfinite(row["loss"]) for row in trace)
    print("[SMOKE_OK]", method, checkpoint, info["parameters"], trace[-1])
print("HPC_FULL600_SMOKE_OK")
PY
