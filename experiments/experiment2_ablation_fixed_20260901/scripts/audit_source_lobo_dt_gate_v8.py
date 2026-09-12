#!/usr/bin/env python3
"""Evaluate real-dt versus zero-dt only on source-body LOBO checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment2_fixed_core_v2 import (  # noqa: E402
    build_boundary_dataset,
    collect_rollouts,
    evaluate_model,
    json_dump,
    load_checkpoint_model,
    split_adapt_test_rollouts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rollouts = collect_rollouts(args.data_root)
    loaded = np.load(args.stats)
    dataset = build_boundary_dataset(rollouts, loaded["mean"], loaded["std"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensor_data = dataset.to_tensors(device)
    with (args.seed_dir / "run_info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    rows = []
    for fold in info["folds"]:
        holdout = fold["holdout_body"]
        adapt_ids, test_ids = split_adapt_test_rollouts(rollouts, [holdout])
        model, _ = load_checkpoint_model(Path(fold["checkpoint"]), device)
        full = evaluate_model(
            info["method"], model, dataset, tensor_data, test_ids,
            adapt_rollout_ids=adapt_ids, calibration=None, forced_beta=1.0,
        )
        zero = evaluate_model(
            info["method"], model, dataset, tensor_data, test_ids,
            adapt_rollout_ids=adapt_ids, calibration=None, forced_beta=0.0,
        )
        rows.append({
            "holdout_body": holdout,
            "full": {key: full[key] for key in (
                "bellman_mse_target", "ranking_auc", "sign_consistency"
            )},
            "zero_dt": {key: zero[key] for key in (
                "bellman_mse_target", "ranking_auc", "sign_consistency"
            )},
        })
    aggregate = {}
    for key in ("bellman_mse_target", "ranking_auc", "sign_consistency"):
        full_mean = float(np.mean([row["full"][key] for row in rows]))
        zero_mean = float(np.mean([row["zero_dt"][key] for row in rows]))
        aggregate[key] = {
            "full_mean": full_mean,
            "zero_dt_mean": zero_mean,
            "full_minus_zero": full_mean - zero_mean,
        }
    result = {
        "scope": "source_lobo_only_no_piper_or_ur5_test_evaluation",
        "seed": info["seed"],
        "folds": rows,
        "aggregate": aggregate,
        "dt_gate_pass_auc": aggregate["ranking_auc"]["full_minus_zero"] > 0.0,
        "dt_gate_pass_mse": aggregate["bellman_mse_target"]["full_minus_zero"] < 0.0,
    }
    json_dump(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["dt_gate_pass_auc"]:
        raise SystemExit(
            "source-only real-dt AUC gate failed: target evaluation is forbidden"
        )


if __name__ == "__main__":
    main()
