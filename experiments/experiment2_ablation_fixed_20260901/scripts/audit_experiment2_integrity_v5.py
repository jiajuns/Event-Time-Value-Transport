#!/usr/bin/env python3
"""Hard integrity and CfC-use audit for the frozen 3000-step experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment2_fixed_core_v2 import (  # noqa: E402
    METHODS,
    STAGES,
    _selected_value,
    build_boundary_dataset,
    collect_rollouts,
    evaluate_model,
    file_sha256,
    forward_indices,
    json_dump,
    load_checkpoint_model,
    logical_data_manifest_sha256,
    split_adapt_test_rollouts,
    validate_frozen_protocol,
)
from train_experiment2_fixed_v2 import (  # noqa: E402
    FROZEN_DATA_MANIFEST_SHA256,
    FROZEN_STATS_SHA256,
    TARGET_BODIES,
)


TIME_GROUPS = (
    "time_input",
    "time_sequence",
    "time_output",
    "time_value_head",
    "time_state_head",
    "elapsed_head",
    "duration_head",
)


def parameter_delta(initial: dict, final: dict, prefix: str) -> dict[str, float]:
    names = [name for name in final if name == prefix or name.startswith(prefix + ".")]
    if not names:
        raise AssertionError(f"missing parameter group {prefix}")
    delta_sq = 0.0
    initial_sq = 0.0
    for name in names:
        before = initial[name].float()
        after = final[name].float()
        delta_sq += float((after - before).square().sum())
        initial_sq += float(before.square().sum())
    delta = math.sqrt(delta_sq)
    return {
        "parameter_tensors": len(names),
        "l2_delta": delta,
        "relative_l2_delta": delta / max(math.sqrt(initial_sq), 1e-12),
    }


def loss_window_audit(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3000 or [int(row["step"]) for row in rows] != list(range(1, 3001)):
        raise AssertionError(f"{path}: training trace is not exactly steps 1..3000")
    result = {}
    for name in (
        "loss_duration",
        "loss_state_transport",
        "loss_elapsed",
        "loss_temporal_mc",
        "loss_temporal_alignment",
    ):
        first = np.asarray([float(row[name]) for row in rows[:100]], dtype=np.float64)
        last = np.asarray([float(row[name]) for row in rows[-100:]], dtype=np.float64)
        result[name] = {
            "first100_mean": float(first.mean()),
            "last100_mean": float(last.mean()),
            "last_over_first": float(last.mean() / max(first.mean(), 1e-12)),
            "decreased": bool(last.mean() < first.mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--abc-train-output", type=Path, default=None)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_sha = logical_data_manifest_sha256(args.data_root)
    stats_sha = file_sha256(args.stats)
    if data_sha != FROZEN_DATA_MANIFEST_SHA256 or stats_sha != FROZEN_STATS_SHA256:
        raise AssertionError("frozen data/statistics hash mismatch")
    rollouts = collect_rollouts(args.data_root)
    protocol = validate_frozen_protocol(rollouts)
    loaded = np.load(args.stats)
    dataset = build_boundary_dataset(rollouts, loaded["mean"], loaded["std"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensor_data = dataset.to_tensors(device)
    adapt_ids, test_ids = split_adapt_test_rollouts(rollouts, TARGET_BODIES)

    audit: dict = {
        "protocol": protocol,
        "data_manifest_sha256": data_sha,
        "stats_sha256": stats_sha,
        "target_test_used_for_training_or_selection": False,
        "runs": [],
        "cfc_counterfactual_dt": [],
    }
    for method in METHODS:
        method_root = (
            args.abc_train_output
            if method == "e2_cfc_event_fixed" and args.abc_train_output is not None
            else args.train_output
        )
        seed_dirs = sorted((method_root / method).glob("seed*"))
        if len(seed_dirs) != 5:
            raise AssertionError(f"{method}: expected five seeds, found {len(seed_dirs)}")
        for seed_dir in seed_dirs:
            with (seed_dir / "run_info.json").open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            formal = seed_dir / "formal"
            snapshots = sorted((formal / "snapshots").glob("step_*.pt"))
            expected_names = [f"step_{step:06d}.pt" for step in range(3001)]
            if [path.name for path in snapshots] != expected_names:
                raise AssertionError(f"{formal}: formal snapshots are not exactly 0..3000")
            if int(info["formal_steps"]) != 3000:
                raise AssertionError(f"{seed_dir}: non-3000 formal run")
            if info["data"]["target_used_for_training_or_selection"]:
                raise AssertionError(f"{seed_dir}: target leakage flag is true")
            if info["data"]["manifest_sha256"] != data_sha:
                raise AssertionError(f"{seed_dir}: training data hash mismatch")
            final_path = formal / "step_3000.pt"
            final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
            if int(final_payload["step"]) != 3000 or final_payload["method"] != method:
                raise AssertionError(f"{final_path}: invalid formal checkpoint identity")
            run_record = {
                "method": method,
                "seed": int(info["seed"]),
                "formal_step": 3000,
                "formal_snapshot_count": len(snapshots),
                "target_leakage": False,
                "checkpoint": str(final_path),
            }
            if STAGES[method] == "ABC":
                initial_payload = torch.load(snapshots[0], map_location="cpu", weights_only=False)
                deltas = {
                    group: parameter_delta(initial_payload["model"], final_payload["model"], group)
                    for group in TIME_GROUPS
                }
                if not all(item["l2_delta"] > 1e-8 for item in deltas.values()):
                    raise AssertionError(f"{seed_dir}: an alleged CfC group did not update")
                run_record["cfc_parameter_updates"] = deltas
                run_record["dynamics_loss_windows"] = loss_window_audit(
                    formal / "training_trace.csv"
                )

                model, _ = load_checkpoint_model(final_path, device)
                full = evaluate_model(
                    method,
                    model,
                    dataset,
                    tensor_data,
                    test_ids,
                    adapt_rollout_ids=adapt_ids,
                    calibration=info["calibration"],
                )
                zero_dt = evaluate_model(
                    method,
                    model,
                    dataset,
                    tensor_data,
                    test_ids,
                    adapt_rollout_ids=adapt_ids,
                    calibration=info["calibration"],
                    forced_beta=0.0,
                )
                target_indices_np = dataset.indices_for_rollouts(test_ids)
                target_indices = torch.as_tensor(
                    target_indices_np, dtype=torch.long, device=device
                )
                with torch.no_grad():
                    actual_outputs = forward_indices(model, tensor_data, target_indices, beta=1.0)
                    zero_outputs = forward_indices(model, tensor_data, target_indices, beta=0.0)
                    event_ids = tensor_data.take("event_id", target_indices)
                    sensitivity = float(
                        (
                            _selected_value(actual_outputs, event_ids)
                            - _selected_value(zero_outputs, event_ids)
                        ).abs().mean()
                    )
                if sensitivity <= 1e-7:
                    raise AssertionError(f"{seed_dir}: CfC value is insensitive to dt")
                audit["cfc_counterfactual_dt"].append(
                    {
                        "seed": int(info["seed"]),
                        "mean_abs_value_change_beta1_vs_zero": sensitivity,
                        "full": {
                            key: full[key]
                            for key in ("bellman_mse_target", "ranking_auc", "sign_consistency")
                        },
                        "zero_dt": {
                            key: zero_dt[key]
                            for key in ("bellman_mse_target", "ranking_auc", "sign_consistency")
                        },
                    }
                )
            audit["runs"].append(run_record)

    audit["all_15_runs_verified"] = len(audit["runs"]) == 15
    audit["all_formal_snapshots_verified"] = all(
        row["formal_snapshot_count"] == 3001 for row in audit["runs"]
    )
    json_dump(args.results, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
