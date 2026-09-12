#!/usr/bin/env python3
"""Evaluate fixed A/A+B/A+B+C checkpoints on the frozen target test set.

The A+B+C row written to the ablation table and the horizontal table is read
from the exact same step-3000 checkpoint for every seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment2_fixed_core_v2 import (  # noqa: E402
    METHODS,
    STAGES,
    build_boundary_dataset,
    collect_rollouts,
    compute_rho,
    evaluate_model,
    file_sha256,
    fit_beta,
    json_dump,
    load_checkpoint_model,
    logical_data_manifest_sha256,
    split_adapt_test_rollouts,
    validate_frozen_protocol,
)
from train_experiment2_fixed_v2 import (  # noqa: E402
    FROZEN_DATA_MANIFEST_SHA256,
    FROZEN_STATS_SHA256,
    SOURCE_BODIES,
    TARGET_BODIES,
)


DISPLAY_NAMES = {
    "e2_base_gru_fixed": "base-gru-fixed (A)",
    "e2_gru_event_fixed": "GRU-EVENT-fixed (A+B)",
    "e2_cfc_event_fixed": "ETSF-CfC-optimized (A+B+C)",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def adaptation_wall_seconds(
    method: str,
    model,
    dataset,
    tensor_data,
    adapt_ids: list[int],
) -> tuple[float, dict]:
    if STAGES[method] == "A":
        return 0.0, {"beta": {}, "rho": {}, "mode": "zero_shot"}
    if tensor_data.device.type == "cuda":
        torch.cuda.synchronize(tensor_data.device)
    started = time.perf_counter()
    rho = compute_rho(dataset.rollouts, adapt_ids)
    beta = {}
    for body in TARGET_BODIES:
        body_ids = [i for i in adapt_ids if dataset.rollouts[i].body == body]
        beta[body] = fit_beta(
            method, model, tensor_data, dataset.indices_for_rollouts(body_ids)
        )
    if tensor_data.device.type == "cuda":
        torch.cuda.synchronize(tensor_data.device)
    elapsed = time.perf_counter() - started
    return elapsed, {
        "beta": beta,
        "rho": rho,
        "mode": "statistics_only_zero_target_td",
        "target_td_steps": 0,
        "updated_params": 0,
    }


def summarize(raw_rows: list[dict]) -> list[dict]:
    summaries = []
    for method in METHODS:
        rows = [row for row in raw_rows if row["method"] == method]
        if not rows:
            continue
        summary = {
            "method": method,
            "model": DISPLAY_NAMES[method],
            "stage": STAGES[method],
            "seeds": len(rows),
        }
        for name in (
            "bellman_mse_target", "ranking_auc", "sign_consistency",
            "success_rate_mae", "success_brier", "adaptation_wall_s"
        ):
            values = np.asarray(
                [float(row[name]) for row in rows if row[name] not in (None, "")],
                dtype=np.float64,
            )
            summary[f"{name}_mean"] = round(float(values.mean()), 6) if len(values) else None
            summary[f"{name}_std"] = round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0 if len(values) else None
        summary["updated_params"] = 0
        summary["formal_step"] = 3000
        summaries.append(summary)

    quality = {
        "bellman_mse_target_mean": min,
        "ranking_auc_mean": max,
        "sign_consistency_mean": max,
        "success_rate_mae_mean": min,
    }
    for name, operation in quality.items():
        available = [row[name] for row in summaries if row[name] is not None]
        best = operation(available) if available else None
        for row in summaries:
            row[f"best_{name}"] = (
                best is not None
                and row[name] is not None
                and abs(float(row[name]) - float(best)) < 1e-12
            )
    return summaries


def horizontal_rows(legacy_path: Path, full_summary: dict) -> list[dict]:
    legacy = read_csv(legacy_path)
    rows = []
    for row in legacy:
        rows.append(
            {
                "baseline": row["baseline"],
                "seeds": row["seeds"],
                "bellman_mse_target_mean": row["bellman_mse_target_mean"],
                "ranking_auc_mean": row["ranking_auc_mean"],
                "sign_consistency_mean": row["sign_consistency_mean"],
                "success_rate_mae_mean": row["success_rate_mae_mean"],
                "updated_params": row["updated_params"],
                "wall_clock_s_mean": row["wall_clock_s_mean"],
                "checkpoint_source": "legacy_frozen_table_ii",
            }
        )
    rows.append(
        {
            "baseline": "b9_fixed_full_abc",
            "seeds": full_summary["seeds"],
            "bellman_mse_target_mean": full_summary["bellman_mse_target_mean"],
            "ranking_auc_mean": full_summary["ranking_auc_mean"],
            "sign_consistency_mean": full_summary["sign_consistency_mean"],
            "success_rate_mae_mean": full_summary["success_rate_mae_mean"],
            "updated_params": 0,
            "wall_clock_s_mean": full_summary["adaptation_wall_s_mean"],
            "checkpoint_source": "same_step3000_as_ablation_abc",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument(
        "--abc-train-output",
        type=Path,
        default=None,
        help="optional C-only rerun root; A and A+B still come from --train-output",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--legacy-horizontal-summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-data-sha256", default=FROZEN_DATA_MANIFEST_SHA256)
    parser.add_argument("--expected-stats-sha256", default=FROZEN_STATS_SHA256)
    args = parser.parse_args()

    data_sha = logical_data_manifest_sha256(args.data_root)
    stats_sha = file_sha256(args.stats)
    if data_sha != args.expected_data_sha256:
        raise SystemExit(f"DATA HASH MISMATCH: {data_sha} != {args.expected_data_sha256}")
    if stats_sha != args.expected_stats_sha256:
        raise SystemExit(f"STATS HASH MISMATCH: {stats_sha} != {args.expected_stats_sha256}")
    rollouts = collect_rollouts(args.data_root)
    protocol = validate_frozen_protocol(rollouts)
    loaded = np.load(args.stats)
    dataset = build_boundary_dataset(rollouts, loaded["mean"], loaded["std"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensor_data = dataset.to_tensors(device)
    adapt_ids, test_ids = split_adapt_test_rollouts(rollouts, TARGET_BODIES)
    if len(adapt_ids) != 10 or len(test_ids) != 50:
        raise AssertionError(f"target split mismatch: adapt={len(adapt_ids)} test={len(test_ids)}")

    raw_rows: list[dict] = []
    adaptation_records: list[dict] = []
    checkpoint_identity: dict[str, dict[str, str]] = {}
    for method in METHODS:
        method_root = (
            args.abc_train_output
            if method == "e2_cfc_event_fixed" and args.abc_train_output is not None
            else args.train_output
        )
        for seed_dir in sorted((method_root / method).glob("seed*")):
            info_path = seed_dir / "run_info.json"
            checkpoint = seed_dir / "formal" / "step_3000.pt"
            if not info_path.exists() or not checkpoint.exists():
                print(f"[SKIP] incomplete run {seed_dir}", flush=True)
                continue
            with info_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            if int(info["formal_steps"]) != 3000:
                raise AssertionError(f"{seed_dir}: non-3000 formal model")
            if info["data"]["manifest_sha256"] != data_sha:
                raise AssertionError(f"{seed_dir}: train/eval data mismatch")
            seed = int(info["seed"])
            model, payload = load_checkpoint_model(checkpoint, device)
            if int(payload["step"]) != 3000:
                raise AssertionError(f"{checkpoint}: checkpoint is not step 3000")
            adapt_seconds, adaptation = adaptation_wall_seconds(
                method, model, dataset, tensor_data, adapt_ids
            )
            metrics = evaluate_model(
                method,
                model,
                dataset,
                tensor_data,
                test_ids,
                adapt_rollout_ids=adapt_ids,
                calibration=info["calibration"],
                return_predictions=False,
            )
            if STAGES[method] == "A":
                official_success_mae = metrics["success_rate_mae_value_s0"]
                official_success_brier = metrics["success_brier_value_s0"]
                official_success_readout = "raw_V_s0"
            else:
                official_success_mae = metrics["success_rate_mae_rho_chain"]
                official_success_brier = metrics["success_brier_rho_chain"]
                official_success_readout = "adapt_N5_product_rho"
            row = {
                "method": method,
                "model": DISPLAY_NAMES[method],
                "stage": STAGES[method],
                "seed": seed,
                "formal_step": 3000,
                "bellman_mse_target": metrics["bellman_mse_target"],
                "ranking_auc": metrics["ranking_auc"],
                "sign_consistency": metrics["sign_consistency"],
                "success_rate_mae": official_success_mae,
                "success_brier": official_success_brier,
                "official_success_readout": official_success_readout,
                "success_head_calibrated_mae": metrics["success_rate_mae"],
                "success_head_calibrated_brier": metrics["success_brier"],
                "success_rate_mae_value_s0": metrics["success_rate_mae_value_s0"],
                "success_brier_value_s0": metrics["success_brier_value_s0"],
                "success_rate_mae_rho_chain": metrics["success_rate_mae_rho_chain"],
                "success_brier_rho_chain": metrics["success_brier_rho_chain"],
                "updated_params": 0,
                "adaptation_wall_s": adapt_seconds,
                "checkpoint": str(checkpoint),
            }
            raw_rows.append(row)
            adaptation_records.append(
                {"method": method, "seed": seed, "adaptation": adaptation}
            )
            checkpoint_identity.setdefault(str(seed), {})[method] = str(checkpoint)
            print(f"[EVAL] {json.dumps(row, ensure_ascii=False)}", flush=True)

    for method in METHODS:
        count = sum(row["method"] == method for row in raw_rows)
        if count != 5:
            raise SystemExit(f"formal evaluation requires five seeds for {method}, found {count}")
    summaries = summarize(raw_rows)
    full_summary = next(row for row in summaries if row["method"] == "e2_cfc_event_fixed")
    horizontal = horizontal_rows(args.legacy_horizontal_summary, full_summary)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "ablation_raw_metrics.csv", raw_rows)
    write_csv(args.results_dir / "ablation_summary.csv", summaries)
    write_csv(args.results_dir / "horizontal_with_same_abc_summary.csv", horizontal)
    json_dump(args.results_dir / "adaptation_records.json", adaptation_records)
    json_dump(
        args.results_dir / "evaluation_manifest.json",
        {
            "data": protocol,
            "data_manifest_sha256": data_sha,
            "stats_sha256": stats_sha,
            "adapt_rollouts": 10,
            "test_rollouts": 50,
            "formal_step": 3000,
            "checkpoint_identity": checkpoint_identity,
            "same_abc_checkpoint_used_in_ablation_and_horizontal": True,
            "target_used_for_training_selection_or_calibration": False,
        },
    )
    print(f"[DONE] results={args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
