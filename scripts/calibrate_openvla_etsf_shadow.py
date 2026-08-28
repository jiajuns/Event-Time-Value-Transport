#!/usr/bin/env python3
"""Validation-only safety calibration for an existing OpenVLA ETSF shadow."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from train_openvla_etsf_shadow import (
    BRIDGE,
    CLOCK,
    EVENTS,
    OpenVLAETSFShadow,
    atomic_json,
    audit_dataset,
    baselines,
    bootstrap_auc_lower,
    evaluate,
    fit_clock_shrinkage,
    fit_probability_calibration,
    load_episodes,
    pack,
    split_episodes,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Never authorize ranking because this is not a new sealed confirmation set.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    audit = audit_dataset(args.data)
    episodes = load_episodes(args.data)
    splits = split_episodes(episodes)
    batches = {name: pack(group, device) for name, group in splits.items()}
    clock_baseline = baselines(splits["train"], batches["train"])

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clock_target = str(payload.get("clock_target", "absolute"))
    model = OpenVLAETSFShadow(clock_target).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    temperature, probability_shrinkage = fit_probability_calibration(
        model, batches["validation"], clock_baseline
    )
    clock_shrinkage = fit_clock_shrinkage(
        model,
        batches["validation"],
        clock_baseline,
        clock_target,
    )
    validation_uncalibrated, _ = evaluate(
        model,
        splits["validation"],
        batches["validation"],
        clock_baseline,
        clock_target_name=clock_target,
    )
    validation, _ = evaluate(
        model,
        splits["validation"],
        batches["validation"],
        clock_baseline,
        temperature,
        clock_target,
        clock_shrinkage,
        probability_shrinkage,
    )
    test_uncalibrated, _ = evaluate(
        model,
        splits["test"],
        batches["test"],
        clock_baseline,
        clock_target_name=clock_target,
    )
    test, per_event = evaluate(
        model,
        splits["test"],
        batches["test"],
        clock_baseline,
        temperature,
        clock_target,
        clock_shrinkage,
        probability_shrinkage,
    )
    lower = bootstrap_auc_lower(
        model,
        splits["test"],
        batches["test"],
        clock_baseline,
        temperature,
        probability_shrinkage,
    )
    test["same_event_auc_episode_bootstrap_lower_95"] = lower
    checks = {
        "rollouts_at_least_100": len(episodes) >= 100,
        "heldout_has_both_outcomes": (
            test["n_success"] >= 4 and test["n_failure"] >= 4
        ),
        "same_event_pairs_at_least_50": test["same_event_pairs"] >= 50,
        "same_event_auc_above_event_counter": test["same_event_micro_auc"] > 0.5,
        "same_event_auc_lower_bound_above_chance": lower > 0.5,
        "semantic_brier_beats_event_rate": (
            test["same_event_brier"] < test["event_rate_baseline_brier"]
        ),
        "clock_beats_event_median": (
            test["clock_duration_mae"] < test["event_median_duration_mae"]
        ),
        "new_confirmation_set_sealed": not args.development_only,
    }
    authorized = all(checks.values())
    calibrated_path = args.output / "openvla_etsf_shadow_calibrated.pt"
    torch.save(
        {
            **payload,
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "temperature": temperature,
            "probability_shrinkage": probability_shrinkage,
            "clock_shrinkage": clock_shrinkage,
            "action_ranking_authorized": authorized,
        },
        calibrated_path,
    )
    summary = {
        "status": "validation_only_calibration_complete",
        "development_only": args.development_only,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "selected_seed": payload.get("selected_seed", payload.get("seed")),
        "openvla_frozen": True,
        "input_dim": 4096,
        "bridge_dim": BRIDGE,
        "clock_dim": CLOCK,
        "events": EVENTS,
        "temperature": temperature,
        "probability_shrinkage": probability_shrinkage,
        "clock_shrinkage": clock_shrinkage,
        "data_audit": audit,
        "validation_metrics_uncalibrated": validation_uncalibrated,
        "validation_metrics": validation,
        "test_metrics_uncalibrated": test_uncalibrated,
        "test_metrics": test,
        "test_per_event": per_event,
        "gate_checks": checks,
        "action_ranking_authorized": authorized,
        "policy_effect_during_calibration": False,
        "next_action": (
            "keep ETSF in shadow mode; collect a newly sealed confirmation set"
        ),
    }
    atomic_json(args.output / "calibration_gate_summary.json", summary)
    print("CALIBRATION_GATE=" + str(summary), flush=True)


if __name__ == "__main__":
    main()
