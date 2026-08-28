#!/usr/bin/env python3
"""Evaluate one frozen SmolVLA+ETSF scorer on a sealed candidate test set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from train_smolvla_etsf_action_q import (
    atomic_json,
    evaluate,
    load_groups,
    model_from_checkpoint,
    pack,
    sha256,
)


def exact_paired_sign_pvalue(improved: int, harmed: int) -> float:
    discordant = improved + harmed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(improved, harmed) + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * tail))


def validate_contract(
    checkpoint: dict[str, Any], manifest: dict[str, Any], test_resolved: set[int]
) -> None:
    prior = set(checkpoint["train_resolved_seeds"]) | set(
        checkpoint["validation_resolved_seeds"]
    )
    overlap = prior & test_resolved
    if overlap:
        raise RuntimeError(f"sealed-test resolved-scene leakage: {sorted(overlap)}")
    expected = {
        "checkpoint": checkpoint["frozen_actor_checkpoint"],
        "task": checkpoint["task"],
        "body": checkpoint["body"],
        "candidate_count": checkpoint["candidate_count"],
        "action_exec_steps": checkpoint["action_exec_steps"],
        "max_steps": checkpoint["max_steps"],
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"test contract differs from frozen checkpoint: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    test_groups, test_manifest = load_groups(args.test)
    test_resolved = {group.resolved_seed for group in test_groups}
    validate_contract(checkpoint, test_manifest, test_resolved)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model = model_from_checkpoint(checkpoint, device)
    batch = pack(test_groups, device)
    guard = checkpoint["guard"]
    guarded = evaluate(
        model,
        batch,
        args.bootstrap_seed,
        minimum_probability_margin=float(guard["minimum_probability_margin"]),
        maximum_normalized_distance=float(guard["maximum_normalized_distance"]),
    )
    unguarded = evaluate(model, batch, args.bootstrap_seed + 1)
    guarded["exact_paired_sign_pvalue"] = exact_paired_sign_pvalue(
        guarded["improved_groups"], guarded["harmed_groups"]
    )
    unguarded["exact_paired_sign_pvalue"] = exact_paired_sign_pvalue(
        unguarded["improved_groups"], unguarded["harmed_groups"]
    )
    result = {
        "schema_version": 1,
        "status": "sealed_test_evaluated_once",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "test_root": str(args.test),
        "test_manifest_sha256": sha256(args.test / "manifest.json"),
        "train_resolved_seeds": checkpoint["train_resolved_seeds"],
        "validation_resolved_seeds": checkpoint["validation_resolved_seeds"],
        "test_requested_seeds": [group.seed for group in test_groups],
        "test_resolved_seeds": [group.resolved_seed for group in test_groups],
        "validation_authorized_ranking": bool(
            checkpoint["action_ranking_authorized"]
        ),
        "gate_checks": checkpoint["gate_checks"],
        "frozen_guard": guard,
        "primary_guarded_test": guarded,
        "diagnostic_unguarded_test": unguarded,
        "interpretation_contract": (
            "A positive ETSF claim requires guarded selected successes above the "
            "same-seed baseline, no post-test tuning, and a confidence interval "
            "reported with the point estimate."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "smolvla_etsf_sealed_test_result.json", result)
    print("SEALED_TEST_COMPLETE=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
