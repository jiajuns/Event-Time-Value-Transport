#!/usr/bin/env python3
"""Evaluate one frozen OpenVLA+ETSF scorer on a sealed candidate test set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from train_openvla_etsf_action_q import (
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
    if not checkpoint.get("action_ranking_authorized", False):
        raise RuntimeError("validation did not authorize opening the sealed test set")
    prior = set(checkpoint["train_resolved_seeds"]) | set(
        checkpoint["validation_resolved_seeds"]
    )
    overlap = prior & test_resolved
    if overlap:
        raise RuntimeError(f"sealed-test resolved-scene leakage: {sorted(overlap)}")
    expected = checkpoint["candidate_contract"]
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
    test_groups = load_groups(args.test)
    test_manifest = json.loads(
        (args.test / "manifest.json").read_text(encoding="utf-8")
    )
    validate_contract(
        checkpoint, test_manifest, {group.resolved_seed for group in test_groups}
    )
    device = torch.device(
        "cuda:0"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
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
    for metrics in [guarded, unguarded]:
        metrics["exact_paired_sign_pvalue"] = exact_paired_sign_pvalue(
            metrics["improved_groups"], metrics["harmed_groups"]
        )
    result = {
        "schema_version": 1,
        "status": "sealed_test_evaluated_once",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "test_root": str(args.test),
        "test_manifest_sha256": sha256(args.test / "manifest.json"),
        "test_requested_seeds": [group.seed for group in test_groups],
        "test_resolved_seeds": [group.resolved_seed for group in test_groups],
        "validation_authorized_ranking": True,
        "gate_checks": checkpoint["gate_checks"],
        "frozen_guard": guard,
        "primary_guarded_test": guarded,
        "diagnostic_unguarded_test": unguarded,
        "interpretation_contract": (
            "A positive ETSF claim requires guarded selected successes above the "
            "same-seed baseline, no post-test tuning, and the paired confidence "
            "interval reported with the point estimate."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "openvla_etsf_sealed_test_result.json", result)
    print("SEALED_TEST_COMPLETE=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
