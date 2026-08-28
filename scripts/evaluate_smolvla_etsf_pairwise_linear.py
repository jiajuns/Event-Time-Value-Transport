#!/usr/bin/env python3
"""Evaluate a frozen low-capacity SmolVLA ETSF pairwise ranker."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

from train_smolvla_etsf_action_q import atomic_json, load_groups, sha256
from train_smolvla_etsf_pairwise_linear import evaluate, model_from_state


def exact_sign_pvalue(improved: int, harmed: int) -> float:
    discordant = improved + harmed
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(improved, harmed) + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * tail))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    model = model_from_state(checkpoint["model_state"])
    groups, manifest = load_groups(args.test)
    test_resolved = {group.resolved_seed for group in groups}
    prior = set(checkpoint["train_resolved_seeds"]) | set(
        checkpoint["validation_resolved_seeds"]
    )
    overlap = prior & test_resolved
    if overlap:
        raise RuntimeError(f"test scene leakage: {sorted(overlap)}")
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in checkpoint["contract"].items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"test contract mismatch: {mismatches}")
    guard = checkpoint["guard"]
    guarded = evaluate(
        model,
        groups,
        20260827,
        float(guard["minimum_probability_margin"]),
        float(guard["maximum_normalized_distance"]),
    )
    unguarded = evaluate(model, groups, 20260828)
    for metrics in [guarded, unguarded]:
        metrics["exact_paired_sign_pvalue"] = exact_sign_pvalue(
            metrics["improved_groups"], metrics["harmed_groups"]
        )
    result = {
        "status": "sealed_test_evaluated_once",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "test_root": str(args.test),
        "test_manifest_sha256": sha256(args.test / "manifest.json"),
        "test_requested_seeds": [group.seed for group in groups],
        "test_resolved_seeds": [group.resolved_seed for group in groups],
        "validation_authorized_ranking": checkpoint[
            "action_ranking_authorized"
        ],
        "gate_checks": checkpoint["gate_checks"],
        "configuration": checkpoint["configuration"],
        "frozen_guard": guard,
        "primary_guarded_test": guarded,
        "diagnostic_unguarded_test": unguarded,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "smolvla_etsf_pairwise_sealed_test.json", result)
    print("PAIRWISE_TEST_COMPLETE=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
