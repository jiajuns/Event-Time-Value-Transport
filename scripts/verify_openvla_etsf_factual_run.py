#!/usr/bin/env python3
"""Fail-closed verification for formal factual-pretraining artifacts.

This verifier is intentionally read-only.  The remote launcher uses it before
skipping a completed seed, before resuming a partial seed, and after every seed
finishes.  It binds artifacts to the current data manifest, split, event spec,
cache schema and training seed so a stale JSON status cannot authorize reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CACHE_SCHEMA = 3
PREDICATE_NAMES = ["moved", "lifted", "near_goal", "stationary", "success"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(path: Path) -> dict[str, list[int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[int]] = {}
    for name in ("train", "validation", "test"):
        if name not in value:
            raise RuntimeError(f"split manifest lacks {name}: {path}")
        seeds = [
            int(row["seed"] if isinstance(row, Mapping) else row)
            for row in value[name]
        ]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"duplicate seed in split {name}")
        result[name] = sorted(seeds)
    if any(
        set(result[left]) & set(result[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("split contains overlapping seeds")
    return result


def read_task_calibration(
    data_root: Path, event_spec: Path
) -> Mapping[str, Any]:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    task = str(manifest.get("task", ""))
    value = json.loads(event_spec.read_text(encoding="utf-8"))
    calibration = value.get("calibration", {}).get(task)
    if not isinstance(calibration, Mapping):
        raise RuntimeError(f"event spec has no calibration for rollout task {task!r}")
    return calibration


def verify_cache(
    cache_path: Path,
    data_root: Path,
    split: Mapping[str, list[int]],
    event_spec_sha256: str,
    task_calibration: Mapping[str, Any],
) -> Mapping[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checks = {
        "schema": cache.get("schema_version") == CACHE_SCHEMA,
        "source_manifest": cache.get("source_manifest_sha256")
        == sha256(data_root / "manifest.json"),
        "event_spec": cache.get("event_spec_sha256") == event_spec_sha256,
        "predicates": cache.get("predicate_names") == PREDICATE_NAMES,
        "task_calibration": cache.get("task_calibration") == task_calibration,
        "split": cache.get("split_seeds") == dict(split),
        "loaded_seeds": cache.get("loaded_episode_seeds")
        == sorted([*split["train"], *split["validation"]]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"factual cache contract mismatch: {checks}")
    array_seeds = set(
        int(seed) for seed in np.asarray(cache["arrays"]["seed"]).tolist()
    )
    leaked = sorted(array_seeds & set(split["test"]))
    if leaked:
        raise RuntimeError(f"sealed test seeds present in cache arrays: {leaked}")
    sealed_files = cache.get("sealed_test_files", [])
    if sorted(int(row["seed"]) for row in sealed_files) != split["test"]:
        raise RuntimeError("sealed-test identity records differ from split")
    for row in sealed_files:
        if "success" in row or "steps" in row:
            raise RuntimeError("sealed-test cache record contains target labels")
        if not row.get("sha256"):
            raise RuntimeError("sealed-test identity record lacks SHA256")
    if not str(cache.get("sealed_test_access", "")).endswith(
        "no_episode_hdf5_open"
    ):
        raise RuntimeError("cache does not declare strict sealed-test access")
    return cache


def verify_contract(
    contract: Mapping[str, Any],
    *,
    seed: int,
    split: Mapping[str, list[int]],
    data_root: Path,
    event_spec_sha256: str,
    task_calibration: Mapping[str, Any],
) -> None:
    checks = {
        "cache_schema": contract.get("cache_schema") == CACHE_SCHEMA,
        "training_seed": contract.get("training_seed") == seed,
        "event_mode": contract.get("event_mode") == "structured",
        "source_manifest": contract.get("source_manifest_sha256")
        == sha256(data_root / "manifest.json"),
        "event_spec": contract.get("event_spec_sha256") == event_spec_sha256,
        "train": contract.get("train_seeds") == split["train"],
        "validation": contract.get("validation_seeds") == split["validation"],
        "sealed_test": contract.get("sealed_test_seeds") == split["test"],
    }
    predicate = contract.get("predicate_contract", {})
    checks.update(
        {
            "predicate_names": predicate.get("names") == PREDICATE_NAMES,
            "predicate_derivation": predicate.get("derivation")
            == "derive_atomic_predicates_v1",
            "predicate_source": predicate.get("source")
            == "simulator_object_poses_at_query_step",
            "predicate_event_spec": predicate.get("event_spec_sha256")
            == event_spec_sha256,
            "predicate_calibration": predicate.get("task_calibration")
            == task_calibration,
            "predicate_online": predicate.get("online_requires_explicit_predicates")
            is True,
            "predicate_missing": predicate.get("missing_policy") == "error",
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"factual checkpoint contract mismatch: {checks}")


def verify_resume_checkpoint(
    checkpoint_path: Path,
    *,
    seed: int,
    requested_steps: int,
    data_root: Path,
    split_manifest: Path,
    event_spec: Path,
    cache_path: Path,
) -> dict[str, Any]:
    split = read_split(split_manifest)
    event_digest = sha256(event_spec)
    task_calibration = read_task_calibration(data_root, event_spec)
    verify_cache(
        cache_path, data_root, split, event_digest, task_calibration
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    verify_contract(
        contract,
        seed=seed,
        split=split,
        data_root=data_root,
        event_spec_sha256=event_digest,
        task_calibration=task_calibration,
    )
    config = checkpoint.get("config", {})
    if config.get("structured_events") is not True:
        raise RuntimeError("resume checkpoint is not a structured event model")
    step = int(checkpoint.get("step", -1))
    best_step = int(checkpoint.get("best_step", -1))
    best_score = float(checkpoint.get("best_score", math.inf))
    if not 0 <= step < requested_steps:
        raise RuntimeError(
            f"resume step {step} must lie in [0,{requested_steps})"
        )
    if not 0 <= best_step <= step or not math.isfinite(best_score):
        raise RuntimeError("resume best-step/score metadata is invalid")
    if best_step > 0:
        best_path = checkpoint_path.parent / "event_world_model_best.pt"
        if not best_path.is_file():
            raise RuntimeError("resume references a missing best checkpoint")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best.get("contract") != contract or int(best.get("step", -1)) != best_step:
            raise RuntimeError("resume/best checkpoint contracts or steps differ")
        if not math.isclose(
            float(best.get("best_score", math.inf)),
            best_score,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError("resume/best checkpoint scores differ")
    return {"status": "resume_verified", "seed": seed, "step": step}


def verify_completed_run(
    summary_path: Path,
    *,
    seed: int,
    requested_steps: int,
    data_root: Path,
    split_manifest: Path,
    event_spec: Path,
    cache_path: Path,
) -> dict[str, Any]:
    split = read_split(split_manifest)
    event_digest = sha256(event_spec)
    task_calibration = read_task_calibration(data_root, event_spec)
    verify_cache(
        cache_path, data_root, split, event_digest, task_calibration
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = summary.get("contract", {})
    verify_contract(
        contract,
        seed=seed,
        split=split,
        data_root=data_root,
        event_spec_sha256=event_digest,
        task_calibration=task_calibration,
    )
    steps = int(summary.get("steps", -1))
    stopped_early = summary.get("stopped_early") is True
    checks = {
        "status": summary.get("status") == "training_complete",
        "requested_steps": summary.get("requested_steps") == requested_steps,
        "completion": steps == requested_steps or stopped_early,
        "sealed": summary.get("sealed_test_evaluated") is False,
        "best_step": 0 < int(summary.get("best_step", 0)) <= steps,
        "best_score": math.isfinite(
            float(summary.get("best_validation_selection_score", math.inf))
        ),
        "best_validation": isinstance(summary.get("best_validation"), Mapping),
    }
    if not all(checks.values()):
        raise RuntimeError(f"training summary is not formally complete: {checks}")

    output = summary_path.parent.resolve()
    best_path = output / "event_world_model_best.pt"
    latest_path = output / "event_world_model_latest.pt"
    if Path(str(summary.get("checkpoint", ""))).resolve() != best_path:
        raise RuntimeError("summary best-checkpoint path differs from seed output")
    if Path(str(summary.get("resume_checkpoint", ""))).resolve() != latest_path:
        raise RuntimeError("summary latest-checkpoint path differs from seed output")
    if not best_path.is_file() or not latest_path.is_file():
        raise RuntimeError("completed summary references missing checkpoint files")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if best.get("contract") != contract or latest.get("contract") != contract:
        raise RuntimeError("summary/checkpoint contracts differ")
    if int(best.get("step", -1)) != int(summary["best_step"]):
        raise RuntimeError("best checkpoint step differs from summary")
    if int(latest.get("step", -1)) != steps:
        raise RuntimeError("latest checkpoint step differs from summary")
    if (
        best.get("config", {}).get("structured_events") is not True
        or latest.get("config", {}).get("structured_events") is not True
    ):
        raise RuntimeError("best/latest checkpoint is not structured")
    if int(latest.get("best_step", -1)) != int(summary["best_step"]):
        raise RuntimeError("latest checkpoint best_step differs from summary")
    if not math.isclose(
        float(best.get("best_score", math.inf)),
        float(summary["best_validation_selection_score"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError("best checkpoint score differs from summary")
    if not math.isclose(
        float(latest.get("best_score", math.inf)),
        float(summary["best_validation_selection_score"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError("latest checkpoint best score differs from summary")
    return {
        "status": "complete_verified",
        "seed": seed,
        "steps": steps,
        "best_step": int(summary["best_step"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["complete", "resume"], required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requested-steps", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "seed": args.seed,
        "requested_steps": args.requested_steps,
        "data_root": args.data,
        "split_manifest": args.split_manifest,
        "event_spec": args.event_spec,
        "cache_path": args.cache,
    }
    result = (
        verify_completed_run(args.artifact, **common)
        if args.mode == "complete"
        else verify_resume_checkpoint(args.artifact, **common)
    )
    print("FACTUAL_RUN_VERIFIED=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
