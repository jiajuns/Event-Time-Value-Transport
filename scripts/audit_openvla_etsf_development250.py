#!/usr/bin/env python3
"""Fail-closed CPU audit for old100 + development150 before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from robotwin_development_seed_contract import (
    EXPECTED_COUNT as DEVELOPMENT_GROUPS,
    REGISTRY as DEVELOPMENT_REGISTRY,
    sha256,
    validate_development_manifest,
)


FORMAT = "etsf_openvla_schema_v5_development250_audit_v1"
OLD_GROUPS = 100
TOTAL_GROUPS = OLD_GROUPS + DEVELOPMENT_GROUPS
SCHEMA_VERSION = 5
LANGUAGE_CONTRACT = "same_instruction_for_initial_query_and_all_candidate_branches"
INTERVENTION = "candidate_first_chunk_then_deterministic_actor"
POST_QUERY_ACTION_CONTRACT = "executed_as_next_query_when_nonterminal"
EXPECTED_NAMES = {
    "old100": (
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
    ),
    "development150": (
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
        "sample_blend_1.000",
    ),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def decode_strings(values: Any) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def wait_for_complete(root: Path, timeout: float, poll: float) -> None:
    started = time.monotonic()
    while True:
        path = root / "manifest.json"
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, Mapping) and value.get("status") == "complete":
                return
        if time.monotonic() - started >= timeout:
            raise RuntimeError(f"timed out waiting for complete collection: {root}")
        time.sleep(max(min(poll, 60.0), 0.01))


def _manifest_group_path(root: Path, row: Mapping[str, Any]) -> Path:
    recorded = Path(str(row.get("path", "")))
    candidates = (
        [recorded]
        if recorded.is_absolute()
        else [root / recorded, root / "groups" / recorded.name]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(recorded)


def audit_group(
    path: Path,
    *,
    expected_requested: int,
    expected_resolved: int,
    expected_names: tuple[str, ...],
) -> dict[str, Any]:
    candidate_count = len(expected_names)
    with h5py.File(path, "r") as handle:
        attrs = handle.attrs
        if (
            int(attrs.get("schema_version", -1)) != SCHEMA_VERSION
            or int(attrs.get("seed", -1)) != expected_requested
            or int(attrs.get("requested_seed", -1)) != expected_requested
            or int(attrs.get("resolved_seed", -1)) != expected_resolved
            or int(attrs.get("candidate_count", -1)) != candidate_count
            or str(attrs.get("language_contract", "")) != LANGUAGE_CONTRACT
            or not bool(attrs.get("branch_instruction_consistent", False))
            or str(attrs.get("intervention", "")) != INTERVENTION
            or str(attrs.get("post_query_action_contract", ""))
            != POST_QUERY_ACTION_CONTRACT
        ):
            raise RuntimeError(f"schema/language/identity contract mismatch: {path}")
        names = decode_strings(handle["candidate_names"][:])
        if names != expected_names:
            raise RuntimeError(f"candidate names/order mismatch: {path}: {names}")
        vector_keys = (
            "success",
            "steps",
            "duration_observed",
            "next_event_duration_observed",
            "pre_event_id",
            "post_event_id",
            "next_event_id",
            "first_chunk_executed_length",
        )
        if any(key not in handle or handle[key].shape != (candidate_count,) for key in vector_keys):
            raise RuntimeError(f"dense vector label shape mismatch: {path}")
        if (
            "candidate_actions" not in handle
            or handle["candidate_actions"].shape != (candidate_count, 25, 14)
            or "first_chunk_action_mask" not in handle
            or handle["first_chunk_action_mask"].shape != (candidate_count, 25)
            or "branches" not in handle
            or len(handle["branches"]) != candidate_count
        ):
            raise RuntimeError(f"action/branch schema mismatch: {path}")
        success = handle["success"][:].astype(bool)
        steps = handle["steps"][:].astype(np.int64)
        duration_observed = handle["duration_observed"][:].astype(bool)
        reach_observed = handle["next_event_duration_observed"][:].astype(bool)
        pre_event = handle["pre_event_id"][:].astype(np.int64)
        post_event = handle["post_event_id"][:].astype(np.int64)
        next_event = handle["next_event_id"][:].astype(np.int64)
        query_transitions = 0
        raw_event_counter: Counter[str] = Counter()
        for index in range(candidate_count):
            branch = handle["branches"][f"candidate_{index:03d}"]
            required = (
                "query_steps",
                "query_post_steps",
                "query_hidden",
                "query_post_hidden",
                "query_actions",
                "query_action_mask",
                "object_poses",
                "proprio",
                "event_names",
            )
            if any(key not in branch for key in required):
                raise RuntimeError(f"continuation/trajectory field missing: {path}")
            query_count = int(branch["query_steps"].shape[0])
            if (
                query_count < 1
                or branch["query_post_steps"].shape != (query_count,)
                or branch["query_hidden"].shape != (query_count, 4096)
                or branch["query_post_hidden"].shape != (query_count, 4096)
                or branch["query_actions"].shape != (query_count, 25, 14)
                or branch["query_action_mask"].shape != (query_count, 25)
                or branch["object_poses"].shape[0] != int(steps[index]) + 1
                or branch["proprio"].shape != (int(steps[index]) + 1, 14)
            ):
                raise RuntimeError(f"continuation/trajectory shape mismatch: {path}")
            query_transitions += query_count
            raw_event_counter.update(decode_strings(branch["event_names"][:]))
        return {
            "success": success.astype(int).tolist(),
            "steps": steps.tolist(),
            "duration_observed": duration_observed.astype(int).tolist(),
            "reach_observed": reach_observed.astype(int).tolist(),
            "pre_event": pre_event.tolist(),
            "post_event": post_event.tolist(),
            "next_event": next_event.tolist(),
            "query_transitions": query_transitions,
            "raw_event_counts": dict(raw_event_counter),
            "sha256": sha256(path),
        }


def audit_collection(
    root: Path,
    *,
    name: str,
    expected_groups: int,
    expected_names: tuple[str, ...],
    event_spec_sha256: str,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("groups")
    requested = [int(value) for value in manifest.get("requested_seeds", [])]
    resolved = [int(value) for value in manifest.get("resolved_seeds", [])]
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or int(manifest.get("completed", -1)) != expected_groups
        or int(manifest.get("candidate_count", -1)) != len(expected_names)
        or manifest.get("event_spec_sha256") != event_spec_sha256
        or not isinstance(groups, list)
        or len(groups) != expected_groups
        or len(requested) != expected_groups
        or len(resolved) != expected_groups
        or len(set(requested)) != expected_groups
        or len(set(resolved)) != expected_groups
    ):
        raise RuntimeError(f"{name} root manifest/completion contract mismatch")
    if name == "old100":
        registry = manifest.get("seed_registry")
        if registry not in (None, "", "official_150"):
            raise RuntimeError("old100 seed registry is not official/legacy")
        if any(
            manifest.get(key) not in (None, "")
            for key in (
                "fresh_seed_manifest",
                "fresh_seed_manifest_sha256",
                "development_seed_manifest",
                "development_seed_manifest_sha256",
            )
        ):
            raise RuntimeError("old100 unexpectedly binds fresh/development registry")
        registry_audit = (
            "official_150" if registry == "official_150" else "legacy_missing_registry"
        )
    else:
        if (
            manifest.get("seed_registry") != DEVELOPMENT_REGISTRY
            or manifest.get("fresh_seed_manifest") not in (None, "")
            or manifest.get("fresh_seed_manifest_sha256") not in (None, "")
            or manifest.get("blends") != [0.25, 0.5, 0.75, 1.0]
            or manifest.get("preserve_grippers") is not True
        ):
            raise RuntimeError("development150 registry/fresh/candidate contract mismatch")
        registry_audit = DEVELOPMENT_REGISTRY
    by_resolved = {int(row.get("resolved_seed", -1)): row for row in groups}
    if set(by_resolved) != set(resolved):
        raise RuntimeError(f"{name} group rows differ from resolved seed mirror")
    successes: list[int] = []
    steps: list[int] = []
    duration_observed: list[int] = []
    reach_observed: list[int] = []
    event_counts = {key: Counter() for key in ("pre", "post", "next", "raw")}
    query_transitions = 0
    outcome_variation = 0
    files = []
    per_candidate_success = np.zeros(len(expected_names), dtype=np.int64)
    for requested_seed, resolved_seed in zip(requested, resolved):
        row = by_resolved[resolved_seed]
        if int(row.get("requested_seed", row.get("seed", -1))) != requested_seed:
            raise RuntimeError(f"{name} requested/resolved ordering changed")
        path = _manifest_group_path(root, row)
        group = audit_group(
            path,
            expected_requested=requested_seed,
            expected_resolved=resolved_seed,
            expected_names=expected_names,
        )
        successes.extend(group["success"])
        steps.extend(group["steps"])
        duration_observed.extend(group["duration_observed"])
        reach_observed.extend(group["reach_observed"])
        per_candidate_success += np.asarray(group["success"], dtype=np.int64)
        outcome_variation += int(len(set(group["success"])) > 1)
        query_transitions += int(group["query_transitions"])
        event_counts["pre"].update(map(str, group["pre_event"]))
        event_counts["post"].update(map(str, group["post_event"]))
        event_counts["next"].update(map(str, group["next_event"]))
        event_counts["raw"].update(group["raw_event_counts"])
        files.append(
            {
                "requested_seed": requested_seed,
                "resolved_seed": resolved_seed,
                "path": str(path),
                "sha256": group["sha256"],
            }
        )
    branches = len(successes)
    return {
        "name": name,
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "groups": expected_groups,
        "candidate_count": len(expected_names),
        "candidate_names": list(expected_names),
        "seed_registry_audit": registry_audit,
        "requested_seeds": requested,
        "resolved_seeds": resolved,
        "files": files,
        "label_density": {
            "branches": branches,
            "success_positives": int(sum(successes)),
            "success_rate": float(np.mean(successes)),
            "successes_per_candidate": per_candidate_success.tolist(),
            "groups_with_outcome_variation": outcome_variation,
            "duration_observed": int(sum(duration_observed)),
            "duration_observed_rate": float(np.mean(duration_observed)),
            "next_reached_event_observed": int(sum(reach_observed)),
            "next_reached_event_observed_rate": float(np.mean(reach_observed)),
            "query_transitions": query_transitions,
            "mean_query_transitions_per_branch": query_transitions / branches,
            "mean_terminal_steps": float(np.mean(steps)),
            "event_id_histograms": {
                key: dict(sorted(counter.items())) for key, counter in event_counts.items()
            },
        },
    }


def audit_development250(args: argparse.Namespace) -> dict[str, Any]:
    event_spec = args.event_spec.expanduser().resolve()
    event_digest = sha256(event_spec)
    development_manifest = validate_development_manifest(
        args.development_seed_manifest, task=args.task
    )
    old = audit_collection(
        args.old_data.expanduser().resolve(),
        name="old100",
        expected_groups=OLD_GROUPS,
        expected_names=EXPECTED_NAMES["old100"],
        event_spec_sha256=event_digest,
    )
    development = audit_collection(
        args.development_data.expanduser().resolve(),
        name="development150",
        expected_groups=DEVELOPMENT_GROUPS,
        expected_names=EXPECTED_NAMES["development150"],
        event_spec_sha256=event_digest,
    )
    development_root_manifest = json.loads(
        (args.development_data / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        development_root_manifest.get("development_seed_manifest_sha256")
        != development_manifest["sha256"]
        or development_root_manifest.get("requested_seeds")
        != development_manifest["requested_seeds"]
        or development_root_manifest.get("resolved_seeds")
        != development_manifest["resolved_seeds"]
    ):
        raise RuntimeError("development collection differs from frozen seed manifest")
    fresh = json.loads(args.fresh_seed_manifest.read_text(encoding="utf-8"))
    if (
        fresh.get("status") != "fresh_confirmation_preregistered_resolved"
        or str(fresh.get("task", "")) != args.task
        or sha256(args.fresh_seed_manifest)
        != development_manifest["fresh_seed_manifest"]["sha256"]
    ):
        raise RuntimeError("fresh exclusion manifest is invalid or differs from preregistration")
    fresh_requested = {int(value) for value in fresh.get("requested_seeds", [])}
    fresh_resolved = {int(value) for value in fresh.get("resolved_seeds", [])}
    if len(fresh_requested) != 50 or len(fresh_resolved) != 50:
        raise RuntimeError("fresh exclusion manifest must contain 50 unique scenes")
    old_resolved = set(old["resolved_seeds"])
    development_resolved = set(development["resolved_seeds"])
    overlap = {
        "old100_vs_development150_resolved": sorted(old_resolved & development_resolved),
        "development150_vs_fresh50_resolved": sorted(development_resolved & fresh_resolved),
        "development150_resolved_vs_fresh50_requested": sorted(
            development_resolved & fresh_requested
        ),
        "development150_requested_vs_fresh50_resolved": sorted(
            set(development["requested_seeds"]) & fresh_resolved
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"development250 seed leakage detected: {overlap}")
    combined_density = {}
    for key in (
        "branches",
        "success_positives",
        "groups_with_outcome_variation",
        "duration_observed",
        "next_reached_event_observed",
        "query_transitions",
    ):
        combined_density[key] = sum(
            int(root["label_density"][key]) for root in (old, development)
        )
    combined_density["success_rate"] = combined_density["success_positives"] / combined_density["branches"]
    result = {
        "format": FORMAT,
        "status": "training_ready",
        "task": args.task,
        "event_spec": {"path": str(event_spec), "sha256": event_digest},
        "old100": old,
        "development150": development,
        "development_seed_manifest": development_manifest,
        "fresh50_exclusion": {
            "path": str(args.fresh_seed_manifest.resolve()),
            "sha256": sha256(args.fresh_seed_manifest),
            "labels_read": False,
            "requested_count": len(fresh_requested),
            "resolved_count": len(fresh_resolved),
        },
        "overlap_audit": overlap,
        "combined": {
            "groups": TOTAL_GROUPS,
            "candidate_branches": old["label_density"]["branches"]
            + development["label_density"]["branches"],
            "label_density": combined_density,
        },
        "training_authorized": True,
    }
    result["audit_payload_sha256"] = canonical_sha256(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--development-seed-manifest", type=Path, required=True)
    parser.add_argument("--fresh-seed-manifest", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_can_pot")
    parser.add_argument("--wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite development250 audit: {args.output}")
    if args.wait_timeout_seconds < 0 or not 0 < args.poll_seconds <= 60:
        raise ValueError("invalid wait/poll interval")
    wait_for_complete(args.old_data, args.wait_timeout_seconds, args.poll_seconds)
    wait_for_complete(
        args.development_data, args.wait_timeout_seconds, args.poll_seconds
    )
    result = audit_development250(args)
    atomic_json(args.output, result)
    print(
        "DEVELOPMENT250_AUDIT_COMPLETE="
        + json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
                "audit_payload_sha256": result["audit_payload_sha256"],
                "groups": result["combined"]["groups"],
                "candidate_branches": result["combined"]["candidate_branches"],
                "training_authorized": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
