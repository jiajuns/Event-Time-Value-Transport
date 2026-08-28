#!/usr/bin/env python3
"""Frozen, label-free seed contract for RoboTwin development expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CANDIDATE_FORMAT = "etsf_robotwin_development_seed_candidates_v1"
MANIFEST_FORMAT = "etsf_robotwin_development_seed_manifest_v1"
CANDIDATE_STATUS = "development_expansion_candidates_preregistered_unresolved"
MANIFEST_STATUS = "development_expansion_preregistered_resolved"
REGISTRY = "explicit_development_expansion"
PURPOSE = "model_development_only_never_fresh_confirmation"
LABEL_ACCESS_CONTRACT = (
    "reset_identity_only_no_policy_no_action_no_event_no_success_no_reward"
)
EXPECTED_COUNT = 150


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def resolve_recorded_path(recorded: str, anchor: Path) -> Path:
    path = Path(recorded).expanduser()
    if path.is_file():
        return path.resolve()
    portable = anchor / path.name
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(path)


def expand_candidate_requested_seeds(candidate: Mapping[str, Any]) -> list[int]:
    explicit = candidate.get("candidate_requested_seeds")
    compact = candidate.get("candidate_seed_range")
    if (explicit is None) == (compact is None):
        raise RuntimeError(
            "candidate manifest must contain exactly one of "
            "candidate_requested_seeds or candidate_seed_range"
        )
    if explicit is not None:
        if not isinstance(explicit, list):
            raise RuntimeError("candidate_requested_seeds must be a list")
        seeds = [int(value) for value in explicit]
    else:
        if not isinstance(compact, Mapping):
            raise RuntimeError("candidate_seed_range must be a mapping")
        start = int(compact.get("start", -1))
        count = int(compact.get("count", 0))
        step = int(compact.get("step", 0))
        if start < 0 or count <= 0 or step <= 0:
            raise RuntimeError("candidate_seed_range start/count/step are invalid")
        seeds = [start + step * index for index in range(count)]
    if len(seeds) < EXPECTED_COUNT or len(set(seeds)) != len(seeds):
        raise RuntimeError("development candidate pool is too small or duplicated")
    return seeds


def official_seeds(registry: Mapping[str, Any], task: str) -> list[int]:
    task_record = registry.get(task)
    if not isinstance(task_record, Mapping):
        raise RuntimeError("official registry lacks development task")
    values = task_record.get("success_seeds")
    if not isinstance(values, list):
        raise RuntimeError("official registry lacks success_seeds")
    seeds = [int(value) for value in values]
    if len(seeds) != 150 or len(set(seeds)) != 150:
        raise RuntimeError("official registry must contain exactly 150 unique seeds")
    return seeds


def fresh_seed_sets(fresh: Mapping[str, Any], task: str) -> tuple[list[int], list[int]]:
    if (
        fresh.get("status") != "fresh_confirmation_preregistered_resolved"
        or str(fresh.get("task", "")) != task
    ):
        raise RuntimeError("fresh exclusion manifest is not frozen/resolved")
    requested = [int(value) for value in fresh.get("requested_seeds", [])]
    resolved = [int(value) for value in fresh.get("resolved_seeds", [])]
    if (
        len(requested) != 50
        or len(resolved) != 50
        or len(set(requested)) != 50
        or len(set(resolved)) != 50
    ):
        raise RuntimeError("fresh exclusion manifest must contain exactly 50 scenes")
    return requested, resolved


def validate_development_manifest(
    path: Path,
    *,
    task: str,
    verify_source_sha256: bool = True,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("development manifest must contain a JSON object")
    unsigned = dict(manifest)
    payload_digest = str(unsigned.pop("manifest_payload_sha256", ""))
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or int(manifest.get("schema_version", -1)) != 1
        or manifest.get("status") != MANIFEST_STATUS
        or str(manifest.get("task", "")) != task
        or manifest.get("purpose") != PURPOSE
        or manifest.get("seed_registry") != REGISTRY
        or manifest.get("label_access_contract") != LABEL_ACCESS_CONTRACT
        or payload_digest != canonical_sha256(unsigned)
    ):
        raise RuntimeError("development seed manifest/signature contract is invalid")
    rows = manifest.get("train")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise RuntimeError("development manifest must contain exactly 150 train rows")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("development seed row is invalid")
        requested = int(row.get("requested_seed", row.get("seed", -1)))
        resolved = int(row.get("resolved_seed", -1))
        if int(row.get("seed", requested)) != requested or min(requested, resolved) < 0:
            raise RuntimeError("development seed identity row is invalid")
        normalized.append(
            {"seed": requested, "requested_seed": requested, "resolved_seed": resolved}
        )
    requested = [row["requested_seed"] for row in normalized]
    resolved = [row["resolved_seed"] for row in normalized]
    if (
        len(set(requested)) != EXPECTED_COUNT
        or len(set(resolved)) != EXPECTED_COUNT
        or [int(value) for value in manifest.get("requested_seeds", [])] != requested
        or [int(value) for value in manifest.get("resolved_seeds", [])] != resolved
    ):
        raise RuntimeError("development requested/resolved seed mirror is invalid")

    candidate_path = resolve_recorded_path(
        str(manifest.get("candidate_manifest", "")), path.parent
    )
    official_path = resolve_recorded_path(
        str(manifest.get("official_seed_registry", "")), path.parent
    )
    fresh_path = resolve_recorded_path(
        str(manifest.get("fresh_seed_manifest", "")), path.parent
    )
    expected_sha = {
        candidate_path: str(manifest.get("candidate_manifest_sha256", "")),
        official_path: str(manifest.get("official_seed_registry_sha256", "")),
        fresh_path: str(manifest.get("fresh_seed_manifest_sha256", "")),
    }
    if verify_source_sha256 and any(
        sha256(source) != digest for source, digest in expected_sha.items()
    ):
        raise RuntimeError("development seed source artifact SHA256 mismatch")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("format") != CANDIDATE_FORMAT
        or candidate.get("status") != CANDIDATE_STATUS
        or candidate.get("purpose") != PURPOSE
        or str(candidate.get("task", "")) != task
    ):
        raise RuntimeError("development candidate manifest is invalid")
    candidates = expand_candidate_requested_seeds(candidate)
    positions = [candidates.index(seed) for seed in requested]
    if positions != sorted(positions):
        raise RuntimeError("development selected seeds changed preregistered order")
    official = json.loads(official_path.read_text(encoding="utf-8"))
    official_values = official_seeds(official, task)
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    fresh_requested, fresh_resolved = fresh_seed_sets(fresh, task)
    excluded = set(official_values) | set(fresh_requested) | set(fresh_resolved)
    overlap = (set(requested) | set(resolved)) & excluded
    if overlap:
        raise RuntimeError(
            f"development seeds overlap official/fresh exclusion sets: {sorted(overlap)}"
        )
    audit = manifest.get("audit")
    if not isinstance(audit, list):
        raise RuntimeError("development reset-only audit is missing")
    selected_audit = [
        {
            "requested_seed": int(row["requested_seed"]),
            "resolved_seed": int(row["resolved_seed"]),
        }
        for row in audit
        if isinstance(row, Mapping) and row.get("decision") == "selected"
    ]
    if selected_audit != [
        {"requested_seed": row["requested_seed"], "resolved_seed": row["resolved_seed"]}
        for row in normalized
    ]:
        raise RuntimeError("development selected rows differ from reset-only audit")
    if any(
        isinstance(row, Mapping)
        and set(row) & {"success", "reward", "event", "policy_prediction", "action"}
        for row in audit
    ):
        raise RuntimeError("development reset audit contains forbidden label/policy fields")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "manifest_payload_sha256": payload_digest,
        "task": task,
        "rows": normalized,
        "requested_seeds": requested,
        "resolved_seeds": resolved,
        "candidate_manifest": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "official_seed_registry": {"path": str(official_path), "sha256": sha256(official_path)},
        "fresh_seed_manifest": {"path": str(fresh_path), "sha256": sha256(fresh_path)},
        "label_access_contract": LABEL_ACCESS_CONTRACT,
        "seed_registry": REGISTRY,
    }


__all__ = [
    "CANDIDATE_FORMAT",
    "CANDIDATE_STATUS",
    "EXPECTED_COUNT",
    "LABEL_ACCESS_CONTRACT",
    "MANIFEST_FORMAT",
    "MANIFEST_STATUS",
    "PURPOSE",
    "REGISTRY",
    "canonical_sha256",
    "expand_candidate_requested_seeds",
    "fresh_seed_sets",
    "official_seeds",
    "sha256",
    "validate_development_manifest",
]
