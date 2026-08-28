#!/usr/bin/env python3
"""Authenticated, data-blind physical integrity audit for schema-v5 object poses.

The audit reads only adaptive-development D250 artifacts.  It authenticates the
R3 OOF manifest and its holdout payloads, authenticates every referenced schema5
HDF, and checks simulator-recorded pose geometry plus R3 physical xyz deltas.
Thresholds are loaded from a separately hashed task/engine specification and
must explicitly state that they were not fitted from trajectory statistics.

This script never trains a model, imports a simulator, starts an environment, or
accepts Fresh/confirmation paths.  The authenticated development payload is a
monolithic file that also contains outcome fields; they are deserialized but are
never accessed by audit logic or used to choose thresholds.  A geometrically
clean result is not evidence that the object head was learned or that pose
quality was independently known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from enum import IntFlag
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


FORMAT = "etsf_schema5_object_pose_integrity_audit_v1"
SPEC_FORMAT = "etsf_schema5_object_pose_integrity_spec_v1"
EXPECTED_R3_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
EXPECTED_HOLDOUT_FORMAT = "etsf_v8_detached_adapter_holdout_input_v1"
QUALITY_UNAVAILABLE = "unavailable_schema5_collector_has_no_quality_field_fail_closed"


class InvalidReason(IntFlag):
    MISSING_REQUIRED_FIELD = 1 << 0
    METADATA_MISMATCH = 1 << 1
    OBJECT_REGISTRY_MISMATCH = 1 << 2
    NONFINITE_POSE = 1 << 3
    QUATERNION_NORM = 1 << 4
    WORLD_AABB = 1 << 5
    STEP_TRANSLATION = 1 << 6
    STEP_ROTATION = 1 << 7
    CANDIDATE_RESET_MISMATCH = 1 << 8
    ENDPOINT_MISMATCH = 1 << 9
    STATIC_OBJECT_MOTION = 1 << 10
    MATERIALIZED_DELTA_MISMATCH = 1 << 11
    MATERIALIZED_LAYOUT_MISMATCH = 1 << 12


INVALID_REASON_NAMES = {int(reason): reason.name.lower() for reason in InvalidReason}
COVERAGE_GAPS = {
    1: "schema5_has_no_signed_pose_quality_valid_field",
    2: "r3_object_adapter_has_no_learned_object_output",
    4: "r3_materialization_retains_xyz_delta_only_not_orientation_delta",
}


class AuditContractError(ValueError):
    """Authentication or immutable audit-contract violation."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _has_fresh(path: Path) -> bool:
    return any("fresh" in part.casefold() or "confirmation" in part.casefold() for part in path.parts)


def safe_path(value: str | Path, name: str, *, must_exist: bool = True) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise AuditContractError(f"{name} must be absolute")
    resolved = raw.resolve(strict=False)
    if _has_fresh(raw) or _has_fresh(resolved):
        raise AuditContractError(f"{name} must not reference Fresh/confirmation paths")
    if must_exist and not resolved.is_file():
        raise AuditContractError(f"{name} does not exist")
    return resolved


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AuditContractError(f"{name} must contain a JSON object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise AuditContractError(
            f"{name} fields differ: missing={sorted(fields-set(value))}, "
            f"extra={sorted(set(value)-fields)}"
        )


def _artifact(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AuditContractError(f"{name} must be an artifact mapping")
    _exact(value, {"path", "sha256"}, name)
    path = safe_path(str(value["path"]), f"{name}.path")
    sha = str(value["sha256"])
    if not _is_sha(sha) or file_sha256(path) != sha:
        raise AuditContractError(f"{name} SHA mismatch")
    return {"path": str(path), "sha256": sha}


def validate_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a pre-scan task/engine threshold specification."""

    _exact(
        value,
        {
            "format",
            "timing_scope",
            "task",
            "body",
            "schema_version",
            "expected_groups",
            "expected_candidates",
            "expected_object_names",
            "materialized_object_names",
            "pose_layout",
            "thresholds",
            "threshold_basis",
            "source_artifacts",
            "access_contract",
        },
        "spec",
    )
    if value["format"] != SPEC_FORMAT:
        raise AuditContractError("unexpected pose-integrity spec format")
    if value["timing_scope"] != "adaptive_development_audit_frozen_before_full_d250_pose_scan":
        raise AuditContractError("audit threshold timing scope changed")
    if value["task"] != "move_can_pot" or value["body"] != "piper_piper_0.6":
        raise AuditContractError("spec task/body differs from authenticated D250")
    if value["schema_version"] != 5 or value["expected_groups"] != 250 or value["expected_candidates"] != 4:
        raise AuditContractError("spec schema/group/candidate cardinality changed")
    expected_names = ["can", "cluttered_obj", "pot", "table", "wall"]
    if value["expected_object_names"] != expected_names:
        raise AuditContractError("task object name/order registry changed")
    if value["materialized_object_names"] != ["can"]:
        raise AuditContractError("R3 materialized object registry must remain can-only")
    if value["pose_layout"] != {
        "shape_suffix": [7],
        "translation_indices": [0, 1, 2],
        "quaternion_indices": [3, 4, 5, 6],
        "quaternion_order": "wxyz",
        "frame": "simulator_world",
        "translation_unit": "metre",
    }:
        raise AuditContractError("pose layout changed")
    thresholds = value["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise AuditContractError("thresholds must be a mapping")
    _exact(
        thresholds,
        {
            "world_aabb_m",
            "quaternion_norm_abs_tolerance",
            "max_step_translation_m",
            "max_step_rotation_rad",
            "endpoint_match_atol",
            "candidate_reset_match_atol",
            "materialized_delta_match_atol",
            "static_object_max_step_translation_m",
            "static_object_max_step_rotation_rad",
            "static_object_names",
        },
        "thresholds",
    )
    aabb = np.asarray(thresholds["world_aabb_m"], dtype=np.float64)
    if aabb.shape != (3, 2) or not np.isfinite(aabb).all() or np.any(aabb[:, 0] >= aabb[:, 1]):
        raise AuditContractError("world AABB must be three finite increasing pairs")
    diagonal = float(np.linalg.norm(aabb[:, 1] - aabb[:, 0]))
    numeric = {
        key: float(thresholds[key])
        for key in (
            "quaternion_norm_abs_tolerance",
            "max_step_translation_m",
            "max_step_rotation_rad",
            "endpoint_match_atol",
            "candidate_reset_match_atol",
            "materialized_delta_match_atol",
            "static_object_max_step_translation_m",
            "static_object_max_step_rotation_rad",
        )
    }
    if not np.isfinite(list(numeric.values())).all() or any(value < 0 for value in numeric.values()):
        raise AuditContractError("physical thresholds must be finite/non-negative")
    if not math.isclose(numeric["max_step_translation_m"], diagonal, rel_tol=0.0, abs_tol=1e-12):
        raise AuditContractError("max translation must be the code-derived AABB diagonal")
    if not math.isclose(numeric["max_step_rotation_rad"], math.pi, rel_tol=0.0, abs_tol=1e-12):
        raise AuditContractError("max rotation must be the quaternion geodesic bound pi")
    if thresholds["static_object_names"] != ["table", "wall"]:
        raise AuditContractError("static-object registry changed")
    basis = value["threshold_basis"]
    if basis != {
        "thresholds_fit_from_pose_data": False,
        "world_aabb_basis": "conservative_envelope_from_base_task_static_wall_and_table_geometry",
        "max_step_translation_basis": "world_aabb_diagonal_not_empirical_quantile",
        "max_step_rotation_basis": "unit_quaternion_geodesic_upper_bound_pi",
        "static_object_basis": "base_task_create_table_and_wall_is_static_true",
        "note": "audit bounds are code-derived conservative integrity limits, not simulator hard limits",
    }:
        raise AuditContractError("threshold basis must be code-derived and data-independent")
    access = value["access_contract"]
    if access != {
        "fresh_inputs_allowed": False,
        "outcome_fields_used_for_thresholds_or_integrity_decisions": False,
        "gpu_allowed": False,
        "simulator_import_or_execution_allowed": False,
        "training_allowed": False,
    }:
        raise AuditContractError("audit access contract changed")
    sources = value["source_artifacts"]
    expected_sources = {
        "task_definition",
        "base_task_engine",
        "pose_discovery_collector",
        "schema5_branch_collector",
        "event_spec",
        "collection_manifest",
        "r3_materialization_manifest",
    }
    if not isinstance(sources, Mapping):
        raise AuditContractError("source_artifacts must be a mapping")
    _exact(sources, expected_sources, "source_artifacts")
    authenticated = {name: _artifact(sources[name], f"source_artifacts.{name}") for name in sorted(sources)}
    result = dict(value)
    result["source_artifacts"] = authenticated
    return result


def _decode(values: Any) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def quaternion_step_angles(quaternions: np.ndarray) -> np.ndarray:
    """Sign-invariant geodesic angles between adjacent wxyz quaternions."""

    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4 or q.shape[0] < 1:
        raise ValueError("quaternions must have shape [T,O,4]")
    if q.shape[0] == 1:
        return np.empty((0, q.shape[1]), dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1)
    denom = norms[:-1] * norms[1:]
    dots = np.sum(q[:-1] * q[1:], axis=-1)
    cosine = np.divide(np.abs(dots), denom, out=np.full_like(dots, np.nan), where=denom > 0)
    return 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))


def audit_pose_trajectory(
    poses: Any,
    *,
    object_names: Sequence[str],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure geometric audit for one candidate trajectory."""

    reason = InvalidReason(0)
    value = np.asarray(poses)
    if value.ndim != 3 or value.shape[1:] != (len(object_names), 7) or value.shape[0] < 1:
        return {
            "invalid_reason_bitset": int(InvalidReason.MISSING_REQUIRED_FIELD),
            "pose_samples": 0,
            "step_transitions": 0,
            "max_quaternion_norm_error": None,
            "max_step_translation_m": None,
            "max_step_rotation_rad": None,
        }
    if not np.isfinite(value).all():
        reason |= InvalidReason.NONFINITE_POSE
    finite = np.isfinite(value).all()
    quat = value[..., 3:7].astype(np.float64)
    norms = np.linalg.norm(quat, axis=-1)
    norm_error = float(np.nanmax(np.abs(norms - 1.0))) if norms.size else 0.0
    if not np.isfinite(norm_error) or norm_error > float(thresholds["quaternion_norm_abs_tolerance"]):
        reason |= InvalidReason.QUATERNION_NORM
    xyz = value[..., :3].astype(np.float64)
    aabb = np.asarray(thresholds["world_aabb_m"], dtype=np.float64)
    if finite and (np.any(xyz < aabb[:, 0]) or np.any(xyz > aabb[:, 1])):
        reason |= InvalidReason.WORLD_AABB
    translation = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    max_translation = float(np.nanmax(translation)) if translation.size else 0.0
    if not np.isfinite(max_translation) or max_translation > float(thresholds["max_step_translation_m"]):
        reason |= InvalidReason.STEP_TRANSLATION
    angles = quaternion_step_angles(quat)
    max_rotation = float(np.nanmax(angles)) if angles.size else 0.0
    if not np.isfinite(max_rotation) or max_rotation > float(thresholds["max_step_rotation_rad"]) + 1e-12:
        reason |= InvalidReason.STEP_ROTATION
    static_names = set(map(str, thresholds["static_object_names"]))
    for index, name in enumerate(object_names):
        if name not in static_names or value.shape[0] <= 1:
            continue
        if (
            float(np.nanmax(translation[:, index]))
            > float(thresholds["static_object_max_step_translation_m"])
            or float(np.nanmax(angles[:, index]))
            > float(thresholds["static_object_max_step_rotation_rad"])
        ):
            reason |= InvalidReason.STATIC_OBJECT_MOTION
    return {
        "invalid_reason_bitset": int(reason),
        "pose_samples": int(value.shape[0] * value.shape[1]),
        "step_transitions": int(max(0, value.shape[0] - 1) * value.shape[1]),
        "max_quaternion_norm_error": norm_error,
        "max_step_translation_m": max_translation,
        "max_step_rotation_rad": max_rotation,
    }


def _set_reason(bits: list[int], reason: InvalidReason, index: int | None = None) -> None:
    if index is None:
        for item in range(len(bits)):
            bits[item] |= int(reason)
    else:
        bits[index] |= int(reason)


def audit_schema5_group(
    path: Path,
    *,
    logical_key: str,
    materialized_record: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one authenticated HDF and its matching R3 holdout record."""

    candidate_count = int(spec["expected_candidates"])
    expected_names = list(map(str, spec["expected_object_names"]))
    materialized_names = list(map(str, spec["materialized_object_names"]))
    thresholds = spec["thresholds"]
    bits = [0] * candidate_count
    pose_samples = step_transitions = endpoint_checks = reset_checks = 0
    maxima = {"quat": 0.0, "translation": 0.0, "rotation": 0.0, "delta_error": 0.0}
    candidate_deltas: list[np.ndarray] = []
    candidate_row_candidates: list[int] = []
    continuation_deltas: list[np.ndarray] = []
    continuation_row_candidates: list[int] = []

    with h5py.File(path, "r") as handle:
        if (
            int(handle.attrs.get("schema_version", -1)) != 5
            or str(handle.attrs.get("task", "")) != spec["task"]
            or str(handle.attrs.get("body", "")) != spec["body"]
            or int(handle.attrs.get("candidate_count", -1)) != candidate_count
        ):
            _set_reason(bits, InvalidReason.METADATA_MISMATCH)
        required = {
            "object_names",
            "pre_object_poses",
            "post_object_poses",
            "first_chunk_executed_length",
            "steps",
            "candidate_names",
            "branches",
        }
        if any(name not in handle for name in required):
            _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD)
            return _group_result(
                logical_key,
                path,
                bits,
                pose_samples,
                step_transitions,
                endpoint_checks,
                reset_checks,
                0,
                maxima,
            )
        names = _decode(handle["object_names"][:])
        if names != expected_names:
            _set_reason(bits, InvalidReason.OBJECT_REGISTRY_MISMATCH)
        if any(name not in names for name in materialized_names):
            _set_reason(bits, InvalidReason.OBJECT_REGISTRY_MISMATCH)
            materialized_indices: list[int] = []
        else:
            materialized_indices = [names.index(name) for name in materialized_names]
        pre = np.asarray(handle["pre_object_poses"][:])
        post = np.asarray(handle["post_object_poses"][:])
        if pre.shape != (candidate_count, len(names), 7) or post.shape != pre.shape:
            _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD)
        else:
            if not np.isfinite(pre).all() or not np.isfinite(post).all():
                _set_reason(bits, InvalidReason.NONFINITE_POSE)
            reset_atol = float(thresholds["candidate_reset_match_atol"])
            for candidate in range(1, candidate_count):
                reset_checks += len(names)
                if not np.allclose(pre[candidate], pre[0], rtol=0.0, atol=reset_atol):
                    _set_reason(bits, InvalidReason.CANDIDATE_RESET_MISMATCH, candidate)
                    _set_reason(bits, InvalidReason.CANDIDATE_RESET_MISMATCH, 0)

        steps = np.asarray(handle["steps"][:], dtype=np.int64)
        executed = np.asarray(handle["first_chunk_executed_length"][:], dtype=np.int64)
        candidate_names = _decode(handle["candidate_names"][:])
        branches = handle["branches"]
        expected_materialized_candidate_names = list(candidate_names)
        continuation_index = 0
        for candidate in range(candidate_count):
            branch_name = f"candidate_{candidate:03d}"
            if branch_name not in branches:
                _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD, candidate)
                continue
            branch = branches[branch_name]
            if "object_poses" not in branch:
                _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD, candidate)
                continue
            trajectory = np.asarray(branch["object_poses"][:])
            audit = audit_pose_trajectory(
                trajectory, object_names=names, thresholds=thresholds
            )
            bits[candidate] |= int(audit["invalid_reason_bitset"])
            pose_samples += int(audit["pose_samples"])
            step_transitions += int(audit["step_transitions"])
            for key, source in (
                ("quat", "max_quaternion_norm_error"),
                ("translation", "max_step_translation_m"),
                ("rotation", "max_step_rotation_rad"),
            ):
                value = audit[source]
                if value is not None and np.isfinite(value):
                    maxima[key] = max(maxima[key], float(value))
            if (
                trajectory.shape != (int(steps[candidate]) + 1, len(names), 7)
                or int(executed[candidate]) < 0
                or int(executed[candidate]) >= trajectory.shape[0]
            ):
                _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD, candidate)
                continue
            endpoint_checks += 2 * len(names)
            endpoint_atol = float(thresholds["endpoint_match_atol"])
            if not np.allclose(trajectory[0], pre[candidate], rtol=0.0, atol=endpoint_atol) or not np.allclose(
                trajectory[int(executed[candidate])], post[candidate], rtol=0.0, atol=endpoint_atol
            ):
                _set_reason(bits, InvalidReason.ENDPOINT_MISMATCH, candidate)
            if materialized_indices:
                candidate_deltas.append(
                    (post[candidate, materialized_indices, :3] - pre[candidate, materialized_indices, :3]).reshape(-1)
                )
                candidate_row_candidates.append(candidate)
            query_required = {"query_steps", "query_post_steps"}
            if not query_required.issubset(branch):
                _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD, candidate)
                continue
            query_steps = np.asarray(branch["query_steps"][:], dtype=np.int64)
            query_post = np.asarray(branch["query_post_steps"][:], dtype=np.int64)
            if query_steps.shape != query_post.shape or len(query_steps) < 1:
                _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD, candidate)
                continue
            for query in range(1, len(query_steps)):
                start, stop = int(query_steps[query]), int(query_post[query])
                expected_materialized_candidate_names.append(f"continuation_{continuation_index}")
                continuation_index += 1
                if not (0 <= start < stop < trajectory.shape[0]):
                    _set_reason(bits, InvalidReason.MISSING_REQUIRED_FIELD, candidate)
                    continue
                if materialized_indices:
                    continuation_deltas.append(
                        (
                            trajectory[stop, materialized_indices, :3]
                            - trajectory[start, materialized_indices, :3]
                        ).reshape(-1)
                    )
                    continuation_row_candidates.append(candidate)

    batch = materialized_record.get("batch")
    if not isinstance(batch, Mapping):
        _set_reason(bits, InvalidReason.MATERIALIZED_LAYOUT_MISMATCH)
        materialized_rows = 0
    else:
        physical = materialized_record.get("object_delta_physical")
        batch_physical = batch.get("object_delta_physical")
        quality = batch.get("object_pose_quality_valid")
        quality_status = batch.get("object_pose_quality_status")
        if quality is not None or quality_status != QUALITY_UNAVAILABLE:
            _set_reason(bits, InvalidReason.MATERIALIZED_LAYOUT_MISMATCH)
        if not isinstance(physical, torch.Tensor) or not isinstance(batch_physical, torch.Tensor) or not torch.equal(
            physical, batch_physical
        ):
            _set_reason(bits, InvalidReason.MATERIALIZED_LAYOUT_MISMATCH)
            materialized_rows = 0
        else:
            observed = physical.detach().cpu().numpy()
            materialized_rows = len(observed)
            # collate_groups emits all terminal candidate rows first and only
            # then appends the flattened continuation rows.
            expected = np.asarray(candidate_deltas + continuation_deltas, dtype=np.float32)
            expected_row_candidates = candidate_row_candidates + continuation_row_candidates
            observed_names = list(map(str, batch.get("candidate_names", [])))
            if observed.shape != expected.shape or observed_names != expected_materialized_candidate_names:
                _set_reason(bits, InvalidReason.MATERIALIZED_LAYOUT_MISMATCH)
            elif expected.size:
                errors = np.max(np.abs(observed - expected), axis=1)
                maxima["delta_error"] = float(errors.max())
                tolerance = float(thresholds["materialized_delta_match_atol"])
                for row, error in enumerate(errors):
                    if float(error) > tolerance:
                        _set_reason(
                            bits,
                            InvalidReason.MATERIALIZED_DELTA_MISMATCH,
                            expected_row_candidates[row],
                        )
    return _group_result(
        logical_key,
        path,
        bits,
        pose_samples,
        step_transitions,
        endpoint_checks,
        reset_checks,
        materialized_rows,
        maxima,
    )


def _group_result(
    logical_key: str,
    path: Path,
    bits: list[int],
    pose_samples: int,
    step_transitions: int,
    endpoint_checks: int,
    reset_checks: int,
    materialized_rows: int,
    maxima: Mapping[str, float],
) -> dict[str, Any]:
    group_bits = 0
    for value in bits:
        group_bits |= int(value)
    return {
        "logical_group_key": logical_key,
        "source_path": str(path),
        "source_sha256": file_sha256(path),
        "candidate_invalid_reason_bitsets": bits,
        "group_invalid_reason_bitset": group_bits,
        "pose_samples": pose_samples,
        "step_transitions": step_transitions,
        "endpoint_checks": endpoint_checks,
        "candidate_reset_object_checks": reset_checks,
        "materialized_delta_rows": materialized_rows,
        "max_quaternion_norm_error": maxima["quat"],
        "max_step_translation_m": maxima["translation"],
        "max_step_rotation_rad": maxima["rotation"],
        "max_materialized_delta_abs_error": maxima["delta_error"],
    }


def _source_group_map(manifest: Mapping[str, Any], expected: int) -> dict[str, dict[str, Any]]:
    audit = manifest.get("source_collection_audit")
    if not isinstance(audit, Mapping):
        raise AuditContractError("R3 manifest lacks source_collection_audit")
    if audit.get("status") != "complete_schema5_signed_source_verified" or audit.get("task") != "move_can_pot" or audit.get("body") != "piper_piper_0.6":
        raise AuditContractError("R3 source collection audit is not authenticated D250")
    groups = audit.get("groups")
    if not isinstance(groups, list) or len(groups) != expected:
        raise AuditContractError("R3 source collection group cardinality differs")
    result: dict[str, dict[str, Any]] = {}
    for row in groups:
        if not isinstance(row, Mapping):
            raise AuditContractError("invalid R3 source group descriptor")
        key = str(row.get("logical_key", ""))
        path = safe_path(str(row.get("path", "")), f"source_group[{key}]")
        sha = str(row.get("sha256", ""))
        if not key or key in result or int(row.get("schema_version", -1)) != 5 or not _is_sha(sha):
            raise AuditContractError("invalid/duplicate R3 source group descriptor")
        if file_sha256(path) != sha:
            raise AuditContractError(f"source HDF SHA mismatch for {key}")
        result[key] = {"path": path, "sha256": sha}
    return result


def _load_holdout_records(manifest: Mapping[str, Any], expected: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise AuditContractError("R3 manifest must contain five folds")
    records: dict[str, dict[str, Any]] = {}
    fold_receipts: list[dict[str, Any]] = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise AuditContractError("invalid R3 fold descriptor")
        fold_id = int(fold.get("outer_fold_id", -1))
        path = safe_path(str(fold.get("holdout_artifact", "")), f"fold_{fold_id}_holdout")
        expected_sha = str(fold.get("holdout_artifact_sha256", ""))
        actual_sha = file_sha256(path)
        if not _is_sha(expected_sha) or expected_sha != actual_sha:
            raise AuditContractError(f"fold {fold_id} holdout artifact SHA mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("format") != EXPECTED_HOLDOUT_FORMAT or payload.get("schema_version") != 5:
            raise AuditContractError(f"fold {fold_id} holdout payload format changed")
        if payload.get("payload_sha256") != fold.get("holdout_payload_sha256"):
            raise AuditContractError(f"fold {fold_id} payload SHA receipt mismatch")
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("fresh_confirmation_data_or_labels_read") is not False:
            raise AuditContractError(f"fold {fold_id} is not development-only/Fresh-blind")
        holdout_keys = list(map(str, fold.get("oof_holdout_groups", [])))
        batches = payload.get("batches")
        if not isinstance(batches, list) or len(batches) != len(holdout_keys):
            raise AuditContractError(f"fold {fold_id} holdout record cardinality mismatch")
        seen: list[str] = []
        for record in batches:
            if not isinstance(record, Mapping):
                raise AuditContractError(f"fold {fold_id} has an invalid record")
            key = str(record.get("logical_group_key", ""))
            metadata = record.get("group_metadata")
            if (
                key in records
                or not isinstance(metadata, Mapping)
                or metadata.get("logical_group_key") != key
                or metadata.get("schema_version") != 5
                or metadata.get("task") != "move_can_pot"
                or metadata.get("body") != "piper_piper_0.6"
                or record.get("split_role") != "outer_holdout"
                or record.get("outer_fold_id") != fold_id
            ):
                raise AuditContractError(f"fold {fold_id} record identity mismatch")
            records[key] = dict(record)
            seen.append(key)
        if seen != holdout_keys:
            raise AuditContractError(f"fold {fold_id} record ordering/ownership mismatch")
        fold_receipts.append(
            {
                "outer_fold_id": fold_id,
                "holdout_artifact": str(path),
                "holdout_artifact_sha256": actual_sha,
                "holdout_payload_sha256": str(payload["payload_sha256"]),
                "groups": len(batches),
            }
        )
    if len(records) != expected:
        raise AuditContractError("authenticated holdouts do not cover every D250 group exactly once")
    return records, fold_receipts


def _reason_counts(groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in INVALID_REASON_NAMES.values()}
    for group in groups:
        for bit, name in INVALID_REASON_NAMES.items():
            if int(group["group_invalid_reason_bitset"]) & bit:
                counts[name] += 1
    return counts


def run_audit(spec_path: Path, r3_manifest_path: Path) -> dict[str, Any]:
    """Run the complete authenticated CPU audit."""

    spec_path = safe_path(spec_path, "spec")
    r3_manifest_path = safe_path(r3_manifest_path, "r3_manifest")
    raw_spec = _json(spec_path, "spec")
    spec = validate_spec(raw_spec)
    bound_r3 = spec["source_artifacts"]["r3_materialization_manifest"]
    if Path(bound_r3["path"]) != r3_manifest_path or bound_r3["sha256"] != file_sha256(r3_manifest_path):
        raise AuditContractError("CLI R3 manifest differs from task/engine spec binding")
    manifest = _json(r3_manifest_path, "r3_manifest")
    if (
        manifest.get("format") != EXPECTED_R3_FORMAT
        or manifest.get("status") != "complete_development_only"
        or manifest.get("fresh_confirmation_data_or_labels_read") is not False
        or manifest.get("prospective_claim_for_v8") is not False
        or len(manifest.get("development_groups", [])) != int(spec["expected_groups"])
    ):
        raise AuditContractError("R3 manifest is not authenticated adaptive-development D250")
    source_groups = _source_group_map(manifest, int(spec["expected_groups"]))
    records, fold_receipts = _load_holdout_records(manifest, int(spec["expected_groups"]))
    expected_keys = list(map(str, manifest["development_groups"]))
    if set(expected_keys) != set(source_groups) or set(expected_keys) != set(records):
        raise AuditContractError("R3/source/holdout group universes differ")
    group_results: list[dict[str, Any]] = []
    for key in expected_keys:
        source = source_groups[key]
        record = records[key]
        metadata = record["group_metadata"]
        if Path(str(metadata.get("source_path", ""))).resolve() != source["path"]:
            raise AuditContractError(f"materialized source path mismatch for {key}")
        result = audit_schema5_group(
            source["path"], logical_key=key, materialized_record=record, spec=spec
        )
        if result["source_sha256"] != source["sha256"]:
            raise AuditContractError(f"post-read source SHA mismatch for {key}")
        group_results.append(result)
    invalid_groups = sum(int(row["group_invalid_reason_bitset"]) != 0 for row in group_results)
    totals = {
        name: sum(int(row[name]) for row in group_results)
        for name in (
            "pose_samples",
            "step_transitions",
            "endpoint_checks",
            "candidate_reset_object_checks",
            "materialized_delta_rows",
        )
    }
    maxima = {
        name: max(float(row[name]) for row in group_results)
        for name in (
            "max_quaternion_norm_error",
            "max_step_translation_m",
            "max_step_rotation_rad",
            "max_materialized_delta_abs_error",
        )
    }
    status = "failed_closed" if invalid_groups else "passed_schema5_geometric_only"
    return {
        "format": FORMAT,
        "status": status,
        "scope": "adaptive_development_only_nonfresh_d250",
        "data_access": {
            "fresh_inputs_read": False,
            "development_payload_deserialized_including_unused_outcome_fields": True,
            "outcome_fields_consumed_by_audit_logic": False,
            "outcome_fields_used_to_choose_thresholds": False,
            "gpu_used": False,
            "simulator_imported_or_executed": False,
            "training_performed": False,
        },
        "spec": {
            "path": str(spec_path),
            "file_sha256": file_sha256(spec_path),
            "canonical_payload_sha256": canonical_sha256(raw_spec),
            "source_artifact_sha256": {
                name: artifact["sha256"] for name, artifact in spec["source_artifacts"].items()
            },
            "thresholds": spec["thresholds"],
            "threshold_basis": spec["threshold_basis"],
        },
        "r3_authentication": {
            "manifest": str(r3_manifest_path),
            "manifest_sha256": file_sha256(r3_manifest_path),
            "materialization_sha256": manifest.get("materialization_sha256"),
            "development_groups_sha256": manifest.get("development_groups_sha256"),
            "fold_receipts": fold_receipts,
            "source_group_hashes_verified": len(source_groups),
        },
        "coverage": {
            "groups_expected": int(spec["expected_groups"]),
            "groups_audited": len(group_results),
            "candidates_audited": len(group_results) * int(spec["expected_candidates"]),
            **totals,
            "source_pose_fields": {
                "object_names_and_order": True,
                "xyz": True,
                "quaternion_wxyz": True,
                "pre_post_endpoints": True,
                "per_step_trajectory": True,
                "signed_pose_quality_valid": False,
            },
            "r3_materialized_fields": {
                "simulator_recorded_can_xyz_delta": True,
                "orientation_delta": False,
                "learned_object_output": False,
                "pose_quality_valid": False,
            },
            "coverage_gap_bitset": 1 | 2 | 4,
            "coverage_gap_names": list(COVERAGE_GAPS.values()),
        },
        "integrity": {
            "invalid_groups": invalid_groups,
            "valid_groups": len(group_results) - invalid_groups,
            "invalid_reason_bits": {str(bit): name for bit, name in INVALID_REASON_NAMES.items()},
            "groups_by_invalid_reason": _reason_counts(group_results),
            "global_maxima": maxima,
        },
        "groups": group_results,
        "claim_ceiling": (
            "schema5 simulator-recorded full-pose geometric consistency and R3 can-xyz-delta "
            "reconstruction only; no independent pose-quality, learned-object-head, or transfer claim"
        ),
        "schema6_required_fields": [
            "object_registry_sha256_and_stable_sim_actor_id",
            "object_asset_model_id_and_role",
            "pose_frame_quaternion_order_and_units_in_hdf",
            "per_object_per_step_pose_quality_valid_and_reason_bitset",
            "simulator_timestamp_control_step_and_physics_substep_count",
            "reset_generation_and_teleport_flags",
            "embedded_pose_integrity_spec_sha256",
        ],
        "implementation_sha256": file_sha256(Path(__file__)),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--r3-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = safe_path(args.output, "output", must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    result = run_audit(args.spec, args.r3_manifest)
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "groups_audited": result["coverage"]["groups_audited"],
                "invalid_groups": result["integrity"]["invalid_groups"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
