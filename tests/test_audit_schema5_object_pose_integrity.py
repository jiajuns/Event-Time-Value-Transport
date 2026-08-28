from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_schema5_object_pose_integrity import (  # noqa: E402
    EXPECTED_HOLDOUT_FORMAT,
    InvalidReason,
    AuditContractError,
    QUALITY_UNAVAILABLE,
    SPEC_FORMAT,
    _load_holdout_records,
    audit_pose_trajectory,
    audit_schema5_group,
    file_sha256,
    quaternion_step_angles,
    safe_path,
    validate_spec,
)


OBJECTS = ["can", "cluttered_obj", "pot", "table", "wall"]


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _thresholds() -> dict[str, object]:
    aabb = [[-3.0, 3.0], [-1.0, 2.0], [-0.5, 3.0]]
    return {
        "world_aabb_m": aabb,
        "quaternion_norm_abs_tolerance": 1e-3,
        "max_step_translation_m": float(np.linalg.norm(np.ptp(np.asarray(aabb), axis=1))),
        "max_step_rotation_rad": math.pi,
        "endpoint_match_atol": 0.0,
        "candidate_reset_match_atol": 0.0,
        "materialized_delta_match_atol": 1e-5,
        "static_object_max_step_translation_m": 1e-6,
        "static_object_max_step_rotation_rad": 1e-6,
        "static_object_names": ["table", "wall"],
    }


@pytest.fixture()
def spec(tmp_path: Path) -> dict[str, object]:
    sources = {}
    for name in (
        "task_definition",
        "base_task_engine",
        "pose_discovery_collector",
        "schema5_branch_collector",
        "event_spec",
        "collection_manifest",
        "r3_materialization_manifest",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        sources[name] = _artifact(path)
    return {
        "format": SPEC_FORMAT,
        "timing_scope": "adaptive_development_audit_frozen_before_full_d250_pose_scan",
        "task": "move_can_pot",
        "body": "piper_piper_0.6",
        "schema_version": 5,
        "expected_groups": 250,
        "expected_candidates": 4,
        "expected_object_names": OBJECTS,
        "materialized_object_names": ["can"],
        "pose_layout": {
            "shape_suffix": [7],
            "translation_indices": [0, 1, 2],
            "quaternion_indices": [3, 4, 5, 6],
            "quaternion_order": "wxyz",
            "frame": "simulator_world",
            "translation_unit": "metre",
        },
        "thresholds": _thresholds(),
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "world_aabb_basis": "conservative_envelope_from_base_task_static_wall_and_table_geometry",
            "max_step_translation_basis": "world_aabb_diagonal_not_empirical_quantile",
            "max_step_rotation_basis": "unit_quaternion_geodesic_upper_bound_pi",
            "static_object_basis": "base_task_create_table_and_wall_is_static_true",
            "note": "audit bounds are code-derived conservative integrity limits, not simulator hard limits",
        },
        "source_artifacts": sources,
        "access_contract": {
            "fresh_inputs_allowed": False,
            "outcome_fields_used_for_thresholds_or_integrity_decisions": False,
            "gpu_allowed": False,
            "simulator_import_or_execution_allowed": False,
            "training_allowed": False,
        },
    }


def _pose() -> np.ndarray:
    value = np.zeros((5, 7), dtype=np.float32)
    value[:, 3] = 1.0
    value[:, :3] = np.array(
        [
            [-0.2, 0.1, 0.74],
            [0.2, 0.1, 0.74],
            [0.0, 0.1, 0.74],
            [0.0, 0.0, 0.74],
            [0.0, 1.0, 1.5],
        ],
        dtype=np.float32,
    )
    return value


def _group(path: Path, *, continuations: bool = False) -> tuple[np.ndarray, np.ndarray]:
    pre = np.repeat(_pose()[None], 4, axis=0)
    post = pre.copy()
    post[:, 0, 0] += np.arange(1, 5, dtype=np.float32) * 0.01
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 5
        handle.attrs["task"] = "move_can_pot"
        handle.attrs["body"] = "piper_piper_0.6"
        handle.attrs["candidate_count"] = 4
        handle.create_dataset("object_names", data=np.asarray(OBJECTS, dtype=object), dtype=string)
        handle.create_dataset("candidate_names", data=np.asarray(["deterministic", "c1", "c2", "c3"], dtype=object), dtype=string)
        handle.create_dataset("pre_object_poses", data=pre)
        handle.create_dataset("post_object_poses", data=post)
        handle.create_dataset(
            "steps",
            data=np.full(4, 2 if continuations else 1, dtype=np.int64),
        )
        handle.create_dataset("first_chunk_executed_length", data=np.ones(4, dtype=np.int64))
        branches = handle.create_group("branches")
        for candidate in range(4):
            branch = branches.create_group(f"candidate_{candidate:03d}")
            trajectory = [pre[candidate], post[candidate]]
            if continuations:
                final = post[candidate].copy()
                final[0, 1] += 0.005 * (candidate + 1)
                trajectory.append(final)
            branch.create_dataset("object_poses", data=np.stack(trajectory))
            branch.create_dataset(
                "query_steps",
                data=np.array([0, 1] if continuations else [0], dtype=np.int64),
            )
            branch.create_dataset(
                "query_post_steps",
                data=np.array([1, 2] if continuations else [1], dtype=np.int64),
            )
    return pre, post


def _record(pre: np.ndarray, post: np.ndarray) -> dict[str, object]:
    delta = torch.from_numpy((post[:, 0, :3] - pre[:, 0, :3]).astype(np.float32))
    batch = {
        "object_delta_physical": delta.clone(),
        "object_pose_quality_valid": None,
        "object_pose_quality_status": QUALITY_UNAVAILABLE,
        "candidate_names": ["deterministic", "c1", "c2", "c3"],
    }
    return {"batch": batch, "object_delta_physical": delta.clone()}


def _record_with_continuations(path: Path, pre: np.ndarray, post: np.ndarray) -> dict[str, object]:
    candidate_delta = post[:, 0, :3] - pre[:, 0, :3]
    continuation_delta = []
    with h5py.File(path, "r") as handle:
        for candidate in range(4):
            trajectory = handle[f"branches/candidate_{candidate:03d}/object_poses"][:]
            continuation_delta.append(trajectory[2, 0, :3] - trajectory[1, 0, :3])
    delta = torch.from_numpy(
        np.concatenate([candidate_delta, np.asarray(continuation_delta)], axis=0).astype(np.float32)
    )
    names = ["deterministic", "c1", "c2", "c3"] + [
        f"continuation_{index}" for index in range(4)
    ]
    batch = {
        "object_delta_physical": delta.clone(),
        "object_pose_quality_valid": None,
        "object_pose_quality_status": QUALITY_UNAVAILABLE,
        "candidate_names": names,
    }
    return {"batch": batch, "object_delta_physical": delta.clone()}


def test_spec_authenticates_code_derived_not_empirical_thresholds(spec: dict[str, object]) -> None:
    result = validate_spec(spec)
    assert result["threshold_basis"]["thresholds_fit_from_pose_data"] is False
    assert result["expected_object_names"] == OBJECTS


def test_spec_rejects_source_sha_and_empirical_threshold_tamper(spec: dict[str, object]) -> None:
    changed = copy.deepcopy(spec)
    changed["source_artifacts"]["task_definition"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(AuditContractError, match="SHA mismatch"):
        validate_spec(changed)
    changed = copy.deepcopy(spec)
    changed["threshold_basis"]["thresholds_fit_from_pose_data"] = True  # type: ignore[index]
    with pytest.raises(AuditContractError, match="data-independent"):
        validate_spec(changed)


def test_fresh_and_confirmation_paths_are_rejected(tmp_path: Path) -> None:
    for name in ("Fresh50", "confirmation_set"):
        path = tmp_path / name / "artifact.json"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(AuditContractError, match="Fresh/confirmation"):
            safe_path(path, "artifact")


def test_quaternion_angle_is_sign_invariant() -> None:
    q = np.array([[[1.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]]])
    assert quaternion_step_angles(q)[0, 0] == pytest.approx(0.0)


def test_pose_pure_function_reason_bits() -> None:
    thresholds = _thresholds()
    valid = np.stack([_pose(), _pose()])
    assert audit_pose_trajectory(valid, object_names=OBJECTS, thresholds=thresholds)["invalid_reason_bitset"] == 0

    bad = valid.copy()
    bad[1, 0, 0] = np.nan
    bits = audit_pose_trajectory(bad, object_names=OBJECTS, thresholds=thresholds)["invalid_reason_bitset"]
    assert bits & int(InvalidReason.NONFINITE_POSE)

    bad = valid.copy()
    bad[1, 0, 3] = 0.5
    bits = audit_pose_trajectory(bad, object_names=OBJECTS, thresholds=thresholds)["invalid_reason_bitset"]
    assert bits & int(InvalidReason.QUATERNION_NORM)

    bad = valid.copy()
    bad[1, 0, 0] = 4.0
    bits = audit_pose_trajectory(bad, object_names=OBJECTS, thresholds=thresholds)["invalid_reason_bitset"]
    assert bits & int(InvalidReason.WORLD_AABB)

    bad = valid.copy()
    bad[1, OBJECTS.index("table"), 0] += 0.01
    bits = audit_pose_trajectory(bad, object_names=OBJECTS, thresholds=thresholds)["invalid_reason_bitset"]
    assert bits & int(InvalidReason.STATIC_OBJECT_MOTION)


def test_group_audit_validates_names_endpoints_reset_and_materialized_delta(
    spec: dict[str, object], tmp_path: Path
) -> None:
    path = tmp_path / "group.hdf5"
    pre, post = _group(path)
    result = audit_schema5_group(
        path,
        logical_key="move_can_pot|piper_piper_0.6|1",
        materialized_record=_record(pre, post),
        spec=spec,
    )
    assert result["group_invalid_reason_bitset"] == 0
    assert result["pose_samples"] == 4 * 2 * 5
    assert result["endpoint_checks"] == 4 * 2 * 5
    assert result["candidate_reset_object_checks"] == 3 * 5
    assert result["materialized_delta_rows"] == 4
    assert result["max_materialized_delta_abs_error"] == pytest.approx(0.0)


def test_group_audit_matches_candidates_then_continuation_row_layout(
    spec: dict[str, object], tmp_path: Path
) -> None:
    path = tmp_path / "group_with_continuations.hdf5"
    pre, post = _group(path, continuations=True)
    result = audit_schema5_group(
        path,
        logical_key="move_can_pot|piper_piper_0.6|2",
        materialized_record=_record_with_continuations(path, pre, post),
        spec=spec,
    )
    assert result["group_invalid_reason_bitset"] == 0
    assert result["materialized_delta_rows"] == 8
    assert result["max_materialized_delta_abs_error"] == pytest.approx(0.0)


def test_group_audit_reports_endpoint_reset_and_delta_reason_bits(
    spec: dict[str, object], tmp_path: Path
) -> None:
    path = tmp_path / "group.hdf5"
    pre, post = _group(path)
    with h5py.File(path, "r+") as handle:
        handle["pre_object_poses"][1, 0, 0] += 0.02
        handle["branches/candidate_002/object_poses"][1, 0, 1] += 0.03
    record = _record(pre, post)
    record["object_delta_physical"][3, 2] += 0.1  # type: ignore[index]
    record["batch"]["object_delta_physical"] = record["object_delta_physical"].clone()  # type: ignore[index,union-attr]
    result = audit_schema5_group(
        path,
        logical_key="move_can_pot|piper_piper_0.6|1",
        materialized_record=record,
        spec=spec,
    )
    bits = result["group_invalid_reason_bitset"]
    assert bits & int(InvalidReason.CANDIDATE_RESET_MISMATCH)
    assert bits & int(InvalidReason.ENDPOINT_MISMATCH)
    assert bits & int(InvalidReason.MATERIALIZED_DELTA_MISMATCH)


def test_holdout_authentication_requires_exact_five_fold_ownership(tmp_path: Path) -> None:
    folds = []
    for fold_id in range(5):
        key = f"move_can_pot|piper_piper_0.6|{fold_id}"
        payload = {
            "format": EXPECTED_HOLDOUT_FORMAT,
            "schema_version": 5,
            "payload_sha256": f"{fold_id + 1:064x}",
            "provenance": {"fresh_confirmation_data_or_labels_read": False},
            "batches": [
                {
                    "logical_group_key": key,
                    "split_role": "outer_holdout",
                    "outer_fold_id": fold_id,
                    "group_metadata": {
                        "logical_group_key": key,
                        "schema_version": 5,
                        "task": "move_can_pot",
                        "body": "piper_piper_0.6",
                    },
                }
            ],
        }
        path = tmp_path / f"fold_{fold_id}.pt"
        torch.save(payload, path)
        folds.append(
            {
                "outer_fold_id": fold_id,
                "holdout_artifact": str(path.resolve()),
                "holdout_artifact_sha256": file_sha256(path),
                "holdout_payload_sha256": payload["payload_sha256"],
                "oof_holdout_groups": [key],
            }
        )
    records, receipts = _load_holdout_records({"folds": folds}, 5)
    assert len(records) == 5
    assert len(receipts) == 5
    folds[0]["holdout_artifact_sha256"] = "0" * 64
    with pytest.raises(AuditContractError, match="artifact SHA mismatch"):
        _load_holdout_records({"folds": folds}, 5)
