from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_smolvla_piper_causal_event_observer_dataset_v1 as bridge  # noqa: E402
from etsf_schema6_pose_quality import (  # noqa: E402
    registry_sha256,
    spec_sha256,
    write_pose_quality_v6,
)


TASK = "move_can_pot"
SCHEMA5_BODY = "aloha-agilex"
SCHEMA6_BODY = "piper-agilex"
POLICY = "smolvla"
SCHEMA5_SOURCE_SHA = "a" * 64
SCHEMA6_SOURCE_SHA = "b" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _event_spec(path: Path) -> str:
    value = {
        "calibration": {
            TASK: {
                "moving": "can",
                "anchor": "pot",
                "offset": [0.0, 0.0, 0.0],
                "delta_move": 0.02,
                "delta_z": 0.05,
                "tau_d": 0.06,
                "tau_motion": 0.001,
                "stationary_steps": 2,
            }
        },
        "chains": {TASK: ["e0", "e12", "e3", "e4", "eK"]},
    }
    _write_json(path, value)
    return bridge.file_sha256(path)


def _poses(steps: int, *, near_goal: bool = False) -> np.ndarray:
    values = np.zeros((steps + 1, 2, 7), dtype=np.float64)
    values[..., 3] = 1.0
    values[:, 1, 0] = 0.5
    if near_goal:
        values[:, 0, 0] = np.linspace(0.49, 0.5, steps + 1)
    else:
        values[:, 0, 0] = np.linspace(0.0, 0.5, steps + 1)
    return values


def _write_schema5_group(
    path: Path,
    *,
    requested_seed: int,
    resolved_seed: int,
    event_spec_sha: str,
    query_count: int,
    success: bool,
    future_offset: float = 0.0,
) -> None:
    terminal = query_count
    hidden = np.repeat(
        np.arange(query_count, dtype=np.float32)[:, None], bridge.STATE_DIM, axis=1
    )
    if future_offset:
        hidden[-1] += np.float32(future_offset)
    post_hidden = np.repeat(
        np.arange(1, query_count + 1, dtype=np.float32)[:, None],
        bridge.STATE_DIM,
        axis=1,
    )
    if future_offset and query_count > 1:
        # Preserve the collector continuation invariant: the post state from
        # q-1 is the actor-visible current state at q.
        post_hidden[-2] += np.float32(future_offset)
    actions = np.zeros((query_count, 2, bridge.PROPRIO_DIM), dtype=np.float32)
    for query in range(query_count):
        actions[query, 0] = query + 0.25
    masks = np.zeros((query_count, 2), dtype=np.bool_)
    masks[:, 0] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 5,
                "task": TASK,
                "body": SCHEMA5_BODY,
                "policy": POLICY,
                "requested_seed": requested_seed,
                "resolved_seed": resolved_seed,
                "event_spec_sha256": event_spec_sha,
                "shared_state_contract_id": SCHEMA5_SOURCE_SHA,
                "candidate_hidden_forbidden": True,
            }
        )
        strings = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "candidate_names",
            data=np.asarray(["deterministic"], dtype=object),
            dtype=strings,
        )
        handle.create_dataset(
            "object_names", data=np.asarray(["can", "pot"], dtype=object), dtype=strings
        )
        handle.create_dataset("success", data=np.asarray([success], dtype=np.bool_))
        handle.create_dataset("steps", data=np.asarray([terminal], dtype=np.int32))
        branches = handle.create_group("branches")
        branch = branches.create_group("candidate_000")
        branch.create_dataset("query_hidden", data=hidden.astype(np.float16))
        branch.create_dataset("query_post_hidden", data=post_hidden.astype(np.float16))
        branch.create_dataset("query_steps", data=np.arange(query_count, dtype=np.int32))
        branch.create_dataset(
            "query_post_steps", data=np.arange(1, query_count + 1, dtype=np.int32)
        )
        branch.create_dataset("query_actions", data=actions)
        branch.create_dataset("query_action_mask", data=masks)
        branch.create_dataset(
            "proprio",
            data=np.repeat(
                np.arange(terminal + 1, dtype=np.float32)[:, None],
                bridge.PROPRIO_DIM,
                axis=1,
            ),
        )
        branch.create_dataset("object_poses", data=_poses(terminal))


def _registry(index: int) -> dict[str, Any]:
    return {
        "format": "etsf_schema6_object_registry_v1",
        "objects": [
            {
                "name": "can",
                "stable_sim_actor_id": f"can-{index}",
                "asset_model_id": "can/base0",
                "role": "manipulated",
                "is_static": False,
            },
            {
                "name": "pot",
                "stable_sim_actor_id": f"pot-{index}",
                "asset_model_id": "pot/base0",
                "role": "receptacle",
                "is_static": False,
            },
        ],
    }


def _pose_spec(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "etsf_schema6_pose_quality_spec_v1",
        "schema_version": 6,
        "object_registry_sha256": registry_sha256(registry),
        "pose_layout": {
            "shape_suffix": [7],
            "translation_indices": [0, 1, 2],
            "quaternion_indices": [3, 4, 5, 6],
            "quaternion_order": "wxyz",
            "frame": "simulator_world",
            "translation_unit": "metre",
            "rotation_unit": "radian",
        },
        "time_layout": {
            "timestamp_unit": "second",
            "timestamp_clock": "simulator_monotonic",
            "control_step_semantics": "sample_after_completed_control_step",
            "physics_substep_semantics": "substeps_since_previous_sample_zero_at_reset",
        },
        "thresholds": {
            "world_aabb_m": [[-3.0, 3.0], [-1.0, 2.0], [-0.5, 3.0]],
            "quaternion_norm_abs_tolerance": 1e-3,
            "max_step_translation_m": 7.5,
            "max_step_rotation_rad": math.pi,
            "static_object_max_step_translation_m": 1e-6,
            "static_object_max_step_rotation_rad": 1e-6,
            "timestamp_step_min_s": 0.004,
            "timestamp_step_max_s": 4.0,
            "max_physics_substeps_per_control_step": 1000,
        },
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "source": "synthetic geometry only",
            "frozen_before_collection": True,
        },
    }


def _write_schema6_group(path: Path, *, index: int, invalid_pose: bool = False) -> None:
    terminal = 2
    query_count = 2
    registry = _registry(index)
    spec = _pose_spec(registry)
    poses = _poses(terminal)
    if invalid_pose:
        poses[1, 0, 0] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 6,
                "format": "etsf_smolvla_piper_dense_event_branches_schema6_v1",
                "object_registry_sha256": registry_sha256(registry),
                "pose_integrity_spec_sha256": spec_sha256(
                    spec, expected_registry_sha256=registry_sha256(registry)
                ),
            }
        )
        root = handle.create_group("root")
        root.create_dataset("hidden", data=np.zeros(bridge.STATE_DIM, dtype=np.float32))
        root.create_dataset(
            "processed_state", data=np.zeros(bridge.PROPRIO_DIM, dtype=np.float32)
        )
        root.create_dataset(
            "eligible_original_candidate_indices", data=np.asarray([0], dtype=np.int16)
        )
        branches = handle.create_group("branches")
        branch = branches.create_group("branch_000")
        branch.attrs["steps"] = terminal
        branch.attrs["success_diagnostic_only"] = False
        branch.attrs["original_candidate_index"] = 0
        branch.create_dataset(
            "query_hidden",
            data=np.repeat(
                np.arange(query_count, dtype=np.float32)[:, None],
                bridge.STATE_DIM,
                axis=1,
            ),
        )
        branch.create_dataset(
            "query_processed_state",
            data=np.repeat(
                np.arange(query_count, dtype=np.float32)[:, None],
                bridge.PROPRIO_DIM,
                axis=1,
            ),
        )
        branch.create_dataset(
            "query_selected_original_candidate_index",
            data=np.asarray([0, 0], dtype=np.int16),
        )
        branch.create_dataset(
            "query_executed_action",
            data=np.ones((query_count, bridge.PROPRIO_DIM), dtype=np.float32),
        )
        masks = np.zeros((query_count, 50), dtype=np.bool_)
        masks[:, 0] = True
        branch.create_dataset("query_executed_action_mask", data=masks)
        branch.create_dataset("object_poses", data=poses)
        branch.create_dataset(
            "proprio", data=np.zeros((terminal + 1, bridge.PROPRIO_DIM), dtype=np.float32)
        )
        write_pose_quality_v6(
            branch,
            registry=registry,
            spec=spec,
            simulator_timestamp_s=np.asarray([0.0, 0.1, 0.2]),
            control_step=np.asarray([0, 1, 2], dtype=np.int64),
            physics_substep_count=np.asarray([0, 1, 1], dtype=np.int64),
            reset_generation=np.asarray([0, 0, 0], dtype=np.int64),
            reset_flag=np.asarray([True, False, False], dtype=np.bool_),
            teleport_flag=np.zeros((terminal + 1, 2), dtype=np.bool_),
            simulator_pose_error_flag=np.zeros((terminal + 1, 2), dtype=np.bool_),
        )


def _schema5_manifest(
    path: Path,
    group_root: Path,
    event_spec_sha: str,
    group_count: int,
    *,
    future_offset: float = 0.0,
) -> tuple[dict[str, Any], list[Path]]:
    groups: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index in range(group_count):
        requested = 1000 + index
        resolved = 2000 + index
        group_path = group_root / f"source_{index:03d}.hdf5"
        _write_schema5_group(
            group_path,
            requested_seed=requested,
            resolved_seed=resolved,
            event_spec_sha=event_spec_sha,
            query_count=10 if index == 0 else 2,
            success=index == 0,
            future_offset=future_offset if index == 0 else 0.0,
        )
        paths.append(group_path)
        groups.append(
            {
                "index": index,
                "path": group_path.name,
                "status": "collected",
                "seed": requested,
                "resolved_seed": resolved,
            }
        )
    value: dict[str, Any] = {
        "status": "complete",
        "schema_version": 5,
        "task": TASK,
        "body": SCHEMA5_BODY,
        "policy": POLICY,
        "hidden_dim": bridge.STATE_DIM,
        "action_dim": bridge.PROPRIO_DIM,
        "event_vocab": list(bridge.EVENT_NAMES),
        "event_spec_sha256": event_spec_sha,
        "shared_state_contract": {"calibration_id": SCHEMA5_SOURCE_SHA},
        "groups": groups,
    }
    _write_json(path, value)
    return value, paths


def _schema6_manifest(
    path: Path,
    group_root: Path,
    event_spec_sha: str,
    group_count: int,
    *,
    invalid_group: int | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    groups: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index in range(group_count):
        group_path = group_root / f"target_{index:03d}.hdf5"
        _write_schema6_group(group_path, index=index, invalid_pose=index == invalid_group)
        paths.append(group_path)
        groups.append(
            {
                "logical_group_id": f"target/{index:03d}",
                "requested_seed": 3000 + index,
                "resolved_seed": 4000 + index,
                "task": TASK,
                "body": SCHEMA6_BODY,
                "policy": POLICY,
                "path": group_path.name,
                "file_sha256": bridge.file_sha256(group_path),
            }
        )
    value: dict[str, Any] = {
        "format": "etsf_smolvla_piper_schema6_training_manifest_v1",
        "status": "complete",
        "groups": groups,
        "fresh_inputs_used": False,
        "sealed_test_labels_disclosed": False,
        "event_spec_sha256": event_spec_sha,
    }
    value["manifest_sha256"] = bridge.canonical_sha256(value)
    _write_json(path, value)
    return value, paths


def _write_request(
    path: Path,
    *,
    event_path: Path,
    source5_manifest_path: Path,
    source5_manifest: dict[str, Any],
    source5_root: Path,
    source5_paths: list[Path],
    source6_manifest_path: Path,
    source6_manifest: dict[str, Any],
    source6_root: Path,
    source6_paths: list[Path],
    overlap: bool = False,
) -> dict[str, Any]:
    source5_ids = [
        bridge.schema5_logical_group_id(
            "source63", TASK, SCHEMA5_BODY, POLICY, 1000 + index
        )
        for index in range(3)
    ]
    source6_ids = [
        bridge.schema6_logical_group_id("piper", f"target/{index:03d}")
        for index in range(3)
    ]
    splits: dict[str, list[dict[str, str]]] = {}
    for index, split in enumerate(bridge.SPLIT_NAMES):
        source_index = 0 if overlap and split == "calibration" else index
        splits[split] = [
            {
                "source_name": "source63",
                "logical_group_id": source5_ids[source_index],
                "source_file_sha256": bridge.file_sha256(source5_paths[source_index]),
            },
            {
                "source_name": "piper",
                "logical_group_id": source6_ids[index],
                "source_file_sha256": bridge.file_sha256(source6_paths[index]),
            },
        ]
    value: dict[str, Any] = {
        "format": bridge.REQUEST_FORMAT,
        "status": bridge.REQUEST_STATUS,
        "event_spec": {
            "path": str(event_path),
            "file_sha256": bridge.file_sha256(event_path),
        },
        "actors": [
            {
                "actor_name": "smolvla_aloha",
                "policy_family": POLICY,
                "body": SCHEMA5_BODY,
                "policy": POLICY,
                "state_feature_source_sha256": SCHEMA5_SOURCE_SHA,
            },
            {
                "actor_name": "smolvla_piper",
                "policy_family": POLICY,
                "body": SCHEMA6_BODY,
                "policy": POLICY,
                "state_feature_source_sha256": SCHEMA6_SOURCE_SHA,
            },
        ],
        "sources": [
            {
                "source_name": "source63",
                "schema_version": 5,
                "manifest_path": str(source5_manifest_path),
                "manifest_file_sha256": bridge.file_sha256(source5_manifest_path),
                "manifest_logical_sha256": bridge.canonical_sha256(source5_manifest),
                "group_root": str(source5_root),
                "actor_name": "smolvla_aloha",
            },
            {
                "source_name": "piper",
                "schema_version": 6,
                "manifest_path": str(source6_manifest_path),
                "manifest_file_sha256": bridge.file_sha256(source6_manifest_path),
                "manifest_logical_sha256": source6_manifest["manifest_sha256"],
                "group_root": str(source6_root),
                "actor_name": "smolvla_piper",
            },
        ],
        "splits": splits,
        "split_unit": "logical_reset_group",
        "split_leakage_allowed": False,
        "privileged_label_source_available_to_model_inputs": False,
        "future_query_features_available_to_model_inputs": False,
    }
    value["request_sha256"] = bridge.canonical_sha256(value)
    _write_json(path, value)
    return value


def make_minimal_materialized_dataset_fixture(
    tmp_path: Path,
    *,
    invalid_schema6_group: int | None = None,
    schema5_future_offset: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    """Reusable schema5+schema6 synthetic fixture for trainer E2E tests."""

    event_path = tmp_path / "event_spec.json"
    event_sha = _event_spec(event_path)
    source5_root = tmp_path / "source5_groups"
    source6_root = tmp_path / "source6_groups"
    source5_manifest_path = tmp_path / "source5_manifest.json"
    source6_manifest_path = tmp_path / "source6_manifest.json"
    source5_manifest, source5_paths = _schema5_manifest(
        source5_manifest_path,
        source5_root,
        event_sha,
        3,
        future_offset=schema5_future_offset,
    )
    source6_manifest, source6_paths = _schema6_manifest(
        source6_manifest_path,
        source6_root,
        event_sha,
        3,
        invalid_group=invalid_schema6_group,
    )
    request_path = tmp_path / "request.json"
    _write_request(
        request_path,
        event_path=event_path,
        source5_manifest_path=source5_manifest_path,
        source5_manifest=source5_manifest,
        source5_root=source5_root,
        source5_paths=source5_paths,
        source6_manifest_path=source6_manifest_path,
        source6_manifest=source6_manifest,
        source6_root=source6_root,
        source6_paths=source6_paths,
    )
    output = tmp_path / "dataset"
    manifest = bridge.materialize(request_path, output)
    return output / "manifest.json", manifest


def test_materializes_multiactor_causal_dataset_and_validates(tmp_path: Path) -> None:
    manifest_path, manifest = make_minimal_materialized_dataset_fixture(tmp_path)
    assert bridge.validate_dataset_manifest(manifest_path, verify_npz=True) == manifest
    assert manifest["split_group_disjoint"] is True
    assert [row["actor_index"] for row in manifest["actor_registry"]] == [0, 1]
    split_groups = [set(manifest["splits"][name]["logical_group_ids"]) for name in bridge.SPLIT_NAMES]
    assert not (split_groups[0] & split_groups[1] or split_groups[0] & split_groups[2] or split_groups[1] & split_groups[2])
    train = bridge.load_split(manifest_path, "train")
    assert train["history"].shape[1:] == (8, 960)
    assert train["proprio"].shape[1:] == (14,)
    assert train["predicate_label"].shape[1:] == (5,)
    assert set(train["actor_index"].tolist()) == {0, 1}
    assert "object_poses" not in train
    assert all(
        token not in name
        for name in train
        for token in ("future", "outcome", "success_or_terminal")
    )


def test_history_is_right_padded_left_truncated_and_same_branch(tmp_path: Path) -> None:
    manifest_path, _ = make_minimal_materialized_dataset_fixture(tmp_path)
    train = bridge.load_split(manifest_path, "train")
    source_rows = np.flatnonzero(train["actor_index"] == 0)
    q0 = source_rows[train["current_query_index"][source_rows] == 0][0]
    q9 = source_rows[train["current_query_index"][source_rows] == 9][0]
    q10 = source_rows[train["current_query_index"][source_rows] == 10][0]
    assert train["history_mask"][q0].tolist() == [True] + [False] * 7
    assert np.all(train["history"][q0, 1:] == 0)
    assert train["history_mask"][q9].all()
    assert train["history"][q9, :, 0].tolist() == list(map(float, range(2, 10)))
    assert train["history"][q10, :, 0].tolist() == list(map(float, range(3, 11)))


def test_execution_receipt_is_absent_only_at_root_and_binds_action(tmp_path: Path) -> None:
    manifest_path, _ = make_minimal_materialized_dataset_fixture(tmp_path)
    for split in bridge.SPLIT_NAMES:
        arrays = bridge.load_split(manifest_path, split)
        for index, query in enumerate(arrays["current_query_index"]):
            if query == 0:
                assert not arrays["prior_execution_present"][index]
                assert arrays["prior_executed_control_steps"][index] == 0
                assert arrays["prior_action_sha256"][index] == ""
            else:
                assert arrays["prior_execution_present"][index]
                assert arrays["prior_executed_control_steps"][index] > 0
                assert len(arrays["prior_action_sha256"][index]) == 64


def test_schema5_terminal_post_state_supplies_terminal_success_label(tmp_path: Path) -> None:
    manifest_path, _ = make_minimal_materialized_dataset_fixture(tmp_path)
    train = bridge.load_split(manifest_path, "train")
    terminal = (train["actor_index"] == 0) & (train["current_query_index"] == 10)
    assert terminal.sum() == 1
    row = int(np.flatnonzero(terminal)[0])
    assert train["event_label"][row] == bridge.EVENT_NAMES.index("eK")
    assert train["predicate_label"][row, -1] == 1.0
    earlier = (train["actor_index"] == 0) & (train["current_query_index"] < 10)
    assert not train["predicate_label"][earlier, -1].any()


def test_future_query_hidden_cannot_change_earlier_samples(tmp_path: Path) -> None:
    first_path, _ = make_minimal_materialized_dataset_fixture(tmp_path / "first")
    second_path, _ = make_minimal_materialized_dataset_fixture(
        tmp_path / "second", schema5_future_offset=100.0
    )
    first = bridge.load_split(first_path, "train")
    second = bridge.load_split(second_path, "train")
    first_rows = np.flatnonzero(
        (first["actor_index"] == 0) & (first["current_query_index"] < 9)
    )
    second_rows = np.flatnonzero(
        (second["actor_index"] == 0) & (second["current_query_index"] < 9)
    )
    assert np.array_equal(first["current_query_index"][first_rows], second["current_query_index"][second_rows])
    for name in ("history", "history_mask", "proprio", "event_label", "predicate_label"):
        assert np.array_equal(first[name][first_rows], second[name][second_rows])
    # The changed q=9 state itself must remain observable at q=9, proving that
    # the comparison did not accidentally ignore the fixture intervention.
    first_q9 = np.flatnonzero(
        (first["actor_index"] == 0) & (first["current_query_index"] == 9)
    )[0]
    second_q9 = np.flatnonzero(
        (second["actor_index"] == 0) & (second["current_query_index"] == 9)
    )[0]
    assert not np.array_equal(first["history"][first_q9], second["history"][second_q9])


def test_overlap_fails_before_any_hdf_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event_path = tmp_path / "event.json"
    event_sha = _event_spec(event_path)
    source5_root = tmp_path / "s5"
    source6_root = tmp_path / "s6"
    m5_path, m6_path = tmp_path / "m5.json", tmp_path / "m6.json"
    m5, p5 = _schema5_manifest(m5_path, source5_root, event_sha, 3)
    m6, p6 = _schema6_manifest(m6_path, source6_root, event_sha, 3)
    request = tmp_path / "request.json"
    _write_request(
        request,
        event_path=event_path,
        source5_manifest_path=m5_path,
        source5_manifest=m5,
        source5_root=source5_root,
        source5_paths=p5,
        source6_manifest_path=m6_path,
        source6_manifest=m6,
        source6_root=source6_root,
        source6_paths=p6,
        overlap=True,
    )

    def forbidden_hdf_open(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("HDF opened before split membership passed")

    monkeypatch.setattr(bridge.h5py, "File", forbidden_hdf_open)
    with pytest.raises(bridge.ObserverDatasetContractError, match="duplicated"):
        bridge.materialize(request, tmp_path / "out")


def test_group_file_sha_mismatch_fails_before_hdf_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = make_minimal_materialized_dataset_fixture(tmp_path / "valid")
    request_path = tmp_path / "valid" / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["splits"]["train"][0]["source_file_sha256"] = "f" * 64
    unsigned = dict(request)
    unsigned.pop("request_sha256")
    request["request_sha256"] = bridge.canonical_sha256(unsigned)
    bad = tmp_path / "bad_request.json"
    _write_json(bad, request)

    def forbidden_hdf_open(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("HDF opened before source SHA passed")

    monkeypatch.setattr(bridge.h5py, "File", forbidden_hdf_open)
    with pytest.raises(bridge.ObserverDatasetContractError, match="SHA changed"):
        bridge.materialize(bad, tmp_path / "bad_out")
    assert manifest_path.exists()


def test_manifest_or_npz_tamper_fails_closed(tmp_path: Path) -> None:
    manifest_path, manifest = make_minimal_materialized_dataset_fixture(tmp_path)
    manifest_path.chmod(0o644)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["splits"]["train"]["row_count"] += 1
    _write_json(manifest_path, value)
    with pytest.raises(bridge.ObserverDatasetContractError):
        bridge.validate_dataset_manifest(manifest_path, verify_npz=True)
    # Restore the exact file, then mutate NPZ bytes without changing manifest.
    _write_json(manifest_path, manifest)
    npz = manifest_path.parent / "train.npz"
    npz.chmod(0o644)
    with npz.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(bridge.ObserverDatasetContractError, match="file SHA"):
        bridge.validate_dataset_manifest(manifest_path, verify_npz=True)


def test_invalid_schema6_current_pose_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        make_minimal_materialized_dataset_fixture(tmp_path, invalid_schema6_group=0)


def test_array_sha_binds_dtype_shape_and_bytes() -> None:
    first = np.asarray([[1.0, 2.0]], dtype=np.float32)
    assert bridge.array_sha256(first) != bridge.array_sha256(first.astype(np.float64))
    assert bridge.array_sha256(first) != bridge.array_sha256(first.reshape(2, 1))
    changed = first.copy()
    changed[0, 1] = 3.0
    assert bridge.array_sha256(first) != bridge.array_sha256(changed)
