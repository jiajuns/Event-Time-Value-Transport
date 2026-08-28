from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from etsf_schema6_pose_quality import (  # noqa: E402
    GROUP_NAME,
    REGISTRY_FORMAT,
    SPEC_FORMAT,
    PoseQualityContractError,
    PoseQualityReason,
    derive_interval_supervision_mask,
    derive_pose_quality,
    load_object_delta_supervision_v6,
    registry_sha256,
    validate_pose_quality_v6,
    validate_registry,
    validate_spec,
    write_pose_quality_v6,
)


def registry() -> dict[str, object]:
    return {
        "format": REGISTRY_FORMAT,
        "objects": [
            {
                "name": "can",
                "stable_sim_actor_id": "scene/task-object/can/actor-0",
                "asset_model_id": "asset:can:v1",
                "role": "manipulated",
                "is_static": False,
            },
            {
                "name": "table",
                "stable_sim_actor_id": "scene/support/table/actor-0",
                "asset_model_id": "asset:table:v1",
                "role": "support",
                "is_static": True,
            },
        ],
    }


def spec(registry_value: dict[str, object] | None = None) -> dict[str, object]:
    registry_value = registry_value or registry()
    return {
        "format": SPEC_FORMAT,
        "schema_version": 6,
        "object_registry_sha256": registry_sha256(registry_value),
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
            "world_aabb_m": [[-2.0, 2.0], [-2.0, 2.0], [-0.1, 2.0]],
            "quaternion_norm_abs_tolerance": 1e-3,
            "max_step_translation_m": 0.2,
            "max_step_rotation_rad": math.pi / 2,
            "static_object_max_step_translation_m": 1e-6,
            "static_object_max_step_rotation_rad": 1e-6,
            "timestamp_step_min_s": 0.01,
            "timestamp_step_max_s": 0.2,
            "max_physics_substeps_per_control_step": 16,
        },
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "source": "task geometry, simulator control dt, and engine configuration",
            "frozen_before_collection": True,
        },
    }


def poses(steps: int = 4) -> np.ndarray:
    value = np.zeros((steps, 2, 7), dtype=np.float64)
    value[..., 3] = 1.0
    value[:, 0, 0] = np.arange(steps) * 0.01
    value[:, :, 2] = 0.7
    value[:, 1, 0] = 0.5
    return value


def telemetry(steps: int = 4) -> dict[str, np.ndarray]:
    return {
        "simulator_timestamp_s": np.arange(steps, dtype=np.float64) * 0.05,
        "control_step": np.arange(steps, dtype=np.uint64),
        "physics_substep_count": np.asarray([0] + [4] * (steps - 1), dtype=np.uint32),
        "reset_generation": np.zeros(steps, dtype=np.uint32),
        "reset_flag": np.asarray([True] + [False] * (steps - 1), dtype=bool),
        "teleport_flag": np.zeros((steps, 2), dtype=bool),
        "simulator_pose_error_flag": np.zeros((steps, 2), dtype=bool),
    }


def derive(value: np.ndarray | None = None, **changes: np.ndarray):
    fields = telemetry(len(value) if value is not None else 4)
    fields.update(changes)
    return derive_pose_quality(
        poses() if value is None else value,
        registry=registry(),
        spec=spec(),
        **fields,
    )


def write_file(path: Path) -> dict[str, object]:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 6
        branch = handle.create_group("branch")
        branch.create_dataset("object_poses", data=poses())
        receipt = write_pose_quality_v6(
            branch,
            registry=registry(),
            spec=spec(),
            **telemetry(),
        )
    return receipt


def test_registry_and_spec_are_content_bound_and_strict() -> None:
    value = registry()
    digest = registry_sha256(value)
    assert validate_registry(value)["objects"][0]["stable_sim_actor_id"]
    assert validate_spec(spec(value), expected_registry_sha256=digest)["schema_version"] == 6

    duplicate = copy.deepcopy(value)
    duplicate["objects"][1]["stable_sim_actor_id"] = duplicate["objects"][0][  # type: ignore[index]
        "stable_sim_actor_id"
    ]
    with pytest.raises(PoseQualityContractError, match="unique"):
        validate_registry(duplicate)

    string_boolean = copy.deepcopy(value)
    string_boolean["objects"][0]["is_static"] = "false"  # type: ignore[index]
    with pytest.raises(PoseQualityContractError, match="must be a boolean"):
        validate_registry(string_boolean)

    empirical = spec(value)
    empirical["threshold_basis"]["thresholds_fit_from_pose_data"] = True  # type: ignore[index]
    with pytest.raises(PoseQualityContractError, match="frozen before collection"):
        validate_spec(empirical)

    wrong_frame = spec(value)
    wrong_frame["pose_layout"]["frame"] = "camera"  # type: ignore[index]
    with pytest.raises(PoseQualityContractError, match="frame/quaternion/unit"):
        validate_spec(wrong_frame)


def test_valid_synthetic_trajectory_has_all_required_fields() -> None:
    batch = derive()
    assert batch.valid.all()
    assert not batch.reason_bitset.any()
    assert batch.reason_bitset.dtype == np.uint32
    assert batch.valid.shape == (4, 2)
    assert len(batch.registry_sha256) == len(batch.spec_sha256) == 64


def test_telemetry_types_are_not_silently_coerced() -> None:
    fields = telemetry()
    fields["control_step"] = fields["control_step"].astype(np.float64)
    with pytest.raises(PoseQualityContractError, match="integer array"):
        derive_pose_quality(poses(), registry=registry(), spec=spec(), **fields)
    fields = telemetry()
    fields["teleport_flag"] = fields["teleport_flag"].astype(np.uint8)
    with pytest.raises(PoseQualityContractError, match="boolean array"):
        derive_pose_quality(poses(), registry=registry(), spec=spec(), **fields)


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda pose, fields: pose.__setitem__((2, 0, 0), np.nan), PoseQualityReason.NONFINITE_POSE),
        (lambda pose, fields: pose.__setitem__((2, 0, 3), 0.5), PoseQualityReason.QUATERNION_NORM),
        (lambda pose, fields: pose.__setitem__((2, 0, 2), 3.0), PoseQualityReason.WORLD_AABB),
        (lambda pose, fields: pose.__setitem__((2, 0, 0), 0.5), PoseQualityReason.STEP_TRANSLATION),
        (lambda pose, fields: pose.__setitem__((2, 1, 0), 0.51), PoseQualityReason.STATIC_OBJECT_MOTION),
        (lambda pose, fields: fields["teleport_flag"].__setitem__((2, 0), True), PoseQualityReason.TELEPORT),
        (
            lambda pose, fields: fields["simulator_pose_error_flag"].__setitem__((2, 0), True),
            PoseQualityReason.SIMULATOR_REPORTED_INVALID,
        ),
        (
            lambda pose, fields: fields["simulator_timestamp_s"].__setitem__(2, fields["simulator_timestamp_s"][1]),
            PoseQualityReason.TIMESTAMP_NONMONOTONIC,
        ),
        (lambda pose, fields: fields["control_step"].__setitem__(2, 7), PoseQualityReason.CONTROL_STEP_INVALID),
        (
            lambda pose, fields: fields["physics_substep_count"].__setitem__(2, 0),
            PoseQualityReason.PHYSICS_SUBSTEP_INVALID,
        ),
        (lambda pose, fields: fields["reset_flag"].__setitem__(2, True), PoseQualityReason.RESET_FLAG_INCONSISTENT),
    ],
)
def test_synthetic_corruption_sets_explicit_reason(mutator, reason: PoseQualityReason) -> None:
    value = poses()
    fields = telemetry()
    mutator(value, fields)
    batch = derive_pose_quality(value, registry=registry(), spec=spec(), **fields)
    assert int(batch.reason_bitset[2].max()) & int(reason)
    assert not batch.valid[2].all()


def test_quaternion_rotation_is_sign_invariant_but_large_rotation_is_rejected() -> None:
    value = poses()
    value[1, 0, 3] = -1.0
    batch = derive(value)
    assert not int(batch.reason_bitset[1, 0]) & int(PoseQualityReason.STEP_ROTATION)

    value = poses()
    value[2, 0, 3:7] = [0.0, 1.0, 0.0, 0.0]
    batch = derive(value)
    assert int(batch.reason_bitset[2, 0]) & int(PoseQualityReason.STEP_ROTATION)


def test_reset_boundary_is_recorded_and_interval_mask_fails_closed() -> None:
    fields = telemetry()
    fields["simulator_timestamp_s"][2:] = [0.0, 0.05]
    fields["control_step"][2:] = [0, 1]
    fields["physics_substep_count"][2:] = [0, 4]
    fields["reset_generation"][2:] = 1
    fields["reset_flag"][2] = True
    batch = derive_pose_quality(poses(), registry=registry(), spec=spec(), **fields)
    assert int(batch.reason_bitset[2, 0]) & int(PoseQualityReason.RESET_DISCONTINUITY)
    assert batch.valid[3].all()
    mask, reason = derive_interval_supervision_mask(
        pose_quality_valid=batch.valid,
        pose_quality_reason_bitset=batch.reason_bitset,
        reset_generation=batch.reset_generation,
        reset_flag=batch.reset_flag,
        teleport_flag=batch.teleport_flag,
        start_steps=[0, 2],
        end_steps=[3, 3],
    )
    assert not mask[0].any()
    assert mask[1].all()
    assert np.all(reason[0] & int(PoseQualityReason.RESET_DISCONTINUITY))


def test_teleport_only_poisons_intervals_that_cross_it() -> None:
    fields = telemetry()
    fields["teleport_flag"][2, 0] = True
    batch = derive_pose_quality(poses(), registry=registry(), spec=spec(), **fields)
    mask, reason = derive_interval_supervision_mask(
        pose_quality_valid=batch.valid,
        pose_quality_reason_bitset=batch.reason_bitset,
        reset_generation=batch.reset_generation,
        reset_flag=batch.reset_flag,
        teleport_flag=batch.teleport_flag,
        start_steps=[0, 2],
        end_steps=[2, 3],
    )
    assert not mask[0, 0] and mask[0, 1]
    assert mask[1].all()
    assert int(reason[0, 0]) & int(PoseQualityReason.TELEPORT)


def test_hdf_roundtrip_and_training_loader(tmp_path: Path) -> None:
    path = tmp_path / "schema6.hdf5"
    receipt = write_file(path)
    with h5py.File(path, "r") as handle:
        branch = handle["branch"]
        audit = validate_pose_quality_v6(
            branch,
            expected_registry_sha256=receipt["object_registry_sha256"],
            expected_spec_sha256=receipt["pose_integrity_spec_sha256"],
        )
        targets = load_object_delta_supervision_v6(
            branch, start_steps=[0, 1], end_steps=[1, 3]
        )
        quality = branch[GROUP_NAME]
        assert quality.attrs["frame"] == "simulator_world"
        assert quality.attrs["quaternion_order"] == "wxyz"
        assert quality.attrs["translation_unit"] == "metre"
        assert quality.attrs["rotation_unit"] == "radian"
        assert set(
            [
                "simulator_timestamp_s",
                "control_step",
                "physics_substep_count",
                "reset_generation",
                "reset_flag",
                "teleport_flag",
                "pose_quality_valid",
                "pose_quality_reason_bitset",
            ]
        ).issubset(quality.keys())
    assert audit["invalid_samples"] == 0
    assert targets["object_delta_xyz_m"].shape == (2, 2, 3)
    assert targets["object_delta_supervision_valid"].all()


@pytest.mark.parametrize("tamper", ["quality", "pose", "frame", "registry", "payload_sha"])
def test_hdf_tamper_is_rejected(tmp_path: Path, tamper: str) -> None:
    path = tmp_path / f"tamper_{tamper}.hdf5"
    write_file(path)
    with h5py.File(path, "r+") as handle:
        branch = handle["branch"]
        quality = branch[GROUP_NAME]
        if tamper == "quality":
            quality["pose_quality_valid"][1, 0] = False
        elif tamper == "pose":
            branch["object_poses"][1, 0, 0] += 0.001
        elif tamper == "frame":
            quality.attrs["frame"] = "camera"
        elif tamper == "registry":
            value = quality["object_registry_json"][()].decode("utf-8")
            changed = json.loads(value)
            changed["objects"][0]["asset_model_id"] = "asset:other:v1"
            quality["object_registry_json"][()] = json.dumps(changed, sort_keys=True, separators=(",", ":"))
        else:
            quality.attrs["logical_payload_sha256"] = "0" * 64
    with h5py.File(path, "r") as handle:
        with pytest.raises(PoseQualityContractError):
            validate_pose_quality_v6(handle["branch"])


def test_schema5_without_quality_never_becomes_implicitly_valid(tmp_path: Path) -> None:
    path = tmp_path / "schema5.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 5
        branch = handle.create_group("branch")
        branch.create_dataset("object_poses", data=poses())
    with h5py.File(path, "r") as handle:
        with pytest.raises(PoseQualityContractError, match="unlabelled poses fail closed"):
            validate_pose_quality_v6(handle["branch"])
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["schema_version"]) == 5
        assert GROUP_NAME not in handle["branch"]
    with h5py.File(path, "r+") as handle:
        with pytest.raises(PoseQualityContractError, match="must not be upgraded in place"):
            write_pose_quality_v6(
                handle["branch"], registry=registry(), spec=spec(), **telemetry()
            )


def test_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "schema6.hdf5"
    write_file(path)
    with h5py.File(path, "r+") as handle:
        with pytest.raises(PoseQualityContractError, match="refusing to overwrite"):
            write_pose_quality_v6(
                handle["branch"], registry=registry(), spec=spec(), **telemetry()
            )
