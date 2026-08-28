#!/usr/bin/env python3
"""Schema-v6 object-pose quality contract and HDF5 helpers.

This module is deliberately independent from the signed schema-v5 collectors.
Collectors may opt into it when producing *new* schema-v6 artifacts.  It does
not upgrade schema-v5 files in place and it never infers that an unlabelled
schema-v5 pose is valid.

The contract separates two questions:

* ``pose_quality_valid[t, object]`` says whether the pose at step ``t`` and the
  transition ending at that step passed the frozen integrity checks.
* ``derive_interval_supervision_mask`` says whether an endpoint delta spanning
  several steps is safe as a training label.  Every intervening destination
  step must be valid and no reset or teleport may be crossed.

All hashes are logical SHA256 values over canonical JSON/array bytes, not HDF5
file hashes.  The reader recomputes the labels and payload hash, so editing a
quality bit, time field, registry, specification, or pose fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


FORMAT = "etsf_schema6_pose_quality_v1"
SPEC_FORMAT = "etsf_schema6_pose_quality_spec_v1"
REGISTRY_FORMAT = "etsf_schema6_object_registry_v1"
GROUP_NAME = "pose_quality_v6"
SCHEMA_VERSION = 6


class PoseQualityContractError(ValueError):
    """Raised when schema-v6 metadata or data fails closed."""


class PoseQualityReason(IntFlag):
    """Per-object, per-step reasons that make a destination label unusable."""

    NONFINITE_POSE = 1 << 0
    QUATERNION_NORM = 1 << 1
    WORLD_AABB = 1 << 2
    STEP_TRANSLATION = 1 << 3
    STEP_ROTATION = 1 << 4
    STATIC_OBJECT_MOTION = 1 << 5
    RESET_DISCONTINUITY = 1 << 6
    TELEPORT = 1 << 7
    TIMESTAMP_INVALID = 1 << 8
    TIMESTAMP_NONMONOTONIC = 1 << 9
    CONTROL_STEP_INVALID = 1 << 10
    PHYSICS_SUBSTEP_INVALID = 1 << 11
    RESET_FLAG_INCONSISTENT = 1 << 12
    SIMULATOR_REPORTED_INVALID = 1 << 13


REASON_NAMES = {int(reason): reason.name.lower() for reason in PoseQualityReason}
KNOWN_REASON_MASK = sum(REASON_NAMES)
ALLOWED_ROLES = {
    "manipulated",
    "receptacle",
    "distractor",
    "support",
    "boundary",
    "tool",
    "other",
}


@dataclass(frozen=True)
class PoseQualityBatch:
    """Validated arrays ready to write beside one ``object_poses`` trajectory."""

    valid: np.ndarray
    reason_bitset: np.ndarray
    simulator_timestamp_s: np.ndarray
    control_step: np.ndarray
    physics_substep_count: np.ndarray
    reset_generation: np.ndarray
    reset_flag: np.ndarray
    teleport_flag: np.ndarray
    simulator_pose_error_flag: np.ndarray
    registry_sha256: str
    spec_sha256: str


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PoseQualityContractError(
            f"{name} fields differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _finite_nonnegative(value: Any, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise PoseQualityContractError(f"{name} must be finite and {qualifier}")
    return number


def validate_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise a stable object registry.

    Registry order is the object axis order.  Actor and asset ids must be
    supplied by the simulator integration; object display names are not used as
    substitutes for either identity.
    """

    if not isinstance(value, Mapping):
        raise PoseQualityContractError("registry must be a mapping")
    _exact_fields(value, {"format", "objects"}, "registry")
    if value["format"] != REGISTRY_FORMAT:
        raise PoseQualityContractError("unexpected object registry format")
    objects = value["objects"]
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        raise PoseQualityContractError("registry.objects must be a non-empty sequence")
    canonical: list[dict[str, Any]] = []
    for index, raw in enumerate(objects):
        if not isinstance(raw, Mapping):
            raise PoseQualityContractError(f"registry.objects[{index}] must be a mapping")
        _exact_fields(
            raw,
            {
                "name",
                "stable_sim_actor_id",
                "asset_model_id",
                "role",
                "is_static",
            },
            f"registry.objects[{index}]",
        )
        item = {
            "name": str(raw["name"]),
            "stable_sim_actor_id": str(raw["stable_sim_actor_id"]),
            "asset_model_id": str(raw["asset_model_id"]),
            "role": str(raw["role"]),
            "is_static": bool(raw["is_static"]),
        }
        for field in ("name", "stable_sim_actor_id", "asset_model_id"):
            if not item[field] or item[field].strip() != item[field]:
                raise PoseQualityContractError(
                    f"registry.objects[{index}].{field} must be a non-empty stable id"
                )
        if item["role"] not in ALLOWED_ROLES:
            raise PoseQualityContractError(f"registry.objects[{index}].role is unsupported")
        if not isinstance(raw["is_static"], (bool, np.bool_)):
            raise PoseQualityContractError(
                f"registry.objects[{index}].is_static must be a boolean"
            )
        canonical.append(item)
    for field in ("name", "stable_sim_actor_id"):
        values = [str(item[field]) for item in canonical]
        if len(set(values)) != len(values):
            raise PoseQualityContractError(f"registry {field} values must be unique")
    return {"format": REGISTRY_FORMAT, "objects": canonical}


def registry_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(validate_registry(value))


def validate_spec(
    value: Mapping[str, Any], *, expected_registry_sha256: str | None = None
) -> dict[str, Any]:
    """Validate a frozen, data-independent schema-v6 integrity spec."""

    if not isinstance(value, Mapping):
        raise PoseQualityContractError("spec must be a mapping")
    _exact_fields(
        value,
        {
            "format",
            "schema_version",
            "object_registry_sha256",
            "pose_layout",
            "time_layout",
            "thresholds",
            "threshold_basis",
        },
        "spec",
    )
    if value["format"] != SPEC_FORMAT or int(value["schema_version"]) != SCHEMA_VERSION:
        raise PoseQualityContractError("unexpected schema-v6 pose-quality spec")
    bound_registry = str(value["object_registry_sha256"])
    if not _is_sha256(bound_registry):
        raise PoseQualityContractError("spec.object_registry_sha256 is invalid")
    if expected_registry_sha256 is not None and bound_registry != expected_registry_sha256:
        raise PoseQualityContractError("spec is bound to a different object registry")

    pose_layout = value["pose_layout"]
    expected_layout = {
        "shape_suffix": [7],
        "translation_indices": [0, 1, 2],
        "quaternion_indices": [3, 4, 5, 6],
        "quaternion_order": "wxyz",
        "frame": "simulator_world",
        "translation_unit": "metre",
        "rotation_unit": "radian",
    }
    if pose_layout != expected_layout:
        raise PoseQualityContractError("pose frame/quaternion/unit contract changed")
    time_layout = value["time_layout"]
    expected_time = {
        "timestamp_unit": "second",
        "timestamp_clock": "simulator_monotonic",
        "control_step_semantics": "sample_after_completed_control_step",
        "physics_substep_semantics": "substeps_since_previous_sample_zero_at_reset",
    }
    if time_layout != expected_time:
        raise PoseQualityContractError("timestamp/control-step contract changed")

    thresholds = value["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise PoseQualityContractError("spec.thresholds must be a mapping")
    _exact_fields(
        thresholds,
        {
            "world_aabb_m",
            "quaternion_norm_abs_tolerance",
            "max_step_translation_m",
            "max_step_rotation_rad",
            "static_object_max_step_translation_m",
            "static_object_max_step_rotation_rad",
            "timestamp_step_min_s",
            "timestamp_step_max_s",
            "max_physics_substeps_per_control_step",
        },
        "spec.thresholds",
    )
    aabb = np.asarray(thresholds["world_aabb_m"], dtype=np.float64)
    if aabb.shape != (3, 2) or not np.isfinite(aabb).all() or np.any(aabb[:, 0] >= aabb[:, 1]):
        raise PoseQualityContractError("world_aabb_m must be three finite increasing pairs")
    for field in (
        "quaternion_norm_abs_tolerance",
        "max_step_translation_m",
        "max_step_rotation_rad",
        "static_object_max_step_translation_m",
        "static_object_max_step_rotation_rad",
        "timestamp_step_min_s",
        "timestamp_step_max_s",
    ):
        _finite_nonnegative(thresholds[field], f"spec.thresholds.{field}")
    if float(thresholds["timestamp_step_min_s"]) >= float(thresholds["timestamp_step_max_s"]):
        raise PoseQualityContractError("timestamp min must be smaller than max")
    substeps = thresholds["max_physics_substeps_per_control_step"]
    if isinstance(substeps, bool) or int(substeps) != substeps or int(substeps) < 1:
        raise PoseQualityContractError("max physics substeps must be a positive integer")

    basis = value["threshold_basis"]
    if not isinstance(basis, Mapping):
        raise PoseQualityContractError("spec.threshold_basis must be a mapping")
    _exact_fields(
        basis,
        {
            "thresholds_fit_from_pose_data",
            "source",
            "frozen_before_collection",
        },
        "spec.threshold_basis",
    )
    if (
        basis["thresholds_fit_from_pose_data"] is not False
        or basis["frozen_before_collection"] is not True
        or not str(basis["source"]).strip()
    ):
        raise PoseQualityContractError(
            "pose-quality thresholds must be source-documented and frozen before collection"
        )
    return json.loads(json.dumps(value, sort_keys=True))


def spec_sha256(value: Mapping[str, Any], *, expected_registry_sha256: str | None = None) -> str:
    return canonical_sha256(
        validate_spec(value, expected_registry_sha256=expected_registry_sha256)
    )


def _array(value: Any, *, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise PoseQualityContractError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def _integer_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise PoseQualityContractError(f"{name} must be an integer array with shape {shape}")
    result = raw.astype(np.int64)
    if not np.array_equal(raw, result):
        raise PoseQualityContractError(f"{name} values do not fit exact int64")
    return result


def _boolean_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind != "b":
        raise PoseQualityContractError(f"{name} must be a boolean array with shape {shape}")
    return raw.astype(bool, copy=False)


def _quaternion_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    denominator = first_norm * second_norm
    dot = np.sum(first * second, axis=-1)
    cosine = np.divide(
        np.abs(dot), dot * 0 + denominator, out=np.full_like(dot, np.nan), where=denominator > 0
    )
    return 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))


def derive_pose_quality(
    poses: Any,
    *,
    registry: Mapping[str, Any],
    spec: Mapping[str, Any],
    simulator_timestamp_s: Any,
    control_step: Any,
    physics_substep_count: Any,
    reset_generation: Any,
    reset_flag: Any,
    teleport_flag: Any,
    simulator_pose_error_flag: Any,
) -> PoseQualityBatch:
    """Derive deterministic per-object/per-step quality labels.

    ``reset_flag[0]`` must be true and the first substep count must be zero.
    Later resets are represented by both a reset-generation increment and a
    reset flag.  They are retained in the file but excluded from supervision.
    """

    canonical_registry = validate_registry(registry)
    registry_hash = canonical_sha256(canonical_registry)
    canonical_spec = validate_spec(spec, expected_registry_sha256=registry_hash)
    spec_hash = canonical_sha256(canonical_spec)
    pose = np.asarray(poses, dtype=np.float64)
    object_count = len(canonical_registry["objects"])
    if pose.ndim != 3 or pose.shape[1:] != (object_count, 7) or pose.shape[0] < 1:
        raise PoseQualityContractError(
            f"poses must have shape [T,{object_count},7] with T>=1, got {pose.shape}"
        )
    steps = pose.shape[0]
    timestamp = _array(
        simulator_timestamp_s, dtype=np.float64, shape=(steps,), name="simulator_timestamp_s"
    )
    controls = _integer_array(control_step, shape=(steps,), name="control_step")
    substeps = _integer_array(
        physics_substep_count, shape=(steps,), name="physics_substep_count"
    )
    generation = _integer_array(
        reset_generation, shape=(steps,), name="reset_generation"
    )
    reset = _boolean_array(reset_flag, shape=(steps,), name="reset_flag")
    teleport = _boolean_array(
        teleport_flag, shape=(steps, object_count), name="teleport_flag"
    )
    simulator_error = _boolean_array(
        simulator_pose_error_flag,
        shape=(steps, object_count),
        name="simulator_pose_error_flag",
    )
    if np.any(controls < 0) or np.any(substeps < 0) or np.any(generation < 0):
        raise PoseQualityContractError("control/substep/reset-generation fields must be non-negative")

    reason = np.zeros((steps, object_count), dtype=np.uint32)

    def add(step_mask: np.ndarray, bit: PoseQualityReason) -> None:
        mask = np.asarray(step_mask, dtype=bool)
        if mask.shape == (steps,):
            mask = np.broadcast_to(mask[:, None], reason.shape)
        if mask.shape != reason.shape:
            raise AssertionError(f"internal reason mask shape {mask.shape}")
        reason[mask] |= np.uint32(int(bit))

    finite_pose = np.isfinite(pose).all(axis=-1)
    add(~finite_pose, PoseQualityReason.NONFINITE_POSE)
    quaternion = pose[..., 3:7]
    quat_norm = np.linalg.norm(quaternion, axis=-1)
    add(
        ~np.isfinite(quat_norm)
        | (np.abs(quat_norm - 1.0) > float(canonical_spec["thresholds"]["quaternion_norm_abs_tolerance"])),
        PoseQualityReason.QUATERNION_NORM,
    )
    xyz = pose[..., :3]
    aabb = np.asarray(canonical_spec["thresholds"]["world_aabb_m"], dtype=np.float64)
    outside = np.any((xyz < aabb[:, 0]) | (xyz > aabb[:, 1]), axis=-1)
    add(outside | ~np.isfinite(xyz).all(axis=-1), PoseQualityReason.WORLD_AABB)
    add(simulator_error, PoseQualityReason.SIMULATOR_REPORTED_INVALID)

    add(~np.isfinite(timestamp), PoseQualityReason.TIMESTAMP_INVALID)
    reset_consistent = np.zeros(steps, dtype=bool)
    reset_consistent[0] = bool(reset[0])
    if steps > 1:
        generation_delta = np.diff(generation)
        reset_consistent[1:] = (generation_delta == 1) == reset[1:]
        reset_consistent[1:] &= (generation_delta == 0) | (generation_delta == 1)
    add(~reset_consistent, PoseQualityReason.RESET_FLAG_INCONSISTENT)
    if not reset[0]:
        add(np.arange(steps) == 0, PoseQualityReason.RESET_FLAG_INCONSISTENT)
    if substeps[0] != 0:
        add(np.arange(steps) == 0, PoseQualityReason.PHYSICS_SUBSTEP_INVALID)
    if controls[0] != 0:
        add(np.arange(steps) == 0, PoseQualityReason.CONTROL_STEP_INVALID)

    if steps > 1:
        normal = (~reset[1:]) & reset_consistent[1:]
        reset_transition = reset[1:] & reset_consistent[1:]
        timestamp_delta = np.diff(timestamp)
        timestamp_bad = np.zeros(steps, dtype=bool)
        minimum = float(canonical_spec["thresholds"]["timestamp_step_min_s"])
        maximum = float(canonical_spec["thresholds"]["timestamp_step_max_s"])
        timestamp_bad[1:] = normal & (
            ~np.isfinite(timestamp_delta)
            | (timestamp_delta < minimum)
            | (timestamp_delta > maximum)
        )
        add(timestamp_bad, PoseQualityReason.TIMESTAMP_NONMONOTONIC)

        control_bad = np.zeros(steps, dtype=bool)
        control_bad[1:] = normal & (np.diff(controls) != 1)
        control_bad[1:] |= reset_transition & (controls[1:] != 0)
        add(control_bad, PoseQualityReason.CONTROL_STEP_INVALID)

        substep_bad = np.zeros(steps, dtype=bool)
        max_substeps = int(canonical_spec["thresholds"]["max_physics_substeps_per_control_step"])
        substep_bad[1:] = normal & ((substeps[1:] < 1) | (substeps[1:] > max_substeps))
        substep_bad[1:] |= reset_transition & (substeps[1:] != 0)
        add(substep_bad, PoseQualityReason.PHYSICS_SUBSTEP_INVALID)

        reset_discontinuity = np.zeros(steps, dtype=bool)
        reset_discontinuity[1:] = reset_transition
        add(reset_discontinuity, PoseQualityReason.RESET_DISCONTINUITY)

        translation = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
        angles = _quaternion_angles(quaternion[:-1], quaternion[1:])
        translation_bad = np.zeros(reason.shape, dtype=bool)
        rotation_bad = np.zeros(reason.shape, dtype=bool)
        translation_bad[1:] = normal[:, None] & (
            ~np.isfinite(translation)
            | (translation > float(canonical_spec["thresholds"]["max_step_translation_m"]))
        )
        rotation_bad[1:] = normal[:, None] & (
            ~np.isfinite(angles)
            | (angles > float(canonical_spec["thresholds"]["max_step_rotation_rad"]))
        )
        add(translation_bad, PoseQualityReason.STEP_TRANSLATION)
        add(rotation_bad, PoseQualityReason.STEP_ROTATION)

        static = np.asarray(
            [bool(item["is_static"]) for item in canonical_registry["objects"]], dtype=bool
        )
        static_bad = np.zeros(reason.shape, dtype=bool)
        static_bad[1:] = normal[:, None] & static[None, :] & (
            (translation > float(canonical_spec["thresholds"]["static_object_max_step_translation_m"]))
            | (angles > float(canonical_spec["thresholds"]["static_object_max_step_rotation_rad"]))
        )
        add(static_bad, PoseQualityReason.STATIC_OBJECT_MOTION)

    add(teleport, PoseQualityReason.TELEPORT)
    valid = reason == 0
    return PoseQualityBatch(
        valid=valid,
        reason_bitset=reason,
        simulator_timestamp_s=timestamp,
        control_step=controls.astype(np.uint64),
        physics_substep_count=substeps.astype(np.uint32),
        reset_generation=generation.astype(np.uint32),
        reset_flag=reset,
        teleport_flag=teleport,
        simulator_pose_error_flag=simulator_error,
        registry_sha256=registry_hash,
        spec_sha256=spec_hash,
    )


def _hash_array(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def pose_quality_payload_sha256(
    poses: Any, batch: PoseQualityBatch, *, registry_sha: str, spec_sha: str
) -> str:
    digest = hashlib.sha256()
    digest.update(FORMAT.encode("ascii") + b"\0")
    digest.update(registry_sha.encode("ascii") + b"\0")
    digest.update(spec_sha.encode("ascii") + b"\0")
    arrays = {
        "poses": np.asarray(poses, dtype=np.float64),
        "pose_quality_valid": np.asarray(batch.valid, dtype=bool),
        "pose_quality_reason_bitset": np.asarray(batch.reason_bitset, dtype=np.uint32),
        "simulator_timestamp_s": np.asarray(batch.simulator_timestamp_s, dtype=np.float64),
        "control_step": np.asarray(batch.control_step, dtype=np.uint64),
        "physics_substep_count": np.asarray(batch.physics_substep_count, dtype=np.uint32),
        "reset_generation": np.asarray(batch.reset_generation, dtype=np.uint32),
        "reset_flag": np.asarray(batch.reset_flag, dtype=bool),
        "teleport_flag": np.asarray(batch.teleport_flag, dtype=bool),
        "simulator_pose_error_flag": np.asarray(batch.simulator_pose_error_flag, dtype=bool),
    }
    for name in sorted(arrays):
        _hash_array(digest, name, arrays[name])
    return digest.hexdigest()


def write_pose_quality_v6(
    trajectory_group: h5py.Group,
    *,
    registry: Mapping[str, Any],
    spec: Mapping[str, Any],
    simulator_timestamp_s: Any,
    control_step: Any,
    physics_substep_count: Any,
    reset_generation: Any,
    reset_flag: Any,
    teleport_flag: Any,
    simulator_pose_error_flag: Any,
    pose_dataset_name: str = "object_poses",
    quality_group_name: str = GROUP_NAME,
) -> dict[str, Any]:
    """Write a self-authenticating schema-v6 quality group beside poses.

    Existing groups are never overwritten.  The caller should write to a new
    temporary HDF5 file and atomically rename that file after all branches pass
    ``validate_pose_quality_v6``.
    """

    if int(trajectory_group.file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
        raise PoseQualityContractError(
            "writer requires a new schema-v6 HDF root; schema-v5 files must not be upgraded in place"
        )
    if pose_dataset_name not in trajectory_group:
        raise PoseQualityContractError(f"missing pose dataset {pose_dataset_name}")
    if quality_group_name in trajectory_group:
        raise PoseQualityContractError(f"refusing to overwrite {quality_group_name}")
    pose_dataset = trajectory_group[pose_dataset_name]
    if not isinstance(pose_dataset, h5py.Dataset):
        raise PoseQualityContractError("pose dataset path must identify a dataset")
    poses = np.asarray(pose_dataset[:], dtype=np.float64)
    canonical_registry = validate_registry(registry)
    batch = derive_pose_quality(
        poses,
        registry=canonical_registry,
        spec=spec,
        simulator_timestamp_s=simulator_timestamp_s,
        control_step=control_step,
        physics_substep_count=physics_substep_count,
        reset_generation=reset_generation,
        reset_flag=reset_flag,
        teleport_flag=teleport_flag,
        simulator_pose_error_flag=simulator_pose_error_flag,
    )
    canonical_spec = validate_spec(spec, expected_registry_sha256=batch.registry_sha256)
    payload_sha = pose_quality_payload_sha256(
        poses, batch, registry_sha=batch.registry_sha256, spec_sha=batch.spec_sha256
    )
    group = trajectory_group.create_group(quality_group_name)
    group.attrs["format"] = FORMAT
    group.attrs["schema_version"] = SCHEMA_VERSION
    group.attrs["pose_dataset_name"] = pose_dataset_name
    group.attrs["frame"] = canonical_spec["pose_layout"]["frame"]
    group.attrs["translation_unit"] = canonical_spec["pose_layout"]["translation_unit"]
    group.attrs["rotation_unit"] = canonical_spec["pose_layout"]["rotation_unit"]
    group.attrs["quaternion_order"] = canonical_spec["pose_layout"]["quaternion_order"]
    group.attrs["object_registry_sha256"] = batch.registry_sha256
    group.attrs["pose_integrity_spec_sha256"] = batch.spec_sha256
    group.attrs["logical_payload_sha256"] = payload_sha
    strings = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        "object_registry_json",
        data=json.dumps(canonical_registry, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        dtype=strings,
    )
    group.create_dataset(
        "pose_integrity_spec_json",
        data=json.dumps(canonical_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        dtype=strings,
    )
    datasets = {
        "pose_quality_valid": batch.valid,
        "pose_quality_reason_bitset": batch.reason_bitset,
        "simulator_timestamp_s": batch.simulator_timestamp_s,
        "control_step": batch.control_step,
        "physics_substep_count": batch.physics_substep_count,
        "reset_generation": batch.reset_generation,
        "reset_flag": batch.reset_flag,
        "teleport_flag": batch.teleport_flag,
        "simulator_pose_error_flag": batch.simulator_pose_error_flag,
    }
    for name, value in datasets.items():
        array = np.asarray(value)
        group.create_dataset(name, data=array, compression="gzip" if array.size > 64 else None)
    return {
        "format": FORMAT,
        "steps": int(poses.shape[0]),
        "objects": int(poses.shape[1]),
        "valid_samples": int(batch.valid.sum()),
        "invalid_samples": int(batch.valid.size - batch.valid.sum()),
        "object_registry_sha256": batch.registry_sha256,
        "pose_integrity_spec_sha256": batch.spec_sha256,
        "logical_payload_sha256": payload_sha,
    }


def _decode_scalar(dataset: h5py.Dataset, name: str) -> dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        decoded = json.loads(str(value))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PoseQualityContractError(f"{name} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise PoseQualityContractError(f"{name} must contain a JSON object")
    return decoded


def validate_pose_quality_v6(
    trajectory_group: h5py.Group,
    *,
    expected_registry_sha256: str | None = None,
    expected_spec_sha256: str | None = None,
    quality_group_name: str = GROUP_NAME,
) -> dict[str, Any]:
    """Authenticate, rederive, and compare one schema-v6 trajectory contract."""

    if quality_group_name not in trajectory_group:
        raise PoseQualityContractError(f"missing {quality_group_name}; unlabelled poses fail closed")
    if int(trajectory_group.file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
        raise PoseQualityContractError("pose-quality data is not contained in a schema-v6 HDF root")
    group = trajectory_group[quality_group_name]
    if not isinstance(group, h5py.Group):
        raise PoseQualityContractError("pose-quality path must identify a group")
    required_attrs = {
        "format",
        "schema_version",
        "pose_dataset_name",
        "frame",
        "translation_unit",
        "rotation_unit",
        "quaternion_order",
        "object_registry_sha256",
        "pose_integrity_spec_sha256",
        "logical_payload_sha256",
    }
    if not required_attrs.issubset(group.attrs.keys()):
        raise PoseQualityContractError("pose-quality metadata is incomplete")
    if str(group.attrs["format"]) != FORMAT or int(group.attrs["schema_version"]) != SCHEMA_VERSION:
        raise PoseQualityContractError("unexpected pose-quality format/schema")
    required_datasets = {
        "object_registry_json",
        "pose_integrity_spec_json",
        "pose_quality_valid",
        "pose_quality_reason_bitset",
        "simulator_timestamp_s",
        "control_step",
        "physics_substep_count",
        "reset_generation",
        "reset_flag",
        "teleport_flag",
        "simulator_pose_error_flag",
    }
    if not required_datasets.issubset(group.keys()):
        raise PoseQualityContractError("pose-quality datasets are incomplete")
    registry = validate_registry(_decode_scalar(group["object_registry_json"], "registry"))
    actual_registry_sha = canonical_sha256(registry)
    stored_registry_sha = str(group.attrs["object_registry_sha256"])
    if stored_registry_sha != actual_registry_sha or (
        expected_registry_sha256 is not None and actual_registry_sha != expected_registry_sha256
    ):
        raise PoseQualityContractError("object registry SHA mismatch")
    spec = validate_spec(
        _decode_scalar(group["pose_integrity_spec_json"], "pose integrity spec"),
        expected_registry_sha256=actual_registry_sha,
    )
    actual_spec_sha = canonical_sha256(spec)
    stored_spec_sha = str(group.attrs["pose_integrity_spec_sha256"])
    if stored_spec_sha != actual_spec_sha or (
        expected_spec_sha256 is not None and actual_spec_sha != expected_spec_sha256
    ):
        raise PoseQualityContractError("pose integrity spec SHA mismatch")
    layout_attrs = {
        "frame": "frame",
        "translation_unit": "translation_unit",
        "rotation_unit": "rotation_unit",
        "quaternion_order": "quaternion_order",
    }
    for attr, key in layout_attrs.items():
        if str(group.attrs[attr]) != str(spec["pose_layout"][key]):
            raise PoseQualityContractError(f"pose layout attr {attr} was tampered")
    pose_name = str(group.attrs["pose_dataset_name"])
    if pose_name not in trajectory_group or not isinstance(trajectory_group[pose_name], h5py.Dataset):
        raise PoseQualityContractError("bound pose dataset is missing")
    poses = np.asarray(trajectory_group[pose_name][:], dtype=np.float64)
    recomputed = derive_pose_quality(
        poses,
        registry=registry,
        spec=spec,
        simulator_timestamp_s=group["simulator_timestamp_s"][:],
        control_step=group["control_step"][:],
        physics_substep_count=group["physics_substep_count"][:],
        reset_generation=group["reset_generation"][:],
        reset_flag=group["reset_flag"][:],
        teleport_flag=group["teleport_flag"][:],
        simulator_pose_error_flag=group["simulator_pose_error_flag"][:],
    )
    stored_valid = np.asarray(group["pose_quality_valid"][:], dtype=bool)
    stored_reason = np.asarray(group["pose_quality_reason_bitset"][:], dtype=np.uint32)
    if np.any(stored_reason & np.uint32(~KNOWN_REASON_MASK & 0xFFFFFFFF)):
        raise PoseQualityContractError("pose-quality reason bitset contains unknown bits")
    if not np.array_equal(stored_valid, recomputed.valid) or not np.array_equal(
        stored_reason, recomputed.reason_bitset
    ):
        raise PoseQualityContractError("stored pose-quality labels differ from recomputation")
    payload_sha = pose_quality_payload_sha256(
        poses, recomputed, registry_sha=actual_registry_sha, spec_sha=actual_spec_sha
    )
    if str(group.attrs["logical_payload_sha256"]) != payload_sha:
        raise PoseQualityContractError("pose-quality logical payload SHA mismatch")
    return {
        "format": FORMAT,
        "steps": int(poses.shape[0]),
        "objects": int(poses.shape[1]),
        "valid_samples": int(stored_valid.sum()),
        "invalid_samples": int(stored_valid.size - stored_valid.sum()),
        "object_registry_sha256": actual_registry_sha,
        "pose_integrity_spec_sha256": actual_spec_sha,
        "logical_payload_sha256": payload_sha,
    }


def derive_interval_supervision_mask(
    *,
    pose_quality_valid: Any,
    pose_quality_reason_bitset: Any,
    reset_generation: Any,
    reset_flag: Any,
    teleport_flag: Any,
    start_steps: Any,
    end_steps: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fail-closed masks/reasons for endpoint-delta supervision.

    An interval ``[start, end]`` is safe only when every destination sample in
    ``(start, end]`` is valid and it crosses neither a reset nor a teleport.
    The source snapshot itself must have no snapshot-integrity error; temporal
    bits attached to the transition ending at ``start`` do not poison the next
    otherwise-valid transition.
    """

    valid = np.asarray(pose_quality_valid, dtype=bool)
    reason = np.asarray(pose_quality_reason_bitset, dtype=np.uint32)
    if valid.ndim != 2 or reason.shape != valid.shape:
        raise PoseQualityContractError("pose quality arrays must align as [T,O]")
    steps, objects = valid.shape
    generation = _integer_array(reset_generation, shape=(steps,), name="reset_generation")
    reset = _boolean_array(reset_flag, shape=(steps,), name="reset_flag")
    teleport = _boolean_array(teleport_flag, shape=(steps, objects), name="teleport_flag")
    starts = np.asarray(start_steps, dtype=np.int64)
    ends = np.asarray(end_steps, dtype=np.int64)
    if starts.ndim != 1 or ends.shape != starts.shape:
        raise PoseQualityContractError("start_steps/end_steps must align as [N]")
    if np.any(starts < 0) or np.any(ends >= steps) or np.any(ends <= starts):
        raise PoseQualityContractError("each interval must satisfy 0 <= start < end < T")
    masks = np.ones((len(starts), objects), dtype=bool)
    reasons = np.zeros((len(starts), objects), dtype=np.uint32)
    snapshot_bits = int(
        PoseQualityReason.NONFINITE_POSE
        | PoseQualityReason.QUATERNION_NORM
        | PoseQualityReason.WORLD_AABB
        | PoseQualityReason.SIMULATOR_REPORTED_INVALID
    )
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        interval_reason = np.bitwise_or.reduce(reason[start + 1 : end + 1], axis=0)
        interval_reason |= reason[start] & np.uint32(snapshot_bits)
        if generation[start] != generation[end] or np.any(reset[start + 1 : end + 1]):
            interval_reason |= np.uint32(int(PoseQualityReason.RESET_DISCONTINUITY))
        teleported = np.any(teleport[start + 1 : end + 1], axis=0)
        interval_reason[teleported] |= np.uint32(int(PoseQualityReason.TELEPORT))
        reasons[row] = interval_reason
        masks[row] = interval_reason == 0
    return masks, reasons


def load_object_delta_supervision_v6(
    trajectory_group: h5py.Group,
    *,
    start_steps: Any,
    end_steps: Any,
    expected_registry_sha256: str | None = None,
    expected_spec_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Authenticate schema-v6 and return xyz deltas plus safe-label masks."""

    validate_pose_quality_v6(
        trajectory_group,
        expected_registry_sha256=expected_registry_sha256,
        expected_spec_sha256=expected_spec_sha256,
    )
    quality = trajectory_group[GROUP_NAME]
    pose_name = str(quality.attrs["pose_dataset_name"])
    poses = np.asarray(trajectory_group[pose_name][:], dtype=np.float64)
    starts = np.asarray(start_steps, dtype=np.int64)
    ends = np.asarray(end_steps, dtype=np.int64)
    mask, reason = derive_interval_supervision_mask(
        pose_quality_valid=quality["pose_quality_valid"][:],
        pose_quality_reason_bitset=quality["pose_quality_reason_bitset"][:],
        reset_generation=quality["reset_generation"][:],
        reset_flag=quality["reset_flag"][:],
        teleport_flag=quality["teleport_flag"][:],
        start_steps=starts,
        end_steps=ends,
    )
    return {
        "object_delta_xyz_m": poses[ends, :, :3] - poses[starts, :, :3],
        "object_delta_supervision_valid": mask,
        "object_delta_invalid_reason_bitset": reason,
    }


__all__ = [
    "FORMAT",
    "GROUP_NAME",
    "REGISTRY_FORMAT",
    "REASON_NAMES",
    "SCHEMA_VERSION",
    "SPEC_FORMAT",
    "PoseQualityBatch",
    "PoseQualityContractError",
    "PoseQualityReason",
    "canonical_sha256",
    "derive_interval_supervision_mask",
    "derive_pose_quality",
    "file_sha256",
    "load_object_delta_supervision_v6",
    "pose_quality_payload_sha256",
    "registry_sha256",
    "spec_sha256",
    "validate_pose_quality_v6",
    "validate_registry",
    "validate_spec",
    "write_pose_quality_v6",
]
