#!/usr/bin/env python3
"""Data-blind, fail-closed SmolVLA(Aloha-trained) -> Piper preflight.

This module authenticates policy/body artifacts and validates already-produced
forward outputs.  It deliberately has no simulator, environment-step, reward,
success-label, rollout, or task-outcome input.  A passing report authorizes only
an offline policy forward probe.  It never authorizes robot execution or a
cross-embodiment performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


FORMAT = "smolvla_piper_zero_shot_preflight_v2"
ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
AUTHORIZATION = "forward_only"
SOURCE_DATASET_REPO = "pepijn223/robotwin_unified_v3"
SOURCE_DATASET_FRAME_COUNT = 6_075_103
EXPECTED_CANDIDATE_SHAPE = (4, 50, 14)
SHA256_LENGTH = 64

ARTIFACT_SUFFIXES = {
    "checkpoint_config": (".json",),
    "model_weights": (".safetensors",),
    "train_config": (".json",),
    "policy_preprocessor": (".json",),
    "policy_postprocessor": (".json",),
    "preprocessor_stats": (".safetensors",),
    "postprocessor_stats": (".safetensors",),
    "piper_body_config": (".yml", ".yaml"),
    "aloha_body_config": (".yml", ".yaml"),
    "piper_urdf": (".urdf",),
    "aloha_urdf": (".urdf",),
}

PROBE_ARTIFACT_SUFFIXES = {
    "forward_probe_receipt": (".json",),
    "candidate_actions": (".npy",),
    "shared_prefixes": (".npy",),
    "source_image": (".png", ".jpg", ".jpeg"),
}

POLICY_ARTIFACT_FILENAMES = {
    "checkpoint_config": "config.json",
    "model_weights": "model.safetensors",
    "train_config": "train_config.json",
    "policy_preprocessor": "policy_preprocessor.json",
    "policy_postprocessor": "policy_postprocessor.json",
    "preprocessor_stats": "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "postprocessor_stats": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
}

BODY_ASSET_LAYOUT = {
    "piper": {
        "directory": "piper",
        "config_filename": "config.yml",
        "urdf_relative_path": "piper.urdf",
    },
    "aloha": {
        "directory": "aloha-agilex",
        "config_filename": "config.yml",
        "urdf_relative_path": "urdf/arx5_description_isaac.urdf",
    },
}

ALOHA_FEATURE_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
ALOHA_PHYSICAL_JOINTS = tuple(
    [f"fl_joint{i}" for i in range(1, 8)]
    + [f"fr_joint{i}" for i in range(1, 8)]
)
PIPER_QUALIFIED_JOINTS = tuple(
    [f"left:joint{i}" for i in range(1, 8)]
    + [f"right:joint{i}" for i in range(1, 8)]
)

PIPER_ARM_LIMITS = (
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.697, 2.697),
    (-1.832, 1.832),
    (-1.22, 1.22),
    (-3.14, 3.14),
)
ALOHA_ARM_LIMITS = tuple([(-10.0, 10.0)] * 6)


class PreflightError(ValueError):
    """A fail-closed preflight contract violation."""


class StateDimensionConflict(PreflightError):
    """The checkpoint's declared and actual state widths conflict."""


@dataclass(frozen=True)
class SlotSpec:
    index: int
    side: str
    ordinal: int
    source_feature_name: str
    source_joint_name: str
    target_joint_name: str
    target_config_joint_name: str
    kind: str
    numeric_transform: str
    lower: float
    upper: float


def _slot_specs() -> tuple[SlotSpec, ...]:
    specs: list[SlotSpec] = []
    for index in range(14):
        side = "left" if index < 7 else "right"
        within_arm = index % 7
        if within_arm == 6:
            kind = "normalized_gripper_[0,1]"
            transform = "normalized_fraction_preserved_target_scale_not_applied"
            lower, upper = 0.0, 1.0
        else:
            kind = "arm_qpos_radian"
            transform = "angle_radians_preserved_by_named_side_and_ordinal"
            lower, upper = PIPER_ARM_LIMITS[within_arm]
        specs.append(
            SlotSpec(
                index=index,
                side=side,
                ordinal=within_arm + 1,
                source_feature_name=ALOHA_FEATURE_NAMES[index],
                source_joint_name=ALOHA_PHYSICAL_JOINTS[index],
                target_joint_name=PIPER_QUALIFIED_JOINTS[index],
                target_config_joint_name=f"joint{within_arm + 1}",
                kind=kind,
                numeric_transform=transform,
                lower=lower,
                upper=upper,
            )
        )
    return tuple(specs)


PIPER_ACTION_SLOTS = _slot_specs()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == SHA256_LENGTH and all(c in "0123456789abcdef" for c in text)


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise PreflightError(
            f"{name} fields differ: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _path_has_fresh_token(path: Path) -> bool:
    return any("fresh" in component.casefold() for component in path.parts)


def reject_fresh_path(path: str | Path, name: str) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve(strict=False)
    if _path_has_fresh_token(candidate) or _path_has_fresh_token(resolved):
        raise PreflightError(f"{name} must not reference a Fresh path")
    return resolved


def _authenticate_artifact(
    value: Any,
    name: str,
    *,
    suffixes: Mapping[str, tuple[str, ...]] = ARTIFACT_SUFFIXES,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"artifacts.{name} must be a mapping")
    _exact_fields(value, {"path", "sha256"}, f"artifacts.{name}")
    raw_path = Path(str(value["path"])).expanduser()
    if not raw_path.is_absolute():
        raise PreflightError(f"artifacts.{name}.path must be absolute")
    path = reject_fresh_path(raw_path, f"artifacts.{name}.path")
    if not path.is_file():
        raise PreflightError(f"artifacts.{name} is not an existing file")
    if path.suffix.casefold() not in suffixes[name]:
        raise PreflightError(f"artifacts.{name} has an unexpected file type")
    expected_sha = str(value["sha256"])
    if not _is_sha256(expected_sha):
        raise PreflightError(f"artifacts.{name}.sha256 must be lowercase SHA256")
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise PreflightError(f"artifacts.{name} SHA256 mismatch")
    return {"path": str(path), "sha256": actual_sha}


def authenticate_artifacts(value: Any) -> dict[str, dict[str, str]]:
    """Authenticate every static policy/body artifact before parsing it."""

    if not isinstance(value, Mapping):
        raise PreflightError("artifacts must be a mapping")
    _exact_fields(value, set(ARTIFACT_SUFFIXES), "artifacts")
    return {name: _authenticate_artifact(value[name], name) for name in sorted(value)}


def authenticate_probe_artifacts(value: Any) -> dict[str, dict[str, str]]:
    """Authenticate every already-produced offline probe artifact."""

    if not isinstance(value, Mapping):
        raise PreflightError("probe_artifacts must be a mapping")
    _exact_fields(value, set(PROBE_ARTIFACT_SUFFIXES), "probe_artifacts")
    return {
        name: _authenticate_artifact(
            value[name], name, suffixes=PROBE_ARTIFACT_SUFFIXES
        )
        for name in sorted(value)
    }


def _load_json(path: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{name} must contain a JSON object")
    return value


def _shape(feature: Any, name: str) -> tuple[int, ...]:
    if not isinstance(feature, Mapping) or not isinstance(feature.get("shape"), list):
        raise PreflightError(f"{name} lacks an explicit shape")
    shape = tuple(feature["shape"])
    if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise PreflightError(f"{name}.shape must contain positive integers")
    return shape


def _normalizer_step(value: Mapping[str, Any], registry_name: str, name: str) -> Mapping[str, Any]:
    steps = value.get("steps")
    if not isinstance(steps, list):
        raise PreflightError(f"{name}.steps must be a list")
    matches = [step for step in steps if isinstance(step, Mapping) and step.get("registry_name") == registry_name]
    if len(matches) != 1:
        raise PreflightError(f"{name} must contain exactly one {registry_name}")
    return matches[0]


_SAFETENSOR_DTYPES: dict[str, tuple[np.dtype[Any], int]] = {
    "BOOL": (np.dtype("?"), 1),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "I16": (np.dtype("<i2"), 2),
    "U16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U64": (np.dtype("<u8"), 8),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "F64": (np.dtype("<f8"), 8),
}


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Read numeric safetensors with strict header/offset validation."""

    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise PreflightError("safetensors file is truncated")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size <= 0 or 8 + header_size > len(raw):
        raise PreflightError("safetensors header size is invalid")
    try:
        header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError("safetensors header is invalid") from exc
    if not isinstance(header, dict):
        raise PreflightError("safetensors header must be a mapping")
    data = memoryview(raw)[8 + header_size :]
    result: dict[str, np.ndarray] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, Mapping):
            raise PreflightError(f"safetensors tensor {name} has invalid metadata")
        dtype_name = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype_name not in _SAFETENSOR_DTYPES:
            raise PreflightError(f"unsupported safetensors dtype for {name}: {dtype_name}")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise PreflightError(f"safetensors tensor {name} has invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > len(data)
        ):
            raise PreflightError(f"safetensors tensor {name} has invalid offsets")
        dtype, width = _SAFETENSOR_DTYPES[str(dtype_name)]
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if offsets[1] - offsets[0] != elements * width:
            raise PreflightError(f"safetensors tensor {name} byte size is inconsistent")
        array = np.frombuffer(data[offsets[0] : offsets[1]], dtype=dtype).reshape(shape)
        result[str(name)] = array.copy()
    return result


def _require_stats(
    tensors: Mapping[str, np.ndarray], prefixes: Sequence[str], name: str
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for prefix in prefixes:
        result[prefix] = {}
        for statistic in ("count", "min", "max", "mean", "std"):
            key = f"{prefix}.{statistic}"
            if key not in tensors:
                raise PreflightError(f"{name} lacks required tensor {key}")
            array = tensors[key]
            expected_shape = (1,) if statistic == "count" else (14,)
            if array.shape != expected_shape:
                raise PreflightError(
                    f"{name}.{key} shape {array.shape} != {expected_shape}"
                )
            if not np.all(np.isfinite(array)):
                raise PreflightError(f"{name}.{key} contains non-finite values")
            result[prefix][statistic] = array
        count = float(result[prefix]["count"][0])
        if count != SOURCE_DATASET_FRAME_COUNT:
            raise PreflightError(
                f"{name}.{prefix}.count {count} != audited Aloha frame count "
                f"{SOURCE_DATASET_FRAME_COUNT}"
            )
        if np.any(result[prefix]["std"] <= 0):
            raise PreflightError(f"{name}.{prefix}.std must be strictly positive")
        if np.any(result[prefix]["min"] > result[prefix]["max"]):
            raise PreflightError(f"{name}.{prefix} has inverted min/max")
    return result


def _yaml_mapping(path: str, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PreflightError(f"{name} is not valid YAML") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{name} must contain a YAML mapping")
    return value


def _validate_policy_artifact_layout(
    artifacts: Mapping[str, Mapping[str, str]],
) -> str:
    """Bind config, weights and pre/postprocessor files to one actor directory."""

    parents: set[Path] = set()
    for name, expected_filename in POLICY_ARTIFACT_FILENAMES.items():
        path = Path(artifacts[name]["path"])
        if path.name != expected_filename:
            raise PreflightError(
                f"artifacts.{name} filename {path.name} != {expected_filename}"
            )
        parents.add(path.parent)
    if len(parents) != 1:
        raise PreflightError("policy artifacts must resolve inside one actor directory")
    return str(next(iter(parents)))


def _validate_body_asset_layout(
    artifacts: Mapping[str, Mapping[str, str]],
    body_config: Mapping[str, Any],
    body: str,
) -> dict[str, str]:
    """Bind RoboTwin ``assets/embodiments/*/config.yml`` to its declared URDF."""

    expected = BODY_ASSET_LAYOUT[body]
    config_path = Path(artifacts[f"{body}_body_config"]["path"])
    urdf_path = Path(artifacts[f"{body}_urdf"]["path"])
    if config_path.name != expected["config_filename"]:
        raise PreflightError(f"{body} body config must be the real config.yml asset")
    if config_path.parent.name != expected["directory"]:
        raise PreflightError(f"{body} body config directory differs from RoboTwin registry")
    if config_path.parent.parent.name != "embodiments" or config_path.parent.parent.parent.name != "assets":
        raise PreflightError(
            f"{body} body config must resolve under assets/embodiments/{expected['directory']}"
        )
    declared = body_config.get("urdf_path")
    accepted_declared = {
        expected["urdf_relative_path"],
        f"./{expected['urdf_relative_path']}",
    }
    if declared not in accepted_declared:
        raise PreflightError(f"{body}.urdf_path differs from the real RoboTwin asset")
    declared_path = (config_path.parent / str(declared)).resolve(strict=False)
    if declared_path != urdf_path:
        raise PreflightError(f"{body} authenticated URDF is not the config.yml urdf_path target")
    if urdf_path.name != Path(expected["urdf_relative_path"]).name:
        raise PreflightError(f"{body} URDF filename differs from the real RoboTwin asset")
    return {
        "config_path": str(config_path),
        "config_relative_registry_path": (
            f"assets/embodiments/{expected['directory']}/config.yml"
        ),
        "declared_urdf_path": str(declared),
        "resolved_urdf_path": str(urdf_path),
    }


def _numeric_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise PreflightError(f"{name} must be a numeric pair")
    try:
        pair = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{name} must be a numeric pair") from exc
    if not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
        raise PreflightError(f"{name} must be finite and increasing")
    return pair


def _validate_body_config(value: Mapping[str, Any], body: str) -> dict[str, Any]:
    if body == "piper":
        expected_arms = [[f"joint{i}" for i in range(1, 7)]] * 2
        expected_grippers = ["joint7", "joint7"]
        expected_scale = (-0.01, 0.04)
        expected_dual = False
    elif body == "aloha":
        expected_arms = [
            [f"fl_joint{i}" for i in range(1, 7)],
            [f"fr_joint{i}" for i in range(1, 7)],
        ]
        expected_grippers = ["fl_joint7", "fr_joint7"]
        expected_scale = (-0.01, 0.045)
        expected_dual = True
    else:  # pragma: no cover - internal invariant
        raise AssertionError(body)
    if value.get("arm_joints_name") != expected_arms:
        raise PreflightError(f"{body} arm_joints_name does not match the frozen registry")
    grippers = value.get("gripper_name")
    if (
        not isinstance(grippers, list)
        or len(grippers) != 2
        or any(not isinstance(item, Mapping) for item in grippers)
        or [item.get("base") for item in grippers] != expected_grippers
    ):
        raise PreflightError(f"{body} gripper joint ordering is invalid")
    scale = _numeric_pair(value.get("gripper_scale"), f"{body}.gripper_scale")
    if scale != expected_scale:
        raise PreflightError(f"{body} gripper_scale differs from the frozen contract")
    if value.get("dual_arm") is not expected_dual:
        raise PreflightError(f"{body}.dual_arm differs from the frozen contract")
    return {
        "arm_joints_name": expected_arms,
        "gripper_base_names": expected_grippers,
        "gripper_scale": list(scale),
        "dual_arm": expected_dual,
    }


def _urdf_limits(path: str, joint_names: Sequence[str], name: str) -> dict[str, tuple[float, float]]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PreflightError(f"{name} is not valid URDF XML") from exc
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    result: dict[str, tuple[float, float]] = {}
    for joint_name in joint_names:
        joint = joints.get(joint_name)
        if joint is None:
            raise PreflightError(f"{name} lacks joint {joint_name}")
        expected_type = "prismatic" if joint_name.endswith("joint7") else "revolute"
        if joint.get("type") != expected_type:
            raise PreflightError(
                f"{name}.{joint_name} type {joint.get('type')} != {expected_type}"
            )
        limit = joint.find("limit")
        if limit is None:
            raise PreflightError(f"{name}.{joint_name} lacks limits")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightError(f"{name}.{joint_name} has invalid limits") from exc
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise PreflightError(f"{name}.{joint_name} limits must be finite/increasing")
        result[joint_name] = (lower, upper)
    return result


def _assert_limits(
    actual: Mapping[str, tuple[float, float]],
    names: Sequence[str],
    expected: Sequence[tuple[float, float]],
    body: str,
) -> None:
    for name, pair in zip(names, expected, strict=True):
        if not np.allclose(actual[name], pair, rtol=0.0, atol=1e-9):
            raise PreflightError(f"{body} URDF limit mismatch for {name}")


def expected_slot_mapping_contract() -> dict[str, Any]:
    """Return the only accepted named, non-identity-by-dimension adapter contract."""

    return {
        "mode": "explicit_named_ordinal_angle_preserving_mapping",
        "source_registry": "robotwin_unified_v3_aloha_14d",
        "target_registry": "robotwin_dual_piper_14d",
        "mapping_basis": "explicit_ordinal_chain_not_equal_dimension_identity",
        "derived_from_equal_dimension": False,
        "arm_angle_units": "radians",
        "arm_numeric_transform": "identity_value_copy_by_named_side_and_ordinal",
        "angle_values_preserved": True,
        "joint_axes_or_kinematics_equivalent": False,
        "physical_equivalence_claimed": False,
        "gripper_abstraction": (
            "normalized_[0,1]_preserved; body-specific physical scale is not applied "
            "during this forward-only preflight"
        ),
        "execution_authorized": False,
        "mapping": [
            {
                "index": slot.index,
                "side": slot.side,
                "ordinal": slot.ordinal,
                "source_feature_name": slot.source_feature_name,
                "source_joint_name": slot.source_joint_name,
                "target_joint_name": slot.target_joint_name,
                "target_config_joint_name": slot.target_config_joint_name,
                "kind": slot.kind,
                "numeric_transform": slot.numeric_transform,
            }
            for slot in PIPER_ACTION_SLOTS
        ],
    }


def expected_state_dimension_resolution() -> dict[str, Any]:
    """Explicitly retain, rather than hide, the 6-vs-14 checkpoint conflict."""

    return {
        "mode": "explicit_forward_only_runtime_shape_probe",
        "checkpoint_policy_state_dim": 6,
        "train_policy_state_dim": 6,
        "train_env_state_dim": 14,
        "normalizer_state_dim": 14,
        "runtime_probe_state_dim": 14,
        "conflict_resolved_for_execution": False,
        "execution_authorized": False,
    }


def _validate_slot_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError("slot_mapping_contract must be an explicit mapping")
    if value.get("mode") in {"identity", "14d_identity", "identity_by_dimension"}:
        raise PreflightError("14D identity-by-dimension action shortcut is forbidden")
    if value.get("derived_from_equal_dimension") is not False:
        raise PreflightError("equal action dimensions cannot establish slot semantics")
    expected = expected_slot_mapping_contract()
    if value != expected:
        raise PreflightError("slot mapping/order differs from the frozen named registry")
    return expected


def _validate_static_artifact_semantics(
    artifacts: Mapping[str, Mapping[str, str]],
    state_resolution: Any,
) -> dict[str, Any]:
    actor_directory = _validate_policy_artifact_layout(artifacts)
    config = _load_json(artifacts["checkpoint_config"]["path"], "checkpoint_config")
    train = _load_json(artifacts["train_config"]["path"], "train_config")
    pre = _load_json(artifacts["policy_preprocessor"]["path"], "policy_preprocessor")
    post = _load_json(artifacts["policy_postprocessor"]["path"], "policy_postprocessor")

    config_state_dim = _shape(
        config.get("input_features", {}).get("observation.state"),
        "checkpoint_config.input_features.observation.state",
    )
    config_action_dim = _shape(
        config.get("output_features", {}).get("action"),
        "checkpoint_config.output_features.action",
    )
    train_policy_state_dim = _shape(
        train.get("policy", {}).get("input_features", {}).get("observation.state"),
        "train_config.policy.input_features.observation.state",
    )
    train_policy_action_dim = _shape(
        train.get("policy", {}).get("output_features", {}).get("action"),
        "train_config.policy.output_features.action",
    )
    train_env_state_dim = _shape(
        train.get("env", {}).get("features", {}).get("agent_pos"),
        "train_config.env.features.agent_pos",
    )
    train_env_action_dim = _shape(
        train.get("env", {}).get("features", {}).get("action"),
        "train_config.env.features.action",
    )
    dataset_repo = train.get("dataset", {}).get("repo_id")
    if dataset_repo != SOURCE_DATASET_REPO:
        raise PreflightError("train_config dataset is not the frozen audited Aloha source")
    if train.get("dataset", {}).get("episodes", "missing") is not None:
        raise PreflightError("train_config must bind the audited all-episodes dataset scope")
    if config.get("repo_id") != "pepijn223/smolvla_robotwin":
        raise PreflightError("checkpoint_config repo_id is not the audited SmolVLA actor")
    if config.get("use_delta_joint_actions_aloha") is not False:
        raise PreflightError("checkpoint action mode differs from the audited absolute-qpos actor")
    if config_action_dim != (14,) or train_policy_action_dim != (14,) or train_env_action_dim != (14,):
        raise PreflightError("all declared action widths must be exactly 14")
    norm_map = config.get("normalization_mapping")
    if not isinstance(norm_map, Mapping) or norm_map.get("STATE") != "MEAN_STD" or norm_map.get("ACTION") != "MEAN_STD":
        raise PreflightError("checkpoint must use audited MEAN_STD state/action normalization")

    pre_step = _normalizer_step(pre, "normalizer_processor", "policy_preprocessor")
    post_step = _normalizer_step(post, "unnormalizer_processor", "policy_postprocessor")
    pre_features = pre_step.get("config", {}).get("features", {})
    post_features = post_step.get("config", {}).get("features", {})
    pre_declared_state_dim = _shape(pre_features.get("observation.state"), "preprocessor.observation.state")
    pre_declared_action_dim = _shape(pre_features.get("action"), "preprocessor.action")
    post_declared_action_dim = _shape(post_features.get("action"), "postprocessor.action")
    if pre_declared_action_dim != (14,) or post_declared_action_dim != (14,):
        raise PreflightError("pre/postprocessor action widths must be exactly 14")
    if Path(str(pre_step.get("state_file"))).name != Path(artifacts["preprocessor_stats"]["path"]).name:
        raise PreflightError("preprocessor stats filename is not bound by processor config")
    if Path(str(post_step.get("state_file"))).name != Path(artifacts["postprocessor_stats"]["path"]).name:
        raise PreflightError("postprocessor stats filename is not bound by processor config")

    pre_tensors = read_safetensors(artifacts["preprocessor_stats"]["path"])
    post_tensors = read_safetensors(artifacts["postprocessor_stats"]["path"])
    pre_stats = _require_stats(pre_tensors, ("action", "observation.state"), "preprocessor_stats")
    post_stats = _require_stats(post_tensors, ("action",), "postprocessor_stats")
    for statistic in ("count", "min", "max", "mean", "std"):
        if not np.array_equal(pre_stats["action"][statistic], post_stats["action"][statistic]):
            raise PreflightError(f"pre/post action {statistic} statistics differ")
    normalizer_state_dim = int(pre_stats["observation.state"]["mean"].shape[0])

    dimensions = {
        "checkpoint_policy_state_dim": config_state_dim[0],
        "train_policy_state_dim": train_policy_state_dim[0],
        "train_env_state_dim": train_env_state_dim[0],
        "preprocessor_declared_state_dim": pre_declared_state_dim[0],
        "normalizer_state_dim": normalizer_state_dim,
    }
    conflict = len(set(dimensions.values())) != 1
    if conflict:
        if state_resolution is None:
            raise StateDimensionConflict(
                "state width conflict (policy/preprocessor=6, env/stats=14) is blocked by default"
            )
        expected_resolution = expected_state_dimension_resolution()
        if state_resolution != expected_resolution:
            raise StateDimensionConflict(
                "state width conflict lacks the exact forward-only resolution contract"
            )
    elif state_resolution is not None:
        raise PreflightError("state_dimension_resolution supplied without an observed conflict")

    piper_body_raw = _yaml_mapping(
        artifacts["piper_body_config"]["path"], "piper_body_config"
    )
    aloha_body_raw = _yaml_mapping(
        artifacts["aloha_body_config"]["path"], "aloha_body_config"
    )
    piper_asset_layout = _validate_body_asset_layout(
        artifacts, piper_body_raw, "piper"
    )
    aloha_asset_layout = _validate_body_asset_layout(
        artifacts, aloha_body_raw, "aloha"
    )
    piper_body = _validate_body_config(piper_body_raw, "piper")
    aloha_body = _validate_body_config(aloha_body_raw, "aloha")
    piper_names = [f"joint{i}" for i in range(1, 8)]
    aloha_names = [f"fl_joint{i}" for i in range(1, 8)] + [f"fr_joint{i}" for i in range(1, 8)]
    piper_limits = _urdf_limits(artifacts["piper_urdf"]["path"], piper_names, "piper_urdf")
    aloha_limits = _urdf_limits(artifacts["aloha_urdf"]["path"], aloha_names, "aloha_urdf")
    _assert_limits(piper_limits, piper_names[:6], PIPER_ARM_LIMITS, "piper")
    _assert_limits(aloha_limits, aloha_names[:6], ALOHA_ARM_LIMITS, "aloha-left")
    _assert_limits(aloha_limits, aloha_names[7:13], ALOHA_ARM_LIMITS, "aloha-right")
    if not np.allclose(piper_limits["joint7"], (0.0, 0.04), rtol=0.0, atol=1e-9):
        raise PreflightError("piper URDF gripper limit differs from the frozen contract")

    return {
        "source_dataset": {
            "repo_id": SOURCE_DATASET_REPO,
            "robot_type": "aloha",
            "stats_frame_count": SOURCE_DATASET_FRAME_COUNT,
            "feature_names": list(ALOHA_FEATURE_NAMES),
        },
        "actor_directory": actor_directory,
        "dimensions": dimensions,
        "observed_state_dimension_conflict": conflict,
        "state_dimension_resolution": state_resolution,
        "piper_body": piper_body,
        "aloha_body": aloha_body,
        "piper_asset_layout": piper_asset_layout,
        "aloha_asset_layout": aloha_asset_layout,
        "piper_action_bounds": [[slot.lower, slot.upper] for slot in PIPER_ACTION_SLOTS],
        "quantile_statistics_available": any(
            "quantile" in name.casefold() for name in pre_tensors
        ),
    }


def validate_static_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the checkpoint/body contract without reading model outputs."""

    _exact_fields(
        value,
        {
            "format",
            "actor_id",
            "source_body",
            "target_body",
            "artifacts",
            "probe_artifacts",
            "slot_mapping_contract",
            "state_dimension_resolution",
            "capability_contract",
        },
        "preflight",
    )
    if value["format"] != FORMAT:
        raise PreflightError("unexpected preflight format")
    if value["actor_id"] != ACTOR_ID:
        raise PreflightError(f"actor_id must be exactly {ACTOR_ID}")
    if value["source_body"] != "aloha" or value["target_body"] != "piper":
        raise PreflightError("source/target bodies must be explicit Aloha -> Piper")
    capability = value["capability_contract"]
    expected_capability = {
        "fresh_inputs_allowed": False,
        "environment_step_allowed": False,
        "outcome_inputs_allowed": False,
        "execution_authorized": False,
        "transfer_claim_authorized": False,
        "maximum_authorization": AUTHORIZATION,
    }
    if capability != expected_capability:
        raise PreflightError("capability_contract must remain data-blind and forward-only")
    artifacts = authenticate_artifacts(value["artifacts"])
    slot_mapping = _validate_slot_mapping(value["slot_mapping_contract"])
    semantics = _validate_static_artifact_semantics(
        artifacts, value["state_dimension_resolution"]
    )
    return {
        "format": FORMAT,
        "actor_id": ACTOR_ID,
        "artifact_sha256": {name: artifact["sha256"] for name, artifact in artifacts.items()},
        "slot_mapping_contract_sha256": canonical_sha256(slot_mapping),
        "static_semantics": semantics,
        "authorization_ceiling": AUTHORIZATION,
        "environment_execution_authorized": False,
        "transfer_claim_authorized": False,
    }


def array_sha256(value: np.ndarray) -> str:
    """Hash an array together with its dtype and shape for contract receipts."""

    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def adapt_aloha_source_actions_to_piper_forward_interface(
    source_actions: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the explicit named ordinal, angle-preserving *offline* mapping.

    Equal width is deliberately not used to infer semantics.  Every source
    column is looked up through the frozen Aloha feature registry and copied to
    its explicitly qualified Piper side/ordinal slot.  Arm radians and the
    normalized gripper fraction are numerically preserved.  No kinematic
    transform, body-specific gripper scaling, controller call, or execution is
    performed.
    """

    source = np.asarray(source_actions)
    if source.shape != EXPECTED_CANDIDATE_SHAPE:
        raise PreflightError(
            f"Aloha source actions shape {source.shape} != {EXPECTED_CANDIDATE_SHAPE}"
        )
    if source.dtype.kind not in "fc" or source.dtype.kind == "c":
        raise PreflightError("Aloha source actions must be real floating-point values")
    if not np.all(np.isfinite(source)):
        raise PreflightError("Aloha source actions contain NaN or infinity")
    source_index_by_name = {
        name: index for index, name in enumerate(ALOHA_FEATURE_NAMES)
    }
    target = np.empty_like(source)
    applied: list[dict[str, Any]] = []
    for target_index, slot in enumerate(PIPER_ACTION_SLOTS):
        source_index = source_index_by_name[slot.source_feature_name]
        target[:, :, target_index] = source[:, :, source_index]
        applied.append(
            {
                "source_index": source_index,
                "source_feature_name": slot.source_feature_name,
                "source_joint_name": slot.source_joint_name,
                "target_index": target_index,
                "target_joint_name": slot.target_joint_name,
                "side": slot.side,
                "ordinal": slot.ordinal,
                "numeric_transform": slot.numeric_transform,
            }
        )
    return target, {
        "mode": "explicit_named_ordinal_angle_preserving_mapping",
        "identity_inferred_from_equal_dimension": False,
        "angle_values_preserved": True,
        "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False,
        "body_specific_gripper_scale_applied": False,
        "execution_authorized": False,
        "source_array_sha256": array_sha256(source),
        "mapped_array_sha256": array_sha256(target),
        "mapping": applied,
    }


def validate_candidate_actions(
    candidate_actions: Any,
    *,
    bounds: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Validate offline candidate outputs; no model or environment is invoked."""

    actions = np.asarray(candidate_actions)
    if actions.shape != EXPECTED_CANDIDATE_SHAPE:
        raise PreflightError(
            f"candidate actions shape {actions.shape} != {EXPECTED_CANDIDATE_SHAPE}"
        )
    if actions.dtype.kind not in "fc" or actions.dtype.kind == "c":
        raise PreflightError("candidate actions must be real floating-point values")
    if not np.all(np.isfinite(actions)):
        raise PreflightError("candidate actions contain NaN or infinity")
    resolved_bounds = (
        tuple((slot.lower, slot.upper) for slot in PIPER_ACTION_SLOTS)
        if bounds is None
        else tuple(tuple(float(x) for x in pair) for pair in bounds)
    )
    if len(resolved_bounds) != 14 or any(len(pair) != 2 for pair in resolved_bounds):
        raise PreflightError("Piper action bounds must contain exactly 14 pairs")
    violations: list[dict[str, Any]] = []
    for slot, (lower, upper) in enumerate(resolved_bounds):
        values = actions[:, :, slot]
        observed_min = float(values.min())
        observed_max = float(values.max())
        if observed_min < lower or observed_max > upper:
            violations.append(
                {
                    "slot": slot,
                    "name": PIPER_ACTION_SLOTS[slot].target_joint_name,
                    "allowed": [lower, upper],
                    "observed": [observed_min, observed_max],
                }
            )
    if violations:
        first = violations[0]
        raise PreflightError(
            f"Piper action limit violation at slot {first['slot']} ({first['name']})"
        )
    hashes = [array_sha256(actions[index]) for index in range(4)]
    if len(set(hashes)) != 4:
        raise PreflightError("candidate set is degenerate; all four candidates must be distinct")
    max_from_baseline = max(
        float(np.max(np.abs(actions[index] - actions[0]))) for index in range(1, 4)
    )
    if max_from_baseline <= 0.0:
        raise PreflightError("candidate set has no action variation from candidate0")
    return {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "candidate_sha256": hashes,
        "all_candidates_distinct": True,
        "max_abs_delta_from_candidate0": max_from_baseline,
        "piper_limits_satisfied": True,
    }


def validate_shared_prefix(
    shared_prefixes: Any, *, claimed_sha256: str
) -> dict[str, Any]:
    """Require four candidate prefixes to be finite and bit-exact identical."""

    prefixes = np.asarray(shared_prefixes)
    if prefixes.ndim < 2 or prefixes.shape[0] != 4 or prefixes.size == 0:
        raise PreflightError("shared prefixes must have shape [4, ...] with non-empty payload")
    if prefixes.dtype.kind not in "biuf":
        raise PreflightError("shared prefixes must be a numeric or boolean array")
    if prefixes.dtype.kind in "f" and not np.all(np.isfinite(prefixes)):
        raise PreflightError("shared prefixes contain NaN or infinity")
    hashes = [array_sha256(prefixes[index]) for index in range(4)]
    if len(set(hashes)) != 1:
        raise PreflightError("candidate prefixes are not bit-exact identical")
    if not _is_sha256(claimed_sha256) or claimed_sha256 != hashes[0]:
        raise PreflightError("shared prefix SHA256 claim does not match the array")
    return {
        "shape": list(prefixes.shape),
        "dtype": str(prefixes.dtype),
        "bit_exact_across_candidates": True,
        "shared_prefix_sha256": hashes[0],
    }


def _load_authenticated_npy(
    artifact: Mapping[str, str], name: str
) -> np.ndarray:
    path = Path(artifact["path"])
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"probe_artifacts.{name} is not a safe NumPy array") from exc


def _validate_forward_probe_receipt(
    receipt: Mapping[str, Any],
    *,
    probe_artifacts: Mapping[str, Mapping[str, str]],
    actor_directory: str,
    candidate_actions: np.ndarray,
    shared_prefixes: np.ndarray,
) -> dict[str, Any]:
    """Bind the CUDA producer receipt to the exact authenticated offline arrays."""

    fixed = {
        "schema_version": 1,
        "experiment_type": "interface_smoke_not_task_success",
        "candidate_generator": "native_smolvla_flow_matching_explicit_noise",
        "preprocessing": "checkpoint_preprocessor_and_postprocessor",
        "candidate_count": 4,
        "candidate_shape": list(EXPECTED_CANDIDATE_SHAPE),
        "native_multi_candidate_verified": True,
        "identical_candidate_pairs": 0,
        "task_success_claimed": False,
        "runtime_observation_state_dim": 14,
        "state_dimension_override_used": True,
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise PreflightError(
                f"forward probe receipt {field} does not match the frozen contract"
            )
    processed_shape = receipt.get("runtime_preprocessed_observation_state_shape")
    if (
        not isinstance(processed_shape, list)
        or len(processed_shape) < 1
        or processed_shape[-1] != 14
        or any(type(dim) is not int or dim <= 0 for dim in processed_shape)
    ):
        raise PreflightError(
            "forward probe receipt lacks a valid preprocessed 14D observation.state shape"
        )
    if receipt.get("model_config_observation_state_dim") != 6:
        raise PreflightError("forward probe receipt must retain the checkpoint 6D declaration")
    if Path(str(receipt.get("model_path", ""))).resolve(strict=False) != Path(
        actor_directory
    ):
        raise PreflightError("forward probe model_path differs from authenticated actor")
    image_sources = receipt.get("image_source")
    expected_image = probe_artifacts["source_image"]["path"]
    if image_sources != [expected_image]:
        raise PreflightError("forward probe image_source differs from authenticated source image")

    outputs = receipt.get("array_outputs")
    if not isinstance(outputs, Mapping):
        raise PreflightError("forward probe receipt lacks array_outputs")
    expected_outputs = {
        "candidate_actions": probe_artifacts["candidate_actions"]["path"],
        "candidate_actions_array_sha256": array_sha256(candidate_actions),
        "shared_prefixes": probe_artifacts["shared_prefixes"]["path"],
        "shared_prefix_array_sha256": array_sha256(shared_prefixes[0]),
    }
    if outputs != expected_outputs:
        raise PreflightError("forward probe array_outputs do not bind the authenticated arrays")

    hook = receipt.get("etsf_shared_state_hook")
    if not isinstance(hook, Mapping):
        raise PreflightError("forward probe receipt lacks shared-state hook evidence")
    required_hook = {
        "shape": [4, 960],
        "feature_dim": 960,
        "hook_calls_per_candidate": [1, 1, 1, 1],
        "max_abs_delta_from_candidate_0": [0.0, 0.0, 0.0, 0.0],
        "bit_exact_across_noise_candidates": True,
        "candidate_specific_expert_hidden_saved": False,
        "status": "verified",
    }
    for field, expected in required_hook.items():
        if hook.get(field) != expected:
            raise PreflightError(
                f"forward probe shared-state hook {field} differs from the contract"
            )
    if list(shared_prefixes.shape) != [4, 960]:
        raise PreflightError("authenticated shared prefixes must have exact shape [4,960]")
    return {
        "receipt_sha256": probe_artifacts["forward_probe_receipt"]["sha256"],
        "runtime_observation_state_dim": 14,
        "runtime_preprocessed_observation_state_shape": processed_shape,
        "checkpoint_declared_state_dim": 6,
        "state_dimension_conflict_retained": True,
        "proper_checkpoint_pre_and_postprocessing": True,
        "authenticated_source_image_sha256": probe_artifacts["source_image"]["sha256"],
        "authenticated_candidate_file_sha256": probe_artifacts["candidate_actions"]["sha256"],
        "authenticated_shared_prefix_file_sha256": probe_artifacts["shared_prefixes"]["sha256"],
    }


def run_preflight(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the complete data-blind check and emit a forward-only receipt."""

    static = validate_static_preflight(manifest)
    probe_artifacts = authenticate_probe_artifacts(manifest["probe_artifacts"])
    candidate_actions = _load_authenticated_npy(
        probe_artifacts["candidate_actions"], "candidate_actions"
    )
    shared_prefixes = _load_authenticated_npy(
        probe_artifacts["shared_prefixes"], "shared_prefixes"
    )
    mapped_actions, mapping_check = adapt_aloha_source_actions_to_piper_forward_interface(
        candidate_actions
    )
    action_check = validate_candidate_actions(
        mapped_actions,
        bounds=static["static_semantics"]["piper_action_bounds"],
    )
    prefix_check = validate_shared_prefix(
        shared_prefixes, claimed_sha256=array_sha256(shared_prefixes[0])
    )
    probe_receipt = _load_json(
        probe_artifacts["forward_probe_receipt"]["path"], "forward_probe_receipt"
    )
    probe_check = _validate_forward_probe_receipt(
        probe_receipt,
        probe_artifacts=probe_artifacts,
        actor_directory=static["static_semantics"]["actor_directory"],
        candidate_actions=candidate_actions,
        shared_prefixes=shared_prefixes,
    )
    return {
        "format": FORMAT,
        "status": "passed_forward_only",
        "actor_id": ACTOR_ID,
        "authorization": AUTHORIZATION,
        "environment_execution_authorized": False,
        "transfer_claim_authorized": False,
        "data_blind": True,
        "static_contract": static,
        "candidate_validation": action_check,
        "action_mapping_validation": mapping_check,
        "shared_prefix_validation": prefix_check,
        "forward_probe_validation": probe_check,
        "implementation_sha256": file_sha256(Path(__file__)),
        "reason": (
            "offline forward outputs satisfy interface/safety checks; this is not "
            "execution authorization or evidence of Piper task transfer"
        ),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_npy(path: Path, name: str) -> np.ndarray:
    resolved = reject_fresh_path(path, name)
    if resolved.suffix.casefold() != ".npy" or not resolved.is_file():
        raise PreflightError(f"{name} must be an existing .npy file")
    try:
        return np.load(resolved, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"{name} is not a safe NumPy array") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = reject_fresh_path(args.manifest, "manifest")
    output_path = reject_fresh_path(args.output, "output")
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = _load_json(str(manifest_path), "manifest")
    result = run_preflight(manifest)
    atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "actor_id": ACTOR_ID,
                "authorization": AUTHORIZATION,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
