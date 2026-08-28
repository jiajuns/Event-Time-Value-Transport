#!/usr/bin/env python3
"""Materialize causal, actor-visible event-observer supervision.

The materializer is the only component in this pipeline that may read
privileged simulator poses.  Poses are used transiently to derive labels at
the *current* query.  They are never written to the resulting NPZ files.  All
model inputs are actor-visible and every sample is bound to a content-addressed
source group, branch, actor and query.

The train/calibration/validation membership is frozen by a signed request
before any HDF5 container is opened.  Membership is expressed in logical reset
groups, not rows, so no branch or continuation query can cross a split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from etsf_schema6_pose_quality import GROUP_NAME as POSE_QUALITY_GROUP
from etsf_schema6_pose_quality import validate_pose_quality_v6
from smolvla_piper_causal_event_observer_v1 import (
    EXPECTED_EVENTS,
    EXPECTED_PREDICATES,
    MAX_HISTORY_STEPS,
    STATE_DIM,
    build_causal_history_window,
    causal_history_contract,
)


FORMAT = "etsf_smolvla_piper_causal_event_observer_dataset_v1"
SPLIT_FORMAT = "etsf_smolvla_piper_causal_event_observer_split_v1"
REQUEST_FORMAT = (
    "etsf_smolvla_piper_causal_event_observer_materialization_request_v1"
)
STATUS = "complete_actor_visible_causal_supervision_content_addressed"
REQUEST_STATUS = "frozen_before_hdf_access"

HISTORY_STEPS = MAX_HISTORY_STEPS
PROPRIO_DIM = 14
EVENT_NAMES = EXPECTED_EVENTS
PREDICATE_NAMES = EXPECTED_PREDICATES
SPLIT_NAMES = ("train", "calibration", "validation")
ARRAY_NAMES = (
    "history",
    "history_mask",
    "proprio",
    "event_label",
    "predicate_label",
    "actor_index",
    "current_query_index",
    "query_step",
    "prior_execution_present",
    "prior_executed_control_steps",
    "prior_action_sha256",
    "sample_id",
    "logical_group_id",
    "branch_id",
    "source_file_sha256",
)
SHA_CHARS = frozenset("0123456789abcdef")
PROTECTED_HDF_COMPONENTS = frozenset(
    {"fresh", "confirmation", "evaluation", "evaluation400", "formal_target_validation"}
)


class ObserverDatasetContractError(RuntimeError):
    """The source, split, causality, label or content-address contract failed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _exact_mapping(value: Any, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ObserverDatasetContractError(f"{role} fields changed")
    return value


def _load_json(path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ObserverDatasetContractError(f"{role} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObserverDatasetContractError(f"{role} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise ObserverDatasetContractError(f"{role} must be a JSON object")
    return value


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            if current.is_symlink():
                raise ObserverDatasetContractError(
                    f"{role} path contains a symbolic link"
                )
        except OSError as error:
            raise ObserverDatasetContractError(f"{role} path is unavailable") from error


def _reject_protected_hdf_path(path: Path, role: str) -> None:
    for component in path.parts:
        lowered = component.casefold()
        if (
            lowered in PROTECTED_HDF_COMPONENTS
            or "confirmation" in lowered
            or lowered.startswith("fresh_")
            or lowered.startswith("evaluation400_")
        ):
            raise ObserverDatasetContractError(
                f"{role} identifies protected fresh/formal/evaluation data"
            )


def _resolve_file(raw: Any, *, base: Path, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ObserverDatasetContractError(f"{role} path is invalid")
    path = Path(raw)
    unresolved = path if path.is_absolute() else base / path
    _reject_symlink_components(unresolved, role)
    path = unresolved.resolve()
    if path.is_symlink() or not path.is_file():
        raise ObserverDatasetContractError(f"{role} is not a regular file")
    return path


def _resolve_directory(raw: Any, *, base: Path, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ObserverDatasetContractError(f"{role} path is invalid")
    path = Path(raw)
    unresolved = path if path.is_absolute() else base / path
    _reject_symlink_components(unresolved, role)
    path = unresolved.resolve()
    if path.is_symlink() or not path.is_dir():
        raise ObserverDatasetContractError(f"{role} is not a directory")
    return path


def schema5_logical_group_id(
    source_name: str, task: str, body: str, policy: str, requested_seed: int
) -> str:
    if any(not isinstance(item, str) or not item for item in (source_name, task, body, policy)):
        raise ObserverDatasetContractError("schema5 logical group components are invalid")
    if not _strict_int(requested_seed):
        raise ObserverDatasetContractError("schema5 requested seed is invalid")
    return (
        f"{source_name}/schema5/{task}/{body}/{policy}/"
        f"requested_seed/{requested_seed}"
    )


def schema6_logical_group_id(source_name: str, native_logical_group_id: str) -> str:
    if (
        not isinstance(source_name, str)
        or not source_name
        or not isinstance(native_logical_group_id, str)
        or not native_logical_group_id
    ):
        raise ObserverDatasetContractError("schema6 logical group components are invalid")
    return f"{source_name}/schema6/{native_logical_group_id}"


@dataclass(frozen=True)
class ActorDescriptor:
    actor_index: int
    actor_name: str
    policy_family: str
    body: str
    policy: str
    state_feature_source_sha256: str


@dataclass(frozen=True)
class SourceDescriptor:
    source_name: str
    schema_version: int
    manifest_path: Path
    manifest_file_sha256: str
    manifest_logical_sha256: str
    group_root: Path
    actor: ActorDescriptor


@dataclass(frozen=True)
class GroupDescriptor:
    source: SourceDescriptor
    logical_group_id: str
    native_logical_group_id: str
    requested_seed: int
    resolved_seed: int
    task: str
    body: str
    policy: str
    path: Path
    expected_file_sha256: str


@dataclass(frozen=True)
class FrozenRequest:
    path: Path
    file_sha256: str
    logical_sha256: str
    event_spec_path: Path
    event_spec_file_sha256: str
    event_spec: dict[str, Any]
    actors: tuple[ActorDescriptor, ...]
    sources: tuple[SourceDescriptor, ...]
    split_groups: dict[str, tuple[GroupDescriptor, ...]]


def _validate_actor_registry(raw: Any) -> tuple[ActorDescriptor, ...]:
    if not isinstance(raw, list) or not raw:
        raise ObserverDatasetContractError("actor registry must be non-empty")
    result: list[ActorDescriptor] = []
    names: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for index, value in enumerate(raw):
        item = _exact_mapping(
            value,
            {
                "actor_name",
                "policy_family",
                "body",
                "policy",
                "state_feature_source_sha256",
            },
            "request actor",
        )
        strings = [item[key] for key in ("actor_name", "policy_family", "body", "policy")]
        if any(not isinstance(text, str) or not text or text.strip() != text for text in strings):
            raise ObserverDatasetContractError("request actor strings are invalid")
        if item["actor_name"] in names or (item["body"], item["policy"]) in identities:
            raise ObserverDatasetContractError("actor registry contains duplicate identity")
        if not _is_sha(item["state_feature_source_sha256"]):
            raise ObserverDatasetContractError("actor state-feature source SHA is invalid")
        names.add(str(item["actor_name"]))
        identities.add((str(item["body"]), str(item["policy"])))
        result.append(
            ActorDescriptor(
                actor_index=index,
                actor_name=str(item["actor_name"]),
                policy_family=str(item["policy_family"]),
                body=str(item["body"]),
                policy=str(item["policy"]),
                state_feature_source_sha256=str(item["state_feature_source_sha256"]),
            )
        )
    return tuple(result)


def _manifest_logical_sha(value: Mapping[str, Any], schema_version: int) -> str:
    if schema_version == 5:
        return canonical_sha256(value)
    if schema_version != 6:
        raise ObserverDatasetContractError("only schema5 and schema6 are supported")
    logical = dict(value)
    recorded = logical.pop("manifest_sha256", None)
    computed = canonical_sha256(logical)
    if not _is_sha(recorded) or recorded != computed:
        raise ObserverDatasetContractError("schema6 manifest logical SHA is invalid")
    return computed


def _scan_source_manifest(
    raw: Mapping[str, Any], *, request_root: Path, actors: Mapping[str, ActorDescriptor]
) -> tuple[SourceDescriptor, dict[str, GroupDescriptor], dict[str, Any]]:
    item = _exact_mapping(
        raw,
        {
            "source_name",
            "schema_version",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_logical_sha256",
            "group_root",
            "actor_name",
        },
        "request source",
    )
    source_name = item["source_name"]
    schema_version = item["schema_version"]
    actor_name = item["actor_name"]
    if not isinstance(source_name, str) or not source_name or source_name.strip() != source_name:
        raise ObserverDatasetContractError("source name is invalid")
    if schema_version not in (5, 6) or actor_name not in actors:
        raise ObserverDatasetContractError("source schema/actor binding is invalid")
    if not _is_sha(item["manifest_file_sha256"]) or not _is_sha(
        item["manifest_logical_sha256"]
    ):
        raise ObserverDatasetContractError("source manifest SHA is invalid")
    manifest_path = _resolve_file(
        item["manifest_path"], base=request_root, role=f"source {source_name} manifest"
    )
    group_root = _resolve_directory(
        item["group_root"], base=request_root, role=f"source {source_name} group root"
    )
    actual_file_sha = file_sha256(manifest_path)
    if actual_file_sha != item["manifest_file_sha256"]:
        raise ObserverDatasetContractError("source manifest file SHA changed")
    manifest = _load_json(manifest_path, f"source {source_name} manifest")
    logical_sha = _manifest_logical_sha(manifest, int(schema_version))
    if logical_sha != item["manifest_logical_sha256"]:
        raise ObserverDatasetContractError("source manifest logical SHA changed")
    source = SourceDescriptor(
        source_name=source_name,
        schema_version=int(schema_version),
        manifest_path=manifest_path,
        manifest_file_sha256=actual_file_sha,
        manifest_logical_sha256=logical_sha,
        group_root=group_root,
        actor=actors[str(actor_name)],
    )
    if schema_version == 5:
        groups = _scan_schema5_groups(source, manifest)
    else:
        groups = _scan_schema6_groups(source, manifest)
    return source, groups, manifest


def _group_path(raw: Any, source: SourceDescriptor, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ObserverDatasetContractError(f"{role} path is invalid")
    path = Path(raw)
    unresolved = path if path.is_absolute() else source.group_root / path
    _reject_symlink_components(unresolved, role)
    resolved = unresolved.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ObserverDatasetContractError(f"{role} is not a regular file")
    if not path.is_absolute() and source.group_root not in resolved.parents:
        raise ObserverDatasetContractError(f"{role} escapes its group root")
    return resolved


def _scan_schema5_groups(
    source: SourceDescriptor, manifest: Mapping[str, Any]
) -> dict[str, GroupDescriptor]:
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version") != 5
        or manifest.get("hidden_dim") != STATE_DIM
        or manifest.get("action_dim") != PROPRIO_DIM
        or manifest.get("event_vocab") != list(EVENT_NAMES)
        or not _is_sha(manifest.get("event_spec_sha256"))
    ):
        raise ObserverDatasetContractError("schema5 collector manifest is incompatible")
    task, body, policy = (manifest.get(key) for key in ("task", "body", "policy"))
    state_contract = manifest.get("shared_state_contract")
    if (
        any(not isinstance(value, str) or not value for value in (task, body, policy))
        or (body, policy) != (source.actor.body, source.actor.policy)
    ):
        raise ObserverDatasetContractError("schema5 actor body/policy differs")
    if not isinstance(state_contract, Mapping) or state_contract.get(
        "calibration_id"
    ) != (
        source.actor.state_feature_source_sha256
    ):
        raise ObserverDatasetContractError("schema5 state-feature source differs")
    rows = manifest.get("groups")
    if not isinstance(rows, list) or not rows:
        raise ObserverDatasetContractError("schema5 manifest has no groups")
    result: dict[str, GroupDescriptor] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ObserverDatasetContractError("schema5 group identity is invalid")
        requested = raw.get("seed")
        resolved = raw.get("resolved_seed")
        if (
            raw.get("index") != index
            or not _strict_int(requested)
            or not _strict_int(resolved)
            or raw.get("status") not in ("collected", "existing")
        ):
            raise ObserverDatasetContractError("schema5 group identity changed")
        logical = schema5_logical_group_id(
            source.source_name, str(task), str(body), str(policy), int(requested)
        )
        if logical in result:
            raise ObserverDatasetContractError("duplicate schema5 logical group")
        path = _group_path(raw.get("path"), source, f"schema5 group {logical}")
        result[logical] = GroupDescriptor(
            source=source,
            logical_group_id=logical,
            native_logical_group_id=str(requested),
            requested_seed=int(requested),
            resolved_seed=int(resolved),
            task=str(task),
            body=str(body),
            policy=str(policy),
            path=path,
            expected_file_sha256="",
        )
    return result


def _scan_schema6_groups(
    source: SourceDescriptor, manifest: Mapping[str, Any]
) -> dict[str, GroupDescriptor]:
    if (
        manifest.get("format") != "etsf_smolvla_piper_schema6_training_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("fresh_inputs_used") is not False
        or manifest.get("sealed_test_labels_disclosed") is not False
        or not _is_sha(manifest.get("event_spec_sha256"))
    ):
        raise ObserverDatasetContractError("schema6 training manifest is incompatible")
    rows = manifest.get("groups")
    if not isinstance(rows, list) or not rows:
        raise ObserverDatasetContractError("schema6 manifest has no groups")
    result: dict[str, GroupDescriptor] = {}
    for raw in rows:
        item = _exact_mapping(
            raw,
            {
                "logical_group_id",
                "requested_seed",
                "resolved_seed",
                "task",
                "body",
                "policy",
                "path",
                "file_sha256",
            },
            "schema6 manifest group",
        )
        if (
            not isinstance(item["logical_group_id"], str)
            or not item["logical_group_id"]
            or not _strict_int(item["requested_seed"])
            or not _strict_int(item["resolved_seed"])
            or not isinstance(item["task"], str)
            or not item["task"]
            or (item["body"], item["policy"])
            != (source.actor.body, source.actor.policy)
            or not _is_sha(item["file_sha256"])
        ):
            raise ObserverDatasetContractError("schema6 group identity is invalid")
        logical = schema6_logical_group_id(
            source.source_name, str(item["logical_group_id"])
        )
        if logical in result:
            raise ObserverDatasetContractError("duplicate schema6 logical group")
        result[logical] = GroupDescriptor(
            source=source,
            logical_group_id=logical,
            native_logical_group_id=str(item["logical_group_id"]),
            requested_seed=int(item["requested_seed"]),
            resolved_seed=int(item["resolved_seed"]),
            task=str(item["task"]),
            body=str(item["body"]),
            policy=str(item["policy"]),
            path=_group_path(item["path"], source, f"schema6 group {logical}"),
            expected_file_sha256=str(item["file_sha256"]),
        )
    return result


def freeze_request(request_path: Path) -> FrozenRequest:
    """Freeze all logical memberships before opening any HDF5 container."""

    path = request_path.resolve()
    value = _load_json(path, "observer materialization request")
    item = _exact_mapping(
        value,
        {
            "format",
            "status",
            "event_spec",
            "actors",
            "sources",
            "splits",
            "split_unit",
            "split_leakage_allowed",
            "privileged_label_source_available_to_model_inputs",
            "future_query_features_available_to_model_inputs",
            "request_sha256",
        },
        "observer materialization request",
    )
    logical = dict(item)
    recorded = logical.pop("request_sha256")
    if (
        item["format"] != REQUEST_FORMAT
        or item["status"] != REQUEST_STATUS
        or item["split_unit"] != "logical_reset_group"
        or item["split_leakage_allowed"] is not False
        or item["privileged_label_source_available_to_model_inputs"] is not False
        or item["future_query_features_available_to_model_inputs"] is not False
        or not _is_sha(recorded)
        or recorded != canonical_sha256(logical)
    ):
        raise ObserverDatasetContractError("materialization request is invalid")
    request_root = path.parent
    event_record = _exact_mapping(
        item["event_spec"], {"path", "file_sha256"}, "request event spec"
    )
    if not _is_sha(event_record["file_sha256"]):
        raise ObserverDatasetContractError("event spec SHA is invalid")
    event_path = _resolve_file(
        event_record["path"], base=request_root, role="event specification"
    )
    if file_sha256(event_path) != event_record["file_sha256"]:
        raise ObserverDatasetContractError("event specification file SHA changed")
    event_spec = _load_json(event_path, "event specification")
    actors = _validate_actor_registry(item["actors"])
    actor_by_name = {actor.actor_name: actor for actor in actors}
    raw_sources = item["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ObserverDatasetContractError("request sources must be non-empty")
    sources: list[SourceDescriptor] = []
    groups: dict[tuple[str, str], GroupDescriptor] = {}
    source_manifests: dict[str, Mapping[str, Any]] = {}
    for raw_source in raw_sources:
        source, discovered, manifest = _scan_source_manifest(
            raw_source, request_root=request_root, actors=actor_by_name
        )
        if source.source_name in source_manifests:
            raise ObserverDatasetContractError("duplicate request source name")
        sources.append(source)
        source_manifests[source.source_name] = manifest
        for logical_id, descriptor in discovered.items():
            groups[(source.source_name, logical_id)] = descriptor
    splits = _exact_mapping(item["splits"], set(SPLIT_NAMES), "request splits")
    frozen: dict[str, tuple[GroupDescriptor, ...]] = {}
    seen: set[tuple[str, str]] = set()
    for split_name in SPLIT_NAMES:
        raw_rows = splits[split_name]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ObserverDatasetContractError(f"{split_name} split must be non-empty")
        selected: list[GroupDescriptor] = []
        for raw_ref in raw_rows:
            ref = _exact_mapping(
                raw_ref,
                {"source_name", "logical_group_id", "source_file_sha256"},
                f"{split_name} group reference",
            )
            key = (ref["source_name"], ref["logical_group_id"])
            if (
                not isinstance(key[0], str)
                or not isinstance(key[1], str)
                or not _is_sha(ref["source_file_sha256"])
                or key not in groups
                or key in seen
            ):
                raise ObserverDatasetContractError(
                    "split group is missing, duplicated, or lacks a valid source SHA"
                )
            descriptor = groups[key]
            _reject_protected_hdf_path(
                descriptor.path, f"{split_name} source group"
            )
            manifest_sha = descriptor.expected_file_sha256
            if manifest_sha and manifest_sha != ref["source_file_sha256"]:
                raise ObserverDatasetContractError(
                    "request group SHA differs from source manifest"
                )
            selected.append(
                GroupDescriptor(
                    **{
                        **descriptor.__dict__,
                        "expected_file_sha256": str(ref["source_file_sha256"]),
                    }
                )
            )
            seen.add(key)
        if {descriptor.source.actor.actor_name for descriptor in selected} != set(
            actor_by_name
        ):
            raise ObserverDatasetContractError(
                f"{split_name} split lacks support for a registered actor"
            )
        frozen[split_name] = tuple(selected)
    event_sha = str(event_record["file_sha256"])
    for source in sources:
        manifest = source_manifests[source.source_name]
        if manifest.get("event_spec_sha256") != event_sha:
            raise ObserverDatasetContractError("source event-spec binding differs")
    return FrozenRequest(
        path=path,
        file_sha256=file_sha256(path),
        logical_sha256=str(recorded),
        event_spec_path=event_path,
        event_spec_file_sha256=event_sha,
        event_spec=event_spec,
        actors=actors,
        sources=tuple(sources),
        split_groups=frozen,
    )


def _decode_strings(value: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value]


def _task_calibration(event_spec: Mapping[str, Any], task: str) -> dict[str, Any]:
    calibration_root = event_spec.get("calibration")
    if not isinstance(calibration_root, Mapping) or task not in calibration_root:
        raise ObserverDatasetContractError("event spec lacks task calibration")
    calibration = calibration_root[task]
    required = {
        "moving",
        "delta_move",
        "delta_z",
        "tau_d",
        "tau_motion",
        "stationary_steps",
    }
    if not isinstance(calibration, Mapping) or not required.issubset(calibration):
        raise ObserverDatasetContractError("event calibration fields are incomplete")
    if ("anchor" in calibration) == ("centers" in calibration):
        raise ObserverDatasetContractError(
            "event calibration must define exactly one goal representation"
        )
    for key in ("delta_move", "delta_z", "tau_d", "tau_motion"):
        value = calibration[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ObserverDatasetContractError("event calibration threshold is invalid")
    if not _strict_int(calibration["stationary_steps"], minimum=1):
        raise ObserverDatasetContractError("stationary_steps is invalid")
    return dict(calibration)


def _derive_current_labels(
    *,
    poses: np.ndarray,
    object_names: Sequence[str],
    current_step: int,
    terminal_success: bool,
    calibration: Mapping[str, Any],
) -> tuple[np.int64, np.ndarray]:
    """Derive labels using only pose prefix through the current query.

    The terminal success flag is consulted only when ``current_step`` is the
    final available snapshot; it can never make an earlier input positive.
    """

    values = np.asarray(poses, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[2] != 7
        or not np.isfinite(values).all()
        or not _strict_int(current_step)
        or current_step >= values.shape[0]
    ):
        raise ObserverDatasetContractError("privileged pose label source is invalid")
    names = [str(name) for name in object_names]
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ObserverDatasetContractError("object registry does not match poses")
    moving = str(calibration["moving"])
    if moving not in names:
        raise ObserverDatasetContractError("moving object is absent")
    prefix = values[: current_step + 1]
    position = prefix[:, names.index(moving), :3]
    step_motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    cumulative_motion = np.cumsum(step_motion)
    anchor_name = calibration.get("anchor")
    if anchor_name is not None:
        anchor_name = str(anchor_name)
        if anchor_name not in names:
            raise ObserverDatasetContractError("goal anchor is absent")
        anchor = prefix[:, names.index(anchor_name), :3]
        offset = np.asarray(calibration.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ObserverDatasetContractError("goal offset is invalid")
        goal_distance = np.linalg.norm(position - anchor - offset, axis=1)
    else:
        centers = np.asarray(calibration["centers"], dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ObserverDatasetContractError("goal centers are invalid")
        goal_distance = np.linalg.norm(position[:, None] - centers[None], axis=2).min(axis=1)
    moved = bool(cumulative_motion[-1] >= float(calibration["delta_move"]))
    lifted = bool(position[-1, 2] >= position[0, 2] + float(calibration["delta_z"]))
    near_goal = bool(goal_distance[-1] <= float(calibration["tau_d"]))
    instant_stationary = (goal_distance <= float(calibration["tau_d"])) & (
        step_motion <= float(calibration["tau_motion"])
    )
    width = int(calibration["stationary_steps"])
    stationary = bool(
        len(instant_stationary) >= width and instant_stationary[-width:].all()
    )
    success = bool(terminal_success and current_step == values.shape[0] - 1)
    predicates = np.asarray(
        [moved, lifted, near_goal, stationary, success], dtype=np.float32
    )
    if success:
        event = EVENT_NAMES.index("eK")
    elif stationary:
        event = EVENT_NAMES.index("e4")
    elif near_goal:
        event = EVENT_NAMES.index("e3")
    elif moved or lifted:
        event = EVENT_NAMES.index("e12")
    else:
        event = EVENT_NAMES.index("e0")
    return np.int64(event), predicates


def _valid_prefix_mask(mask: np.ndarray, width: int) -> int:
    value = np.asarray(mask, dtype=np.bool_)
    if value.shape != (width,):
        raise ObserverDatasetContractError("executed action mask shape changed")
    length = int(value.sum())
    if not np.array_equal(value, np.arange(width) < length):
        raise ObserverDatasetContractError("executed action mask is not a prefix")
    return length


def _sample(
    *,
    descriptor: GroupDescriptor,
    source_sha: str,
    branch_id: str,
    hidden_prefix: np.ndarray,
    proprio: np.ndarray,
    event_label: np.int64,
    predicate_label: np.ndarray,
    current_query_index: int,
    query_step: int,
    prior_execution_present: bool,
    prior_executed_control_steps: int,
    prior_action_sha256: str,
) -> dict[str, Any]:
    history, history_mask = build_causal_history_window(
        np.asarray(hidden_prefix, dtype=np.float32)
    )
    state = np.asarray(proprio, dtype=np.float32)
    if state.shape != (PROPRIO_DIM,) or not np.isfinite(state).all():
        raise ObserverDatasetContractError("actor-visible proprio is invalid")
    if current_query_index < 0 or query_step < 0:
        raise ObserverDatasetContractError("query identity is invalid")
    if prior_execution_present:
        if prior_executed_control_steps < 1 or not _is_sha(prior_action_sha256):
            raise ObserverDatasetContractError("past execution receipt is unbound")
    elif prior_executed_control_steps != 0 or prior_action_sha256 != "":
        raise ObserverDatasetContractError("root query has a fabricated execution receipt")
    identity = {
        "source_name": descriptor.source.source_name,
        "source_file_sha256": source_sha,
        "logical_group_id": descriptor.logical_group_id,
        "branch_id": branch_id,
        "current_query_index": current_query_index,
        "actor_name": descriptor.source.actor.actor_name,
        "prior_action_sha256": prior_action_sha256,
    }
    return {
        "history": history,
        "history_mask": history_mask,
        "proprio": state,
        "event_label": event_label,
        "predicate_label": np.asarray(predicate_label, dtype=np.float32),
        "actor_index": np.int64(descriptor.source.actor.actor_index),
        "current_query_index": np.int64(current_query_index),
        "query_step": np.int64(query_step),
        "prior_execution_present": np.bool_(prior_execution_present),
        "prior_executed_control_steps": np.int64(prior_executed_control_steps),
        "prior_action_sha256": prior_action_sha256,
        "sample_id": canonical_sha256(identity),
        "logical_group_id": descriptor.logical_group_id,
        "branch_id": branch_id,
        "source_file_sha256": source_sha,
    }


def _materialize_schema5_group(
    descriptor: GroupDescriptor,
    *,
    source_sha: str,
    event_spec: Mapping[str, Any],
    event_spec_sha256: str,
) -> list[dict[str, Any]]:
    calibration = _task_calibration(event_spec, descriptor.task)
    samples: list[dict[str, Any]] = []
    with h5py.File(descriptor.path, "r") as handle:
        attrs = handle.attrs
        if (
            int(attrs.get("schema_version", -1)) != 5
            or str(attrs.get("task", "")) != descriptor.task
            or str(attrs.get("body", "")) != descriptor.body
            or str(attrs.get("policy", "")) != descriptor.policy
            or int(attrs.get("requested_seed", -1)) != descriptor.requested_seed
            or int(attrs.get("resolved_seed", -1)) != descriptor.resolved_seed
            or str(attrs.get("event_spec_sha256", "")) != event_spec_sha256
            or str(attrs.get("shared_state_contract_id", ""))
            != descriptor.source.actor.state_feature_source_sha256
            or bool(attrs.get("candidate_hidden_forbidden", False)) is not True
        ):
            raise ObserverDatasetContractError("schema5 HDF identity changed")
        if "candidate_hidden" in handle:
            raise ObserverDatasetContractError("candidate-specific hidden is forbidden")
        required = {
            "candidate_names",
            "object_names",
            "success",
            "steps",
            "branches",
        }
        if not required.issubset(handle.keys()):
            raise ObserverDatasetContractError("schema5 HDF supervision is incomplete")
        object_names = _decode_strings(handle["object_names"][:])
        candidate_names = _decode_strings(handle["candidate_names"][:])
        branches = handle["branches"]
        branch_names = sorted(branches.keys())
        success = np.asarray(handle["success"][:], dtype=np.bool_)
        terminal_steps = np.asarray(handle["steps"][:], dtype=np.int64)
        if (
            len(branch_names) != len(success)
            or len(branch_names) != len(terminal_steps)
            or len(branch_names) != len(candidate_names)
            or len(set(candidate_names)) != len(candidate_names)
        ):
            raise ObserverDatasetContractError("schema5 branch accounting changed")
        for branch_index, name in enumerate(branch_names):
            if name != f"candidate_{branch_index:03d}":
                raise ObserverDatasetContractError("schema5 branch order is not canonical")
            branch = branches[name]
            required_branch = {
                "query_hidden",
                "query_post_hidden",
                "query_steps",
                "query_post_steps",
                "query_actions",
                "query_action_mask",
                "proprio",
                "object_poses",
            }
            if not required_branch.issubset(branch.keys()):
                raise ObserverDatasetContractError("schema5 branch fields are incomplete")
            hidden = np.asarray(branch["query_hidden"][:], dtype=np.float32)
            post_hidden = np.asarray(branch["query_post_hidden"][:], dtype=np.float32)
            query_steps = np.asarray(branch["query_steps"][:], dtype=np.int64)
            query_post = np.asarray(branch["query_post_steps"][:], dtype=np.int64)
            actions = np.asarray(branch["query_actions"][:], dtype=np.float32)
            action_masks = np.asarray(branch["query_action_mask"][:], dtype=np.bool_)
            poses = np.asarray(branch["object_poses"][:], dtype=np.float64)
            proprio = np.asarray(branch["proprio"][:], dtype=np.float32)
            query_count = hidden.shape[0]
            terminal = int(terminal_steps[branch_index])
            if (
                hidden.shape != (query_count, STATE_DIM)
                or post_hidden.shape != (query_count, STATE_DIM)
                or query_steps.shape != (query_count,)
                or query_post.shape != (query_count,)
                or actions.ndim != 3
                or actions.shape[0] != query_count
                or actions.shape[2] != PROPRIO_DIM
                or action_masks.shape != actions.shape[:2]
                or poses.shape != (terminal + 1, len(object_names), 7)
                or proprio.shape != (terminal + 1, PROPRIO_DIM)
                or query_count < 1
                or query_steps[0] != 0
                or query_post[-1] != terminal
                or not np.array_equal(query_steps[1:], query_post[:-1])
                or np.any(query_post <= query_steps)
                or (
                    query_count > 1
                    and not np.array_equal(hidden[1:], post_hidden[:-1])
                )
                or not np.isfinite(hidden).all()
                or not np.isfinite(post_hidden).all()
                or not np.isfinite(actions).all()
                or not np.isfinite(proprio).all()
            ):
                raise ObserverDatasetContractError("schema5 causal branch is invalid")
            for query_index in range(query_count):
                current_step = int(query_steps[query_index])
                if current_step < 0 or current_step > terminal:
                    raise ObserverDatasetContractError("schema5 query step is out of range")
                if query_index == 0:
                    prior_present, prior_steps, prior_sha = False, 0, ""
                else:
                    prior_mask = action_masks[query_index - 1]
                    prior_steps = _valid_prefix_mask(prior_mask, actions.shape[1])
                    if prior_steps != current_step - int(query_steps[query_index - 1]):
                        raise ObserverDatasetContractError("schema5 execution/query chain differs")
                    prior_present = True
                    prior_sha = array_sha256(actions[query_index - 1, prior_mask])
                event, predicates = _derive_current_labels(
                    poses=poses,
                    object_names=object_names,
                    current_step=current_step,
                    terminal_success=bool(success[branch_index]),
                    calibration=calibration,
                )
                samples.append(
                    _sample(
                        descriptor=descriptor,
                        source_sha=source_sha,
                        branch_id=name,
                        hidden_prefix=hidden[: query_index + 1],
                        proprio=proprio[current_step],
                        event_label=event,
                        predicate_label=predicates,
                        current_query_index=query_index,
                        query_step=current_step,
                        prior_execution_present=prior_present,
                        prior_executed_control_steps=prior_steps,
                        prior_action_sha256=prior_sha,
                    )
                )
            # Schema5 records one actor-visible post-query state after every
            # action, including the terminal/censored state.  Only the final
            # post state is additional; earlier post states equal the next
            # query_hidden and would duplicate samples.
            final_mask = action_masks[-1]
            final_steps = _valid_prefix_mask(final_mask, actions.shape[1])
            if final_steps != terminal - int(query_steps[-1]):
                raise ObserverDatasetContractError("schema5 terminal execution differs")
            terminal_prefix = np.concatenate(
                [hidden, post_hidden[-1:].astype(np.float32)], axis=0
            )
            event, predicates = _derive_current_labels(
                poses=poses,
                object_names=object_names,
                current_step=terminal,
                terminal_success=bool(success[branch_index]),
                calibration=calibration,
            )
            samples.append(
                _sample(
                    descriptor=descriptor,
                    source_sha=source_sha,
                    branch_id=name,
                    hidden_prefix=terminal_prefix,
                    proprio=proprio[terminal],
                    event_label=event,
                    predicate_label=predicates,
                    current_query_index=query_count,
                    query_step=terminal,
                    prior_execution_present=True,
                    prior_executed_control_steps=final_steps,
                    prior_action_sha256=array_sha256(actions[-1, final_mask]),
                )
            )
    return samples


def _decode_json_scalar(dataset: h5py.Dataset, role: str) -> dict[str, Any]:
    raw = dataset[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ObserverDatasetContractError(f"{role} is not JSON text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ObserverDatasetContractError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ObserverDatasetContractError(f"{role} is not an object")
    return value


def _materialize_schema6_group(
    descriptor: GroupDescriptor, *, source_sha: str, event_spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    calibration = _task_calibration(event_spec, descriptor.task)
    samples: list[dict[str, Any]] = []
    with h5py.File(descriptor.path, "r") as handle:
        if (
            int(handle.attrs.get("schema_version", -1)) != 6
            or str(handle.attrs.get("format", ""))
            != "etsf_smolvla_piper_dense_event_branches_schema6_v1"
        ):
            raise ObserverDatasetContractError("schema6 HDF identity changed")
        if "branches" not in handle or "root" not in handle:
            raise ObserverDatasetContractError("schema6 root/branches are absent")
        root = handle["root"]
        if not {
            "hidden",
            "processed_state",
            "eligible_original_candidate_indices",
        }.issubset(root.keys()):
            raise ObserverDatasetContractError("schema6 root identity is incomplete")
        root_hidden = np.asarray(root["hidden"][:], dtype=np.float32)
        root_processed = np.asarray(root["processed_state"][:], dtype=np.float32)
        eligible = np.asarray(
            root["eligible_original_candidate_indices"][:], dtype=np.int64
        )
        if (
            root_hidden.shape != (STATE_DIM,)
            or root_processed.shape != (PROPRIO_DIM,)
            or eligible.ndim != 1
            or len(eligible) < 1
            or len(set(eligible.tolist())) != len(eligible)
            or not np.isfinite(root_hidden).all()
            or not np.isfinite(root_processed).all()
        ):
            raise ObserverDatasetContractError("schema6 root state is invalid")
        branch_names = sorted(handle["branches"].keys())
        if len(branch_names) != len(eligible):
            raise ObserverDatasetContractError(
                "schema6 group branch/eligible accounting changed"
            )
        for branch_index, name in enumerate(branch_names):
            if name != f"branch_{branch_index:03d}":
                raise ObserverDatasetContractError("schema6 branch order is not canonical")
            branch = handle["branches"][name]
            required = {
                "query_hidden",
                "query_processed_state",
                "query_selected_original_candidate_index",
                "query_executed_action",
                "query_executed_action_mask",
                "object_poses",
                "proprio",
                POSE_QUALITY_GROUP,
            }
            if not required.issubset(branch.keys()):
                raise ObserverDatasetContractError("schema6 branch fields are incomplete")
            validate_pose_quality_v6(
                branch,
                expected_registry_sha256=str(handle.attrs.get("object_registry_sha256", "")),
                expected_spec_sha256=str(handle.attrs.get("pose_integrity_spec_sha256", "")),
            )
            quality = branch[POSE_QUALITY_GROUP]
            registry = _decode_json_scalar(
                quality["object_registry_json"], "schema6 object registry"
            )
            objects = registry.get("objects")
            if not isinstance(objects, list) or not objects:
                raise ObserverDatasetContractError("schema6 object registry is empty")
            object_names = [str(item.get("name", "")) for item in objects]
            hidden = np.asarray(branch["query_hidden"][:], dtype=np.float32)
            processed = np.asarray(branch["query_processed_state"][:], dtype=np.float32)
            selected = np.asarray(
                branch["query_selected_original_candidate_index"][:], dtype=np.int64
            )
            executed = np.asarray(branch["query_executed_action"][:], dtype=np.float32)
            action_masks = np.asarray(
                branch["query_executed_action_mask"][:], dtype=np.bool_
            )
            poses = np.asarray(branch["object_poses"][:], dtype=np.float64)
            proprio = np.asarray(branch["proprio"][:], dtype=np.float32)
            pose_valid = np.asarray(quality["pose_quality_valid"][:], dtype=np.bool_)
            query_count = hidden.shape[0]
            terminal = int(branch.attrs.get("steps", -1))
            terminal_success = bool(branch.attrs.get("success_diagnostic_only", False))
            if (
                int(branch.attrs.get("original_candidate_index", -1))
                != int(eligible[branch_index])
                or hidden.shape != (query_count, STATE_DIM)
                or processed.shape != (query_count, PROPRIO_DIM)
                or selected.shape != (query_count,)
                or executed.shape != (query_count, PROPRIO_DIM)
                or action_masks.ndim != 2
                or action_masks.shape[0] != query_count
                or poses.shape != (terminal + 1, len(object_names), 7)
                or proprio.shape != (terminal + 1, PROPRIO_DIM)
                or pose_valid.shape != poses.shape[:2]
                or query_count < 1
                or query_count not in (terminal, terminal + 1)
                or not np.isfinite(hidden).all()
                or not np.isfinite(processed).all()
                or not np.isfinite(executed).all()
                or not np.isfinite(proprio).all()
            ):
                raise ObserverDatasetContractError("schema6 causal branch is invalid")
            if not np.array_equal(hidden[0], root_hidden) or not np.array_equal(
                processed[0], root_processed
            ):
                raise ObserverDatasetContractError(
                    "schema6 branch root actor-visible state changed"
                )
            executed_lengths = np.asarray(
                [
                    _valid_prefix_mask(mask, action_masks.shape[1])
                    for mask in action_masks
                ],
                dtype=np.int64,
            )
            if (
                np.any((selected >= 0) & (executed_lengths != 1))
                or np.any((selected < 0) & (executed_lengths != 0))
                or (query_count == terminal and np.any(selected < 0))
                or (query_count == terminal + 1 and selected[-1] >= 0)
            ):
                raise ObserverDatasetContractError(
                    "schema6 selected/executed query chronology changed"
                )
            for query_index in range(query_count):
                current_step = query_index
                if current_step > terminal or not bool(pose_valid[current_step].all()):
                    raise ObserverDatasetContractError(
                        "schema6 current-query pose label is not quality-valid"
                    )
                if query_index == 0:
                    prior_present, prior_steps, prior_sha = False, 0, ""
                else:
                    prior_mask = action_masks[query_index - 1]
                    prior_steps = _valid_prefix_mask(prior_mask, action_masks.shape[1])
                    if prior_steps != 1 or selected[query_index - 1] < 0:
                        raise ObserverDatasetContractError(
                            "schema6 previous query did not execute the recorded H=1 action"
                        )
                    prior_present = True
                    prior_sha = array_sha256(executed[query_index - 1])
                if selected[query_index] < 0 and query_index != terminal:
                    raise ObserverDatasetContractError(
                        "schema6 censored query is not the terminal observation"
                    )
                event, predicates = _derive_current_labels(
                    poses=poses,
                    object_names=object_names,
                    current_step=current_step,
                    terminal_success=terminal_success,
                    calibration=calibration,
                )
                samples.append(
                    _sample(
                        descriptor=descriptor,
                        source_sha=source_sha,
                        branch_id=name,
                        hidden_prefix=hidden[: query_index + 1],
                        proprio=processed[query_index],
                        event_label=event,
                        predicate_label=predicates,
                        current_query_index=query_index,
                        query_step=current_step,
                        prior_execution_present=prior_present,
                        prior_executed_control_steps=prior_steps,
                        prior_action_sha256=prior_sha,
                    )
                )
    return samples


def _stack_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ObserverDatasetContractError("a split produced no samples")
    result = {
        "history": np.stack([row["history"] for row in samples]).astype(np.float32),
        "history_mask": np.stack([row["history_mask"] for row in samples]).astype(np.bool_),
        "proprio": np.stack([row["proprio"] for row in samples]).astype(np.float32),
        "event_label": np.asarray([row["event_label"] for row in samples], dtype=np.int64),
        "predicate_label": np.stack([row["predicate_label"] for row in samples]).astype(np.float32),
        "actor_index": np.asarray([row["actor_index"] for row in samples], dtype=np.int64),
        "current_query_index": np.asarray([row["current_query_index"] for row in samples], dtype=np.int64),
        "query_step": np.asarray([row["query_step"] for row in samples], dtype=np.int64),
        "prior_execution_present": np.asarray([row["prior_execution_present"] for row in samples], dtype=np.bool_),
        "prior_executed_control_steps": np.asarray([row["prior_executed_control_steps"] for row in samples], dtype=np.int64),
        "prior_action_sha256": np.asarray([row["prior_action_sha256"] for row in samples], dtype="U64"),
        "sample_id": np.asarray([row["sample_id"] for row in samples], dtype="U64"),
        "logical_group_id": np.asarray([row["logical_group_id"] for row in samples], dtype=str),
        "branch_id": np.asarray([row["branch_id"] for row in samples], dtype=str),
        "source_file_sha256": np.asarray([row["source_file_sha256"] for row in samples], dtype="U64"),
    }
    _validate_arrays(result)
    return result


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(ARRAY_NAMES):
        raise ObserverDatasetContractError("dataset array registry changed")
    n = len(arrays["event_label"])
    exact = {
        "history": ((n, HISTORY_STEPS, STATE_DIM), np.dtype(np.float32)),
        "history_mask": ((n, HISTORY_STEPS), np.dtype(np.bool_)),
        "proprio": ((n, PROPRIO_DIM), np.dtype(np.float32)),
        "event_label": ((n,), np.dtype(np.int64)),
        "predicate_label": ((n, len(PREDICATE_NAMES)), np.dtype(np.float32)),
        "actor_index": ((n,), np.dtype(np.int64)),
        "current_query_index": ((n,), np.dtype(np.int64)),
        "query_step": ((n,), np.dtype(np.int64)),
        "prior_execution_present": ((n,), np.dtype(np.bool_)),
        "prior_executed_control_steps": ((n,), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in exact.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ObserverDatasetContractError(f"array {name} shape/dtype changed")
    for name in (
        "prior_action_sha256",
        "sample_id",
        "logical_group_id",
        "branch_id",
        "source_file_sha256",
    ):
        if arrays[name].shape != (n,) or arrays[name].dtype.kind != "U":
            raise ObserverDatasetContractError(f"identity array {name} changed")
    if n < 1 or not np.isfinite(arrays["history"]).all() or not np.isfinite(arrays["proprio"]).all() or not np.isfinite(arrays["predicate_label"]).all():
        raise ObserverDatasetContractError("dataset contains empty/non-finite tensors")
    masks = arrays["history_mask"]
    valid = masks.sum(axis=1)
    expected = np.arange(HISTORY_STEPS)[None, :] < valid[:, None]
    if (
        np.any(valid < 1)
        or not np.array_equal(masks, expected)
        or np.any(arrays["history"][~masks] != 0)
        or np.any(arrays["event_label"] < 0)
        or np.any(arrays["event_label"] >= len(EVENT_NAMES))
        or np.any(arrays["predicate_label"] < 0)
        or np.any(arrays["predicate_label"] > 1)
        or len(set(arrays["sample_id"].tolist())) != n
    ):
        raise ObserverDatasetContractError("dataset causality/label identity failed")
    for index in range(n):
        present = bool(arrays["prior_execution_present"][index])
        steps = int(arrays["prior_executed_control_steps"][index])
        action_sha = str(arrays["prior_action_sha256"][index])
        query_index = int(arrays["current_query_index"][index])
        if query_index == 0:
            if present or steps != 0 or action_sha:
                raise ObserverDatasetContractError("root execution receipt is not absent")
        elif not present or steps < 1 or not _is_sha(action_sha):
            raise ObserverDatasetContractError("continuation execution receipt is absent")
        if not _is_sha(str(arrays["sample_id"][index])) or not _is_sha(
            str(arrays["source_file_sha256"][index])
        ):
            raise ObserverDatasetContractError("sample/source identity SHA is invalid")


def _array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "sha256": array_sha256(arrays[name]),
        }
        for name in sorted(arrays)
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def materialize(request_path: Path, output_directory: Path) -> dict[str, Any]:
    frozen = freeze_request(request_path)
    output = output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    split_records: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    source_sha_cache: dict[Path, str] = {}
    try:
        for split_name in SPLIT_NAMES:
            samples: list[dict[str, Any]] = []
            logical_ids: list[str] = []
            for descriptor in frozen.split_groups[split_name]:
                source_sha = source_sha_cache.get(descriptor.path)
                if source_sha is None:
                    source_sha = file_sha256(descriptor.path)
                    source_sha_cache[descriptor.path] = source_sha
                if source_sha != descriptor.expected_file_sha256:
                    raise ObserverDatasetContractError("source group file SHA changed")
                if descriptor.logical_group_id in all_ids:
                    raise ObserverDatasetContractError("logical reset group crossed a split")
                if descriptor.source.schema_version == 5:
                    rows = _materialize_schema5_group(
                        descriptor,
                        source_sha=source_sha,
                        event_spec=frozen.event_spec,
                        event_spec_sha256=frozen.event_spec_file_sha256,
                    )
                else:
                    rows = _materialize_schema6_group(
                        descriptor, source_sha=source_sha, event_spec=frozen.event_spec
                    )
                samples.extend(rows)
                logical_ids.append(descriptor.logical_group_id)
                all_ids.add(descriptor.logical_group_id)
            arrays = _stack_samples(samples)
            npz_path = output / f"{split_name}.npz"
            _atomic_npz(npz_path, arrays)
            logical_group_ids = sorted(logical_ids)
            base: dict[str, Any] = {
                "format": SPLIT_FORMAT,
                "split": split_name,
                "path": npz_path.name,
                "row_count": len(samples),
                "logical_group_ids": logical_group_ids,
                "logical_group_ids_sha256": canonical_sha256(logical_group_ids),
                "arrays": _array_records(arrays),
            }
            logical_sha = canonical_sha256(base)
            split_records[split_name] = {
                "path": npz_path.name,
                "file_sha256": file_sha256(npz_path),
                "logical_sha256": logical_sha,
                "row_count": len(samples),
                "logical_group_ids": logical_group_ids,
                "logical_group_ids_sha256": base["logical_group_ids_sha256"],
                "arrays": base["arrays"],
            }
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "status": STATUS,
            "event_names": list(EVENT_NAMES),
            "predicate_names": list(PREDICATE_NAMES),
            "state_dim": STATE_DIM,
            "history_steps": HISTORY_STEPS,
            "proprio_dim": PROPRIO_DIM,
            "image_feature_dim": 0,
            "history_contract_sha256": causal_history_contract()["contract_sha256"],
            "event_spec": {
                "path": str(frozen.event_spec_path),
                "file_sha256": frozen.event_spec_file_sha256,
            },
            "request": {
                "path": str(frozen.path),
                "file_sha256": frozen.file_sha256,
                "logical_sha256": frozen.logical_sha256,
            },
            "actor_registry": [
                {
                    "actor_name": actor.actor_name,
                    "policy_family": actor.policy_family,
                    "body": actor.body,
                    "policy": actor.policy,
                    "state_feature_source_sha256": actor.state_feature_source_sha256,
                    "actor_index": actor.actor_index,
                }
                for actor in frozen.actors
            ],
            "source_manifests": [
                {
                    "source_name": source.source_name,
                    "schema_version": source.schema_version,
                    "path": str(source.manifest_path),
                    "file_sha256": source.manifest_file_sha256,
                    "logical_sha256": source.manifest_logical_sha256,
                }
                for source in frozen.sources
            ],
            "split_unit": "logical_reset_group",
            "split_group_disjoint": True,
            "privileged_label_source_available_to_model_inputs": False,
            "future_query_features_available_to_model_inputs": False,
            "splits": split_records,
            "all_logical_group_ids_sha256": canonical_sha256(sorted(all_ids)),
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = output / "manifest.json"
        _atomic_json(manifest_path, manifest)
        validate_dataset_manifest(manifest_path, verify_npz=True)
        return manifest
    except BaseException:
        # No valid manifest is published on failure.  Partial content-addressed
        # files remain diagnostic-only and cannot be consumed by the loader.
        raise


def _validate_split_record(
    manifest_root: Path,
    split_name: str,
    raw: Any,
    *,
    verify_npz: bool,
) -> tuple[dict[str, Any], set[str]]:
    item = _exact_mapping(
        raw,
        {
            "path",
            "file_sha256",
            "logical_sha256",
            "row_count",
            "logical_group_ids",
            "logical_group_ids_sha256",
            "arrays",
        },
        f"dataset {split_name} split",
    )
    logical_ids = item["logical_group_ids"]
    if (
        not isinstance(logical_ids, list)
        or not logical_ids
        or any(not isinstance(value, str) or not value for value in logical_ids)
        or logical_ids != sorted(logical_ids)
        or len(set(logical_ids)) != len(logical_ids)
        or item["logical_group_ids_sha256"] != canonical_sha256(logical_ids)
        or not _strict_int(item["row_count"], minimum=1)
        or not _is_sha(item["file_sha256"])
        or not _is_sha(item["logical_sha256"])
    ):
        raise ObserverDatasetContractError("dataset split identity is invalid")
    arrays_meta = item["arrays"]
    if not isinstance(arrays_meta, Mapping) or set(arrays_meta) != set(ARRAY_NAMES):
        raise ObserverDatasetContractError("dataset split array metadata changed")
    logical_base = {
        "format": SPLIT_FORMAT,
        "split": split_name,
        "path": item["path"],
        "row_count": item["row_count"],
        "logical_group_ids": logical_ids,
        "logical_group_ids_sha256": item["logical_group_ids_sha256"],
        "arrays": arrays_meta,
    }
    if item["logical_sha256"] != canonical_sha256(logical_base):
        raise ObserverDatasetContractError("dataset split logical SHA changed")
    path = _resolve_file(item["path"], base=manifest_root, role=f"{split_name} NPZ")
    if verify_npz:
        if file_sha256(path) != item["file_sha256"]:
            raise ObserverDatasetContractError("dataset split file SHA changed")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(ARRAY_NAMES):
                    raise ObserverDatasetContractError("NPZ array registry changed")
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        except (OSError, ValueError) as error:
            raise ObserverDatasetContractError("dataset split NPZ is unreadable") from error
        _validate_arrays(arrays)
        if len(arrays["event_label"]) != item["row_count"]:
            raise ObserverDatasetContractError("dataset split row count changed")
        if set(arrays["logical_group_id"].tolist()) != set(logical_ids):
            raise ObserverDatasetContractError("NPZ logical groups differ from manifest")
        for name, array in arrays.items():
            meta = _exact_mapping(
                arrays_meta[name], {"dtype", "shape", "sha256"}, f"array {name} metadata"
            )
            if (
                meta["dtype"] != array.dtype.str
                or meta["shape"] != list(array.shape)
                or meta["sha256"] != array_sha256(array)
            ):
                raise ObserverDatasetContractError(f"array {name} content address changed")
    return dict(item), set(logical_ids)


def validate_dataset_manifest(
    path: Path, *, verify_npz: bool = True
) -> dict[str, Any]:
    manifest_path = path.resolve()
    item = _load_json(manifest_path, "observer dataset manifest")
    expected = {
        "format",
        "status",
        "event_names",
        "predicate_names",
        "state_dim",
        "history_steps",
        "proprio_dim",
        "image_feature_dim",
        "history_contract_sha256",
        "event_spec",
        "request",
        "actor_registry",
        "source_manifests",
        "split_unit",
        "split_group_disjoint",
        "privileged_label_source_available_to_model_inputs",
        "future_query_features_available_to_model_inputs",
        "splits",
        "all_logical_group_ids_sha256",
        "manifest_sha256",
    }
    _exact_mapping(item, expected, "observer dataset manifest")
    logical = dict(item)
    digest = logical.pop("manifest_sha256")
    if (
        item["format"] != FORMAT
        or item["status"] != STATUS
        or item["event_names"] != list(EVENT_NAMES)
        or item["predicate_names"] != list(PREDICATE_NAMES)
        or item["state_dim"] != STATE_DIM
        or item["history_steps"] != HISTORY_STEPS
        or item["proprio_dim"] != PROPRIO_DIM
        or item["image_feature_dim"] != 0
        or item["history_contract_sha256"]
        != causal_history_contract()["contract_sha256"]
        or item["split_unit"] != "logical_reset_group"
        or item["split_group_disjoint"] is not True
        or item["privileged_label_source_available_to_model_inputs"] is not False
        or item["future_query_features_available_to_model_inputs"] is not False
        or not _is_sha(digest)
        or digest != canonical_sha256(logical)
    ):
        raise ObserverDatasetContractError("observer dataset manifest is invalid")
    root = manifest_path.parent
    frozen_request: FrozenRequest | None = None
    for role in ("event_spec", "request"):
        record = item[role]
        fields = {"path", "file_sha256"} | ({"logical_sha256"} if role == "request" else set())
        _exact_mapping(record, fields, f"manifest {role}")
        source = _resolve_file(record["path"], base=root, role=f"manifest {role}")
        if not _is_sha(record["file_sha256"]) or (
            verify_npz and file_sha256(source) != record["file_sha256"]
        ):
            raise ObserverDatasetContractError(f"manifest {role} content changed")
        if role == "request" and not _is_sha(record["logical_sha256"]):
            raise ObserverDatasetContractError("request logical SHA is invalid")
        if role == "request":
            frozen_request = freeze_request(source)
            if (
                frozen_request.logical_sha256 != record["logical_sha256"]
                or frozen_request.file_sha256 != record["file_sha256"]
            ):
                raise ObserverDatasetContractError(
                    "manifest request logical/file identity changed"
                )
    if frozen_request is None:
        raise AssertionError("validated manifest request was not frozen")
    if (
        str(frozen_request.event_spec_path) != str(
            _resolve_file(
                item["event_spec"]["path"], base=root, role="manifest event spec"
            )
        )
        or frozen_request.event_spec_file_sha256
        != item["event_spec"]["file_sha256"]
    ):
        raise ObserverDatasetContractError("manifest event spec differs from request")
    actors = item["actor_registry"]
    if not isinstance(actors, list) or not actors:
        raise ObserverDatasetContractError("manifest actor registry is empty")
    actor_indices: set[int] = set()
    for index, raw in enumerate(actors):
        actor = _exact_mapping(
            raw,
            {
                "actor_name",
                "policy_family",
                "body",
                "policy",
                "state_feature_source_sha256",
                "actor_index",
            },
            "manifest actor",
        )
        if actor["actor_index"] != index or not _is_sha(actor["state_feature_source_sha256"]):
            raise ObserverDatasetContractError("manifest actor registry changed")
        actor_indices.add(index)
    expected_actors = [
        {
            "actor_name": actor.actor_name,
            "policy_family": actor.policy_family,
            "body": actor.body,
            "policy": actor.policy,
            "state_feature_source_sha256": actor.state_feature_source_sha256,
            "actor_index": actor.actor_index,
        }
        for actor in frozen_request.actors
    ]
    if actors != expected_actors:
        raise ObserverDatasetContractError("manifest actors differ from request")
    sources = item["source_manifests"]
    if not isinstance(sources, list) or not sources:
        raise ObserverDatasetContractError("manifest source registry is empty")
    for raw in sources:
        source = _exact_mapping(
            raw,
            {"source_name", "schema_version", "path", "file_sha256", "logical_sha256"},
            "manifest source",
        )
        if source["schema_version"] not in (5, 6) or not _is_sha(source["file_sha256"]) or not _is_sha(source["logical_sha256"]):
            raise ObserverDatasetContractError("manifest source identity is invalid")
        source_path = _resolve_file(source["path"], base=root, role="source manifest")
        if verify_npz and file_sha256(source_path) != source["file_sha256"]:
            raise ObserverDatasetContractError("source manifest content changed")
    expected_sources = [
        {
            "source_name": source.source_name,
            "schema_version": source.schema_version,
            "path": str(source.manifest_path),
            "file_sha256": source.manifest_file_sha256,
            "logical_sha256": source.manifest_logical_sha256,
        }
        for source in frozen_request.sources
    ]
    if sources != expected_sources:
        raise ObserverDatasetContractError("manifest sources differ from request")
    split_map = _exact_mapping(item["splits"], set(SPLIT_NAMES), "manifest splits")
    seen: set[str] = set()
    all_ids: set[str] = set()
    for split_name in SPLIT_NAMES:
        _, ids = _validate_split_record(
            root, split_name, split_map[split_name], verify_npz=verify_npz
        )
        expected_descriptors = frozen_request.split_groups[split_name]
        expected_ids = {descriptor.logical_group_id for descriptor in expected_descriptors}
        if ids != expected_ids:
            raise ObserverDatasetContractError(
                "manifest split membership differs from frozen request"
            )
        if seen & ids:
            raise ObserverDatasetContractError("logical group crossed dataset splits")
        seen |= ids
        all_ids |= ids
        if verify_npz:
            arrays = load_split(manifest_path, split_name, _already_validated=True)
            if not set(arrays["actor_index"].tolist()).issubset(actor_indices):
                raise ObserverDatasetContractError("NPZ actor index is unregistered")
            descriptors = {
                descriptor.logical_group_id: descriptor
                for descriptor in expected_descriptors
            }
            verified_group_files: set[Path] = set()
            sample_keys: set[tuple[str, str, int]] = set()
            for row in range(len(arrays["event_label"])):
                group_id = str(arrays["logical_group_id"][row])
                branch_id = str(arrays["branch_id"][row])
                query_index = int(arrays["current_query_index"][row])
                descriptor = descriptors.get(group_id)
                if descriptor is None:
                    raise ObserverDatasetContractError(
                        "NPZ sample belongs to an unregistered logical group"
                    )
                if descriptor.path not in verified_group_files:
                    if file_sha256(descriptor.path) != descriptor.expected_file_sha256:
                        raise ObserverDatasetContractError(
                            "NPZ source group file content changed"
                        )
                    verified_group_files.add(descriptor.path)
                if (
                    str(arrays["source_file_sha256"][row])
                    != descriptor.expected_file_sha256
                    or int(arrays["actor_index"][row])
                    != descriptor.source.actor.actor_index
                ):
                    raise ObserverDatasetContractError(
                        "NPZ source/actor binding differs from frozen request"
                    )
                sample_key = (group_id, branch_id, query_index)
                if sample_key in sample_keys:
                    raise ObserverDatasetContractError(
                        "NPZ contains a duplicate branch/query sample"
                    )
                sample_keys.add(sample_key)
                identity = {
                    "source_name": descriptor.source.source_name,
                    "source_file_sha256": descriptor.expected_file_sha256,
                    "logical_group_id": group_id,
                    "branch_id": branch_id,
                    "current_query_index": query_index,
                    "actor_name": descriptor.source.actor.actor_name,
                    "prior_action_sha256": str(
                        arrays["prior_action_sha256"][row]
                    ),
                }
                if str(arrays["sample_id"][row]) != canonical_sha256(identity):
                    raise ObserverDatasetContractError(
                        "NPZ sample identity SHA is not reproducible"
                    )
    if item["all_logical_group_ids_sha256"] != canonical_sha256(sorted(all_ids)):
        raise ObserverDatasetContractError("all-group identity SHA changed")
    return item


def load_split(
    manifest_path: Path, split: str, *, _already_validated: bool = False
) -> dict[str, np.ndarray]:
    if split not in SPLIT_NAMES:
        raise ObserverDatasetContractError("unknown observer dataset split")
    path = manifest_path.resolve()
    manifest = (
        _load_json(path, "observer dataset manifest")
        if _already_validated
        else validate_dataset_manifest(path, verify_npz=True)
    )
    record = manifest["splits"][split]
    npz_path = _resolve_file(record["path"], base=path.parent, role=f"{split} NPZ")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    _validate_arrays(arrays)
    return arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize(args.request, args.output_directory)
    print("OBSERVER_DATASET_MATERIALIZED=" + json.dumps(
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "splits": {
                name: manifest["splits"][name]["row_count"] for name in SPLIT_NAMES
            },
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARRAY_NAMES",
    "EVENT_NAMES",
    "FORMAT",
    "HISTORY_STEPS",
    "ObserverDatasetContractError",
    "PREDICATE_NAMES",
    "PROPRIO_DIM",
    "REQUEST_FORMAT",
    "REQUEST_STATUS",
    "SPLIT_FORMAT",
    "SPLIT_NAMES",
    "STATE_DIM",
    "array_sha256",
    "canonical_sha256",
    "file_sha256",
    "freeze_request",
    "load_split",
    "materialize",
    "schema5_logical_group_id",
    "schema6_logical_group_id",
    "validate_dataset_manifest",
]
