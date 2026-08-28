#!/usr/bin/env python3
"""Train the action-conditioned OpenVLA ETSF event world model.

The input rollouts already contain frozen OpenVLA hidden states.  This script
turns every policy query into one factual transition, keeps the split at the
episode/seed level, and never evaluates the sealed test split.  It is intended
to run both as a small CPU smoke test and as a single-GPU training job.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DEFAULT_EVENTS = ("e0", "e12", "e3", "e4", "eK")
DEFAULT_PREDICATES = ("moved", "lifted", "near_goal", "stationary", "success")
RELATIVE_TRANSITIONS = ("stay", "advance", "skip", "regress")
SPLIT_SEED = 20260826
CACHE_SCHEMA = 3


@dataclasses.dataclass(frozen=True)
class EpisodeDescriptor:
    """Label-free rollout identity used before opening any episode HDF5.

    These fields come only from the collection manifest.  In particular,
    outcome, length, events and trajectory statistics are deliberately absent
    so a sealed episode can be assigned without inspecting its targets.
    """

    index: int
    seed: int
    relative_path: str
    path: str
    manifest_schema: int
    task: str
    body: str
    policy: str


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def current_event_at(
    query_step: int,
    event_names: Sequence[str],
    event_steps: Sequence[int],
    event_to_id: Mapping[str, int],
) -> int:
    current = event_to_id[event_names[0]]
    for name, step in sorted(zip(event_names, event_steps), key=lambda item: item[1]):
        if int(step) <= query_step:
            current = event_to_id[name]
    return current


def event_transition_target(
    query_step: int,
    end_step: int,
    terminal_step: int,
    event_names: Sequence[str],
    event_steps: Sequence[int],
    event_to_id: Mapping[str, int],
) -> tuple[int, int, float, bool]:
    """Return current event, first reached event, duration, and observation flag.

    The target is the next event under the frozen actor continuation used by the
    counterfactual branch contract.  Object and latent effects remain local to
    ``end_step``; event duration may extend beyond the first chunk.  If the
    episode never reaches another event, its terminal step is a right-censoring
    lower bound.
    """

    current = current_event_at(query_step, event_names, event_steps, event_to_id)
    future = sorted(
        (
            (int(step), event_to_id[name])
            for name, step in zip(event_names, event_steps)
            if query_step < int(step) <= terminal_step
        ),
        key=lambda item: item[0],
    )
    if future:
        boundary_step, next_event = future[0]
        return current, next_event, float(boundary_step - query_step), True
    return current, current, float(terminal_step - query_step), False


def derive_atomic_predicates(
    poses: np.ndarray,
    object_names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    """Derive dynamic, reversible predicates at every simulator step.

    Unlike first-hit canonical events, ``lifted``, ``near_goal`` and
    ``stationary`` may turn off again.  This supplies genuine regression
    supervision whenever the recorded trajectory contains it.  ``moved`` is
    cumulative by definition and ``success`` is true only at a successful
    terminal state, avoiding episode-outcome leakage into earlier queries.
    """

    names = list(object_names)
    moving_name = str(calibration["moving"])
    if moving_name not in names:
        raise ValueError(f"moving object {moving_name!r} absent from {names}")
    position = poses[:, names.index(moving_name), :3]
    motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    cumulative_motion = np.cumsum(motion)

    anchor_name = calibration.get("anchor")
    if anchor_name:
        anchor_name = str(anchor_name)
        if anchor_name not in names:
            raise ValueError(f"anchor object {anchor_name!r} absent from {names}")
        anchor = poses[:, names.index(anchor_name), :3]
        offset = np.asarray(calibration.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
        goal_distance = np.linalg.norm(position - anchor - offset, axis=1)
    else:
        centers = np.asarray(calibration["centers"], dtype=np.float32)
        goal_distance = np.linalg.norm(
            position[:, None] - centers[None], axis=2
        ).min(axis=1)

    moved = cumulative_motion >= float(calibration["delta_move"])
    lifted = position[:, 2] >= position[0, 2] + float(calibration["delta_z"])
    near_goal = goal_distance <= float(calibration["tau_d"])
    instant_stationary = near_goal & (
        motion <= float(calibration["tau_motion"])
    )
    width = int(calibration["stationary_steps"])
    if width <= 0:
        raise ValueError("stationary_steps must be positive")
    stationary = np.zeros(len(poses), dtype=bool)
    for index in range(width - 1, len(poses)):
        stationary[index] = bool(
            instant_stationary[index - width + 1 : index + 1].all()
        )
    succeeded = np.zeros(len(poses), dtype=bool)
    if success:
        succeeded[-1] = True
    predicates = np.stack(
        [moved, lifted, near_goal, stationary, succeeded], axis=-1
    )
    return predicates.astype(np.float32)


def dynamic_event_ids(
    predicates: np.ndarray, event_to_id: Mapping[str, int]
) -> np.ndarray:
    """Map dynamic predicate truth to the shared ordered event vocabulary."""

    required = set(DEFAULT_EVENTS)
    if not required.issubset(event_to_id):
        raise ValueError(
            "structured event mode requires event vocabulary "
            f"{DEFAULT_EVENTS}, got {tuple(event_to_id)}"
        )
    if predicates.ndim != 2 or predicates.shape[1] != len(DEFAULT_PREDICATES):
        raise ValueError(
            f"predicates must be [T,{len(DEFAULT_PREDICATES)}]"
        )
    moved, lifted, near_goal, stationary, success = predicates.T > 0.5
    event = np.full(len(predicates), event_to_id["e0"], dtype=np.int64)
    event[moved | lifted] = event_to_id["e12"]
    event[near_goal] = event_to_id["e3"]
    event[stationary] = event_to_id["e4"]
    event[success] = event_to_id["eK"]
    return event


def relative_transition_ids(
    current_event_id: np.ndarray, post_event_id: np.ndarray
) -> np.ndarray:
    """Return stay/advance/skip/regress ids for ordered dynamic phases."""

    difference = post_event_id - current_event_id
    return np.select(
        [difference == 0, difference == 1, difference > 1, difference < 0],
        [0, 1, 2, 3],
    ).astype(np.int64)


def _validate_finite(name: str, value: np.ndarray, path: Path) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"non-finite {name} in {path}")


def read_rollout_descriptors(
    data_root: Path,
) -> tuple[dict[str, Any], list[EpisodeDescriptor]]:
    """Read the rollout manifest without opening an episode HDF5 file."""

    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"rollout collection is not complete: {data_root}")
    items = manifest.get("episodes", [])
    if not items:
        raise RuntimeError(f"manifest contains no episodes: {manifest_path}")

    episodes_root = (data_root / "episodes").resolve()
    descriptors: list[EpisodeDescriptor] = []
    seen_seeds: set[int] = set()
    seen_paths: set[str] = set()
    for position, item in enumerate(items):
        if "seed" not in item or "path" not in item:
            raise RuntimeError(
                f"episode descriptor {position} lacks seed/path in {manifest_path}"
            )
        seed = int(item["seed"])
        relative_path = str(item["path"])
        resolved_path = (episodes_root / relative_path).resolve()
        try:
            resolved_path.relative_to(episodes_root)
        except ValueError as error:
            raise RuntimeError(
                f"episode path escapes rollout root: {relative_path}"
            ) from error
        if not resolved_path.is_file():
            raise RuntimeError(f"episode file does not exist: {resolved_path}")
        if seed in seen_seeds:
            raise RuntimeError(f"duplicate rollout seed in manifest: {seed}")
        if str(resolved_path) in seen_paths:
            raise RuntimeError(f"duplicate rollout path in manifest: {resolved_path}")
        seen_seeds.add(seed)
        seen_paths.add(str(resolved_path))
        descriptors.append(
            EpisodeDescriptor(
                index=int(item.get("index", position)),
                seed=seed,
                relative_path=relative_path,
                path=str(resolved_path),
                manifest_schema=int(manifest.get("schema_version", -1)),
                task=str(manifest.get("task", "")),
                body=str(manifest.get("body", "unknown")),
                policy=str(
                    manifest.get("model_path", manifest.get("policy", "openvla"))
                ),
            )
        )
    requested_seeds = manifest.get("requested_seeds")
    if requested_seeds is not None and {int(value) for value in requested_seeds} != seen_seeds:
        raise RuntimeError(
            "manifest requested_seeds differs from completed episode descriptors"
        )
    return manifest, descriptors


def descriptor_record(
    descriptor: EpisodeDescriptor, *, include_sha256: bool
) -> dict[str, Any]:
    """Serialize identity metadata; raw hashing never opens HDF5 datasets."""

    result: dict[str, Any] = {
        "index": descriptor.index,
        "seed": descriptor.seed,
        "relative_path": descriptor.relative_path,
        "path": descriptor.path,
        "manifest_schema": descriptor.manifest_schema,
        "task": descriptor.task,
        "body": descriptor.body,
        "policy": descriptor.policy,
    }
    if include_sha256:
        result["sha256"] = sha256(Path(descriptor.path))
    return result


def build_transition_cache(
    data_root: Path,
    events: Sequence[str],
    object_names: Sequence[str],
    manifest: Mapping[str, Any],
    episode_descriptors: Sequence[EpisodeDescriptor],
    sealed_test_descriptors: Sequence[EpisodeDescriptor],
    split_seeds: Mapping[str, Sequence[int]],
    event_spec_path: Path | None = None,
    require_predicates: bool = False,
) -> dict[str, Any]:
    """Build transitions only for preselected train/validation episodes.

    ``sealed_test_descriptors`` are raw-hashed for immutable identity auditing,
    but their HDF5 containers are never opened by this function.
    """

    manifest_path = data_root / "manifest.json"
    selected_seeds = {descriptor.seed for descriptor in episode_descriptors}
    sealed_seeds = {descriptor.seed for descriptor in sealed_test_descriptors}
    expected_selected = set(int(seed) for seed in split_seeds["train"]) | set(
        int(seed) for seed in split_seeds["validation"]
    )
    if selected_seeds != expected_selected:
        raise RuntimeError(
            "cache episode descriptors differ from train/validation split: "
            f"selected={sorted(selected_seeds)} expected={sorted(expected_selected)}"
        )
    if sealed_seeds != {int(seed) for seed in split_seeds.get("test", [])}:
        raise RuntimeError("sealed descriptors differ from test split")
    overlap = sorted(selected_seeds & sealed_seeds)
    if overlap:
        raise RuntimeError(f"sealed test leaked into cache selection: {overlap}")
    if not episode_descriptors:
        raise RuntimeError("no train/validation descriptors selected for cache")
    event_to_id = {name: index for index, name in enumerate(events)}
    if len(event_to_id) != len(events):
        raise ValueError("event names must be unique")
    resolved_event_spec: Path | None = event_spec_path
    if resolved_event_spec is None and manifest.get("event_spec"):
        resolved_event_spec = Path(str(manifest["event_spec"]))
        if not resolved_event_spec.is_absolute():
            resolved_event_spec = data_root / resolved_event_spec
    event_spec: Mapping[str, Any] | None = None
    calibration: Mapping[str, Any] | None = None
    event_spec_sha256: str | None = None
    task_name = str(manifest.get("task", ""))
    if resolved_event_spec is not None:
        if not resolved_event_spec.is_file():
            raise RuntimeError(f"event spec does not exist: {resolved_event_spec}")
        event_spec_sha256 = sha256(resolved_event_spec)
        expected_digest = manifest.get("event_spec_sha256")
        if expected_digest and event_spec_sha256 != expected_digest:
            raise RuntimeError(
                "event spec digest differs from rollout manifest: "
                f"{event_spec_sha256} != {expected_digest}"
            )
        event_spec = json.loads(resolved_event_spec.read_text(encoding="utf-8"))
        if task_name not in event_spec.get("calibration", {}):
            raise RuntimeError(
                f"event spec has no calibration for rollout task {task_name!r}"
            )
        calibration = event_spec["calibration"][task_name]
    elif require_predicates:
        raise RuntimeError(
            "structured event mode requires --event-spec or manifest['event_spec']"
        )

    rows: dict[str, list[Any]] = {
        "hidden_t": [],
        "next_hidden": [],
        "action_chunks": [],
        "action_mask": [],
        "proprio": [],
        "current_event_id": [],
        "next_event_id": [],
        "post_event_id": [],
        "structured_current_event_id": [],
        "structured_post_event_id": [],
        "relative_transition_id": [],
        "current_predicates": [],
        "post_predicates": [],
        "reach": [],
        "duration": [],
        "duration_observed": [],
        "success": [],
        "object_delta": [],
        "episode_index": [],
        "seed": [],
        "query_step": [],
        "end_step": [],
        "horizon": [],
        "body": [],
        "policy": [],
    }
    episode_rows = []
    seen_seeds: set[int] = set()
    action_dim: int | None = None
    chunk_size: int | None = None

    for descriptor in episode_descriptors:
        path = Path(descriptor.path)
        with h5py.File(path, "r") as handle:
            required = {
                "query_steps",
                "hidden",
                "terminal_hidden",
                "action_chunks",
                "object_names",
                "object_poses",
                "proprio",
                "event_names",
                "event_steps",
            }
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise RuntimeError(f"missing fields {missing} in {path}")
            seed = int(handle.attrs["seed"])
            if seed != descriptor.seed:
                raise RuntimeError(
                    f"episode seed differs from manifest descriptor in {path}: "
                    f"{seed} != {descriptor.seed}"
                )
            if seed in seen_seeds:
                raise RuntimeError(f"duplicate rollout seed: {seed}")
            seen_seeds.add(seed)
            success = bool(handle.attrs["success"])
            steps = int(handle.attrs["steps"])
            body = str(handle.attrs.get("body", "unknown"))
            policy = str(handle.attrs.get("model_path", manifest.get("model_path", "openvla")))
            query_steps = handle["query_steps"][:].astype(np.int64)
            hidden = handle["hidden"][:].astype(np.float16)
            terminal_hidden = handle["terminal_hidden"][:].astype(np.float16)
            actions = handle["action_chunks"][:].astype(np.float32)
            proprio = handle["proprio"][:].astype(np.float32)
            poses = handle["object_poses"][:].astype(np.float32)
            available_objects = decode_strings(handle["object_names"][:])
            canonical_names = decode_strings(handle["event_names"][:])
            canonical_steps = handle["event_steps"][:].astype(np.int64)
            predicate_sequence: np.ndarray | None = None
            dynamic_events: np.ndarray | None = None
            if calibration is not None:
                predicate_sequence = derive_atomic_predicates(
                    poses,
                    available_objects,
                    success,
                    calibration,
                )
                dynamic_events = dynamic_event_ids(
                    predicate_sequence, event_to_id
                )

            unknown_events = sorted(set(canonical_names) - set(events))
            if unknown_events:
                raise RuntimeError(f"unknown canonical events {unknown_events} in {path}")
            missing_objects = sorted(set(object_names) - set(available_objects))
            if missing_objects:
                raise RuntimeError(
                    f"requested object targets {missing_objects} absent in {path}; "
                    f"available={available_objects}"
                )
            if len(query_steps) != len(hidden) or len(query_steps) != len(actions):
                raise RuntimeError(f"query/hidden/action count mismatch in {path}")
            if hidden.ndim != 2 or terminal_hidden.shape != hidden.shape[1:]:
                raise RuntimeError(f"invalid hidden shapes in {path}")
            if actions.ndim != 3:
                raise RuntimeError(f"invalid action chunk shape in {path}: {actions.shape}")
            if poses.shape[0] != steps + 1 or proprio.shape[0] != steps + 1:
                raise RuntimeError(f"pose/proprio sequence does not end at terminal step in {path}")
            if len(canonical_names) != len(canonical_steps) or canonical_steps[0] != 0:
                raise RuntimeError(f"invalid canonical event sequence in {path}")
            if np.any(np.diff(query_steps) <= 0) or query_steps[0] != 0:
                raise RuntimeError(f"invalid query steps in {path}")
            _validate_finite("hidden", hidden, path)
            _validate_finite("terminal_hidden", terminal_hidden, path)
            _validate_finite("action_chunks", actions, path)
            _validate_finite("object_poses", poses, path)
            _validate_finite("proprio", proprio, path)

            if action_dim is None:
                chunk_size, action_dim = int(actions.shape[1]), int(actions.shape[2])
            if actions.shape[1:] != (chunk_size, action_dim):
                raise RuntimeError(f"action contract differs in {path}: {actions.shape[1:]}")
            object_indices = [available_objects.index(name) for name in object_names]
            episode_index = descriptor.index
            for query_index, query_step_value in enumerate(query_steps):
                query_step = int(query_step_value)
                if not 0 <= query_step < steps:
                    raise RuntimeError(f"query step {query_step} outside [0,{steps}) in {path}")
                end_step = min(query_step + chunk_size, steps)
                horizon = end_step - query_step
                mask = np.arange(chunk_size) < horizon
                if query_index + 1 < len(query_steps) and int(query_steps[query_index + 1]) == end_step:
                    next_hidden = hidden[query_index + 1]
                elif end_step == steps:
                    next_hidden = terminal_hidden
                else:
                    raise RuntimeError(
                        f"cannot align next hidden at step {end_step} for query {query_step} in {path}"
                    )
                current_event, next_event, duration, observed = event_transition_target(
                    query_step,
                    end_step,
                    steps,
                    canonical_names,
                    canonical_steps,
                    event_to_id,
                )
                post_event = current_event_at(
                    end_step,
                    canonical_names,
                    canonical_steps,
                    event_to_id,
                )
                if predicate_sequence is None or dynamic_events is None:
                    current_predicates = np.zeros(0, dtype=np.float32)
                    post_predicates = np.zeros(0, dtype=np.float32)
                    structured_current_event = current_event
                    structured_post_event = post_event
                else:
                    current_predicates = predicate_sequence[query_step]
                    post_predicates = predicate_sequence[end_step]
                    structured_current_event = int(dynamic_events[query_step])
                    structured_post_event = int(dynamic_events[end_step])
                relative_transition = int(
                    relative_transition_ids(
                        np.asarray([structured_current_event]),
                        np.asarray([structured_post_event]),
                    )[0]
                )
                # Positions are unambiguous across pose conventions.  Multiple
                # selected objects are flattened in manifest order.
                object_delta = (
                    poses[end_step, object_indices, :3]
                    - poses[query_step, object_indices, :3]
                ).reshape(-1)
                rows["hidden_t"].append(hidden[query_index])
                rows["next_hidden"].append(next_hidden)
                rows["action_chunks"].append(actions[query_index])
                rows["action_mask"].append(mask)
                rows["proprio"].append(proprio[query_step])
                rows["current_event_id"].append(current_event)
                rows["next_event_id"].append(next_event)
                rows["post_event_id"].append(post_event)
                rows["structured_current_event_id"].append(
                    structured_current_event
                )
                rows["structured_post_event_id"].append(structured_post_event)
                rows["relative_transition_id"].append(relative_transition)
                rows["current_predicates"].append(current_predicates)
                rows["post_predicates"].append(post_predicates)
                rows["reach"].append(float(observed))
                rows["duration"].append(duration)
                rows["duration_observed"].append(float(observed))
                rows["success"].append(float(success))
                rows["object_delta"].append(object_delta)
                rows["episode_index"].append(episode_index)
                rows["seed"].append(seed)
                rows["query_step"].append(query_step)
                rows["end_step"].append(end_step)
                rows["horizon"].append(float(horizon))
                rows["body"].append(body)
                rows["policy"].append(policy)

            episode_rows.append(
                {
                    "index": episode_index,
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "queries": len(query_steps),
                    "body": body,
                    "policy": policy,
                    "path": descriptor.relative_path,
                }
            )

    numeric_dtypes: dict[str, Any] = {
        "hidden_t": np.float16,
        "next_hidden": np.float16,
        "action_chunks": np.float32,
        "action_mask": np.bool_,
        "proprio": np.float32,
        "current_event_id": np.int64,
        "next_event_id": np.int64,
        "post_event_id": np.int64,
        "structured_current_event_id": np.int64,
        "structured_post_event_id": np.int64,
        "relative_transition_id": np.int64,
        "current_predicates": np.float32,
        "post_predicates": np.float32,
        "reach": np.float32,
        "duration": np.float32,
        "duration_observed": np.float32,
        "success": np.float32,
        "object_delta": np.float32,
        "episode_index": np.int64,
        "seed": np.int64,
        "query_step": np.int64,
        "end_step": np.int64,
        "horizon": np.float32,
    }
    arrays = {key: np.asarray(rows[key], dtype=dtype) for key, dtype in numeric_dtypes.items()}
    bodies = sorted(set(rows["body"]))
    policies = sorted(set(rows["policy"]))
    body_to_id = {name: index for index, name in enumerate(bodies)}
    policy_to_id = {name: index for index, name in enumerate(policies)}
    arrays["body_id"] = np.asarray([body_to_id[name] for name in rows["body"]], dtype=np.int64)
    arrays["policy_id"] = np.asarray([policy_to_id[name] for name in rows["policy"]], dtype=np.int64)
    return {
        "schema_version": CACHE_SCHEMA,
        "source": str(data_root.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "events": list(events),
        "predicate_names": list(DEFAULT_PREDICATES) if calibration is not None else [],
        "relative_transition_names": list(RELATIVE_TRANSITIONS),
        "event_spec": str(resolved_event_spec) if resolved_event_spec else None,
        "event_spec_sha256": event_spec_sha256,
        "task_calibration": dict(calibration) if calibration is not None else None,
        "task": task_name,
        "object_names": list(object_names),
        "object_target": "xyz_delta",
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "hidden_dim": int(arrays["hidden_t"].shape[1]),
        "proprio_dim": int(arrays["proprio"].shape[1]),
        "object_delta_dim": int(arrays["object_delta"].shape[1]),
        "body_to_id": body_to_id,
        "policy_to_id": policy_to_id,
        "episodes": episode_rows,
        "loaded_episode_seeds": sorted(selected_seeds),
        "split_seeds": {
            name: sorted(int(seed) for seed in split_seeds.get(name, []))
            for name in ("train", "validation", "test")
        },
        "sealed_test_access": (
            "manifest_identity_and_raw_file_sha256_only_no_episode_hdf5_open"
        ),
        "sealed_test_files": [
            descriptor_record(descriptor, include_sha256=True)
            for descriptor in sealed_test_descriptors
        ],
        "arrays": arrays,
    }


def load_or_build_cache(
    data_root: Path,
    cache_path: Path,
    events: Sequence[str],
    object_names: Sequence[str],
    rebuild: bool,
    *,
    manifest: Mapping[str, Any],
    episode_descriptors: Sequence[EpisodeDescriptor],
    sealed_test_descriptors: Sequence[EpisodeDescriptor],
    split_seeds: Mapping[str, Sequence[int]],
    event_spec_path: Path | None = None,
    require_predicates: bool = False,
) -> dict[str, Any]:
    expected_sha = sha256(data_root / "manifest.json")
    resolved_event_spec = event_spec_path
    if resolved_event_spec is None and manifest.get("event_spec"):
        resolved_event_spec = Path(str(manifest["event_spec"]))
        if not resolved_event_spec.is_absolute():
            resolved_event_spec = data_root / resolved_event_spec
    expected_event_spec_sha256: str | None = None
    expected_calibration: Mapping[str, Any] | None = None
    if resolved_event_spec is not None:
        if not resolved_event_spec.is_file():
            raise RuntimeError(f"event spec does not exist: {resolved_event_spec}")
        expected_event_spec_sha256 = sha256(resolved_event_spec)
        manifest_event_spec_sha256 = manifest.get("event_spec_sha256")
        if (
            manifest_event_spec_sha256
            and expected_event_spec_sha256 != manifest_event_spec_sha256
        ):
            raise RuntimeError(
                "event spec digest differs from rollout manifest: "
                f"{expected_event_spec_sha256} != {manifest_event_spec_sha256}"
            )
        event_spec_value = json.loads(
            resolved_event_spec.read_text(encoding="utf-8")
        )
        task = str(manifest.get("task", ""))
        expected_calibration = event_spec_value.get("calibration", {}).get(task)
        if expected_calibration is None:
            raise RuntimeError(
                f"event spec has no calibration for rollout task {task!r}"
            )
    elif require_predicates:
        raise RuntimeError(
            "structured event mode requires --event-spec or manifest['event_spec']"
        )
    expected_loaded = sorted(
        int(seed)
        for name in ("train", "validation")
        for seed in split_seeds[name]
    )
    expected_split = {
        name: sorted(int(seed) for seed in split_seeds.get(name, []))
        for name in ("train", "validation", "test")
    }
    expected_sealed = [
        descriptor_record(descriptor, include_sha256=True)
        for descriptor in sealed_test_descriptors
    ]
    if cache_path.is_file() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        checks = {
            "schema": cache.get("schema_version") == CACHE_SCHEMA,
            "source_manifest": cache.get("source_manifest_sha256") == expected_sha,
            "events": cache.get("events") == list(events),
            "objects": cache.get("object_names") == list(object_names),
            "predicates": (
                not require_predicates
                or cache.get("predicate_names") == list(DEFAULT_PREDICATES)
            ),
            "event_spec": (
                cache.get("event_spec_sha256") == expected_event_spec_sha256
            ),
            "task_calibration": cache.get("task_calibration") == expected_calibration,
            "loaded_episode_seeds": cache.get("loaded_episode_seeds") == expected_loaded,
            "split_seeds": cache.get("split_seeds") == expected_split,
            "sealed_test_files": cache.get("sealed_test_files") == expected_sealed,
        }
        if not all(checks.values()):
            raise RuntimeError(f"transition cache contract mismatch: {checks}; use --rebuild-cache")
        cached_seeds = set(int(seed) for seed in cache["arrays"]["seed"])
        leaked = sorted(cached_seeds & set(expected_split["test"]))
        if leaked:
            raise RuntimeError(f"sealed test seeds present in transition cache: {leaked}")
        return cache
    cache = build_transition_cache(
        data_root,
        events,
        object_names,
        manifest,
        episode_descriptors,
        sealed_test_descriptors,
        split_seeds,
        event_spec_path=event_spec_path,
        require_predicates=require_predicates,
    )
    atomic_torch_save(cache_path, cache)
    return cache


def read_split_manifest(
    path: Path, descriptors: Sequence[EpisodeDescriptor]
) -> dict[str, list[int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    available = {descriptor.seed for descriptor in descriptors}
    result: dict[str, list[int]] = {}
    for name in ("train", "validation", "test"):
        if name not in value:
            if name == "test":
                result[name] = []
                continue
            raise RuntimeError(f"split manifest lacks {name}: {path}")
        seeds = [int(row["seed"] if isinstance(row, dict) else row) for row in value[name]]
        if len(set(seeds)) != len(seeds):
            raise RuntimeError(f"duplicate seeds in split {name}")
        unknown = sorted(set(seeds) - available)
        if unknown:
            raise RuntimeError(f"split {name} contains unknown seeds: {unknown}")
        result[name] = seeds
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(set(result[left]) & set(result[right]))
        if overlap:
            raise RuntimeError(f"episode leakage between {left}/{right}: {overlap}")
    assigned = set().union(*(set(values) for values in result.values()))
    omitted = sorted(available - assigned)
    if omitted:
        raise RuntimeError(f"split manifest omits rollout seeds: {omitted}")
    return result


def make_default_split(
    descriptors: Sequence[EpisodeDescriptor], seed: int
) -> dict[str, list[int]]:
    """Create deterministic 100/25/25-style splits without reading labels."""

    if len(descriptors) < 3:
        raise RuntimeError("default sealed split requires at least three episodes")
    rng = np.random.default_rng(seed)
    seeds = np.asarray(sorted(descriptor.seed for descriptor in descriptors), dtype=np.int64)
    rng.shuffle(seeds)
    holdout = max(1, round(len(seeds) / 6))
    holdout = min(holdout, (len(seeds) - 1) // 2)
    test = seeds[:holdout]
    validation = seeds[holdout : 2 * holdout]
    train = seeds[2 * holdout :]
    return {
        "train": sorted(int(value) for value in train),
        "validation": sorted(int(value) for value in validation),
        "test": sorted(int(value) for value in test),
    }


def transition_indices(arrays: Mapping[str, np.ndarray], seeds: Iterable[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(arrays["seed"], np.asarray(list(seeds), dtype=np.int64)))


class TransitionDataset(Dataset):
    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        indices: np.ndarray,
        object_mean: np.ndarray,
        object_std: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.object_mean = object_mean.astype(np.float32)
        self.object_std = object_std.astype(np.float32)
        # Map each transition to the complete OpenVLA query prefix from the same
        # episode.  The shadow GRU was trained over episode prefixes; resetting
        # it on every query silently changes the semantic-state contract.
        self.history_indices: dict[int, np.ndarray] = {}
        for index in self.indices:
            same_episode = np.flatnonzero(
                (arrays["seed"] == arrays["seed"][index])
                & (arrays["query_step"] <= arrays["query_step"][index])
            )
            order = np.argsort(arrays["query_step"][same_episode])
            self.history_indices[int(index)] = same_episode[order]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        history = self.arrays["hidden_t"][self.history_indices[index]]
        next_history = np.concatenate(
            [history, self.arrays["next_hidden"][index][None]], axis=0
        )
        result = {
            key: torch.from_numpy(np.asarray(self.arrays[key][index]))
            for key in (
                "action_chunks",
                "action_mask",
                "proprio",
                "current_event_id",
                "next_event_id",
                "post_event_id",
                "structured_current_event_id",
                "structured_post_event_id",
                "relative_transition_id",
                "current_predicates",
                "post_predicates",
                "reach",
                "duration",
                "duration_observed",
                "success",
                "body_id",
                "policy_id",
                "horizon",
            )
        }
        result["hidden_t"] = torch.from_numpy(history)
        result["next_hidden"] = torch.from_numpy(next_history)
        normalized = (
            self.arrays["object_delta"][index].astype(np.float32) - self.object_mean
        ) / self.object_std
        result["object_delta"] = torch.from_numpy(normalized)
        return result


def collate_transitions(items: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in items[0]:
        if key in {"hidden_t", "next_hidden"}:
            sequences = [item[key] for item in items]
            result[key] = pad_sequence(sequences, batch_first=True)
            mask_name = "history_mask" if key == "hidden_t" else "next_history_mask"
            lengths = torch.as_tensor([len(value) for value in sequences])
            result[mask_name] = (
                torch.arange(result[key].shape[1])[None] < lengths[:, None]
            )
        else:
            result[key] = torch.stack([item[key] for item in items])
    return result


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    result["hidden_t"] = result["hidden_t"].float()
    result["next_hidden"] = result["next_hidden"].float()
    return result


def lognormal_nll(
    log_mean: torch.Tensor,
    log_scale: torch.Tensor,
    duration: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Log-normal likelihood with right-censoring at the executed horizon."""

    target = torch.log1p(duration.clamp_min(0.0))
    scale = torch.exp(log_scale.clamp(-5.0, 3.0)).clamp_min(1e-4)
    z = (target - log_mean) / scale
    observed_nll = 0.5 * z.square() + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)
    censored_nll = -torch.special.log_ndtr(-z)
    return torch.where(observed.bool(), observed_nll, censored_nll).mean()


def _gather_event(values: torch.Tensor, event_ids: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values
    return values.gather(1, event_ids[:, None]).squeeze(1)


def encode_semantic(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_state"):
        return model.encode_state(hidden)  # type: ignore[attr-defined]
    if hasattr(model, "semantic_encoder"):
        return model.semantic_encoder(hidden)  # type: ignore[attr-defined]
    if hasattr(model, "semantic"):
        mask = torch.ones((len(hidden), 1), device=hidden.device, dtype=torch.bool)
        return model.semantic(hidden[:, None], mask)[:, 0]  # type: ignore[attr-defined]
    raise RuntimeError("event world model does not expose encode_state/semantic_encoder")


def forward_model(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    structured = bool(getattr(model.config, "structured_events", False))  # type: ignore[attr-defined]
    return model(
        batch["hidden_t"],
        batch["action_chunks"],
        history_mask=batch["history_mask"],
        action_mask=batch["action_mask"],
        proprio=batch["proprio"],
        body_id=batch["body_id"],
        policy_id=batch["policy_id"],
        current_event_id=(
            batch["structured_current_event_id"]
            if structured
            else batch["current_event_id"]
        ),
        clock_event_id=batch["current_event_id"] if structured else None,
        current_predicates=batch["current_predicates"] if structured else None,
        dt=batch["horizon"],
    )


def masked_weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy that remains finite when a minibatch has no valid label."""

    losses = F.cross_entropy(logits, target, reduction="none")
    effective_weight = torch.ones_like(losses)
    if class_weight is not None:
        effective_weight = effective_weight * class_weight[target].to(losses)
    if sample_mask is not None:
        effective_weight = effective_weight * sample_mask.to(
            device=losses.device, dtype=losses.dtype
        )
    valid = effective_weight > 0
    if not bool(valid.any()):
        anchor = logits.reshape(-1)[0]
        # Do not reduce every masked finfo.min logit: their finite sum can
        # overflow to -inf and turn ``* 0`` back into NaN.
        anchor = torch.where(torch.isfinite(anchor), anchor, anchor.new_zeros(()))
        return anchor * 0.0
    denominator = effective_weight[valid].sum()
    # Index before multiplying so an invalid-label infinite loss cannot create
    # ``inf * 0 = NaN`` in a partially supported minibatch.
    return (losses[valid] * effective_weight[valid]).sum() / denominator


def compute_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    reach_pos_weight: torch.Tensor,
    success_pos_weight: torch.Tensor,
    event_class_weight: torch.Tensor,
    destination_class_weight: torch.Tensor | None = None,
    relative_class_weight: torch.Tensor | None = None,
    relative_supported: torch.Tensor | None = None,
    predicate_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = forward_model(model, batch)
    structured = bool(getattr(model.config, "structured_events", False))  # type: ignore[attr-defined]
    if structured:
        event = masked_weighted_cross_entropy(
            output["next_event_logits"],
            batch["structured_post_event_id"],
            class_weight=event_class_weight,
        )
        if relative_class_weight is None or relative_supported is None:
            raise ValueError("structured loss requires relative class metadata")
        relative_target = batch["relative_transition_id"]
        relative = masked_weighted_cross_entropy(
            output["relative_transition_logits"],
            relative_target,
            class_weight=relative_class_weight,
            sample_mask=relative_supported[relative_target],
        )
        destination = masked_weighted_cross_entropy(
            output["next_reached_event_logits"],
            batch["next_event_id"],
            class_weight=destination_class_weight,
            # A censored next event is unknown, not a fabricated self-loop.
            sample_mask=batch["duration_observed"],
        )
        if predicate_pos_weight is None:
            raise ValueError("structured loss requires predicate_pos_weight")
        predicate = F.binary_cross_entropy_with_logits(
            output["post_predicate_logits"],
            batch["post_predicates"],
            pos_weight=predicate_pos_weight,
        )
    else:
        event = F.cross_entropy(
            output["next_event_logits"],
            batch["next_event_id"],
            weight=event_class_weight,
        )
    reach = F.binary_cross_entropy_with_logits(
        output["reach_logit"], batch["reach"], pos_weight=reach_pos_weight
    )
    duration_mean = _gather_event(
        output["duration_log_mean"], batch["current_event_id"]
    )
    duration_scale = _gather_event(
        output["duration_log_scale"], batch["current_event_id"]
    )
    duration = lognormal_nll(
        duration_mean,
        duration_scale,
        batch["duration"],
        batch["duration_observed"],
    )
    success = F.binary_cross_entropy_with_logits(
        output["success_logit"], batch["success"], pos_weight=success_pos_weight
    )
    # The current factual dataset supervises terminal failure/success.  The
    # third recovery class remains present in the reusable core and will gain
    # positive labels from schema-v3 branches; for now it must not remain a
    # random head because its entropy participates in deployment uncertainty.
    outcome_logits = output["outcome_logits"]
    if structured and not bool(
        getattr(model.config, "recovery_supervised", False)  # type: ignore[attr-defined]
    ):
        # Factual data has no operational recovery label.  Optimize only the
        # observed failure/success subspace instead of treating every rollout
        # as hard negative evidence for recovery.
        outcome_logits = outcome_logits[:, :2]
    outcome = F.cross_entropy(outcome_logits, batch["success"].long())
    object_scale = torch.exp(output["object_delta_log_scale"].clamp(-5.0, 3.0))
    object_delta = (
        0.5
        * torch.square(
            (batch["object_delta"] - output["object_delta_mean"])
            / object_scale.clamp_min(1e-4)
        )
        + torch.log(object_scale.clamp_min(1e-4))
    ).mean()
    with torch.no_grad():
        next_semantic = model.encode_state(  # type: ignore[attr-defined]
            batch["next_hidden"], batch["next_history_mask"]
        )
    latent = (1.0 - F.cosine_similarity(
        output["predicted_next_semantic"], next_semantic.detach(), dim=-1
    )).mean()
    pieces = {
        "event": event,
        "reach": reach,
        "duration": duration,
        "success": success,
        "outcome": outcome,
        "object": object_delta,
        "latent": latent,
    }
    if structured:
        pieces.update(
            {
                "relative": relative,
                "destination": destination,
                "predicate": predicate,
            }
        )
    total = sum(weights.get(name, 0.0) * value for name, value in pieces.items())
    pieces["total"] = total
    return total, pieces


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if not len(positive) or not len(negative):
        return None
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).sum() + 0.5 * (comparison == 0).sum()) / comparison.size)


def multiclass_macro_f1(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> float:
    scores = []
    for class_id in range(num_classes):
        true_positive = int(((labels == class_id) & (predictions == class_id)).sum())
        false_positive = int(((labels != class_id) & (predictions == class_id)).sum())
        false_negative = int(((labels == class_id) & (predictions != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        # Classes absent from both labels and predictions carry no validation
        # evidence and are omitted instead of receiving a misleading perfect F1.
        if denominator:
            scores.append(2.0 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    true_positive = int(((labels > 0.5) & (predictions > 0.5)).sum())
    false_positive = int(((labels <= 0.5) & (predictions > 0.5)).sum())
    false_negative = int(((labels > 0.5) & (predictions <= 0.5)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else None


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    object_mean: np.ndarray,
    object_std: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    structured = bool(getattr(model.config, "structured_events", False))  # type: ignore[attr-defined]
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "next_event",
            "next_event_prediction",
            "reach",
            "reach_probability",
            "success",
            "success_probability",
            "duration",
            "duration_prediction",
            "duration_observed",
            "object_target",
            "object_prediction",
        )
    }
    if structured:
        rows.update(
            {
                key: []
                for key in (
                    "relative_transition",
                    "relative_transition_prediction",
                    "structured_current_event",
                    "next_reached_event",
                    "next_reached_event_prediction",
                    "post_predicate",
                    "post_predicate_probability",
                )
            }
        )
    latent_cosines = []
    num_events: int | None = None
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = forward_model(model, batch)
        num_events = int(output["next_event_logits"].shape[-1])
        duration_mean = _gather_event(output["duration_log_mean"], batch["current_event_id"])
        duration_prediction = torch.expm1(duration_mean).clamp_min(0.0)
        next_semantic = model.encode_state(  # type: ignore[attr-defined]
            batch["next_hidden"], batch["next_history_mask"]
        )
        latent_cosines.append(
            F.cosine_similarity(output["predicted_next_semantic"], next_semantic, dim=-1)
            .cpu()
            .numpy()
        )
        rows["next_event"].append(
            (
                batch["structured_post_event_id"]
                if structured
                else batch["next_event_id"]
            ).cpu().numpy()
        )
        rows["next_event_prediction"].append(
            output["next_event_logits"].argmax(-1).cpu().numpy()
        )
        rows["reach"].append(batch["reach"].cpu().numpy())
        rows["reach_probability"].append(torch.sigmoid(output["reach_logit"]).cpu().numpy())
        rows["success"].append(batch["success"].cpu().numpy())
        rows["success_probability"].append(
            torch.sigmoid(output["success_logit"]).cpu().numpy()
        )
        rows["duration"].append(batch["duration"].cpu().numpy())
        rows["duration_prediction"].append(duration_prediction.cpu().numpy())
        rows["duration_observed"].append(batch["duration_observed"].cpu().numpy())
        rows["object_target"].append(batch["object_delta"].cpu().numpy())
        rows["object_prediction"].append(output["object_delta_mean"].cpu().numpy())
        if structured:
            rows["relative_transition"].append(
                batch["relative_transition_id"].cpu().numpy()
            )
            rows["relative_transition_prediction"].append(
                output["relative_transition_logits"].argmax(-1).cpu().numpy()
            )
            rows["structured_current_event"].append(
                batch["structured_current_event_id"].cpu().numpy()
            )
            rows["next_reached_event"].append(
                batch["next_event_id"].cpu().numpy()
            )
            rows["next_reached_event_prediction"].append(
                output["next_reached_event_logits"].argmax(-1).cpu().numpy()
            )
            rows["post_predicate"].append(
                batch["post_predicates"].cpu().numpy()
            )
            rows["post_predicate_probability"].append(
                torch.sigmoid(output["post_predicate_logits"]).cpu().numpy()
            )
    values = {key: np.concatenate(parts) for key, parts in rows.items()}
    observed = values["duration_observed"] > 0.5
    raw_object_target = values["object_target"] * object_std + object_mean
    raw_object_prediction = values["object_prediction"] * object_std + object_mean
    assert num_events is not None
    metrics: dict[str, Any] = {
        "transitions": int(len(values["reach"])),
        "event_accuracy": float(
            np.mean(values["next_event_prediction"] == values["next_event"])
        ),
        "event_macro_f1": multiclass_macro_f1(
            values["next_event"], values["next_event_prediction"], num_events
        ),
        "reach_auc": binary_auc(values["reach"], values["reach_probability"]),
        "reach_brier": float(np.mean(np.square(values["reach_probability"] - values["reach"]))),
        "success_auc": binary_auc(values["success"], values["success_probability"]),
        "success_brier": float(
            np.mean(np.square(values["success_probability"] - values["success"]))
        ),
        "duration_observed_count": int(observed.sum()),
        "duration_observed_mae_steps": (
            float(np.mean(np.abs(values["duration_prediction"][observed] - values["duration"][observed])))
            if observed.any()
            else None
        ),
        "object_delta_mae": float(np.mean(np.abs(raw_object_prediction - raw_object_target))),
        "future_semantic_cosine": float(np.mean(np.concatenate(latent_cosines))),
    }
    if structured:
        relative_prediction = values["relative_transition_prediction"]
        relative_target = values["relative_transition"]
        observed_target = values["duration_observed"] > 0.5
        predicate_target = values["post_predicate"]
        predicate_probability = values["post_predicate_probability"]
        predicate_metrics = {}
        predicate_f1_values = []
        for index, name in enumerate(model.config.predicate_names):  # type: ignore[attr-defined]
            f1 = binary_f1(
                predicate_target[:, index],
                predicate_probability[:, index] >= 0.5,
            )
            if f1 is not None:
                predicate_f1_values.append(f1)
            predicate_metrics[str(name)] = {
                "positive_count": int((predicate_target[:, index] > 0.5).sum()),
                "f1": f1,
                "auc": binary_auc(
                    predicate_target[:, index], predicate_probability[:, index]
                ),
                "brier": float(
                    np.mean(
                        np.square(
                            predicate_probability[:, index]
                            - predicate_target[:, index]
                        )
                    )
                ),
            }
        predicted_absolute_relative = relative_transition_ids(
            values["structured_current_event"],
            values["next_event_prediction"],
        )
        metrics.update(
            {
                "relative_transition_accuracy": float(
                    np.mean(relative_prediction == relative_target)
                ),
                "relative_transition_macro_f1": multiclass_macro_f1(
                    relative_target,
                    relative_prediction,
                    len(RELATIVE_TRANSITIONS),
                ),
                "relative_absolute_consistency": float(
                    np.mean(predicted_absolute_relative == relative_prediction)
                ),
                "next_reached_event_observed_accuracy": (
                    float(
                        np.mean(
                            values["next_reached_event_prediction"][observed_target]
                            == values["next_reached_event"][observed_target]
                        )
                    )
                    if observed_target.any()
                    else None
                ),
                "next_reached_event_observed_macro_f1": (
                    multiclass_macro_f1(
                        values["next_reached_event"][observed_target],
                        values["next_reached_event_prediction"][observed_target],
                        num_events,
                    )
                    if observed_target.any()
                    else None
                ),
                "predicate_macro_f1": (
                    float(np.mean(predicate_f1_values))
                    if predicate_f1_values
                    else 0.0
                ),
                "predicate_metrics": predicate_metrics,
            }
        )
    return metrics


def initialize_shadow_semantic(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    source = {
        key.removeprefix("semantic."): value
        for key, value in source.items()
        if key.startswith("semantic.")
    }
    module = getattr(model, "semantic_encoder", None)
    if module is None:
        module = getattr(model, "semantic", None)
    if module is None:
        raise RuntimeError("model has no semantic encoder for shadow initialization")
    missing, unexpected = module.load_state_dict(source, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"shadow semantic state mismatch: missing={missing}, unexpected={unexpected}"
        )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "frozen": True}


def make_model_config(cache: Mapping[str, Any], structured_events: bool = False) -> Any:
    from openvla_etsf_event_world_model import EventWorldModelConfig

    available = {field.name for field in dataclasses.fields(EventWorldModelConfig)}
    candidates = {
        "state_input_dim": int(cache["hidden_dim"]),
        "hidden_dim": int(cache["hidden_dim"]),
        "input_dim": int(cache["hidden_dim"]),
        "action_dim": int(cache["action_dim"]),
        "chunk_size": int(cache["chunk_size"]),
        "num_events": len(cache["events"]),
        "event_count": len(cache["events"]),
        "event_names": tuple(cache["events"]),
        "predicate_names": tuple(cache.get("predicate_names", [])),
        "relative_transition_names": tuple(
            cache.get("relative_transition_names", RELATIVE_TRANSITIONS)
        ),
        "structured_events": structured_events,
        "object_delta_dim": int(cache["object_delta_dim"]),
        "object_dim": int(cache["object_delta_dim"]),
        "proprio_dim": int(cache["proprio_dim"]),
        "num_bodies": len(cache["body_to_id"]),
        "num_policies": len(cache["policy_to_id"]),
    }
    return EventWorldModelConfig(**{key: value for key, value in candidates.items() if key in available})


def build_model(
    cache: Mapping[str, Any],
    shadow_checkpoint: Path | None,
    structured_events: bool = False,
) -> tuple[nn.Module, Any, Any]:
    from openvla_etsf_event_world_model import ActionConditionedEventWorldModel

    config = make_model_config(cache, structured_events=structured_events)
    model = ActionConditionedEventWorldModel(config)
    shadow_info = None
    if shadow_checkpoint is not None:
        shadow_info = initialize_shadow_semantic(model, shadow_checkpoint)
    return model, config, shadow_info


def class_weights(
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    num_events: int,
    device: torch.device,
    *,
    structured: bool,
    min_relative_support: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    def balanced(labels: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(labels, minlength=classes).astype(np.float64)
        supported = counts > 0
        weights = np.zeros(classes, dtype=np.float64)
        weights[supported] = np.sqrt(counts.sum() / counts[supported])
        weights[supported] /= weights[supported].mean()
        return weights, counts

    reach = arrays["reach"][train_indices]
    success = arrays["success"][train_indices]
    event = arrays[
        "structured_post_event_id" if structured else "next_event_id"
    ][train_indices]
    reach_pos = min(float((reach < 0.5).sum() / max((reach > 0.5).sum(), 1)), 10.0)
    success_pos = min(float((success < 0.5).sum() / max((success > 0.5).sum(), 1)), 10.0)
    event_weights, _ = balanced(event, num_events)
    observed = arrays["duration_observed"][train_indices] > 0.5
    destination_weights, _ = balanced(
        arrays["next_event_id"][train_indices][observed], num_events
    )
    relative = arrays["relative_transition_id"][train_indices]
    relative_weights, relative_counts = balanced(
        relative, len(RELATIVE_TRANSITIONS)
    )
    relative_supported = relative_counts >= min_relative_support
    relative_weights[~relative_supported] = 0.0
    predicates = arrays["post_predicates"][train_indices]
    predicate_positive = predicates.sum(0)
    predicate_negative = len(predicates) - predicate_positive
    predicate_pos_weight = np.minimum(
        predicate_negative / np.maximum(predicate_positive, 1.0), 20.0
    ).astype(np.float32)
    return (
        torch.tensor(reach_pos, device=device, dtype=torch.float32),
        torch.tensor(success_pos, device=device, dtype=torch.float32),
        torch.tensor(event_weights, device=device, dtype=torch.float32),
        torch.tensor(destination_weights, device=device, dtype=torch.float32),
        torch.tensor(relative_weights, device=device, dtype=torch.float32),
        torch.tensor(relative_supported, device=device, dtype=torch.bool),
        torch.tensor(predicate_pos_weight, device=device, dtype=torch.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--shadow-checkpoint", type=Path)
    parser.add_argument(
        "--event-mode",
        choices=["absolute", "structured"],
        default="structured",
        help=(
            "structured predicts dynamic post-chunk events and predicates; "
            "absolute retains the v1 checkpoint contract"
        ),
    )
    parser.add_argument(
        "--event-spec",
        type=Path,
        help="Override the rollout manifest event-spec used for predicate labels.",
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--events", nargs="+", default=list(DEFAULT_EVENTS))
    parser.add_argument(
        "--object-names",
        nargs="+",
        default=["can"],
        help="Tracked objects whose xyz deltas are predicted (current task: can).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1000,
        help="Stop after this many steps without a better validation checkpoint; 0 disables.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=1.0)
    parser.add_argument("--destination-weight", type=float, default=0.25)
    parser.add_argument("--predicate-weight", type=float, default=0.5)
    parser.add_argument(
        "--min-relative-class-support",
        type=int,
        default=5,
        help=(
            "Do not optimize a relative class with fewer train examples; "
            "this prevents a single regression from being presented as learned recovery."
        ),
    )
    parser.add_argument("--reach-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.5)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=0.25)
    parser.add_argument("--object-weight", type=float, default=0.5)
    parser.add_argument("--latent-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda:0" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    amp_name = args.amp
    if amp_name == "auto":
        amp_name = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "off"
        )
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and amp_name != "off"

    # Establish the episode split from label-free manifest descriptors before
    # opening any HDF5 file.  This is the sealed-test boundary.
    rollout_manifest, descriptors = read_rollout_descriptors(args.data)
    if args.split_manifest:
        splits = read_split_manifest(args.split_manifest, descriptors)
        split_source = str(args.split_manifest)
    else:
        splits = make_default_split(descriptors, SPLIT_SEED)
        split_source = "generated_label_free_episode_100_25_25_style"
    descriptor_by_seed = {descriptor.seed: descriptor for descriptor in descriptors}
    loaded_descriptors = [
        descriptor_by_seed[seed]
        for name in ("train", "validation")
        for seed in splits[name]
    ]
    sealed_test_descriptors = [
        descriptor_by_seed[seed] for seed in splits.get("test", [])
    ]
    cache_path = args.cache or (args.output / "query_transitions.pt")
    cache = load_or_build_cache(
        args.data,
        cache_path,
        args.events,
        args.object_names,
        args.rebuild_cache,
        manifest=rollout_manifest,
        episode_descriptors=loaded_descriptors,
        sealed_test_descriptors=sealed_test_descriptors,
        split_seeds=splits,
        event_spec_path=args.event_spec,
        require_predicates=args.event_mode == "structured",
    )
    arrays = cache["arrays"]
    train_indices = transition_indices(arrays, splits["train"])
    validation_indices = transition_indices(arrays, splits["validation"])
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("train and validation must both contain transitions")
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("transition-level train/validation leakage")
    cached_seed_set = set(int(seed) for seed in arrays["seed"])
    leaked_test_seeds = sorted(cached_seed_set & set(splits.get("test", [])))
    if leaked_test_seeds:
        raise RuntimeError(
            f"sealed test episode datasets leaked into cache: {leaked_test_seeds}"
        )

    object_mean = arrays["object_delta"][train_indices].mean(0).astype(np.float32)
    object_std = np.maximum(
        arrays["object_delta"][train_indices].std(0), 1e-4
    ).astype(np.float32)
    train_duration_observed = arrays["duration_observed"][train_indices] > 0.5
    duration_selection_scale = float(cache["chunk_size"])
    if train_duration_observed.any():
        duration_selection_scale = float(
            max(
                arrays["duration"][train_indices][train_duration_observed].mean(),
                cache["chunk_size"],
            )
        )
    object_selection_scale = float(max(object_std.mean(), 1e-3))
    train_dataset = TransitionDataset(arrays, train_indices, object_mean, object_std)
    validation_dataset = TransitionDataset(
        arrays, validation_indices, object_mean, object_std
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_transitions,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=False, **loader_kwargs
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    structured_events = args.event_mode == "structured"
    model, config, shadow_info = build_model(
        cache, args.shadow_checkpoint, structured_events=structured_events
    )
    model = model.to(device)
    valid_actions = arrays["action_chunks"][train_indices][
        arrays["action_mask"][train_indices]
    ]
    action_mean = torch.as_tensor(valid_actions.mean(0), device=device)
    action_std = torch.as_tensor(
        np.maximum(valid_actions.std(0), 1e-4), device=device
    )
    model.action_encoder.set_normalization(action_mean, action_std)  # type: ignore[attr-defined]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_name == "fp16"
    )
    (
        reach_pos,
        success_pos,
        event_weight,
        destination_weight,
        relative_weight,
        relative_supported,
        predicate_pos_weight,
    ) = class_weights(
        arrays,
        train_indices,
        len(cache["events"]),
        device,
        structured=structured_events,
        min_relative_support=args.min_relative_class_support,
    )
    loss_weights = {
        "event": args.event_weight,
        "relative": args.relative_weight,
        "destination": args.destination_weight,
        "predicate": args.predicate_weight,
        "reach": args.reach_weight,
        "duration": args.duration_weight,
        "success": args.success_weight,
        "outcome": args.outcome_weight,
        "object": args.object_weight,
        "latent": args.latent_weight,
    }
    contract = {
        "cache_schema": CACHE_SCHEMA,
        "training_seed": args.seed,
        "event_mode": args.event_mode,
        "source_manifest_sha256": cache["source_manifest_sha256"],
        "event_spec_sha256": cache.get("event_spec_sha256"),
        "split_source": split_source,
        "events": cache["events"],
        "predicate_names": cache.get("predicate_names", []),
        "relative_transition_names": cache.get(
            "relative_transition_names", []
        ),
        "object_names": cache["object_names"],
        "object_target": cache["object_target"],
        "body_to_id": cache["body_to_id"],
        "policy_to_id": cache["policy_to_id"],
        "train_seeds": sorted(splits["train"]),
        "validation_seeds": sorted(splits["validation"]),
        "sealed_test_seeds": sorted(splits.get("test", [])),
        "sealed_test_access": cache["sealed_test_access"],
        "sealed_test_files": cache["sealed_test_files"],
        "predicate_contract": {
            "names": cache.get("predicate_names", []),
            "derivation": "derive_atomic_predicates_v1",
            "source": "simulator_object_poses_at_query_step",
            "event_spec_sha256": cache.get("event_spec_sha256"),
            "task_calibration": cache.get("task_calibration"),
            "online_requires_explicit_predicates": True,
            "missing_policy": "error",
        },
    }
    atomic_json(
        args.output / "split_manifest.json",
        {
            "source": split_source,
            "train": sorted(splits["train"]),
            "validation": sorted(splits["validation"]),
            "test": sorted(splits.get("test", [])),
            "test_policy": (
                "sealed_manifest_identity_and_raw_sha256_only_"
                "episode_hdf5_never_opened_not_evaluated"
            ),
        },
    )
    atomic_json(
        args.output / "data_audit.json",
        {
            "manifest_episode_descriptors": len(descriptors),
            "loaded_train_validation_episodes": len(cache["episodes"]),
            "loaded_train_validation_transitions": int(len(arrays["seed"])),
            "train_episodes": len(splits["train"]),
            "validation_episodes": len(splits["validation"]),
            "sealed_test_episodes": len(splits.get("test", [])),
            "sealed_test_episode_datasets_opened": 0,
            "sealed_test_transition_count": "unknown_not_loaded",
            "sealed_test_access": cache["sealed_test_access"],
            "sealed_test_files": cache["sealed_test_files"],
            "train_transitions": int(len(train_indices)),
            "validation_transitions": int(len(validation_indices)),
            "train_validation_partial_terminal_chunks": int(
                (arrays["horizon"] < cache["chunk_size"]).sum()
            ),
            "train_validation_observed_event_transitions": int(
                arrays["duration_observed"].sum()
            ),
            "train_validation_right_censored_transitions": int(
                (1.0 - arrays["duration_observed"]).sum()
            ),
            "train_validation_stored_next_equals_current_but_censored": int(
                (
                    (arrays["next_event_id"] == arrays["current_event_id"])
                    & (arrays["duration_observed"] < 0.5)
                ).sum()
            ),
            "train_structured_event_counts": np.bincount(
                arrays["structured_post_event_id"][train_indices],
                minlength=len(cache["events"]),
            ).tolist(),
            "validation_structured_event_counts": np.bincount(
                arrays["structured_post_event_id"][validation_indices],
                minlength=len(cache["events"]),
            ).tolist(),
            "train_relative_transition_counts": np.bincount(
                arrays["relative_transition_id"][train_indices],
                minlength=len(RELATIVE_TRANSITIONS),
            ).tolist(),
            "validation_relative_transition_counts": np.bincount(
                arrays["relative_transition_id"][validation_indices],
                minlength=len(RELATIVE_TRANSITIONS),
            ).tolist(),
            "relative_classes_optimized": {
                name: bool(relative_supported[index].item())
                for index, name in enumerate(RELATIVE_TRANSITIONS)
            },
            "validation_selection_scales": {
                "duration_steps": duration_selection_scale,
                "object_delta": object_selection_scale,
            },
            "contract": contract,
        },
    )

    step = 0
    best_step = 0
    best_score = math.inf
    best_validation: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("contract") != contract:
            raise RuntimeError("resume checkpoint data/split contract differs from this run")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scaler") is not None:
            scaler.load_state_dict(resume["scaler"])
        step = int(resume["step"])
        best_step = int(resume.get("best_step", 0))
        best_score = float(resume.get("best_score", math.inf))
        last_validation = resume.get("last_validation")
        if step < 0 or step > args.steps:
            raise RuntimeError(
                f"resume step {step} is outside requested training range [0,{args.steps}]"
            )
        if best_step < 0 or best_step > step:
            raise RuntimeError(
                f"resume best_step {best_step} is outside completed range [0,{step}]"
            )
        best_path = args.output / "event_world_model_best.pt"
        if best_step > 0 and not best_path.is_file():
            raise RuntimeError(
                "resume checkpoint references a best step but best checkpoint is missing"
            )
        if best_path.is_file():
            previous_best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            if previous_best.get("contract") != contract:
                raise RuntimeError("best checkpoint contract differs from resume contract")
            if int(previous_best.get("step", -1)) != best_step:
                raise RuntimeError("best checkpoint step differs from resume best_step")
            previous_score = float(previous_best.get("best_score", math.inf))
            if not math.isclose(previous_score, best_score, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError("best checkpoint score differs from resume best_score")
            best_validation = previous_best.get("validation")
            if not isinstance(best_validation, Mapping):
                raise RuntimeError("best checkpoint has no validation metrics")

    log_path = args.output / "train_log.jsonl"
    started = time.time()
    iterator = iter(train_loader)
    stopped_early = False
    while step < args.steps:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, pieces = compute_loss(
                model,
                batch,
                loss_weights,
                reach_pos,
                success_pos,
                event_weight,
                destination_weight,
                relative_weight,
                relative_supported,
                predicate_pos_weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {pieces}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        validate_now = step % args.eval_every == 0 or step == args.steps
        save_now = step % args.save_every == 0 or validate_now
        row: dict[str, Any] = {
            "step": step,
            "wall_seconds": time.time() - started,
            **{f"train_{key}": float(value.detach()) for key, value in pieces.items()},
        }
        if validate_now:
            last_validation = evaluate(
                model, validation_loader, device, object_mean, object_std
            )
            # Selection uses validation only and covers every trained factor;
            # otherwise the heavily censored clock can improve likelihood while
            # its observed-event MAE silently becomes unusable.
            duration_mae = last_validation["duration_observed_mae_steps"]
            if duration_mae is None:
                raise RuntimeError(
                    "validation split contains no observed event duration; "
                    "best-checkpoint selection is undefined"
                )
            selection_components = {
                "reach_brier": last_validation["reach_brier"],
                "success_brier": last_validation["success_brier"],
                "event_macro_error": 1.0 - last_validation["event_macro_f1"],
                "future_semantic_cosine_error": 1.0
                - last_validation["future_semantic_cosine"],
                "duration_relative_mae": duration_mae / duration_selection_scale,
                "object_relative_mae": last_validation["object_delta_mae"]
                / object_selection_scale,
            }
            if structured_events:
                selection_components.update(
                    {
                        "relative_macro_error": 1.0
                        - last_validation["relative_transition_macro_f1"],
                        "predicate_macro_error": 1.0
                        - last_validation["predicate_macro_f1"],
                        "next_reached_macro_error": 1.0
                        - (
                            last_validation[
                                "next_reached_event_observed_macro_f1"
                            ]
                            or 0.0
                        ),
                    }
                )
            non_finite_components = {
                name: value
                for name, value in selection_components.items()
                if not math.isfinite(float(value))
            }
            if non_finite_components:
                raise RuntimeError(
                    "non-finite validation selection components: "
                    f"{non_finite_components}"
                )
            selection_score = (
                selection_components["reach_brier"]
                + selection_components["success_brier"]
                + selection_components["event_macro_error"]
                + selection_components["future_semantic_cosine_error"]
                + 0.10 * selection_components["duration_relative_mae"]
                + 0.10 * selection_components["object_relative_mae"]
            )
            if structured_events:
                selection_score += (
                    0.50 * selection_components["relative_macro_error"]
                    + 0.25 * selection_components["predicate_macro_error"]
                    + 0.10 * selection_components["next_reached_macro_error"]
                )
            row["validation"] = last_validation
            row["validation_selection_components"] = selection_components
            row["validation_selection_score"] = selection_score
            if selection_score < best_score:
                best_score = selection_score
                best_step = step
                best_validation = dict(last_validation)
                atomic_torch_save(
                    args.output / "event_world_model_best.pt",
                    {
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(config),
                        "step": step,
                        "best_step": best_step,
                        "best_score": best_score,
                        "contract": contract,
                        "normalization": {
                            "object_delta_mean": object_mean,
                            "object_delta_std": object_std,
                        },
                        "shadow_semantic": shadow_info,
                        "validation": last_validation,
                        "validation_selection_components": selection_components,
                    },
                )
            if (
                args.early_stopping_patience > 0
                and step - best_step >= args.early_stopping_patience
            ):
                stopped_early = True
                row["early_stopping"] = {
                    "triggered": True,
                    "patience_steps": args.early_stopping_patience,
                    "best_step": best_step,
                }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("EVENT_WORLD_MODEL=" + json.dumps(row, sort_keys=True), flush=True)

        if save_now:
            atomic_torch_save(
                args.output / "event_world_model_latest.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": dataclasses.asdict(config),
                    "step": step,
                    "best_step": best_step,
                    "best_score": best_score,
                    "last_validation": last_validation,
                    "contract": contract,
                    "normalization": {
                        "object_delta_mean": object_mean,
                        "object_delta_std": object_std,
                    },
                    "shadow_semantic": shadow_info,
                },
            )
        if stopped_early:
            break

    summary = {
        "status": "training_complete",
        "device": str(device),
        "amp": amp_name,
        "steps": step,
        "requested_steps": args.steps,
        "stopped_early": stopped_early,
        "best_step": best_step,
        "best_validation_selection_score": best_score,
        "best_validation": best_validation,
        "last_validation": last_validation,
        "sealed_test_evaluated": False,
        "checkpoint": str(args.output / "event_world_model_best.pt"),
        "resume_checkpoint": str(args.output / "event_world_model_latest.pt"),
        "contract": contract,
    }
    atomic_json(args.output / "training_summary.json", summary)
    print("TRAINING_COMPLETE=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
