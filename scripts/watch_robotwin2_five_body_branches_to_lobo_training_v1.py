#!/usr/bin/env python3
"""Wait for the full public five-body branch set, then train all LOBO folds.

The watcher is a detached, CPU-only orchestrator.  Before training it validates
exactly 2,000 four-candidate decision artifacts from the five public RoboTwin2
embodiments without interpreting outcome/event payload arrays.  It then binds
the immutable actor identity, public materialization receipt and five body
manifests, and runs the five source-only outer-LOBO folds sequentially on the
authorized RTX 4090.

The final aggregate is deliberately source-validation evidence.  It records
best-of-four ranking effect and prediction diagnostics, but never relabels
those diagnostics as held-out task-success evidence; that requires the later
paired live-simulator evaluation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event


FORMAT = "etsf_robotwin2_five_body_branches_to_lobo_watcher_v1"
FINAL_FORMAT = "etsf_robotwin2_five_body_lobo_source_validation_aggregate_v1"
BINDING_FORMAT = "etsf_robotwin2_five_body_lobo_training_binding_v1"
ACTOR_FORMAT = "etsf_robotwin2_frozen_native_actor_authority_v1"
MANIFEST_FORMAT = "etsf_robotwin2_canonical_transition_manifest_v1"
COLLECTOR_FORMAT = "etsf_robotwin2_five_body_ee_candidate_branches_v1"
DATASET_REPO = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
TASK = "move_can_pot"
DEFAULT_INSTRUCTION = "Move the can to the side of the pot."
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
STATE_SCHEMA = "dual_ee_object_relative_state_27d_v2"
ACTION_SCHEMA = "dual_ee_se3_gripper_delta_14d_v2"
EVENTS = ("e0", "e12", "e3", "e4", "eK")
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
CANDIDATE_COUNT = 4
SEED_START = 2026082000
DEVELOPMENT_SEED_STOP = 2026090000
SEEDS_PER_CONDITION_QUERY = 5
# Preserve exactly 200 decisions per body/condition while covering every
# remaining-budget value used by the formal five-action online scorer.
QUERY_INDICES = tuple(range(40))
CANDIDATE_NOISE_CONTRACT = {
    "distribution": "antithetic_standard_normal_pairs_each_marginal_N_0_I",
    "candidate_indices": [0, 1, 2, 3],
    "base_noise_indices": [0, 0, 2, 2],
    "signs": [1, -1, 1, -1],
    "candidate_zero_legacy_noise_unchanged": True,
}
TERMINAL_SUPERVISION_CONTRACT = {
    "terminal_max_event_id": (
        "maximum_canonical_event_from_candidate_root_through_continuation"
    ),
    "terminal_event_mask": "finite_horizon_terminal_event_is_valid",
    "terminal_stage_progress": "one_if_success_else_terminal_max_event_id_div_4",
    "terminal_goal_distance": "euclidean_goal_residual_at_full_continuation_terminal",
    "terminal_goal_progress": "root_goal_distance_minus_terminal_goal_distance",
    "terminal_goal_progress_mask": "finite_horizon_terminal_goal_is_valid",
    "terminal_stop_reason_id": {
        "success": 0,
        "formal_action_limit": 1,
    },
    "planner_status_failure_without_exception": "valid_finite_horizon_outcome",
    "action_execution_exception": "invalidate_complete_four_candidate_decision",
    "same_stage_progress_definition_as_formal_paired_runner": True,
}
EVENT_AGE_CONTRACT = {
    "array": "event_age_seconds",
    "semantics": "elapsed_physical_seconds_since_current_canonical_event_entry",
    "clock_source": "counted_successful_sapien_scene_step_calls",
    "available_before_candidate_execution": True,
    "same_value_for_all_candidates_at_one_root": True,
}
TERMINAL_HORIZON_CONTRACT = {
    "array": "remaining_action_budget",
    "semantics": "max_episode_action_steps_minus_pre_action_take_action_count",
    "available_before_candidate_execution": True,
    "same_value_for_all_candidates_at_one_root": True,
    "conditions_only_terminal_consequence_heads": True,
    "direct_rank_path": False,
    "formal_episode_action_steps": 200,
    "formal_actor_query_stride_actions": 5,
    "development_remaining_action_budgets": list(range(200, 0, -5)),
}
ROOT_POSE_RESTORE_ATOL = 2.384185791015625e-7
BRANCH_ROOT_SNAPSHOT_CONTRACT = {
    "format": "etsf_sapien_explicit_fresh_scene_branch_root_v2_float32_roundtrip",
    "physics_state": "keyed_rigid_articulation_drive_task_render_rng_snapshot",
    "candidate_scene_isolation": "one_fresh_scene_per_candidate",
    "contact_cache_reconstruction": "one_counted_raw_scene_step",
    "derived_articulation_qacc": (
        "recorded_for_provenance_not_required_pre_step_then_recomputed_and_"
        "strictly_hashed_after_canonicalization_step"
    ),
    "precanonical_restore_exact_except_articulation_root_pose_float32_roundtrip": True,
    "articulation_root_pose_component_atol": ROOT_POSE_RESTORE_ATOL,
    "articulation_root_pose_component_rtol": 0.0,
    "all_non_root_pose_restorable_fields_bit_exact": True,
    "post_canonicalization_full_snapshot_bit_exact": True,
    "simulation_clock_restored": True,
    "task_counters_restored": ["take_action_cnt", "eval_success"],
    "rng_restored": ["python", "numpy", "torch_cpu", "torch_cuda"],
    "reset_and_action_prefix_replay_used_for_candidates": False,
}
OBJECT_EFFECT_SCHEMA = {
    "format": "etsf_robotwin2_moving_object_se3_effect_6d_v1",
    "channels": [
        "moving_delta_x",
        "moving_delta_y",
        "moving_delta_z",
        "moving_delta_axis_angle_x",
        "moving_delta_axis_angle_y",
        "moving_delta_axis_angle_z",
    ],
    "rotation": "q_post_times_conjugate_q_root_shortest_axis_angle_wxyz",
    "redundant_relative_goal_delta_removed": True,
}
DIAGNOSTIC_FORMAT = "etsf_robotwin2_candidate_branch_diagnostics_v1"
BRANCH_DIAGNOSTIC_CONTRACT = {
    "format": DIAGNOSTIC_FORMAT,
    "first_executed": "successful_or_physics_advancing_actions_in_planned_first_chunk",
    "branch_error": "all_false_execution_exception_invalidates_complete_decision",
    "candidate_action_pairwise_rms": (
        "symmetric_raw_canonical_effect_rms_over_planned_first_five_actions"
    ),
}
DECISIONS_PER_BODY_CONDITION = SEEDS_PER_CONDITION_QUERY * len(QUERY_INDICES)
DECISIONS_PER_BODY = len(CONDITIONS) * DECISIONS_PER_BODY_CONDITION
TOTAL_DECISIONS = len(BODIES) * DECISIONS_PER_BODY
TOTAL_BRANCHES = TOTAL_DECISIONS * CANDIDATE_COUNT
EXPECTED_GPU_UUID = "GPU-06f6e50e-5296-258f-dd86-8f838390a7d1"
ENSEMBLE_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
REQUIRED_ARRAYS = {
    "state",
    "actions",
    "action_mask",
    "current_event_id",
    "post_event_id",
    "post_event_mask",
    "next_event_id",
    "next_event_mask",
    "duration",
    "duration_observed",
    "duration_mask",
    "success",
    "success_mask",
    "recovery",
    "recovery_mask",
    "object_delta",
    "object_delta_mask",
    "terminal_max_event_id",
    "terminal_event_mask",
    "terminal_stage_progress",
    "terminal_goal_distance",
    "terminal_goal_progress",
    "terminal_goal_progress_mask",
    "terminal_stop_reason_id",
    "candidate_index",
    "event_age_seconds",
    "remaining_action_budget",
    "dt",
}
FLOAT_ARRAYS = {
    "state",
    "actions",
    "post_event_mask",
    "next_event_mask",
    "duration",
    "duration_observed",
    "duration_mask",
    "success",
    "success_mask",
    "recovery",
    "recovery_mask",
    "object_delta",
    "object_delta_mask",
    "terminal_stage_progress",
    "terminal_goal_distance",
    "terminal_goal_progress",
    "terminal_event_mask",
    "terminal_goal_progress_mask",
    "event_age_seconds",
    "remaining_action_budget",
    "dt",
}
INTEGER_ARRAYS = {
    "current_event_id", "post_event_id", "next_event_id",
    "terminal_max_event_id", "terminal_stop_reason_id", "candidate_index"
}
DIAGNOSTIC_ARRAYS = {
    "first_executed",
    "branch_error",
    "candidate_action_pairwise_rms",
}
_FAILURE_STATE_PATH: Path | None = None
_FAILURE_RUN_EXIT_PATH: Path | None = None


class LoboWatcherError(RuntimeError):
    """The public branch, actor, GPU, or training contract failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    root = path.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise LoboWatcherError("actor checkpoint must be a real directory")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise LoboWatcherError("actor checkpoint contains a symbolic link")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise LoboWatcherError("actor checkpoint contains a special file")
        rows.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise LoboWatcherError("actor checkpoint directory is empty")
    return canonical_sha256(rows), len(rows), sum(row["size_bytes"] for row in rows)


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["logical_sha256"] = canonical_sha256(result)
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def create_once_or_verify(path: Path, value: Any, label: str) -> None:
    if path.exists():
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != value:
            raise LoboWatcherError(f"existing {label} differs from the exact contract")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".create-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise LoboWatcherError(f"racing {label} differs from exact contract")
    finally:
        temporary.unlink(missing_ok=True)


def contained_relative(base: Path, target: Path, label: str) -> str:
    base = base.expanduser().resolve()
    target = target.expanduser().resolve()
    try:
        result = target.relative_to(base)
    except ValueError as error:
        raise LoboWatcherError(f"{label} is not contained by binding directory") from error
    if not result.parts or any(part == ".." for part in result.parts):
        raise LoboWatcherError(f"{label} has an unsafe relative path")
    cursor = base
    for part in result.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise LoboWatcherError(f"{label} path contains a symbolic link")
    return result.as_posix()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LoboWatcherError(f"{label} is not a real file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LoboWatcherError(f"{label} must be a JSON object")
    return value


def verify_logical_sha(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("logical_sha256", None)
    if observed != canonical_sha256(unsigned):
        raise LoboWatcherError(f"{label} logical SHA-256 mismatch")


def _read_exact(stream: BinaryIO, count: int, label: str) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise LoboWatcherError(f"truncated NPY {label}")
    return value


def read_npy_header(stream: BinaryIO, label: str) -> dict[str, Any]:
    if _read_exact(stream, 6, label) != b"\x93NUMPY":
        raise LoboWatcherError(f"{label} is not an NPY member")
    major, minor = _read_exact(stream, 2, label)
    if (major, minor) == (1, 0):
        header_size = struct.unpack("<H", _read_exact(stream, 2, label))[0]
    elif major in (2, 3):
        header_size = struct.unpack("<I", _read_exact(stream, 4, label))[0]
    else:
        raise LoboWatcherError(f"unsupported NPY version for {label}: {(major, minor)}")
    try:
        header = ast.literal_eval(_read_exact(stream, header_size, label).decode("latin1"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise LoboWatcherError(f"invalid NPY header for {label}") from error
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise LoboWatcherError(f"invalid NPY header fields for {label}")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape
    ):
        raise LoboWatcherError(f"invalid NPY shape for {label}")
    if header["fortran_order"] is not False:
        raise LoboWatcherError(f"Fortran-order NPY is forbidden for {label}")
    return {"descr": header["descr"], "shape": shape}


def expected_shape(name: str, horizon: int) -> tuple[int, ...]:
    if name == "state":
        return (CANDIDATE_COUNT, 27)
    if name == "actions":
        return (CANDIDATE_COUNT, horizon, 14)
    if name == "action_mask":
        return (CANDIDATE_COUNT, horizon)
    if name == "object_delta":
        return (CANDIDATE_COUNT, 6)
    return (CANDIDATE_COUNT,)


def validate_decision_npz(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LoboWatcherError(f"decision payload is missing/symlink: {path}")
    if sha256_file(path) != expected_sha256:
        raise LoboWatcherError(f"decision payload SHA-256 mismatch: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            member_names = archive.namelist()
            expected_members = {f"{name}.npy" for name in REQUIRED_ARRAYS}
            if len(member_names) != len(expected_members) or set(member_names) != expected_members:
                raise LoboWatcherError(f"decision NPZ member set mismatch: {path}")
            headers: dict[str, dict[str, Any]] = {}
            for name in sorted(REQUIRED_ARRAYS):
                with archive.open(f"{name}.npy") as stream:
                    headers[name] = read_npy_header(stream, f"{path}:{name}")
            action_shape = headers["actions"]["shape"]
            if len(action_shape) != 3 or action_shape[0] != 4 or action_shape[2] != 14:
                raise LoboWatcherError(f"decision action shape mismatch: {path}")
            horizon = int(action_shape[1])
            if horizon <= 0:
                raise LoboWatcherError(f"decision action horizon is empty: {path}")
            for name, header in headers.items():
                if tuple(header["shape"]) != expected_shape(name, horizon):
                    raise LoboWatcherError(f"decision {name} shape mismatch: {path}")
                if name == "action_mask":
                    allowed = {"|b1"}
                elif name in FLOAT_ARRAYS:
                    allowed = {"<f4", "=f4"}
                else:
                    allowed = {"<i8", "=i8"}
                if (
                    name not in FLOAT_ARRAYS | INTEGER_ARRAYS | {"action_mask"}
                    or header["descr"] not in allowed
                ):
                    raise LoboWatcherError(f"decision {name} dtype mismatch: {path}")
            with archive.open("action_mask.npy") as stream:
                read_npy_header(stream, f"{path}:action_mask")
                action_mask = _read_exact(
                    stream, CANDIDATE_COUNT * horizon, "action_mask"
                )
            planned_row = bytes([1] * 5 + [0] * (horizon - 5)) if horizon >= 5 else b""
            if horizon < 5 or action_mask != planned_row * CANDIDATE_COUNT:
                raise LoboWatcherError(
                    f"decision action mask is not the full planned first five steps: {path}"
                )
            with archive.open("candidate_index.npy") as stream:
                header = read_npy_header(stream, f"{path}:candidate_index")
                byte_order = "<" if header["descr"] == "<i8" else "="
                candidate_index = struct.unpack(
                    byte_order + "4q", _read_exact(stream, 32, "candidate_index")
                )
            if candidate_index != (0, 1, 2, 3):
                raise LoboWatcherError(f"candidate order is not [0,1,2,3]: {path}")
            with archive.open("dt.npy") as stream:
                header = read_npy_header(stream, f"{path}:dt")
                byte_order = "<" if header["descr"] == "<f4" else "="
                elapsed = struct.unpack(byte_order + "4f", _read_exact(stream, 16, "dt"))
            if any(not math.isfinite(value) or value <= 0 for value in elapsed):
                raise LoboWatcherError(f"decision contains non-positive planned dt: {path}")
            if any(abs(value - 5.0 / 15.0) > 1e-6 for value in elapsed):
                raise LoboWatcherError(f"decision planned dt is not fixed 5/15 seconds: {path}")
            with archive.open("event_age_seconds.npy") as stream:
                header = read_npy_header(stream, f"{path}:event_age_seconds")
                byte_order = "<" if header["descr"] == "<f4" else "="
                event_ages = struct.unpack(
                    byte_order + "4f", _read_exact(stream, 16, "event_age_seconds")
                )
            if any(not math.isfinite(value) or value < 0.0 for value in event_ages):
                raise LoboWatcherError(f"decision contains invalid event age: {path}")
            if any(abs(value - event_ages[0]) > 1e-6 for value in event_ages[1:]):
                raise LoboWatcherError(
                    f"decision candidates do not share one pre-action event age: {path}"
                )
            with archive.open("remaining_action_budget.npy") as stream:
                header = read_npy_header(stream, f"{path}:remaining_action_budget")
                byte_order = "<" if header["descr"] == "<f4" else "="
                remaining_budget = struct.unpack(
                    byte_order + "4f",
                    _read_exact(stream, 16, "remaining_action_budget"),
                )
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in remaining_budget
            ) or any(value != remaining_budget[0] for value in remaining_budget[1:]):
                raise LoboWatcherError(
                    f"decision candidates do not share one positive remaining budget: {path}"
                )
            if remaining_budget[0] not in TERMINAL_HORIZON_CONTRACT[
                "development_remaining_action_budgets"
            ]:
                raise LoboWatcherError(
                    f"decision remaining budget is outside the formal query grid: {path}"
                )
            with archive.open("actions.npy") as stream:
                header = read_npy_header(stream, f"{path}:actions")
                byte_order = "<" if header["descr"] == "<f4" else "="
                action_count = CANDIDATE_COUNT * horizon * 14
                action_values = struct.unpack(
                    byte_order + f"{action_count}f",
                    _read_exact(stream, action_count * 4, "actions"),
                )
                if stream.read(1):
                    raise LoboWatcherError(f"decision actions contain trailing bytes: {path}")
            pairwise_rms: list[list[float]] = []
            for left in range(CANDIDATE_COUNT):
                row = []
                for right in range(CANDIDATE_COUNT):
                    squared = 0.0
                    for step in range(5):
                        for channel in range(14):
                            left_index = (left * horizon + step) * 14 + channel
                            right_index = (right * horizon + step) * 14 + channel
                            difference = (
                                action_values[left_index] - action_values[right_index]
                            )
                            squared += difference * difference
                    row.append(math.sqrt(squared / float(5 * 14)))
                pairwise_rms.append(row)
    except zipfile.BadZipFile as error:
        raise LoboWatcherError(f"decision NPZ is corrupt: {path}") from error
    return {
        "candidate_count": 4,
        "action_horizon": horizon,
        "remaining_action_budget": float(remaining_budget[0]),
        "candidate_action_pairwise_rms": pairwise_rms,
    }


def validate_diagnostic_npz(
    path: Path,
    expected_sha256: str,
    expected_pairwise_rms: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Validate the bound proposal-coverage sidecar without using it as a label."""

    if not path.is_file() or path.is_symlink():
        raise LoboWatcherError(f"diagnostic payload is missing/symlink: {path}")
    if sha256_file(path) != expected_sha256:
        raise LoboWatcherError(f"diagnostic payload SHA-256 mismatch: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            expected_members = {f"{name}.npy" for name in DIAGNOSTIC_ARRAYS}
            member_names = archive.namelist()
            if len(member_names) != len(expected_members) or set(member_names) != expected_members:
                raise LoboWatcherError(f"diagnostic NPZ member set mismatch: {path}")
            headers: dict[str, dict[str, Any]] = {}
            for name in sorted(DIAGNOSTIC_ARRAYS):
                with archive.open(f"{name}.npy") as stream:
                    headers[name] = read_npy_header(stream, f"{path}:{name}")
            expected = {
                "first_executed": ((CANDIDATE_COUNT,), {"<i8", "=i8"}),
                "branch_error": ((CANDIDATE_COUNT,), {"|b1"}),
                "candidate_action_pairwise_rms": (
                    (CANDIDATE_COUNT, CANDIDATE_COUNT),
                    {"<f4", "=f4"},
                ),
            }
            for name, (shape, dtypes) in expected.items():
                if tuple(headers[name]["shape"]) != shape or headers[name]["descr"] not in dtypes:
                    raise LoboWatcherError(
                        f"diagnostic {name} shape/dtype mismatch: {path}"
                    )
            with archive.open("first_executed.npy") as stream:
                header = read_npy_header(stream, f"{path}:first_executed")
                byte_order = "<" if header["descr"] == "<i8" else "="
                first_executed = struct.unpack(
                    byte_order + "4q", _read_exact(stream, 32, "first_executed")
                )
                if stream.read(1):
                    raise LoboWatcherError(
                        f"diagnostic first_executed contains trailing bytes: {path}"
                    )
            if any(value < 0 or value > 5 for value in first_executed):
                raise LoboWatcherError(
                    f"diagnostic first_executed is outside planned first five steps: {path}"
                )
            with archive.open("branch_error.npy") as stream:
                read_npy_header(stream, f"{path}:branch_error")
                branch_error = _read_exact(stream, CANDIDATE_COUNT, "branch_error")
                if stream.read(1) or any(value != 0 for value in branch_error):
                    raise LoboWatcherError(
                        "action execution exceptions must invalidate the complete "
                        f"decision instead of reaching diagnostics: {path}"
                    )
            with archive.open("candidate_action_pairwise_rms.npy") as stream:
                header = read_npy_header(
                    stream, f"{path}:candidate_action_pairwise_rms"
                )
                byte_order = "<" if header["descr"] == "<f4" else "="
                pairwise = struct.unpack(
                    byte_order + "16f",
                    _read_exact(stream, 64, "candidate_action_pairwise_rms"),
                )
                if stream.read(1):
                    raise LoboWatcherError(
                        f"diagnostic pairwise RMS contains trailing bytes: {path}"
                    )
            for left in range(CANDIDATE_COUNT):
                for right in range(CANDIDATE_COUNT):
                    observed = float(pairwise[left * CANDIDATE_COUNT + right])
                    expected_value = float(expected_pairwise_rms[left][right])
                    reverse = float(pairwise[right * CANDIDATE_COUNT + left])
                    if (
                        not math.isfinite(observed)
                        or observed < 0.0
                        or abs(observed - reverse) > 1e-7
                        or (left == right and abs(observed) > 1e-7)
                        or abs(observed - expected_value) > 2e-6
                    ):
                        raise LoboWatcherError(
                            f"diagnostic pairwise RMS disagrees with core actions: {path}"
                        )
    except zipfile.BadZipFile as error:
        raise LoboWatcherError(f"diagnostic NPZ is corrupt: {path}") from error
    return {"diagnostic_format": DIAGNOSTIC_FORMAT}


def collection_progress(branches_root: Path) -> dict[str, Any]:
    bodies: dict[str, Any] = {}
    total = 0
    for body in BODIES:
        path = branches_root / body / "manifest.json"
        if not path.is_file():
            bodies[body] = {"manifest_present": False, "decisions": 0}
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            groups = value.get("groups", []) if isinstance(value, dict) else []
            counts = {
                condition: sum(
                    1 for item in groups
                    if isinstance(item, Mapping) and item.get("condition") == condition
                )
                for condition in CONDITIONS
            }
            query_counts = {
                f"{condition}|query={query}": sum(
                    1 for item in groups
                    if isinstance(item, Mapping)
                    and item.get("condition") == condition
                    and item.get("root_query_index") == query
                )
                for condition in CONDITIONS
                for query in QUERY_INDICES
            }
            count = len(groups) if isinstance(groups, list) else 0
            bodies[body] = {
                "manifest_present": True,
                "decisions": count,
                "conditions": counts,
                "condition_queries": query_counts,
            }
            total += count
        except (OSError, json.JSONDecodeError):
            bodies[body] = {"manifest_present": True, "decisions": 0, "transient_read": True}
    return {
        "bodies": bodies,
        "decisions": total,
        "declared_branches": total * CANDIDATE_COUNT,
        "expected_decisions": TOTAL_DECISIONS,
        "expected_branches": TOTAL_BRANCHES,
    }


def progress_is_complete(progress: Mapping[str, Any]) -> bool:
    if progress.get("decisions") != TOTAL_DECISIONS:
        return False
    for body in BODIES:
        row = progress["bodies"].get(body, {})
        if row.get("decisions") != DECISIONS_PER_BODY:
            return False
        if row.get("conditions") != {
            condition: DECISIONS_PER_BODY_CONDITION for condition in CONDITIONS
        }:
            return False
        if row.get("condition_queries") != {
            f"{condition}|query={query}": SEEDS_PER_CONDITION_QUERY
            for condition in CONDITIONS
            for query in QUERY_INDICES
        }:
            return False
    return True


def reject_irrecoverable_progress(progress: Mapping[str, Any]) -> None:
    if int(progress.get("decisions", 0)) > TOTAL_DECISIONS:
        raise LoboWatcherError("collection contains more than 2000 declared decisions")
    for body in BODIES:
        row = progress["bodies"].get(body, {})
        if int(row.get("decisions", 0)) > DECISIONS_PER_BODY:
            raise LoboWatcherError(f"{body} contains more than 400 declared decisions")
        conditions = row.get("conditions", {})
        for condition in CONDITIONS:
            if int(conditions.get(condition, 0)) > DECISIONS_PER_BODY_CONDITION:
                raise LoboWatcherError(
                    f"{body}/{condition} contains more than 200 declared decisions"
                )
        units = row.get("condition_queries", {})
        for condition in CONDITIONS:
            for query in QUERY_INDICES:
                key = f"{condition}|query={query}"
                if int(units.get(key, 0)) > SEEDS_PER_CONDITION_QUERY:
                    raise LoboWatcherError(
                        f"{body}/{key} contains more than five declared decisions"
                    )


def validate_complete_collection(
    branches_root: Path, actor_checkpoint: Path
) -> dict[str, Any]:
    manifest_bindings: dict[str, dict[str, str]] = {}
    manifest_audits: dict[str, Any] = {}
    shared_adapter_sha: str | None = None
    shared_collector_sha: str | None = None
    shared_event_spec_sha: str | None = None
    shared_event_implementation_sha: str | None = None
    decision_count = branch_count = 0
    action_horizons: set[int] = set()
    for body in BODIES:
        manifest_path = branches_root / body / "manifest.json"
        before_sha = sha256_file(manifest_path)
        manifest = read_json(manifest_path, f"{body} manifest")
        verify_logical_sha(manifest, f"{body} manifest")
        adapter = manifest.get("schema_adapter")
        physical_time = manifest.get("physical_time_contract")
        candidate_action = manifest.get("candidate_action_contract")
        try:
            analytic_event.validate_event_contract(
                manifest.get("analytic_event_contract")
            )
        except analytic_event.AnalyticEventSpecError as error:
            raise LoboWatcherError(
                f"{body} manifest changed the analytic event contract"
            ) from error
        if (
            manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("collector_format") != COLLECTOR_FORMAT
            or manifest.get("dataset_repo") != DATASET_REPO
            or manifest.get("dataset_revision") != DATASET_REVISION
            or manifest.get("task") != TASK
            or manifest.get("instruction") != DEFAULT_INSTRUCTION
            or manifest.get("body") != body
            or manifest.get("status")
            != "complete_400_decisions_1600_candidate_branches"
            or not isinstance(manifest.get("collector_file_sha256"), str)
            or len(manifest["collector_file_sha256"]) != 64
            or manifest.get("actor_checkpoint") != str(actor_checkpoint)
            or manifest.get("candidate_count") != CANDIDATE_COUNT
            or manifest.get("action_exec_steps") != 5
            or manifest.get("max_episode_action_steps") != 200
            or manifest.get("candidate_zero_is_actor_baseline") is not True
            or manifest.get("same_ordered_candidate_set_for_baseline_and_etsf") is not True
            or manifest.get("root_query_indices") != list(QUERY_INDICES)
            or manifest.get("candidate_noise_contract") != CANDIDATE_NOISE_CONTRACT
            or manifest.get("terminal_supervision_contract")
            != TERMINAL_SUPERVISION_CONTRACT
            or manifest.get("event_age_contract") != EVENT_AGE_CONTRACT
            or manifest.get("terminal_horizon_contract")
            != TERMINAL_HORIZON_CONTRACT
            or manifest.get("branch_root_snapshot_contract")
            != BRANCH_ROOT_SNAPSHOT_CONTRACT
            or manifest.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
            or manifest.get("branch_diagnostic_contract")
            != BRANCH_DIAGNOSTIC_CONTRACT
            or not isinstance(adapter, Mapping)
            or adapter.get("kind") != "analytic_label_free_canonical_v1"
            or adapter.get("trainable") is not False
            or adapter.get("labels_or_outcomes_used_to_fit") is not False
            or adapter.get("heldout_supervision_allowed") is not False
            or adapter.get("state_dim") != 27
            or adapter.get("action_dim") != 14
            or adapter.get("state_schema") != STATE_SCHEMA
            or adapter.get("action_schema") != ACTION_SCHEMA
            or adapter.get("elapsed_time_unit") != "seconds"
            or adapter.get("duration_unit") != "seconds"
            or adapter.get("event_names") != list(EVENTS)
            or manifest.get("event_spec_sha256") != EVENT_SPEC_SHA256
            or not isinstance(
                manifest.get("event_derivation_implementation_sha256"), str
            )
            or len(manifest["event_derivation_implementation_sha256"]) != 64
            or manifest.get("state27_relative_goal_contract")
            != (
                "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
                "event_labels_and_online_state27_channels_0_2"
            )
            or not isinstance(physical_time, Mapping)
            or physical_time.get("source")
            != "counted_successful_sapien_scene_step_calls"
            or physical_time.get("simulator_timestep_source") != "scene.get_timestep"
            or physical_time.get("policy_action_call_count_used_as_time") is not False
            or physical_time.get("wall_clock_used_as_time") is not False
            or physical_time.get("dt_semantics")
            != "planned_first_candidate_chunk_seconds"
            or physical_time.get("planned_action_steps") != 5
            or physical_time.get("actor_control_hz") != 15.0
            or physical_time.get("planned_dt_seconds") != 5.0 / 15.0
            or physical_time.get("duration_semantics")
            != "simulator_elapsed_seconds_to_event_boundary"
            or physical_time.get("zero_elapsed_duration_masked") is not True
            or physical_time.get("stationary_window_seconds")
            != analytic_event.THRESHOLDS["stationary_window_seconds"]
            or physical_time.get("stationary_speed_threshold_m_per_s")
            != analytic_event.THRESHOLDS["stationary_speed_m_per_s"]
            or candidate_action
            != {
                "critic_observation_time": "before_candidate_execution",
                "planned_action_horizon": 5,
                "action_mask_source": "planned_first_chunk_not_executed_count",
                "executed_action_count_used_for_action_mask": False,
                "executed_action_count_used_for_sim_time_accounting_only": True,
                "planner_status_fail_is_a_valid_action_outcome": True,
                "python_execution_exception_invalidates_complete_decision": True,
            }
        ):
            raise LoboWatcherError(f"{body} manifest violates the canonical collection contract")
        adapter_sha = str(adapter.get("implementation_sha256", ""))
        event_spec_sha = str(manifest.get("event_spec_sha256", ""))
        event_implementation_sha = str(
            manifest.get("event_derivation_implementation_sha256", "")
        )
        if (
            len(adapter_sha) != 64
            or len(event_spec_sha) != 64
            or len(event_implementation_sha) != 64
        ):
            raise LoboWatcherError(f"{body} lacks adapter/event-spec SHA-256")
        if shared_adapter_sha is None:
            shared_adapter_sha = adapter_sha
            shared_collector_sha = str(manifest["collector_file_sha256"])
            shared_event_spec_sha = event_spec_sha
            shared_event_implementation_sha = event_implementation_sha
        if (
            adapter_sha != shared_adapter_sha
            or manifest.get("collector_file_sha256") != shared_collector_sha
            or event_spec_sha != shared_event_spec_sha
            or event_implementation_sha != shared_event_implementation_sha
        ):
            raise LoboWatcherError(
                "five bodies do not share one adapter/event specification/implementation"
            )
        groups = manifest.get("groups")
        if not isinstance(groups, list) or len(groups) != DECISIONS_PER_BODY:
            raise LoboWatcherError(f"{body} does not contain exactly 400 decisions")
        observed_ids: set[str] = set()
        observed_paths: set[str] = set()
        observed_diagnostic_paths: set[str] = set()
        condition_counts = {condition: 0 for condition in CONDITIONS}
        seeds_by_unit = {
            (condition, query): set()
            for condition in CONDITIONS
            for query in QUERY_INDICES
        }
        for item in groups:
            if not isinstance(item, Mapping):
                raise LoboWatcherError(f"{body} manifest has a non-object group")
            group_id = item.get("group_id")
            condition = item.get("condition")
            seed = item.get("requested_seed")
            query = item.get("root_query_index")
            if (
                not isinstance(group_id, str)
                or group_id in observed_ids
                or condition not in CONDITIONS
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or not SEED_START <= seed < DEVELOPMENT_SEED_STOP
                or isinstance(query, bool)
                or query not in QUERY_INDICES
                or seed in seeds_by_unit[(str(condition), int(query))]
            ):
                raise LoboWatcherError(f"{body} has an unexpected/duplicate group identity")
            if group_id != f"{condition}|seed={seed}|query={query}":
                raise LoboWatcherError(f"{body}/{group_id} group identity is inconsistent")
            expected_filename = f"{condition}_seed_{seed}_query_{query}.npz"
            expected_diagnostic_filename = (
                f"{condition}_seed_{seed}_query_{query}.diagnostics.npz"
            )
            relative = item.get("path")
            diagnostics_relative = item.get("diagnostics_path")
            snapshot_hashes = (
                item.get("branch_root_snapshot_sha256"),
                item.get("branch_root_restorable_snapshot_sha256"),
                item.get("canonical_root_snapshot_sha256"),
            )
            if (
                item.get("condition") != condition
                or item.get("requested_seed") != seed
                or item.get("root_query_index") != query
                or relative != f"groups/{expected_filename}"
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                or item.get("diagnostic_format") != DIAGNOSTIC_FORMAT
                or diagnostics_relative
                != f"groups/{expected_diagnostic_filename}"
                or not isinstance(item.get("diagnostics_sha256"), str)
                or len(item["diagnostics_sha256"]) != 64
                or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in snapshot_hashes
                )
            ):
                raise LoboWatcherError(f"{body}/{group_id} identity fields changed")
            payload = branches_root / body / str(relative)
            try:
                payload.resolve().relative_to((branches_root / body).resolve())
            except ValueError as error:
                raise LoboWatcherError(f"{body}/{group_id} payload escapes body root") from error
            decision = validate_decision_npz(payload, str(item["sha256"]))
            if decision["remaining_action_budget"] != 200.0 - 5.0 * query:
                raise LoboWatcherError(
                    f"{body}/{group_id} remaining budget disagrees with query index"
                )
            diagnostics_payload = branches_root / body / str(diagnostics_relative)
            try:
                diagnostics_payload.resolve().relative_to(
                    (branches_root / body).resolve()
                )
            except ValueError as error:
                raise LoboWatcherError(
                    f"{body}/{group_id} diagnostic payload escapes body root"
                ) from error
            validate_diagnostic_npz(
                diagnostics_payload,
                str(item["diagnostics_sha256"]),
                decision["candidate_action_pairwise_rms"],
            )
            action_horizons.add(int(decision["action_horizon"]))
            observed_ids.add(str(group_id))
            observed_paths.add(str(relative))
            observed_diagnostic_paths.add(str(diagnostics_relative))
            seeds_by_unit[(str(condition), int(query))].add(int(seed))
            condition_counts[condition] += 1
            decision_count += 1
            branch_count += int(decision["candidate_count"])
        if any(
            len(seeds_by_unit[(condition, query)]) != SEEDS_PER_CONDITION_QUERY
            for condition in CONDITIONS
            for query in QUERY_INDICES
        ):
            raise LoboWatcherError(
                f"{body} does not contain five unique development seeds per condition/query"
            )
        disk_paths = {
            path.relative_to(branches_root / body).as_posix()
            for path in (branches_root / body / "groups").glob("*.npz")
            if path.is_file() and not path.name.endswith(".diagnostics.npz")
        }
        if disk_paths != observed_paths:
            raise LoboWatcherError(f"{body} has missing or extra decision NPZ files")
        disk_diagnostic_paths = {
            path.relative_to(branches_root / body).as_posix()
            for path in (branches_root / body / "groups").glob("*.diagnostics.npz")
            if path.is_file()
        }
        if disk_diagnostic_paths != observed_diagnostic_paths:
            raise LoboWatcherError(
                f"{body} has missing or extra diagnostic NPZ files"
            )
        if sha256_file(manifest_path) != before_sha:
            raise LoboWatcherError(f"{body} manifest changed during validation")
        manifest_bindings[body] = {
            "path": "",  # populated relative to the binding directory later
            "sha256": before_sha,
        }
        manifest_audits[body] = {
            "decisions": len(groups),
            "branches": len(groups) * CANDIDATE_COUNT,
            "conditions": condition_counts,
            "condition_queries": {
                f"{condition}|query={query}": {
                    "decisions": len(seeds_by_unit[(condition, query)]),
                    "minimum_seed": min(seeds_by_unit[(condition, query)]),
                    "maximum_seed": max(seeds_by_unit[(condition, query)]),
                    "seed_identity_sha256": canonical_sha256(
                        sorted(seeds_by_unit[(condition, query)])
                    ),
                }
                for condition in CONDITIONS
                for query in QUERY_INDICES
            },
            "manifest_file_sha256": before_sha,
            "manifest_logical_sha256": manifest["logical_sha256"],
            "diagnostic_sidecars": len(observed_diagnostic_paths),
        }
    if decision_count != TOTAL_DECISIONS or branch_count != TOTAL_BRANCHES:
        raise LoboWatcherError("five-body collection total is not exactly 2000/8000")
    return {
        "status": "complete_public_five_body_candidate_collection_verified",
        "decisions": decision_count,
        "branches": branch_count,
        "bodies": manifest_audits,
        "adapter_implementation_sha256": shared_adapter_sha,
        "collector_file_sha256": shared_collector_sha,
        "event_spec_sha256": shared_event_spec_sha,
        "event_derivation_implementation_sha256": shared_event_implementation_sha,
        "action_horizons": sorted(action_horizons),
        "outcome_or_event_arrays_interpreted_by_watcher": False,
        "candidate_index_and_dt_arrays_interpreted_by_watcher": True,
        "event_age_array_interpreted_as_pre_action_input": True,
        "diagnostic_arrays_interpreted_as_training_labels": False,
        "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
        "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
        "event_age_contract": EVENT_AGE_CONTRACT,
        "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
        "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
        "object_effect_schema": OBJECT_EFFECT_SCHEMA,
        "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
        "manifest_bindings": manifest_bindings,
    }


def gpu_identity() -> dict[str, str]:
    command = [
        "nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader,nounits"
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise LoboWatcherError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or "," not in rows[0]:
        raise LoboWatcherError("expected exactly one visible GPU")
    name, uuid = (part.strip() for part in rows[0].split(",", 1))
    return {"name": name, "uuid": uuid}


def gpu_compute_pids() -> list[int]:
    command = [
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise LoboWatcherError(f"GPU process query failed: {result.stderr.strip()}")
    return sorted(
        int(row.strip())
        for row in result.stdout.splitlines()
        if row.strip().isdigit()
    )


def numeric_summary(values: Sequence[float | int | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "member_values": [None if value is None else float(value) for value in values],
        "available_members": len(present),
        "mean": statistics.fmean(present) if present else None,
        "median": statistics.median(present) if present else None,
        "minimum": min(present) if present else None,
        "maximum": max(present) if present else None,
    }


def nested_values(members: Sequence[Mapping[str, Any]], *keys: str) -> list[Any]:
    result = []
    for member in members:
        value: Any = member["source_validation"]
        for key in keys:
            value = value.get(key) if isinstance(value, Mapping) else None
        result.append(value)
    return result


def summarize_fold(
    path: Path, held_out_body: str, expected_binding_sha256: str
) -> dict[str, Any]:
    summary_path = path / "training_summary.json"
    summary = read_json(summary_path, f"{held_out_body} training summary")
    members = summary.get("members")
    ensemble_selection = summary.get("ensemble_checkpoint_selection")
    if (
        summary.get("status") != "source_only_checkpoint_selection_complete"
        or summary.get("held_out_body") != held_out_body
        or not isinstance(members, list)
        or len(members) != 5
        or [member.get("seed") for member in members if isinstance(member, Mapping)]
        != list(ENSEMBLE_SEEDS)
        or summary.get("heldout_group_npz_opened") != 0
        or summary.get("heldout_labels_used_for_normalization_training_or_selection") is not False
        or summary.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or summary.get("event_derivation_implementation_sha256")
        != summary.get("preflight", {}).get(
            "event_derivation_implementation_sha256"
        )
        or not isinstance(summary.get("preflight"), Mapping)
        or summary["preflight"].get("binding_file_sha256")
        != expected_binding_sha256
        or not isinstance(summary.get("trainer_file_sha256"), str)
        or len(summary["trainer_file_sha256"]) != 64
        or summary.get("rank_supervision_available") is not True
        or summary.get("candidate_rank_parameters_received_direct_supervision")
        is not True
        or summary.get("synthetic_success_labels") != 0
        or not isinstance(ensemble_selection, Mapping)
        or ensemble_selection.get("common_step_required_for_all_five_members")
        is not True
        or ensemble_selection.get("rank_aggregation", {}).get("format")
        != "etsf_bounded_utility_epistemic_lcb_ensemble_v1"
        or not isinstance(
            ensemble_selection.get("selected_ensemble_candidate_ranking"),
            Mapping,
        )
        or any(
            member.get("best_step") != ensemble_selection.get("selected_step")
            or member.get("trainer_file_sha256")
            != summary.get("trainer_file_sha256")
            for member in members
        )
    ):
        raise LoboWatcherError(f"{held_out_body} training summary violates outer-LOBO")
    for member in members:
        checkpoint = Path(str(member.get("checkpoint", "")))
        if not checkpoint.is_file() or sha256_file(checkpoint) != member.get("checkpoint_sha256"):
            raise LoboWatcherError(f"{held_out_body} member checkpoint is missing/tampered")
    return {
        "held_out_body": held_out_body,
        "source_bodies": summary.get("source_bodies"),
        "member_count": len(members),
        "steps_per_member": 3000,
        "training_summary": str(summary_path),
        "training_summary_file_sha256": sha256_file(summary_path),
        "event_spec_sha256": summary.get("event_spec_sha256"),
        "event_derivation_implementation_sha256": summary.get(
            "event_derivation_implementation_sha256"
        ),
        "mixed_outcome_source_decisions": summary.get("mixed_outcome_source_decisions"),
        "observed_success_classes": summary.get("observed_success_classes"),
        "source_success_rows": summary.get("source_success_rows"),
        "source_failure_rows": summary.get("source_failure_rows"),
        "success_probability_identifiability": summary.get(
            "success_probability_identifiability"
        ),
        "informative_dense_rank_groups": summary.get(
            "informative_dense_rank_groups"
        ),
        "rank_supervision_mode": summary.get("rank_supervision_mode"),
        "selection_evidence_mode": summary.get("selection_evidence_mode"),
        "source_negative_to_positive_ratio": summary.get(
            "source_negative_to_positive_ratio"
        ),
        "source_validation": {
            "macro_one_deviation_branch_success_gain": numeric_summary(
                nested_values(
                    members,
                    "candidate_ranking",
                    "macro_one_deviation_branch_success_gain",
                )
            ),
            "macro_selected_success_rate": numeric_summary(
                nested_values(members, "candidate_ranking", "macro_selected_success_rate")
            ),
            "candidate_pairwise_accuracy": numeric_summary(
                nested_values(members, "candidate_ranking", "pairwise_accuracy")
            ),
            "success_calibration": {
                "brier": numeric_summary(nested_values(members, "success_brier")),
                "auroc_diagnostic": numeric_summary(
                    nested_values(members, "success_auroc")
                ),
                "support_by_member": nested_values(members, "success_support"),
            },
            "events": {
                "post_event_macro_f1": numeric_summary(
                    nested_values(members, "post_event", "macro_f1")
                ),
                "post_event_accuracy": numeric_summary(
                    nested_values(members, "post_event", "accuracy")
                ),
                "next_event_macro_f1": numeric_summary(
                    nested_values(members, "next_event", "macro_f1")
                ),
                "next_event_accuracy": numeric_summary(
                    nested_values(members, "next_event", "accuracy")
                ),
                "post_support_by_member": nested_values(members, "post_event", "support"),
                "next_support_by_member": nested_values(members, "next_event", "support"),
            },
            "duration_seconds": {
                "observed_mae": numeric_summary(
                    nested_values(members, "observed_duration_mae")
                ),
                "observed_nll": numeric_summary(
                    nested_values(members, "observed_duration_nll")
                ),
                "support_by_member": nested_values(members, "duration_support"),
            },
            "object_effect": {
                "rmse": numeric_summary(nested_values(members, "object_rmse")),
                "nll": numeric_summary(nested_values(members, "object_nll")),
                "support_by_member": nested_values(members, "object_support"),
            },
        },
        "one_deviation_ensemble_source_validation": (
            ensemble_selection["selected_ensemble_candidate_ranking"]
        ),
        "ensemble_common_selection_step": ensemble_selection["selected_step"],
        "trainer_file_sha256": summary["trainer_file_sha256"],
        "member_best_steps": [member.get("best_step") for member in members],
        "member_checkpoints": [
            {
                "member": member.get("member"),
                "seed": member.get("seed"),
                "checkpoint": member.get("checkpoint"),
                "checkpoint_sha256": member.get("checkpoint_sha256"),
            }
            for member in members
        ],
        "heldout_labels_used_for_training_normalization_or_selection": False,
        "heldout_task_success_measured": False,
    }


def validate_existing_authorities(
    args: argparse.Namespace,
    collection: Mapping[str, Any],
    checkpoint_sha: str,
    checkpoint_files: int,
    checkpoint_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse the immutable artifacts produced by actor-to-branches.

    The upstream watcher owns these files.  Reconstructing them here used to
    create a second, incompatible schema at the same paths and would stop the
    pipeline after all 8,000 branches had already been collected.
    """

    actor_authority = read_json(args.actor_authority, "actor authority")
    verify_logical_sha(actor_authority, "actor authority")
    sampling_contract = actor_authority.get("sampling_contract")
    actors = actor_authority.get("actors")
    if (
        actor_authority.get("format") != ACTOR_FORMAT
        or actor_authority.get("task") != TASK
        or actor_authority.get("dataset_repo") != DATASET_REPO
        or actor_authority.get("dataset_revision") != DATASET_REVISION
        or actor_authority.get("public_expert_episode_count") != 2750
        or actor_authority.get("one_universal_actor_for_all_five_bodies") is not True
        or not isinstance(sampling_contract, Mapping)
        or sampling_contract.get("format")
        != "etsf_robotwin2_five_body_fixed_flow_candidate_sampling_v1"
        or sampling_contract.get("frozen_actor_checkpoint_tree_sha256")
        != checkpoint_sha
        or sampling_contract.get("collector_file_sha256")
        != collection["collector_file_sha256"]
        or sampling_contract.get("canonical_adapter_file_sha256")
        != collection["adapter_implementation_sha256"]
        or sampling_contract.get("candidate_count") != CANDIDATE_COUNT
        or sampling_contract.get("candidate_indices") != list(range(CANDIDATE_COUNT))
        or sampling_contract.get("candidate_zero_is_actor_baseline") is not True
        or sampling_contract.get("same_ordered_candidate_set_for_baseline_and_etsf")
        is not True
        or sampling_contract.get("candidate_noise_contract")
        != CANDIDATE_NOISE_CONTRACT
        or sampling_contract.get("instruction") != DEFAULT_INSTRUCTION
        or sampling_contract.get("conditions") != list(CONDITIONS)
        or sampling_contract.get("root_query_indices") != list(QUERY_INDICES)
        or sampling_contract.get("action_exec_steps") != 5
        or sampling_contract.get("max_policy_action_calls") != 200
        or sampling_contract.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or sampling_contract.get("event_derivation_implementation_sha256")
        != collection["event_derivation_implementation_sha256"]
        or sampling_contract.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
        or sampling_contract.get("terminal_supervision_contract")
        != TERMINAL_SUPERVISION_CONTRACT
        or sampling_contract.get("event_age_contract") != EVENT_AGE_CONTRACT
        or sampling_contract.get("terminal_horizon_contract")
        != TERMINAL_HORIZON_CONTRACT
        or sampling_contract.get("branch_root_snapshot_contract")
        != BRANCH_ROOT_SNAPSHOT_CONTRACT
        or sampling_contract.get("branch_diagnostic_contract")
        != BRANCH_DIAGNOSTIC_CONTRACT
        or not isinstance(actors, Mapping)
        or set(actors) != set(BODIES)
    ):
        raise LoboWatcherError("existing actor authority violates the frozen collection")
    checkpoint_relative = contained_relative(
        args.actor_authority.parent.resolve(), args.actor_checkpoint, "actor checkpoint"
    )
    sampling_sha = canonical_sha256(sampling_contract)
    config_sha: str | None = None
    for body in BODIES:
        actor = actors[body]
        if (
            not isinstance(actor, Mapping)
            or actor.get("embodiment") != body
            or actor.get("shared_checkpoint_across_all_five_bodies") is not True
            or actor.get("frozen") is not True
            or actor.get("optimizer_updates_allowed") is not False
            or actor.get("checkpoint_path") != checkpoint_relative
            or actor.get("checkpoint_kind") != "directory_tree"
            or actor.get("checkpoint_sha256") != checkpoint_sha
            or actor.get("checkpoint_tree_file_count") != checkpoint_files
            or actor.get("checkpoint_tree_size_bytes") != checkpoint_bytes
            or actor.get("policy_type") != "smolvla"
            or actor.get("state_shape") != [16]
            or actor.get("action_shape") != [16]
            or actor.get("sampling_contract_sha256") != sampling_sha
            or actor.get("candidate_count") != CANDIDATE_COUNT
            or actor.get("candidate_zero_is_actor_baseline") is not True
            or actor.get("same_ordered_candidate_set_for_baseline_and_etsf") is not True
            or not isinstance(actor.get("config_file_sha256"), str)
            or len(actor["config_file_sha256"]) != 64
        ):
            raise LoboWatcherError(f"existing actor authority changed for {body}")
        if config_sha is None:
            config_sha = str(actor["config_file_sha256"])
        elif actor["config_file_sha256"] != config_sha:
            raise LoboWatcherError("five actor rows do not share one frozen config")
    actor_file_sha = sha256_file(args.actor_authority)

    binding = read_json(args.binding, "training binding")
    verify_logical_sha(binding, "training binding")
    binding_dir = args.binding.parent.resolve()
    materialization_relative = contained_relative(
        binding_dir, args.materialization_receipt, "materialization receipt"
    )
    actor_relative = contained_relative(binding_dir, args.actor_authority, "actor authority")
    expected_body_bindings: dict[str, Any] = {}
    for body in BODIES:
        manifest_path = args.branches_root / body / "manifest.json"
        expected_body_bindings[body] = {
            "path": contained_relative(binding_dir, manifest_path, f"{body} manifest"),
            "sha256": collection["bodies"][body]["manifest_file_sha256"],
        }
    if (
        binding.get("format") != BINDING_FORMAT
        or binding.get("dataset_repo") != DATASET_REPO
        or binding.get("dataset_revision") != DATASET_REVISION
        or binding.get("task") != TASK
        or binding.get("instruction") != DEFAULT_INSTRUCTION
        or binding.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or binding.get("candidate_noise_contract") != CANDIDATE_NOISE_CONTRACT
        or binding.get("terminal_supervision_contract")
        != TERMINAL_SUPERVISION_CONTRACT
        or binding.get("event_age_contract") != EVENT_AGE_CONTRACT
        or binding.get("terminal_horizon_contract") != TERMINAL_HORIZON_CONTRACT
        or binding.get("branch_root_snapshot_contract")
        != BRANCH_ROOT_SNAPSHOT_CONTRACT
        or binding.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
        or binding.get("branch_diagnostic_contract") != BRANCH_DIAGNOSTIC_CONTRACT
        or binding.get("heldout_labels_may_train_fit_calibrate_or_select") is not False
        or binding.get("canonical_shared_body_rows") != 1
        or binding.get("execution_authority")
        != {
            "explicit_user_training_request_recorded": True,
            "public_data_only": True,
            "protected_internal_data_allowed": False,
            "remote_cuda_only": True,
        }
        or binding.get("materialization_receipt")
        != {
            "path": materialization_relative,
            "sha256": sha256_file(args.materialization_receipt),
        }
        or binding.get("actor_authority")
        != {"path": actor_relative, "sha256": actor_file_sha}
        or binding.get("body_manifests") != expected_body_bindings
    ):
        raise LoboWatcherError("existing training binding changed after collection")
    return actor_authority, binding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-exit", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=Path(__file__).with_name(
        "train_robotwin2_five_body_lobo_shared_event_head_v1.py"
    ))
    parser.add_argument(
        "--training-python", type=Path,
        default=Path("/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--expected-gpu-uuid", default=EXPECTED_GPU_UUID)
    return parser.parse_args()


def normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "branches_root", "actor_checkpoint", "materialization_receipt",
        "actor_authority", "binding", "output_root", "state", "run_exit",
        "trainer", "training_python",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.poll_seconds <= 0:
        raise LoboWatcherError("poll interval must be positive")
    return args


def main() -> int:
    global _FAILURE_RUN_EXIT_PATH, _FAILURE_STATE_PATH
    args = normalized_args(parse_args())
    _FAILURE_STATE_PATH = args.state
    _FAILURE_RUN_EXIT_PATH = args.run_exit

    def write_state(status: str, **extra: Any) -> None:
        atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "branches_root": str(args.branches_root),
                "actor_checkpoint": str(args.actor_checkpoint),
                "binding": str(args.binding),
                "output_root": str(args.output_root),
                "expected_decisions": TOTAL_DECISIONS,
                "expected_branches": TOTAL_BRANCHES,
                **extra,
            },
        )

    for required in (args.materialization_receipt, args.trainer, args.training_python):
        if not required.exists():
            raise FileNotFoundError(required)
    gpu = gpu_identity()
    if "4090" not in gpu["name"] or gpu["uuid"] != args.expected_gpu_uuid:
        raise LoboWatcherError(f"unexpected GPU authority: {gpu}")

    collection: dict[str, Any] | None = None
    while collection is None:
        progress = collection_progress(args.branches_root)
        reject_irrecoverable_progress(progress)
        actor_ready = args.actor_checkpoint.is_dir()
        upstream_authorities_ready = (
            args.actor_authority.is_file() and args.binding.is_file()
        )
        if (
            not progress_is_complete(progress)
            or not actor_ready
            or not upstream_authorities_ready
        ):
            write_state(
                "waiting_for_complete_public_branches",
                collection_progress=progress,
                actor_checkpoint_present=actor_ready,
                upstream_authorities_present=upstream_authorities_ready,
                gpu=gpu,
                gpu_reserved_by_watcher=False,
            )
            time.sleep(args.poll_seconds)
            continue
        write_state(
            "validating_exact_2000_decisions_8000_branches",
            collection_progress=progress,
            actor_checkpoint_present=True,
            gpu=gpu,
            outcome_or_event_arrays_interpreted_by_watcher=False,
        )
        collection = validate_complete_collection(args.branches_root, args.actor_checkpoint)

    write_state("hashing_frozen_actor_checkpoint", collection_audit=collection, gpu=gpu)
    checkpoint_sha, checkpoint_files, checkpoint_bytes = sha256_tree(args.actor_checkpoint)
    actor_authority, binding = validate_existing_authorities(
        args, collection, checkpoint_sha, checkpoint_files, checkpoint_bytes
    )
    binding_sha = sha256_file(args.binding)
    fold_results: list[dict[str, Any]] = []
    environment = os.environ.copy()
    code_root = args.trainer.parent
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(code_root),
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    logs.mkdir(exist_ok=True)
    for fold_index, held_out_body in enumerate(BODIES):
        fold_output = args.output_root / f"outer_lobo_{held_out_body}"
        completed_summary = fold_output / "training_summary.json"
        if completed_summary.is_file():
            fold_results.append(
                summarize_fold(fold_output, held_out_body, binding_sha)
            )
            write_state(
                "fold_already_complete",
                collection_audit=collection,
                completed_folds=[row["held_out_body"] for row in fold_results],
                current_fold=held_out_body,
                gpu=gpu,
            )
            continue
        if fold_output.exists():
            raise LoboWatcherError(
                f"incomplete fold output requires a new formal version: {fold_output}"
            )
        while True:
            compute_pids = gpu_compute_pids()
            if not compute_pids:
                break
            write_state(
                "waiting_for_gpu_after_complete_collection",
                collection_audit=collection,
                completed_folds=[row["held_out_body"] for row in fold_results],
                next_fold=held_out_body,
                external_gpu_compute_pids=compute_pids,
                gpu_reserved_by_watcher=False,
            )
            time.sleep(args.poll_seconds)
        command = [
            str(args.training_python),
            str(args.trainer),
            "--mode", "train-fold",
            "--binding", str(args.binding),
            "--binding-sha256", binding_sha,
            "--held-out-body", held_out_body,
            "--split-seed", "20260901",
            "--output", str(fold_output),
            "--device", "cuda",
            "--steps", "3000",
            "--eval-every", "100",
            "--batch-size", "64",
            "--learning-rate", "0.0003",
            "--ensemble-seeds", *[str(seed) for seed in ENSEMBLE_SEEDS],
        ]
        log_path = logs / f"outer_lobo_{held_out_body}.log"
        write_state(
            "training_fold",
            collection_audit=collection,
            completed_folds=[row["held_out_body"] for row in fold_results],
            fold_index=fold_index,
            current_fold=held_out_body,
            command=command,
            log=str(log_path),
            ensemble_members=5,
            steps_per_member=3000,
            gpu=gpu,
        )
        with log_path.open("x", encoding="utf-8") as stream:
            training = subprocess.run(
                command,
                cwd=code_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if training.returncode != 0:
            raise LoboWatcherError(
                f"outer LOBO fold {held_out_body} failed with exit {training.returncode}"
            )
        fold_results.append(summarize_fold(fold_output, held_out_body, binding_sha))
        write_state(
            "fold_complete",
            collection_audit=collection,
            completed_folds=[row["held_out_body"] for row in fold_results],
            last_fold_result=fold_results[-1],
            gpu=gpu,
        )

    macro_deltas = [
        row["one_deviation_ensemble_source_validation"][
            "macro_one_deviation_branch_success_gain"
        ]
        for row in fold_results
    ]
    final = signed(
        {
            "format": FINAL_FORMAT,
            "status": "five_outer_lobo_source_only_training_complete",
            "completed_at_utc": datetime.fromtimestamp(
                max(
                    (args.output_root / f"outer_lobo_{body}" / "training_summary.json")
                    .stat()
                    .st_mtime
                    for body in BODIES
                ),
                timezone.utc,
            ).isoformat(),
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "event_derivation_implementation_sha256": collection[
                "event_derivation_implementation_sha256"
            ],
            "collection_audit": collection,
            "actor_authority": {
                "path": str(args.actor_authority),
                "file_sha256": sha256_file(args.actor_authority),
                "logical_sha256": actor_authority["logical_sha256"],
                "checkpoint_sha256": checkpoint_sha,
            },
            "training_binding": {
                "path": str(args.binding),
                "file_sha256": binding_sha,
                "logical_sha256": binding["logical_sha256"],
            },
            "outer_folds": fold_results,
            "fold_count": len(fold_results),
            "members_per_fold": 5,
            "steps_per_member": 3000,
            "source_validation_fold_mean_macro_one_deviation_branch_success_gain": numeric_summary(
                macro_deltas
            ),
            "heldout_labels_used_for_training_normalization_or_selection": False,
            "heldout_task_success_measured": False,
            "cross_embodiment_task_success_claim_authorized": False,
            "next_required_stage": "five_body_live_paired_baseline_vs_best_of_4_evaluation",
        }
    )
    final_path = args.output_root / "five_fold_training_summary.json"
    create_once_or_verify(final_path, final, "five-fold aggregate")
    args.run_exit.write_text("0\n", encoding="utf-8")
    write_state(
        "complete",
        collection_audit=collection,
        completed_folds=list(BODIES),
        final_summary=str(final_path),
        final_summary_file_sha256=sha256_file(final_path),
        source_validation_fold_mean_macro_one_deviation_branch_success_gain=final[
            "source_validation_fold_mean_macro_one_deviation_branch_success_gain"
        ],
        gpu=gpu,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if _FAILURE_STATE_PATH is not None:
            atomic_json(
                _FAILURE_STATE_PATH,
                {
                    "format": FORMAT,
                    "status": "failed",
                    "updated_at_utc": utc_now(),
                    "pid": os.getpid(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        if _FAILURE_RUN_EXIT_PATH is not None:
            _FAILURE_RUN_EXIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _FAILURE_RUN_EXIT_PATH.write_text("1\n", encoding="utf-8")
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
