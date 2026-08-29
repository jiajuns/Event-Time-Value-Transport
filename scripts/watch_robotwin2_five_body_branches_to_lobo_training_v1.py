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
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
STATE_SCHEMA = "dual_ee_object_relative_state_27d_v2"
ACTION_SCHEMA = "dual_ee_se3_gripper_delta_14d_v2"
EVENTS = ("e0", "e12", "e3", "e4", "eK")
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
CANDIDATE_COUNT = 4
SEED_START = 2026081000
DEVELOPMENT_SEED_STOP = 2026090000
SEEDS_PER_CONDITION_QUERY = 50
QUERY_INDICES = (0, 5, 10, 15)
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
    "candidate_index",
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
    "dt",
}
INTEGER_ARRAYS = {
    "current_event_id", "post_event_id", "next_event_id", "candidate_index"
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
    except zipfile.BadZipFile as error:
        raise LoboWatcherError(f"decision NPZ is corrupt: {path}") from error
    return {"candidate_count": 4, "action_horizon": horizon}


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
                        f"{body}/{key} contains more than 50 declared decisions"
                    )


def validate_complete_collection(
    branches_root: Path, actor_checkpoint: Path
) -> dict[str, Any]:
    manifest_bindings: dict[str, dict[str, str]] = {}
    manifest_audits: dict[str, Any] = {}
    shared_adapter_sha: str | None = None
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
            or manifest.get("body") != body
            or manifest.get("actor_checkpoint") != str(actor_checkpoint)
            or manifest.get("candidate_count") != CANDIDATE_COUNT
            or manifest.get("candidate_zero_is_actor_baseline") is not True
            or manifest.get("same_ordered_candidate_set_for_baseline_and_etsf") is not True
            or manifest.get("root_query_indices") != list(QUERY_INDICES)
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
                "zero_step_infeasible_candidate_keeps_failure_and_action_binding": True,
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
            shared_event_spec_sha = event_spec_sha
            shared_event_implementation_sha = event_implementation_sha
        if (
            adapter_sha != shared_adapter_sha
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
            relative = item.get("path")
            if (
                item.get("condition") != condition
                or item.get("requested_seed") != seed
                or item.get("root_query_index") != query
                or relative != f"groups/{expected_filename}"
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
            ):
                raise LoboWatcherError(f"{body}/{group_id} identity fields changed")
            payload = branches_root / body / str(relative)
            try:
                payload.resolve().relative_to((branches_root / body).resolve())
            except ValueError as error:
                raise LoboWatcherError(f"{body}/{group_id} payload escapes body root") from error
            decision = validate_decision_npz(payload, str(item["sha256"]))
            action_horizons.add(int(decision["action_horizon"]))
            observed_ids.add(str(group_id))
            observed_paths.add(str(relative))
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
                f"{body} does not contain 50 unique development seeds per condition/query"
            )
        disk_paths = {
            path.relative_to(branches_root / body).as_posix()
            for path in (branches_root / body / "groups").glob("*.npz")
            if path.is_file()
        }
        if disk_paths != observed_paths:
            raise LoboWatcherError(f"{body} has missing or extra decision NPZ files")
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
        }
    if decision_count != TOTAL_DECISIONS or branch_count != TOTAL_BRANCHES:
        raise LoboWatcherError("five-body collection total is not exactly 2000/8000")
    return {
        "status": "complete_public_five_body_candidate_collection_verified",
        "decisions": decision_count,
        "branches": branch_count,
        "bodies": manifest_audits,
        "adapter_implementation_sha256": shared_adapter_sha,
        "event_spec_sha256": shared_event_spec_sha,
        "event_derivation_implementation_sha256": shared_event_implementation_sha,
        "action_horizons": sorted(action_horizons),
        "outcome_or_event_arrays_interpreted_by_watcher": False,
        "candidate_index_and_dt_arrays_interpreted_by_watcher": True,
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
        "source_negative_to_positive_ratio": summary.get(
            "source_negative_to_positive_ratio"
        ),
        "source_validation": {
            "macro_best_of_4_delta_success_rate": numeric_summary(
                nested_values(members, "candidate_ranking", "macro_delta_success_rate")
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


def build_authorities(
    args: argparse.Namespace,
    collection: Mapping[str, Any],
    checkpoint_sha: str,
    checkpoint_files: int,
    checkpoint_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority_dir = args.actor_authority.parent.resolve()
    checkpoint_relative = contained_relative(
        authority_dir, args.actor_checkpoint, "actor checkpoint"
    )
    sampling_contract = {
        "format": "etsf_robotwin2_five_body_ordered_flow_candidate_contract_v1",
        "collector_format": COLLECTOR_FORMAT,
        "task": TASK,
        "conditions": list(CONDITIONS),
        "development_seed_interval": {
            "start_inclusive": SEED_START,
            "stop_exclusive": DEVELOPMENT_SEED_STOP,
        },
        "seeds_per_body_condition_query": SEEDS_PER_CONDITION_QUERY,
        "supplemental_development_seeds_after_terminal_root_skip_allowed": True,
        "root_query_indices": list(QUERY_INDICES),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_zero_is_actor_baseline": True,
        "same_ordered_candidate_set_for_baseline_and_etsf": True,
        "state_schema": STATE_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "actor_checkpoint_sha256": checkpoint_sha,
        "adapter_implementation_sha256": collection["adapter_implementation_sha256"],
        "event_spec_sha256": collection["event_spec_sha256"],
        "planned_dt_seconds": 5.0 / 15.0,
        "duration_time_source": "counted_successful_sapien_scene_step_calls",
    }
    sampling_sha = canonical_sha256(sampling_contract)
    actors = {
        body: {
            "family": "smolvla_v0.4.4_universal_five_body_ee16_actor",
            "frozen": True,
            "optimizer_updates_allowed": False,
            "checkpoint_path": checkpoint_relative,
            "checkpoint_kind": "directory_tree",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_file_count": checkpoint_files,
            "checkpoint_total_bytes": checkpoint_bytes,
            "sampling_contract_sha256": sampling_sha,
            "candidate_count": CANDIDATE_COUNT,
            "candidate_zero_is_actor_baseline": True,
            "same_ordered_candidate_set_for_baseline_and_etsf": True,
        }
        for body in BODIES
    }
    actor_authority = signed(
        {
            "format": ACTOR_FORMAT,
            "task": TASK,
            "universal_actor_same_checkpoint_for_all_bodies": True,
            "sampling_contract": sampling_contract,
            "actors": actors,
        }
    )
    create_once_or_verify(args.actor_authority, actor_authority, "actor authority")
    actor_file_sha = sha256_file(args.actor_authority)
    binding_dir = args.binding.parent.resolve()
    materialization_relative = contained_relative(
        binding_dir, args.materialization_receipt, "materialization receipt"
    )
    actor_relative = contained_relative(binding_dir, args.actor_authority, "actor authority")
    body_bindings: dict[str, Any] = {}
    for body in BODIES:
        manifest_path = args.branches_root / body / "manifest.json"
        body_bindings[body] = {
            "path": contained_relative(binding_dir, manifest_path, f"{body} manifest"),
            "sha256": collection["bodies"][body]["manifest_file_sha256"],
        }
    binding = signed(
        {
            "format": BINDING_FORMAT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "heldout_labels_may_train_fit_calibrate_or_select": False,
            "canonical_shared_body_rows": 1,
            "execution_authority": {
                "explicit_user_training_request_recorded": True,
                "public_data_only": True,
                "protected_internal_data_allowed": False,
                "remote_cuda_only": True,
            },
            "materialization_receipt": {
                "path": materialization_relative,
                "sha256": sha256_file(args.materialization_receipt),
            },
            "actor_authority": {"path": actor_relative, "sha256": actor_file_sha},
            "body_manifests": body_bindings,
            "collection_audit": {
                "decisions": collection["decisions"],
                "branches": collection["branches"],
                "five_bodies": list(BODIES),
                "conditions": list(CONDITIONS),
                "outcome_or_event_arrays_interpreted_by_watcher": False,
            },
        }
    )
    create_once_or_verify(args.binding, binding, "training binding")
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
        if not progress_is_complete(progress) or not actor_ready:
            write_state(
                "waiting_for_complete_public_branches",
                collection_progress=progress,
                actor_checkpoint_present=actor_ready,
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
    actor_authority, binding = build_authorities(
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
        row["source_validation"]["macro_best_of_4_delta_success_rate"]["mean"]
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
            "source_validation_fold_mean_macro_best_of_4_delta_success_rate": numeric_summary(
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
        source_validation_fold_mean_macro_best_of_4_delta_success_rate=final[
            "source_validation_fold_mean_macro_best_of_4_delta_success_rate"
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
