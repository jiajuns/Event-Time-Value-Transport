#!/usr/bin/env python3
"""Frozen, data-free event semantics for RoboTwin2 ``move_can_pot``.

The public task code defines the two objects, the initial-side rule and the
0.18 m pot-relative target.  This module is the sole implementation used to
materialize LOBO training rows and to build online paired-runner state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT = "etsf_robotwin2_move_can_pot_five_body_analytic_event_spec_v1"
TASK = "move_can_pot"
EVENT_NAMES = ("e0", "e12", "e3", "e4", "eK")
EVENT_TO_ID = {name: index for index, name in enumerate(EVENT_NAMES)}
EVENT_SPEC_SHA256 = "4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee"
PUBLIC_TASK_COMMIT = "30954692d06ba7e89f7a6b76064f4062c488fa81"
REQUIRED_OBJECTS = ("can", "pot")

GOAL_RULE = {
    "anchor_pose_source": "episode_initial_pot_xyz",
    "kind": "initial_moving_relative_anchor_x_sign",
    "lateral_offset_m": 0.18,
    "negative_initial_relative_x_offset_m": -0.18,
    "positive_initial_relative_x_offset_m": 0.18,
    "side_sign_source": "sign(initial_can_x_minus_initial_pot_x)",
    "y_offset_m": 0.0,
    "z_offset_m": 0.0,
    "zero_sign_policy": "error_unreachable_under_public_task_initialization",
}
THRESHOLDS = {
    "lifted_delta_z_m": 0.01,
    "moved_displacement_m": 0.01,
    "near_goal_euclidean_m": 0.02,
    "stationary_speed_m_per_s": 0.01,
    "stationary_window_seconds": 0.2,
}
EVENT_RULES = {
    "e0": (
        "not_moved_and_not_lifted_and_not_near_goal_and_not_stationary_"
        "and_not_terminal_success"
    ),
    "e12": "moved_or_lifted",
    "e3": "near_goal",
    "e4": "near_goal_and_stationary_for_window",
    "eK": "terminal_simulator_success",
    "precedence_low_to_high": list(EVENT_NAMES),
}


class AnalyticEventSpecError(RuntimeError):
    """The frozen analytic specification or a task trajectory is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_event_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact public-code/prior contract and return calibration."""

    construction = value.get("construction")
    provenance = value.get("provenance")
    public_code = provenance.get("public_task_code") if isinstance(provenance, Mapping) else None
    threshold_provenance = provenance.get("thresholds") if isinstance(provenance, Mapping) else None
    calibration_root = value.get("calibration")
    calibration = (
        calibration_root.get(TASK) if isinstance(calibration_root, Mapping) else None
    )
    if (
        value.get("format") != FORMAT
        or value.get("status") != "frozen_data_and_label_free_analytic_spec"
        or value.get("task") != TASK
        or value.get("event_names") != list(EVENT_NAMES)
        or construction
        != {
            "data_or_label_files_opened": 0,
            "labels_or_outcomes_used_to_fit": False,
            "method": "analytic_public_task_code_and_explicit_physical_priors_only",
            "protected_hdf_or_labels_read": False,
            "trainable": False,
        }
        or not isinstance(calibration_root, Mapping)
        or set(calibration_root) != {TASK}
        or not isinstance(calibration, Mapping)
        or calibration.get("moving") != "can"
        or calibration.get("anchor") != "pot"
        or calibration.get("required_objects") != list(REQUIRED_OBJECTS)
        or calibration.get("goal_rule") != GOAL_RULE
        or calibration.get("thresholds") != THRESHOLDS
        or calibration.get("event_rules") != EVENT_RULES
        or not isinstance(public_code, Mapping)
        or public_code.get("commit") != PUBLIC_TASK_COMMIT
        or public_code.get("path") != "envs/move_can_pot.py"
        or not isinstance(threshold_provenance, Mapping)
        or set(threshold_provenance) != set(THRESHOLDS)
    ):
        raise AnalyticEventSpecError("analytic move_can_pot event spec changed")
    for name, expected in THRESHOLDS.items():
        row = threshold_provenance.get(name)
        if (
            not isinstance(row, Mapping)
            or isinstance(row.get("value"), bool)
            or not isinstance(row.get("value"), (int, float))
            or float(row["value"]) != expected
            or row.get("kind")
            not in {
                "analytic_from_public_task_code",
                "explicit_physical_prior_not_fitted",
            }
        ):
            raise AnalyticEventSpecError(f"threshold provenance changed for {name}")
    return dict(calibration)


def load_event_spec(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise AnalyticEventSpecError("analytic event spec must be a real file")
    if sha256_file(resolved) != EVENT_SPEC_SHA256:
        raise AnalyticEventSpecError("analytic event spec SHA-256 mismatch")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalyticEventSpecError("analytic event spec must be a JSON object")
    return value, validate_event_spec(value)


def event_contract(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Small exact contract copied into every body manifest/checkpoint."""

    validate_event_spec(
        {
            "format": FORMAT,
            "status": "frozen_data_and_label_free_analytic_spec",
            "task": TASK,
            "event_names": list(EVENT_NAMES),
            "construction": {
                "data_or_label_files_opened": 0,
                "labels_or_outcomes_used_to_fit": False,
                "method": "analytic_public_task_code_and_explicit_physical_priors_only",
                "protected_hdf_or_labels_read": False,
                "trainable": False,
            },
            "calibration": {TASK: dict(calibration)},
            "provenance": {
                "public_task_code": {
                    "commit": PUBLIC_TASK_COMMIT,
                    "path": "envs/move_can_pot.py",
                },
                "thresholds": {
                    name: {
                        "kind": (
                            "analytic_from_public_task_code"
                            if name == "near_goal_euclidean_m"
                            else "explicit_physical_prior_not_fitted"
                        ),
                        "value": threshold,
                    }
                    for name, threshold in THRESHOLDS.items()
                },
            },
        }
    )
    return {
        "format": FORMAT,
        "event_names": list(EVENT_NAMES),
        "required_objects": list(REQUIRED_OBJECTS),
        "moving": "can",
        "anchor": "pot",
        "goal_rule": dict(GOAL_RULE),
        "thresholds": dict(THRESHOLDS),
        "event_rules": dict(EVENT_RULES),
        "labels_or_outcomes_used_to_fit_spec": False,
        "data_or_label_files_opened_to_construct_spec": 0,
        "public_task_code_commit": PUBLIC_TASK_COMMIT,
    }


def validate_event_contract(value: Any) -> None:
    expected_calibration = {
        "moving": "can",
        "anchor": "pot",
        "required_objects": list(REQUIRED_OBJECTS),
        "goal_rule": dict(GOAL_RULE),
        "thresholds": dict(THRESHOLDS),
        "event_rules": dict(EVENT_RULES),
    }
    if value != event_contract(expected_calibration):
        raise AnalyticEventSpecError("embedded analytic event contract changed")


def goal_vector(
    poses: np.ndarray,
    names: Sequence[str],
    step: int,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return current can xyz and the identical training/online goal residual."""

    if not 0 <= step < len(poses):
        raise AnalyticEventSpecError("goal-vector step is outside the trajectory")
    if calibration.get("goal_rule") != GOAL_RULE:
        raise AnalyticEventSpecError("goal-vector rule is not the frozen analytic rule")
    name_list = list(names)
    if not set(REQUIRED_OBJECTS).issubset(name_list):
        raise AnalyticEventSpecError("trajectory lacks can/pot required by goal rule")
    moving_index = name_list.index("can")
    anchor_index = name_list.index("pot")
    initial_relative_x = float(
        poses[0, moving_index, 0] - poses[0, anchor_index, 0]
    )
    if not np.isfinite(initial_relative_x) or initial_relative_x == 0.0:
        raise AnalyticEventSpecError(
            "initial can-pot x sign is zero/non-finite; public initialization forbids it"
        )
    x_offset = 0.18 if initial_relative_x > 0.0 else -0.18
    goal = np.asarray(poses[0, anchor_index, :3], dtype=np.float32).copy()
    goal += np.asarray([x_offset, 0.0, 0.0], dtype=np.float32)
    moving = np.asarray(poses[step, moving_index, :3], dtype=np.float32)
    return moving, (goal - moving).astype(np.float32)


def derive_predicates_and_events(
    poses: np.ndarray,
    sim_times: np.ndarray,
    names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive e0/e12/e3/e4/eK from geometry and simulator success only."""

    poses = np.asarray(poses)
    sim_times = np.asarray(sim_times, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[0] == 0 or poses.shape[2] < 3:
        raise AnalyticEventSpecError("trajectory poses must be non-empty [T,N,D>=3]")
    if sim_times.shape != (len(poses),) or not np.isfinite(sim_times).all():
        raise AnalyticEventSpecError("trajectory simulation timestamps are invalid")
    if np.any(np.diff(sim_times) <= 0.0):
        raise AnalyticEventSpecError("trajectory simulation timestamps must increase")
    if calibration.get("thresholds") != THRESHOLDS:
        raise AnalyticEventSpecError("trajectory thresholds are not frozen analytic values")
    moving_index = list(names).index("can")
    position = np.asarray(poses[:, moving_index, :3], dtype=np.float64)
    displacement = np.linalg.norm(position - position[0], axis=1)
    moved = displacement >= THRESHOLDS["moved_displacement_m"]
    lifted = position[:, 2] >= (
        position[0, 2] + THRESHOLDS["lifted_delta_z_m"]
    )
    near = np.asarray(
        [
            np.linalg.norm(goal_vector(poses, names, step, calibration)[1])
            <= THRESHOLDS["near_goal_euclidean_m"]
            for step in range(len(poses))
        ],
        dtype=bool,
    )
    speed = np.r_[
        0.0,
        np.linalg.norm(np.diff(position, axis=0), axis=1) / np.diff(sim_times),
    ]
    stationary = np.zeros(len(poses), dtype=bool)
    for step in range(1, len(poses)):
        start = step
        while (
            start > 0
            and near[start]
            and speed[start] <= THRESHOLDS["stationary_speed_m_per_s"]
        ):
            start -= 1
            if (
                near[start]
                and sim_times[step] - sim_times[start]
                >= THRESHOLDS["stationary_window_seconds"]
            ):
                stationary[step] = True
                break
    succeeded = np.zeros(len(poses), dtype=bool)
    if bool(success):
        succeeded[-1] = True
    predicates = np.stack((moved, lifted, near, stationary, succeeded), axis=-1)
    events = np.full(len(poses), EVENT_TO_ID["e0"], dtype=np.int64)
    events[moved | lifted] = EVENT_TO_ID["e12"]
    events[near] = EVENT_TO_ID["e3"]
    events[stationary] = EVENT_TO_ID["e4"]
    events[succeeded] = EVENT_TO_ID["eK"]
    return predicates.astype(np.float32), events


__all__ = [
    "AnalyticEventSpecError",
    "EVENT_NAMES",
    "EVENT_SPEC_SHA256",
    "EVENT_TO_ID",
    "FORMAT",
    "GOAL_RULE",
    "PUBLIC_TASK_COMMIT",
    "REQUIRED_OBJECTS",
    "TASK",
    "THRESHOLDS",
    "derive_predicates_and_events",
    "event_contract",
    "goal_vector",
    "load_event_spec",
    "sha256_file",
    "validate_event_contract",
    "validate_event_spec",
]
