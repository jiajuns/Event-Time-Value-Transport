#!/usr/bin/env python3
"""Paired RoboTwin2 actor deployment study: execute-5/replan vs native-50.

This runner deliberately contains no critic and performs no training.  Both
arms execute proposal zero from the same frozen EE16 actor.  They differ only
in how many tokens of each native 50-token chunk are executed before querying
the actor again.  Every body/condition/seed pair is run from two fresh scenes
whose complete observable reset, canonical query-0 state, proposal-zero chunk,
and first token are bound before either arm is executed.

The output protocol is create-once.  A completed method is never rerun, an
interrupted method may be restarted once under an explicit non-informative
interruption assumption, and an ordinary Python/simulator exception creates an
immutable failure receipt.  This supports host/process restart without
silently turning outcome-aware retries into extra samples.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_robotwin2_five_body_paired_success_v1 as formal
import robotwin2_move_can_pot_analytic_event_spec_v2 as analytic_event
import materialize_robotwin2_stable_seed_roster_v1 as stable_roster


collector = formal.collector
shared_head = formal.shared_head

FORMAT = "etsf_robotwin2_five_body_actor_execute5_vs_execute50_v2_stable_roster"
BINDING_FORMAT = "etsf_robotwin2_actor_deployment_protocol_binding_v2_stable_roster"
ATTEMPT_FORMAT = "etsf_robotwin2_actor_deployment_pair_attempt_v1"
COMMITMENT_FORMAT = "etsf_robotwin2_actor_deployment_initial_commitment_v1"
METHOD_START_FORMAT = "etsf_robotwin2_actor_deployment_method_start_v1"
METHOD_RESUME_FORMAT = "etsf_robotwin2_actor_deployment_method_resume_v1"
METHOD_RESULT_FORMAT = "etsf_robotwin2_actor_deployment_method_result_v1"
METHOD_FAILURE_FORMAT = "etsf_robotwin2_actor_deployment_method_failure_v1"
PAIR_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_pair_v1"
OUTCOME_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_outcomes_v1"
REPORT_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_report_v1"
COMPLETION_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_completion_v1"

BENCHMARK = formal.BENCHMARK
TASK = formal.TASK
BODIES = formal.BODIES
CONDITIONS = formal.CONDITIONS
METHOD_EXECUTE5 = "actor_candidate0_execute5_replan"
METHOD_EXECUTE50 = "actor_candidate0_execute50_native"
METHODS = (METHOD_EXECUTE5, METHOD_EXECUTE50)
EXECUTION_STEPS = {METHOD_EXECUTE5: 5, METHOD_EXECUTE50: 50}
NATIVE_CHUNK_STEPS = 50
NATIVE_EE_DIM = formal.NATIVE_EE_DIM
MAX_EPISODE_ACTION_STEPS = 200
ACTOR_DATASET_FPS = formal.ACTOR_DATASET_FPS
QUERY_CANONICALIZATION_STEPS = formal.QUERY_CANONICALIZATION_STEPS
STAGE_DENOMINATOR = formal.STAGE_DENOMINATOR
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256

# Runtime values are replaced by the validated create-once stable-roster file
# before any actor/simulator setup.  The initial values only keep pure helper
# imports deterministic; they do not authorize a production run.
SEED_BASE = stable_roster.CANDIDATE_SEED_START
SEED_COUNT = 20
SEED_ROSTER = tuple(range(SEED_BASE, SEED_BASE + SEED_COUNT))
STABLE_SEED_ROSTER: dict[str, Any] | None = None
STABLE_SEED_ROSTER_PATH: Path | None = None
STABLE_SEED_ROSTER_FILE_SHA256: str | None = None
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261019
METHOD_RESUME_LIMIT = 1


canonical_sha256 = formal.canonical_sha256
sha256_file = formal.sha256_file
array_sha256 = formal.array_sha256


class ActorDeploymentProtocolError(RuntimeError):
    """The frozen actor, paired reset, recovery chain, or report changed."""


def configure_stable_seed_roster(
    value: Mapping[str, Any], *, path: Path, file_sha256: str
) -> dict[str, Any]:
    try:
        validated = stable_roster.validate_stable_seed_roster(value)
    except stable_roster.StableSeedRosterError as error:
        raise ActorDeploymentProtocolError(str(error)) from error
    selected = tuple(int(seed) for seed in validated["selected_seeds"])
    if len(selected) != SEED_COUNT or len(set(selected)) != SEED_COUNT:
        raise ActorDeploymentProtocolError("stable roster does not contain 20 unique seeds")
    global SEED_ROSTER, SEED_BASE, STABLE_SEED_ROSTER
    global STABLE_SEED_ROSTER_PATH, STABLE_SEED_ROSTER_FILE_SHA256
    SEED_ROSTER = selected
    SEED_BASE = min(selected)
    STABLE_SEED_ROSTER = dict(validated)
    STABLE_SEED_ROSTER_PATH = path.expanduser().resolve()
    STABLE_SEED_ROSTER_FILE_SHA256 = file_sha256
    return dict(validated)


def require_stable_seed_roster() -> dict[str, Any]:
    if (
        STABLE_SEED_ROSTER is None
        or STABLE_SEED_ROSTER_PATH is None
        or STABLE_SEED_ROSTER_FILE_SHA256 is None
    ):
        raise ActorDeploymentProtocolError(
            "production actor-v2 requires a file-bound common-stable seed roster"
        )
    return stable_roster.validate_stable_seed_roster(STABLE_SEED_ROSTER)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _staged_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.staged")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_create_once_json(
    path: Path, *, label: str
) -> tuple[dict[str, Any] | None, bool]:
    """Read a final/staged immutable artifact, rejecting divergent copies."""

    stage = _staged_path(path)
    final_exists = path.exists()
    stage_exists = stage.exists()
    if (final_exists and (not path.is_file() or path.is_symlink())) or (
        stage_exists and (not stage.is_file() or stage.is_symlink())
    ):
        raise ActorDeploymentProtocolError(f"{label} is symbolic or non-file")
    if not final_exists and not stage_exists:
        return None, False

    def load(candidate: Path) -> dict[str, Any]:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ActorDeploymentProtocolError(
                f"{label} staged/final JSON is incomplete"
            ) from error
        if not isinstance(value, dict):
            raise ActorDeploymentProtocolError(f"{label} must be a JSON object")
        return value

    final_value = load(path) if final_exists else None
    stage_value = load(stage) if stage_exists else None
    if final_value is not None and stage_value is not None and final_value != stage_value:
        raise ActorDeploymentProtocolError(f"{label} final/staged copies differ")
    return (
        final_value if final_value is not None else stage_value,
        final_value is None and stage_value is not None,
    )


def promote_create_once_json(
    path: Path, value: Mapping[str, Any], *, label: str
) -> str:
    """Persist JSON by fsync and hard-link creation; never replace a value."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = _staged_path(path)
    existing, staged_only = read_create_once_json(path, label=label)
    if existing is None:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        try:
            with stage.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            stage.chmod(0o444)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        existing, staged_only = read_create_once_json(path, label=label)
    if existing != dict(value):
        raise ActorDeploymentProtocolError(f"{label} create-once value changed")
    if staged_only:
        try:
            os.link(stage, path)
        except FileExistsError:
            current, _ = read_create_once_json(path, label=label)
            if current != dict(value):
                raise ActorDeploymentProtocolError(f"{label} link race changed value")
        _fsync_directory(path.parent)
    current, _ = read_create_once_json(path, label=label)
    if current != dict(value) or not path.is_file() or path.is_symlink():
        raise ActorDeploymentProtocolError(f"{label} final promotion failed")
    if stage.exists():
        stage.unlink()
        _fsync_directory(path.parent)
    return sha256_file(path)


def evaluation_protocol() -> dict[str, Any]:
    roster = require_stable_seed_roster()
    formal_seeds = set(range(formal.SEED_BASE, formal.SEED_BASE + formal.SEED_COUNT))
    nested_seeds = set(range(2026091000, 2026091100))
    seeds = set(SEED_ROSTER)
    if seeds & formal_seeds or seeds & nested_seeds:
        raise ActorDeploymentProtocolError("deployment protocol reuses an inspected seed")
    base = {
        "format": "etsf_robotwin2_actor_execute5_vs_execute50_protocol_v2_stable_roster",
        "task": TASK,
        "bodies": list(BODIES),
        "conditions": list(CONDITIONS),
        "selected_seeds": list(SEED_ROSTER),
        "seed_count": SEED_COUNT,
        "stable_seed_roster_logical_sha256": roster["logical_sha256"],
        "stable_seed_roster_file_sha256": STABLE_SEED_ROSTER_FILE_SHA256,
        "pair_count": len(BODIES) * len(CONDITIONS) * SEED_COUNT,
        "rollout_count": len(BODIES) * len(CONDITIONS) * SEED_COUNT * 2,
        "methods": list(METHODS),
        "method_execution_steps": dict(EXECUTION_STEPS),
        "native_actor_chunk_steps": NATIVE_CHUNK_STEPS,
        "candidate_count": 1,
        "candidate_index": 0,
        "same_initial_reset_and_candidate0_chunk": True,
        "fresh_scene_per_method": True,
        "method_order_counterbalanced_before_outcomes": True,
        "query_canonicalization": {
            "raw_scene_steps_before_every_actor_query": (
                QUERY_CANONICALIZATION_STEPS
            ),
            "formal_actor_action_count_advanced": False,
            "physical_time_and_event_age_advanced": True,
            "occurs_once_per_policy_query_and_therefore_frequency_is_part_of_"
            "the_execute5_vs_execute50_deployment_protocol": True,
            "same_as_formal_paired_and_nested_fresh_scene_protocol": True,
        },
        "prompt_factor_held_fixed": collector.DEFAULT_INSTRUCTION,
        "critic_loaded_or_called": False,
        "training_performed": False,
        "bootstrap_unit": (
            "requested_seed_cluster_with_all_selected_body_condition_rows_together"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "method_interruption_resume_limit": METHOD_RESUME_LIMIT,
    }
    return {**base, "logical_sha256": canonical_sha256(base)}


def method_order(body: str, condition: str, seed: int) -> list[str]:
    if body not in BODIES or condition not in CONDITIONS:
        raise ActorDeploymentProtocolError("unknown body/condition in method order")
    if int(seed) not in SEED_ROSTER:
        raise ActorDeploymentProtocolError("seed is outside the frozen stable roster")
    parity = (
        BODIES.index(body)
        + CONDITIONS.index(condition)
        + SEED_ROSTER.index(int(seed))
    ) % 2
    return list(METHODS if parity == 0 else reversed(METHODS))


def evaluation_schedule() -> list[dict[str, Any]]:
    result = []
    for body in BODIES:
        for condition in CONDITIONS:
            for seed in SEED_ROSTER:
                result.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": seed,
                        "method_order": method_order(body, condition, seed),
                    }
                )
    return result


def pair_id(body: str, condition: str, seed: int) -> str:
    return formal.pair_id(body, condition, seed)


def validate_actor_candidate0(value: Any) -> np.ndarray:
    candidates = np.asarray(value, dtype=np.float32)
    if (
        candidates.shape != (1, NATIVE_CHUNK_STEPS, NATIVE_EE_DIM)
        or not np.isfinite(candidates).all()
    ):
        raise ActorDeploymentProtocolError(
            "actor proposal zero must be finite [1,50,16] EE actions"
        )
    return candidates


def generate_actor_candidate0(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: Any,
    instruction: str,
    scene_seed: int,
    query_index: int,
    device: torch.device,
) -> np.ndarray:
    return validate_actor_candidate0(
        collector.generate_candidates(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            instruction=instruction,
            scene_seed=scene_seed,
            query_index=query_index,
            candidate_count=1,
            device=device,
        )
    )


def _effect_metrics(reference_ee16: np.ndarray, token_ee16: np.ndarray) -> dict[str, Any]:
    reference = collector.normalize_ee_chunk(
        np.asarray(reference_ee16, dtype=np.float32).reshape(1, NATIVE_EE_DIM)
    )[0]
    token = collector.normalize_ee_chunk(
        np.asarray(token_ee16, dtype=np.float32).reshape(1, NATIVE_EE_DIM)
    )[0]
    effect = collector.canonical_action_chunk(reference, token[None])[0]
    translation = effect[[0, 1, 2, 7, 8, 9]]
    rotation = effect[[3, 4, 5, 10, 11, 12]]
    gripper = effect[[6, 13]]

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))

    return {
        "effect14": effect.astype(float).tolist(),
        "effect14_rms": rms(effect),
        "translation_rms_m": rms(translation),
        "rotation_axis_angle_rms_rad": rms(rotation),
        "gripper_rms": rms(gripper),
    }


def first_token_continuity(
    *,
    live_ee16: np.ndarray,
    first_token_ee16: np.ndarray,
    previous_executed_token_ee16: np.ndarray | None,
    previous_native_next_token_ee16: np.ndarray | None,
) -> dict[str, Any]:
    token = np.asarray(first_token_ee16, dtype=np.float32)
    return {
        "first_token_sha256": array_sha256(token),
        "live_state_to_first_token": _effect_metrics(live_ee16, token),
        "previous_executed_to_first_token": (
            None
            if previous_executed_token_ee16 is None
            else _effect_metrics(previous_executed_token_ee16, token)
        ),
        "previous_native_next_to_replanned_first_token": (
            None
            if previous_native_next_token_ee16 is None
            else _effect_metrics(previous_native_next_token_ee16, token)
        ),
        "computed_in_canonical_two_arm_ee_effect14_frame": True,
        "raw_quaternion_component_rms_used": False,
    }


def summarize_first_token_continuity(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not decisions:
        raise ActorDeploymentProtocolError("continuity summary requires decisions")

    def summary(path: str) -> dict[str, Any]:
        values = []
        for decision in decisions:
            record = decision["first_token_continuity"].get(path)
            if record is not None:
                values.append(float(record["effect14_rms"]))
        return {
            "count": len(values),
            "mean_effect14_rms": None if not values else float(np.mean(values)),
            "max_effect14_rms": None if not values else float(np.max(values)),
        }

    return {
        "query_count": len(decisions),
        "live_state_to_first_token": summary("live_state_to_first_token"),
        "previous_executed_to_first_token": summary(
            "previous_executed_to_first_token"
        ),
        "previous_native_next_to_replanned_first_token": summary(
            "previous_native_next_to_replanned_first_token"
        ),
    }


def _quaternion_roll_pitch_degrees(quaternion_wxyz: Sequence[float]) -> tuple[float, float]:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ActorDeploymentProtocolError("terminal can quaternion is invalid")
    # Use the same transforms3d implementation as the public task checker and
    # analytic event v2; an algebraically equivalent custom formula can differ
    # at the strict 15-degree boundary because of floating-point rounding.
    roll, pitch = analytic_event._roll_pitch_degrees_wxyz(q.reshape(1, 4))
    return float(roll[0]), float(pitch[0])


def native_success_components(task: Any) -> dict[str, Any]:
    """Persist every public move_can_pot checker term at rollout terminal."""

    pot_pose = task.pot.get_pose()
    can_pose = task.can.get_pose()
    pot_xyz = np.asarray(pot_pose.p, dtype=np.float64)
    can_xyz = np.asarray(can_pose.p, dtype=np.float64)
    can_q = np.asarray(can_pose.q, dtype=np.float64)
    roll, pitch = _quaternion_roll_pitch_degrees(can_q)
    arm_side = "left" if task.arm_tag == "left" else "right"
    signed_x = (
        float(pot_xyz[0] - can_xyz[0])
        if arm_side == "left"
        else float(can_xyz[0] - pot_xyz[0])
    )
    abs_y = float(abs(pot_xyz[1] - can_xyz[1]))
    roll_error = float(abs(roll - 90.0))
    pitch_error = float(abs(pitch))
    z_limit = float(task.orig_z) + 0.001
    left_open = bool(task.robot.is_left_gripper_open())
    right_open = bool(task.robot.is_right_gripper_open())
    checks = {
        "correct_signed_side": signed_x > 0.0,
        "signed_x_below_0.2_m": abs(signed_x) < 0.2,
        "absolute_y_below_0.035_m": abs_y < 0.035,
        "roll_error_below_15_deg": roll_error < 15.0,
        "pitch_error_below_15_deg": pitch_error < 15.0,
        "can_z_at_or_below_orig_z_plus_0.001_m": float(can_xyz[2]) <= z_limit,
        "left_gripper_open": left_open,
        "right_gripper_open": right_open,
    }
    recomputed = bool(all(checks.values()))
    official = bool(task.check_success())
    if recomputed != official:
        raise ActorDeploymentProtocolError(
            "persisted move_can_pot success components disagree with check_success"
        )
    return {
        "pot_xyz": pot_xyz.astype(float).tolist(),
        "can_xyz": can_xyz.astype(float).tolist(),
        "can_quaternion_wxyz": (can_q / np.linalg.norm(can_q)).astype(float).tolist(),
        "arm_side": arm_side,
        "signed_x_m": signed_x,
        "absolute_y_m": abs_y,
        "can_roll_degrees": roll,
        "can_pitch_degrees": pitch,
        "roll_error_from_90_degrees": roll_error,
        "pitch_error_from_zero_degrees": pitch_error,
        "can_z_m": float(can_xyz[2]),
        "orig_z_plus_0.001_m": z_limit,
        "checks": checks,
        "recomputed_terminal_check_success": recomputed,
        "official_terminal_check_success": official,
    }


def _commitment_base(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "commitment_sha256"}


def prepare_initial_commitment(
    *,
    body: str,
    condition: str,
    seed: int,
    task_class: Any,
    task_args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    device: torch.device,
) -> dict[str, Any]:
    task = collector._new_task(task_class, task_args, seed, instruction)
    try:
        names, objects = collector.discover_pose_objects(
            task, set(analytic_event.REQUIRED_OBJECTS)
        )
        reset_snapshot = formal.capture_reset_snapshot(task, names, objects)
        task.scene.step()
        canonical_snapshot = formal.capture_reset_snapshot(task, names, objects)
        candidates = generate_actor_candidate0(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            instruction=instruction,
            scene_seed=seed,
            query_index=0,
            device=device,
        )
        after = formal.capture_reset_snapshot(task, names, objects)
        if after != canonical_snapshot:
            raise ActorDeploymentProtocolError(
                "initial candidate0 generation changed observable simulator state"
            )
        base = {
            "format": COMMITMENT_FORMAT,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "frozen_before_any_method_execution": True,
            "candidate_count": 1,
            "candidate_index": 0,
            "candidate_shape": list(candidates.shape),
            "candidate0_chunk_sha256": array_sha256(candidates[0]),
            "candidate0_first_token_sha256": array_sha256(candidates[0, 0]),
            "reset_snapshot": reset_snapshot,
            "reset_identity_sha256": formal.reset_identity(reset_snapshot),
            "canonical_query_snapshot": canonical_snapshot,
            "canonical_query_identity_sha256": formal.reset_identity(
                canonical_snapshot
            ),
            "query_canonicalization_steps": QUERY_CANONICALIZATION_STEPS,
            "candidate_generation_advanced_simulator": False,
            "actor_candidate0_native_ee16": True,
        }
        return {**base, "commitment_sha256": canonical_sha256(base)}
    finally:
        task.close_env(clear_cache=False)


def validate_stored_initial_commitment(
    commitment: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if (
        commitment.get("format") != COMMITMENT_FORMAT
        or commitment.get("heldout_body") != expected["heldout_body"]
        or commitment.get("condition") != expected["condition"]
        or commitment.get("requested_seed") != expected["requested_seed"]
        or commitment.get("resolved_seed") != expected["requested_seed"]
        or commitment.get("frozen_before_any_method_execution") is not True
        or commitment.get("candidate_count") != 1
        or commitment.get("candidate_index") != 0
        or commitment.get("candidate_shape")
        != [1, NATIVE_CHUNK_STEPS, NATIVE_EE_DIM]
        or not _is_sha256(commitment.get("candidate0_chunk_sha256"))
        or not _is_sha256(commitment.get("candidate0_first_token_sha256"))
        or commitment.get("reset_identity_sha256")
        != formal.reset_identity(commitment.get("reset_snapshot", {}))
        or commitment.get("canonical_query_identity_sha256")
        != formal.reset_identity(commitment.get("canonical_query_snapshot", {}))
        or commitment.get("query_canonicalization_steps")
        != QUERY_CANONICALIZATION_STEPS
        or commitment.get("candidate_generation_advanced_simulator") is not False
        or commitment.get("actor_candidate0_native_ee16") is not True
        or commitment.get("commitment_sha256")
        != canonical_sha256(_commitment_base(commitment))
    ):
        raise ActorDeploymentProtocolError("stored initial commitment changed")


def verify_initial_commitment(
    commitment: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    reset_snapshot: Mapping[str, Any],
    canonical_snapshot: Mapping[str, Any],
    candidates: np.ndarray,
) -> None:
    validate_stored_initial_commitment(commitment, expected)
    if (
        commitment["reset_snapshot"] != reset_snapshot
        or commitment["canonical_query_snapshot"] != canonical_snapshot
        or commitment["candidate0_chunk_sha256"] != array_sha256(candidates[0])
        or commitment["candidate0_first_token_sha256"]
        != array_sha256(candidates[0, 0])
    ):
        raise ActorDeploymentProtocolError(
            "method reset/query0 candidate differs from immutable commitment"
        )


def execute_rollout(
    *,
    method: str,
    body: str,
    condition: str,
    seed: int,
    task_class: Any,
    task_args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    calibration: Mapping[str, Any],
    initial_commitment: Mapping[str, Any],
    instruction: str,
    max_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in METHODS or max_steps != MAX_EPISODE_ACTION_STEPS:
        raise ActorDeploymentProtocolError("unknown method or non-formal horizon")
    execute_steps = EXECUTION_STEPS[method]
    expected = {
        "heldout_body": body,
        "condition": condition,
        "requested_seed": seed,
        "method_order": method_order(body, condition, seed),
    }
    validate_stored_initial_commitment(initial_commitment, expected)
    task = collector._new_task(task_class, task_args, seed, instruction)
    decisions: list[dict[str, Any]] = []
    try:
        names, objects = collector.discover_pose_objects(
            task, set(analytic_event.REQUIRED_OBJECTS)
        )
        initial_poses = collector.read_poses(objects)
        initial_ee = collector.current_ee_action16(task)
        initial_snapshot = formal.capture_reset_snapshot(task, names, objects)
        trajectory = [initial_poses]
        sim_times = [collector._sim_time(task)]
        query_index = 0
        initial_canonical_snapshot: Mapping[str, Any] | None = None
        previous_executed: np.ndarray | None = None
        previous_native_next: np.ndarray | None = None
        while not collector._episode_done(task, max_steps):
            # This is the formal paired/nested fresh-scene contact-cache
            # canonicalization, not an actor action.  It occurs once per query,
            # so its physical-time frequency is intentionally part of the two
            # deployment protocols and is frozen in evaluation_protocol().
            task.scene.step()
            collector._append_physical_observation(
                task, objects, trajectory, sim_times
            )
            live_ee = collector.current_ee_action16(task)
            pre_candidate = formal.capture_reset_snapshot(task, names, objects)
            candidates = generate_actor_candidate0(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                instruction=instruction,
                scene_seed=seed,
                query_index=query_index,
                device=device,
            )
            after_candidate = formal.capture_reset_snapshot(task, names, objects)
            if after_candidate != pre_candidate:
                raise ActorDeploymentProtocolError(
                    "actor generation changed observable simulator state"
                )
            if query_index == 0:
                initial_canonical_snapshot = pre_candidate
                verify_initial_commitment(
                    initial_commitment,
                    expected=expected,
                    reset_snapshot=initial_snapshot,
                    canonical_snapshot=pre_candidate,
                    candidates=candidates,
                )
            chunk = candidates[0]
            continuity = first_token_continuity(
                live_ee16=live_ee,
                first_token_ee16=chunk[0],
                previous_executed_token_ee16=previous_executed,
                previous_native_next_token_ee16=previous_native_next,
            )
            executed = 0
            chunk_start = collector._sim_time(task)
            for action in chunk[:execute_steps]:
                if collector._episode_done(task, max_steps):
                    break
                task.take_action(action, action_type="ee")
                executed += 1
                previous_executed = np.asarray(action, dtype=np.float32).copy()
                collector._append_physical_observation(
                    task, objects, trajectory, sim_times
                )
            previous_native_next = (
                np.asarray(chunk[execute_steps], dtype=np.float32).copy()
                if executed == execute_steps and execute_steps < NATIVE_CHUNK_STEPS
                else None
            )
            decisions.append(
                {
                    "query_index": query_index,
                    "candidate_count": 1,
                    "candidate_index": 0,
                    "candidate0_chunk_sha256": array_sha256(chunk),
                    "candidate0_first_token_sha256": array_sha256(chunk[0]),
                    "native_chunk_steps": NATIVE_CHUNK_STEPS,
                    "protocol_execute_steps": execute_steps,
                    "query_canonicalization_physical_steps_before_generation": (
                        QUERY_CANONICALIZATION_STEPS
                    ),
                    "executed_action_count": executed,
                    "first_token_continuity": continuity,
                    "physical_sim_seconds": collector._sim_time(task) - chunk_start,
                    "critic_scores": None,
                }
            )
            query_index += 1

        latched_success = bool(getattr(task, "eval_success", False))
        terminal_components = native_success_components(task)
        terminal_success = bool(
            terminal_components["official_terminal_check_success"]
        )
        success = latched_success or terminal_success
        trajectory_array = np.stack(trajectory).astype(np.float64)
        time_array = np.asarray(sim_times, dtype=np.float64)
        _predicates, events = analytic_event.derive_predicates_and_events(
            trajectory_array,
            time_array,
            names,
            success,
            calibration,
            collector.success_height_reference_z(task),
        )
        _moving_initial, initial_goal = analytic_event.goal_vector(
            trajectory_array, names, 0, calibration
        )
        _moving_terminal, terminal_goal = analytic_event.goal_vector(
            trajectory_array, names, len(trajectory_array) - 1, calibration
        )
        initial_distance = float(np.linalg.norm(initial_goal))
        terminal_distance = float(np.linalg.norm(terminal_goal))
        return {
            "method": method,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "execution_stride_actions": execute_steps,
            "native_chunk_steps": NATIVE_CHUNK_STEPS,
            "candidate_count": 1,
            "candidate_index": 0,
            "initial_reset_identity_sha256": formal.reset_identity(
                initial_snapshot
            ),
            "initial_reset_snapshot": initial_snapshot,
            "initial_canonical_query_snapshot": initial_canonical_snapshot,
            "initial_candidate_commitment_sha256": initial_commitment[
                "commitment_sha256"
            ],
            "initial_candidate0_chunk_sha256": decisions[0][
                "candidate0_chunk_sha256"
            ],
            "initial_candidate0_first_token_sha256": decisions[0][
                "candidate0_first_token_sha256"
            ],
            "tracked_object_names": list(names),
            "initial_object_poses": initial_poses.astype(float).tolist(),
            "initial_ee16": initial_ee.astype(float).tolist(),
            "binary_success": int(success),
            "latched_eval_success": latched_success,
            "terminal_check_success": terminal_success,
            "terminal_native_success_components": terminal_components,
            "stop_reason": "success" if success else "formal_action_limit",
            "stage_progress": formal.stage_progress(events, success),
            "max_event_id": int(events.max()),
            "initial_goal_distance_m": initial_distance,
            "terminal_goal_distance_m": terminal_distance,
            "goal_progress_m": initial_distance - terminal_distance,
            "executed_control_steps": int(getattr(task, "take_action_cnt", 0)),
            "physical_sim_seconds": collector._sim_time(task) - sim_times[0],
            "sim_timestep_seconds": float(task.scene.timestep_seconds),
            "policy_query_count": len(decisions),
            "first_token_continuity_summary": summarize_first_token_continuity(
                decisions
            ),
            "action_execution_error": None,
            "decisions": decisions,
        }
    finally:
        task.close_env(clear_cache=False)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_rollout(
    rollout: Mapping[str, Any], *, method: str, expected: Mapping[str, Any]
) -> None:
    stride = EXECUTION_STEPS[method]
    if (
        rollout.get("method") != method
        or rollout.get("heldout_body") != expected["heldout_body"]
        or rollout.get("condition") != expected["condition"]
        or rollout.get("requested_seed") != expected["requested_seed"]
        or rollout.get("resolved_seed") != expected["requested_seed"]
        or rollout.get("execution_stride_actions") != stride
        or rollout.get("native_chunk_steps") != NATIVE_CHUNK_STEPS
        or rollout.get("candidate_count") != 1
        or rollout.get("candidate_index") != 0
        or rollout.get("action_execution_error") is not None
        or type(rollout.get("binary_success")) is not int
        or rollout.get("binary_success") not in (0, 1)
        or type(rollout.get("max_event_id")) is not int
        or not 0 <= rollout["max_event_id"] <= STAGE_DENOMINATOR
        or not _is_sha256(rollout.get("initial_candidate_commitment_sha256"))
        or not _is_sha256(rollout.get("initial_candidate0_chunk_sha256"))
        or not _is_sha256(rollout.get("initial_candidate0_first_token_sha256"))
    ):
        raise ActorDeploymentProtocolError(f"{method} rollout identity changed")
    expected_stage = (
        1.0
        if rollout["binary_success"]
        else rollout["max_event_id"] / float(STAGE_DENOMINATOR)
    )
    if abs(float(rollout.get("stage_progress", -1.0)) - expected_stage) > 1e-9:
        raise ActorDeploymentProtocolError(f"{method} stage progress changed")
    for name in (
        "initial_goal_distance_m",
        "terminal_goal_distance_m",
        "goal_progress_m",
    ):
        if not _finite_number(rollout.get(name)):
            raise ActorDeploymentProtocolError(f"{method} goal metric is invalid")
    if rollout["initial_goal_distance_m"] < 0 or rollout["terminal_goal_distance_m"] < 0:
        raise ActorDeploymentProtocolError(f"{method} goal distance is negative")
    if not math.isclose(
        float(rollout["goal_progress_m"]),
        float(rollout["initial_goal_distance_m"])
        - float(rollout["terminal_goal_distance_m"]),
        abs_tol=1e-8,
        rel_tol=0.0,
    ):
        raise ActorDeploymentProtocolError(f"{method} goal progress changed")
    decisions = rollout.get("decisions")
    if (
        not isinstance(decisions, list)
        or not decisions
        or rollout.get("policy_query_count") != len(decisions)
    ):
        raise ActorDeploymentProtocolError(f"{method} decisions changed")
    total_executed = 0
    for query_index, decision in enumerate(decisions):
        count = decision.get("executed_action_count")
        if (
            not isinstance(decision, Mapping)
            or decision.get("query_index") != query_index
            or decision.get("candidate_count") != 1
            or decision.get("candidate_index") != 0
            or decision.get("native_chunk_steps") != NATIVE_CHUNK_STEPS
            or decision.get("protocol_execute_steps") != stride
            or decision.get(
                "query_canonicalization_physical_steps_before_generation"
            )
            != QUERY_CANONICALIZATION_STEPS
            or type(count) is not int
            or not 0 <= count <= stride
            or decision.get("critic_scores") is not None
            or not _is_sha256(decision.get("candidate0_chunk_sha256"))
            or not _is_sha256(decision.get("candidate0_first_token_sha256"))
        ):
            raise ActorDeploymentProtocolError(f"{method} decision changed")
        continuity = decision.get("first_token_continuity")
        if (
            not isinstance(continuity, Mapping)
            or continuity.get("first_token_sha256")
            != decision["candidate0_first_token_sha256"]
            or continuity.get(
                "computed_in_canonical_two_arm_ee_effect14_frame"
            )
            is not True
            or continuity.get("raw_quaternion_component_rms_used") is not False
        ):
            raise ActorDeploymentProtocolError(f"{method} continuity changed")
        for key in (
            "live_state_to_first_token",
            "previous_executed_to_first_token",
            "previous_native_next_to_replanned_first_token",
        ):
            item = continuity.get(key)
            if key == "live_state_to_first_token" and not isinstance(item, Mapping):
                raise ActorDeploymentProtocolError(f"{method} lacks live continuity")
            if item is not None and (
                np.asarray(item.get("effect14"), dtype=np.float64).shape != (14,)
                or not _finite_number(item.get("effect14_rms"))
            ):
                raise ActorDeploymentProtocolError(f"{method} continuity metric invalid")
        if query_index == 0 and (
            continuity["previous_executed_to_first_token"] is not None
            or continuity["previous_native_next_to_replanned_first_token"] is not None
        ):
            raise ActorDeploymentProtocolError("query0 cannot have a prior-token metric")
        if query_index > 0 and continuity["previous_executed_to_first_token"] is None:
            raise ActorDeploymentProtocolError("replan boundary lacks prior executed token")
        if method == METHOD_EXECUTE50 and continuity[
            "previous_native_next_to_replanned_first_token"
        ] is not None:
            raise ActorDeploymentProtocolError("native50 cannot abandon a token50")
        total_executed += count
    if total_executed != rollout.get("executed_control_steps"):
        raise ActorDeploymentProtocolError(f"{method} executed step sum changed")
    if rollout["binary_success"] == 0 and total_executed != MAX_EPISODE_ACTION_STEPS:
        raise ActorDeploymentProtocolError("failed rollout did not exhaust formal horizon")
    if decisions[0]["candidate0_chunk_sha256"] != rollout[
        "initial_candidate0_chunk_sha256"
    ] or decisions[0]["candidate0_first_token_sha256"] != rollout[
        "initial_candidate0_first_token_sha256"
    ]:
        raise ActorDeploymentProtocolError("rollout query0 SHA binding changed")
    summary = rollout.get("first_token_continuity_summary")
    if summary != summarize_first_token_continuity(decisions):
        raise ActorDeploymentProtocolError(f"{method} continuity summary changed")
    terminal = rollout.get("terminal_native_success_components")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("official_terminal_check_success")
        is not rollout.get("terminal_check_success")
        or rollout.get("binary_success")
        != int(bool(rollout.get("latched_eval_success")) or bool(rollout.get("terminal_check_success")))
        or rollout.get("stop_reason")
        != ("success" if rollout["binary_success"] else "formal_action_limit")
    ):
        raise ActorDeploymentProtocolError(f"{method} native success audit changed")


def build_attempt(
    expected: Mapping[str, Any],
    *,
    binding_logical_sha256: str,
    binding_file_sha256: str,
) -> dict[str, Any]:
    base = {
        "format": ATTEMPT_FORMAT,
        "status": "started_once_fixed_pair_no_outcome_selection",
        **dict(expected),
        "binding_logical_sha256": binding_logical_sha256,
        "binding_file_sha256": binding_file_sha256,
        "attempt_number": 1,
    }
    return {**base, "attempt_sha256": canonical_sha256(base)}


def build_method_start(
    expected: Mapping[str, Any],
    *,
    method: str,
    method_ordinal: int,
    attempt_sha256: str,
    commitment_sha256: str,
    binding_logical_sha256: str,
    completed_prefix_result_sha256: Sequence[str],
) -> dict[str, Any]:
    base = {
        "format": METHOD_START_FORMAT,
        "status": "started_once_fixed_order_no_outcome_selection",
        **dict(expected),
        "method": method,
        "method_ordinal": method_ordinal,
        "attempt_sha256": attempt_sha256,
        "commitment_sha256": commitment_sha256,
        "binding_logical_sha256": binding_logical_sha256,
        "completed_prefix_result_sha256": list(completed_prefix_result_sha256),
        "automatic_noninformative_resume_limit": METHOD_RESUME_LIMIT,
        "result_or_failure_never_overwritten": True,
    }
    return {**base, "method_start_sha256": canonical_sha256(base)}


def build_method_result(
    expected: Mapping[str, Any],
    *,
    method: str,
    method_ordinal: int,
    rollout: Mapping[str, Any],
    method_start_sha256: str,
    attempt_sha256: str,
    commitment_sha256: str,
    binding_logical_sha256: str,
    binding_file_sha256: str,
    completed_prefix_result_sha256: Sequence[str],
) -> dict[str, Any]:
    validate_rollout(rollout, method=method, expected=expected)
    base = {
        "format": METHOD_RESULT_FORMAT,
        "status": "complete_create_once",
        **dict(expected),
        "method": method,
        "method_ordinal": method_ordinal,
        "method_start_sha256": method_start_sha256,
        "attempt_sha256": attempt_sha256,
        "commitment_sha256": commitment_sha256,
        "binding_logical_sha256": binding_logical_sha256,
        "binding_file_sha256": binding_file_sha256,
        "completed_prefix_result_sha256": list(completed_prefix_result_sha256),
        "rollout": dict(rollout),
    }
    return {**base, "method_result_sha256": canonical_sha256(base)}


def validate_method_result(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    method: str,
    method_ordinal: int,
    method_start_sha256: str,
    attempt_sha256: str,
    commitment_sha256: str,
    binding_logical_sha256: str,
    binding_file_sha256: str,
    completed_prefix_result_sha256: Sequence[str],
) -> dict[str, Any]:
    base = {key: item for key, item in value.items() if key != "method_result_sha256"}
    if (
        value.get("format") != METHOD_RESULT_FORMAT
        or value.get("status") != "complete_create_once"
        or any(value.get(key) != expected[key] for key in expected)
        or value.get("method") != method
        or value.get("method_ordinal") != method_ordinal
        or value.get("method_start_sha256") != method_start_sha256
        or value.get("attempt_sha256") != attempt_sha256
        or value.get("commitment_sha256") != commitment_sha256
        or value.get("binding_logical_sha256") != binding_logical_sha256
        or value.get("binding_file_sha256") != binding_file_sha256
        or value.get("completed_prefix_result_sha256")
        != list(completed_prefix_result_sha256)
        or value.get("method_result_sha256") != canonical_sha256(base)
        or not isinstance(value.get("rollout"), Mapping)
    ):
        raise ActorDeploymentProtocolError("stored method result binding changed")
    validate_rollout(value["rollout"], method=method, expected=expected)
    if value["rollout"].get("initial_candidate_commitment_sha256") != commitment_sha256:
        raise ActorDeploymentProtocolError("method result rollout commitment changed")
    return dict(value["rollout"])


def materialize_pair(
    expected: Mapping[str, Any],
    rollouts: Mapping[str, Mapping[str, Any]],
    *,
    attempt_sha256: str,
    commitment: Mapping[str, Any],
    binding_logical_sha256: str,
    binding_file_sha256: str,
    method_result_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if set(rollouts) != set(METHODS) or set(method_result_bindings) != set(METHODS):
        raise ActorDeploymentProtocolError("pair lacks both deployment methods")
    for method in METHODS:
        validate_rollout(rollouts[method], method=method, expected=expected)
    left = rollouts[METHOD_EXECUTE5]
    right = rollouts[METHOD_EXECUTE50]
    same_reset = (
        left["initial_reset_snapshot"]
        == right["initial_reset_snapshot"]
        == commitment["reset_snapshot"]
    )
    same_query = (
        left["initial_canonical_query_snapshot"]
        == right["initial_canonical_query_snapshot"]
        == commitment["canonical_query_snapshot"]
    )
    same_chunk = (
        left["initial_candidate0_chunk_sha256"]
        == right["initial_candidate0_chunk_sha256"]
        == commitment["candidate0_chunk_sha256"]
    )
    same_first = (
        left["initial_candidate0_first_token_sha256"]
        == right["initial_candidate0_first_token_sha256"]
        == commitment["candidate0_first_token_sha256"]
    )
    if not (same_reset and same_query and same_chunk and same_first):
        raise ActorDeploymentProtocolError("paired deployment methods did not share query0")
    base = {
        "format": PAIR_FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        **dict(expected),
        "attempt_sha256": attempt_sha256,
        "commitment_sha256": commitment["commitment_sha256"],
        "binding_logical_sha256": binding_logical_sha256,
        "binding_file_sha256": binding_file_sha256,
        "method_result_bindings": {
            key: dict(value) for key, value in method_result_bindings.items()
        },
        "same_complete_observable_reset_snapshot": same_reset,
        "same_canonical_query0_snapshot": same_query,
        "same_initial_candidate0_chunk": same_chunk,
        "same_initial_candidate0_first_token": same_first,
        "discordance": (
            "execute5_only"
            if left["binary_success"] > right["binary_success"]
            else "execute50_only"
            if right["binary_success"] > left["binary_success"]
            else "concordant_success"
            if left["binary_success"]
            else "concordant_failure"
        ),
        "rollouts": {key: dict(value) for key, value in rollouts.items()},
    }
    return {**base, "pair_sha256": canonical_sha256(base)}


def validate_pair(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    binding_logical_sha256: str,
    binding_file_sha256: str,
    expected_method_result_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    base = {key: item for key, item in value.items() if key != "pair_sha256"}
    if (
        value.get("format") != PAIR_FORMAT
        or value.get("benchmark") != BENCHMARK
        or value.get("task") != TASK
        or any(value.get(key) != expected[key] for key in expected)
        or value.get("binding_logical_sha256") != binding_logical_sha256
        or value.get("binding_file_sha256") != binding_file_sha256
        or not _is_sha256(value.get("attempt_sha256"))
        or not _is_sha256(value.get("commitment_sha256"))
        or value.get("same_complete_observable_reset_snapshot") is not True
        or value.get("same_canonical_query0_snapshot") is not True
        or value.get("same_initial_candidate0_chunk") is not True
        or value.get("same_initial_candidate0_first_token") is not True
        or value.get("pair_sha256") != canonical_sha256(base)
    ):
        raise ActorDeploymentProtocolError("stored pair changed")
    rollouts = value.get("rollouts")
    bindings = value.get("method_result_bindings")
    if (
        not isinstance(rollouts, Mapping)
        or set(rollouts) != set(METHODS)
        or not isinstance(bindings, Mapping)
        or set(bindings) != set(METHODS)
    ):
        raise ActorDeploymentProtocolError("pair method roster changed")
    if expected_method_result_bindings is not None and dict(bindings) != {
        key: dict(item) for key, item in expected_method_result_bindings.items()
    }:
        raise ActorDeploymentProtocolError("pair/result SHA binding changed")
    for method in METHODS:
        validate_rollout(rollouts[method], method=method, expected=expected)
        if rollouts[method]["initial_candidate_commitment_sha256"] != value[
            "commitment_sha256"
        ]:
            raise ActorDeploymentProtocolError("pair rollout commitment changed")
    if (
        rollouts[METHOD_EXECUTE5]["initial_reset_snapshot"]
        != rollouts[METHOD_EXECUTE50]["initial_reset_snapshot"]
        or rollouts[METHOD_EXECUTE5]["initial_candidate0_chunk_sha256"]
        != rollouts[METHOD_EXECUTE50]["initial_candidate0_chunk_sha256"]
        or rollouts[METHOD_EXECUTE5]["initial_candidate0_first_token_sha256"]
        != rollouts[METHOD_EXECUTE50]["initial_candidate0_first_token_sha256"]
    ):
        raise ActorDeploymentProtocolError("pair query0 evidence disagrees")


def outcome_row(pair: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "benchmark": BENCHMARK,
        "task": TASK,
        "heldout_body": pair["heldout_body"],
        "condition": pair["condition"],
        "requested_seed": pair["requested_seed"],
        "method_order": pair["method_order"],
        "pair_sha256": pair["pair_sha256"],
    }
    for method in METHODS:
        rollout = pair["rollouts"][method]
        continuity = rollout["first_token_continuity_summary"]
        result[f"{method}_binary_success"] = rollout["binary_success"]
        result[f"{method}_stage_progress"] = rollout["stage_progress"]
        result[f"{method}_terminal_goal_distance_m"] = rollout[
            "terminal_goal_distance_m"
        ]
        result[f"{method}_goal_progress_m"] = rollout["goal_progress_m"]
        result[f"{method}_live_first_token_effect14_rms_mean"] = continuity[
            "live_state_to_first_token"
        ]["mean_effect14_rms"]
        result[f"{method}_command_boundary_effect14_rms_mean"] = continuity[
            "previous_executed_to_first_token"
        ]["mean_effect14_rms"]
    return result


def validate_complete_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    schedule = evaluation_schedule()
    if len(rows) != len(schedule):
        raise ActorDeploymentProtocolError("outcomes do not cover all 200 pairs")
    for row, expected in zip(rows, schedule):
        if not isinstance(row, Mapping) or any(row.get(k) != expected[k] for k in expected):
            raise ActorDeploymentProtocolError("outcome schedule changed")
        if not _is_sha256(row.get("pair_sha256")):
            raise ActorDeploymentProtocolError("outcome row lacks pair SHA")
        for method in METHODS:
            if row.get(f"{method}_binary_success") not in (0, 1):
                raise ActorDeploymentProtocolError("outcome success is invalid")
            for suffix in (
                "stage_progress",
                "terminal_goal_distance_m",
                "goal_progress_m",
                "live_first_token_effect14_rms_mean",
            ):
                if not _finite_number(row.get(f"{method}_{suffix}")):
                    raise ActorDeploymentProtocolError("outcome metric is invalid")
            boundary = row.get(f"{method}_command_boundary_effect14_rms_mean")
            if boundary is not None and not _finite_number(boundary):
                raise ActorDeploymentProtocolError("boundary continuity is invalid")


def _exact_mcnemar_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


REPORT_METRICS = {
    "binary_success": "higher",
    "stage_progress": "higher",
    "terminal_goal_distance_m": "lower",
    "goal_progress_m": "higher",
    "live_first_token_effect14_rms_mean": "lower",
}


def _comparison_summary(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_seed: int
) -> dict[str, Any]:
    if not rows:
        raise ActorDeploymentProtocolError("comparison scope is empty")
    selected_cells = sorted(
        {(str(row["heldout_body"]), str(row["condition"])) for row in rows}
    )
    seeds = list(SEED_ROSTER)
    by_seed: dict[int, list[int]] = {seed: [] for seed in seeds}
    for index, row in enumerate(rows):
        seed = row.get("requested_seed")
        if seed not in by_seed:
            raise ActorDeploymentProtocolError("comparison uses a non-frozen seed")
        by_seed[int(seed)].append(index)
    rows_per_cluster = len(selected_cells)
    if rows_per_cluster == 0 or any(
        len(indices) != rows_per_cluster for indices in by_seed.values()
    ):
        raise ActorDeploymentProtocolError("seed clusters lack complete cell coverage")
    for seed, indices in by_seed.items():
        observed = {
            (str(rows[index]["heldout_body"]), str(rows[index]["condition"]))
            for index in indices
        }
        if observed != set(selected_cells):
            raise ActorDeploymentProtocolError(f"seed cluster {seed} changed cells")
    generator = np.random.default_rng(bootstrap_seed)
    sampled = generator.integers(
        0, SEED_COUNT, size=(BOOTSTRAP_REPLICATES, SEED_COUNT)
    )
    metric_reports: dict[str, Any] = {}
    for metric, favorable in REPORT_METRICS.items():
        left = np.asarray(
            [row[f"{METHOD_EXECUTE5}_{metric}"] for row in rows], dtype=np.float64
        )
        right = np.asarray(
            [row[f"{METHOD_EXECUTE50}_{metric}"] for row in rows], dtype=np.float64
        )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ActorDeploymentProtocolError(f"report metric {metric} is non-finite")
        left_cluster = np.asarray(
            [left[indices].mean() for indices in by_seed.values()], dtype=np.float64
        )
        right_cluster = np.asarray(
            [right[indices].mean() for indices in by_seed.values()], dtype=np.float64
        )
        left_boot = left_cluster[sampled].mean(axis=1)
        right_boot = right_cluster[sampled].mean(axis=1)
        delta_boot = right_boot - left_boot
        metric_reports[metric] = {
            "favorable_direction": favorable,
            "execute5_mean": float(left.mean()),
            "execute50_mean": float(right.mean()),
            "paired_delta_execute50_minus_execute5": float((right - left).mean()),
            "execute5_cluster_bootstrap_95_interval": np.quantile(
                left_boot, [0.025, 0.975]
            ).astype(float).tolist(),
            "execute50_cluster_bootstrap_95_interval": np.quantile(
                right_boot, [0.025, 0.975]
            ).astype(float).tolist(),
            "paired_delta_cluster_bootstrap_95_interval": np.quantile(
                delta_boot, [0.025, 0.975]
            ).astype(float).tolist(),
        }
    left_success = np.asarray(
        [row[f"{METHOD_EXECUTE5}_binary_success"] for row in rows]
    )
    right_success = np.asarray(
        [row[f"{METHOD_EXECUTE50}_binary_success"] for row in rows]
    )
    left_only = int(np.sum((left_success == 1) & (right_success == 0)))
    right_only = int(np.sum((left_success == 0) & (right_success == 1)))
    return {
        "pair_count": len(rows),
        "selected_body_condition_cells": [list(cell) for cell in selected_cells],
        "metrics": metric_reports,
        "execute5_only_successes": left_only,
        "execute50_only_successes": right_only,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_two_sided(
            left_only, right_only
        ),
        "mcnemar_role": "inferential" if len(selected_cells) == 1 else "descriptive_only",
        "confidence_interval_contract": {
            "method": "paired_requested_seed_cluster_percentile_bootstrap",
            "cluster_count": SEED_COUNT,
            "rows_per_cluster": rows_per_cluster,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": bootstrap_seed,
            "replacement": True,
        },
    }


def build_report(
    rows: Sequence[Mapping[str, Any]], *, outcome_document_sha256: str
) -> dict[str, Any]:
    validate_complete_rows(rows)
    normalized = [dict(row) for row in rows]
    overall = _comparison_summary(normalized, bootstrap_seed=BOOTSTRAP_SEED)
    ordered_selection_metrics = (
        "binary_success",
        "stage_progress",
        "goal_progress_m",
    )
    selected_metric = ordered_selection_metrics[-1]
    for metric in ordered_selection_metrics[:-1]:
        if abs(
            float(
                overall["metrics"][metric][
                    "paired_delta_execute50_minus_execute5"
                ]
            )
        ) > 0.0:
            selected_metric = metric
            break
    selected_delta = float(
        overall["metrics"][selected_metric][
            "paired_delta_execute50_minus_execute5"
        ]
    )
    hierarchical_selection = {
        "ordered_criteria": list(ordered_selection_metrics),
        "rule": (
            "compare paired success first; only if exactly tied compare paired "
            "stage progress; only if still tied compare paired goal progress"
        ),
        "selected_criterion": selected_metric,
        "selected_delta_execute50_minus_execute5": selected_delta,
        "preferred_protocol": (
            METHOD_EXECUTE50
            if selected_delta > 0.0
            else METHOD_EXECUTE5
            if selected_delta < 0.0
            else "tie"
        ),
        "mcnemar_and_bootstrap_intervals_are_uncertainty_not_sole_gate": True,
    }
    by_body = {
        body: _comparison_summary(
            [row for row in normalized if row["heldout_body"] == body],
            bootstrap_seed=BOOTSTRAP_SEED + 10 + index,
        )
        for index, body in enumerate(BODIES)
    }
    by_cell = {}
    for body_index, body in enumerate(BODIES):
        for condition_index, condition in enumerate(CONDITIONS):
            by_cell[f"{body}|{condition}"] = _comparison_summary(
                [
                    row
                    for row in normalized
                    if row["heldout_body"] == body
                    and row["condition"] == condition
                ],
                bootstrap_seed=(
                    BOOTSTRAP_SEED + 100 + body_index * 10 + condition_index
                ),
            )
    base = {
        "format": REPORT_FORMAT,
        "status": "complete_five_body_two_condition_paired_actor_deployment_report",
        "outcome_document_sha256": outcome_document_sha256,
        "evaluation_protocol_logical_sha256": evaluation_protocol()[
            "logical_sha256"
        ],
        "primary_estimand": "execute50_native_minus_execute5_replan",
        "primary_hierarchical_selection": hierarchical_selection,
        "overall": overall,
        "by_body": by_body,
        "by_body_condition": by_cell,
    }
    return {**base, "report_sha256": canonical_sha256(base)}


def build_outcome_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    binding_logical_sha256: str,
    binding_file_sha256: str,
) -> dict[str, Any]:
    validate_complete_rows(rows)
    normalized = [dict(row) for row in rows]
    base = {
        "format": OUTCOME_FORMAT,
        "status": "complete_200_pairs_400_rollouts",
        "binding_logical_sha256": binding_logical_sha256,
        "binding_file_sha256": binding_file_sha256,
        "rows": normalized,
        "rows_sha256": canonical_sha256(normalized),
        "ordered_pair_sha256s_sha256": canonical_sha256(
            [row["pair_sha256"] for row in normalized]
        ),
    }
    return {**base, "document_sha256": canonical_sha256(base)}


def _artifact_paths(output: Path, identity: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    starts = {}
    resumes = {}
    results = {}
    failures = {}
    for ordinal, method in enumerate(expected["method_order"]):
        stem = f"{identity}.{ordinal:02d}.{method}"
        starts[method] = output / "method_starts" / f"{stem}.json"
        resumes[method] = output / "method_resumes" / f"{stem}.json"
        results[method] = output / "method_results" / f"{stem}.json"
        failures[method] = output / "method_failures" / f"{stem}.json"
    return {
        "pair": output / "pairs" / f"{identity}.json",
        "attempt": output / "attempts" / f"{identity}.json",
        "commitment": output / "initial_commitments" / f"{identity}.json",
        "pair_failure": output / "pair_failures" / f"{identity}.json",
        "starts": starts,
        "resumes": resumes,
        "results": results,
        "failures": failures,
    }


def load_complete_pair_chain(
    *,
    output: Path,
    identity: str,
    expected: Mapping[str, Any],
    attempt: Mapping[str, Any],
    binding_logical_sha256: str,
    binding_file_sha256: str,
) -> dict[str, Any] | None:
    paths = _artifact_paths(output, identity, expected)
    pair_value, _ = read_create_once_json(paths["pair"], label="deployment pair")
    if pair_value is None:
        return None
    pair_failure, _ = read_create_once_json(
        paths["pair_failure"], label="pair failure"
    )
    if pair_failure is not None:
        raise ActorDeploymentProtocolError(
            "completed pair coexists with an immutable pair failure"
        )
    stored_attempt, _ = read_create_once_json(paths["attempt"], label="pair attempt")
    commitment, _ = read_create_once_json(
        paths["commitment"], label="initial commitment"
    )
    if stored_attempt != dict(attempt) or commitment is None:
        raise ActorDeploymentProtocolError("existing pair lacks attempt/commitment chain")
    validate_stored_initial_commitment(commitment, expected)
    if pair_value.get("commitment_sha256") != commitment["commitment_sha256"]:
        raise ActorDeploymentProtocolError("existing pair commitment chain changed")
    rollouts = {}
    bindings = {}
    prefix = []
    for ordinal, method in enumerate(expected["method_order"]):
        method_failure, _ = read_create_once_json(
            paths["failures"][method], label="method failure"
        )
        if method_failure is not None:
            raise ActorDeploymentProtocolError(
                "completed pair coexists with an immutable method failure"
            )
        start, _ = read_create_once_json(
            paths["starts"][method], label="method start"
        )
        result, _ = read_create_once_json(
            paths["results"][method], label="method result"
        )
        if start is None or result is None:
            raise ActorDeploymentProtocolError(
                "existing pair lacks complete method start/result chain"
            )
        expected_start = build_method_start(
            expected,
            method=method,
            method_ordinal=ordinal,
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            binding_logical_sha256=binding_logical_sha256,
            completed_prefix_result_sha256=prefix,
        )
        if start != expected_start:
            raise ActorDeploymentProtocolError("existing method start changed")
        rollout = validate_method_result(
            result,
            expected,
            method=method,
            method_ordinal=ordinal,
            method_start_sha256=start["method_start_sha256"],
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            binding_logical_sha256=binding_logical_sha256,
            binding_file_sha256=binding_file_sha256,
            completed_prefix_result_sha256=prefix,
        )
        result_file_sha = promote_create_once_json(
            paths["results"][method], result, label="method result"
        )
        rollouts[method] = rollout
        bindings[method] = {
            "logical_sha256": result["method_result_sha256"],
            "file_sha256": result_file_sha,
        }
        prefix.append(result["method_result_sha256"])
    computed = materialize_pair(
        expected,
        rollouts,
        attempt_sha256=attempt["attempt_sha256"],
        commitment=commitment,
        binding_logical_sha256=binding_logical_sha256,
        binding_file_sha256=binding_file_sha256,
        method_result_bindings=bindings,
    )
    if pair_value != computed:
        raise ActorDeploymentProtocolError("pair differs from immutable method results")
    validate_pair(
        pair_value,
        expected,
        binding_logical_sha256=binding_logical_sha256,
        binding_file_sha256=binding_file_sha256,
        expected_method_result_bindings=bindings,
    )
    promote_create_once_json(paths["pair"], pair_value, label="deployment pair")
    return dict(pair_value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--stable-seed-roster", type=Path, required=True)
    parser.add_argument("--stable-seed-roster-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", default=collector.DEFAULT_INSTRUCTION)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_ACTION_STEPS)
    return parser.parse_args(argv)


def _binding(
    args: argparse.Namespace,
    *,
    actor_checkpoint: Path,
    actor_tree: tuple[str, int, int],
    vlm_metadata: Path,
    vlm_tree: tuple[str, int, int],
    robotwin_root: Path,
    event_spec: Path,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    roster_document = require_stable_seed_roster()
    roster_materializer_path = Path(
        inspect.getsourcefile(stable_roster) or ""
    ).resolve()
    base = {
        "format": BINDING_FORMAT,
        "runner_format": FORMAT,
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "evaluation_protocol": evaluation_protocol(),
        "stable_seed_roster_binding": {
            "path": str(STABLE_SEED_ROSTER_PATH),
            "file_sha256": STABLE_SEED_ROSTER_FILE_SHA256,
            "logical_sha256": roster_document["logical_sha256"],
            "preregistration_file_sha256": roster_document[
                "preregistration_file_sha256"
            ],
            "preregistration_logical_sha256": roster_document[
                "preregistration_logical_sha256"
            ],
            "materializer_path": str(roster_materializer_path),
            "materializer_file_sha256": sha256_file(roster_materializer_path),
            "selection_uses_labels_or_outcomes": False,
            "actor_inference_calls_during_selection": 0,
        },
        "actor_checkpoint": str(actor_checkpoint),
        "actor_checkpoint_tree_sha256": actor_tree[0],
        "actor_checkpoint_file_count": actor_tree[1],
        "actor_checkpoint_size_bytes": actor_tree[2],
        "vlm_metadata_path": str(vlm_metadata),
        "vlm_metadata_tree_sha256": vlm_tree[0],
        "vlm_metadata_file_count": vlm_tree[1],
        "vlm_metadata_size_bytes": vlm_tree[2],
        "robotwin_root": str(robotwin_root),
        "runtime_binding": formal.implementation_binding(robotwin_root),
        "collector_implementation_sha256": sha256_file(
            Path(inspect.getsourcefile(collector) or "").resolve()
        ),
        "event_implementation_sha256": sha256_file(
            Path(inspect.getsourcefile(analytic_event) or "").resolve()
        ),
        "event_spec": str(event_spec),
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "instruction": args.instruction,
        "prompt_factor_status": "held_fixed_generic_not_part_of_primary_comparison",
        "max_episode_action_steps": MAX_EPISODE_ACTION_STEPS,
        "actor_dataset_fps": ACTOR_DATASET_FPS,
        "native_actor_action_frame": "two_arm_absolute_ee_xyz_quaternion_wxyz_gripper_16d",
        "execution_calls_task_take_action_action_type": "ee",
        "continuity_frame": "canonical_two_arm_relative_se3_axis_angle_gripper_effect14",
        "query_canonicalization": evaluation_protocol()[
            "query_canonicalization"
        ],
        "no_critic_loaded_or_called": True,
        "no_training": True,
        "official_expert_or_protected_payloads_opened": False,
    }
    return {**base, "logical_sha256": canonical_sha256(base)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        roster_value = stable_roster.load_stable_seed_roster_file(
            args.stable_seed_roster,
            args.stable_seed_roster_sha256,
        )
    except stable_roster.StableSeedRosterError as error:
        raise ActorDeploymentProtocolError(str(error)) from error
    configure_stable_seed_roster(
        roster_value,
        path=args.stable_seed_roster,
        file_sha256=args.stable_seed_roster_sha256,
    )
    if args.max_steps != MAX_EPISODE_ACTION_STEPS:
        raise ActorDeploymentProtocolError("formal deployment roster/horizon is frozen")
    if args.instruction != collector.DEFAULT_INSTRUCTION:
        raise ActorDeploymentProtocolError("primary comparison holds generic prompt fixed")
    required = (
        args.actor_checkpoint,
        args.vlm_metadata_path,
        args.robotwin_root,
        args.event_spec,
        args.stable_seed_roster,
    )
    if any(not path.expanduser().resolve().exists() for path in required):
        raise FileNotFoundError("one or more static deployment inputs are missing")

    robotwin_root = args.robotwin_root.expanduser().resolve()
    actor_checkpoint = args.actor_checkpoint.expanduser().resolve()
    vlm_metadata = args.vlm_metadata_path.expanduser().resolve()
    event_spec = args.event_spec.expanduser().resolve()
    if sha256_file(event_spec) != EVENT_SPEC_SHA256:
        raise ActorDeploymentProtocolError("event spec SHA changed")
    try:
        _spec, calibration = analytic_event.load_event_spec(event_spec)
    except analytic_event.AnalyticEventSpecError as error:
        raise ActorDeploymentProtocolError(str(error)) from error
    actor_tree = shared_head.sha256_tree(actor_checkpoint)
    vlm_tree = shared_head.sha256_tree(vlm_metadata)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "pairs",
        "attempts",
        "initial_commitments",
        "method_starts",
        "method_resumes",
        "method_results",
        "method_failures",
        "pair_failures",
    ):
        (output / directory).mkdir(exist_ok=True)

    binding = _binding(
        args,
        actor_checkpoint=actor_checkpoint,
        actor_tree=actor_tree,
        vlm_metadata=vlm_metadata,
        vlm_tree=vlm_tree,
        robotwin_root=robotwin_root,
        event_spec=event_spec,
        calibration=calibration,
    )
    binding_path = output / "immutable_deployment_binding.json"
    binding_file_sha = promote_create_once_json(
        binding_path, binding, label="immutable deployment binding"
    )
    binding_logical_sha = binding["logical_sha256"]

    schedule = evaluation_schedule()
    recovered: dict[str, dict[str, Any]] = {}
    missing = []
    for expected in schedule:
        identity = pair_id(
            str(expected["heldout_body"]),
            str(expected["condition"]),
            int(expected["requested_seed"]),
        )
        attempt = build_attempt(
            expected,
            binding_logical_sha256=binding_logical_sha,
            binding_file_sha256=binding_file_sha,
        )
        pair = load_complete_pair_chain(
            output=output,
            identity=identity,
            expected=expected,
            attempt=attempt,
            binding_logical_sha256=binding_logical_sha,
            binding_file_sha256=binding_file_sha,
        )
        if pair is None:
            missing.append((expected, identity, attempt))
        else:
            recovered[identity] = pair

    policy = preprocessor = postprocessor = task_class = device = None
    if missing:
        if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
            raise ActorDeploymentProtocolError(
                "missing deployment rollouts require the remote RTX 4090"
            )
        os.environ["ASSETS_PATH"] = str(robotwin_root)
        os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, str(robotwin_root))
        from envs import CONFIGS_PATH  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        module = __import__(f"envs.{TASK}", fromlist=[TASK])
        task_class = getattr(module, TASK)
        device = torch.device("cuda:0")
        config = PreTrainedConfig.from_pretrained(
            actor_checkpoint, local_files_only=True
        )
        config.device = str(device)
        config.vlm_model_name = str(vlm_metadata)
        config.load_vlm_weights = False
        if (
            config.action_feature is None
            or int(config.action_feature.shape[0]) != NATIVE_EE_DIM
            or config.input_features.get("observation.state") is None
            or int(config.input_features["observation.state"].shape[0])
            != NATIVE_EE_DIM
            or int(config.chunk_size) != NATIVE_CHUNK_STEPS
        ):
            raise ActorDeploymentProtocolError("actor is not native EE16 chunk50")
        policy = SmolVLAPolicy.from_pretrained(
            actor_checkpoint, config=config, local_files_only=True, strict=True
        ).eval().to(device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(actor_checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "tokenizer_processor": {"tokenizer_name": str(vlm_metadata)},
            },
        )

    started = time.time()
    for expected, identity, attempt in missing:
        assert task_class is not None and device is not None and policy is not None
        paths = _artifact_paths(output, identity, expected)
        prior_pair_failure, _ = read_create_once_json(
            paths["pair_failure"], label="pair failure"
        )
        if prior_pair_failure is not None:
            raise ActorDeploymentProtocolError("pair has an immutable prior failure")
        promote_create_once_json(paths["attempt"], attempt, label="pair attempt")
        body = str(expected["heldout_body"])
        condition = str(expected["condition"])
        seed = int(expected["requested_seed"])
        task_args = collector._load_task_args(robotwin_root, body, condition)
        task_args["step_lim"] = MAX_EPISODE_ACTION_STEPS
        try:
            commitment, _ = read_create_once_json(
                paths["commitment"], label="initial commitment"
            )
            if commitment is None:
                commitment = prepare_initial_commitment(
                    body=body,
                    condition=condition,
                    seed=seed,
                    task_class=task_class,
                    task_args=task_args,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    instruction=args.instruction,
                    device=device,
                )
            validate_stored_initial_commitment(commitment, expected)
            promote_create_once_json(
                paths["commitment"], commitment, label="initial commitment"
            )
            rollouts = {}
            result_bindings = {}
            prefix = []
            for ordinal, method in enumerate(expected["method_order"]):
                start_value = build_method_start(
                    expected,
                    method=method,
                    method_ordinal=ordinal,
                    attempt_sha256=attempt["attempt_sha256"],
                    commitment_sha256=commitment["commitment_sha256"],
                    binding_logical_sha256=binding_logical_sha,
                    completed_prefix_result_sha256=prefix,
                )
                existing_start, _ = read_create_once_json(
                    paths["starts"][method], label="method start"
                )
                start_created_now = existing_start is None
                if existing_start is not None and existing_start != start_value:
                    raise ActorDeploymentProtocolError("method start changed")
                promote_create_once_json(
                    paths["starts"][method], start_value, label="method start"
                )
                prior_failure, _ = read_create_once_json(
                    paths["failures"][method], label="method failure"
                )
                if prior_failure is not None:
                    raise ActorDeploymentProtocolError(
                        f"{method} has an immutable prior failure"
                    )
                result, _ = read_create_once_json(
                    paths["results"][method], label="method result"
                )
                if result is None:
                    if not start_created_now:
                        existing_resume, _ = read_create_once_json(
                            paths["resumes"][method], label="method resume"
                        )
                        if existing_resume is not None:
                            raise ActorDeploymentProtocolError(
                                f"{method} exceeded its one interruption resume"
                            )
                        resume_base = {
                            "format": METHOD_RESUME_FORMAT,
                            "status": "single_noninformative_interruption_resume",
                            **dict(expected),
                            "method": method,
                            "method_ordinal": ordinal,
                            "method_start_sha256": start_value[
                                "method_start_sha256"
                            ],
                            "attempt_sha256": attempt["attempt_sha256"],
                            "commitment_sha256": commitment["commitment_sha256"],
                            "resume_number": 1,
                            "resume_trigger": "missing_result_and_failure_after_process_or_host_interruption",
                            "outcome_not_observed_or_used_to_choose_resume": True,
                        }
                        resume = {
                            **resume_base,
                            "method_resume_sha256": canonical_sha256(resume_base),
                        }
                        promote_create_once_json(
                            paths["resumes"][method], resume, label="method resume"
                        )
                    try:
                        rollout = execute_rollout(
                            method=method,
                            body=body,
                            condition=condition,
                            seed=seed,
                            task_class=task_class,
                            task_args=task_args,
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            calibration=calibration,
                            initial_commitment=commitment,
                            instruction=args.instruction,
                            max_steps=MAX_EPISODE_ACTION_STEPS,
                            device=device,
                        )
                        result = build_method_result(
                            expected,
                            method=method,
                            method_ordinal=ordinal,
                            rollout=rollout,
                            method_start_sha256=start_value[
                                "method_start_sha256"
                            ],
                            attempt_sha256=attempt["attempt_sha256"],
                            commitment_sha256=commitment["commitment_sha256"],
                            binding_logical_sha256=binding_logical_sha,
                            binding_file_sha256=binding_file_sha,
                            completed_prefix_result_sha256=prefix,
                        )
                        promote_create_once_json(
                            paths["results"][method], result, label="method result"
                        )
                    except Exception as method_error:
                        failure_base = {
                            "format": METHOD_FAILURE_FORMAT,
                            "status": "failed_once_no_retry",
                            **dict(expected),
                            "method": method,
                            "method_ordinal": ordinal,
                            "method_start_sha256": start_value[
                                "method_start_sha256"
                            ],
                            "attempt_sha256": attempt["attempt_sha256"],
                            "commitment_sha256": commitment["commitment_sha256"],
                            "error_type": type(method_error).__name__,
                            "error_message": str(method_error),
                        }
                        failure = {
                            **failure_base,
                            "method_failure_sha256": canonical_sha256(failure_base),
                        }
                        promote_create_once_json(
                            paths["failures"][method], failure, label="method failure"
                        )
                        raise
                rollout = validate_method_result(
                    result,
                    expected,
                    method=method,
                    method_ordinal=ordinal,
                    method_start_sha256=start_value["method_start_sha256"],
                    attempt_sha256=attempt["attempt_sha256"],
                    commitment_sha256=commitment["commitment_sha256"],
                    binding_logical_sha256=binding_logical_sha,
                    binding_file_sha256=binding_file_sha,
                    completed_prefix_result_sha256=prefix,
                )
                result_file_sha = promote_create_once_json(
                    paths["results"][method], result, label="method result"
                )
                rollouts[method] = rollout
                result_bindings[method] = {
                    "logical_sha256": result["method_result_sha256"],
                    "file_sha256": result_file_sha,
                }
                prefix.append(result["method_result_sha256"])
            pair = materialize_pair(
                expected,
                rollouts,
                attempt_sha256=attempt["attempt_sha256"],
                commitment=commitment,
                binding_logical_sha256=binding_logical_sha,
                binding_file_sha256=binding_file_sha,
                method_result_bindings=result_bindings,
            )
            existing_pair, _ = read_create_once_json(
                paths["pair"], label="deployment pair"
            )
            if existing_pair is not None and existing_pair != pair:
                raise ActorDeploymentProtocolError(
                    "existing pair differs from completed method results"
                )
            validate_pair(
                pair,
                expected,
                binding_logical_sha256=binding_logical_sha,
                binding_file_sha256=binding_file_sha,
                expected_method_result_bindings=result_bindings,
            )
            promote_create_once_json(paths["pair"], pair, label="deployment pair")
            recovered[identity] = pair
        except Exception as error:
            failure_base = {
                "format": "etsf_robotwin2_actor_deployment_pair_failure_v1",
                "status": "failed_no_automatic_retry",
                "pair_id": identity,
                "attempt_sha256": attempt["attempt_sha256"],
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            failure = {
                **failure_base,
                "failure_sha256": canonical_sha256(failure_base),
            }
            promote_create_once_json(
                paths["pair_failure"], failure, label="pair failure"
            )
            raise
        completed = len(recovered)
        formal.atomic_json(
            output / "progress.json",
            {
                "format": FORMAT,
                "status": "running" if completed < len(schedule) else "rollouts_complete",
                "completed_pairs": completed,
                "completed_rollouts": completed * 2,
                "total_pairs": len(schedule),
                "last_pair": identity,
                "wall_seconds_this_process": time.time() - started,
                "binding_logical_sha256": binding_logical_sha,
            },
        )
        print(
            "ACTOR_DEPLOYMENT_PAIR_COMPLETE="
            + json.dumps(
                {
                    "completed": completed,
                    "total": len(schedule),
                    "body": body,
                    "condition": condition,
                    "seed": seed,
                    "discordance": pair["discordance"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rows = []
    for expected in schedule:
        identity = pair_id(
            str(expected["heldout_body"]),
            str(expected["condition"]),
            int(expected["requested_seed"]),
        )
        if identity not in recovered:
            raise ActorDeploymentProtocolError("complete schedule lost a pair")
        rows.append(outcome_row(recovered[identity]))
    outcome = build_outcome_document(
        rows,
        binding_logical_sha256=binding_logical_sha,
        binding_file_sha256=binding_file_sha,
    )
    outcome_path = output / "paired_outcomes.json"
    promote_create_once_json(outcome_path, outcome, label="paired outcomes")
    report = build_report(
        rows, outcome_document_sha256=outcome["document_sha256"]
    )
    report_path = output / "paired_report.json"
    promote_create_once_json(report_path, report, label="paired report")
    completion_base = {
        "format": COMPLETION_FORMAT,
        "status": "complete_200_pairs_400_rollouts_frozen",
        "binding_logical_sha256": binding_logical_sha,
        "binding_file_sha256": binding_file_sha,
        "outcome_document_sha256": outcome["document_sha256"],
        "outcome_file_sha256": sha256_file(outcome_path),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": sha256_file(report_path),
        "pair_count": len(rows),
        "rollout_count": len(rows) * 2,
    }
    completion = {
        **completion_base,
        "logical_sha256": canonical_sha256(completion_base),
    }
    promote_create_once_json(
        output / "run.complete.json", completion, label="completion receipt"
    )
    print(
        "ACTOR_EXECUTE5_VS_EXECUTE50_COMPLETE="
        + json.dumps(
            {
                "pairs": len(rows),
                "rollouts": len(rows) * 2,
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del policy, preprocessor, postprocessor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ActorDeploymentProtocolError",
    "BODIES",
    "CONDITIONS",
    "EXECUTION_STEPS",
    "METHODS",
    "METHOD_EXECUTE5",
    "METHOD_EXECUTE50",
    "NATIVE_CHUNK_STEPS",
    "SEED_BASE",
    "SEED_COUNT",
    "build_method_result",
    "build_method_start",
    "build_outcome_document",
    "build_report",
    "evaluation_protocol",
    "evaluation_schedule",
    "execute_rollout",
    "first_token_continuity",
    "load_complete_pair_chain",
    "materialize_pair",
    "method_order",
    "native_success_components",
    "promote_create_once_json",
    "read_create_once_json",
    "summarize_first_token_continuity",
    "validate_complete_rows",
    "validate_method_result",
    "validate_pair",
    "validate_rollout",
]
