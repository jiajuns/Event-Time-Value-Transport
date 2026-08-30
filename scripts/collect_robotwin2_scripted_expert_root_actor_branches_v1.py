#!/usr/bin/env python3
"""Collect an independent scripted-root/frozen-actor RoboTwin2 supplement.

For each pre-registered scene seed, the public ``move_can_pot.play_once``
expert advances the simulator only until the first physical samples satisfying
analytic events e12/e3 and the first non-terminal sample satisfying e4 have
been snapshotted.  The expert is then discarded.  Each snapshot starts a new
frozen-actor decision with four real flow-noise candidates and actor-only
continuation under a label-blind, seed-bound horizon.

This collector deliberately writes its own manifest and output tree.  It does
not modify or append to the formal on-policy branch collection.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import collect_robotwin2_five_body_ee_candidate_branches_v1 as base
import robotwin2_cross_body_canonical_adapter_v1 as canonical_adapter
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event


FORMAT = "etsf_robotwin2_scripted_expert_root_actor_branches_v2"
MANIFEST_FORMAT = "etsf_robotwin2_proper_world_utility_rank_supplement_manifest_v2"
TARGET_EVENTS = ("e12", "e3", "e4")
HORIZON_SCHEDULE = (10, 25, 50, 100, 200)
RESERVE_SEED_START = 2026081000
RESERVE_SEEDS_PER_SLOT = 16
RESERVE_SEED_STOP_EXCLUSIVE = (
    RESERVE_SEED_START
    + len(base.BODIES)
    * len(base.CONDITIONS)
    * len(HORIZON_SCHEDULE)
    * RESERVE_SEEDS_PER_SLOT
)
FORMAL_PRIMARY_SEED_START = 2026082000
ROOT_NOISE_QUERY_INDEX = {"e12": 1, "e3": 2, "e4": 3}
EXPECTED_DECISIONS_PER_BODY = (
    len(base.CONDITIONS) * len(TARGET_EVENTS) * len(HORIZON_SCHEDULE)
)
EXPECTED_BRANCHES_PER_BODY = EXPECTED_DECISIONS_PER_BODY * base.CANDIDATE_COUNT
EXPECTED_FIVE_BODY_DECISIONS = EXPECTED_DECISIONS_PER_BODY * len(base.BODIES)
EXPECTED_FIVE_BODY_BRANCHES = EXPECTED_BRANCHES_PER_BODY * len(base.BODIES)
SUPPLEMENT_PROPER_LOSS_WEIGHT = 0.25
SUPPLEMENT_RANK_LOSS_WEIGHT = 0.25
SUPPLEMENT_USAGE_CONTRACT = {
    "outer_lobo_source_train_only": True,
    "multitask_proper_loss": True,
    "robust_object_effect_proper_loss": True,
    "terminal_event_proper_loss": True,
    "terminal_goal_progress_proper_loss": True,
    "normalization_or_baseline_fit": False,
    "candidate_rank_or_utility_loss": True,
    "candidate_rank_or_utility_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
    "candidate_rank_updates": (
        "bounded_monotone_utility_only_from_detached_consequence_features"
    ),
    "semantic_comparative_loss": False,
    "source_validation_or_checkpoint_selection": False,
    "calibration": False,
    "heldout_payload_access": False,
}
EXPERT_ROOT_PROVENANCE_CONTRACT = {
    "root_prefix_policy": "robotwin_scripted_expert",
    "root_definition": "fresh_frozen_actor_branch_initialized_from_expert_state",
    "expert_prefix_claimed_actor_on_policy": False,
    "candidate_policy": "same_frozen_native_actor_as_primary_binding",
    "continuation_policy": "same_frozen_native_actor_as_primary_binding",
    "expert_terminal_outcome_used_as_branch_label": False,
    "fresh_branch_horizon_starts_at_root": True,
    "formal_actor_prefix_distribution_claimed": False,
}

ROOT_SELECTION_CONTRACT = {
    "controller": "public_RoboTwin_move_can_pot.play_once",
    "observation_granularity": "every_successful_sapien_scene_step",
    "targets": list(TARGET_EVENTS),
    "selection": "first_physical_sample_whose_frozen_analytic_event_equals_target",
    "one_root_per_target_per_scene_seed": True,
    "adjacent_same_event_frames_used_as_additional_roots": False,
    "e4_must_be_nonterminal_simulator_success": True,
    "root_selection_reads_actor_branch_outcomes": False,
    "triplet_acceptance": (
        "e12_e3_e4_all_exist_and_fresh_restore_canonicalize_before_any_actor_"
        "candidate_outcome"
    ),
    "missing_target_policy": (
        "record_reject_and_advance_same_body_condition_horizon_slot_ordered_"
        "reserve_before_any_actor_candidate_outcome"
    ),
    "reserve_selection_reads_actor_candidate_outcomes": False,
    "cross_body_common_success_seed_selection": False,
    "planner_after_root": "scripted_expert_ends_and_is_never_used_for_continuation",
}
HORIZON_CONTRACT = {
    "values": list(HORIZON_SCHEDULE),
    "slot_count_per_condition": len(HORIZON_SCHEDULE),
    "binding": (
        "body_condition_horizon_slot_has_pre_registered_ordered_reserve_"
        "seeds_before_any_rollout"
    ),
    "same_horizon_for_e12_e3_e4_of_one_seed": True,
    "new_actor_branch_take_action_count_at_root": 0,
    "remaining_action_budget_at_root_equals_bound_horizon": True,
    "expert_physics_steps_or_planner_frames_used_to_compute_horizon": False,
    "candidate_or_terminal_outcomes_used_to_choose_horizon": False,
    "actor_query_stride_actions": base.FORMAL_ACTION_EXEC_STEPS,
}
RESERVE_ROSTER_CONTRACT = {
    "scope": "body_local_condition_local_horizon_slot_local",
    "ordered_reserve_seeds_per_slot": RESERVE_SEEDS_PER_SLOT,
    "selection": "first_complete_canonicalizable_e12_e3_e4_triplet",
    "selection_occurs_before_actor_candidate_outcomes": True,
    "rejected_attempts_are_audited": True,
    "rejected_seed_candidate_outcomes_executed": False,
    "one_selected_seed_per_slot": True,
    "heldout_body_availability_changes_source_body_roster": False,
    "python_rng_seeded_from_requested_scene_seed_before_each_fresh_setup": True,
    "reserve_seed_start_inclusive": RESERVE_SEED_START,
    "reserve_seed_stop_exclusive": RESERVE_SEED_STOP_EXCLUSIVE,
    "formal_primary_seed_start": FORMAL_PRIMARY_SEED_START,
}
ROOT_PAIR_BUNDLE_FORMAT = "etsf_robotwin2_scripted_root_triplet_resume_bundle_v2"
ACTOR_BRANCH_CONTRACT = {
    "candidate_count": base.CANDIDATE_COUNT,
    "candidate_generator": (
        "collect_robotwin2_five_body_ee_candidate_branches_v1.generate_candidates"
    ),
    "fresh_scene_candidate_evaluator": (
        "collect_robotwin2_five_body_ee_candidate_branches_v1._evaluate_candidate"
    ),
    "snapshot_restore_contract": base.BRANCH_ROOT_SNAPSHOT_CONTRACT,
    "candidate_noise_contract": base.CANDIDATE_NOISE_CONTRACT,
    "terminal_supervision_contract": base.TERMINAL_SUPERVISION_CONTRACT,
    "expert_actions_after_root": 0,
    "continuation_controller": "same_frozen_actor_as_four_root_candidates",
}


class ScriptedRootCollectionError(base.BranchCollectionError):
    """The scripted root or independent actor branch violates its contract."""


class _AllRequestedRootsCaptured(RuntimeError):
    """Private control-flow sentinel used to stop the expert after all roots."""


def horizon_slot_key(condition: str, horizon_slot: int) -> str:
    if condition not in base.CONDITIONS or horizon_slot not in range(
        len(HORIZON_SCHEDULE)
    ):
        raise ScriptedRootCollectionError("invalid supplement horizon slot")
    return f"{condition}|horizon_slot={horizon_slot}"


def reserve_roster(body: str) -> list[dict[str, Any]]:
    """Return the immutable body-local ordered seed roster for all ten slots."""

    if body not in base.BODIES:
        raise ScriptedRootCollectionError(f"unknown supplement body {body!r}")
    body_index = base.BODIES.index(body)
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(base.CONDITIONS):
        for horizon_slot, horizon in enumerate(HORIZON_SCHEDULE):
            global_slot = (
                (body_index * len(base.CONDITIONS) + condition_index)
                * len(HORIZON_SCHEDULE)
                + horizon_slot
            )
            first = RESERVE_SEED_START + global_slot * RESERVE_SEEDS_PER_SLOT
            seeds = list(range(first, first + RESERVE_SEEDS_PER_SLOT))
            rows.append(
                {
                    "slot_key": horizon_slot_key(condition, horizon_slot),
                    "condition": condition,
                    "horizon_slot": horizon_slot,
                    "remaining_action_budget": int(horizon),
                    "ordered_requested_seeds": seeds,
                }
            )
    flattened = [
        seed for row in rows for seed in row["ordered_requested_seeds"]
    ]
    if (
        len(flattened)
        != len(base.CONDITIONS)
        * len(HORIZON_SCHEDULE)
        * RESERVE_SEEDS_PER_SLOT
        or len(set(flattened)) != len(flattened)
        or min(flattened) < RESERVE_SEED_START
        or max(flattened) >= RESERVE_SEED_STOP_EXCLUSIVE
        or max(flattened) >= FORMAL_PRIMARY_SEED_START
    ):
        raise ScriptedRootCollectionError("supplement reserve roster is invalid")
    return rows


def reserve_horizon_by_seed(body: str) -> dict[int, int]:
    return {
        int(seed): int(row["remaining_action_budget"])
        for row in reserve_roster(body)
        for seed in row["ordered_requested_seeds"]
    }


def sha256_path(path: Path) -> str:
    """Hash a checkpoint with the exact trainer actor-authority convention."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ScriptedRootCollectionError("actor checkpoint may not be a symlink")
    resolved = expanded.resolve()
    if resolved.is_file():
        return base.sha256_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if resolved.is_symlink():
        raise ScriptedRootCollectionError("actor checkpoint tree may not be a symlink")
    rows = []
    for value in sorted(resolved.rglob("*")):
        if value.is_symlink():
            raise ScriptedRootCollectionError(
                "actor checkpoint tree contains a symbolic link"
            )
        if value.is_dir():
            continue
        if not value.is_file():
            raise ScriptedRootCollectionError(
                "actor checkpoint tree contains a special file"
            )
        rows.append(
            {
                "path": value.relative_to(resolved).as_posix(),
                "size_bytes": value.stat().st_size,
                "sha256": base.sha256_file(value),
            }
        )
    if not rows:
        raise ScriptedRootCollectionError("actor checkpoint tree is empty")
    return base.canonical_sha256(rows)


def load_actor_authority(
    path: Path, *, body: str, actor_checkpoint_sha256: str
) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ScriptedRootCollectionError("actor authority must be a real JSON file")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScriptedRootCollectionError("actor authority must be a JSON object")
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    actors = value.get("actors")
    actor = actors.get(body) if isinstance(actors, Mapping) else None
    if (
        logical != base.canonical_sha256(unsigned)
        or value.get("format")
        != "etsf_robotwin2_frozen_native_actor_authority_v1"
        or value.get("task") != base.TASK
        or not isinstance(actor, Mapping)
        or actor.get("frozen") is not True
        or actor.get("optimizer_updates_allowed") is not False
        or actor.get("candidate_count") != base.CANDIDATE_COUNT
        or actor.get("checkpoint_sha256") != actor_checkpoint_sha256
    ):
        raise ScriptedRootCollectionError(
            "actor authority does not bind this body's frozen checkpoint"
        )
    return value, base.sha256_file(resolved)


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path.resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def normalize_snapshot_for_actor_branch(
    snapshot: Mapping[str, Any], horizon: int
) -> dict[str, Any]:
    """End expert planner semantics and start a fresh H-action actor branch."""

    if int(horizon) not in HORIZON_SCHEDULE:
        raise ScriptedRootCollectionError("actor horizon is outside the frozen grid")
    result = copy.deepcopy(dict(snapshot))
    fields = result.get("task_fields")
    if not isinstance(fields, dict):
        raise ScriptedRootCollectionError("expert snapshot lacks restorable task fields")
    required = {"take_action_cnt", "eval_success", "plan_success", "step_lim"}
    if not required.issubset(fields):
        raise ScriptedRootCollectionError(
            "expert snapshot lacks actor branch counter/planner fields"
        )
    fields["take_action_cnt"] = 0
    fields["eval_success"] = False
    fields["plan_success"] = True
    fields["step_lim"] = int(horizon)
    return result


class ScriptedRootObserver:
    """Observe each expert physics step and freeze first e12/e3/e4 roots."""

    def __init__(
        self,
        *,
        task: Any,
        names: Sequence[str],
        objects: Sequence[Any],
        calibration: Mapping[str, Any],
        horizon: int,
        snapshot_fn: Callable[[Any], Mapping[str, Any]] = base.capture_branch_root_snapshot,
    ) -> None:
        self.task = task
        self.names = list(names)
        self.objects = list(objects)
        self.calibration = dict(calibration)
        self.horizon = int(horizon)
        self.snapshot_fn = snapshot_fn
        self.trajectory = [base.read_poses(self.objects)]
        self.sim_times = [base._sim_time(task)]
        self.roots: dict[str, dict[str, Any]] = {}
        self.terminal_target_rejections = {name: 0 for name in TARGET_EVENTS}
        self.expert_move_index = -1
        self.expert_dense_segment_index = -1

    def record_physics_step(self) -> None:
        now = base._sim_time(self.task)
        if now <= self.sim_times[-1]:
            raise ScriptedRootCollectionError(
                "expert observer received a non-advancing simulator step"
            )
        self.trajectory.append(base.read_poses(self.objects))
        self.sim_times.append(now)
        simulator_success = bool(self.task.check_success())
        poses = np.stack(self.trajectory)
        times = np.asarray(self.sim_times, dtype=np.float64)
        try:
            _predicates, events = analytic_event.derive_predicates_and_events(
                poses,
                times,
                self.names,
                simulator_success,
                self.calibration,
            )
        except (analytic_event.AnalyticEventSpecError, ValueError) as error:
            raise ScriptedRootCollectionError(str(error)) from error
        observed_event = str(analytic_event.EVENT_NAMES[int(events[-1])])
        if observed_event == "eK":
            self.terminal_target_rejections["e4"] += 1
            return
        if observed_event not in TARGET_EVENTS or observed_event in self.roots:
            return
        target = observed_event
        if simulator_success:
            self.terminal_target_rejections[target] += 1
            return

        raw_snapshot = copy.deepcopy(dict(self.snapshot_fn(self.task)))
        normalized_snapshot = normalize_snapshot_for_actor_branch(
            raw_snapshot, self.horizon
        )
        self.roots[target] = {
            "target_event": target,
            "target_event_id": int(analytic_event.EVENT_TO_ID[target]),
            "object_names": list(self.names),
            "detector_trajectory": poses.copy(),
            "detector_sim_times": times.copy(),
            "detector_sim_step": int(self.task.scene.step_count),
            "expert_move_index": int(self.expert_move_index),
            "expert_dense_segment_index": int(self.expert_dense_segment_index),
            "raw_expert_snapshot_sha256": base.branch_root_snapshot_sha256(
                raw_snapshot
            ),
            "branch_root_snapshot": normalized_snapshot,
            "branch_root_snapshot_sha256": base.branch_root_snapshot_sha256(
                normalized_snapshot
            ),
            "branch_root_restorable_snapshot_sha256": (
                base.branch_root_restorable_snapshot_sha256(normalized_snapshot)
            ),
            "remaining_action_budget": self.horizon,
        }
        if set(self.roots) == set(TARGET_EVENTS):
            raise _AllRequestedRootsCaptured()


class ExpertObservationScene(base.SimulationClockScene):
    """Simulation-clock proxy that notifies the root observer after each step."""

    def __init__(
        self,
        prior: base.SimulationClockScene,
        callback: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(prior, base.SimulationClockScene):
            raise ScriptedRootCollectionError("expert task lacks the base scene clock")
        super().__init__(prior._scene)
        object.__setattr__(self, "step_count", int(prior.step_count))
        object.__setattr__(self, "_expert_step_callback", callback)

    def set_callback(self, callback: Callable[[], None]) -> None:
        object.__setattr__(self, "_expert_step_callback", callback)

    def step(self) -> Any:
        result = super().step()
        callback = self._expert_step_callback
        if callback is not None:
            callback()
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_expert_step_callback":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


def _close_task_safely(task: Any | None) -> None:
    if task is None or not hasattr(task, "close_env"):
        return
    with contextlib.suppress(Exception):
        task.close_env(clear_cache=False)


def _new_task_with_setup_cleanup(
    task_class: Any,
    task_args: Mapping[str, Any],
    seed: int,
    instruction: str,
) -> Any:
    """Construct the same fresh task as the base collector and close on setup error."""

    task = task_class()
    try:
        # RoboTwin seeds NumPy/Torch in Base_Task but leaves Python's RNG
        # commented out even though CuRobo graph fallback uses random.choice.
        # Reset it per requested scene seed so reject skipping on resume cannot
        # change later root availability.
        random.seed(int(seed))
        task.setup_demo(
            now_ep_num=seed,
            seed=seed,
            is_test=True,
            **dict(task_args),
        )
        if "step_lim" in task_args:
            task.step_lim = int(task_args["step_lim"])
        task.set_instruction(instruction=instruction)
        task.scene = base.SimulationClockScene(task.scene)
        return task
    except BaseException:
        _close_task_safely(task)
        raise


def _task_factory_with_setup_cleanup(task_class: Any) -> Callable[[], Any]:
    """Wrap the base candidate evaluator's task factory against setup leaks."""

    def factory() -> Any:
        task = task_class()
        original_setup = task.setup_demo

        def safe_setup(*args: Any, **kwargs: Any) -> Any:
            try:
                requested_seed = kwargs.get("seed")
                if isinstance(requested_seed, bool) or not isinstance(
                    requested_seed, int
                ):
                    raise ScriptedRootCollectionError(
                        "fresh candidate setup lacks an integer scene seed"
                    )
                random.seed(requested_seed)
                return original_setup(*args, **kwargs)
            except BaseException:
                _close_task_safely(task)
                raise

        task.setup_demo = safe_setup
        return task

    return factory


def capture_scripted_roots(
    *,
    task_class: Any,
    task_args: Mapping[str, Any],
    instruction: str,
    seed: int,
    horizon: int,
    required_pose_names: set[str],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one online expert scene and return label-blind first-event roots."""

    task: Any | None = None
    stopped_after_roots = False
    expert_returned = False
    observer: ScriptedRootObserver | None = None
    try:
        task = _new_task_with_setup_cleanup(
            task_class, task_args, seed, instruction
        )
        task.scene = ExpertObservationScene(task.scene)
        names, objects = base.discover_pose_objects(task, required_pose_names)
        observer = ScriptedRootObserver(
            task=task,
            names=names,
            objects=objects,
            calibration=calibration,
            horizon=horizon,
        )
        task.scene.set_callback(observer.record_physics_step)

        original_move = task.move
        original_take_dense_action = task.take_dense_action

        def tracked_move(*args: Any, **kwargs: Any) -> Any:
            observer.expert_move_index += 1
            return original_move(*args, **kwargs)

        def tracked_take_dense_action(*args: Any, **kwargs: Any) -> Any:
            observer.expert_dense_segment_index += 1
            return original_take_dense_action(*args, **kwargs)

        task.move = tracked_move
        task.take_dense_action = tracked_take_dense_action
        try:
            task.play_once()
            expert_returned = True
        except _AllRequestedRootsCaptured:
            stopped_after_roots = True
    finally:
        _close_task_safely(task)
    if observer is None:
        raise ScriptedRootCollectionError("expert observer was not initialized")
    return {
        "roots": observer.roots,
        "captured_targets": [name for name in TARGET_EVENTS if name in observer.roots],
        "missing_targets": [name for name in TARGET_EVENTS if name not in observer.roots],
        "terminal_target_rejections": dict(observer.terminal_target_rejections),
        "expert_play_once_returned": expert_returned,
        "expert_stopped_immediately_after_all_roots": stopped_after_roots,
        "expert_plan_success": bool(getattr(task, "plan_success", False)),
        "observed_physics_steps": len(observer.trajectory) - 1,
    }


def canonicalize_actor_root(
    *,
    captured: Mapping[str, Any],
    task_class: Any,
    task_args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    seed: int,
    required_pose_names: set[str],
    calibration: Mapping[str, Any],
    device: torch.device,
    generate_actor_candidates: bool = True,
) -> dict[str, Any]:
    """Restore one expert snapshot, canonicalize it, and generate four candidates."""

    target = str(captured["target_event"])
    horizon = int(captured["remaining_action_budget"])
    root_query = int(ROOT_NOISE_QUERY_INDEX[target])
    snapshot = captured["branch_root_snapshot"]
    args = dict(task_args)
    args["step_lim"] = horizon
    reference: Any | None = None
    try:
        reference = _new_task_with_setup_cleanup(
            task_class, args, seed, instruction
        )
        base.restore_branch_root_snapshot(reference, snapshot)
        restored = base.capture_branch_root_snapshot(reference)
        if not base.branch_root_restorable_snapshots_equal(snapshot, restored):
            differences = base.branch_root_snapshot_difference_summary(
                base.branch_root_restorable_snapshot(snapshot),
                base.branch_root_restorable_snapshot(restored),
            )
            raise ScriptedRootCollectionError(
                "fresh scene did not reproduce scripted event root: "
                + json.dumps(differences, sort_keys=True)
            )
        reference.scene.step()
        names, objects = base.discover_pose_objects(reference, required_pose_names)
        captured_names = captured.get("object_names")
        if not isinstance(captured_names, list) or list(names) != captured_names:
            raise ScriptedRootCollectionError(
                "tracked object registry changed after scripted root restore"
            )
        trajectory = np.concatenate(
            (
                np.asarray(captured["detector_trajectory"], dtype=np.float32),
                base.read_poses(objects)[None],
            ),
            axis=0,
        )
        sim_times = np.r_[
            np.asarray(captured["detector_sim_times"], dtype=np.float64),
            base._sim_time(reference),
        ]
        success = bool(reference.check_success())
        _predicates, events = base.derive_predicates_and_events(
            trajectory, sim_times, names, success, calibration
        )
        canonical_event = str(analytic_event.EVENT_NAMES[int(events[-1])])
        if canonical_event != target or success:
            raise ScriptedRootCollectionError(
                f"canonical actor root changed from {target} to {canonical_event}"
            )
        if (
            int(getattr(reference, "take_action_cnt", -1)) != 0
            or int(getattr(reference, "step_lim", -1)) != horizon
            or bool(getattr(reference, "eval_success", True))
        ):
            raise ScriptedRootCollectionError(
                "scripted snapshot did not start a fresh horizon-bound actor branch"
            )
        current = base.current_ee_action16(reference)
        canonical_snapshot_sha256 = base.branch_root_snapshot_sha256(
            base.capture_branch_root_snapshot(reference)
        )
        candidates = None
        if generate_actor_candidates:
            candidates = base.generate_candidates(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=reference,
                instruction=instruction,
                scene_seed=seed,
                query_index=root_query,
                candidate_count=base.CANDIDATE_COUNT,
                device=device,
            )
        root = dict(captured)
        root.update(
            {
                "object_names": list(names),
                "root_object_poses": trajectory[-1].copy(),
                "root_ee_action": current,
                "prefix_trajectory": trajectory,
                "prefix_sim_times": sim_times,
                "root_sim_steps": int(reference.scene.step_count),
                "sim_timestep_seconds": float(reference.scene.timestep_seconds),
                "remaining_action_budget": horizon,
                "canonical_root_snapshot_sha256": canonical_snapshot_sha256,
                "candidate_noise_query_index": root_query,
            }
        )
        if candidates is not None:
            root["candidates"] = candidates
        return root
    finally:
        _close_task_safely(reference)


def _manifest_logical_write(path: Path, manifest: dict[str, Any]) -> None:
    unsigned = dict(manifest)
    unsigned.pop("logical_sha256", None)
    manifest["logical_sha256"] = base.canonical_sha256(unsigned)
    base.atomic_json(path, manifest)


def _attempt_index(manifest: Mapping[str, Any], attempt_id: str) -> int | None:
    for index, value in enumerate(manifest.get("attempts", [])):
        if str(value.get("attempt_id")) == attempt_id:
            return index
    return None


def _upsert_attempt(manifest: dict[str, Any], attempt: Mapping[str, Any]) -> None:
    attempt_id = str(attempt["attempt_id"])
    index = _attempt_index(manifest, attempt_id)
    if index is None:
        manifest["attempts"].append(dict(attempt))
    else:
        manifest["attempts"][index] = dict(attempt)


def reserve_attempt_id(condition: str, horizon_slot: int, seed: int) -> str:
    return (
        f"{horizon_slot_key(condition, horizon_slot)}|requested_seed={int(seed)}"
    )


def reserve_group_id(
    condition: str, horizon_slot: int, seed: int, target: str
) -> str:
    if target not in TARGET_EVENTS:
        raise ScriptedRootCollectionError("unknown scripted root event")
    return (
        f"{horizon_slot_key(condition, horizon_slot)}|requested_seed={int(seed)}"
        f"|scripted_root={target}"
    )


def _contained_real_file(root: Path, relative_value: Any, label: str) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ScriptedRootCollectionError(f"{label} path is not contained")
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ScriptedRootCollectionError(f"{label} path escapes output") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ScriptedRootCollectionError(f"{label} file is missing or symbolic")
    return resolved


def verify_existing_group_files(output: Path, item: Mapping[str, Any]) -> None:
    """Hash both persisted files before an existing group may be skipped."""

    group = _contained_real_file(output, item.get("path"), "existing group")
    diagnostics = _contained_real_file(
        output, item.get("diagnostics_path"), "existing group diagnostics"
    )
    if base.sha256_file(group) != item.get("sha256"):
        raise ScriptedRootCollectionError("existing group SHA-256 mismatch")
    if base.sha256_file(diagnostics) != item.get("diagnostics_sha256"):
        raise ScriptedRootCollectionError(
            "existing group diagnostics SHA-256 mismatch"
        )


def _root_pair_bundle_relative(
    condition: str, horizon_slot: int, seed: int
) -> Path:
    return Path("root_pairs") / (
        f"{condition}_slot_{horizon_slot}_seed_{int(seed)}.pt"
    )


def _root_pair_fingerprint(capture: Mapping[str, Any]) -> dict[str, str]:
    roots = capture.get("roots")
    if not isinstance(roots, Mapping) or set(roots) != set(TARGET_EVENTS):
        raise ScriptedRootCollectionError(
            "root-triplet bundle is not a complete triplet"
        )
    result: dict[str, str] = {}
    for target in TARGET_EVENTS:
        root = roots[target]
        if not isinstance(root, Mapping):
            raise ScriptedRootCollectionError("root-triplet entry is invalid")
        snapshot = root.get("branch_root_snapshot")
        observed = base.branch_root_snapshot_sha256(snapshot)
        if (
            root.get("target_event") != target
            or root.get("branch_root_snapshot_sha256") != observed
            or not isinstance(root.get("object_names"), list)
        ):
            raise ScriptedRootCollectionError(
                "root-triplet snapshot contract changed"
            )
        result[target] = observed
    return result


def _validate_root_pair_bundle(
    value: Any,
    *,
    body: str,
    condition: str,
    horizon_slot: int,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScriptedRootCollectionError(
            "root-triplet resume bundle is not a mapping"
        )
    capture = value.get("capture")
    if (
        value.get("format") != ROOT_PAIR_BUNDLE_FORMAT
        or value.get("body") != body
        or value.get("condition") != condition
        or value.get("horizon_slot") != horizon_slot
        or value.get("remaining_action_budget") != horizon
        or value.get("requested_seed") != seed
        or not isinstance(capture, Mapping)
        or capture.get("captured_targets") != list(TARGET_EVENTS)
        or capture.get("missing_targets") != []
        or capture.get("expert_plan_success") is not True
        or value.get("actor_candidate_outcomes_executed_before_bundle") is not False
    ):
        raise ScriptedRootCollectionError(
            "root-triplet resume bundle identity changed"
        )
    for target in TARGET_EVENTS:
        root = capture["roots"].get(target)
        if (
            not isinstance(root, Mapping)
            or root.get("remaining_action_budget") != horizon
        ):
            raise ScriptedRootCollectionError("root-triplet horizon changed")
    _root_pair_fingerprint(capture)
    return dict(value)


def load_root_pair_bundle(
    path: Path,
    *,
    expected_sha256: str | None,
    body: str,
    condition: str,
    horizon_slot: int,
    horizon: int,
    seed: int,
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ScriptedRootCollectionError(
            "root-triplet resume bundle is missing/symbolic"
        )
    observed_sha = base.sha256_file(path)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise ScriptedRootCollectionError(
            "root-triplet resume bundle SHA-256 mismatch"
        )
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (
        _validate_root_pair_bundle(
            value,
            body=body,
            condition=condition,
            horizon_slot=horizon_slot,
            horizon=horizon,
            seed=seed,
        ),
        observed_sha,
    )


def persist_root_pair_bundle_create_once(
    path: Path,
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    """Atomically create a restorable triplet; never overwrite a prior root."""

    identity = {
        key: value[key]
        for key in (
            "body",
            "condition",
            "horizon_slot",
            "remaining_action_budget",
            "requested_seed",
        )
    }
    if path.is_symlink() or path.parent.is_symlink():
        raise ScriptedRootCollectionError(
            "root-triplet bundle path may not contain a symbolic link"
        )
    if path.exists():
        existing, observed_sha = load_root_pair_bundle(
            path,
            expected_sha256=None,
            body=str(identity["body"]),
            condition=str(identity["condition"]),
            horizon_slot=int(identity["horizon_slot"]),
            horizon=int(identity["remaining_action_budget"]),
            seed=int(identity["requested_seed"]),
        )
        if _root_pair_fingerprint(existing["capture"]) != _root_pair_fingerprint(
            value["capture"]
        ):
            raise ScriptedRootCollectionError(
                "existing root-triplet bundle differs for the same roster seed"
            )
        return existing, observed_sha, False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ScriptedRootCollectionError("stale root-triplet partial exists")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()
    observed_sha = base.sha256_file(path)
    loaded, _ = load_root_pair_bundle(
        path,
        expected_sha256=observed_sha,
        body=str(identity["body"]),
        condition=str(identity["condition"]),
        horizon_slot=int(identity["horizon_slot"]),
        horizon=int(identity["remaining_action_budget"]),
        seed=int(identity["requested_seed"]),
    )
    return loaded, observed_sha, True


def validate_completed_design_metadata(
    value: Mapping[str, Any], *, body: str
) -> set[tuple[str, int]]:
    """Validate ordered reserve resolution without opening any NPZ payload."""

    roster = reserve_roster(body)
    flattened = [
        seed for row in roster for seed in row["ordered_requested_seeds"]
    ]
    expected_horizons = reserve_horizon_by_seed(body)
    declared_horizons = value.get("pre_registered_horizon_by_seed")
    try:
        normalized_horizons = {
            int(seed): int(horizon)
            for seed, horizon in declared_horizons.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ScriptedRootCollectionError(
            f"{body} reserve horizon map is invalid"
        ) from error
    selected = value.get("selected_seed_by_slot")
    groups = value.get("groups")
    attempts = value.get("attempts")
    if (
        value.get("collection_status") != "complete"
        or value.get("reserve_roster_contract") != RESERVE_ROSTER_CONTRACT
        or value.get("reserve_roster") != roster
        or value.get("pre_registered_seeds") != flattened
        or normalized_horizons != expected_horizons
        or not isinstance(selected, Mapping)
        or set(selected) != {row["slot_key"] for row in roster}
        or not isinstance(groups, list)
        or len(groups) != EXPECTED_DECISIONS_PER_BODY
        or not isinstance(attempts, list)
    ):
        raise ScriptedRootCollectionError(
            f"{body} supplement reserve design is incomplete"
        )
    attempt_by_id: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ScriptedRootCollectionError("reserve attempt is not a mapping")
        attempt_id = attempt.get("attempt_id")
        if (
            not isinstance(attempt_id, str)
            or attempt_id in attempt_by_id
            or attempt.get("actor_candidate_outcomes_executed_before_selection")
            is not False
        ):
            raise ScriptedRootCollectionError("reserve attempt audit is invalid")
        attempt_by_id[attempt_id] = attempt

    observed_groups: set[tuple[str, int, int, str]] = set()
    selected_pairs: set[tuple[str, int]] = set()
    consumed_attempt_ids: set[str] = set()
    for row in roster:
        condition = str(row["condition"])
        slot = int(row["horizon_slot"])
        horizon = int(row["remaining_action_budget"])
        seeds = [int(seed) for seed in row["ordered_requested_seeds"]]
        selected_seed = selected.get(row["slot_key"])
        if isinstance(selected_seed, bool) or selected_seed not in seeds:
            raise ScriptedRootCollectionError("selected reserve seed is invalid")
        selected_seed = int(selected_seed)
        selected_index = seeds.index(selected_seed)
        for rejected_seed in seeds[:selected_index]:
            rejected = attempt_by_id.get(
                reserve_attempt_id(condition, slot, rejected_seed)
            )
            if (
                not isinstance(rejected, Mapping)
                or rejected.get("status") != "rejected_before_actor_outcomes"
                or rejected.get("condition") != condition
                or rejected.get("horizon_slot") != slot
                or rejected.get("requested_seed") != rejected_seed
                or rejected.get("pre_registered_horizon") != horizon
                or not isinstance(rejected.get("reject_reason"), str)
                or not rejected.get("reject_reason")
            ):
                raise ScriptedRootCollectionError(
                    "ordered reserve rejection history is incomplete"
                )
        selected_attempt = attempt_by_id.get(
            reserve_attempt_id(condition, slot, selected_seed)
        )
        if (
            not isinstance(selected_attempt, Mapping)
            or selected_attempt.get("status") != "complete"
            or selected_attempt.get("condition") != condition
            or selected_attempt.get("horizon_slot") != slot
            or selected_attempt.get("pre_registered_horizon") != horizon
            or selected_attempt.get("requested_seed") != selected_seed
            or selected_attempt.get("selected_before_actor_candidate_outcomes")
            is not True
            or not isinstance(selected_attempt.get("root_triplet_bundle_sha256"), str)
            or len(selected_attempt["root_triplet_bundle_sha256"]) != 64
        ):
            raise ScriptedRootCollectionError("selected reserve attempt is incomplete")
        allowed_attempt_ids = {
            reserve_attempt_id(condition, slot, seed)
            for seed in seeds[: selected_index + 1]
        }
        actual_attempt_ids = {
            attempt_id
            for attempt_id, attempt in attempt_by_id.items()
            if attempt.get("condition") == condition
            and attempt.get("horizon_slot") == slot
        }
        if actual_attempt_ids != allowed_attempt_ids:
            raise ScriptedRootCollectionError(
                "reserve attempts continued after the selected seed"
            )
        consumed_attempt_ids.update(allowed_attempt_ids)
        selected_pairs.add((condition, selected_seed))
        for target in TARGET_EVENTS:
            expected_group_id = reserve_group_id(
                condition, slot, selected_seed, target
            )
            matching = [
                group
                for group in groups
                if isinstance(group, Mapping)
                and group.get("group_id") == expected_group_id
            ]
            if len(matching) != 1:
                raise ScriptedRootCollectionError(
                    "selected reserve root group is missing or duplicated"
                )
            group = matching[0]
            if (
                group.get("condition") != condition
                or group.get("horizon_slot") != slot
                or group.get("requested_seed") != selected_seed
                or group.get("pre_registered_horizon") != horizon
                or group.get("scripted_root_event") != target
                or group.get("root_event_id")
                != int(analytic_event.EVENT_TO_ID[target])
            ):
                raise ScriptedRootCollectionError(
                    "selected reserve group contract changed"
                )
            observed_groups.add((condition, slot, selected_seed, target))
    if len(observed_groups) != EXPECTED_DECISIONS_PER_BODY:
        raise ScriptedRootCollectionError("reserve group design is incomplete")
    if consumed_attempt_ids != set(attempt_by_id):
        raise ScriptedRootCollectionError("reserve manifest has an unknown attempt")
    return selected_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", choices=base.BODIES, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions", nargs="+", choices=base.CONDITIONS, default=list(base.CONDITIONS)
    )
    parser.add_argument(
        "--action-exec-steps", type=int, default=base.FORMAL_ACTION_EXEC_STEPS
    )
    parser.add_argument("--instruction", default=base.DEFAULT_INSTRUCTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise ScriptedRootCollectionError(
            "real scripted-root collection requires remote RTX 4090 CUDA"
        )
    if args.action_exec_steps != base.FORMAL_ACTION_EXEC_STEPS:
        raise ScriptedRootCollectionError("actor query stride is fixed to five actions")
    if list(args.conditions) != list(base.CONDITIONS):
        raise ScriptedRootCollectionError(
            "the complete supplement requires clean and randomized in frozen order"
        )
    if args.instruction != base.DEFAULT_INSTRUCTION:
        raise ScriptedRootCollectionError("the supplement fixes the actor instruction")
    if len(set(args.conditions)) != len(args.conditions):
        raise ScriptedRootCollectionError("conditions must be unique")
    body_roster = reserve_roster(args.body)
    seed_horizons = reserve_horizon_by_seed(args.body)
    flattened_roster_seeds = [
        int(seed)
        for row in body_roster
        for seed in row["ordered_requested_seeds"]
    ]
    for path in (
        args.actor_checkpoint,
        args.actor_authority,
        args.vlm_metadata_path,
        args.robotwin_root,
        args.event_spec,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    random.seed(20260903)
    np.random.seed(20260903)
    torch.manual_seed(20260903)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root.resolve())
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.robotwin_root.resolve()))

    from envs import CONFIGS_PATH  # noqa: F401 - initializes RoboTwin paths
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    module = __import__(f"envs.{base.TASK}", fromlist=[base.TASK])
    task_class = getattr(module, base.TASK)
    device = torch.device("cuda:0")
    config = PreTrainedConfig.from_pretrained(
        args.actor_checkpoint, local_files_only=True
    )
    config.device = str(device)
    config.vlm_model_name = str(args.vlm_metadata_path.resolve())
    config.load_vlm_weights = False
    if config.action_feature is None or int(config.action_feature.shape[0]) != base.NATIVE_EE_DIM:
        raise ScriptedRootCollectionError("actor checkpoint must have 16-D EE actions")
    if config.input_features.get("observation.state") is None or int(
        config.input_features["observation.state"].shape[0]
    ) != base.NATIVE_EE_DIM:
        raise ScriptedRootCollectionError("actor checkpoint must have 16-D EE state")
    policy = SmolVLAPolicy.from_pretrained(
        args.actor_checkpoint, config=config, local_files_only=True, strict=True
    ).eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.actor_checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {
                "tokenizer_name": str(args.vlm_metadata_path)
            },
        },
    )

    try:
        _event_spec, calibration = analytic_event.load_event_spec(args.event_spec)
    except analytic_event.AnalyticEventSpecError as error:
        raise ScriptedRootCollectionError(str(error)) from error
    required_pose_names = set(analytic_event.REQUIRED_OBJECTS)
    collector_path = Path(__file__).resolve()
    base_collector_path = Path(inspect.getsourcefile(base) or "").resolve()
    event_source = Path(inspect.getsourcefile(analytic_event) or "").resolve()
    adapter_source = Path(inspect.getsourcefile(canonical_adapter) or "").resolve()
    public_task_source = args.robotwin_root / "envs" / f"{base.TASK}.py"
    collector_sha = base.sha256_file(collector_path)
    base_collector_sha = base.sha256_file(base_collector_path)
    actor_sha = sha256_path(args.actor_checkpoint)
    _actor_authority, actor_authority_sha = load_actor_authority(
        args.actor_authority,
        body=args.body,
        actor_checkpoint_sha256=actor_sha,
    )
    robotwin_commit = git_head(args.robotwin_root)

    output = args.output.expanduser().resolve()
    groups_dir = output / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    immutable = {
        "format": MANIFEST_FORMAT,
        "collector_format": FORMAT,
        "dataset_repo": base.DATASET_REPO,
        "dataset_revision": base.DATASET_REVISION,
        "task": base.TASK,
        "body": args.body,
        "conditions": list(args.conditions),
        "pre_registered_seeds": flattened_roster_seeds,
        "pre_registered_horizon_by_seed": {
            str(seed): horizon for seed, horizon in seed_horizons.items()
        },
        "reserve_roster_contract": RESERVE_ROSTER_CONTRACT,
        "reserve_roster": body_roster,
        "target_events": list(TARGET_EVENTS),
        "collector_file_sha256": collector_sha,
        "base_collector_file_sha256": base_collector_sha,
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "actor_checkpoint_tree_or_file_sha256": actor_sha,
        "actor_authority_sha256": actor_authority_sha,
        "vlm_metadata_path": str(args.vlm_metadata_path.resolve()),
        "instruction": base.DEFAULT_INSTRUCTION,
        "candidate_count": base.CANDIDATE_COUNT,
        "action_exec_steps": base.FORMAL_ACTION_EXEC_STEPS,
        "supplement_role": (
            "expert_event_root_proper_world_and_utility_rank_source_train_only"
        ),
        "root_policy": "robotwin_scripted_expert",
        "candidate_and_continuation_policy": (
            "same_frozen_native_actor_as_primary_binding"
        ),
        "proper_loss_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
        "rank_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
        "usage_contract": SUPPLEMENT_USAGE_CONTRACT,
        "expert_root_provenance_contract": EXPERT_ROOT_PROVENANCE_CONTRACT,
        "root_selection_contract": ROOT_SELECTION_CONTRACT,
        "horizon_contract": HORIZON_CONTRACT,
        "actor_branch_contract": ACTOR_BRANCH_CONTRACT,
        "candidate_noise_contract": base.CANDIDATE_NOISE_CONTRACT,
        "terminal_supervision_contract": base.TERMINAL_SUPERVISION_CONTRACT,
        "event_age_contract": base.EVENT_AGE_CONTRACT,
        "terminal_horizon_contract": base.TERMINAL_HORIZON_CONTRACT,
        "branch_root_snapshot_contract": base.BRANCH_ROOT_SNAPSHOT_CONTRACT,
        "object_effect_schema": base.OBJECT_EFFECT_SCHEMA,
        "branch_diagnostic_contract": base.BRANCH_DIAGNOSTIC_CONTRACT,
        "event_spec_sha256": analytic_event.EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": base.sha256_file(event_source),
        "canonical_adapter_implementation_sha256": base.sha256_file(adapter_source),
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "schema_adapter": {
            "kind": "analytic_label_free_canonical_v1",
            "trainable": False,
            "labels_or_outcomes_used_to_fit": False,
            "heldout_supervision_allowed": False,
            "state_dim": base.STATE_DIM,
            "action_dim": base.CANONICAL_ACTION_DIM,
            "state_schema": base.STATE_SCHEMA,
            "action_schema": base.ACTION_SCHEMA,
            "elapsed_time_unit": "seconds",
            "duration_unit": "seconds",
            "event_names": list(base.CANONICAL_EVENTS),
            "implementation_sha256": base.sha256_file(adapter_source),
        },
        "state27_relative_goal_contract": (
            "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
            "event_labels_and_online_state27_channels_0_2"
        ),
        "physical_time_contract": {
            "source": "counted_successful_sapien_scene_step_calls",
            "simulator_timestep_source": "scene.get_timestep",
            "policy_action_call_count_used_as_time": False,
            "wall_clock_used_as_time": False,
            "dt_semantics": "planned_first_candidate_chunk_seconds",
            "planned_action_steps": base.FORMAL_ACTION_EXEC_STEPS,
            "actor_control_hz": base.SOURCE_EVENT_SAMPLING_HZ,
            "planned_dt_seconds": (
                base.FORMAL_ACTION_EXEC_STEPS / base.SOURCE_EVENT_SAMPLING_HZ
            ),
            "duration_semantics": "simulator_elapsed_seconds_to_event_boundary",
            "zero_elapsed_duration_masked": True,
            "stationary_window_seconds": float(
                calibration["thresholds"]["stationary_window_seconds"]
            ),
            "stationary_speed_threshold_m_per_s": float(
                calibration["thresholds"]["stationary_speed_m_per_s"]
            ),
        },
        "candidate_action_contract": {
            "critic_observation_time": "before_candidate_execution",
            "planned_action_horizon": base.FORMAL_ACTION_EXEC_STEPS,
            "action_mask_source": "planned_first_chunk_not_executed_count",
            "executed_action_count_used_for_action_mask": False,
            "executed_action_count_used_for_sim_time_accounting_only": True,
            "planner_status_fail_is_a_valid_action_outcome": True,
            "python_execution_exception_invalidates_complete_decision": True,
        },
        "robotwin_public_task": {
            "repository_head": robotwin_commit,
            "event_spec_expected_commit": analytic_event.PUBLIC_TASK_COMMIT,
            "path": f"envs/{base.TASK}.py",
            "file_sha256": base.sha256_file(public_task_source),
            "play_once_source_sha256": hashlib.sha256(
                inspect.getsource(task_class.play_once).encode("utf-8")
            ).hexdigest(),
            "online_fresh_scene_seeds": True,
            "official_expert_zip_opened": False,
        },
        "expected_scale": {
            "decisions_this_body": EXPECTED_DECISIONS_PER_BODY,
            "branches_this_body": EXPECTED_BRANCHES_PER_BODY,
            "decisions_five_bodies": EXPECTED_FIVE_BODY_DECISIONS,
            "branches_five_bodies": EXPECTED_FIVE_BODY_BRANCHES,
        },
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = dict(manifest)
        logical = unsigned.pop("logical_sha256", None)
        if logical != base.canonical_sha256(unsigned) or any(
            manifest.get(key) != value for key, value in immutable.items()
        ):
            raise ScriptedRootCollectionError(
                "existing scripted-root manifest does not match this collection"
            )
    else:
        manifest = {
            **immutable,
            "collection_status": "in_progress",
            "selected_seed_by_slot": {},
            "attempts": [],
            "groups": [],
        }
        _manifest_logical_write(manifest_path, manifest)

    existing_groups: dict[str, Mapping[str, Any]] = {}
    for item in manifest["groups"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("group_id"), str):
            raise ScriptedRootCollectionError("existing group entry is invalid")
        group_id = str(item["group_id"])
        if group_id in existing_groups:
            raise ScriptedRootCollectionError("existing group id is duplicated")
        verify_existing_group_files(output, item)
        existing_groups[group_id] = item

    if manifest.get("collection_status") == "complete":
        validate_completed_design_metadata(manifest, body=args.body)
        print(
            "COLLECTION_COMPLETE="
            + json.dumps(
                {
                    "body": args.body,
                    "attempts": len(manifest["attempts"]),
                    "groups": len(manifest["groups"]),
                    "branches": len(manifest["groups"])
                    * base.CANDIDATE_COUNT,
                    "manifest": str(manifest_path),
                    "resumed_verified_complete": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    manifest["collection_status"] = "in_progress"
    selected_seed_by_slot = manifest.get("selected_seed_by_slot")
    if not isinstance(selected_seed_by_slot, dict):
        raise ScriptedRootCollectionError("selected seed map is invalid")

    for roster_row in body_roster:
        condition = str(roster_row["condition"])
        horizon_slot = int(roster_row["horizon_slot"])
        horizon = int(roster_row["remaining_action_budget"])
        slot_key = str(roster_row["slot_key"])
        roster_seeds = [
            int(seed) for seed in roster_row["ordered_requested_seeds"]
        ]
        expert_args = base._load_task_args(
            args.robotwin_root, args.body, condition
        )
        # Public expert dense physics is independent of the fresh actor budget.
        expert_args["step_lim"] = max(HORIZON_SCHEDULE)
        selected_seed = selected_seed_by_slot.get(slot_key)
        capture: dict[str, Any] | None = None
        attempt: dict[str, Any] | None = None
        started = time.time()

        candidate_seeds = [int(selected_seed)] if selected_seed is not None else roster_seeds
        for seed in candidate_seeds:
            if seed not in roster_seeds:
                raise ScriptedRootCollectionError(
                    "selected seed is outside its immutable reserve slot"
                )
            attempt_id = reserve_attempt_id(condition, horizon_slot, seed)
            previous_index = _attempt_index(manifest, attempt_id)
            previous = (
                manifest["attempts"][previous_index]
                if previous_index is not None
                else None
            )
            if selected_seed is None and isinstance(previous, Mapping):
                if previous.get("status") == "rejected_before_actor_outcomes":
                    continue
                if previous.get("status") not in {
                    "roots_observed_before_actor_outcomes",
                    "selected_triplet_before_actor_outcomes",
                    "complete",
                }:
                    raise ScriptedRootCollectionError(
                        "unfinished reserve attempt has an invalid status"
                    )

            bundle_relative = _root_pair_bundle_relative(
                condition, horizon_slot, seed
            )
            bundle_path = output / bundle_relative
            if bundle_path.exists():
                expected_bundle_sha = (
                    str(previous.get("root_triplet_bundle_sha256"))
                    if isinstance(previous, Mapping)
                    and previous.get("root_triplet_bundle_sha256") is not None
                    else None
                )
                bundle, bundle_sha = load_root_pair_bundle(
                    bundle_path,
                    expected_sha256=expected_bundle_sha,
                    body=args.body,
                    condition=condition,
                    horizon_slot=horizon_slot,
                    horizon=horizon,
                    seed=seed,
                )
                capture = dict(bundle["capture"])
            else:
                try:
                    capture = capture_scripted_roots(
                        task_class=task_class,
                        task_args=expert_args,
                        instruction=args.instruction,
                        seed=seed,
                        horizon=horizon,
                        required_pose_names=required_pose_names,
                        calibration=calibration,
                    )
                except BaseException as error:
                    reason = (
                        "unstable_reset"
                        if type(error).__name__ == "UnStableError"
                        else "fatal_expert_capture_exception"
                    )
                    rejected_or_fatal = {
                        "attempt_id": attempt_id,
                        "condition": condition,
                        "horizon_slot": horizon_slot,
                        "requested_seed": seed,
                        "pre_registered_horizon": horizon,
                        "status": (
                            "rejected_before_actor_outcomes"
                            if reason == "unstable_reset"
                            else "fatal_before_actor_outcomes"
                        ),
                        "reject_reason": reason,
                        "reject_details": {
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                        "selected_before_actor_candidate_outcomes": False,
                        "actor_candidate_outcomes_executed_before_selection": False,
                        "wall_seconds": time.time() - started,
                    }
                    if isinstance(previous, Mapping):
                        rejected_or_fatal["resume_history"] = [
                            *list(previous.get("resume_history", [])),
                            {
                                "prior_status": previous.get("status"),
                                "prior_frozen_roots": previous.get(
                                    "frozen_roots"
                                ),
                                "reason": (
                                    "process_ended_before_root_pair_selection"
                                ),
                            },
                        ]
                    _upsert_attempt(manifest, rejected_or_fatal)
                    _manifest_logical_write(manifest_path, manifest)
                    if reason == "unstable_reset":
                        continue
                    raise

                frozen_roots = {
                    target: {
                        "target_event_id": int(value["target_event_id"]),
                        "detector_sim_step": int(value["detector_sim_step"]),
                        "expert_move_index": int(value["expert_move_index"]),
                        "expert_dense_segment_index": int(
                            value["expert_dense_segment_index"]
                        ),
                        "raw_expert_snapshot_sha256": value[
                            "raw_expert_snapshot_sha256"
                        ],
                        "branch_root_snapshot_sha256": value[
                            "branch_root_snapshot_sha256"
                        ],
                        "branch_root_restorable_snapshot_sha256": value[
                            "branch_root_restorable_snapshot_sha256"
                        ],
                    }
                    for target, value in capture["roots"].items()
                }
                attempt = {
                    "attempt_id": attempt_id,
                    "condition": condition,
                    "horizon_slot": horizon_slot,
                    "requested_seed": seed,
                    "pre_registered_horizon": horizon,
                    "status": "roots_observed_before_actor_outcomes",
                    "captured_targets": capture["captured_targets"],
                    "missing_targets": capture["missing_targets"],
                    "terminal_target_rejections": capture[
                        "terminal_target_rejections"
                    ],
                    "expert_play_once_returned": capture[
                        "expert_play_once_returned"
                    ],
                    "expert_stopped_immediately_after_all_roots": capture[
                        "expert_stopped_immediately_after_all_roots"
                    ],
                    "expert_plan_success": capture["expert_plan_success"],
                    "observed_expert_physics_steps": int(
                        capture["observed_physics_steps"]
                    ),
                    "frozen_roots": frozen_roots,
                    "selected_before_actor_candidate_outcomes": False,
                    "actor_candidate_outcomes_executed_before_selection": False,
                }
                if isinstance(previous, Mapping):
                    resume_history = list(previous.get("resume_history", []))
                    resume_history.append(
                        {
                            "prior_status": previous.get("status"),
                            "prior_frozen_roots": previous.get("frozen_roots"),
                            "reason": "process_ended_before_root_pair_selection",
                        }
                    )
                    attempt["resume_history"] = resume_history
                _upsert_attempt(manifest, attempt)
                _manifest_logical_write(manifest_path, manifest)

                reject_reason = None
                if capture["expert_plan_success"] is not True:
                    reject_reason = "expert_plan_failed"
                elif capture["missing_targets"]:
                    reject_reason = "missing_" + "_and_".join(
                        str(target) for target in capture["missing_targets"]
                    )
                if reject_reason is not None:
                    attempt["status"] = "rejected_before_actor_outcomes"
                    attempt["reject_reason"] = reject_reason
                    attempt["wall_seconds"] = time.time() - started
                    _upsert_attempt(manifest, attempt)
                    _manifest_logical_write(manifest_path, manifest)
                    capture = None
                    continue

            if capture is None:
                raise ScriptedRootCollectionError("reserve capture disappeared")

            # Both roots must independently survive a fresh restore and one
            # canonical physics step before this seed is selected.  Candidate
            # generation and all candidate outcomes remain disabled here.
            try:
                for target in TARGET_EVENTS:
                    canonicalize_actor_root(
                        captured=capture["roots"][target],
                        task_class=task_class,
                        task_args=expert_args,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        instruction=args.instruction,
                        seed=seed,
                        required_pose_names=required_pose_names,
                        calibration=calibration,
                        device=device,
                        generate_actor_candidates=False,
                    )
            except (
                base.BranchCollectionError,
                analytic_event.AnalyticEventSpecError,
                ValueError,
            ) as error:
                if selected_seed is not None:
                    raise ScriptedRootCollectionError(
                        "previously selected root pair no longer canonicalizes"
                    ) from error
                attempt = dict(attempt or previous or {})
                attempt.update(
                    {
                        "attempt_id": attempt_id,
                        "condition": condition,
                        "horizon_slot": horizon_slot,
                        "requested_seed": seed,
                        "pre_registered_horizon": horizon,
                        "status": "rejected_before_actor_outcomes",
                        "reject_reason": "canonicalization_failed",
                        "reject_details": {
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                        "selected_before_actor_candidate_outcomes": False,
                        "actor_candidate_outcomes_executed_before_selection": False,
                        "wall_seconds": time.time() - started,
                    }
                )
                _upsert_attempt(manifest, attempt)
                _manifest_logical_write(manifest_path, manifest)
                capture = None
                continue

            bundle_value = {
                "format": ROOT_PAIR_BUNDLE_FORMAT,
                "body": args.body,
                "condition": condition,
                "horizon_slot": horizon_slot,
                "remaining_action_budget": horizon,
                "requested_seed": seed,
                "actor_candidate_outcomes_executed_before_bundle": False,
                "capture": capture,
            }
            bundle, bundle_sha, _created = persist_root_pair_bundle_create_once(
                bundle_path, bundle_value
            )
            capture = dict(bundle["capture"])
            attempt = dict(attempt or previous or {})
            attempt.update(
                {
                    "attempt_id": attempt_id,
                    "condition": condition,
                    "horizon_slot": horizon_slot,
                    "requested_seed": seed,
                    "pre_registered_horizon": horizon,
                    "status": "selected_triplet_before_actor_outcomes",
                    "selected_before_actor_candidate_outcomes": True,
                    "actor_candidate_outcomes_executed_before_selection": False,
                    "root_triplet_bundle_path": bundle_relative.as_posix(),
                    "root_triplet_bundle_sha256": bundle_sha,
                }
            )
            selected_seed_by_slot[slot_key] = seed
            selected_seed = seed
            _upsert_attempt(manifest, attempt)
            _manifest_logical_write(manifest_path, manifest)
            break

        if selected_seed is None or capture is None or attempt is None:
            manifest["collection_status"] = "reserve_exhausted"
            manifest["reserve_exhausted_slot"] = slot_key
            _manifest_logical_write(manifest_path, manifest)
            raise ScriptedRootCollectionError(
                f"ordered reserve exhausted before a complete root triplet: {slot_key}"
            )

        expected_group_ids = [
            reserve_group_id(condition, horizon_slot, selected_seed, target)
            for target in TARGET_EVENTS
        ]
        missing_targets = [
            target
            for target, group_id in zip(
                TARGET_EVENTS, expected_group_ids, strict=True
            )
            if group_id not in existing_groups
        ]
        generated_roots: dict[str, dict[str, Any]] = {}
        # Selection is already immutable.  Only now may the frozen actor be
        # queried; all missing roots are generated before the first outcome.
        for target in missing_targets:
            generated_roots[target] = canonicalize_actor_root(
                captured=capture["roots"][target],
                task_class=task_class,
                task_args=expert_args,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                instruction=args.instruction,
                seed=selected_seed,
                required_pose_names=required_pose_names,
                calibration=calibration,
                device=device,
                generate_actor_candidates=True,
            )

        for target in missing_targets:
            group_id = reserve_group_id(
                condition, horizon_slot, selected_seed, target
            )
            root = generated_roots[target]
            root_query = int(root["candidate_noise_query_index"])
            branch_args = dict(expert_args)
            branch_args["step_lim"] = horizon
            outcomes = [
                base._evaluate_candidate(
                    task_class=_task_factory_with_setup_cleanup(task_class),
                    args=branch_args,
                    root=root,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    instruction=args.instruction,
                    seed=selected_seed,
                    root_query=root_query,
                    candidate=candidate,
                    action_exec_steps=args.action_exec_steps,
                    max_steps=horizon,
                    required_pose_names=required_pose_names,
                    device=device,
                )
                for candidate in root["candidates"]
            ]
            arrays = base.materialize_group(
                root=root,
                outcomes=outcomes,
                calibration=calibration,
                action_exec_steps=args.action_exec_steps,
            )
            diagnostics = base.materialize_branch_diagnostics(
                root=root,
                outcomes=outcomes,
                action_exec_steps=args.action_exec_steps,
            )
            stem = (
                f"{condition}_slot_{horizon_slot}_h_{horizon}_"
                f"seed_{selected_seed}_scripted_root_{target}"
            )
            group_path = groups_dir / f"{stem}.npz"
            diagnostic_path = groups_dir / f"{stem}.diagnostics.npz"
            base.atomic_npz(group_path, arrays)
            base.atomic_npz(diagnostic_path, diagnostics)
            item = {
                "group_id": group_id,
                "collector_file_sha256": collector_sha,
                "base_collector_file_sha256": base_collector_sha,
                "condition": condition,
                "horizon_slot": horizon_slot,
                "requested_seed": selected_seed,
                "scripted_root_event": target,
                "scripted_root_event_id": int(root["target_event_id"]),
                "root_event_id": int(root["target_event_id"]),
                "pre_registered_horizon": horizon,
                "candidate_noise_query_index": root_query,
                "detector_sim_step": int(root["detector_sim_step"]),
                "raw_expert_snapshot_sha256": root[
                    "raw_expert_snapshot_sha256"
                ],
                "branch_root_snapshot_sha256": root[
                    "branch_root_snapshot_sha256"
                ],
                "branch_root_restorable_snapshot_sha256": root[
                    "branch_root_restorable_snapshot_sha256"
                ],
                "canonical_root_snapshot_sha256": root[
                    "canonical_root_snapshot_sha256"
                ],
                "path": f"groups/{group_path.name}",
                "sha256": base.sha256_file(group_path),
                "diagnostic_format": base.DIAGNOSTIC_FORMAT,
                "diagnostics_path": f"groups/{diagnostic_path.name}",
                "diagnostics_sha256": base.sha256_file(diagnostic_path),
                "wall_seconds": time.time() - started,
            }
            verify_existing_group_files(output, item)
            manifest["groups"].append(item)
            existing_groups[group_id] = item
            _manifest_logical_write(manifest_path, manifest)
            print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)

        for group_id in expected_group_ids:
            item = existing_groups.get(group_id)
            if not isinstance(item, Mapping):
                raise ScriptedRootCollectionError(
                    "selected triplet did not produce all event groups"
                )
            verify_existing_group_files(output, item)
        attempt["status"] = "complete"
        attempt["produced_group_ids"] = expected_group_ids
        attempt["actor_candidate_outcomes_executed_after_selection"] = True
        attempt["wall_seconds"] = time.time() - started
        _upsert_attempt(manifest, attempt)
        _manifest_logical_write(manifest_path, manifest)

    manifest.pop("reserve_exhausted_slot", None)
    manifest["collection_status"] = "complete"
    _manifest_logical_write(manifest_path, manifest)
    validate_completed_design_metadata(manifest, body=args.body)
    for item in manifest["groups"]:
        verify_existing_group_files(output, item)
    print(
        "COLLECTION_COMPLETE="
        + json.dumps(
            {
                "body": args.body,
                "attempts": len(manifest["attempts"]),
                "groups": len(manifest["groups"]),
                "branches": len(manifest["groups"]) * base.CANDIDATE_COUNT,
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
