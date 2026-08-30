#!/usr/bin/env python3
"""Collect real four-candidate RoboTwin2 branches for the five-body head.

The collector is intentionally operational rather than a data-contract tool:
it loads one frozen 16-D EE SmolVLA actor, creates four fixed-flow-noise action
candidates at several fixed query indices, executes each candidate after an
explicitly restored fresh-scene root snapshot plus one canonical physics step,
and writes one four-row canonical NPZ plus a non-trainable diagnostic sidecar
per decision.  Events, terminal
progress, SE(3) object effects and success are derived from the simulator
trajectory, never from the public expert archive.

Run this program only on the remote 4090 with the public RoboTwin simulator.
One invocation collects both clean and randomized groups for one embodiment;
the five embodiments are run sequentially because they share one GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

import robotwin2_cross_body_canonical_adapter_v1 as canonical_adapter
import robotwin2_move_can_pot_analytic_event_spec_v2 as analytic_event
import robotwin2_actor_execution_protocol_v1 as actor_execution


FORMAT = (
    "etsf_robotwin2_five_body_ee_candidate_branches_v3_actor_execution_protocol"
)
MANIFEST_FORMAT = (
    "etsf_robotwin2_canonical_transition_manifest_v3_actor_execution_protocol"
)
DIAGNOSTIC_FORMAT = "etsf_robotwin2_candidate_branch_diagnostics_v2_endpose_frame"
DATASET_REPO = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
TASK = "move_can_pot"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
CANDIDATE_COUNT = 4
NATIVE_EE_DIM = 16
CANONICAL_ACTION_DIM = 14
STATE_DIM = 27
OBJECT_DELTA_DIM = 6
SOURCE_EVENT_SAMPLING_HZ = 15.0
FORMAL_ACTION_EXEC_STEPS = 5
FORMAL_MAX_STEPS = 200
DEFAULT_INSTRUCTION = "Move the can to the side of the pot."
CANONICAL_EVENTS = ("e0", "e12", "e3", "e4", "eK")
EVENT_TO_ID = {name: index for index, name in enumerate(CANONICAL_EVENTS)}
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
ROOT_POSE_RESTORE_ATOL = 2.0 * float(np.finfo(np.float32).eps)
STATE_SCHEMA = canonical_adapter.STATE_SCHEMA
ACTION_SCHEMA = canonical_adapter.ACTION_SCHEMA
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
STATE_ACTION_FRAME_CONTRACT = {
    "format": "etsf_robotwin2_native_ee16_state_action_frame_v2",
    "training_state_source": "public_hdf5_endpose_left_right_endpose",
    "runtime_state_api": "task.get_arm_pose(left/right)",
    "runtime_state_pose_semantics": "robot.get_*_ee_pose(is_endpose=False)",
    "native_action_pose_semantics": (
        "same_absolute_world_ee_frame_as_training_endpose"
    ),
    "environment_call": "task.take_action(native_ee16, action_type=ee)",
    "pose_convention": "xyz_plus_quaternion_wxyz",
    "tcp_tool_axis_offset_m_excluded": 0.12,
    "state_and_action_same_frame": True,
}
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
def terminal_horizon_contract(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind terminal budgets to one validated actor execution protocol."""

    validated = actor_execution.validate_execution_protocol(protocol)
    return {
        "array": "remaining_action_budget",
        "semantics": "max_episode_action_steps_minus_pre_action_take_action_count",
        "available_before_candidate_execution": True,
        "same_value_for_all_candidates_at_one_root": True,
        "conditions_only_terminal_consequence_heads": True,
        "direct_rank_path": False,
        "formal_episode_action_steps": validated["max_steps"],
        "formal_actor_query_stride_actions": validated["stride"],
        "development_remaining_action_budgets": list(
            validated["primary_remaining_action_budgets"]
        ),
        "actor_execution_protocol_logical_sha256": validated["logical_sha256"],
    }


# Import-time compatibility value for code that only inspects the module.  A
# production invocation always replaces this with its explicitly file-bound
# protocol before creating or validating any artifact.
TERMINAL_HORIZON_CONTRACT = terminal_horizon_contract(
    actor_execution.execution_protocol(FORMAL_ACTION_EXEC_STEPS)
)
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
BRANCH_DIAGNOSTIC_CONTRACT = {
    "format": DIAGNOSTIC_FORMAT,
    "first_executed": "successful_or_physics_advancing_actions_in_planned_first_chunk",
    "branch_error": "all_false_execution_exception_invalidates_complete_decision",
    "candidate_action_pairwise_rms": (
        "symmetric_raw_canonical_effect_rms_over_planned_first_five_actions"
    ),
    "candidate_first_token_translation_norm_m": (
        "label_free_left_right_translation_norm_from_same_frame_root_state_to_"
        "candidate_token_zero"
    ),
    "candidate_later_token_translation_norm_median_m": (
        "label_free_left_right_median_translation_norm_between_subsequent_"
        "candidate_tokens"
    ),
}
DIAGNOSTIC_ACTION_PREFIX_STEPS = 5
BODY_EMBODIMENT = {
    "aloha-agilex": ["aloha-agilex"],
    "arx-x5": ["ARX-X5", "ARX-X5", 0.6],
    "franka": ["franka-panda", "franka-panda", 0.8],
    "piper": ["piper", "piper", 0.6],
    "ur5": ["ur5-wsg", "ur5-wsg", 0.8],
}


class BranchCollectionError(RuntimeError):
    """The actor, reset, replay or canonical output is invalid."""


class SimulationClockScene:
    """Transparent SAPIEN scene proxy with an exact physical-step clock.

    RoboTwin's EE ``take_action`` executes a variable number of internal
    ``scene.step`` calls.  Counting policy calls or wall time therefore cannot
    produce a cross-embodiment duration target.  The proxy observes the actual
    simulation steps without changing the scene implementation.
    """

    def __init__(self, scene: Any) -> None:
        object.__setattr__(self, "_scene", scene)
        timestep = float(scene.get_timestep())
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise BranchCollectionError("RoboTwin scene timestep must be positive")
        object.__setattr__(self, "timestep_seconds", timestep)
        object.__setattr__(self, "step_count", 0)

    def step(self) -> Any:
        result = self._scene.step()
        object.__setattr__(self, "step_count", int(self.step_count) + 1)
        return result

    @property
    def sim_seconds(self) -> float:
        return float(self.step_count) * float(self.timestep_seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_scene", "timestep_seconds", "step_count"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._scene, name, value)


TASK_SNAPSHOT_FIELDS = (
    "take_action_cnt",
    "eval_success",
    "plan_success",
    "left_cnt",
    "right_cnt",
    "stage_success_tag",
    "FRAME_IDX",
    "step_lim",
    "left_js",
    "right_js",
)
ROBOT_SNAPSHOT_FIELDS = (
    "left_gripper_val",
    "right_gripper_val",
    "left_js",
    "right_js",
)


def _pose7(value: Any) -> np.ndarray:
    return np.asarray([*value.p, *value.q], dtype=np.float64)


def _sapien_pose(value: Sequence[float]) -> Any:
    import sapien

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise BranchCollectionError("snapshot contains an invalid SAPIEN pose")
    return sapien.Pose(p=array[:3], q=array[3:])


def _component_key(component: Any) -> str:
    entity = component.entity
    same_type = [
        value
        for value in entity.components
        if type(value) is type(component)
        and str(getattr(value, "name", "")) == str(getattr(component, "name", ""))
    ]
    occurrence = same_type.index(component)
    return "|".join(
        (
            str(entity.per_scene_id),
            str(entity.name),
            f"{type(component).__module__}.{type(component).__name__}",
            str(getattr(component, "name", "")),
            str(occurrence),
        )
    )


def _articulation_key(articulation: Any) -> str:
    return canonical_sha256(
        {
            "name": str(articulation.name),
            "dof": int(articulation.dof),
            "links": [
                [int(link.entity.per_scene_id), str(link.entity.name), str(link.name)]
                for link in articulation.links
            ],
            "joints": [str(joint.name) for joint in articulation.joints],
        }
    )


def _scene_inventory(native_scene: Any) -> list[dict[str, Any]]:
    return [
        {
            "per_scene_id": int(entity.per_scene_id),
            "name": str(entity.name),
            "components": [
                [
                    f"{type(component).__module__}.{type(component).__name__}",
                    str(getattr(component, "name", "")),
                ]
                for component in entity.components
            ],
        }
        for entity in native_scene.entities
    ]


def _jsonable_snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable_snapshot(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise BranchCollectionError(
        f"snapshot field has unsupported type {type(value).__name__}"
    )


def branch_root_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    return canonical_sha256(_jsonable_snapshot(snapshot))


def branch_root_restorable_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return only independent state that the SAPIEN API can restore.

    ``qacc`` is the acceleration derived during the previous PhysX solve.
    SAPIEN 3 exposes a setter, but a fresh articulation immediately reports a
    recomputed/cache value instead of the supplied value.  It is retained in
    the full provenance snapshot, excluded only from the pre-step restore
    equality check, and included again in the strict post-canonicalization
    snapshot hash.
    """

    value = copy.deepcopy(_jsonable_snapshot(snapshot))
    for articulation in value.get("articulations", {}).values():
        articulation.pop("qacc", None)
    return value


def branch_root_restorable_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    return canonical_sha256(branch_root_restorable_snapshot(snapshot))


def branch_root_restorable_snapshots_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    """Check a fresh-scene restore while isolating SAPIEN float32 pose roundoff.

    ``set_root_pose`` normalizes an articulation quaternion in float32, so a
    subsequent getter can differ from the captured value by float32 roundoff.
    Only the seven articulation-root-pose components receive an absolute
    tolerance of two float32 machine epsilons.  After removing those values,
    the complete restorable snapshot must still have the same canonical hash;
    qpos, velocities, forces, task state, clocks and RNG therefore remain
    bit-exact.
    """

    left = branch_root_restorable_snapshot(expected)
    right = branch_root_restorable_snapshot(observed)
    left_articulations = left.get("articulations")
    right_articulations = right.get("articulations")
    if not isinstance(left_articulations, Mapping) or not isinstance(
        right_articulations, Mapping
    ):
        return False
    if set(left_articulations) != set(right_articulations):
        return False
    for key in left_articulations:
        left_value = left_articulations[key]
        right_value = right_articulations[key]
        if not isinstance(left_value, Mapping) or not isinstance(right_value, Mapping):
            return False
        try:
            left_pose = np.asarray(left_value["root_pose"], dtype=np.float64)
            right_pose = np.asarray(right_value["root_pose"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        if (
            left_pose.shape != (7,)
            or right_pose.shape != (7,)
            or not np.isfinite(left_pose).all()
            or not np.isfinite(right_pose).all()
            or not np.allclose(
                left_pose,
                right_pose,
                atol=ROOT_POSE_RESTORE_ATOL,
                rtol=0.0,
            )
        ):
            return False
        del left_value["root_pose"]
        del right_value["root_pose"]
    return canonical_sha256(left) == canonical_sha256(right)


def branch_root_snapshot_section_sha256(
    snapshot: Mapping[str, Any]
) -> dict[str, str]:
    return {
        str(name): canonical_sha256(_jsonable_snapshot(value))
        for name, value in snapshot.items()
    }


def branch_root_snapshot_difference_summary(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    sections: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return compact field-level diagnostics for an inexact restore.

    Snapshot hashes remain the fast equality check.  When one differs, this
    summary identifies the actual state variable instead of encouraging a
    blind tolerance increase.
    """

    selected = list(sections) if sections is not None else sorted(expected)
    differences: dict[str, Any] = {}

    def visit(path: str, left: Any, right: Any) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                differences[path] = {
                    "missing": sorted(set(left).difference(right)),
                    "unexpected": sorted(set(right).difference(left)),
                }
            for key in sorted(set(left).intersection(right), key=str):
                visit(f"{path}.{key}" if path else str(key), left[key], right[key])
            return
        if isinstance(left, (list, tuple, np.ndarray, torch.Tensor)) and isinstance(
            right, (list, tuple, np.ndarray, torch.Tensor)
        ):
            try:
                left_array = np.asarray(_jsonable_snapshot(left))
                right_array = np.asarray(_jsonable_snapshot(right))
            except (TypeError, ValueError):
                if _jsonable_snapshot(left) != _jsonable_snapshot(right):
                    differences[path] = "non_numeric_sequence_changed"
                return
            if left_array.shape != right_array.shape:
                differences[path] = {
                    "expected_shape": list(left_array.shape),
                    "observed_shape": list(right_array.shape),
                }
                return
            if left_array.dtype.kind in "biufc" and right_array.dtype.kind in "biufc":
                left_numeric = left_array.astype(np.float64, copy=False)
                right_numeric = right_array.astype(np.float64, copy=False)
                delta = np.abs(left_numeric - right_numeric)
                if not np.array_equal(left_numeric, right_numeric):
                    flat_index = int(np.argmax(delta)) if delta.size else 0
                    differences[path] = {
                        "max_abs": float(delta.reshape(-1)[flat_index]) if delta.size else 0.0,
                        "argmax": [
                            int(index)
                            for index in np.unravel_index(flat_index, delta.shape)
                        ]
                        if delta.size
                        else [],
                        "expected_at_argmax": float(left_numeric.reshape(-1)[flat_index]) if delta.size else None,
                        "observed_at_argmax": float(right_numeric.reshape(-1)[flat_index]) if delta.size else None,
                    }
                return
            if left_array.tolist() != right_array.tolist():
                differences[path] = "sequence_changed"
            return
        if _jsonable_snapshot(left) != _jsonable_snapshot(right):
            differences[path] = {
                "expected": _jsonable_snapshot(left),
                "observed": _jsonable_snapshot(right),
            }

    for section in selected:
        if section not in expected or section not in observed:
            differences[section] = "section_missing"
        else:
            visit(section, expected[section], observed[section])
    return differences


def capture_branch_root_snapshot(task: Any) -> dict[str, Any]:
    """Capture an explicit state that can be restored into a fresh scene."""

    scene = task.scene
    if not isinstance(scene, SimulationClockScene):
        raise BranchCollectionError("branch snapshot requires the simulation clock proxy")
    native_scene = scene._scene
    dynamic: dict[str, Any] = {}
    drives: dict[str, Any] = {}
    render: dict[str, Any] = {}
    static_entity_pose: dict[str, Any] = {}
    for entity in native_scene.entities:
        physical_types = {type(component).__name__ for component in entity.components}
        if not physical_types.intersection(
            {"PhysxRigidDynamicComponent", "PhysxArticulationLinkComponent"}
        ):
            static_entity_pose[str(entity.per_scene_id)] = _pose7(entity.pose)
        for component in entity.components:
            name = type(component).__name__
            key = _component_key(component)
            if name == "PhysxRigidDynamicComponent":
                kinematic = bool(component.get_kinematic())
                dynamic[key] = {
                    "pose": _pose7(component.get_pose()),
                    "linear_velocity": np.asarray(
                        component.get_linear_velocity(), dtype=np.float64
                    ),
                    "angular_velocity": np.asarray(
                        component.get_angular_velocity(), dtype=np.float64
                    ),
                    "kinematic": kinematic,
                    "kinematic_target": (
                        _pose7(component.get_kinematic_target())
                        if kinematic
                        else None
                    ),
                    "sleeping": bool(
                        component.is_sleeping()
                        if callable(component.is_sleeping)
                        else component.is_sleeping
                    ),
                }
            elif name == "PhysxDriveComponent":
                linear_target, angular_target = component.get_drive_velocity_target()
                drives[key] = {
                    "pose": _pose7(component.get_pose()),
                    "drive_target": _pose7(component.get_drive_target()),
                    "drive_linear_velocity_target": np.asarray(
                        linear_target, dtype=np.float64
                    ),
                    "drive_angular_velocity_target": np.asarray(
                        angular_target, dtype=np.float64
                    ),
                }
            elif "RenderCameraComponent" in name or "LightComponent" in name:
                row = {
                    "local_pose": _pose7(component.get_local_pose()),
                    "enabled": bool(
                        component.is_enabled()
                        if callable(component.is_enabled)
                        else component.is_enabled
                    ),
                }
                if "LightComponent" in name:
                    row["color"] = np.asarray(component.get_color(), dtype=np.float64)
                render[key] = row

    articulations: dict[str, Any] = {}
    for articulation in native_scene.get_all_articulations():
        root = articulation.root
        articulations[_articulation_key(articulation)] = {
            "root_pose": _pose7(articulation.get_root_pose()),
            "root_linear_velocity": np.asarray(
                articulation.get_root_linear_velocity(), dtype=np.float64
            ),
            "root_angular_velocity": np.asarray(
                articulation.get_root_angular_velocity(), dtype=np.float64
            ),
            "qpos": np.asarray(articulation.get_qpos(), dtype=np.float64),
            "qvel": np.asarray(articulation.get_qvel(), dtype=np.float64),
            "qacc": np.asarray(articulation.get_qacc(), dtype=np.float64),
            "qf": np.asarray(articulation.get_qf(), dtype=np.float64),
            "joint_names": [str(joint.name) for joint in articulation.active_joints],
            "joint_drive_target": np.asarray(
                [joint.get_drive_target() for joint in articulation.active_joints],
                dtype=np.float64,
            ),
            "joint_drive_velocity_target": np.asarray(
                [joint.get_drive_velocity_target() for joint in articulation.active_joints],
                dtype=np.float64,
            ),
            "sleeping": bool(root.sleeping),
        }
    task_fields = {
        name: copy.deepcopy(getattr(task, name))
        for name in TASK_SNAPSHOT_FIELDS
        if hasattr(task, name)
    }
    robot = getattr(task, "robot", None)
    robot_fields = {
        name: copy.deepcopy(getattr(robot, name))
        for name in ROBOT_SNAPSHOT_FIELDS
        if robot is not None and hasattr(robot, name)
    }
    import sapien

    return {
        "format": "etsf_sapien_explicit_fresh_scene_root_snapshot_v1",
        "sapien_version": str(sapien.__version__),
        "timestep_seconds": float(scene.timestep_seconds),
        "inventory": _scene_inventory(native_scene),
        "static_entity_pose": static_entity_pose,
        "dynamic": dynamic,
        "articulations": articulations,
        "drives": drives,
        "render": {
            "ambient_light": np.asarray(native_scene.ambient_light, dtype=np.float64),
            "components": render,
        },
        "simulation_step_count": int(scene.step_count),
        "task_fields": task_fields,
        "robot_fields": robot_fields,
        "python_rng": copy.deepcopy(random.getstate()),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_branch_root_snapshot(task: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore an explicit root into an independently constructed scene."""

    scene = task.scene
    if not isinstance(scene, SimulationClockScene):
        raise BranchCollectionError("branch restore requires the simulation clock proxy")
    import sapien

    native_scene = scene._scene
    if (
        snapshot.get("format")
        != "etsf_sapien_explicit_fresh_scene_root_snapshot_v1"
        or snapshot.get("sapien_version") != str(sapien.__version__)
        or not np.isclose(
            float(snapshot["timestep_seconds"]),
            float(scene.timestep_seconds),
            atol=1e-12,
            rtol=0.0,
        )
        or snapshot.get("inventory") != _scene_inventory(native_scene)
    ):
        raise BranchCollectionError("fresh scene inventory/timestep differs from root")

    entities = {str(entity.per_scene_id): entity for entity in native_scene.entities}
    components = {
        _component_key(component): component
        for entity in native_scene.entities
        for component in entity.components
    }
    articulations = {
        _articulation_key(articulation): articulation
        for articulation in native_scene.get_all_articulations()
    }
    if set(articulations) != set(snapshot["articulations"]):
        raise BranchCollectionError("fresh articulation inventory differs from root")

    for entity_id, pose in snapshot["static_entity_pose"].items():
        entities[entity_id].pose = _sapien_pose(pose)
    for key, value in snapshot["articulations"].items():
        articulation = articulations[key]
        if [str(joint.name) for joint in articulation.active_joints] != value[
            "joint_names"
        ]:
            raise BranchCollectionError("fresh articulation joint ordering changed")
        articulation.set_root_pose(_sapien_pose(value["root_pose"]))
        articulation.set_qpos(np.asarray(value["qpos"]))
        articulation.set_qvel(np.asarray(value["qvel"]))
        articulation.set_qf(np.asarray(value["qf"]))
        articulation.set_root_linear_velocity(
            np.asarray(value["root_linear_velocity"])
        )
        articulation.set_root_angular_velocity(
            np.asarray(value["root_angular_velocity"])
        )
        for index, joint in enumerate(articulation.active_joints):
            joint.set_drive_target(
                float(np.asarray(value["joint_drive_target"][index]).reshape(-1)[0])
            )
            joint.set_drive_velocity_target(
                float(
                    np.asarray(
                        value["joint_drive_velocity_target"][index]
                    ).reshape(-1)[0]
                )
            )
    for key, value in snapshot["dynamic"].items():
        component = components[key]
        component.set_kinematic(bool(value["kinematic"]))
        component.set_pose(_sapien_pose(value["pose"]))
        component.set_linear_velocity(np.asarray(value["linear_velocity"]))
        component.set_angular_velocity(np.asarray(value["angular_velocity"]))
        if value["kinematic_target"] is not None:
            component.set_kinematic_target(_sapien_pose(value["kinematic_target"]))
    for key, value in snapshot["drives"].items():
        component = components[key]
        component.set_pose(_sapien_pose(value["pose"]))
        component.set_drive_target(_sapien_pose(value["drive_target"]))
        component.set_drive_velocity_target(
            np.asarray(value["drive_linear_velocity_target"]),
            np.asarray(value["drive_angular_velocity_target"]),
        )
    native_scene.set_ambient_light(snapshot["render"]["ambient_light"])
    for key, value in snapshot["render"]["components"].items():
        component = components[key]
        component.set_local_pose(_sapien_pose(value["local_pose"]))
        if "color" in value:
            component.set_color(np.asarray(value["color"]))
        (component.enable if value["enabled"] else component.disable)()
    for name, value in snapshot["task_fields"].items():
        setattr(task, name, copy.deepcopy(value))
    robot = getattr(task, "robot", None)
    for name, value in snapshot["robot_fields"].items():
        setattr(robot, name, copy.deepcopy(value))
    for side in ("left_planner", "right_planner"):
        planner = getattr(robot, side, None)
        motion_gen = getattr(planner, "motion_gen", None)
        if motion_gen is not None:
            motion_gen.reset(reset_seed=True)
    random.setstate(copy.deepcopy(snapshot["python_rng"]))
    np.random.set_state(copy.deepcopy(snapshot["numpy_rng"]))
    torch.random.set_rng_state(snapshot["torch_cpu_rng"].clone())
    torch.cuda.set_rng_state_all(
        [state.clone() for state in snapshot["torch_cuda_rng"]]
    )
    object.__setattr__(scene, "step_count", int(snapshot["simulation_step_count"]))
    for key, value in snapshot["articulations"].items():
        root = articulations[key].root
        root.put_to_sleep() if value["sleeping"] else root.wake_up()
    for key, value in snapshot["dynamic"].items():
        component = components[key]
        component.put_to_sleep() if value["sleeping"] else component.wake_up()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def make_noise(
    config: Any,
    scene_seed: int,
    query_index: int,
    candidate_index: int,
    device: torch.device,
) -> torch.Tensor:
    if candidate_index < 0:
        raise BranchCollectionError("candidate index must be non-negative")
    # Preserve the legacy candidate-zero draw exactly, while pairing every
    # even draw with its antithetic negative.  Each candidate marginal remains
    # N(0,I), but four proposals cover both directions of two independent flow
    # draws instead of relying on four potentially clustered draws.
    base_candidate_index = int(candidate_index) - int(candidate_index) % 2
    sign = 1.0 if int(candidate_index) % 2 == 0 else -1.0
    seed = int(
        (
            20260903
            + int(scene_seed) * 1_000_003
            + int(query_index) * 10_007
            + base_candidate_index * 101
        )
        % (2**63 - 1)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (1, int(config.chunk_size), int(config.max_action_dim)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).mul_(sign)


def _pose_vector(value: Any) -> np.ndarray | None:
    try:
        pose = value.get_pose() if callable(getattr(value, "get_pose", None)) else value.pose
        result = np.r_[np.asarray(pose.p), np.asarray(pose.q)].astype(np.float32)
        return result if result.shape == (7,) and np.isfinite(result).all() else None
    except Exception:
        return None


def discover_pose_objects(
    task: Any, required_names: set[str]
) -> tuple[list[str], list[Any]]:
    excluded = {"robot", "scene", "viewer", "engine", "renderer", "cameras"}
    found = [
        (name, value)
        for name, value in vars(task).items()
        if not name.startswith("_")
        and name not in excluded
        and _pose_vector(value) is not None
    ]
    found.sort(key=lambda item: item[0])
    names = [name for name, _value in found]
    missing = sorted(required_names - set(names))
    if missing:
        raise BranchCollectionError(
            f"required task objects are missing: {missing}; discovered={names}"
        )
    return names, [value for _name, value in found]


def read_poses(objects: Sequence[Any]) -> np.ndarray:
    values = [_pose_vector(value) for value in objects]
    if any(value is None for value in values):
        raise BranchCollectionError("a tracked object stopped exposing a pose")
    # Keep the simulator's float64 quaternion until the official strict Euler
    # checks have been evaluated.  Casting to float32 can flip a strict 15°
    # boundary relative to RoboTwin ``check_success``.
    return np.stack(values).astype(np.float64)


def _goal_vector(
    poses: np.ndarray,
    names: Sequence[str],
    step: int,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return analytic_event.goal_vector(poses, names, step, calibration)
    except analytic_event.AnalyticEventSpecError as error:
        raise BranchCollectionError(str(error)) from error


def derive_predicates_and_events(
    poses: np.ndarray,
    sim_times: np.ndarray,
    names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
    success_height_reference_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return analytic_event.derive_predicates_and_events(
            poses,
            sim_times,
            names,
            success,
            calibration,
            success_height_reference_z,
        )
    except (analytic_event.AnalyticEventSpecError, ValueError) as error:
        raise BranchCollectionError(str(error)) from error


def event_age_seconds(
    events: np.ndarray, sim_times: np.ndarray, step: int | None = None
) -> float:
    """Return physical time elapsed since entry into the current event."""

    event_values = np.asarray(events, dtype=np.int64).reshape(-1)
    time_values = np.asarray(sim_times, dtype=np.float64).reshape(-1)
    if (
        len(event_values) == 0
        or event_values.shape != time_values.shape
        or not np.isfinite(time_values).all()
        or np.any(np.diff(time_values) < -1e-12)
    ):
        raise BranchCollectionError("event age requires aligned monotone physical time")
    selected = len(event_values) - 1 if step is None else int(step)
    if selected < 0 or selected >= len(event_values):
        raise BranchCollectionError("event age step is outside the observed trajectory")
    current = int(event_values[selected])
    entry = selected
    while entry > 0 and int(event_values[entry - 1]) == current:
        entry -= 1
    age = float(time_values[selected] - time_values[entry])
    if not np.isfinite(age) or age < -1e-9:
        raise BranchCollectionError("derived event age is invalid")
    return max(age, 0.0)


def _image_chw(value: Any) -> torch.Tensor:
    image = torch.as_tensor(np.asarray(value))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise BranchCollectionError(f"camera must be HWC RGB, got {tuple(image.shape)}")
    return image.permute(2, 0, 1).contiguous().float().div(255.0)


def current_ee_action16(task: Any) -> np.ndarray:
    """Read the actor state in the exact end-pose frame used by its HDF5 data."""

    get_arm_pose = getattr(task, "get_arm_pose", None)
    if not callable(get_arm_pose):
        raise BranchCollectionError(
            "RoboTwin task lacks get_arm_pose(left/right) required by the end-pose frame"
        )
    left_pose = np.asarray(get_arm_pose("left"), dtype=np.float32)
    right_pose = np.asarray(get_arm_pose("right"), dtype=np.float32)
    if left_pose.shape != (7,) or right_pose.shape != (7,):
        raise BranchCollectionError(
            "RoboTwin task.get_arm_pose(left/right) must return xyz+quaternion_wxyz"
        )
    value = np.concatenate(
        (
            left_pose,
            [float(task.robot.get_left_gripper_val())],
            right_pose,
            [float(task.robot.get_right_gripper_val())],
        )
    ).astype(np.float32)
    if value.shape != (NATIVE_EE_DIM,) or not np.isfinite(value).all():
        raise BranchCollectionError("RoboTwin current EE state is not finite 16-D")
    return value


def _camera_sources(observation: Mapping[str, Any]) -> list[Any]:
    vision = observation.get("observation", {})
    keys = ("head_camera", "left_camera", "right_camera")
    result = []
    for key in keys:
        camera = vision.get(key)
        if not isinstance(camera, Mapping) or "rgb" not in camera:
            raise BranchCollectionError(f"RoboTwin observation lacks {key} RGB")
        result.append(camera["rgb"])
    return result


def raw_policy_input(
    task: Any, image_keys: Sequence[str], instruction: str
) -> dict[str, Any]:
    observation = task.get_obs()
    images = _camera_sources(observation)
    result: dict[str, Any] = {
        "observation.state": torch.from_numpy(current_ee_action16(task)),
        "task": instruction,
    }
    named = {
        "observation.images.cam_high": images[0],
        "observation.images.camera1": images[0],
        "observation.images.cam_left_wrist": images[1],
        "observation.images.camera2": images[1],
        "observation.images.cam_right_wrist": images[2],
        "observation.images.camera3": images[2],
    }
    for index, key in enumerate(image_keys):
        result[key] = _image_chw(named.get(key, images[min(index, 2)]))
    return result


def normalize_ee_chunk(value: Any) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != NATIVE_EE_DIM:
        raise BranchCollectionError(
            f"actor must return [H,{NATIVE_EE_DIM}], got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise BranchCollectionError("actor EE candidate contains non-finite values")
    for start in (0, 8):
        quaternion = actions[:, start + 3 : start + 7]
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norms < 1e-6):
            raise BranchCollectionError("actor EE candidate has a zero quaternion")
        actions[:, start + 3 : start + 7] = quaternion / norms
        actions[:, start + 7] = np.clip(actions[:, start + 7], 0.0, 1.0)
    return actions


def generate_candidates(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: Any,
    instruction: str,
    scene_seed: int,
    query_index: int,
    candidate_count: int,
    device: torch.device,
) -> np.ndarray:
    raw = raw_policy_input(task, list(policy.config.image_features), instruction)
    processed = preprocessor(raw)
    candidates = []
    for candidate_index in range(candidate_count):
        policy.reset()
        noise = make_noise(
            policy.config, scene_seed, query_index, candidate_index, device
        )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(dict(processed), noise=noise)
        actions = postprocessor(normalized)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        candidates.append(normalize_ee_chunk(actions))
    result = np.stack(candidates)
    if candidate_count > 1 and not np.any(result[1:] != result[0]):
        raise BranchCollectionError("four flow-noise candidates are identical")
    return result


def canonical_action_chunk(current: np.ndarray, actions: np.ndarray) -> np.ndarray:
    left_pose = np.vstack((current[None, 0:7], actions[:, 0:7]))
    left_gripper = np.r_[current[7], actions[:, 7]]
    right_pose = np.vstack((current[None, 8:15], actions[:, 8:15]))
    right_gripper = np.r_[current[15], actions[:, 15]]
    effect = canonical_adapter.task_space_action_effect14(
        left_pose, left_gripper, right_pose, right_gripper
    )
    if effect.shape != (len(actions), CANONICAL_ACTION_DIM):
        raise BranchCollectionError("canonical action adapter changed shape")
    return effect


def _load_task_args(robotwin_root: Path, body: str, condition: str) -> dict[str, Any]:
    config_root = robotwin_root / "env_cfg" / "task_config"
    with (config_root / f"demo_{condition}.yml").open("r", encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    with (config_root / "_embodiment_config.yml").open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    embodiment = list(BODY_EMBODIMENT[body])
    args["embodiment"] = embodiment

    def robot_file(name: str) -> Path:
        declared = Path(str(registry[name]["file_path"]))
        return (robotwin_root / declared).resolve()

    if len(embodiment) == 1:
        left_file = right_file = robot_file(str(embodiment[0]))
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    else:
        left_file = robot_file(str(embodiment[0]))
        right_file = robot_file(str(embodiment[1]))
        args["dual_arm_embodied"] = False
        args["embodiment_dis"] = float(embodiment[2])
        args["embodiment_name"] = f"{embodiment[0]}_{embodiment[1]}"

    def embodiment_config(path: Path) -> dict[str, Any]:
        with (path / "config.yml").open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    args.update(
        {
            "task_name": TASK,
            "task_config": f"demo_{condition}",
            "left_robot_file": str(left_file),
            "right_robot_file": str(right_file),
            "left_embodiment_config": embodiment_config(left_file),
            "right_embodiment_config": embodiment_config(right_file),
            "eval_mode": True,
            "eval_video_log": False,
            "collect_data": False,
            "render_freq": 0,
            "save_data": False,
        }
    )
    return args


def _new_task(task_class: Any, args: Mapping[str, Any], seed: int, instruction: str):
    task = task_class()
    task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **dict(args))
    # BaseTask's eval setup reloads the task YAML and otherwise restores its
    # built-in 400-step limit.  Keep simulator and collector termination on the
    # same explicit branch horizon.
    if "step_lim" in args:
        task.step_lim = int(args["step_lim"])
    task.set_instruction(instruction=instruction)
    task.scene = SimulationClockScene(task.scene)
    return task


def _sim_time(task: Any) -> float:
    scene = task.scene
    if not isinstance(scene, SimulationClockScene):
        raise BranchCollectionError("task scene is missing the physical simulation clock")
    return scene.sim_seconds


def success_height_reference_z(task: Any) -> float:
    """Return the exact per-episode height authority used by check_success."""

    value = getattr(task, "orig_z", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not np.isfinite(float(value))
    ):
        raise BranchCollectionError("task.orig_z success height reference is invalid")
    return float(value)


def _append_physical_observation(
    task: Any,
    objects: Sequence[Any],
    trajectory: list[np.ndarray],
    sim_times: list[float],
) -> None:
    now = _sim_time(task)
    if now <= sim_times[-1]:
        raise BranchCollectionError("simulator operation advanced no physical steps")
    trajectory.append(read_poses(objects))
    sim_times.append(now)


def _episode_done(task: Any, max_steps: int) -> bool:
    return bool(getattr(task, "eval_success", False)) or int(
        getattr(task, "take_action_cnt", 0)
    ) >= max_steps


def _root_prefix(
    *,
    task_class: Any,
    args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    seed: int,
    root_query: int,
    action_exec_steps: int,
    max_steps: int,
    required_pose_names: set[str],
    device: torch.device,
) -> dict[str, Any] | None:
    task = _new_task(task_class, args, seed, instruction)
    try:
        names, objects = discover_pose_objects(task, required_pose_names)
        trajectory = [read_poses(objects)]
        sim_times = [_sim_time(task)]
        for query_index in range(root_query):
            if _episode_done(task, max_steps):
                return None
            # The live paired policy scores after the same one-step contact
            # cache canonicalization at every actor query.  Reproduce it for
            # every historical prefix query, not only the sampled root.
            task.scene.step()
            _append_physical_observation(task, objects, trajectory, sim_times)
            chunk = generate_candidates(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                instruction=instruction,
                scene_seed=seed,
                query_index=query_index,
                candidate_count=1,
                device=device,
            )[0]
            for action in chunk[:action_exec_steps]:
                if _episode_done(task, max_steps):
                    break
                task.take_action(action, action_type="ee")
                _append_physical_observation(task, objects, trajectory, sim_times)
        if _episode_done(task, max_steps):
            return None
        remaining_action_budget = max_steps - int(
            getattr(task, "take_action_cnt", 0)
        )
        expected_remaining_action_budget = max_steps - root_query * action_exec_steps
        if remaining_action_budget != expected_remaining_action_budget:
            raise BranchCollectionError(
                "actor prefix action budget disagrees with the frozen query grid: "
                f"observed={remaining_action_budget}, "
                f"expected={expected_remaining_action_budget}"
            )
        if remaining_action_budget <= 0:
            raise BranchCollectionError(
                "non-terminal branch root has no remaining action budget"
            )
        snapshot = capture_branch_root_snapshot(task)
        snapshot_sha = branch_root_snapshot_sha256(snapshot)
        restorable_snapshot_sha = branch_root_restorable_snapshot_sha256(snapshot)
        height_reference_z = success_height_reference_z(task)
    finally:
        task.close_env(clear_cache=False)

    # Generate the candidate set in its own restored scene.  The single raw
    # PhysX step rebuilds contact manifolds identically in this reference and
    # in every candidate scene; it is part of the observed prefix clock.
    reference = _new_task(task_class, args, seed, instruction)
    try:
        restore_branch_root_snapshot(reference, snapshot)
        restored_snapshot = capture_branch_root_snapshot(reference)
        if not branch_root_restorable_snapshots_equal(snapshot, restored_snapshot):
            expected_restorable = branch_root_restorable_snapshot(snapshot)
            observed_restorable = branch_root_restorable_snapshot(
                restored_snapshot
            )
            expected_sections = branch_root_snapshot_section_sha256(
                expected_restorable
            )
            observed_sections = branch_root_snapshot_section_sha256(
                observed_restorable
            )
            changed = {
                name: [expected_sections[name], observed_sections.get(name)]
                for name in expected_sections
                if expected_sections[name] != observed_sections.get(name)
            }
            details = branch_root_snapshot_difference_summary(
                expected_restorable,
                observed_restorable,
                sections=changed,
            )
            raise BranchCollectionError(
                "fresh scene did not reproduce the saved root; changed_sections="
                + json.dumps(changed, sort_keys=True)
                + "; differences="
                + json.dumps(details, sort_keys=True)
            )
        reference.scene.step()
        if not np.isclose(
            success_height_reference_z(reference),
            height_reference_z,
            atol=0.0,
            rtol=0.0,
        ):
            raise BranchCollectionError(
                "fresh scene changed task.orig_z success height reference"
            )
        reference_names, reference_objects = discover_pose_objects(
            reference, required_pose_names
        )
        if list(reference_names) != list(names):
            raise BranchCollectionError("tracked object registry changed after restore")
        trajectory.append(read_poses(reference_objects))
        sim_times.append(_sim_time(reference))
        root_pose = trajectory[-1].copy()
        current = current_ee_action16(reference)
        canonical_snapshot_sha = branch_root_snapshot_sha256(
            capture_branch_root_snapshot(reference)
        )
        candidates = generate_candidates(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=reference,
            instruction=instruction,
            scene_seed=seed,
            query_index=root_query,
            candidate_count=CANDIDATE_COUNT,
            device=device,
        )
        root_sim_steps = int(reference.scene.step_count)
        sim_timestep_seconds = float(reference.scene.timestep_seconds)
    finally:
        reference.close_env(clear_cache=False)
    return {
        "branch_root_snapshot": snapshot,
        "branch_root_snapshot_sha256": snapshot_sha,
        "branch_root_restorable_snapshot_sha256": restorable_snapshot_sha,
        "canonical_root_snapshot_sha256": canonical_snapshot_sha,
        "object_names": names,
        "root_object_poses": root_pose,
        "root_ee_action": current,
        "prefix_trajectory": np.stack(trajectory),
        "prefix_sim_times": np.asarray(sim_times, dtype=np.float64),
        "root_sim_steps": root_sim_steps,
        "sim_timestep_seconds": sim_timestep_seconds,
        "remaining_action_budget": int(remaining_action_budget),
        "success_height_reference_z": height_reference_z,
        "candidates": candidates,
    }


def _execute_candidate_from_restored_root(
    *,
    task: Any,
    objects: Sequence[Any],
    root: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    seed: int,
    root_query: int,
    candidate: np.ndarray,
    action_exec_steps: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    trajectory = [
        # Preserve simulator float64 quaternions until the official strict
        # roll/pitch success checks have been evaluated.  Quantizing a saved
        # prefix to float32 can flip a sample exactly at the 15-degree bound.
        np.asarray(value, dtype=np.float64).copy()
        for value in np.asarray(root["prefix_trajectory"])
    ]
    sim_times = [float(value) for value in np.asarray(root["prefix_sim_times"])]
    root_step = len(trajectory) - 1
    first_executed = 0
    terminal_stop_reason_id = 1
    for action in candidate[:action_exec_steps]:
        if _episode_done(task, max_steps):
            break
        # RoboTwin reports an ordinary CuRobo planning failure by returning a
        # ``Fail`` plan and advancing the formal action; that is a valid policy
        # outcome.  A Python exception here instead signals broken collection
        # infrastructure and must invalidate the complete four-candidate root.
        task.take_action(action, action_type="ee")
        first_executed += 1
        _append_physical_observation(task, objects, trajectory, sim_times)
    post_step = len(trajectory) - 1
    query_index = root_query + 1
    while not _episode_done(task, max_steps):
        # Match the recursive live policy: every future actor query begins at
        # the same one-step canonicalized simulator time used for scoring.
        task.scene.step()
        _append_physical_observation(task, objects, trajectory, sim_times)
        # Policy/observation/runtime generation failures are collection
        # failures, not negative action outcomes.  They invalidate the whole
        # decision and intentionally propagate to the caller.
        continuation = generate_candidates(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            instruction=instruction,
            scene_seed=seed,
            query_index=query_index,
            candidate_count=1,
            device=device,
        )[0]
        for action in continuation[:action_exec_steps]:
            if _episode_done(task, max_steps):
                break
            task.take_action(action, action_type="ee")
            _append_physical_observation(task, objects, trajectory, sim_times)
        query_index += 1
    success = bool(getattr(task, "eval_success", False))
    if not success:
        success = bool(task.check_success())
    if success:
        terminal_stop_reason_id = 0
    return {
        "trajectory": np.stack(trajectory),
        "sim_times": np.asarray(sim_times, dtype=np.float64),
        "root_step": root_step,
        "post_step": post_step,
        "first_executed": first_executed,
        "sim_timestep_seconds": float(task.scene.timestep_seconds),
        "success": success,
        "branch_error": None,
        "terminal_stop_reason_id": terminal_stop_reason_id,
    }


def _evaluate_candidate(
    *,
    task_class: Any,
    args: Mapping[str, Any],
    root: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    seed: int,
    root_query: int,
    candidate: np.ndarray,
    action_exec_steps: int,
    max_steps: int,
    required_pose_names: set[str],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one candidate in an independently restored fresh scene."""

    task = _new_task(task_class, args, seed, instruction)
    try:
        restore_branch_root_snapshot(task, root["branch_root_snapshot"])
        restored_snapshot = capture_branch_root_snapshot(task)
        if not branch_root_restorable_snapshots_equal(
            root["branch_root_snapshot"], restored_snapshot
        ):
            expected_restorable = branch_root_restorable_snapshot(
                root["branch_root_snapshot"]
            )
            observed_restorable = branch_root_restorable_snapshot(
                restored_snapshot
            )
            expected_sections = branch_root_snapshot_section_sha256(
                expected_restorable
            )
            observed_sections = branch_root_snapshot_section_sha256(
                observed_restorable
            )
            changed = {
                name: [expected_sections[name], observed_sections.get(name)]
                for name in expected_sections
                if expected_sections[name] != observed_sections.get(name)
            }
            details = branch_root_snapshot_difference_summary(
                expected_restorable,
                observed_restorable,
                sections=changed,
            )
            raise BranchCollectionError(
                "fresh candidate scene changed the saved root; changed_sections="
                + json.dumps(changed, sort_keys=True)
                + "; differences="
                + json.dumps(details, sort_keys=True)
            )
        task.scene.step()
        names, objects = discover_pose_objects(task, required_pose_names)
        if list(names) != list(root["object_names"]):
            raise BranchCollectionError("fresh candidate object registry changed")
        if (
            branch_root_snapshot_sha256(capture_branch_root_snapshot(task))
            != root["canonical_root_snapshot_sha256"]
            or int(task.scene.step_count) != int(root["root_sim_steps"])
            or not np.isclose(
                float(task.scene.timestep_seconds),
                float(root["sim_timestep_seconds"]),
                atol=1e-12,
                rtol=0.0,
            )
            or not np.allclose(
                read_poses(objects),
                root["root_object_poses"],
                atol=2e-5,
                rtol=0.0,
            )
            or not np.allclose(
                current_ee_action16(task),
                root["root_ee_action"],
                atol=2e-5,
                rtol=0.0,
            )
            or not np.isclose(
                success_height_reference_z(task),
                float(root["success_height_reference_z"]),
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise BranchCollectionError(
                "fresh candidate canonical root differs from candidate-generation root"
            )
        # Match the observation/render call made while generating the frozen
        # candidate set without invoking the actor again.
        task.get_obs()
        return _execute_candidate_from_restored_root(
            task=task,
            objects=objects,
            root=root,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            instruction=instruction,
            seed=seed,
            root_query=root_query,
            candidate=candidate,
            action_exec_steps=action_exec_steps,
            max_steps=max_steps,
            device=device,
        )
    finally:
        task.close_env(clear_cache=False)


def _state27(
    *,
    poses: np.ndarray,
    names: Sequence[str],
    step: int,
    initial_moving_position: np.ndarray,
    ee_action: np.ndarray,
    event: int,
    predicates: np.ndarray,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    moving_index = list(names).index(str(calibration["moving"]))
    moving = poses[step, moving_index]
    _moving, relative_goal = _goal_vector(poses, names, step, calibration)
    event_onehot = np.zeros(5, dtype=np.float32)
    event_onehot[event] = 1.0
    return canonical_adapter.pack_shared_critic_state27(
        relative_goal,
        moving[:3] - ee_action[0:3],
        moving[:3] - ee_action[8:11],
        moving[:3] - initial_moving_position,
        np.asarray([ee_action[7], ee_action[15]], dtype=np.float32),
        moving[3:7],
        event_onehot,
        predicates[step, :4],
    )


def materialize_group(
    *,
    root: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    action_exec_steps: int,
) -> dict[str, np.ndarray]:
    if len(outcomes) != CANDIDATE_COUNT or len(root["candidates"]) != CANDIDATE_COUNT:
        raise BranchCollectionError("materialization requires exactly four complete branches")
    if any(outcome.get("branch_error") is not None for outcome in outcomes):
        raise BranchCollectionError(
            "action execution exceptions invalidate the complete candidate decision"
        )
    names = list(root["object_names"])
    moving_index = names.index(str(calibration["moving"]))
    prefix = np.asarray(root["prefix_trajectory"], dtype=np.float64)
    initial_moving = prefix[0, moving_index, :3]
    prefix_times = np.asarray(root["prefix_sim_times"], dtype=np.float64)
    prefix_predicates, prefix_events = derive_predicates_and_events(
        prefix,
        prefix_times,
        names,
        False,
        calibration,
        float(root["success_height_reference_z"]),
    )
    current_event = int(prefix_events[-1])
    root_event_age = event_age_seconds(prefix_events, prefix_times)
    state = _state27(
        poses=prefix,
        names=names,
        step=len(prefix) - 1,
        initial_moving_position=initial_moving,
        ee_action=np.asarray(root["root_ee_action"]),
        event=current_event,
        predicates=prefix_predicates,
        calibration=calibration,
    )
    horizon = int(np.asarray(root["candidates"]).shape[1])
    actions = []
    masks = []
    post_event = []
    next_event = []
    next_mask = []
    duration = []
    duration_observed = []
    duration_mask = []
    success = []
    recovery = []
    recovery_mask = []
    object_delta = []
    object_delta_mask = []
    terminal_max_event = []
    terminal_event_mask = []
    terminal_stage_progress = []
    terminal_goal_distance = []
    terminal_goal_progress = []
    terminal_goal_progress_mask = []
    terminal_stop_reason = []
    for candidate, outcome in zip(root["candidates"], outcomes):
        action = canonical_action_chunk(root["root_ee_action"], candidate)
        # The critic scores the proposal before execution, so it must see the
        # complete planned first chunk even when planning/execution later
        # fails.  Executed action count is used only for physical timing and
        # must never censor the action that caused a negative outcome.
        planned_steps = min(
            int(action_exec_steps),
            int(root["remaining_action_budget"]),
            horizon,
        )
        mask = np.arange(horizon) < planned_steps
        trajectory = np.asarray(outcome["trajectory"], dtype=np.float64)
        sim_times = np.asarray(outcome["sim_times"], dtype=np.float64)
        predicates, events = derive_predicates_and_events(
            trajectory,
            sim_times,
            names,
            bool(outcome["success"]),
            calibration,
            float(root["success_height_reference_z"]),
        )
        root_step = int(outcome["root_step"])
        post_step = int(outcome["post_step"])
        if int(events[root_step]) != current_event:
            raise BranchCollectionError("candidate replay changed the root event")
        future = np.flatnonzero(events[root_step + 1 :] != current_event)
        if len(future):
            boundary = root_step + 1 + int(future[0])
            next_id = int(events[boundary])
            duration_seconds = sim_times[boundary] - sim_times[root_step]
            observed = True
        else:
            next_id = current_event
            duration_seconds = sim_times[-1] - sim_times[root_step]
            observed = False
        regressed = int(events[post_step]) < current_event
        recovered = bool(regressed and np.any(events[post_step + 1 :] >= current_event))
        moving_start, relative_start = _goal_vector(
            trajectory, names, root_step, calibration
        )
        moving_post, _relative_post = _goal_vector(
            trajectory, names, post_step, calibration
        )
        _moving_terminal, relative_terminal = _goal_vector(
            trajectory, names, len(trajectory) - 1, calibration
        )
        moving_rotation_delta = canonical_adapter.relative_axis_angle_wxyz(
            trajectory[root_step, moving_index, 3:7],
            trajectory[post_step, moving_index, 3:7],
        )
        # This head predicts consequences that remain changeable by the
        # candidate.  Prefix achievements before the branch root are common
        # to all four candidates and must not enter the action target.
        maximum_event = int(events[root_step:].max())
        branch_success = bool(outcome["success"])
        actions.append(action)
        masks.append(mask)
        post_event.append(int(events[post_step]))
        next_event.append(next_id)
        next_mask.append(observed)
        duration.append(float(duration_seconds))
        duration_observed.append(observed)
        # A zero-step planning failure has no temporal exposure.  Keep its
        # success/ranking/zero-object-effect supervision, but do not turn a
        # vacuous censored duration at t=0 into a clock gradient.
        duration_mask.append(
            float(duration_seconds) > 0.0
            and (observed or outcome.get("branch_error") is None)
        )
        success.append(branch_success)
        recovery.append(recovered)
        recovery_mask.append(regressed)
        object_delta.append(np.r_[moving_post - moving_start, moving_rotation_delta])
        object_delta_mask.append(True)
        terminal_max_event.append(maximum_event)
        terminal_event_mask.append(True)
        terminal_stage_progress.append(
            1.0 if branch_success else maximum_event / float(len(CANONICAL_EVENTS) - 1)
        )
        terminal_distance = float(np.linalg.norm(relative_terminal))
        terminal_goal_distance.append(terminal_distance)
        terminal_goal_progress.append(float(np.linalg.norm(relative_start)) - terminal_distance)
        terminal_goal_progress_mask.append(True)
        terminal_stop_reason.append(int(outcome["terminal_stop_reason_id"]))

    count = CANDIDATE_COUNT
    arrays = {
        "state": np.repeat(state[None], count, axis=0).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "action_mask": np.asarray(masks, dtype=bool),
        "current_event_id": np.full(count, current_event, dtype=np.int64),
        "post_event_id": np.asarray(post_event, dtype=np.int64),
        "post_event_mask": np.ones(count, dtype=np.float32),
        "next_event_id": np.asarray(next_event, dtype=np.int64),
        "next_event_mask": np.asarray(next_mask, dtype=np.float32),
        "duration": np.asarray(duration, dtype=np.float32),
        "duration_observed": np.asarray(duration_observed, dtype=np.float32),
        "duration_mask": np.asarray(duration_mask, dtype=np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "success_mask": np.ones(count, dtype=np.float32),
        "recovery": np.asarray(recovery, dtype=np.float32),
        "recovery_mask": np.asarray(recovery_mask, dtype=np.float32),
        "object_delta": np.asarray(object_delta, dtype=np.float32),
        "object_delta_mask": np.asarray(object_delta_mask, dtype=np.float32),
        "terminal_max_event_id": np.asarray(terminal_max_event, dtype=np.int64),
        "terminal_event_mask": np.asarray(terminal_event_mask, dtype=np.float32),
        "terminal_stage_progress": np.asarray(terminal_stage_progress, dtype=np.float32),
        "terminal_goal_distance": np.asarray(terminal_goal_distance, dtype=np.float32),
        "terminal_goal_progress": np.asarray(terminal_goal_progress, dtype=np.float32),
        "terminal_goal_progress_mask": np.asarray(
            terminal_goal_progress_mask, dtype=np.float32
        ),
        "terminal_stop_reason_id": np.asarray(terminal_stop_reason, dtype=np.int64),
        "candidate_index": np.arange(count, dtype=np.int64),
        "event_age_seconds": np.full(count, root_event_age, dtype=np.float32),
        "remaining_action_budget": np.full(
            count, int(root["remaining_action_budget"]), dtype=np.float32
        ),
        "success_height_reference_z": np.full(
            count, float(root["success_height_reference_z"]), dtype=np.float64
        ),
        # ``dt`` is an execution-time critic input, not an outcome.  Keep it
        # equal across the four candidates and known before execution; only
        # event ``duration`` above uses counted simulator seconds.
        "dt": np.full(
            count,
            min(
                int(action_exec_steps),
                int(root["remaining_action_budget"]),
                horizon,
            )
            / SOURCE_EVENT_SAMPLING_HZ,
            dtype=np.float32,
        ),
    }
    if arrays["actions"].shape != (count, horizon, CANONICAL_ACTION_DIM):
        raise BranchCollectionError("canonical group action shape changed")
    if arrays["state"].shape != (count, STATE_DIM):
        raise BranchCollectionError("canonical group state shape changed")
    if arrays["object_delta"].shape != (count, OBJECT_DELTA_DIM):
        raise BranchCollectionError("canonical object effect shape changed")
    if not np.array_equal(
        arrays["terminal_stage_progress"],
        np.where(
            arrays["success"] > 0.5,
            1.0,
            arrays["terminal_max_event_id"] / float(len(CANONICAL_EVENTS) - 1),
        ).astype(np.float32),
    ):
        raise BranchCollectionError("terminal stage progress disagrees with formal definition")
    if np.any(
        (arrays["terminal_max_event_id"] < 0)
        | (arrays["terminal_max_event_id"] >= len(CANONICAL_EVENTS))
        | (arrays["terminal_goal_distance"] < 0.0)
    ):
        raise BranchCollectionError("terminal branch supervision is outside its domain")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise BranchCollectionError("canonical group contains non-finite values")
    return arrays


def materialize_branch_diagnostics(
    *,
    root: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    action_exec_steps: int,
) -> dict[str, np.ndarray]:
    """Return label-free/action-execution diagnostics bound beside one group.

    These fields are deliberately separate from the trainable row arrays: they
    quantify proposal coverage and simulator infeasibility without silently
    becoming model inputs or ranking targets.
    """

    candidates = np.asarray(root["candidates"], dtype=np.float32)
    if candidates.ndim != 3 or len(candidates) != CANDIDATE_COUNT:
        raise BranchCollectionError("branch diagnostics require four actor candidates")
    if len(outcomes) != CANDIDATE_COUNT:
        raise BranchCollectionError("branch diagnostics require four outcomes")
    current = np.asarray(root["root_ee_action"], dtype=np.float32)
    effects = np.stack(
        [canonical_action_chunk(current, candidate) for candidate in candidates]
    ).astype(np.float32)
    planned_execution = min(
        int(action_exec_steps),
        int(root["remaining_action_budget"]),
        int(effects.shape[1]),
    )
    diagnostic_prefix = min(DIAGNOSTIC_ACTION_PREFIX_STEPS, planned_execution)
    if diagnostic_prefix < 2:
        raise BranchCollectionError(
            "branch diagnostics require first and subsequent planned action tokens"
        )
    # This label-free continuity/noise diagnostic intentionally remains a
    # first-five-token measurement for both execute-5 and execute-50.  It is
    # not the critic action mask and must never be widened with the stride.
    first = effects[:, None, :diagnostic_prefix, :]
    second = effects[None, :, :diagnostic_prefix, :]
    pairwise_rms = np.sqrt(np.mean(np.square(first - second), axis=(2, 3))).astype(
        np.float32
    )
    translation_norm = np.stack(
        (
            np.linalg.norm(effects[:, :diagnostic_prefix, 0:3], axis=2),
            np.linalg.norm(effects[:, :diagnostic_prefix, 7:10], axis=2),
        ),
        axis=2,
    ).astype(np.float32)
    arrays = {
        "first_executed": np.asarray(
            [int(outcome["first_executed"]) for outcome in outcomes], dtype=np.int64
        ),
        "branch_error": np.asarray(
            [outcome.get("branch_error") is not None for outcome in outcomes], dtype=bool
        ),
        "candidate_action_pairwise_rms": pairwise_rms,
        "candidate_first_token_translation_norm_m": translation_norm[:, 0, :],
        "candidate_later_token_translation_norm_median_m": np.median(
            translation_norm[:, 1:, :], axis=1
        ).astype(np.float32),
    }
    if arrays["first_executed"].shape != (CANDIDATE_COUNT,) or np.any(
        (arrays["first_executed"] < 0)
        | (arrays["first_executed"] > planned_execution)
    ):
        raise BranchCollectionError("first-executed diagnostic is outside planned horizon")
    if arrays["branch_error"].shape != (CANDIDATE_COUNT,):
        raise BranchCollectionError("branch-error diagnostic shape changed")
    if any(
        arrays[name].shape != (CANDIDATE_COUNT, 2)
        for name in (
            "candidate_first_token_translation_norm_m",
            "candidate_later_token_translation_norm_median_m",
        )
    ):
        raise BranchCollectionError("first-token continuity diagnostic shape changed")
    if pairwise_rms.shape != (CANDIDATE_COUNT, CANDIDATE_COUNT) or not np.allclose(
        pairwise_rms, pairwise_rms.T, atol=1e-7, rtol=0.0
    ) or not np.allclose(np.diag(pairwise_rms), 0.0, atol=1e-7, rtol=0.0):
        raise BranchCollectionError("candidate pairwise-distance diagnostic is invalid")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise BranchCollectionError("branch diagnostics contain non-finite values")
    return arrays


def resolve_query_contract(
    requested: Sequence[int], declared: Sequence[int] | None
) -> tuple[list[int], list[int]]:
    """Separate resumable work lanes from the immutable manifest universe."""

    requested_queries = sorted(int(value) for value in requested)
    manifest_queries = sorted(
        int(value) for value in (requested if declared is None else declared)
    )
    if (
        not requested_queries
        or not manifest_queries
        or requested_queries[0] < 0
        or manifest_queries[0] < 0
    ):
        raise BranchCollectionError("root query indices must be non-negative")
    if (
        len(set(requested_queries)) != len(requested_queries)
        or len(set(manifest_queries)) != len(manifest_queries)
    ):
        raise BranchCollectionError("root query indices must be unique")
    if not set(requested_queries).issubset(manifest_queries):
        raise BranchCollectionError(
            "requested root queries must be a subset of the manifest query universe"
        )
    return requested_queries, manifest_queries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", choices=BODIES, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument(
        "--actor-execution-protocol", type=Path, required=True
    )
    parser.add_argument(
        "--actor-execution-protocol-sha256", required=True
    )
    parser.add_argument(
        "--path-root",
        type=Path,
        required=True,
        help="Absolute root used to resolve every relative artifact binding",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seed-start", type=int, default=2026081000)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument(
        "--root-query-indices", nargs="+", type=int
    )
    parser.add_argument(
        "--manifest-root-query-indices",
        nargs="+",
        type=int,
        help=(
            "Immutable query-index universe recorded in the manifest.  A resume "
            "invocation may request a subset via --root-query-indices, but may "
            "not change this universe."
        ),
    )
    parser.add_argument(
        "--action-exec-steps", type=int
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    return parser.parse_args(argv)


def bind_execution_protocol_arguments(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, list[int], list[int]]:
    """Normalize all stride-dependent CLI values from one frozen file."""

    try:
        execution_protocol = actor_execution.load_execution_protocol_file(
            args.actor_execution_protocol,
            args.actor_execution_protocol_sha256,
        )
    except actor_execution.ActorExecutionProtocolError as error:
        raise BranchCollectionError(str(error)) from error
    action_exec_steps = int(execution_protocol["stride"])
    max_steps = int(execution_protocol["max_steps"])
    args.action_exec_steps = (
        action_exec_steps
        if args.action_exec_steps is None
        else int(args.action_exec_steps)
    )
    args.max_steps = max_steps if args.max_steps is None else int(args.max_steps)
    args.seed_count = (
        int(execution_protocol["target_per_condition_query"])
        if args.seed_count is None
        else int(args.seed_count)
    )
    args.root_query_indices = (
        list(execution_protocol["query_indices"])
        if args.root_query_indices is None
        else list(args.root_query_indices)
    )
    if args.manifest_root_query_indices is None:
        args.manifest_root_query_indices = list(execution_protocol["query_indices"])
    try:
        protocol_binding = actor_execution.execution_protocol_file_binding(
            args.actor_execution_protocol,
            args.actor_execution_protocol_sha256,
            path_root=args.path_root,
        )
    except actor_execution.ActorExecutionProtocolError as error:
        raise BranchCollectionError(str(error)) from error
    path_root = Path(protocol_binding["path_root"])
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(path_root)
    except ValueError as error:
        raise BranchCollectionError("collector output must be contained by path_root") from error
    if args.seed_count <= 0 or args.action_exec_steps <= 0:
        raise BranchCollectionError("seed-count/action-exec-steps must be positive")
    if (
        args.action_exec_steps != action_exec_steps
        or args.max_steps != max_steps
    ):
        raise BranchCollectionError(
            "formal consequence collection arguments disagree with the bound "
            "actor execution protocol"
        )
    if args.instruction != DEFAULT_INSTRUCTION:
        raise BranchCollectionError(
            "formal consequence collection fixes the actor instruction"
        )
    requested_queries, manifest_queries = resolve_query_contract(
        args.root_query_indices,
        args.manifest_root_query_indices,
    )
    if manifest_queries != list(execution_protocol["query_indices"]):
        raise BranchCollectionError(
            "manifest root query universe differs from the bound actor execution protocol"
        )
    return (
        execution_protocol,
        protocol_binding,
        path_root,
        requested_queries,
        manifest_queries,
    )


def main() -> None:
    args = parse_args()
    (
        execution_protocol,
        protocol_binding,
        path_root,
        requested_queries,
        manifest_queries,
    ) = bind_execution_protocol_arguments(args)
    action_exec_steps = int(execution_protocol["stride"])
    max_steps = int(execution_protocol["max_steps"])
    terminal_contract = terminal_horizon_contract(execution_protocol)
    output = args.output.expanduser().resolve()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise BranchCollectionError("real branch collection requires remote RTX 4090 CUDA")
    for path in (
        args.actor_checkpoint,
        args.vlm_metadata_path,
        args.robotwin_root,
        args.event_spec,
        args.actor_execution_protocol,
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

    module = __import__(f"envs.{TASK}", fromlist=[TASK])
    task_class = getattr(module, TASK)
    device = torch.device("cuda:0")
    config = PreTrainedConfig.from_pretrained(
        args.actor_checkpoint, local_files_only=True
    )
    config.device = str(device)
    config.vlm_model_name = str(args.vlm_metadata_path.resolve())
    config.load_vlm_weights = False
    if config.action_feature is None or int(config.action_feature.shape[0]) != NATIVE_EE_DIM:
        raise BranchCollectionError("actor checkpoint must have a 16-D EE action feature")
    if config.input_features.get("observation.state") is None or int(
        config.input_features["observation.state"].shape[0]
    ) != NATIVE_EE_DIM:
        raise BranchCollectionError("actor checkpoint must have a 16-D EE state feature")
    if int(config.chunk_size) != int(execution_protocol["native_chunk_steps"]):
        raise BranchCollectionError(
            "actor native action chunk differs from the bound execution protocol"
        )
    policy = SmolVLAPolicy.from_pretrained(
        args.actor_checkpoint, config=config, local_files_only=True, strict=True
    ).eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.actor_checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_metadata_path)},
        },
    )

    try:
        event_spec, calibration = analytic_event.load_event_spec(args.event_spec)
    except analytic_event.AnalyticEventSpecError as error:
        raise BranchCollectionError(str(error)) from error
    required_pose_names = set(analytic_event.REQUIRED_OBJECTS)
    adapter_source = Path(inspect.getsourcefile(canonical_adapter) or "")
    adapter_sha = sha256_file(adapter_source)
    event_source = Path(inspect.getsourcefile(analytic_event) or "")
    event_implementation_sha = sha256_file(event_source)
    collector_sha = sha256_file(Path(__file__).resolve())
    groups_dir = output / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = dict(manifest)
        logical = unsigned.pop("logical_sha256", None)
        if (
            logical != canonical_sha256(unsigned)
            or manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("collector_format") != FORMAT
            or manifest.get("body") != args.body
            or manifest.get("collector_file_sha256") != collector_sha
            or manifest.get("path_root") != str(path_root)
            or manifest.get("actor_execution_protocol_binding")
            != protocol_binding
            or manifest.get("actor_execution_protocol") != execution_protocol
            or manifest.get("actor_execution_protocol_file_sha256")
            != args.actor_execution_protocol_sha256
            or manifest.get("actor_checkpoint") != str(args.actor_checkpoint.resolve())
            or manifest.get("instruction") != DEFAULT_INSTRUCTION
            or manifest.get("candidate_count") != CANDIDATE_COUNT
            or manifest.get("root_query_indices") != manifest_queries
            or manifest.get("action_exec_steps") != args.action_exec_steps
            or manifest.get("max_episode_action_steps") != args.max_steps
            or manifest.get("candidate_noise_contract") != CANDIDATE_NOISE_CONTRACT
            or manifest.get("state_action_frame_contract")
            != STATE_ACTION_FRAME_CONTRACT
            or manifest.get("terminal_supervision_contract")
            != TERMINAL_SUPERVISION_CONTRACT
            or manifest.get("event_age_contract") != EVENT_AGE_CONTRACT
            or manifest.get("terminal_horizon_contract")
            != terminal_contract
            or manifest.get("branch_root_snapshot_contract")
            != BRANCH_ROOT_SNAPSHOT_CONTRACT
            or manifest.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
            or manifest.get("branch_diagnostic_contract")
            != BRANCH_DIAGNOSTIC_CONTRACT
        ):
            raise BranchCollectionError("existing manifest does not match this collection")
    else:
        manifest = {
            "format": MANIFEST_FORMAT,
            "collector_format": FORMAT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "body": args.body,
            "collector_file_sha256": collector_sha,
            "path_root": str(path_root),
            "actor_execution_protocol_binding": protocol_binding,
            "actor_execution_protocol": execution_protocol,
            "actor_execution_protocol_file_sha256": (
                args.actor_execution_protocol_sha256
            ),
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "instruction": DEFAULT_INSTRUCTION,
            "actor_checkpoint_tree_or_file_sha256_recorded_separately": True,
            "candidate_count": CANDIDATE_COUNT,
            "action_exec_steps": int(args.action_exec_steps),
            "max_episode_action_steps": int(args.max_steps),
            "candidate_zero_is_actor_baseline": True,
            "same_ordered_candidate_set_for_baseline_and_etsf": True,
            "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
            "state_action_frame_contract": STATE_ACTION_FRAME_CONTRACT,
            "root_query_indices": manifest_queries,
            "schema_adapter": {
                "kind": "analytic_label_free_canonical_v1",
                "trainable": False,
                "labels_or_outcomes_used_to_fit": False,
                "heldout_supervision_allowed": False,
                "state_dim": STATE_DIM,
                "action_dim": CANONICAL_ACTION_DIM,
                "state_schema": STATE_SCHEMA,
                "action_schema": ACTION_SCHEMA,
                "elapsed_time_unit": "seconds",
                "duration_unit": "seconds",
                "event_names": list(CANONICAL_EVENTS),
                "implementation_sha256": adapter_sha,
            },
            "analytic_event_contract": analytic_event.event_contract(calibration),
            "event_derivation_implementation_sha256": event_implementation_sha,
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
                "planned_action_steps": min(
                    int(args.action_exec_steps), int(config.chunk_size)
                ),
                "actor_control_hz": SOURCE_EVENT_SAMPLING_HZ,
                "planned_dt_seconds": min(
                    int(args.action_exec_steps), int(config.chunk_size)
                )
                / SOURCE_EVENT_SAMPLING_HZ,
                "duration_semantics": "simulator_elapsed_seconds_to_event_boundary",
                "zero_elapsed_duration_masked": True,
                "event_thresholds": dict(calibration["thresholds"]),
                "event_chain_success_aligned": True,
            },
            "candidate_action_contract": {
                "critic_observation_time": "before_candidate_execution",
                "planned_action_horizon": min(
                    int(args.action_exec_steps), int(config.chunk_size)
                ),
                "action_mask_source": "planned_first_chunk_not_executed_count",
                "executed_action_count_used_for_action_mask": False,
                "executed_action_count_used_for_sim_time_accounting_only": True,
                "planner_status_fail_is_a_valid_action_outcome": True,
                "python_execution_exception_invalidates_complete_decision": True,
            },
            "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
            "event_age_contract": EVENT_AGE_CONTRACT,
            "terminal_horizon_contract": terminal_contract,
            "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
            "object_effect_schema": OBJECT_EFFECT_SCHEMA,
            "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "groups": [],
        }

    existing_ids = {str(item["group_id"]) for item in manifest["groups"]}
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    for condition in args.conditions:
        task_args = _load_task_args(args.robotwin_root, args.body, condition)
        task_args["step_lim"] = args.max_steps
        for seed in seeds:
            for root_query in requested_queries:
                group_id = f"{condition}|seed={seed}|query={root_query}"
                if group_id in existing_ids:
                    continue
                started = time.time()
                root = _root_prefix(
                    task_class=task_class,
                    args=task_args,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    instruction=args.instruction,
                    seed=seed,
                    root_query=root_query,
                    action_exec_steps=args.action_exec_steps,
                    max_steps=args.max_steps,
                    required_pose_names=required_pose_names,
                    device=device,
                )
                if root is None:
                    print(
                        "SKIP_TERMINAL_ROOT="
                        + json.dumps({"condition": condition, "seed": seed, "query": root_query}),
                        flush=True,
                    )
                    continue
                outcomes = [
                    _evaluate_candidate(
                        task_class=task_class,
                        args=task_args,
                        root=root,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        instruction=args.instruction,
                        seed=seed,
                        root_query=root_query,
                        candidate=candidate,
                        action_exec_steps=args.action_exec_steps,
                        max_steps=args.max_steps,
                        required_pose_names=required_pose_names,
                        device=device,
                    )
                    for candidate in root["candidates"]
                ]
                arrays = materialize_group(
                    root=root,
                    outcomes=outcomes,
                    calibration=calibration,
                    action_exec_steps=args.action_exec_steps,
                )
                diagnostics = materialize_branch_diagnostics(
                    root=root,
                    outcomes=outcomes,
                    action_exec_steps=args.action_exec_steps,
                )
                filename = f"{condition}_seed_{seed}_query_{root_query}.npz"
                path = groups_dir / filename
                atomic_npz(path, arrays)
                diagnostic_filename = (
                    f"{condition}_seed_{seed}_query_{root_query}.diagnostics.npz"
                )
                diagnostic_path = groups_dir / diagnostic_filename
                atomic_npz(diagnostic_path, diagnostics)
                item = {
                    "group_id": group_id,
                    "collector_file_sha256": collector_sha,
                    "condition": condition,
                    "requested_seed": int(seed),
                    "root_query_index": int(root_query),
                    "branch_root_snapshot_sha256": root[
                        "branch_root_snapshot_sha256"
                    ],
                    "branch_root_restorable_snapshot_sha256": root[
                        "branch_root_restorable_snapshot_sha256"
                    ],
                    "canonical_root_snapshot_sha256": root[
                        "canonical_root_snapshot_sha256"
                    ],
                    "path": f"groups/{filename}",
                    "sha256": sha256_file(path),
                    "diagnostic_format": DIAGNOSTIC_FORMAT,
                    "diagnostics_path": f"groups/{diagnostic_filename}",
                    "diagnostics_sha256": sha256_file(diagnostic_path),
                    "wall_seconds": time.time() - started,
                }
                manifest["groups"].append(item)
                existing_ids.add(group_id)
                unsigned = dict(manifest)
                unsigned.pop("logical_sha256", None)
                manifest["logical_sha256"] = canonical_sha256(unsigned)
                atomic_json(manifest_path, manifest)
                print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)

    unsigned = dict(manifest)
    unsigned.pop("logical_sha256", None)
    manifest["logical_sha256"] = canonical_sha256(unsigned)
    atomic_json(manifest_path, manifest)
    print(
        "COLLECTION_COMPLETE="
        + json.dumps(
            {
                "body": args.body,
                "groups": len(manifest["groups"]),
                "manifest": str(manifest_path),
                "logical_sha256": manifest["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
