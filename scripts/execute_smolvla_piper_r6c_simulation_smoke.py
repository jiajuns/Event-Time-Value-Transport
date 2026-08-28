#!/usr/bin/env python3
"""Execute an R6c SmolVLA(Aloha-trained) candidate in Piper simulation only.

This is a deliberately narrow open-loop interface smoke.  It binds the exact
R6c preflight manifest/receipt, maps all 14 slots by the frozen named
side/ordinal registry, and executes at most eight already-produced actions in
RoboTwin's Piper simulator.  Every command is checked immediately before
``env.step``; an out-of-range or non-finite value aborts and is never clipped.

The command cannot target a real robot, cannot read Fresh trajectories/labels,
does not run a new policy forward, and does not authorize a transfer or task
success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import verify_smolvla_piper_zero_shot_preflight as preflight_module
from verify_smolvla_piper_zero_shot_preflight import (
    ACTOR_ID,
    PIPER_ACTION_SLOTS,
    adapt_aloha_source_actions_to_piper_forward_interface,
    array_sha256,
    file_sha256,
    reject_fresh_path,
    run_preflight,
)


FORMAT = "smolvla_piper_r6c_simulation_smoke_v1"
PREREGISTRATION_FORMAT = "smolvla_piper_r6c_simulation_smoke_preregistration_v1"
R6C_MANIFEST_SHA256 = "c52b3c6deb37011cd8c94d2b8585279f764fc410fabe119c4855116ea7a0662c"
R6C_RECEIPT_SHA256 = "13256b3675bab8b671b161fd49bb7fe93e6d7d2e449fee707ee5318d39e7b674"
R6C_VERIFIER_SHA256 = "6e406a8256ffe1951a7b28e144a8a72dca0612283440ee0b74a72d0d720e38ff"
R6C_DIRECTORY_NAME = "etsf_smolvla_piper_forward_probe_r6c_20260827"
V7_DEVELOPMENT_SEED_MANIFEST_SHA256 = (
    "403bb28a5010a402acb41a68baa02a71355f1c99c0a8e927894292cf803478f1"
)
REQUESTED_DEVELOPMENT_SEED = 100101000
EXPECTED_RESOLVED_DEVELOPMENT_SEED = 100101000
TASK = "move_can_pot"
MAX_SMOKE_STEPS = 8
ACTION_DIM = 14


class SimulationSmokeError(RuntimeError):
    """A fail-closed simulation smoke contract violation."""


class ActionLimitError(SimulationSmokeError):
    """A command would violate a frozen Piper action bound."""


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SimulationSmokeError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SimulationSmokeError(f"{role} must contain a JSON object")
    return value


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)):
        value = value[0]
    return bool(value)


def reset_with_explicit_instruction(
    env: Any,
    *,
    requested_seed: int,
    instruction: str,
) -> tuple[Mapping[str, Any], Any]:
    """Reset once while bypassing RoboTwin's stale ``episode_info_list`` text.

    The audited VectorEnv does not rebuild ``episode_info_list`` for an explicit
    seed reset.  The preregistered instruction is therefore injected only for
    this reset, then the original method is restored before any action runs.
    """

    if not instruction:
        raise SimulationSmokeError("explicit instruction must be non-empty")
    venv = getattr(env, "venv", None)
    subenvs = getattr(venv, "envs", None)
    if not isinstance(subenvs, list) or len(subenvs) != 1:
        raise SimulationSmokeError("RoboTwin must expose exactly one simulation subenv")
    subenv = subenvs[0]
    original = getattr(subenv, "create_instruction", None)
    if not callable(original):
        raise SimulationSmokeError("RoboTwin subenv lacks create_instruction")
    subenv.create_instruction = lambda: instruction
    try:
        observation, info = env.reset(env_seeds=[requested_seed])
    finally:
        subenv.create_instruction = original
    return observation, info


def validate_reset_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and hash the state/main/two-wrist observation interface."""

    def array(name: str) -> np.ndarray:
        value = observation.get(name)
        if value is None:
            raise SimulationSmokeError(f"RoboTwin reset lacks {name}")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value)
        if not np.all(np.isfinite(result)):
            raise SimulationSmokeError(f"RoboTwin reset {name} is non-finite")
        return result

    state = array("states")
    main = array("main_images")
    wrists = array("wrist_images")
    if state.shape != (1, ACTION_DIM):
        raise SimulationSmokeError("RoboTwin reset state must have shape [1,14]")
    if main.ndim != 4 or main.shape[0] != 1 or main.shape[-1] != 3:
        raise SimulationSmokeError("RoboTwin main image must have shape [1,H,W,3]")
    if wrists.ndim != 5 or wrists.shape[0] != 1 or wrists.shape[1] != 2 or wrists.shape[-1] != 3:
        raise SimulationSmokeError(
            "RoboTwin wrist images must have shape [1,2,H,W,3]"
        )
    if main.dtype != np.uint8 or wrists.dtype != np.uint8:
        raise SimulationSmokeError("RoboTwin RGB observations must be uint8")
    state32 = np.ascontiguousarray(state.astype(np.float32, copy=False))
    main8 = np.ascontiguousarray(main)
    wrists8 = np.ascontiguousarray(wrists)
    return {
        "drive_target": state32[0],
        "state_shape": list(state32.shape),
        "state_sha256": array_sha256(state32),
        "main_image_shape": list(main8.shape),
        "main_image_sha256": array_sha256(main8),
        "wrist_images_shape": list(wrists8.shape),
        "wrist_images_sha256": array_sha256(wrists8),
    }


def bind_r6c_preflight(
    manifest_path: Path,
    receipt_path: Path,
    *,
    expected_manifest_sha256: str = R6C_MANIFEST_SHA256,
    expected_receipt_sha256: str = R6C_RECEIPT_SHA256,
    expected_verifier_sha256: str = R6C_VERIFIER_SHA256,
    expected_directory_name: str = R6C_DIRECTORY_NAME,
    runner: Callable[[Mapping[str, Any]], dict[str, Any]] = run_preflight,
) -> dict[str, Any]:
    """Authenticate and recompute the exact forward-only R6c preflight."""

    manifest_file = reject_fresh_path(manifest_path, "R6c preflight manifest")
    receipt_file = reject_fresh_path(receipt_path, "R6c preflight receipt")
    if manifest_file.parent != receipt_file.parent:
        raise SimulationSmokeError("R6c manifest and receipt must share one directory")
    if manifest_file.parent.name != expected_directory_name:
        raise SimulationSmokeError("preflight directory is not the frozen R6c directory")
    if manifest_file.name != "preflight_manifest.json":
        raise SimulationSmokeError("unexpected R6c manifest filename")
    if receipt_file.name != "preflight_receipt.json":
        raise SimulationSmokeError("unexpected R6c receipt filename")
    if not manifest_file.is_file() or not receipt_file.is_file():
        raise FileNotFoundError("R6c manifest/receipt file is missing")
    manifest_sha = file_sha256(manifest_file)
    receipt_sha = file_sha256(receipt_file)
    if manifest_sha != expected_manifest_sha256:
        raise SimulationSmokeError("R6c manifest SHA256 mismatch")
    if receipt_sha != expected_receipt_sha256:
        raise SimulationSmokeError("R6c receipt SHA256 mismatch")

    local_verifier = Path(preflight_module.__file__).resolve()
    if file_sha256(local_verifier) != expected_verifier_sha256:
        raise SimulationSmokeError("local preflight verifier differs from frozen R6c code")
    manifest = _load_json(manifest_file, "R6c preflight manifest")
    receipt = _load_json(receipt_file, "R6c preflight receipt")
    if receipt.get("manifest_file_sha256") != manifest_sha:
        raise SimulationSmokeError("R6c receipt does not bind the manifest file")
    if receipt.get("implementation_sha256") != expected_verifier_sha256:
        raise SimulationSmokeError("R6c receipt does not bind the frozen verifier")
    if (
        receipt.get("status") != "passed_forward_only"
        or receipt.get("authorization") != "forward_only"
        or receipt.get("environment_execution_authorized") is not False
        or receipt.get("transfer_claim_authorized") is not False
        or receipt.get("data_blind") is not True
    ):
        raise SimulationSmokeError("R6c receipt capability boundary changed")
    capability = manifest.get("capability_contract")
    if capability != {
        "fresh_inputs_allowed": False,
        "environment_step_allowed": False,
        "outcome_inputs_allowed": False,
        "execution_authorized": False,
        "transfer_claim_authorized": False,
        "maximum_authorization": "forward_only",
    }:
        raise SimulationSmokeError("R6c manifest capability boundary changed")

    # This rehashes every bound policy/body/probe artifact.  The stored receipt
    # alone is not accepted after any artifact has changed.
    recomputed = runner(manifest)
    for key, value in recomputed.items():
        if receipt.get(key) != value:
            raise SimulationSmokeError(f"R6c recomputation differs at {key}")
    return {
        "manifest": manifest,
        "receipt": receipt,
        "manifest_path": str(manifest_file),
        "receipt_path": str(receipt_file),
        "manifest_sha256": manifest_sha,
        "receipt_sha256": receipt_sha,
        "verifier_sha256": expected_verifier_sha256,
    }


def bind_development_seed_manifest(
    path: Path,
    *,
    expected_sha256: str = V7_DEVELOPMENT_SEED_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Bind one already-frozen, label-free, never-Fresh development scene."""

    manifest_path = reject_fresh_path(path, "development seed manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    digest = file_sha256(manifest_path)
    if digest != expected_sha256:
        raise SimulationSmokeError("development seed manifest SHA256 mismatch")
    value = _load_json(manifest_path, "development seed manifest")
    if (
        value.get("format") != "etsf_robotwin_v7_development_seed_manifest_v1"
        or value.get("status") != "preregistered_resolved_label_free"
        or value.get("purpose")
        != "independent_prospective_development_confirmation_never_fresh"
        or value.get("seed_registry") != "explicit_v7_prospective_development"
        or value.get("fresh_confirmation_eligible") is not False
        or value.get("task") != TASK
        or value.get("label_access_contract")
        != "reset_identity_only_no_policy_no_action_no_event_no_success_no_reward"
    ):
        raise SimulationSmokeError("development seed registry contract changed")
    rows = value.get("train")
    expected_row = {
        "seed": REQUESTED_DEVELOPMENT_SEED,
        "requested_seed": REQUESTED_DEVELOPMENT_SEED,
        "resolved_seed": EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    }
    if not isinstance(rows, list) or not rows or rows[0] != expected_row:
        raise SimulationSmokeError("fixed development smoke seed identity changed")
    return {
        "path": str(manifest_path),
        "sha256": digest,
        "seed_registry": value["seed_registry"],
        "requested_seed": REQUESTED_DEVELOPMENT_SEED,
        "expected_resolved_seed": EXPECTED_RESOLVED_DEVELOPMENT_SEED,
        "fresh_confirmation_eligible": False,
        "label_free": True,
    }


def load_r6c_mapped_candidate(
    binding: Mapping[str, Any], candidate_index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if type(candidate_index) is not int or not 0 <= candidate_index < 4:
        raise SimulationSmokeError("candidate index must be an integer in [0,3]")
    manifest = binding["manifest"]
    receipt = binding["receipt"]
    record = manifest["probe_artifacts"]["candidate_actions"]
    path = reject_fresh_path(Path(record["path"]), "R6c candidate actions")
    if file_sha256(path) != record["sha256"]:
        raise SimulationSmokeError("R6c candidate action file SHA256 changed")
    try:
        source = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SimulationSmokeError("R6c candidate actions are not safe NumPy") from exc
    mapped, mapping_receipt = adapt_aloha_source_actions_to_piper_forward_interface(
        source
    )
    if mapping_receipt != receipt.get("action_mapping_validation"):
        raise SimulationSmokeError("runtime named mapping differs from R6c receipt")
    candidate = np.ascontiguousarray(mapped[candidate_index])
    expected_hashes = receipt["candidate_validation"]["candidate_sha256"]
    if array_sha256(candidate) != expected_hashes[candidate_index]:
        raise SimulationSmokeError("selected mapped candidate hash differs from R6c")
    return candidate, {
        "candidate_index": candidate_index,
        "candidate_sha256": expected_hashes[candidate_index],
        "source_candidate_file_sha256": record["sha256"],
        "mapping_mode": mapping_receipt["mode"],
        "identity_inferred_from_equal_dimension": False,
        "angle_values_preserved": True,
        "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False,
    }


def validate_piper_step(
    action: Any,
    bounds: Sequence[Sequence[float]],
    *,
    step_index: int,
) -> dict[str, Any]:
    """Validate exactly one 14-D command without transforming or clipping it."""

    command = np.asarray(action)
    if command.shape != (ACTION_DIM,):
        raise ActionLimitError(f"step {step_index}: action shape must be [14]")
    if command.dtype.kind not in "fc" or command.dtype.kind == "c":
        raise ActionLimitError(f"step {step_index}: action must be real floating point")
    if not np.all(np.isfinite(command)):
        raise ActionLimitError(f"step {step_index}: action contains NaN or infinity")
    if len(bounds) != ACTION_DIM:
        raise ActionLimitError("Piper bounds must contain 14 named slots")
    slots: list[dict[str, Any]] = []
    for index, (slot, pair) in enumerate(zip(PIPER_ACTION_SLOTS, bounds, strict=True)):
        if len(pair) != 2:
            raise ActionLimitError(f"invalid bound for {slot.target_joint_name}")
        lower, upper = float(pair[0]), float(pair[1])
        value = float(command[index])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ActionLimitError(f"invalid bound for {slot.target_joint_name}")
        if value < lower or value > upper:
            raise ActionLimitError(
                f"step {step_index}: slot {index} {slot.target_joint_name}="
                f"{value} outside [{lower},{upper}]; command rejected, not clipped"
            )
        slots.append(
            {
                "index": index,
                "source_feature_name": slot.source_feature_name,
                "target_joint_name": slot.target_joint_name,
                "side": slot.side,
                "ordinal": slot.ordinal,
                "value": value,
                "allowed": [lower, upper],
            }
        )
    return {
        "step_index": step_index,
        "action_sha256": array_sha256(command),
        "all_14_named_slots_within_bounds": True,
        "clipping_applied": False,
        "slots": slots,
    }


def execute_validated_steps(
    env: Any,
    action_chunk: np.ndarray,
    bounds: Sequence[Sequence[float]],
    *,
    step_limit: int,
    action_converter: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    """Run a bounded prefix, checking each raw command immediately pre-step."""

    if type(step_limit) is not int or not 1 <= step_limit <= MAX_SMOKE_STEPS:
        raise SimulationSmokeError(
            f"step_limit must be in [1,{MAX_SMOKE_STEPS}] for a short smoke"
        )
    chunk = np.asarray(action_chunk)
    if chunk.shape != (50, ACTION_DIM):
        raise SimulationSmokeError("mapped action chunk must have shape [50,14]")
    step_receipts: list[dict[str, Any]] = []
    success_observed = False
    terminated_observed = False
    truncated_observed = False
    for step_index in range(step_limit):
        raw = np.ascontiguousarray(chunk[step_index])
        check = validate_piper_step(raw, bounds, step_index=step_index)
        # Conversion happens only after the fail-closed bound gate.  This module
        # contains no clipping path; the exact pre-conversion hash is retained.
        env_action = action_converter(raw.copy())
        if hasattr(env_action, "detach"):
            echoed = env_action.detach().cpu().numpy()
        else:
            echoed = np.asarray(env_action)
        if echoed.shape != (1, 1, ACTION_DIM) or not np.array_equal(
            echoed.reshape(ACTION_DIM), raw
        ):
            raise ActionLimitError(
                f"step {step_index}: action converter changed the command; "
                "silent clipping/scaling is forbidden"
            )
        _, _, terminated, truncated, infos = env.step(
            env_action, auto_reset=False
        )
        step_receipts.append(check)
        success_observed = success_observed or _scalar_bool(
            infos.get("success", [False])
        )
        terminated_observed = _scalar_bool(terminated)
        truncated_observed = _scalar_bool(truncated)
        if terminated_observed or truncated_observed:
            break
    return {
        "steps_executed": len(step_receipts),
        "step_limit": step_limit,
        "stopped_on_termination": terminated_observed,
        "stopped_on_truncation": truncated_observed,
        "success_observed_diagnostic_only": success_observed,
        "all_steps_prevalidated": True,
        "silent_clipping_possible": False,
        "step_receipts": step_receipts,
    }


def piper_environment_config(
    robotwin_root: Path,
    seeds_path: Path,
    output_parent: Path,
    *,
    step_limit: int,
) -> dict[str, Any]:
    """Return the only simulation environment configuration accepted here."""

    return {
        "env_type": "robotwin",
        "auto_reset": False,
        "ignore_terminations": False,
        "reward_coef": 1.0,
        "use_custom_reward": True,
        "use_rel_reward": True,
        "center_crop": False,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "max_steps_per_rollout_epoch": step_limit,
        "max_episode_steps": step_limit,
        "is_eval": True,
        "assets_path": str(robotwin_root),
        "seeds_path": str(seeds_path),
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": str(output_parent / "video_disabled"),
        },
        "task_config": {
            "task_name": TASK,
            "step_lim": step_limit,
            "planner_backend": "mplib",
            "render_freq": 0,
            "episode_num": 1,
            "use_seed": False,
            "save_freq": 15,
            "embodiment": ["piper", "piper", 0.6],
            "language_num": 100,
            "domain_randomization": {
                "random_background": False,
                "cluttered_table": False,
                "clean_background_rate": 1.0,
                "random_head_camera_dis": 0,
                "random_table_height": 0.0,
                "random_light": False,
                "crazy_random_light_rate": 0.0,
            },
            "camera": {
                "head_camera_type": "D435",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": True,
            },
            "data_type": {
                "rgb": True,
                "third_view": False,
                "depth": False,
                "pointcloud": False,
                "observer": False,
                "endpose": False,
                "qpos": True,
                "mesh_segmentation": False,
                "actor_segmentation": False,
            },
            "pcd_down_sample_num": 1024,
            "pcd_crop": True,
            "save_path": str(output_parent / "data_disabled"),
            "clear_cache_freq": 8,
            "collect_data": False,
            "eval_video_log": False,
        },
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_simulation_preregistration(
    *,
    binding: Mapping[str, Any],
    seed_contract: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    rlinf_root: Path,
    robotwin_root: Path,
    robotwin_code: Path,
    output: Path,
    candidate_index: int,
    step_limit: int,
) -> dict[str, Any]:
    """Build the immutable authority for one narrow simulator-only run."""

    if not 1 <= step_limit <= MAX_SMOKE_STEPS:
        raise SimulationSmokeError("invalid preregistered short-smoke step limit")
    roots = {
        "rlinf_root": str(reject_fresh_path(rlinf_root, "RLinf root")),
        "robotwin_root": str(reject_fresh_path(robotwin_root, "RoboTwin root")),
        "robotwin_code": str(reject_fresh_path(robotwin_code, "RoboTwin code")),
    }
    for role, raw_path in roots.items():
        if not Path(raw_path).is_dir():
            raise NotADirectoryError(f"{role}: {raw_path}")
    runtime_sources = {
        "rlinf_robotwin_env": (
            Path(roots["rlinf_root"]) / "rlinf/envs/robotwin/robotwin_env.py"
        ),
        "robotwin_vector_env": (
            Path(roots["robotwin_code"]) / "robotwin/envs/vector_env.py"
        ),
        "robotwin_base_task": Path(roots["robotwin_code"]) / "envs/_base_task.py",
        "robotwin_robot_controller": (
            Path(roots["robotwin_code"]) / "envs/robot/robot.py"
        ),
    }
    for role, source_path in runtime_sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"{role}: {source_path}")
    output_path = reject_fresh_path(output, "simulation smoke output")
    if output_path.exists():
        raise FileExistsError(output_path)
    base: dict[str, Any] = {
        "format": PREREGISTRATION_FORMAT,
        "status": "preregistered_simulation_only_not_executed",
        "actor_id": ACTOR_ID,
        "task": TASK,
        "source_body": "aloha",
        "target_body": "piper",
        "candidate_index": candidate_index,
        "candidate_sha256": candidate_contract["candidate_sha256"],
        "step_limit": step_limit,
        "preflight": {
            "manifest_path": binding["manifest_path"],
            "manifest_sha256": binding["manifest_sha256"],
            "receipt_path": binding["receipt_path"],
            "receipt_sha256": binding["receipt_sha256"],
            "authorization_remains": "forward_only",
            "environment_execution_authorized": False,
        },
        "development_seed": dict(seed_contract),
        "runtime_roots": roots,
        "runtime_source_artifacts": {
            role: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for role, path in runtime_sources.items()
        },
        "output": str(output_path),
        "simulation_capability": {
            "simulation_execution_authorized": True,
            "real_robot_execution_authorized": False,
            "policy_forward_authorized": False,
            "maximum_steps": MAX_SMOKE_STEPS,
            "performance_evaluation_authorized": False,
            "task_success_claim_authorized": False,
            "transfer_claim_authorized": False,
        },
        "fresh_contract": {
            "fresh_inputs_allowed": False,
            "fresh_seed_manifest_opened": False,
            "fresh_trajectory_or_label_opened": False,
            "development_registry_only": True,
        },
        "action_contract": {
            "shape_per_env_step": [1, 1, 14],
            "horizon_per_env_step": 1,
            "semantics": (
                "absolute arm joint target radians plus normalized gripper [0,1]"
            ),
            "mapping": "explicit_named_side_and_ordinal_angle_value_preserving",
            "derived_from_equal_14d_width": False,
            "bounds_checked_immediately_before_every_env_step": True,
            "nonfinite_rejected": True,
            "clipping_or_scaling_forbidden": True,
            "controller_semantics": (
                "each absolute arm qpos target is expanded by RoboTwin TOPP; "
                "normalized grippers are interpolated over that controller path"
            ),
            "RoboTwin_internal_gripper_clip_present": True,
            "internal_gripper_clip_is_noop_by_prevalidated_convex_path": True,
            "out_of_range_input_can_never_reach_internal_clip": True,
        },
        "observation_contract": {
            "state_dim": 14,
            "state_semantics": (
                "[left drive_target q1..q6, left normalized gripper, "
                "right drive_target q1..q6, right normalized gripper]"
            ),
            "state_is_measured_qpos": False,
            "center_crop": False,
            "collect_wrist_camera": True,
        },
        "language_and_seed_caveat": {
            "policy_conditioning_instruction": "move the can into the pot",
            "scene_seed_and_instruction_strictly_bound": False,
            "reason": (
                "RoboTwin SubEnv explicit-seed reset does not refresh "
                "episode_info_list"
            ),
            "guarantee": (
                "one observed instruction remains fixed only within this "
                "single-reset single-run smoke"
            ),
            "R6c_source_image_strictly_bound_to_reset_scene": False,
            "visual_state_alignment": "not_established_open_loop_interface_only",
        },
        "time_contract": {
            "reported_step_count": "policy action row count",
            "one_policy_row_may_expand_to_many_simulator_control_steps": True,
            "simulator_control_step_count_reported": False,
            "physical_duration_claimed": False,
        },
        "executor_implementation_sha256": file_sha256(Path(__file__)),
    }
    return {**base, "preregistration_sha256": canonical_sha256(base)}


def validate_simulation_preregistration(path: Path) -> dict[str, Any]:
    prereg_path = reject_fresh_path(path, "simulation preregistration")
    if not prereg_path.is_file():
        raise FileNotFoundError(prereg_path)
    value = _load_json(prereg_path, "simulation preregistration")
    recorded_sha = value.get("preregistration_sha256")
    base = {key: item for key, item in value.items() if key != "preregistration_sha256"}
    if recorded_sha != canonical_sha256(base):
        raise SimulationSmokeError("simulation preregistration logical SHA256 mismatch")
    if value.get("format") != PREREGISTRATION_FORMAT:
        raise SimulationSmokeError("unexpected simulation preregistration format")
    if value.get("status") != "preregistered_simulation_only_not_executed":
        raise SimulationSmokeError("simulation preregistration was not frozen pre-run")
    if value.get("executor_implementation_sha256") != file_sha256(Path(__file__)):
        raise SimulationSmokeError("executor code differs from preregistration")
    capability = value.get("simulation_capability")
    if capability != {
        "simulation_execution_authorized": True,
        "real_robot_execution_authorized": False,
        "policy_forward_authorized": False,
        "maximum_steps": MAX_SMOKE_STEPS,
        "performance_evaluation_authorized": False,
        "task_success_claim_authorized": False,
        "transfer_claim_authorized": False,
    }:
        raise SimulationSmokeError("simulation-only capability contract changed")
    fresh = value.get("fresh_contract")
    if fresh != {
        "fresh_inputs_allowed": False,
        "fresh_seed_manifest_opened": False,
        "fresh_trajectory_or_label_opened": False,
        "development_registry_only": True,
    }:
        raise SimulationSmokeError("Fresh=false preregistration contract changed")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()

    prereg = validate_simulation_preregistration(args.preregistration)
    step_limit = int(prereg["step_limit"])
    candidate_index = int(prereg["candidate_index"])
    output = reject_fresh_path(Path(prereg["output"]), "simulation smoke output")
    if output.exists():
        raise FileExistsError(output)
    roots = {
        role: reject_fresh_path(Path(path), role)
        for role, path in prereg["runtime_roots"].items()
    }
    for role, path in roots.items():
        if not path.is_dir():
            raise NotADirectoryError(f"{role}: {path}")

    binding = bind_r6c_preflight(
        Path(prereg["preflight"]["manifest_path"]),
        Path(prereg["preflight"]["receipt_path"]),
    )
    seed_contract = bind_development_seed_manifest(
        Path(prereg["development_seed"]["path"])
    )
    chunk, candidate_contract = load_r6c_mapped_candidate(
        binding, candidate_index
    )
    expected_prereg = build_simulation_preregistration(
        binding=binding,
        seed_contract=seed_contract,
        candidate_contract=candidate_contract,
        rlinf_root=roots["rlinf_root"],
        robotwin_root=roots["robotwin_root"],
        robotwin_code=roots["robotwin_code"],
        output=output,
        candidate_index=candidate_index,
        step_limit=step_limit,
    )
    if prereg != expected_prereg:
        raise SimulationSmokeError("preregistration differs from recomputed run contract")
    bounds = binding["receipt"]["static_contract"]["static_semantics"][
        "piper_action_bounds"
    ]

    os.environ["ASSETS_PATH"] = str(roots["robotwin_root"])
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(roots["rlinf_root"]))
    sys.path.insert(0, str(roots["robotwin_code"]))

    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise SimulationSmokeError("simulation smoke must run on the designated RTX 4090")
    seeds_path = (
        roots["rlinf_root"] / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    )
    if not seeds_path.is_file():
        raise FileNotFoundError(seeds_path)
    config = piper_environment_config(
        roots["robotwin_root"], seeds_path, output.parent, step_limit=step_limit
    )
    if config["task_config"]["embodiment"] != ["piper", "piper", 0.6]:
        raise SimulationSmokeError("internal target embodiment is not dual Piper")

    env = None
    try:
        env = RoboTwinEnv(
            cfg=OmegaConf.create(config),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            record_metrics=True,
        )
        expected_instruction = prereg["language_and_seed_caveat"][
            "policy_conditioning_instruction"
        ]
        reset_observation, _ = reset_with_explicit_instruction(
            env,
            requested_seed=REQUESTED_DEVELOPMENT_SEED,
            instruction=expected_instruction,
        )
        subenv = env.venv.envs[0]
        if not hasattr(subenv.task, "ep_num"):
            raise SimulationSmokeError("RoboTwin did not expose the resolved seed")
        resolved_seed = int(subenv.task.ep_num)
        if resolved_seed != EXPECTED_RESOLVED_DEVELOPMENT_SEED:
            raise SimulationSmokeError(
                "RoboTwin resolved seed differs from frozen development identity"
            )
        descriptions = reset_observation.get("task_descriptions")
        if descriptions is None or len(descriptions) != 1:
            raise SimulationSmokeError("RoboTwin reset lacks one observed instruction")
        observed_instruction = str(descriptions[0])
        if not observed_instruction:
            raise SimulationSmokeError("RoboTwin returned an empty instruction")
        if observed_instruction != expected_instruction:
            raise SimulationSmokeError(
                "single-run instruction differs from the R6c policy-conditioning text"
            )
        reset_interface = validate_reset_observation(reset_observation)
        reset_drive_target = reset_interface.pop("drive_target")

        def to_cuda_action(action: np.ndarray) -> Any:
            return torch.from_numpy(action.reshape(1, 1, ACTION_DIM)).to(
                device="cuda:0", dtype=torch.float32
            )

        execution = execute_validated_steps(
            env,
            chunk,
            bounds,
            step_limit=step_limit,
            action_converter=to_cuda_action,
        )
    finally:
        if env is not None:
            offload = getattr(env, "offload", None)
            if callable(offload):
                offload(clear_cache=True)
            else:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    result = {
        "format": FORMAT,
        "status": "completed_simulation_interface_smoke",
        "actor_id": ACTOR_ID,
        "source_body": "aloha",
        "target_body": "piper",
        "target_runtime": "RoboTwin_simulation_only",
        "real_robot_execution": False,
        "preflight_authorization_remains": "forward_only",
        "simulation_execution_basis": (
            "independent_frozen_simulation_only_preregistration_plus_exact_R6c_binding"
        ),
        "preregistration": {
            "path": str(reject_fresh_path(args.preregistration, "simulation preregistration")),
            "file_sha256": file_sha256(args.preregistration),
            "logical_sha256": prereg["preregistration_sha256"],
        },
        "fresh_inputs_used": False,
        "fresh_seed_manifest_opened": False,
        "fresh_trajectory_or_label_opened": False,
        "policy_forward_performed": False,
        "task_success_claimed": False,
        "transfer_claim_authorized": False,
        "performance_evaluation_authorized": False,
        "preflight": {
            "manifest_path": binding["manifest_path"],
            "manifest_sha256": binding["manifest_sha256"],
            "receipt_path": binding["receipt_path"],
            "receipt_sha256": binding["receipt_sha256"],
            "verifier_sha256": binding["verifier_sha256"],
            "stored_environment_execution_authorized": False,
        },
        "development_seed_contract": seed_contract,
        "candidate_contract": candidate_contract,
        "runtime_source_artifacts": prereg["runtime_source_artifacts"],
        "environment_contract": {
            "env_class": "rlinf.envs.robotwin.robotwin_env.RoboTwinEnv",
            "embodiment": ["piper", "piper", 0.6],
            "requested_seed": REQUESTED_DEVELOPMENT_SEED,
            "resolved_seed": EXPECTED_RESOLVED_DEVELOPMENT_SEED,
            "domain_randomization": False,
            "collect_data": False,
            "save_video": False,
            "center_crop": False,
            "collect_wrist_camera": True,
            "action_horizon_per_env_step": 1,
            "action_semantics": (
                "absolute arm qpos target radians plus normalized gripper [0,1]"
            ),
            "controller_semantics": (
                "RoboTwin TOPP arm path plus normalized gripper interpolation"
            ),
            "RoboTwin_internal_gripper_clip_present": True,
            "internal_gripper_clip_is_noop_by_prevalidated_convex_path": True,
            "observation_state_semantics": (
                "[left drive_target q1..q6,left normalized gripper,"
                "right drive_target q1..q6,right normalized gripper]"
            ),
            "observation_state_is_measured_qpos": False,
            "reset_drive_target_state_sha256": array_sha256(reset_drive_target),
            "reset_observation_interface": reset_interface,
            "scene_seed_and_instruction_strictly_bound": False,
            "observed_instruction": observed_instruction,
            "R6c_source_image_strictly_bound_to_reset_scene": False,
            "visual_state_alignment": "not_established_open_loop_interface_only",
            "instruction_guarantee": (
                "observed once and unchanged within this single-reset run only"
            ),
            "device": torch.cuda.get_device_name(0),
        },
        "time_contract": {
            "steps_executed_unit": "policy action rows",
            "one_policy_row_may_expand_to_many_simulator_control_steps": True,
            "simulator_control_step_count_reported": False,
            "physical_duration_claimed": False,
        },
        "execution": execution,
        "implementation_sha256": file_sha256(Path(__file__)),
        "interpretation": (
            "only simulator action-interface survival; not zero-shot transfer, "
            "task success, safety, or real-robot evidence"
        ),
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "steps_executed": execution["steps_executed"],
                "task_success_claimed": False,
                "transfer_claim_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
