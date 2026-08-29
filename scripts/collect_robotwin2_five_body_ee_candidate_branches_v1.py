#!/usr/bin/env python3
"""Collect real four-candidate RoboTwin2 branches for the five-body head.

The collector is intentionally operational rather than a data-contract tool:
it loads one frozen 16-D EE SmolVLA actor, creates four fixed-flow-noise action
candidates at several fixed query indices, executes each candidate after an
identical reset/replayed baseline prefix, and writes one four-row canonical
NPZ per decision.  Events, object effects and success are derived from the
simulator trajectory, never from the public expert archive.

Run this program only on the remote 4090 with the public RoboTwin simulator.
One invocation collects both clean and randomized groups for one embodiment;
the five embodiments are run sequentially because they share one GPU.
"""

from __future__ import annotations

import argparse
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


FORMAT = "etsf_robotwin2_five_body_ee_candidate_branches_v1"
MANIFEST_FORMAT = "etsf_robotwin2_canonical_transition_manifest_v1"
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
DEFAULT_INSTRUCTION = "Move the can to the side of the pot."
CANONICAL_EVENTS = ("e0", "e12", "e3", "e4", "eK")
EVENT_TO_ID = {name: index for index, name in enumerate(CANONICAL_EVENTS)}
STATE_SCHEMA = canonical_adapter.STATE_SCHEMA
ACTION_SCHEMA = canonical_adapter.ACTION_SCHEMA
BODY_EMBODIMENT = {
    "aloha-agilex": ["aloha-agilex"],
    "arx-x5": ["ARX-X5", "ARX-X5", 0.6],
    "franka": ["franka-panda", "franka-panda", 0.8],
    "piper": ["piper", "piper", 0.6],
    "ur5": ["ur5-wsg", "ur5-wsg", 0.8],
}


class BranchCollectionError(RuntimeError):
    """The actor, reset, replay or canonical output is invalid."""


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
    seed = int(
        (
            20260903
            + int(scene_seed) * 1_000_003
            + int(query_index) * 10_007
            + int(candidate_index) * 101
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
    )


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
    return np.stack(values).astype(np.float32)


def _goal_vector(
    poses: np.ndarray,
    names: Sequence[str],
    step: int,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    moving_index = list(names).index(str(calibration["moving"]))
    moving = poses[step, moving_index, :3]
    anchor_name = str(calibration.get("anchor", ""))
    if anchor_name:
        anchor_index = list(names).index(anchor_name)
        goal = poses[step, anchor_index, :3] + np.asarray(
            calibration.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32
        )
    else:
        centers = np.asarray(calibration["centers"], dtype=np.float32)
        goal = centers[np.linalg.norm(centers - moving[None], axis=1).argmin()]
    return moving.astype(np.float32), (goal - moving).astype(np.float32)


def derive_predicates_and_events(
    poses: np.ndarray,
    names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    moving_index = list(names).index(str(calibration["moving"]))
    position = poses[:, moving_index, :3]
    displacement = np.linalg.norm(position - position[0], axis=1)
    moved = displacement >= float(calibration["delta_move"])
    lifted = position[:, 2] >= position[0, 2] + float(calibration["delta_z"])
    near = np.asarray(
        [
            np.linalg.norm(_goal_vector(poses, names, step, calibration)[1])
            <= float(calibration["tau_d"])
            for step in range(len(poses))
        ],
        dtype=bool,
    )
    motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    instant_stationary = near & (motion <= float(calibration["tau_motion"]))
    stationary = np.zeros(len(poses), dtype=bool)
    width = int(calibration["stationary_steps"])
    for step in range(width - 1, len(poses)):
        stationary[step] = bool(instant_stationary[step - width + 1 : step + 1].all())
    succeeded = np.zeros(len(poses), dtype=bool)
    if success:
        succeeded[-1] = True
    predicates = np.stack((moved, lifted, near, stationary, succeeded), axis=-1)
    events = np.full(len(poses), EVENT_TO_ID["e0"], dtype=np.int64)
    events[moved | lifted] = EVENT_TO_ID["e12"]
    events[near] = EVENT_TO_ID["e3"]
    events[stationary] = EVENT_TO_ID["e4"]
    events[succeeded] = EVENT_TO_ID["eK"]
    return predicates.astype(np.float32), events


def _image_chw(value: Any) -> torch.Tensor:
    image = torch.as_tensor(np.asarray(value))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise BranchCollectionError(f"camera must be HWC RGB, got {tuple(image.shape)}")
    return image.permute(2, 0, 1).contiguous().float().div(255.0)


def current_ee_action16(task: Any) -> np.ndarray:
    left_pose = np.asarray(task.robot.get_left_tcp_pose(), dtype=np.float32)
    right_pose = np.asarray(task.robot.get_right_tcp_pose(), dtype=np.float32)
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
    return task


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
    prefix_chunks: list[np.ndarray] = []
    try:
        names, objects = discover_pose_objects(task, required_pose_names)
        trajectory = [read_poses(objects)]
        for query_index in range(root_query):
            if _episode_done(task, max_steps):
                return None
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
            prefix_chunks.append(chunk[:action_exec_steps].copy())
            for action in chunk[:action_exec_steps]:
                if _episode_done(task, max_steps):
                    break
                task.take_action(action, action_type="ee")
                trajectory.append(read_poses(objects))
        if _episode_done(task, max_steps):
            return None
        current = current_ee_action16(task)
        root_pose = read_poses(objects)
        candidates = generate_candidates(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            instruction=instruction,
            scene_seed=seed,
            query_index=root_query,
            candidate_count=CANDIDATE_COUNT,
            device=device,
        )
        return {
            "object_names": names,
            "root_object_poses": root_pose,
            "root_ee_action": current,
            "prefix_chunks": prefix_chunks,
            "prefix_trajectory": np.stack(trajectory),
            "candidates": candidates,
        }
    finally:
        task.close_env(clear_cache=False)


def _evaluate_candidate(
    *,
    task_class: Any,
    args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    seed: int,
    root_query: int,
    prefix_chunks: Sequence[np.ndarray],
    candidate: np.ndarray,
    action_exec_steps: int,
    max_steps: int,
    required_pose_names: set[str],
    expected_names: Sequence[str],
    expected_root_pose: np.ndarray,
    expected_root_ee: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    task = _new_task(task_class, args, seed, instruction)
    try:
        names, objects = discover_pose_objects(task, required_pose_names)
        if list(names) != list(expected_names):
            raise BranchCollectionError("tracked object registry changed during reset")
        trajectory = [read_poses(objects)]
        for chunk in prefix_chunks:
            for action in chunk:
                if _episode_done(task, max_steps):
                    raise BranchCollectionError("baseline prefix terminated during replay")
                task.take_action(action, action_type="ee")
                trajectory.append(read_poses(objects))
        root_step = len(trajectory) - 1
        if not np.allclose(trajectory[-1], expected_root_pose, atol=2e-5, rtol=0.0):
            raise BranchCollectionError("object state changed across identical prefix replay")
        if not np.allclose(current_ee_action16(task), expected_root_ee, atol=2e-5, rtol=0.0):
            raise BranchCollectionError("robot state changed across identical prefix replay")

        first_executed = 0
        branch_error = None
        try:
            for action in candidate[:action_exec_steps]:
                if _episode_done(task, max_steps):
                    break
                task.take_action(action, action_type="ee")
                first_executed += 1
                trajectory.append(read_poses(objects))
        except Exception as error:  # an infeasible candidate is a real failed branch
            branch_error = f"{type(error).__name__}: {error}"
        post_step = len(trajectory) - 1
        query_index = root_query + 1
        while branch_error is None and not _episode_done(task, max_steps):
            try:
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
                    trajectory.append(read_poses(objects))
                query_index += 1
            except Exception as error:
                branch_error = f"{type(error).__name__}: {error}"
                break
        success = bool(getattr(task, "eval_success", False))
        if not success:
            try:
                success = bool(task.check_success())
            except Exception:
                success = False
        return {
            "trajectory": np.stack(trajectory),
            "root_step": root_step,
            "post_step": post_step,
            "first_executed": first_executed,
            "success": success,
            "branch_error": branch_error,
        }
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
    dt: float,
) -> dict[str, np.ndarray]:
    names = list(root["object_names"])
    moving_index = names.index(str(calibration["moving"]))
    prefix = np.asarray(root["prefix_trajectory"], dtype=np.float32)
    initial_moving = prefix[0, moving_index, :3]
    prefix_predicates, prefix_events = derive_predicates_and_events(
        prefix, names, False, calibration
    )
    current_event = int(prefix_events[-1])
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
    success = []
    recovery = []
    recovery_mask = []
    object_delta = []
    object_delta_mask = []
    for candidate, outcome in zip(root["candidates"], outcomes):
        action = canonical_action_chunk(root["root_ee_action"], candidate)
        mask = np.arange(horizon) < int(outcome["first_executed"])
        trajectory = np.asarray(outcome["trajectory"], dtype=np.float32)
        predicates, events = derive_predicates_and_events(
            trajectory, names, bool(outcome["success"]), calibration
        )
        root_step = int(outcome["root_step"])
        post_step = int(outcome["post_step"])
        if int(events[root_step]) != current_event:
            raise BranchCollectionError("candidate replay changed the root event")
        future = np.flatnonzero(events[root_step + 1 :] != current_event)
        if len(future):
            boundary = root_step + 1 + int(future[0])
            next_id = int(events[boundary])
            duration_steps = boundary - root_step
            observed = True
        else:
            next_id = current_event
            duration_steps = max(len(events) - 1 - root_step, 1)
            observed = False
        regressed = int(events[post_step]) < current_event
        recovered = bool(regressed and np.any(events[post_step + 1 :] >= current_event))
        moving_start, relative_start = _goal_vector(
            trajectory, names, root_step, calibration
        )
        moving_post, relative_post = _goal_vector(
            trajectory, names, post_step, calibration
        )
        actions.append(action)
        masks.append(mask)
        post_event.append(int(events[post_step]))
        next_event.append(next_id)
        next_mask.append(observed)
        duration.append(max(float(duration_steps) * dt, dt))
        duration_observed.append(observed)
        success.append(bool(outcome["success"]))
        recovery.append(recovered)
        recovery_mask.append(regressed)
        object_delta.append(np.r_[moving_post - moving_start, relative_post - relative_start])
        object_delta_mask.append(bool(outcome["first_executed"]))

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
        "duration_mask": np.ones(count, dtype=np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "success_mask": np.ones(count, dtype=np.float32),
        "recovery": np.asarray(recovery, dtype=np.float32),
        "recovery_mask": np.asarray(recovery_mask, dtype=np.float32),
        "object_delta": np.asarray(object_delta, dtype=np.float32),
        "object_delta_mask": np.asarray(object_delta_mask, dtype=np.float32),
        "candidate_index": np.arange(count, dtype=np.int64),
        "dt": np.asarray(
            [max(int(outcome["first_executed"]), 1) * dt for outcome in outcomes],
            dtype=np.float32,
        ),
    }
    if arrays["actions"].shape != (count, horizon, CANONICAL_ACTION_DIM):
        raise BranchCollectionError("canonical group action shape changed")
    if arrays["state"].shape != (count, STATE_DIM):
        raise BranchCollectionError("canonical group state shape changed")
    if arrays["object_delta"].shape != (count, OBJECT_DELTA_DIM):
        raise BranchCollectionError("canonical object effect shape changed")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise BranchCollectionError("canonical group contains non-finite values")
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", choices=BODIES, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seed-start", type=int, default=2026081000)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--root-query-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--action-exec-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise BranchCollectionError("real branch collection requires remote RTX 4090 CUDA")
    if args.seed_count <= 0 or args.action_exec_steps <= 0 or args.fps <= 0:
        raise BranchCollectionError("seed-count/action-exec-steps/fps must be positive")
    if not args.root_query_indices or min(args.root_query_indices) < 0:
        raise BranchCollectionError("root query indices must be non-negative")
    if len(set(args.root_query_indices)) != len(args.root_query_indices):
        raise BranchCollectionError("root query indices must be unique")
    for path in (
        args.actor_checkpoint,
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

    event_spec = json.loads(args.event_spec.read_text(encoding="utf-8"))
    calibration = event_spec["calibration"][TASK]
    required_pose_names = {str(calibration["moving"])}
    anchor_name = str(calibration.get("anchor", "")).strip()
    if anchor_name:
        required_pose_names.add(anchor_name)
    adapter_source = Path(inspect.getsourcefile(canonical_adapter) or "")
    adapter_sha = sha256_file(adapter_source)
    dt = 1.0 / float(args.fps)
    output = args.output.expanduser().resolve()
    groups_dir = output / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        unsigned = dict(manifest)
        unsigned.pop("logical_sha256", None)
        if (
            manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("body") != args.body
            or manifest.get("actor_checkpoint") != str(args.actor_checkpoint.resolve())
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
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "actor_checkpoint_tree_or_file_sha256_recorded_separately": True,
            "candidate_count": CANDIDATE_COUNT,
            "candidate_zero_is_actor_baseline": True,
            "same_ordered_candidate_set_for_baseline_and_etsf": True,
            "root_query_indices": sorted(args.root_query_indices),
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
            "event_spec_sha256": sha256_file(args.event_spec),
            "groups": [],
        }

    existing_ids = {str(item["group_id"]) for item in manifest["groups"]}
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    for condition in args.conditions:
        task_args = _load_task_args(args.robotwin_root, args.body, condition)
        task_args["step_lim"] = args.max_steps
        for seed in seeds:
            for root_query in sorted(args.root_query_indices):
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
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        instruction=args.instruction,
                        seed=seed,
                        root_query=root_query,
                        prefix_chunks=root["prefix_chunks"],
                        candidate=candidate,
                        action_exec_steps=args.action_exec_steps,
                        max_steps=args.max_steps,
                        required_pose_names=required_pose_names,
                        expected_names=root["object_names"],
                        expected_root_pose=root["root_object_poses"],
                        expected_root_ee=root["root_ee_action"],
                        device=device,
                    )
                    for candidate in root["candidates"]
                ]
                arrays = materialize_group(
                    root=root,
                    outcomes=outcomes,
                    calibration=calibration,
                    action_exec_steps=args.action_exec_steps,
                    dt=dt,
                )
                filename = f"{condition}_seed_{seed}_query_{root_query}.npz"
                path = groups_dir / filename
                atomic_npz(path, arrays)
                item = {
                    "group_id": group_id,
                    "condition": condition,
                    "requested_seed": int(seed),
                    "root_query_index": int(root_query),
                    "path": f"groups/{filename}",
                    "sha256": sha256_file(path),
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
