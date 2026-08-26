#!/usr/bin/env python3
"""Collect OpenVLA-OFT RoboTwin rollouts for an offline-only ETSF shadow model.

The policy is queried once per action chunk, while each action in the chunk is
executed separately so object poses and the genuine failure terminal frame stay
aligned with environment time.  Episode files are written atomically and the
collector is safe to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


TASK = "move_can_pot"
BODY = "piper_piper_0.6"
ACTION_DIM = 14
CHUNK = 25
MAX_STEPS = 200


def model_config(model_path: Path):
    return OmegaConf.create(
        {
            "model_path": str(model_path),
            "action_dim": ACTION_DIM,
            "num_action_chunks": CHUNK,
            "add_value_head": False,
            "value_type": "action_level",
            "proprio_dim": ACTION_DIM,
            "use_proprio": True,
            "use_film": False,
            "max_prompt_length": 512,
            "unnorm_key": "move_can_pot_1k",
            "num_images_in_input": 1,
        }
    )


def environment_config(robotwin_root: Path, seeds_path: Path):
    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": True,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": MAX_STEPS,
            "max_episode_steps": MAX_STEPS,
            "is_eval": True,
            "assets_path": str(robotwin_root),
            "seeds_path": str(seeds_path),
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/etsf_openvla_collect_video",
            },
            "task_config": {
                "task_name": TASK,
                "step_lim": MAX_STEPS,
                "planner_backend": "mplib",
                "render_freq": 0,
                "episode_num": 150,
                "use_seed": False,
                "save_freq": 15,
                "embodiment": ["piper", "piper", 0.6],
                "language_num": 100,
                "domain_randomization": {
                    "random_background": True,
                    "cluttered_table": True,
                    "clean_background_rate": 0.02,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0.03,
                    "random_light": True,
                    "crazy_random_light_rate": 0.02,
                },
                "camera": {
                    "head_camera_type": "D435",
                    "wrist_camera_type": "D435",
                    "collect_head_camera": True,
                    "collect_wrist_camera": False,
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
                "save_path": "/tmp/etsf_openvla_collect_data",
                "clear_cache_freq": 8,
                "collect_data": False,
                "eval_video_log": False,
            },
        }
    )


def install_hidden_hook(model):
    capture: dict[str, torch.Tensor] = {}
    original = model._discrete_prediction

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        capture["last_hidden_states"] = result[3].detach()
        return result

    model._discrete_prediction = wrapped
    return capture


def predict(model, obs):
    actions, _ = model.predict_action_batch(
        env_obs=obs,
        do_sample=False,
        temperature=1.0,
        top_k=-1,
        calulate_values=False,
    )
    return actions


def scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0])
    return bool(np.asarray(value).reshape(-1)[0])


def pose_vector(value: Any) -> np.ndarray | None:
    try:
        pose = value.get_pose() if callable(getattr(value, "get_pose", None)) else value.pose
        vector = np.r_[np.asarray(pose.p), np.asarray(pose.q)].astype(np.float32)
        return vector if vector.shape == (7,) and np.isfinite(vector).all() else None
    except Exception:
        return None


def discover_pose_objects(task) -> tuple[list[str], list[Any]]:
    excluded = {"robot", "scene", "viewer", "engine", "renderer", "cameras"}
    found: list[tuple[str, Any]] = []
    for name, value in vars(task).items():
        if name.startswith("_") or name in excluded:
            continue
        if pose_vector(value) is not None:
            found.append((name, value))
    found.sort(key=lambda pair: pair[0])
    names = [pair[0] for pair in found]
    if "can" not in names:
        raise RuntimeError(f"cannot locate task.can pose; discovered={names}")
    return names, [pair[1] for pair in found]


def read_poses(objects: list[Any]) -> np.ndarray:
    poses = [pose_vector(value) for value in objects]
    if any(pose is None for pose in poses):
        raise RuntimeError("a tracked object stopped exposing a valid pose")
    return np.stack(poses).astype(np.float32)


def raw_rgb(task) -> np.ndarray:
    image = task.now_obs["observation"]["head_camera"]["rgb"]
    return np.asarray(image, dtype=np.uint8).copy()


def raw_state(task) -> np.ndarray:
    return np.asarray(task.now_obs["joint_action"]["vector"], dtype=np.float32).copy()


def derive_events(
    poses: np.ndarray,
    names: list[str],
    success: bool,
    event_spec: dict[str, Any],
) -> tuple[list[str], list[int], list[str], list[int]]:
    config = event_spec["calibration"][TASK]
    chain = event_spec["chains"][TASK]
    moving = names.index(str(config["moving"]))
    position = poses[:, moving, :3]
    motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    cumulative = np.cumsum(motion)
    if config["anchor"]:
        anchor = poses[:, names.index(str(config["anchor"])), :3]
        distance = np.linalg.norm(position - anchor - np.asarray(config["offset"]), axis=1)
    else:
        centers = np.asarray(config["centers"], dtype=np.float32)
        distance = np.linalg.norm(position[:, None] - centers[None], axis=2).min(1)

    raw: dict[str, int] = {"e0": 0}
    candidates = np.flatnonzero(cumulative >= float(config["delta_move"]))
    if candidates.size:
        raw["e1"] = int(candidates[0])
    candidates = np.flatnonzero(position[:, 2] >= position[0, 2] + float(config["delta_z"]))
    if candidates.size:
        raw["e2"] = int(candidates[0])
    candidates = np.flatnonzero(distance <= float(config["tau_d"]))
    if candidates.size:
        raw["e3"] = int(candidates[0])
    stationary = (distance <= float(config["tau_d"])) & (motion <= float(config["tau_motion"]))
    width = int(config["stationary_steps"])
    for index in range(len(poses) - width + 1):
        if stationary[index : index + width].all():
            raw["e4"] = index + width - 1
            break
    if success:
        raw["eK"] = len(poses) - 1

    raw_items = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    canonical_raw = dict(raw)
    if chain["merge_e1_e2"] and "e1" in raw and "e2" in raw:
        canonical_raw["e12"] = min(raw["e1"], raw["e2"])
    canonical: list[tuple[str, int]] = []
    previous = -1
    for name in chain["chain"]:
        # Keep the observed canonical subsequence rather than forcing a prefix.
        # OpenVLA's environment success can fire before an expert-demo geometric
        # settle event (e3/e4).  eK remains a valid outcome event and must not be
        # discarded merely because such an intermediate event was unobserved.
        if name not in canonical_raw or canonical_raw[name] < previous:
            continue
        canonical.append((name, int(canonical_raw[name])))
        previous = int(canonical_raw[name])
    return (
        [name for name, _ in raw_items],
        [step for _, step in raw_items],
        [name for name, _ in canonical],
        [step for _, step in canonical],
    )


def load_official_seeds(seeds_path: Path, limit: int, offset: int) -> list[int]:
    data = json.loads(seeds_path.read_text(encoding="utf-8"))
    seeds = list(data[TASK]["success_seeds"])
    selected = [int(seed) for seed in seeds[offset : offset + limit]]
    if len(selected) != limit:
        raise ValueError(f"requested {limit} seeds at offset {offset}, only found {len(selected)}")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_episode(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".hdf5.partial")
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        for key in [
            "seed", "success", "steps", "wall_seconds", "instruction", "task", "body",
            "canonical_chain_has_gap",
        ]:
            handle.attrs[key] = record[key]
        handle.attrs["failure_terminal_retained"] = not record["success"]
        handle.attrs["hidden_anchor"] = "token_before_action_block"
        handle.attrs["hidden_dim"] = 4096
        handle.attrs["action_chunk"] = CHUNK
        for key in [
            "query_steps",
            "hidden",
            "action_chunks",
            "executed_actions",
            "object_poses",
            "proprio",
            "initial_rgb",
            "terminal_rgb",
            "terminal_hidden",
            "raw_event_steps",
            "event_steps",
        ]:
            value = record[key]
            compression = "gzip" if np.asarray(value).size > 64 else None
            handle.create_dataset(key, data=value, compression=compression)
        for key in ["object_names", "raw_event_names", "event_names"]:
            handle.create_dataset(key, data=np.asarray(record[key], dtype=object), dtype=strings)
        handle.flush()
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)

    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openvla_oft.official import get_model

    args.output.mkdir(parents=True, exist_ok=True)
    episodes_dir = args.output / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    seeds = load_official_seeds(seeds_path, args.limit, args.offset)
    event_spec = json.loads(args.event_spec.read_text(encoding="utf-8"))

    device = torch.device("cuda:0")
    model = get_model(model_config(args.model_path), torch_dtype=torch.bfloat16).eval().to(device)
    capture = install_hidden_hook(model)
    env = RoboTwinEnv(
        cfg=environment_config(args.robotwin_root, seeds_path),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "collector_seed": 20260826,
        "status": "collecting",
        "task": TASK,
        "body": BODY,
        "requested_seeds": seeds,
        "model_path": str(args.model_path),
        "event_spec": str(args.event_spec),
        "event_spec_sha256": sha256(args.event_spec),
        "hidden_dim": 4096,
        "action_dim": ACTION_DIM,
        "action_chunk": CHUNK,
        "max_steps": MAX_STEPS,
        "episodes": [],
    }
    try:
        for episode_index, seed in enumerate(seeds):
            path = episodes_dir / f"episode_{episode_index:03d}_seed_{seed}.hdf5"
            if path.exists() and not args.overwrite:
                with h5py.File(path, "r") as handle:
                    manifest["episodes"].append(
                        {"index": episode_index, "seed": seed, "path": path.name,
                         "success": bool(handle.attrs["success"]), "steps": int(handle.attrs["steps"]),
                         "status": "existing"}
                    )
                continue

            started = time.time()
            obs, _ = env.reset(env_seeds=[seed])
            subenv = env.venv.envs[0]
            task = subenv.task
            names, objects = discover_pose_objects(task)
            poses = [read_poses(objects)]
            states = [raw_state(task)]
            initial_rgb = raw_rgb(task)
            query_steps: list[int] = []
            hidden: list[np.ndarray] = []
            chunks: list[np.ndarray] = []
            executed: list[np.ndarray] = []
            success = False
            done = False
            steps = 0

            while steps < MAX_STEPS and not done:
                with torch.inference_mode():
                    action_chunk = predict(model, obs)
                last_hidden = capture["last_hidden_states"]
                anchor = last_hidden[:, -model.action_dim * model.num_action_chunks - 1]
                query_steps.append(steps)
                hidden.append(anchor[0].float().cpu().numpy().astype(np.float16))
                chunks.append(action_chunk[0].detach().float().cpu().numpy().astype(np.float32))
                for action_index in range(action_chunk.shape[1]):
                    action = action_chunk[:, action_index : action_index + 1]
                    obs, _, terminated, truncated, infos = env.step(action, auto_reset=False)
                    executed.append(action[0, 0].detach().float().cpu().numpy().astype(np.float32))
                    steps += 1
                    poses.append(read_poses(objects))
                    states.append(raw_state(task))
                    success_value = infos.get("success", [False])
                    success = success or scalar_bool(success_value)
                    done = scalar_bool(terminated) or scalar_bool(truncated)
                    if done or steps >= MAX_STEPS:
                        break

            terminal_rgb = raw_rgb(task)
            # This forward pass is observational only: its action is never sent
            # to the environment.  It gives failure supervision a representation
            # of the actual terminal frame instead of the preceding chunk start.
            with torch.inference_mode():
                _ = predict(model, obs)
            terminal_state = capture["last_hidden_states"]
            terminal_anchor = terminal_state[:, -model.action_dim * model.num_action_chunks - 1]
            raw_names, raw_steps, event_names, event_steps = derive_events(
                np.stack(poses), names, success, event_spec
            )
            canonical_chain = event_spec["chains"][TASK]["chain"]
            observed_positions = [canonical_chain.index(name) for name in event_names]
            canonical_chain_has_gap = any(
                later - earlier > 1
                for earlier, later in zip(observed_positions, observed_positions[1:])
            )
            instruction = str(subenv.get_instruction())
            record = {
                "seed": seed,
                "success": success,
                "steps": steps,
                "wall_seconds": time.time() - started,
                "instruction": instruction,
                "task": TASK,
                "body": BODY,
                "canonical_chain_has_gap": canonical_chain_has_gap,
                "query_steps": np.asarray(query_steps, dtype=np.int32),
                "hidden": np.stack(hidden),
                "action_chunks": np.stack(chunks),
                "executed_actions": np.stack(executed),
                "object_poses": np.stack(poses),
                "object_names": names,
                "proprio": np.stack(states),
                "initial_rgb": initial_rgb,
                "terminal_rgb": terminal_rgb,
                "terminal_hidden": terminal_anchor[0].float().cpu().numpy().astype(np.float16),
                "raw_event_names": raw_names,
                "raw_event_steps": np.asarray(raw_steps, dtype=np.int32),
                "event_names": event_names,
                "event_steps": np.asarray(event_steps, dtype=np.int32),
            }
            save_episode(path, record)
            item = {
                "index": episode_index,
                "seed": seed,
                "path": path.name,
                "success": success,
                "steps": steps,
                "events": dict(zip(event_names, event_steps)),
                "canonical_chain_has_gap": canonical_chain_has_gap,
                "wall_seconds": record["wall_seconds"],
                "status": "collected",
            }
            manifest["episodes"].append(item)
            manifest["completed"] = len(manifest["episodes"])
            manifest["successes"] = sum(entry["success"] for entry in manifest["episodes"])
            atomic_json(args.output / "manifest.json", manifest)
            print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)
    finally:
        env.venv.close(clear_cache=False)

    manifest["status"] = "complete"
    manifest["completed"] = len(manifest["episodes"])
    manifest["successes"] = sum(entry["success"] for entry in manifest["episodes"])
    manifest["failures"] = manifest["completed"] - manifest["successes"]
    atomic_json(args.output / "manifest.json", manifest)
    print("COLLECTION_COMPLETE=" + json.dumps(
        {key: manifest[key] for key in ["completed", "successes", "failures"]}, sort_keys=True
    ), flush=True)


if __name__ == "__main__":
    main()
