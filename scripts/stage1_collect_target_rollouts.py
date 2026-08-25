#!/usr/bin/env python3
"""Collect unfiltered target-embodiment rollouts with images and object poses."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml


PAIR_DISTANCE = {"piper": 0.6, "ur5-wsg": 0.8}


def append_csv(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def object_handles(task: Any) -> tuple[list[str], list[Any]]:
    names: list[str] = []
    handles: list[Any] = []
    seen: set[int] = set()
    excluded = {"robot", "viewer", "scene", "engine", "renderer", "cameras"}
    for attr, value in sorted(vars(task).items()):
        if attr in excluded or attr.startswith("_") or id(value) in seen:
            continue
        get_pose = getattr(value, "get_pose", None)
        if not callable(get_pose):
            continue
        try:
            pose = get_pose()
            if len(pose.p) != 3 or len(pose.q) != 4:
                continue
        except Exception:
            continue
        seen.add(id(value))
        names.append(attr)
        handles.append(value)
    if not handles:
        raise RuntimeError("no task objects with get_pose() found")
    return names, handles


def poses(handles: list[Any]) -> np.ndarray:
    values = []
    for handle in handles:
        pose = handle.get_pose()
        values.append(np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]))
    return np.asarray(values, dtype=np.float32)


def encode_rgb(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    ok, payload = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return payload


def write_episode(
    path: Path,
    *,
    image_frames: dict[str, list[np.ndarray]],
    pose_frames: list[np.ndarray],
    object_names: list[str],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".hdf5", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with h5py.File(tmp_path, "w") as handle:
            for key, value in metadata.items():
                handle.attrs[key] = value
            handle.create_dataset("object_poses", data=np.stack(pose_frames), compression="gzip")
            handle.create_dataset("object_names", data=np.asarray(object_names, dtype="S"))
            group = handle.create_group("images")
            dtype = h5py.vlen_dtype(np.dtype("uint8"))
            for camera, frames in image_frames.items():
                dataset = group.create_dataset(camera, shape=(len(frames),), dtype=dtype)
                for index, frame in enumerate(frames):
                    dataset[index] = frame
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--embodiment", choices=sorted(PAIR_DISTANCE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=200)
    args_cli = parser.parse_args()
    args_cli.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args_cli.output.parent.parent / "target_rollout_manifest.csv"

    existing = sorted(args_cli.output.glob("episode_*.hdf5"))
    rollout_index = len(existing)
    if rollout_index >= args_cli.rollouts:
        print(f"TARGET_READY task={args_cli.task} body={args_cli.embodiment} existing={rollout_index}")
        return

    os.chdir(args_cli.repo)
    sys.path.insert(0, str(args_cli.repo))
    from envs import CONFIGS_PATH  # pylint: disable=import-outside-toplevel
    from scripts.collect_data import class_decorator, get_embodiment_config  # pylint: disable=import-outside-toplevel

    with open(Path(CONFIGS_PATH) / "stage0_gate1.yml", encoding="utf-8") as handle:
        base = yaml.load(handle.read(), Loader=yaml.FullLoader)
    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", encoding="utf-8") as handle:
        body_config = yaml.load(handle.read(), Loader=yaml.FullLoader)
    robot_file = body_config[args_cli.embodiment]["file_path"]
    body_args = get_embodiment_config(robot_file)
    run_args = dict(base)
    run_args.update(
        {
            "task_name": args_cli.task,
            "task_config": "stage1_target_rollout",
            "embodiment": [args_cli.embodiment, args_cli.embodiment, PAIR_DISTANCE[args_cli.embodiment]],
            "embodiment_name": args_cli.embodiment,
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": body_args,
            "right_embodiment_config": body_args,
            "dual_arm_embodied": False,
            "embodiment_dis": PAIR_DISTANCE[args_cli.embodiment],
            "need_plan": True,
            "save_data": False,
            "collect_data": False,
            "render_freq": 0,
            "eval_video_log": False,
            "save_path": str(args_cli.output / "scratch"),
        }
    )
    run_args["data_type"] = {
        "rgb": True,
        "third_view": False,
        "depth": False,
        "pointcloud": False,
        "observer": False,
        "endpose": False,
        "qpos": False,
        "mesh_segmentation": False,
        "actor_segmentation": False,
    }

    task = class_decorator(args_cli.task)
    for seed in range(args_cli.seed_start, args_cli.seed_start + args_cli.max_seeds):
        if rollout_index >= args_cli.rollouts:
            break
        started = time.monotonic()
        image_frames: dict[str, list[np.ndarray]] = {}
        pose_frames: list[np.ndarray] = []
        object_names: list[str] = []
        setup_ok = False
        success = False
        error = ""
        try:
            task.setup_demo(now_ep_num=rollout_index, seed=seed, **run_args)
            setup_ok = True
            object_names, handles = object_handles(task)

            def capture() -> None:
                observation = task.get_obs()["observation"]
                available = {
                    name: value["rgb"]
                    for name, value in observation.items()
                    if isinstance(value, dict) and "rgb" in value
                }
                if len(available) < 3:
                    raise RuntimeError(f"expected >=3 RGB cameras, got {list(available)}")
                if not image_frames:
                    for camera in sorted(available):
                        image_frames[camera] = []
                if set(available) != set(image_frames):
                    raise RuntimeError("camera set changed during rollout")
                for camera, rgb in available.items():
                    image_frames[camera].append(encode_rgb(rgb))
                pose_frames.append(poses(handles))

            task._take_picture = capture
            try:
                task.play_once()
                success = bool(task.plan_success and task.check_success())
            except Exception as exc:
                error = f"play {type(exc).__name__}: {exc}"
                success = False
            if not pose_frames:
                capture()
            output_path = args_cli.output / f"episode_{rollout_index:06d}.hdf5"
            write_episode(
                output_path,
                image_frames=image_frames,
                pose_frames=pose_frames,
                object_names=object_names,
                metadata={
                    "task": args_cli.task,
                    "embodiment": args_cli.embodiment,
                    "rollout_index": rollout_index,
                    "seed": seed,
                    "success": int(success),
                    "total_steps": len(pose_frames),
                    "sim_error": error,
                },
            )
        except Exception as exc:
            error = f"setup {type(exc).__name__}: {exc}"
        finally:
            try:
                task.close_env(clear_cache=False)
            except Exception as exc:
                if not error:
                    error = f"close_env {type(exc).__name__}: {exc}"

        valid = setup_ok and bool(pose_frames)
        row = {
            "task": args_cli.task,
            "embodiment": args_cli.embodiment,
            "rollout_index": rollout_index if valid else -1,
            "seed": seed,
            "valid_rollout": int(valid),
            "success": int(success),
            "total_steps": len(pose_frames),
            "cameras": json.dumps(sorted(image_frames)),
            "objects": json.dumps(object_names),
            "wall_seconds": round(time.monotonic() - started, 4),
            "path": str(args_cli.output / f"episode_{rollout_index:06d}.hdf5") if valid else "",
            "error": error,
        }
        append_csv(manifest_path, row)
        print("TARGET_ROLLOUT=" + json.dumps(row), flush=True)
        if valid:
            rollout_index += 1
        if error or not setup_ok:
            task = class_decorator(args_cli.task)

    if rollout_index < args_cli.rollouts:
        raise SystemExit(
            f"only collected {rollout_index}/{args_cli.rollouts} valid rollouts for "
            f"{args_cli.task}/{args_cli.embodiment}"
        )


if __name__ == "__main__":
    main()
