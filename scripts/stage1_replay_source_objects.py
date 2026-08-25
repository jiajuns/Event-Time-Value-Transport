#!/usr/bin/env python3
"""Replay official successful source plans and record object-only pose sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


PAIR_DISTANCE = {"ARX-X5": 0.6}
ALLOWED_GLOBALS = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"forbidden global: {module}.{name}")
        return ALLOWED_GLOBALS[(module, name)]


def safe_load(path: Path) -> Any:
    with path.open("rb") as handle:
        return RestrictedUnpickler(handle).load()


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
        raise RuntimeError("no task object handles with get_pose() found")
    return names, handles


def pose_row(handles: list[Any]) -> np.ndarray:
    rows = []
    for handle in handles:
        pose = handle.get_pose()
        rows.append(np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]))
    return np.asarray(rows, dtype=np.float32)


def find_episode_file(directory: Path, episode: int, suffix: str) -> Path:
    candidates = [
        directory / f"episode{episode}.{suffix}",
        directory / f"episode_{episode}.{suffix}",
        directory / f"episode_{episode:07d}.{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"episode {episode} .{suffix} below {directory}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    args_cli = parser.parse_args()
    args_cli.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args_cli.output.parent / "source_replay_manifest.csv"

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
    dual = args_cli.embodiment not in PAIR_DISTANCE
    run_args = dict(base)
    run_args.update(
        {
            "task_name": args_cli.task,
            "task_config": "stage1_source_replay",
            "embodiment": [args_cli.embodiment]
            if dual
            else [args_cli.embodiment, args_cli.embodiment, PAIR_DISTANCE[args_cli.embodiment]],
            "embodiment_name": args_cli.embodiment,
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": body_args,
            "right_embodiment_config": body_args,
            "dual_arm_embodied": dual,
            "need_plan": False,
            "save_data": False,
            "collect_data": False,
            "render_freq": 0,
            "eval_video_log": False,
            "save_path": str(args_cli.output / "scratch"),
        }
    )
    if not dual:
        run_args["embodiment_dis"] = PAIR_DISTANCE[args_cli.embodiment]

    seeds = [int(x) for x in (args_cli.source / "seed.txt").read_text().split()]
    task = class_decorator(args_cli.task)
    for episode in range(args_cli.start, min(args_cli.end, len(seeds))):
        target = args_cli.output / f"episode_{episode:06d}.npz"
        if target.exists():
            continue
        started = time.monotonic()
        status = "error"
        error = ""
        success = False
        frame_count = 0
        expected_frames = -1
        object_names: list[str] = []
        frames: list[np.ndarray] = []
        try:
            task.setup_demo(now_ep_num=episode, seed=seeds[episode], **run_args)
            paths = safe_load(find_episode_file(args_cli.source / "_traj_data", episode, "pkl"))
            task.set_path_lst(
                {
                    "need_plan": False,
                    "left_joint_path": paths["left_joint_path"],
                    "right_joint_path": paths["right_joint_path"],
                }
            )
            object_names, handles = object_handles(task)

            def capture() -> None:
                frames.append(pose_row(handles))

            task._take_picture = capture
            task.play_once()
            success = bool(task.plan_success and task.check_success())
            status = "success" if success else "failed"
            frame_count = len(frames)
            hdf5_path = find_episode_file(args_cli.source / "data", episode, "hdf5")
            with h5py.File(hdf5_path, "r") as handle:
                expected_frames = int(handle["joint_action/vector"].shape[0])
            if frame_count != expected_frames:
                raise RuntimeError(f"frame mismatch: replay={frame_count}, hdf5={expected_frames}")
            if not success:
                raise RuntimeError("official successful source replay did not satisfy task success")
            np.savez_compressed(
                target,
                poses=np.stack(frames),
                object_names=np.asarray(object_names),
                task=args_cli.task,
                embodiment=args_cli.embodiment,
                episode=episode,
                seed=seeds[episode],
                success=success,
                hdf5_path=str(hdf5_path),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                task.close_env(clear_cache=False)
            except Exception as exc:
                if not error:
                    error = f"close_env {type(exc).__name__}: {exc}"
        row = {
            "task": args_cli.task,
            "embodiment": args_cli.embodiment,
            "episode": episode,
            "seed": seeds[episode],
            "status": status if not error else "error",
            "success": int(success),
            "frames": frame_count,
            "expected_frames": expected_frames,
            "objects": json.dumps(object_names),
            "wall_seconds": round(time.monotonic() - started, 4),
            "path": str(target) if target.exists() else "",
            "error": error,
        }
        append_csv(manifest_path, row)
        print("SOURCE_REPLAY=" + json.dumps(row), flush=True)
        if error:
            task = class_decorator(args_cli.task)


if __name__ == "__main__":
    main()
