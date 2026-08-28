#!/usr/bin/env python3
"""Record one RoboTwin scripted-expert rollout from a fixed camera.

This utility is deliberately policy-neutral: it records ``task.play_once()``
for visual comparison across embodiments and writes a small JSON manifest next
to the video.  It does not claim to evaluate OpenVLA or ETSF online control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PAIR_DISTANCE = {
    "piper": 0.6,
    "ARX-X5": 0.6,
    "ur5-wsg": 0.8,
    "franka-panda": 0.8,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--fps", type=float, default=15.0)
    args_cli = parser.parse_args()

    args_cli.output = args_cli.output.resolve()
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args_cli.repo)
    sys.path.insert(0, str(args_cli.repo))

    from envs import CONFIGS_PATH  # pylint: disable=import-outside-toplevel
    print("RECORDER_STAGE=import_envs_ok", flush=True)
    from scripts.collect_data import (  # pylint: disable=import-outside-toplevel
        class_decorator,
        get_embodiment_config,
    )
    print("RECORDER_STAGE=import_collect_data_ok", flush=True)
    # RoboTwin imports PyTorch3D's compiled farthest-point sampler even for
    # RGB-only jobs.  If that optional extension is ABI-incompatible, provide
    # a deterministic uniform sampler so setup-time observations cannot call
    # the module's hard-exit fallback.  The recorder disables point clouds
    # after setup, so this is only a compatibility guard.
    from envs.camera import camera as camera_module  # pylint: disable=import-outside-toplevel

    try:
        import pytorch3d.ops  # type: ignore  # pylint: disable=import-outside-toplevel,unused-import
    except Exception:
        import torch  # pylint: disable=import-outside-toplevel

        def rgb_only_fps(
            points: np.ndarray,
            num_points: int = 1024,
            use_cuda: bool = True,  # noqa: ARG001
        ) -> tuple[np.ndarray, torch.Tensor]:
            count = min(len(points), num_points)
            indices = np.linspace(0, max(0, len(points) - 1), count, dtype=np.int64)
            return points[indices], torch.from_numpy(indices[None])

        camera_module.fps = rgb_only_fps
        print("RECORDER_STAGE=installed_rgb_only_fps", flush=True)

    with open(Path(CONFIGS_PATH) / "stage0_gate1.yml", encoding="utf-8") as handle:
        base = yaml.load(handle.read(), Loader=yaml.FullLoader)
    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", encoding="utf-8") as handle:
        body_config = yaml.load(handle.read(), Loader=yaml.FullLoader)

    embodiment = args_cli.embodiment
    robot_file = body_config[embodiment]["file_path"]
    body_args = get_embodiment_config(robot_file)
    dual_arm_embodied = embodiment not in PAIR_DISTANCE
    configured_embodiment: list[Any]
    if dual_arm_embodied:
        configured_embodiment = [embodiment]
    else:
        configured_embodiment = [
            embodiment,
            embodiment,
            PAIR_DISTANCE[embodiment],
        ]

    scratch = args_cli.output.parent / f"scratch_{embodiment.replace('-', '_')}"
    scratch.mkdir(parents=True, exist_ok=True)
    run_args = dict(base)
    run_args.update(
        {
            "task_name": args_cli.task,
            "task_config": "stage0_gate1",
            "embodiment": configured_embodiment,
            "embodiment_name": embodiment,
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": body_args,
            "right_embodiment_config": body_args,
            "dual_arm_embodied": dual_arm_embodied,
            "need_plan": True,
            "save_data": False,
            "collect_data": False,
            "render_freq": 0,
            "eval_video_log": False,
            "save_path": str(scratch),
        }
    )
    if not dual_arm_embodied:
        run_args["embodiment_dis"] = PAIR_DISTANCE[embodiment]
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

    print("RECORDER_STAGE=config_ready", flush=True)
    task = class_decorator(args_cli.task)
    print("RECORDER_STAGE=task_class_ready", flush=True)
    frames: list[np.ndarray] = []
    started = time.monotonic()
    success = False
    error = ""
    try:
        print("RECORDER_STAGE=setup_begin", flush=True)
        task.setup_demo(now_ep_num=0, seed=args_cli.seed, **run_args)
        print("RECORDER_STAGE=setup_done", flush=True)
        # Some RoboTwin task configs overwrite data_type during setup.  Keep
        # point clouds disabled here: this recorder needs RGB only and should
        # not depend on the optional compiled PyTorch3D FPS operator.
        task.data_type = dict(run_args["data_type"])

        def capture() -> None:
            task._update_render()
            task.cameras.update_picture()
            images = task.cameras.get_rgb()
            if args_cli.camera not in images:
                raise KeyError(
                    f"camera {args_cli.camera!r} missing; available={sorted(images)}"
                )
            rgb = np.asarray(images[args_cli.camera]["rgb"])
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            frames.append(np.ascontiguousarray(rgb))

        capture()
        task._take_picture = capture
        task.play_once()
        success = bool(task.plan_success and task.check_success())
        capture()
    except Exception as exc:  # preserve partial video for diagnosis
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            task.close_env(clear_cache=False)
        except Exception as exc:
            if not error:
                error = f"close_env {type(exc).__name__}: {exc}"

    if not frames:
        raise RuntimeError(f"no frames captured: {error}")

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(args_cli.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args_cli.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {args_cli.output}")
    for rgb in frames:
        if rgb.shape[:2] != (height, width):
            raise RuntimeError(f"frame shape changed: {rgb.shape[:2]} != {(height, width)}")
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    manifest = {
        "task": args_cli.task,
        "embodiment": embodiment,
        "seed": args_cli.seed,
        "camera": args_cli.camera,
        "fps": args_cli.fps,
        "frames": len(frames),
        "width": width,
        "height": height,
        "success": success,
        "error": error,
        "wall_seconds": round(time.monotonic() - started, 4),
        "video": str(args_cli.output),
        "controller": "RoboTwin scripted expert task.play_once()",
    }
    args_cli.output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("ROBOTWIN_VIDEO=" + json.dumps(manifest, ensure_ascii=False), flush=True)
    if error or not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
