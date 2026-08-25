#!/usr/bin/env python3
"""Measure expert trajectory horizons for matched seeds across RoboTwin bodies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml


SINGLE_ARM_PAIR_DISTANCE = {
    "piper": 0.6,
    "ARX-X5": 0.6,
    "ur5-wsg": 0.8,
    "franka-panda": 0.8,
}


def motion_steps(task: Any) -> int:
    left = [x for x in task.left_joint_path if "position" in x]
    right = [x for x in task.right_joint_path if "position" in x]
    if left and right:
        if len(left) != len(right):
            # This fallback is only used for reporting raw planner work. The
            # counted saved-control horizon below remains exact.
            return sum(x["position"].shape[0] for x in left + right)
        return sum(
            max(a["position"].shape[0], b["position"].shape[0])
            for a, b in zip(left, right)
        )
    return sum(x["position"].shape[0] for x in (left or right))


def append_row(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", default="stage0_gate1")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--embodiments", nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args_cli = parser.parse_args()

    os.chdir(args_cli.repo)
    sys.path.insert(0, str(args_cli.repo))
    from envs import CONFIGS_PATH  # pylint: disable=import-outside-toplevel
    from scripts.collect_data import (  # pylint: disable=import-outside-toplevel
        class_decorator,
        get_embodiment_config,
    )

    with open(Path(CONFIGS_PATH) / f"{args_cli.config}.yml", encoding="utf-8") as handle:
        base = yaml.load(handle.read(), Loader=yaml.FullLoader)
    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", encoding="utf-8") as handle:
        body_config = yaml.load(handle.read(), Loader=yaml.FullLoader)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[tuple[str, str, int]] = set()
    if args_cli.output.exists():
        with args_cli.output.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing.add((row["task"], row["embodiment"], int(row["seed"])))

    for task_name in args_cli.tasks:
        for embodiment in args_cli.embodiments:
            task = class_decorator(task_name)
            run_args = dict(base)
            robot_file = body_config[embodiment]["file_path"]
            dual_arm_embodied = embodiment not in SINGLE_ARM_PAIR_DISTANCE
            configured_embodiment: list[Any]
            if dual_arm_embodied:
                configured_embodiment = [embodiment]
            else:
                configured_embodiment = [
                    embodiment,
                    embodiment,
                    SINGLE_ARM_PAIR_DISTANCE[embodiment],
                ]
            run_args.update(
                {
                    "task_name": task_name,
                    "task_config": args_cli.config,
                    "embodiment": configured_embodiment,
                    "embodiment_name": embodiment,
                    "left_robot_file": robot_file,
                    "right_robot_file": robot_file,
                    "left_embodiment_config": get_embodiment_config(robot_file),
                    "right_embodiment_config": get_embodiment_config(robot_file),
                    "dual_arm_embodied": dual_arm_embodied,
                    "need_plan": True,
                    "save_data": False,
                    "collect_data": False,
                    "render_freq": 0,
                    "eval_video_log": False,
                    "save_path": str(
                        args_cli.output.parent
                        / "scratch"
                        / task_name
                        / re.sub(r"[^A-Za-z0-9_]+", "_", embodiment).lower()
                    ),
                }
            )
            if not dual_arm_embodied:
                run_args["embodiment_dis"] = SINGLE_ARM_PAIR_DISTANCE[embodiment]
            for seed in range(args_cli.seed_start, args_cli.seed_end):
                key = (task_name, embodiment, seed)
                if key in existing:
                    continue
                started = time.monotonic()
                picture_count = [0]
                status = "error"
                error = ""
                raw_steps = 0
                try:
                    task.setup_demo(now_ep_num=seed, seed=seed, **run_args)

                    def count_picture() -> None:
                        picture_count[0] += 1

                    task._take_picture = count_picture
                    task.play_once()
                    success = bool(task.plan_success and task.check_success())
                    status = "success" if success else "failed"
                    raw_steps = motion_steps(task)
                except Exception as exc:  # keep the full seed audit running
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    try:
                        # Each task/embodiment combination is launched in its
                        # own process. Avoid clearing SAPIEN's global renderer
                        # cache between seeds; process exit releases it safely.
                        task.close_env(clear_cache=False)
                    except Exception as exc:
                        if not error:
                            error = f"close_env {type(exc).__name__}: {exc}"
                row = {
                    "task": task_name,
                    "embodiment": embodiment,
                    "seed": seed,
                    "status": status,
                    "effective_control_steps": picture_count[0],
                    "raw_motion_planner_steps": raw_steps,
                    "wall_seconds": round(time.monotonic() - started, 4),
                    "error": error,
                }
                append_row(args_cli.output, row)
                print("MATCHED_HORIZON=" + json.dumps(row, ensure_ascii=False), flush=True)
                if status == "error":
                    # A setup failure may leave a partially initialized Robot
                    # object that cannot be reset safely on the next seed.
                    task = class_decorator(task_name)


if __name__ == "__main__":
    main()
