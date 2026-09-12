#!/usr/bin/env python3
"""Run and quality-gate a bounded RoboTwin multi-task/multi-body smoke matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from universal_event.schema import EVENT_NAMES, RELATION_NAMES, TASK_SPECS, schema_sha256


PLACEMENT_TASKS = {
    "move_can_pot", "place_container_plate", "place_can_basket",
    "place_empty_cup", "put_object_cabinet", "stack_blocks_two",
    "hanging_mug", "handover_block",
}
OPEN_TASKS = {"open_laptop", "open_microwave"}
ACTIVATION_TASKS = {"press_stapler", "click_bell"}


def representative_matrix() -> list[tuple[str, str]]:
    # One embodiment covers the full semantic task inventory.  Every other
    # embodiment covers grasp/place, articulation and contact activation.
    matrix = [(task, "aloha-agilex") for task in TASK_SPECS]
    for body in ("arx-x5", "franka", "piper", "ur5"):
        matrix.extend(
            [("move_can_pot", body), ("open_laptop", body), ("press_stapler", body)]
        )
    return matrix


def goal_satisfied(handle: h5py.File) -> bool:
    relation = handle["relations"][-1].astype(bool)
    goal = handle["goal"][:]
    mask = handle["goal_mask"][:].astype(bool)
    if not mask.any():
        return False
    expected = goal[mask] >= 0.5
    return bool(np.equal(relation[mask], expected).all())


def inspect_episode(path: Path, task: str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        events = handle["events"][:].astype(bool)
        actions = handle["actions14"][:]
        counts = {name: int(events[:, index].sum())
                  for index, name in enumerate(EVENT_NAMES)}
        checks: dict[str, bool] = {
            "native_success": bool(handle.attrs["native_success"]),
            "nonzero_action": bool((np.linalg.norm(actions, axis=-1) > 1e-5).any()),
            "goal_satisfied": goal_satisfied(handle),
            "no_false_drop": counts["drop"] == 0,
        }
        if task in PLACEMENT_TASKS:
            checks.update(
                approach=counts["approach"] > 0,
                grasp=counts["grasp"] > 0,
                transport=counts["transport"] > 0,
                release=counts["release"] > 0,
            )
        elif task in OPEN_TASKS:
            checks["open"] = counts["open"] > 0
        elif task in ACTIVATION_TASKS:
            checks["activate"] = counts["activate"] > 0
        else:
            raise ValueError(f"task family is not quality-gated: {task}")
        return {
            "path": str(path),
            "frames": int(len(events)),
            "event_counts": counts,
            "planner_clean": not bool(str(handle.attrs["planner_error"])),
            "planner_error": str(handle.attrs["planner_error"]),
            "checks": checks,
            "passed": all(checks.values()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--condition", choices=("clean", "randomized"), default="clean")
    parser.add_argument(
        "--pair", action="append", default=[], metavar="TASK:BODY",
        help="run only this task/body pair; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.robotwin_root.resolve()
    collector = args.collector.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    environment.update(VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd)
    rows: list[dict[str, object]] = []
    matrix = []
    for value in args.pair:
        task, separator, body = value.partition(":")
        if not separator or task not in TASK_SPECS or body not in {
            "aloha-agilex", "arx-x5", "franka", "piper", "ur5"
        }:
            raise ValueError(f"invalid --pair: {value}")
        matrix.append((task, body))
    if not matrix:
        matrix = representative_matrix()
    for index, (task, body) in enumerate(matrix, 1):
        print(f"[{index:02d}/{len(matrix):02d}] {task} / {body}", flush=True)
        command = [
            str(args.python), "-u", str(collector),
            "--robotwin-root", str(root), "--task", task, "--body", body,
            "--condition", args.condition, "--seed-start", str(args.seed),
            "--episodes", "1", "--sample-hz", "15", "--scene-backend", "renderer",
            "--output", str(output),
        ]
        completed = subprocess.run(
            command, cwd=root, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        episode = output / task / body / args.condition / f"episode_seed_{args.seed}.hdf5"
        row: dict[str, object] = {
            "task": task, "body": body, "seed": args.seed,
            "returncode": completed.returncode,
            "collector_output": completed.stdout[-6000:],
        }
        if completed.returncode == 0 and episode.is_file():
            try:
                row.update(inspect_episode(episode, task))
            except Exception as error:
                row.update(passed=False, inspection_error=f"{type(error).__name__}: {error}")
        else:
            row["passed"] = False
        rows.append(row)
        print("  PASS" if row["passed"] else "  FAIL", flush=True)
        receipt = {
            "format": "etsf_robotwin2_small_simulation_receipt_v1",
            "schema_sha256": schema_sha256(),
            "seed": args.seed,
            "condition": args.condition,
            "matrix_size": len(matrix),
            "completed": len(rows),
            "passed": sum(bool(item["passed"]) for item in rows),
            "failed": sum(not bool(item["passed"]) for item in rows),
            "rows": rows,
        }
        (output / "small_simulation_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps({key: receipt[key] for key in
                      ("matrix_size", "passed", "failed")}), flush=True)
    if receipt["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
