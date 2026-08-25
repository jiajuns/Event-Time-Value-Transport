#!/usr/bin/env python3
"""Recover effective RoboTwin control horizons from public planning trajectories.

The public HDF5 trajectories store one action every ``save_freq`` simulator steps.
For each task/embodiment we calibrate the fixed non-arm portion against episode 0,
then recover every episode horizon from the lightweight MoveIt path pickle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py

from inspect_cross_embodiment_pickle import load


EMBODIMENTS = {
    "RoboTwin-AgileX": "Aloha",
    "RoboTwin-X5": "ARX-X5",
    "RoboTwin-Panda": "Franka",
}


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return math.nan
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def motion_lengths(traj: dict[str, Any], save_freq: int) -> tuple[int, int]:
    left = traj["left_joint_path"]
    right = traj["right_joint_path"]
    if left and right:
        if len(left) != len(right):
            raise ValueError("cannot align unequal simultaneous left/right path lists")
        dense = [
            max(a["position"].shape[0], b["position"].shape[0])
            for a, b in zip(left, right)
        ]
    else:
        dense = [x["position"].shape[0] for x in (left or right)]
    # take_dense_action saves once before, ceil(n/freq) times in-loop, once after.
    sampled = sum(math.ceil(n / save_freq) + 2 for n in dense)
    return sum(dense), sampled


def hdf5_length(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["joint_action/vector"].shape[0])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-freq", type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    keyed: dict[tuple[str, str, int], int] = {}

    for repo_folder, embodiment in EMBODIMENTS.items():
        repo_root = args.root / repo_folder
        for task_root in sorted(p for p in repo_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            demo_dirs = list(task_root.glob("demo*"))
            if len(demo_dirs) != 1:
                raise ValueError(f"expected one demo directory below {task_root}, got {demo_dirs}")
            demo = demo_dirs[0]
            seeds = [int(x) for x in (demo / "seed.txt").read_text().split()]
            pkl0 = demo / "_traj_data" / "episode0.pkl"
            hdf50 = demo / "data" / "episode0.hdf5"
            _, sampled0 = motion_lengths(load(pkl0), args.save_freq)
            observed0 = hdf5_length(hdf50)
            fixed = observed0 - sampled0
            if fixed < 0:
                raise ValueError(f"negative fixed-step calibration for {task_root}")
            calibrations.append(
                {
                    "task": task_root.name,
                    "embodiment": embodiment,
                    "save_freq": args.save_freq,
                    "episode0_observed_steps": observed0,
                    "episode0_motion_sampled_steps": sampled0,
                    "fixed_saved_steps": fixed,
                }
            )

            pickles = sorted(
                (demo / "_traj_data").glob("episode*.pkl"),
                key=lambda p: int(p.stem.removeprefix("episode")),
            )
            if len(pickles) != len(seeds):
                raise ValueError(f"pickle/seed count mismatch for {demo}")
            for path in pickles:
                episode = int(path.stem.removeprefix("episode"))
                raw_motion, sampled_motion = motion_lengths(load(path), args.save_freq)
                effective = sampled_motion + fixed
                seed = seeds[episode]
                row = {
                    "task": task_root.name,
                    "embodiment": embodiment,
                    "episode": episode,
                    "seed": seed,
                    "raw_motion_planner_steps": raw_motion,
                    "motion_sampled_steps": sampled_motion,
                    "fixed_saved_steps": fixed,
                    "effective_control_steps": effective,
                }
                episode_rows.append(row)
                keyed[(task_root.name, embodiment, seed)] = effective

    aggregate_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in episode_rows:
        groups[(row["task"], row["embodiment"])].append(row["effective_control_steps"])
    for (task, embodiment), values in sorted(groups.items()):
        aggregate_rows.append(
            {
                "task": task,
                "embodiment": embodiment,
                "n": len(values),
                "mean_steps": statistics.fmean(values),
                "median_steps": statistics.median(values),
                "std_steps": statistics.stdev(values),
                "p10_steps": percentile(values, 0.10),
                "p90_steps": percentile(values, 0.90),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    tasks = sorted({row["task"] for row in episode_rows})
    bodies = list(EMBODIMENTS.values())
    for task in tasks:
        for idx, body_a in enumerate(bodies):
            for body_b in bodies[idx + 1 :]:
                seeds_a = {seed for t, body, seed in keyed if t == task and body == body_a}
                seeds_b = {seed for t, body, seed in keyed if t == task and body == body_b}
                common = sorted(seeds_a & seeds_b)
                directional = [keyed[(task, body_a, seed)] / keyed[(task, body_b, seed)] for seed in common]
                kappas = [max(r, 1.0 / r) for r in directional]
                slower_a = sum(r > 1 for r in directional)
                pair_rows.append(
                    {
                        "task": task,
                        "embodiment_a": body_a,
                        "embodiment_b": body_b,
                        "n_common_seeds": len(common),
                        "median_a_over_b": statistics.median(directional),
                        "geomean_a_over_b": math.exp(statistics.fmean(math.log(x) for x in directional)),
                        "median_kappa": statistics.median(kappas),
                        "p10_kappa": percentile(kappas, 0.10),
                        "p90_kappa": percentile(kappas, 0.90),
                        "fraction_kappa_ge_2": sum(k >= 2 for k in kappas) / len(kappas),
                        "fraction_kappa_le_1_1": sum(k <= 1.1 for k in kappas) / len(kappas),
                        "fraction_a_slower": slower_a / len(common),
                    }
                )

    write_csv(args.output / "episode_horizons.csv", episode_rows)
    write_csv(args.output / "task_embodiment_summary.csv", aggregate_rows)
    write_csv(args.output / "paired_kappa_by_task.csv", pair_rows)
    write_csv(args.output / "calibrations.csv", calibrations)
    summary = {
        "data_scope": "public preliminary audit; not matched four-embodiment GATE",
        "save_freq": args.save_freq,
        "episode_count": len(episode_rows),
        "tasks": tasks,
        "embodiments": bodies,
        "calibration_fixed_steps": sorted({row["fixed_saved_steps"] for row in calibrations}),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
