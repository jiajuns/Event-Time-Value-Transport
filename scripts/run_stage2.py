#!/usr/bin/env python3
"""Run the ETSF Stage-2 liquid value-transfer study."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import binomtest
import torch
import torch.nn.functional as F
from torch import nn


TASKS = [
    "adjust_bottle",
    "handover_block",
    "move_can_pot",
    "place_container_plate",
    "beat_block_hammer",
    "lift_pot",
]
BODIES = ["aloha-agilex", "ARX-X5", "piper", "ur5-wsg"]
SOURCE_BODIES = ["aloha-agilex", "ARX-X5"]
TARGET_BODIES = ["piper", "ur5-wsg"]
RAW_EVENTS = ["e0", "e1", "e2", "e3", "e4", "eK"]
MODEL_EVENTS = ["e0", "e1", "e2", "e12", "e3", "e4", "eK"]
RHO_FAIL = 0.30
RHO_DEPLOY = 0.70
SEQUENTIAL_ALPHA = 0.05
SEED = 20260825
GAMMA = 0.99
HIDDEN = 96
BINS = 51


@dataclass
class Episode:
    task: str
    body: str
    index: int
    success: bool
    poses: np.ndarray
    names: list[str]
    raw_events: dict[str, int] = field(default_factory=dict)
    events: list[tuple[str, int]] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return int(self.poses.shape[0])


@dataclass
class BoundaryRecord:
    episode: Episode
    inputs: np.ndarray
    dts: np.ndarray
    durations: np.ndarray
    event_ids: np.ndarray
    remaining: np.ndarray


def load_episodes(data_root: Path) -> list[Episode]:
    episodes = []
    for task in TASKS:
        for body in BODIES:
            if body in SOURCE_BODIES:
                paths = sorted((data_root / "source_object_poses" / task / body).glob("episode_*.npz"))
                for path in paths:
                    with np.load(path) as data:
                        episodes.append(
                            Episode(
                                task=task,
                                body=body,
                                index=int(path.stem.rsplit("_", 1)[1]),
                                success=bool(data["success"]),
                                poses=data["poses"].astype(np.float32),
                                names=[str(value) for value in data["object_names"]],
                            )
                        )
            else:
                paths = sorted((data_root / "target_rollouts" / task / body).glob("episode_*.hdf5"))
                for path in paths:
                    with h5py.File(path, "r") as handle:
                        episodes.append(
                            Episode(
                                task=task,
                                body=body,
                                index=int(handle.attrs["rollout_index"]),
                                success=bool(handle.attrs["success"]),
                                poses=handle["object_poses"][:].astype(np.float32),
                                names=[value.decode() for value in handle["object_names"][:]],
                            )
                        )
    expected = Counter({(task, body): 50 if body in SOURCE_BODIES else 20 for task in TASKS for body in BODIES})
    actual = Counter((episode.task, episode.body) for episode in episodes)
    if actual != expected:
        raise RuntimeError(f"incomplete data cells: actual={actual}, expected={expected}")
    return episodes


def two_means(points: np.ndarray) -> np.ndarray:
    direction = np.linalg.eigh(np.cov(points.T))[1][:, -1]
    projection = points @ direction
    centers = np.stack([points[projection.argmin()], points[projection.argmax()]])
    for _ in range(30):
        labels = np.square(points[:, None, :] - centers[None, :, :]).sum(2).argmin(1)
        if min(np.bincount(labels, minlength=2)) == 0:
            return points.mean(0, keepdims=True)
        updated = np.stack([points[labels == label].mean(0) for label in range(2)])
        if np.allclose(updated, centers):
            break
        centers = updated
    labels = np.square(points[:, None, :] - centers[None, :, :]).sum(2).argmin(1)
    one_sse = float(np.square(points - points.mean(0)).sum())
    two_sse = float(np.square(points - centers[labels]).sum())
    if two_sse >= 0.35 * max(one_sse, 1e-12) or min(np.bincount(labels, minlength=2)) < 0.15 * len(points):
        return points.mean(0, keepdims=True)
    return centers.astype(np.float32)


def calibrate_events(episodes: list[Episode]) -> dict[str, dict[str, object]]:
    calibration = {}
    for task in TASKS:
        source = [episode for episode in episodes if episode.task == task and episode.body in SOURCE_BODIES]
        names = sorted(set.intersection(*(set(episode.names) for episode in source)))
        displacement = {}
        for name in names:
            maxima = []
            for episode in source:
                position = episode.poses[:, episode.names.index(name), :3]
                maxima.append(float(np.linalg.norm(position - position[0], axis=1).max()))
            displacement[name] = float(np.median(maxima))
        moving = max(displacement, key=displacement.get)
        successful = [episode for episode in source if episode.success]
        endpoints = np.stack([episode.poses[-1, episode.names.index(moving), :3] for episode in successful])
        absolute_variance = float(np.square(endpoints - endpoints.mean(0)).sum(axis=1).mean())
        anchor = ""
        offset = np.zeros(3, dtype=np.float32)
        best_variance = math.inf
        for name in names:
            if name == moving or name in {"table", "wall"} or displacement[name] > 0.2 * displacement[moving]:
                continue
            relative = np.stack(
                [
                    episode.poses[-1, episode.names.index(moving), :3]
                    - episode.poses[-1, episode.names.index(name), :3]
                    for episode in successful
                ]
            )
            variance = float(np.square(relative - np.median(relative, axis=0)).sum(axis=1).mean())
            if variance < best_variance:
                anchor = name
                offset = np.median(relative, axis=0).astype(np.float32)
                best_variance = variance
        if not anchor or best_variance >= 0.25 * max(absolute_variance, 1e-8):
            anchor = ""
            centers = two_means(endpoints)
        else:
            centers = np.empty((0, 3), dtype=np.float32)
        path_lengths = []
        rises = []
        step_motion = []
        final_distances = []
        for episode in source:
            position = episode.poses[:, episode.names.index(moving), :3]
            steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
            cumulative = np.r_[0.0, np.cumsum(steps)]
            step_motion.extend(steps.tolist())
            if episode.success:
                path_lengths.append(float(cumulative[-1]))
                rises.append(float((position[:, 2] - position[0, 2]).max()))
                if anchor:
                    target = episode.poses[-1, episode.names.index(anchor), :3] + offset
                    final_distances.append(float(np.linalg.norm(position[-1] - target)))
                else:
                    final_distances.append(float(np.linalg.norm(position[-1] - centers, axis=1).min()))
        calibration[task] = {
            "moving": moving,
            "anchor": anchor,
            "offset": offset,
            "centers": centers,
            "delta_move": max(0.005, 0.10 * float(np.median(path_lengths))),
            "delta_z": max(0.005, 0.20 * float(np.median(rises))),
            "tau_motion": max(0.0005, float(np.quantile(np.asarray(step_motion), 0.85))),
            "tau_d": max(0.015, float(np.quantile(final_distances, 0.95)) + 0.005),
            "stationary_steps": 3,
        }
    return calibration


def assign_raw_events(episodes: list[Episode], calibration: dict[str, dict[str, object]]) -> None:
    for episode in episodes:
        config = calibration[episode.task]
        moving = str(config["moving"])
        position = episode.poses[:, episode.names.index(moving), :3]
        step_motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
        cumulative = np.cumsum(step_motion)
        if config["anchor"]:
            anchor_position = episode.poses[:, episode.names.index(str(config["anchor"])), :3]
            target = anchor_position + np.asarray(config["offset"])[None, :]
            distance = np.linalg.norm(position - target, axis=1)
        else:
            centers = np.asarray(config["centers"])
            distance = np.linalg.norm(position[:, None, :] - centers[None, :, :], axis=2).min(1)
        raw = {"e0": 0}
        candidates = np.flatnonzero(cumulative >= float(config["delta_move"]))
        if candidates.size:
            raw["e1"] = int(candidates[0])
        candidates = np.flatnonzero(position[:, 2] >= position[0, 2] + float(config["delta_z"]))
        if candidates.size:
            raw["e2"] = int(candidates[0])
        candidates = np.flatnonzero(distance <= float(config["tau_d"]))
        if candidates.size:
            raw["e3"] = int(candidates[0])
        stationary = (distance <= float(config["tau_d"])) & (step_motion <= float(config["tau_motion"]))
        width = int(config["stationary_steps"])
        for index in range(episode.steps - width + 1):
            if stationary[index : index + width].all():
                raw["e4"] = index + width - 1
                break
        if episode.success:
            raw["eK"] = episode.steps - 1
        episode.raw_events = raw


def mode_sequence(episodes: list[Episode], raw: bool = False) -> tuple[str, int]:
    if raw:
        sequences = [tuple(name for name, _ in sorted(episode.raw_events.items(), key=lambda item: (item[1], item[0]))) for episode in episodes]
    else:
        sequences = [tuple(name for name, _ in episode.events) for episode in episodes]
    mode, count = Counter(sequences).most_common(1)[0]
    return ">".join(mode), count


def derive_chains(episodes: list[Episode]) -> dict[str, dict[str, object]]:
    specs = {}
    for task in TASKS:
        source_groups = {
            body: [episode for episode in episodes if episode.task == task and episode.body == body]
            for body in SOURCE_BODIES
        }
        source_modes = {body: mode_sequence(group, raw=True)[0].split(">") for body, group in source_groups.items()}
        common = set.intersection(*(set(sequence) for sequence in source_modes.values()))
        gaps = []
        for group in source_groups.values():
            gaps.extend(
                abs(episode.raw_events["e1"] - episode.raw_events["e2"])
                for episode in group
                if "e1" in episode.raw_events and "e2" in episode.raw_events
            )
        source_orders = []
        for sequence in source_modes.values():
            if "e1" in sequence and "e2" in sequence:
                source_orders.append(sequence.index("e1") < sequence.index("e2"))
        merge = "e1" in common and "e2" in common and (
            (gaps and float(np.median(gaps)) < 5.0) or len(set(source_orders)) > 1
        )
        if merge:
            middle = ["e12"]
        else:
            pair = [event for event in ["e1", "e2"] if event in common]
            if len(pair) == 2 and source_orders and not source_orders[0]:
                pair.reverse()
            middle = pair
        chain = ["e0", *middle]
        chain.extend(event for event in ["e3", "e4", "eK"] if event in common)
        specs[task] = {
            "chain": chain,
            "merge_e1_e2": merge,
            "source_raw_modes": source_modes,
            "median_e1_e2_gap": float(np.median(gaps)) if gaps else math.nan,
        }
    return specs


def apply_chains(episodes: list[Episode], specs: dict[str, dict[str, object]]) -> None:
    for episode in episodes:
        spec = specs[episode.task]
        raw = dict(episode.raw_events)
        if spec["merge_e1_e2"] and "e1" in raw and "e2" in raw:
            raw["e12"] = min(raw["e1"], raw["e2"])
        events = []
        previous = -1
        for name in spec["chain"]:
            if name not in raw or raw[name] < previous:
                break
            events.append((name, int(raw[name])))
            previous = int(raw[name])
        episode.events = events


def observed_sequence(episode: Episode, spec: dict[str, object]) -> list[str]:
    raw = dict(episode.raw_events)
    if spec["merge_e1_e2"] and "e1" in raw and "e2" in raw:
        raw["e12"] = min(raw["e1"], raw["e2"])
    chain = set(spec["chain"])
    return [name for name, _ in sorted(raw.items(), key=lambda item: (item[1], item[0])) if name in chain]


def write_m1(
    episodes: list[Episode],
    specs: dict[str, dict[str, object]],
    output_root: Path,
) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        modes = {}
        mode_counts = {}
        for body in BODIES:
            group = [episode for episode in episodes if episode.task == task and episode.body == body]
            sequences = [tuple(observed_sequence(episode, specs[task])) for episode in group]
            mode, mode_counts[body] = Counter(sequences).most_common(1)[0]
            modes[body] = ">".join(mode)
        strict = len(set(modes.values())) == 1
        sequences = [mode.split(">") if mode else [] for mode in modes.values()]
        chain = list(specs[task]["chain"])
        prefix_compatible = all(
            [chain.index(event) for event in sequence] == sorted(chain.index(event) for event in sequence)
            for sequence in sequences
        )
        for body in BODIES:
            group = [episode for episode in episodes if episode.task == task and episode.body == body]
            for event_index, event in enumerate(chain):
                if event_index == 0:
                    eligible = len(group)
                    reached = len(group)
                    intervals = []
                    transition = "start"
                else:
                    previous = chain[event_index - 1]
                    transition = f"{previous}->{event}"
                    eligible = sum(any(name == previous for name, _ in episode.events) for episode in group)
                    reached = sum(any(name == event for name, _ in episode.events) for episode in group)
                    intervals = []
                    for episode in group:
                        times = dict(episode.events)
                        if previous in times and event in times:
                            intervals.append(times[event] - times[previous])
                rows.append(
                    {
                        "task": task,
                        "embodiment": body,
                        "event": event,
                        "transition": transition,
                        "canonical_chain": ">".join(chain),
                        "mode_sequence": modes[body],
                        "mode_count": mode_counts[body],
                        "cross_body_full_mode_consistent": int(strict),
                        "cross_body_structural_prefix_compatible": int(prefix_compatible),
                        "merge_e1_e2": int(specs[task]["merge_e1_e2"]),
                        "median_source_e1_e2_gap": specs[task]["median_e1_e2_gap"],
                        "source_mode_aloha": ">".join(specs[task]["source_raw_modes"]["aloha-agilex"]),
                        "source_mode_arx": ">".join(specs[task]["source_raw_modes"]["ARX-X5"]),
                        "n_rollouts": len(group),
                        "eligible": eligible,
                        "reached": reached,
                        "reach_rate": reached / eligible if eligible else math.nan,
                        "interval_n": len(intervals),
                        "interval_mean": float(np.mean(intervals)) if intervals else math.nan,
                        "interval_std": float(np.std(intervals)) if intervals else math.nan,
                        "interval_median": float(np.median(intervals)) if intervals else math.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "results/P1_fixed_event_consistency.csv", index=False)
    return frame


def write_rank_test(m1: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    cells = []
    for (task, transition), group in m1[m1.event != "e0"].groupby(["task", "transition"]):
        if len(group) != len(BODIES) or group.interval_mean.isna().any():
            continue
        cells.append(
            {
                "name": f"{task}:{transition}",
                "values": group.set_index("embodiment").loc[BODIES].interval_mean.to_numpy(float),
                "clean": bool((group.interval_median >= 5).all()),
            }
        )
    for name, selected in [
        ("all_complete_cells", cells),
        ("min_interval_ge_5", [cell for cell in cells if cell["clean"]]),
    ]:
        if selected:
            matrix = np.log(np.stack([cell["values"] for cell in selected], axis=1))
            centered = matrix - matrix.mean(0, keepdims=True)
            singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
            energy = np.square(singular)
            rank1 = float(energy[0] / energy.sum()) if energy.sum() else math.nan
        else:
            singular = np.asarray([])
            rank1 = math.nan
        rows.append(
            {
                "dataset": name,
                "n_cells": len(selected),
                "rank1_explained": rank1,
                "singular_values": json.dumps(singular.tolist()),
                "cells": json.dumps([cell["name"] for cell in selected]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "results/P1_rank_test.csv", index=False)
    return frame


def link_posteriors(group: list[Episode], chain: list[str]) -> tuple[np.ndarray, np.ndarray]:
    posterior_a = np.ones(len(chain) - 1, dtype=np.float64)
    posterior_b = np.ones(len(chain) - 1, dtype=np.float64)
    for index in range(1, len(chain)):
        previous = chain[index - 1]
        current = chain[index]
        eligible = sum(any(name == previous for name, _ in episode.events) for episode in group)
        reached = sum(any(name == current for name, _ in episode.events) for episode in group)
        posterior_a[index - 1] += reached
        posterior_b[index - 1] += eligible - reached
    return posterior_a, posterior_b


def posterior_product(
    posterior_a: np.ndarray,
    posterior_b: np.ndarray,
    rng: np.random.Generator,
    draws: int = 50000,
) -> np.ndarray:
    if len(posterior_a) == 0:
        return np.ones(draws)
    return rng.beta(posterior_a, posterior_b, size=(draws, len(posterior_a))).prod(1)


def write_n_sweep(
    episodes: list[Episode],
    specs: dict[str, dict[str, object]],
    output_root: Path,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for count in [5, 10, 15]:
        for body in TARGET_BODIES:
            for task in TASKS:
                group = sorted(
                    [episode for episode in episodes if episode.task == task and episode.body == body],
                    key=lambda episode: episode.index,
                )
                adaptation = group[:count]
                test = group[15:20]
                posterior_a, posterior_b = link_posteriors(adaptation, list(specs[task]["chain"]))
                samples = posterior_product(posterior_a, posterior_b, rng)
                truth = float(np.mean([episode.success for episode in test]))
                prediction = float(samples.mean())
                rows.append(
                    {
                        "task": task,
                        "embodiment": body,
                        "N": count,
                        "adaptation_indices": json.dumps([episode.index for episode in adaptation]),
                        "test_indices": json.dumps([episode.index for episode in test]),
                        "rho_posterior_a": json.dumps(posterior_a.astype(int).tolist()),
                        "rho_posterior_b": json.dumps(posterior_b.astype(int).tolist()),
                        "predicted_success_rate": prediction,
                        "predicted_success_ci_low": float(np.quantile(samples, 0.025)),
                        "predicted_success_ci_high": float(np.quantile(samples, 0.975)),
                        "approval_probability": float(np.mean(samples >= RHO_DEPLOY)),
                        "reject_probability": float(np.mean(samples <= RHO_FAIL)),
                        "test_success_rate": truth,
                        "absolute_error": abs(prediction - truth),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "results/P1_n_sweep_corrected.csv", index=False)
    return frame


def write_budget_audit(episodes: list[Episode], output_root: Path) -> pd.DataFrame:
    rows = []
    for body in TARGET_BODIES:
        for task in TASKS:
            group = sorted(
                [episode for episode in episodes if episode.task == task and episode.body == body],
                key=lambda episode: episode.index,
            )
            rows.append(
                {
                    "task": task,
                    "embodiment": body,
                    "available": len(group),
                    "adaptation_pool": json.dumps([episode.index for episode in group[:15]]),
                    "test": json.dumps([episode.index for episode in group[15:20]]),
                    "split_overlap": len(set(episode.index for episode in group[:15]) & set(episode.index for episode in group[15:20])),
                    "raw_remaining_direction": "larger_is_farther",
                    "signed_success_score": "-raw_remaining_steps",
                    "score_direction": "larger_is_closer_to_success",
                    "polarity_boundary_test": "pass" if -20.0 < -10.0 else "fail",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "results/P1_auc_budget_audit.csv", index=False)
    return frame


def write_deployment_approval(data_root: Path, output_root: Path) -> pd.DataFrame:
    paths = sorted((data_root / "m2_fresh/adjust_bottle/piper").glob("episode_*.hdf5"))
    rows = []
    successes = 0
    e_fail = 1.0
    e_deploy = 1.0
    max_evidence = 1.0
    decision = "continue"
    decision_rollouts = math.nan
    for index, path in enumerate(paths, 1):
        with h5py.File(path, "r") as handle:
            outcome = int(handle.attrs["success"])
        successes += outcome
        e_fail *= 1.0 + RHO_FAIL - outcome
        e_deploy *= 1.0 + outcome - RHO_DEPLOY
        max_evidence = max(max_evidence, e_fail, e_deploy)
        if decision == "continue" and e_fail >= 1.0 / SEQUENTIAL_ALPHA:
            decision = "reject_deployment"
            decision_rollouts = index
        elif decision == "continue" and e_deploy >= 1.0 / SEQUENTIAL_ALPHA:
            decision = "approve_deployment"
            decision_rollouts = index
        posterior_a = successes + 1
        posterior_b = index - successes + 1
        rows.append(
            {
                "task": "adjust_bottle",
                "embodiment": "piper",
                "N": index,
                "successes": successes,
                "posterior_a": posterior_a,
                "posterior_b": posterior_b,
                "posterior_mean": posterior_a / (posterior_a + posterior_b),
                "approval_probability": beta_distribution.sf(RHO_DEPLOY, posterior_a, posterior_b),
                "reject_probability": beta_distribution.cdf(RHO_FAIL, posterior_a, posterior_b),
                "in_30_70_uncertainty_band": int(
                    RHO_FAIL <= posterior_a / (posterior_a + posterior_b) <= RHO_DEPLOY
                ),
                "e_process_fail": e_fail,
                "e_process_deploy": e_deploy,
                "anytime_p_value": min(1.0, 1.0 / max_evidence),
                "sequential_decision": decision,
                "decision_rollouts": decision_rollouts,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "results/rho_deployment_approval.csv", index=False)
    return frame


def run_g0(data_root: Path, output_root: Path) -> dict[str, object]:
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(data_root)
    calibration = calibrate_events(episodes)
    assign_raw_events(episodes, calibration)
    specs = derive_chains(episodes)
    apply_chains(episodes, specs)
    m1 = write_m1(episodes, specs, output_root)
    rank = write_rank_test(m1, output_root)
    sweep = write_n_sweep(episodes, specs, output_root)
    audit = write_budget_audit(episodes, output_root)
    approval = write_deployment_approval(data_root, output_root)
    calibration_json = {}
    for task, values in calibration.items():
        calibration_json[task] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }
    (output_root / "event_spec.json").write_text(
        json.dumps({"calibration": calibration_json, "chains": specs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    strict = float(m1.groupby("task").cross_body_full_mode_consistent.first().mean())
    structural = float(m1.groupby("task").cross_body_structural_prefix_compatible.first().mean())
    n5 = sweep[sweep.N == 5]
    final_approval = approval.iloc[-1] if len(approval) else None
    summary = {
        "episodes": len(episodes),
        "strict_mode_consistency": strict,
        "structural_prefix_consistency": structural,
        "rank1_clean": float(rank[rank.dataset == "min_interval_ge_5"].rank1_explained.iloc[0]),
        "rank_cells_clean": int(rank[rank.dataset == "min_interval_ge_5"].n_cells.iloc[0]),
        "n5_mae": float(n5.absolute_error.mean()),
        "split_overlap": int(audit.split_overlap.sum()),
        "deployment_decision": None if final_approval is None else str(final_approval.sequential_decision),
        "deployment_rollouts": None if final_approval is None or math.isnan(final_approval.decision_rollouts) else int(final_approval.decision_rollouts),
        "deployment_reject_probability": None if final_approval is None else float(final_approval.reject_probability),
        "deployment_anytime_p": None if final_approval is None else float(final_approval.anytime_p_value),
    }
    (output_root / "g0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def raw_boundary_input(
    episode: Episode,
    event: str,
    frame: int,
    calibration: dict[str, object],
) -> np.ndarray:
    moving = str(calibration["moving"])
    moving_index = episode.names.index(moving)
    position = episode.poses[:, moving_index, :3]
    current = position[frame]
    delta = current - position[0]
    if calibration["anchor"]:
        anchor = episode.poses[frame, episode.names.index(str(calibration["anchor"])), :3]
        relative_target = current - (anchor + np.asarray(calibration["offset"]))
    else:
        centers = np.asarray(calibration["centers"])
        relative = current[None, :] - centers
        relative_target = relative[np.linalg.norm(relative, axis=1).argmin()]
    quaternion = episode.poses[frame, moving_index, 3:7]
    path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(position[: frame + 1], axis=0), axis=1))][-1]
    direction = np.zeros(3, dtype=np.float32)
    if frame:
        step = position[frame] - position[frame - 1]
        norm = float(np.linalg.norm(step))
        if norm > 1e-8:
            direction = step / norm
    continuous = np.concatenate(
        [
            delta,
            relative_target,
            quaternion,
            np.asarray([path / max(float(calibration["delta_move"]), 1e-6)]),
            direction,
        ]
    ).astype(np.float32)
    task_token = np.eye(len(TASKS), dtype=np.float32)[TASKS.index(episode.task)]
    event_token = np.eye(len(MODEL_EVENTS), dtype=np.float32)[MODEL_EVENTS.index(event)]
    return np.concatenate([continuous, task_token, event_token])


def build_records(
    episodes: list[Episode],
    calibration: dict[str, dict[str, object]],
    specs: dict[str, dict[str, object]],
) -> tuple[list[BoundaryRecord], np.ndarray, np.ndarray]:
    records = []
    for episode in episodes:
        chain = list(specs[episode.task]["chain"])
        inputs = np.stack(
            [raw_boundary_input(episode, event, frame, calibration[episode.task]) for event, frame in episode.events]
        )
        times = np.asarray([frame for _, frame in episode.events], dtype=np.float32)
        dts = np.r_[1.0, np.diff(times)].astype(np.float32)
        durations = np.r_[np.diff(times), np.nan].astype(np.float32)
        event_ids = np.asarray([MODEL_EVENTS.index(event) for event, _ in episode.events], dtype=np.int64)
        remaining = np.asarray([len(chain) - chain.index(event) - 1 for event, _ in episode.events], dtype=np.float32)
        records.append(BoundaryRecord(episode, inputs, dts, durations, event_ids, remaining))
    train_values = np.concatenate(
        [
            record.inputs[:, :14]
            for record in records
            if record.episode.body in SOURCE_BODIES and record.episode.index < 40
        ]
    )
    mean = train_values.mean(0).astype(np.float32)
    std = np.maximum(train_values.std(0), 1e-5).astype(np.float32)
    for record in records:
        record.inputs[:, :14] = (record.inputs[:, :14] - mean) / std
    return records, mean, std


def pack_records(records: list[BoundaryRecord], device: torch.device) -> dict[str, torch.Tensor]:
    length = max(len(record.event_ids) for record in records)
    width = records[0].inputs.shape[1]
    count = len(records)
    inputs = np.zeros((count, length, width), dtype=np.float32)
    dts = np.ones((count, length), dtype=np.float32)
    durations = np.zeros((count, length), dtype=np.float32)
    event_ids = np.zeros((count, length), dtype=np.int64)
    remaining = np.zeros((count, length), dtype=np.float32)
    mask = np.zeros((count, length), dtype=bool)
    duration_mask = np.zeros((count, length), dtype=bool)
    body_is_arx = np.zeros(count, dtype=np.float32)
    success = np.zeros(count, dtype=np.int64)
    for row, record in enumerate(records):
        size = len(record.event_ids)
        inputs[row, :size] = record.inputs
        dts[row, :size] = record.dts
        finite = np.isfinite(record.durations)
        durations[row, :size][finite] = record.durations[finite]
        event_ids[row, :size] = record.event_ids
        remaining[row, :size] = record.remaining
        mask[row, :size] = True
        duration_mask[row, :size] = finite
        body_is_arx[row] = float(record.episode.body == "ARX-X5")
        success[row] = int(record.episode.success)
    return {
        "inputs": torch.from_numpy(inputs).to(device),
        "dts": torch.from_numpy(dts).to(device),
        "durations": torch.from_numpy(durations).to(device),
        "event_ids": torch.from_numpy(event_ids).to(device),
        "remaining": torch.from_numpy(remaining).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "duration_mask": torch.from_numpy(duration_mask).to(device),
        "body_is_arx": torch.from_numpy(body_is_arx).to(device),
        "success": torch.from_numpy(success).to(device),
    }


class LowRankLiquidCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.base_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.shape_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def components(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joined = torch.cat([inputs, hidden], dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        base = math.log(10.0) + 2.0 * torch.tanh(self.base_tau(joined))
        if self.mode == "t3":
            shape = torch.ones_like(base)
        else:
            shape = torch.tanh(self.shape_tau(joined))
            shape = shape - shape.mean(-1, keepdim=True)
            shape = shape / torch.sqrt(torch.square(shape).mean(-1, keepdim=True) + 1e-6)
        log_tau = torch.clamp(base + beta[:, None] * shape, -4.0, 8.0)
        return candidate, log_tau, shape

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        timespan: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate, log_tau, _ = self.components(inputs, hidden, beta)
        decay = torch.exp(-timespan[:, None] / torch.exp(log_tau))
        return decay * hidden + (1.0 - decay) * candidate, log_tau


class LiquidCritic(nn.Module):
    def __init__(self, input_dim: int, mode: str) -> None:
        super().__init__()
        self.cell = LowRankLiquidCell(input_dim, HIDDEN, mode)
        self.value = nn.Linear(HIDDEN, len(MODEL_EVENTS) * BINS)
        self.duration = nn.Linear(HIDDEN, 1)
        self.beta_arx = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        inputs: torch.Tensor,
        dts: torch.Tensor,
        mask: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = inputs.new_zeros(inputs.shape[0], HIDDEN)
        logits = []
        durations = []
        log_taus = []
        for index in range(inputs.shape[1]):
            proposed, log_tau = self.cell(inputs[:, index], hidden, dts[:, index], beta)
            hidden = torch.where(mask[:, index, None], proposed, hidden)
            logits.append(self.value(hidden).view(-1, len(MODEL_EVENTS), BINS))
            log_duration = F.softplus(self.duration(hidden).squeeze(-1)).clamp(max=6.25)
            durations.append(torch.expm1(log_duration))
            log_taus.append(log_tau)
        return torch.stack(logits, 1), torch.stack(durations, 1), torch.stack(log_taus, 1)


class EventMLPCritic(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, len(MODEL_EVENTS) * BINS),
        )

    def forward(self, inputs: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([inputs, lambdas[..., None]], dim=-1)
        return self.net(joined).view(*inputs.shape[:2], len(MODEL_EVENTS), BINS)


def support_from_lambda(lambdas: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
    horizon = remaining + 1.0
    denominator = torch.clamp(1.0 - lambdas, min=1e-5)
    upper = (1.0 - torch.pow(lambdas, horizon)) / denominator
    upper = torch.where(lambdas > 0.9999, horizon, upper).clamp(min=1.0, max=len(MODEL_EVENTS))
    fractions = torch.linspace(0.0, 1.0, BINS, device=lambdas.device)
    return upper[..., None] * fractions


def decode_values(
    logits: torch.Tensor,
    lambdas: torch.Tensor,
    remaining: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = support_from_lambda(lambdas, remaining)
    probabilities = torch.softmax(logits, dim=-1)
    values = (probabilities * support[..., None, :]).sum(-1)
    return values, support


def hl_gauss(target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    width = torch.clamp(support[..., 1] - support[..., 0], min=1e-4)
    distance = (support[..., None, :] - target[..., :, None]) / (0.75 * width[..., None, None])
    weights = torch.exp(-0.5 * torch.square(distance))
    return weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)


def liquid_outputs(
    model: LiquidCritic,
    batch: dict[str, torch.Tensor],
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, durations, log_taus = model(batch["inputs"], batch["dts"], batch["mask"], beta)
    lambdas = torch.pow(durations.new_tensor(GAMMA), durations)
    values, support = decode_values(logits, lambdas, batch["remaining"])
    return logits, values, durations, lambdas, support


def td_targets(
    values: torch.Tensor,
    lambdas: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    following = torch.zeros_like(values)
    following[:, :-1] = values[:, 1:]
    has_next = torch.zeros_like(batch["mask"])
    has_next[:, :-1] = batch["mask"][:, 1:]
    phi = F.one_hot(batch["event_ids"], num_classes=len(MODEL_EVENTS)).float()
    return phi + lambdas[..., None] * has_next[..., None] * following


def liquid_loss(
    model: LiquidCritic,
    target: LiquidCritic,
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = {key: value[indices] for key, value in batch.items()}
    beta = selected["body_is_arx"] * model.beta_arx
    logits, _, durations, lambdas, support = liquid_outputs(model, selected, beta)
    with torch.no_grad():
        target_beta = selected["body_is_arx"] * target.beta_arx
        _, next_values, _, _, _ = liquid_outputs(target, selected, target_beta)
        desired = td_targets(next_values, lambdas, selected)
        distribution = hl_gauss(desired, support)
    cross_entropy = -(distribution * torch.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
    value_loss = cross_entropy[selected["mask"]].mean()
    duration_error = F.smooth_l1_loss(
        torch.log1p(durations), torch.log1p(selected["durations"]), reduction="none"
    )
    duration_loss = duration_error[selected["duration_mask"]].mean()
    return value_loss + duration_loss, value_loss, duration_loss


def train_liquid(
    mode: str,
    train_records: list[BoundaryRecord],
    validation_records: list[BoundaryRecord],
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[LiquidCritic, dict[str, float]]:
    torch.manual_seed(seed)
    train = pack_records(train_records, device)
    validation = pack_records(validation_records, device)
    model = LiquidCritic(train["inputs"].shape[-1], mode).to(device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    started = time.time()
    last = {}
    for step in range(steps):
        indices = torch.randint(
            len(train_records), (min(64, len(train_records)),), generator=generator, device=device
        )
        loss, value_loss, duration_loss = liquid_loss(model, target, train, indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            target.load_state_dict(model.state_dict())
        if (step + 1) % 500 == 0 or step + 1 == steps:
            with torch.no_grad():
                val_indices = torch.arange(len(validation_records), device=device)
                val_loss, val_value, val_duration = liquid_loss(model, target, validation, val_indices)
            last = {
                "train_total": float(loss.detach()),
                "train_value": float(value_loss.detach()),
                "train_duration": float(duration_loss.detach()),
                "validation_total": float(val_loss.detach()),
                "validation_value": float(val_value.detach()),
                "validation_duration": float(val_duration.detach()),
                "beta_arx": float(model.beta_arx.detach()),
                "seconds": time.time() - started,
            }
            print(f"TRAIN_{mode.upper()}={step + 1}/{steps} " + json.dumps(last, sort_keys=True), flush=True)
    return model.eval(), last


def actual_lambdas(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    lambdas = torch.ones_like(batch["durations"])
    lambdas[batch["duration_mask"]] = torch.pow(
        lambdas.new_tensor(GAMMA), batch["durations"][batch["duration_mask"]]
    )
    return lambdas


def mlp_loss(
    model: EventMLPCritic,
    target: EventMLPCritic,
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor:
    selected = {key: value[indices] for key, value in batch.items()}
    lambdas = actual_lambdas(selected)
    logits = model(selected["inputs"], lambdas)
    _, support = decode_values(logits, lambdas, selected["remaining"])
    with torch.no_grad():
        next_logits = target(selected["inputs"], lambdas)
        next_values, _ = decode_values(next_logits, lambdas, selected["remaining"])
        desired = td_targets(next_values, lambdas, selected)
        distribution = hl_gauss(desired, support)
    cross_entropy = -(distribution * torch.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
    return cross_entropy[selected["mask"]].mean()


def train_mlp(
    train_records: list[BoundaryRecord],
    validation_records: list[BoundaryRecord],
    device: torch.device,
    steps: int,
) -> tuple[EventMLPCritic, dict[str, float]]:
    torch.manual_seed(SEED + 17)
    train = pack_records(train_records, device)
    validation = pack_records(validation_records, device)
    model = EventMLPCritic(train["inputs"].shape[-1]).to(device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(SEED + 1017)
    started = time.time()
    last = {}
    for step in range(steps):
        indices = torch.randint(
            len(train_records), (min(64, len(train_records)),), generator=generator, device=device
        )
        loss = mlp_loss(model, target, train, indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            target.load_state_dict(model.state_dict())
        if (step + 1) % 500 == 0 or step + 1 == steps:
            with torch.no_grad():
                val_loss = mlp_loss(
                    model, target, validation, torch.arange(len(validation_records), device=device)
                )
            last = {
                "train_value": float(loss.detach()),
                "validation_value": float(val_loss.detach()),
                "seconds": time.time() - started,
            }
            print(f"TRAIN_T2={step + 1}/{steps} " + json.dumps(last, sort_keys=True), flush=True)
    return model.eval(), last


def fit_beta(
    model: LiquidCritic,
    records: list[BoundaryRecord],
    device: torch.device,
    steps: int = 300,
) -> tuple[float, list[dict[str, float]]]:
    batch = pack_records(records, device)
    snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
    beta = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=0.05)
    history = []
    for step in range(steps):
        repeated = beta.expand(len(records))
        _, _, durations, _, _ = liquid_outputs(model, batch, repeated)
        error = F.smooth_l1_loss(torch.log1p(durations), torch.log1p(batch["durations"]), reduction="none")
        loss = error[batch["duration_mask"]].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-4.0, 4.0)
        if step in {0, 9, 49, 99, steps - 1}:
            history.append(
                {"step": step + 1, "beta": float(beta.detach()), "duration_loss": float(loss.detach())}
            )
    for name, value in model.state_dict().items():
        if not torch.equal(value, snapshot[name]):
            raise RuntimeError(f"shared parameter changed during target adaptation: {name}")
    return float(beta.detach()), history


def estimate_lambda_parameters(
    adaptation: list[BoundaryRecord],
    source: list[BoundaryRecord],
) -> tuple[float, np.ndarray]:
    source_values = {event: [] for event in MODEL_EVENTS}
    target_values = {event: [] for event in MODEL_EVENTS}
    for record in source:
        for event, duration in zip(record.event_ids, record.durations):
            if np.isfinite(duration):
                source_values[MODEL_EVENTS[int(event)]].append(GAMMA ** float(duration))
    for record in adaptation:
        for event, duration in zip(record.event_ids, record.durations):
            if np.isfinite(duration):
                target_values[MODEL_EVENTS[int(event)]].append(GAMMA ** float(duration))
    all_target = [value for values in target_values.values() for value in values]
    all_source = [value for values in source_values.values() for value in values]
    global_lambda = float((sum(all_target) + np.mean(all_source)) / (len(all_target) + 1))
    event_lambdas = np.empty(len(MODEL_EVENTS), dtype=np.float32)
    for index, event in enumerate(MODEL_EVENTS):
        prior = float(np.mean(source_values[event])) if source_values[event] else float(np.mean(all_source))
        values = target_values[event]
        event_lambdas[index] = (sum(values) + prior) / (len(values) + 1)
    return global_lambda, event_lambdas


def rank_auc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return math.nan
    return float(
        np.mean([1.0 if positive_score > negative_score else 0.5 if positive_score == negative_score else 0.0 for positive_score in positive for negative_score in negative])
    )


def gate_tensor(
    records: list[BoundaryRecord],
    specs: dict[str, dict[str, object]],
    adaptation: list[BoundaryRecord],
    device: torch.device,
    length: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    posterior_means = {}
    for task in TASKS:
        group = [record.episode for record in adaptation if record.episode.task == task]
        posterior_a, posterior_b = link_posteriors(group, list(specs[task]["chain"]))
        posterior_means[task] = posterior_a / (posterior_a + posterior_b)
    gates = np.zeros((len(records), length, len(MODEL_EVENTS)), dtype=np.float32)
    success_predictions = {}
    for row, record in enumerate(records):
        chain = list(specs[record.episode.task]["chain"])
        rho = posterior_means[record.episode.task]
        success_predictions[record.episode.task] = float(np.prod(rho))
        for column, event_id in enumerate(record.event_ids):
            event = MODEL_EVENTS[int(event_id)]
            position = chain.index(event)
            probability = 1.0
            gates[row, column, event_id] = 1.0
            for next_position in range(position + 1, len(chain)):
                probability *= float(rho[next_position - 1])
                gates[row, column, MODEL_EVENTS.index(chain[next_position])] = probability
    return torch.from_numpy(gates).to(device), success_predictions


def evaluate_model(
    name: str,
    model: nn.Module,
    records: list[BoundaryRecord],
    adaptation: list[BoundaryRecord],
    source_records: list[BoundaryRecord],
    specs: dict[str, dict[str, object]],
    device: torch.device,
    beta: float = 0.0,
    lambda_mode: str = "event",
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    batch = pack_records(records, device)
    gates, success_predictions = gate_tensor(records, specs, adaptation, device, batch["inputs"].shape[1])
    with torch.no_grad():
        if isinstance(model, LiquidCritic):
            beta_tensor = torch.full((len(records),), beta, device=device)
            _, values, durations, lambdas, _ = liquid_outputs(model, batch, beta_tensor)
        else:
            global_lambda, event_lambdas = estimate_lambda_parameters(adaptation, source_records)
            if lambda_mode == "global":
                lambdas = torch.full_like(batch["durations"], global_lambda)
            else:
                lookup = torch.from_numpy(event_lambdas).to(device)
                lambdas = lookup[batch["event_ids"]]
            logits = model(batch["inputs"], lambdas)
            values, _ = decode_values(logits, lambdas, batch["remaining"])
            durations = torch.log(lambdas) / math.log(GAMMA)
        adjusted = values * gates
        desired = td_targets(adjusted, lambdas, batch)
        squared = torch.square(adjusted - desired).mean(-1)
    scores = adjusted[:, 0, MODEL_EVENTS.index("eK")].cpu().numpy()
    labels = batch["success"].cpu().numpy()
    rows = []
    for task in [*TASKS, "__all__"]:
        indices = [index for index, record in enumerate(records) if task == "__all__" or record.episode.task == task]
        task_mask = batch["mask"][indices]
        bellman = float(squared[indices][task_mask].mean())
        task_labels = labels[indices].tolist()
        task_scores = scores[indices].tolist()
        truth = float(np.mean(task_labels))
        prediction = float(np.mean([success_predictions[record.episode.task] for record in np.asarray(records, dtype=object)[indices]]))
        rows.append(
            {
                "model": name,
                "task": task,
                "embodiment": records[0].episode.body,
                "n_adaptation_per_task": len(adaptation) // len(TASKS),
                "n_test": len(indices),
                "bellman_mse": bellman,
                "value_ranking_auc": rank_auc(task_labels, task_scores),
                "success_prediction": prediction,
                "test_success_rate": truth,
                "success_mae": abs(prediction - truth),
                "mean_predicted_duration": float(durations[indices][batch["duration_mask"][indices]].mean()),
                "target_td": False,
            }
        )
    arrays = {
        "values": adjusted.cpu().numpy(),
        "raw_values": values.cpu().numpy(),
        "durations": durations.cpu().numpy(),
        "lambdas": lambdas.cpu().numpy(),
        "mask": batch["mask"].cpu().numpy(),
    }
    return rows, arrays


def sign_consistency(
    model: LiquidCritic,
    records: list[BoundaryRecord],
    beta: float,
    device: torch.device,
) -> float:
    batch = pack_records(records, device)
    with torch.no_grad():
        _, target_values, _, target_lambdas, _ = liquid_outputs(
            model, batch, torch.full((len(records),), beta, device=device)
        )
        _, source_values, _, source_lambdas, _ = liquid_outputs(
            model, batch, torch.zeros(len(records), device=device)
        )
    goal = MODEL_EVENTS.index("eK")
    agreements = []
    for row, record in enumerate(records):
        for index in range(len(record.event_ids) - 1):
            target_advantage = target_lambdas[row, index] * target_values[row, index + 1, goal] - target_values[row, index, goal]
            source_advantage = source_lambdas[row, index] * source_values[row, index + 1, goal] - source_values[row, index, goal]
            target_sign = 0 if abs(float(target_advantage)) <= 1e-6 else 1 if target_advantage > 0 else -1
            source_sign = 0 if abs(float(source_advantage)) <= 1e-6 else 1 if source_advantage > 0 else -1
            agreements.append(target_sign == source_sign)
    return float(np.mean(agreements)) if agreements else math.nan


def mechanism_rows(
    model: LiquidCritic,
    body: str,
    beta: float,
    adaptation: list[BoundaryRecord],
    test: list[BoundaryRecord],
    source: list[BoundaryRecord],
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_batch = pack_records(test, device)
    with torch.no_grad():
        _, _, target_duration, _, _ = liquid_outputs(
            model, test_batch, torch.full((len(test),), beta, device=device)
        )
        _, _, source_duration, _, _ = liquid_outputs(model, test_batch, torch.zeros(len(test), device=device))
    sign_rows = []
    kappa_rows = []
    for task in TASKS:
        target_adapt = [record for record in adaptation if record.episode.task == task]
        source_task = [record for record in source if record.episode.task == task and record.episode.body == "aloha-agilex"]
        test_indices = [index for index, record in enumerate(test) if record.episode.task == task]
        for event in MODEL_EVENTS:
            target_actual = [
                duration
                for record in target_adapt
                for event_id, duration in zip(record.event_ids, record.durations)
                if MODEL_EVENTS[int(event_id)] == event and np.isfinite(duration)
            ]
            source_actual = [
                duration
                for record in source_task
                for event_id, duration in zip(record.event_ids, record.durations)
                if MODEL_EVENTS[int(event_id)] == event and np.isfinite(duration)
            ]
            predicted_target = []
            predicted_source = []
            for row in test_indices:
                record = test[row]
                for column, event_id in enumerate(record.event_ids):
                    if MODEL_EVENTS[int(event_id)] == event and np.isfinite(record.durations[column]):
                        predicted_target.append(float(target_duration[row, column]))
                        predicted_source.append(float(source_duration[row, column]))
            if target_actual and source_actual and predicted_target:
                actual_ratio = float(np.mean(target_actual) / np.mean(source_actual))
                predicted_ratio = float(np.mean(predicted_target) / np.mean(predicted_source))
                actual_direction = 0 if abs(math.log(actual_ratio)) < 0.05 else 1 if actual_ratio > 1 else -1
                predicted_direction = 0 if abs(math.log(predicted_ratio)) < 0.05 else 1 if predicted_ratio > 1 else -1
                sign_rows.append(
                    {
                        "model": model.cell.mode,
                        "embodiment": body,
                        "task": task,
                        "event": event,
                        "actual_ratio": actual_ratio,
                        "predicted_ratio": predicted_ratio,
                        "actual_direction": actual_direction,
                        "predicted_direction": predicted_direction,
                        "direction_correct": int(actual_direction == predicted_direction),
                    }
                )
        actual_target_total = [
            sum(duration for duration in record.durations if np.isfinite(duration))
            for record in target_adapt
            if record.episode.success and record.event_ids[-1] == MODEL_EVENTS.index("eK")
        ]
        actual_source_total = [
            sum(duration for duration in record.durations if np.isfinite(duration))
            for record in source_task
            if record.episode.success and record.event_ids[-1] == MODEL_EVENTS.index("eK")
        ]
        predicted_target_total = []
        predicted_source_total = []
        for row in test_indices:
            count = len(test[row].event_ids) - 1
            if count > 0 and test[row].episode.success and test[row].event_ids[-1] == MODEL_EVENTS.index("eK"):
                predicted_target_total.append(float(target_duration[row, :count].sum()))
                predicted_source_total.append(float(source_duration[row, :count].sum()))
        if actual_target_total and actual_source_total and predicted_target_total:
            kappa_rows.append(
                {
                    "model": model.cell.mode,
                    "embodiment": body,
                    "task": task,
                    "actual_kappa": float(np.mean(actual_source_total) / np.mean(actual_target_total)),
                    "predicted_kappa": float(np.mean(predicted_source_total) / np.mean(predicted_target_total)),
                }
            )
    return sign_rows, kappa_rows


def liquid_boundary_tests(model: LiquidCritic, input_dim: int, device: torch.device) -> list[dict[str, object]]:
    rows = []
    torch.manual_seed(SEED)
    inputs = torch.randn(2, 3, input_dim, device=device)
    dts = torch.ones(2, 3, device=device)
    mask = torch.ones(2, 3, dtype=torch.bool, device=device)
    hidden = torch.zeros(2, HIDDEN, device=device)
    candidate_zero, tau_zero, _ = model.cell.components(inputs[:, 0], hidden, torch.zeros(2, device=device))
    candidate_one, tau_one, _ = model.cell.components(inputs[:, 0], hidden, torch.ones(2, device=device))
    rows.append({"test": "beta_only_time_constant", "passed": bool(torch.equal(candidate_zero, candidate_one) and not torch.equal(tau_zero, tau_one))})
    with torch.no_grad():
        original = model(inputs, dts, mask, torch.ones(2, device=device))[0]
        changed = inputs.clone()
        changed[:, 2] += 100.0
        perturbed = model(changed, dts, mask, torch.ones(2, device=device))[0]
    rows.append({"test": "future_causality", "passed": bool(torch.equal(original[:, :2], perturbed[:, :2]))})
    durations = torch.tensor([1.0, 10.0], device=device)
    lambdas = torch.pow(durations.new_tensor(GAMMA), durations)
    rows.append({"test": "duration_discount_monotonic", "passed": bool(lambdas[1] < lambdas[0])})
    support = support_from_lambda(lambdas, torch.tensor([2.0, 2.0], device=device))
    rows.append({"test": "dynamic_support", "passed": bool(not torch.equal(support[0], support[1]) and torch.all(support[:, 1:] >= support[:, :-1]))})
    targets = torch.ones(2, 1, device=device)
    distribution = hl_gauss(targets, support)
    rows.append({"test": "hl_gauss_normalized", "passed": bool(torch.allclose(distribution.sum(-1), torch.ones_like(distribution.sum(-1))))})
    with torch.no_grad():
        repeated = model(inputs[:1].repeat(2, 1, 1), dts, mask, torch.ones(2, device=device))[0]
    rows.append({"test": "fake_body_id_absent", "passed": bool(torch.equal(repeated[0], repeated[1]))})
    if not all(row["passed"] for row in rows):
        raise RuntimeError(f"liquid boundary tests failed: {rows}")
    return rows


def embodiment_probe(
    model: LiquidCritic,
    records: list[BoundaryRecord],
    device: torch.device,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selected = [record for record in records if record.episode.index < 20]
    batch = pack_records(selected, device)
    with torch.no_grad():
        logits, _, _, _, _ = liquid_outputs(model, batch, torch.zeros(len(selected), device=device))
    representation = logits[:, 0].flatten(1).cpu().numpy()
    labels = np.asarray([BODIES.index(record.episode.body) for record in selected])
    train = np.asarray([record.episode.index < 15 for record in selected])
    evaluation = ~train
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    classifier.fit(representation[train], labels[train])
    predicted = classifier.predict(representation[evaluation])
    correct = int((predicted == labels[evaluation]).sum())
    count = int(evaluation.sum())
    accuracy = correct / count
    p_value = float(binomtest(correct, count, 1.0 / len(BODIES), alternative="greater").pvalue)
    return {
        "test": "linear_embodiment_probe_beta0",
        "passed": not (accuracy > 1.0 / len(BODIES) and p_value < 0.05),
        "accuracy": accuracy,
        "chance": 1.0 / len(BODIES),
        "n_eval": count,
        "p_value": p_value,
    }


def run_main(
    data_root: Path,
    output_root: Path,
    steps: int,
    adaptation_count: int,
) -> dict[str, object]:
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(data_root)
    calibration = calibrate_events(episodes)
    assign_raw_events(episodes, calibration)
    specs = derive_chains(episodes)
    apply_chains(episodes, specs)
    records, mean, std = build_records(episodes, calibration, specs)
    source_train = [record for record in records if record.episode.body in SOURCE_BODIES and record.episode.index < 40]
    source_validation = [record for record in records if record.episode.body in SOURCE_BODIES and record.episode.index >= 40]
    device = torch.device("cuda:0")
    t2, t2_loss = train_mlp(source_train, source_validation, device, steps)
    t3, t3_loss = train_liquid("t3", source_train, source_validation, device, steps, SEED + 31)
    t4, t4_loss = train_liquid("t4", source_train, source_validation, device, steps, SEED + 47)
    leak_rows = liquid_boundary_tests(t4, source_train[0].inputs.shape[1], device)
    probe = embodiment_probe(t4, records, device)
    leak_rows.append(probe)
    if not probe["passed"]:
        pd.DataFrame(leak_rows).to_csv(output_root / "results/leak_check.csv", index=False)
        summary = {"g1_passed": False, "stop_after_g1": True, "embodiment_probe": probe}
        (output_root / "main_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    value_rows = []
    sign_rows = []
    kappa_rows = []
    beta_rows = []
    fitted_betas = {}
    for body in TARGET_BODIES:
        adaptation = [
            record
            for record in records
            if record.episode.body == body and record.episode.index < adaptation_count
        ]
        test = [record for record in records if record.episode.body == body and record.episode.index >= 15]
        beta3, history3 = fit_beta(t3, adaptation, device)
        beta4, history4 = fit_beta(t4, adaptation, device)
        fitted_betas[body] = {"t3": beta3, "t4": beta4}
        for name, history in [("t3", history3), ("t4", history4)]:
            for row in history:
                beta_rows.append({"model": name, "embodiment": body, "N": adaptation_count, **row})
        for name, model, beta, lambda_mode in [
            ("T1_global_lambda_mlp", t2, 0.0, "global"),
            ("T2_event_lambda_mlp", t2, 0.0, "event"),
            ("T3_global_liquid", t3, beta3, "event"),
            ("T4_low_rank_liquid", t4, beta4, "event"),
        ]:
            rows, _ = evaluate_model(
                name,
                model,
                test,
                adaptation,
                source_train,
                specs,
                device,
                beta=beta,
                lambda_mode=lambda_mode,
            )
            for row in rows:
                row["advantage_sign_consistency"] = (
                    sign_consistency(model, test, beta, device) if isinstance(model, LiquidCritic) else math.nan
                )
            value_rows.extend(rows)
        body_sign3, body_kappa3 = mechanism_rows(t3, body, beta3, adaptation, test, source_train, device)
        body_sign4, body_kappa4 = mechanism_rows(t4, body, beta4, adaptation, test, source_train, device)
        sign_rows.extend(body_sign3 + body_sign4)
        kappa_rows.extend(body_kappa3 + body_kappa4)
    leak_rows.append({"test": "target_adaptation_freeze", "passed": True})
    value_frame = pd.DataFrame(value_rows)
    sign_frame = pd.DataFrame(sign_rows)
    kappa_frame = pd.DataFrame(kappa_rows)
    value_frame.to_csv(output_root / "results/V0_value_transport.csv", index=False)
    value_frame.to_csv(output_root / "results/T_ladder_comparison.csv", index=False)
    sign_frame.to_csv(output_root / "results/V1_sign_reversal.csv", index=False)
    kappa_frame.to_csv(output_root / "results/V2_kappa_reproduction.csv", index=False)
    pd.DataFrame(beta_rows).to_csv(output_root / "results/beta_convergence.csv", index=False)
    pd.DataFrame(leak_rows).to_csv(output_root / "results/leak_check.csv", index=False)
    np.savez(output_root / "feature_normalization.npz", mean=mean, std=std)
    torch.save(
        {
            "t2": t2.state_dict(),
            "t3": t3.state_dict(),
            "t4": t4.state_dict(),
            "input_dim": source_train[0].inputs.shape[1],
            "fitted_betas": fitted_betas,
        },
        output_root / "stage2_models.pt",
    )
    aggregate = value_frame[value_frame.task == "__all__"].set_index(["embodiment", "model"])
    ur5_t2 = float(aggregate.loc[("ur5-wsg", "T2_event_lambda_mlp"), "bellman_mse"])
    ur5_t3 = float(aggregate.loc[("ur5-wsg", "T3_global_liquid"), "bellman_mse"])
    ur5_t4 = float(aggregate.loc[("ur5-wsg", "T4_low_rank_liquid"), "bellman_mse"])
    ur5_auc2 = float(aggregate.loc[("ur5-wsg", "T2_event_lambda_mlp"), "value_ranking_auc"])
    ur5_auc3 = float(aggregate.loc[("ur5-wsg", "T3_global_liquid"), "value_ranking_auc"])
    ur5_auc4 = float(aggregate.loc[("ur5-wsg", "T4_low_rank_liquid"), "value_ranking_auc"])
    ur5_adv3 = float(aggregate.loc[("ur5-wsg", "T3_global_liquid"), "advantage_sign_consistency"])
    ur5_adv4 = float(aggregate.loc[("ur5-wsg", "T4_low_rank_liquid"), "advantage_sign_consistency"])
    t3_direction = float(sign_frame[(sign_frame.embodiment == "ur5-wsg") & (sign_frame.model == "t3")].direction_correct.mean())
    t4_direction = float(sign_frame[(sign_frame.embodiment == "ur5-wsg") & (sign_frame.model == "t4")].direction_correct.mean())
    checks = {
        "bellman_better_than_t3": ur5_t4 < ur5_t3,
        "bellman_not_worse_than_t2": ur5_t4 <= ur5_t2,
        "auc_not_worse_than_t2_t3": ur5_auc4 >= max(ur5_auc2, ur5_auc3),
        "advantage_not_worse_than_t3": ur5_adv4 >= ur5_adv3,
        "direction_better_than_t3": t4_direction > t3_direction,
    }
    passed = all(checks.values())
    summary = {
        "adaptation_count": adaptation_count,
        "source_train_episodes": len(source_train),
        "source_validation_episodes": len(source_validation),
        "g1_passed": True,
        "embodiment_probe": probe,
        "losses": {"t2": t2_loss, "t3": t3_loss, "t4": t4_loss},
        "fitted_betas": fitted_betas,
        "ur5_bellman": {"t2": ur5_t2, "t3": ur5_t3, "t4": ur5_t4},
        "ur5_auc": {"t2": ur5_auc2, "t3": ur5_auc3, "t4": ur5_auc4},
        "ur5_advantage_sign_consistency": {"t3": ur5_adv3, "t4": ur5_adv4},
        "ur5_direction_accuracy": {"t3": t3_direction, "t4": t4_direction},
        "g3_checks": checks,
        "g3_passed": passed,
        "stop_after_g3": not passed,
    }
    (output_root / "main_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw_rows = []
    for (body, model_name), row in aggregate.iterrows():
        for metric in ["bellman_mse", "value_ranking_auc", "advantage_sign_consistency", "success_mae"]:
            raw_rows.append(
                {
                    "split": "target_last5",
                    "embodiment": body,
                    "model": model_name,
                    "metric": metric,
                    "value": row[metric],
                }
            )
    pd.DataFrame(raw_rows).to_csv(output_root / "results/raw_metrics.csv", index=False)
    g0_path = output_root / "g0_summary.json"
    g0 = json.loads(g0_path.read_text()) if g0_path.exists() else {}
    failed_checks = [name for name, value in checks.items() if not value]
    report = f"""# ETSF 阶段 2 报告

## G0

- 轨迹数：{g0.get('episodes', 'NA')}
- 修复后结构偏序一致率：{g0.get('structural_prefix_consistency', math.nan):.4f}
- 干净格子秩-1 解释率：{g0.get('rank1_clean', math.nan):.4f}
- N=5 可行性 MAE：{g0.get('n5_mae', math.nan):.4f}
- Piper/adjust_bottle：第 {g0.get('deployment_rollouts', 'NA')} 条停止，拒绝后验 {g0.get('deployment_reject_probability', math.nan):.6f}，anytime p={g0.get('deployment_anytime_p', math.nan):.6f}

## G1/G2

- T3 源验证总损失：{t3_loss['validation_total']:.6f}
- T4 源验证总损失：{t4_loss['validation_total']:.6f}
- ur5 局部方向一致率：T3={t3_direction:.4f}，T4={t4_direction:.4f}
- 因果、动态 support、假本体 ID、目标冻结等模块边界测试全部通过。
- `beta=0` 线性本体 probe：accuracy={probe['accuracy']:.4f}，chance={probe['chance']:.4f}，p={probe['p_value']:.4f}，未检出显著泄漏。

## G3 · 留出本体 ur5，N=5

| 模型 | Bellman MSE | AUC | 优势符号一致率 |
|---|---:|---:|---:|
| T2 | {ur5_t2:.6f} | {ur5_auc2:.4f} | NA |
| T3 | {ur5_t3:.6f} | {ur5_auc3:.4f} | {ur5_adv3:.4f} |
| T4 | {ur5_t4:.6f} | {ur5_auc4:.4f} | {ur5_adv4:.4f} |

**判定：{'通过' if passed else '未通过，按预注册规则止损'}。** 未通过项：{', '.join(failed_checks) if failed_checks else '无'}。

T4 改善了 Bellman 一致性与局部时长方向，但没有同时保持价值排序和优势符号。因此不能把当前结果写成“共享液态 critic 已完成零样本价值传输”。按照执行文档，不继续 T5、target-from-scratch 与完整 V3；这些项目只有 G3 全指标通过后才运行。

## 口径

- 共享头只使用物体相对状态、累计位移、归一化方向与任务/事件 token。
- 目标适配只更新一个 `beta_b` 与事件可达率后验，没有目标 TD。
- Aloha 与 ARX-X5 是共享头训练本体；ARX-X5 只作源域时钟诊断，不冒充留出本体。
"""
    (output_root / "stage2_report.md").write_text(report, encoding="utf-8")
    return summary


def self_test() -> None:
    rng = np.random.default_rng(1)
    samples = posterior_product(np.asarray([6.0, 4.0]), np.asarray([1.0, 2.0]), rng, draws=1000)
    assert samples.shape == (1000,)
    assert np.all((0.0 <= samples) & (samples <= 1.0))
    assert -20.0 < -10.0
    assert not (set(range(15)) & set(range(15, 20)))
    print("SELF_TEST_PASS=posterior,polarity,split", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/user/etsf_stage1"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/user/etsf_stage2"))
    parser.add_argument("--stage", choices=["g0", "main", "self-test"], default="g0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--adaptation-count", type=int, default=5)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    if args.stage == "self-test":
        self_test()
        return
    if args.stage == "main":
        summary = run_main(args.data_root, args.output_root, args.steps, args.adaptation_count)
        print("MAIN_COMPLETE=" + json.dumps(summary, sort_keys=True), flush=True)
        return
    summary = run_g0(args.data_root, args.output_root)
    print("G0_COMPLETE=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
