#!/usr/bin/env python3
"""Run the frozen-feature ETSF Stage-1 study."""

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

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_distribution
from torch import nn


ROOT = Path("/home/user/etsf_stage1")
TASKS = [
    "adjust_bottle",
    "handover_block",
    "move_can_pot",
    "place_container_plate",
    "beat_block_hammer",
    "lift_pot",
]
BODIES = ["aloha-agilex", "ARX-X5", "piper", "ur5-wsg"]
SOURCE_BODIES = {"aloha-agilex", "ARX-X5"}
TARGET_BODIES = ["piper", "ur5-wsg"]
EVENTS = ["e0", "e1", "e2", "e3", "e4", "eK"]
LAMBDAS = np.asarray([0.5, 0.7, 0.9, 0.95, 0.99], dtype=np.float32)
GAMMA = 0.99
RHO_FAIL = 0.30
RHO_DEPLOY = 0.70
SEQUENTIAL_ALPHA = 0.05
HIDDEN = 256
STEP_SCALE = 400.0
BACKBONE = "vit_small_patch14_dinov2.lvd142m"
CAMERAS = ["head_camera", "left_camera", "right_camera"]


@dataclass
class Episode:
    task: str
    body: str
    index: int
    success: bool
    poses: np.ndarray
    names: list[str]
    image_path: Path
    feature_path: Path
    events: dict[int, int] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.poses.shape[0])


class Head(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, len(EVENTS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_episodes() -> list[Episode]:
    episodes: list[Episode] = []
    feature_root = ROOT / "features"
    for task in TASKS:
        for body in BODIES:
            if body in SOURCE_BODIES:
                paths = sorted((ROOT / "source_object_poses" / task / body).glob("episode_*.npz"))
                for path in paths:
                    index = int(path.stem.rsplit("_", 1)[1])
                    with np.load(path) as data:
                        poses = data["poses"].astype(np.float32)
                        names = [str(value) for value in data["object_names"]]
                        success = bool(data["success"])
                    episodes.append(
                        Episode(
                            task,
                            body,
                            index,
                            success,
                            poses,
                            names,
                            ROOT / "source_data" / task / body / "data" / f"episode{index}.hdf5",
                            feature_root / task / body / f"episode_{index:06d}.npz",
                        )
                    )
            else:
                paths = sorted((ROOT / "target_rollouts" / task / body).glob("episode_*.hdf5"))
                for path in paths:
                    with h5py.File(path, "r") as handle:
                        index = int(handle.attrs["rollout_index"])
                        episodes.append(
                            Episode(
                                task,
                                body,
                                index,
                                bool(handle.attrs["success"]),
                                handle["object_poses"][:].astype(np.float32),
                                [value.decode() for value in handle["object_names"][:]],
                                path,
                                feature_root / task / body / f"episode_{index:06d}.npz",
                            )
                        )
    return episodes


def event_calibration(episodes: list[Episode]) -> dict[str, dict[str, object]]:
    calibration: dict[str, dict[str, object]] = {}
    for task in TASKS:
        task_eps = [episode for episode in episodes if episode.task == task]
        names = sorted(set.intersection(*(set(episode.names) for episode in task_eps)))
        displacement: dict[str, float] = {}
        for name in names:
            values = []
            for episode in task_eps:
                position = episode.poses[:, episode.names.index(name), :3]
                values.append(float(np.linalg.norm(position - position[0], axis=1).max()))
            displacement[name] = float(np.median(values))
        moving = max(displacement, key=displacement.get)
        successful = [episode for episode in task_eps if episode.success]
        endpoints = np.stack(
            [episode.poses[-1, episode.names.index(moving), :3] for episode in successful]
        )
        absolute_variance = float(np.square(endpoints - endpoints.mean(0)).sum(axis=1).mean())
        anchor = ""
        offset = np.zeros(3, dtype=np.float32)
        best_variance = math.inf
        for name in names:
            if name == moving or name in {"table", "wall"}:
                continue
            if displacement[name] > 0.2 * displacement[moving]:
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
            center = endpoints.mean(0, keepdims=True)
            sse_one = float(np.square(endpoints - center).sum())
            covariance = np.cov(endpoints.T)
            direction = np.linalg.eigh(covariance)[1][:, -1]
            projection = endpoints @ direction
            centers = np.stack([endpoints[projection.argmin()], endpoints[projection.argmax()]])
            for _ in range(30):
                labels = np.square(endpoints[:, None, :] - centers[None, :, :]).sum(2).argmin(1)
                if min(np.bincount(labels, minlength=2)) == 0:
                    break
                updated = np.stack([endpoints[labels == label].mean(0) for label in range(2)])
                if np.allclose(updated, centers):
                    centers = updated
                    break
                centers = updated
            labels = np.square(endpoints[:, None, :] - centers[None, :, :]).sum(2).argmin(1)
            sse_two = float(np.square(endpoints - centers[labels]).sum())
            if sse_two >= 0.35 * max(sse_one, 1e-12) or min(np.bincount(labels, minlength=2)) < 0.15 * len(endpoints):
                centers = center
        else:
            centers = np.empty((0, 3), dtype=np.float32)

        speeds = []
        rises = []
        final_distances = []
        for episode in task_eps:
            position = episode.poses[:, episode.names.index(moving), :3]
            speeds.extend(np.linalg.norm(np.diff(position, axis=0), axis=1))
            if episode.success:
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
            "centers": centers.astype(np.float32),
            "tau_v": max(0.0005, float(np.quantile(np.asarray(speeds), 0.85))),
            "delta_z": max(0.005, 0.2 * float(np.median(rises))),
            "tau_d": max(0.015, float(np.quantile(final_distances, 0.95)) + 0.005),
            "k": 3,
        }
    return calibration


def assign_events(episodes: list[Episode], calibration: dict[str, dict[str, object]]) -> None:
    for episode in episodes:
        config = calibration[episode.task]
        moving = str(config["moving"])
        position = episode.poses[:, episode.names.index(moving), :3]
        speed = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
        if config["anchor"]:
            anchor_position = episode.poses[:, episode.names.index(str(config["anchor"])), :3]
            target = anchor_position + np.asarray(config["offset"])[None, :]
            distance = np.linalg.norm(position - target, axis=1)
        else:
            centers = np.asarray(config["centers"])
            distance = np.linalg.norm(position[:, None, :] - centers[None, :, :], axis=2).min(1)
        event_times = {0: 0}
        candidates = np.flatnonzero(speed > float(config["tau_v"]))
        if candidates.size:
            event_times[1] = int(candidates[0])
        candidates = np.flatnonzero(position[:, 2] > position[0, 2] + float(config["delta_z"]))
        if candidates.size:
            event_times[2] = int(candidates[0])
        candidates = np.flatnonzero(distance < float(config["tau_d"]))
        if candidates.size:
            event_times[3] = int(candidates[0])
        stationary = (distance < float(config["tau_d"])) & (speed < float(config["tau_v"]))
        k = int(config["k"])
        for index in range(episode.steps - k + 1):
            if stationary[index : index + k].all():
                event_times[4] = index + k - 1
                break
        if episode.success:
            event_times[5] = episode.steps - 1
        episode.events = dict(sorted(event_times.items(), key=lambda item: (item[1], item[0])))


def write_m1(episodes: list[Episode]) -> pd.DataFrame:
    modes: dict[tuple[str, str], str] = {}
    for task in TASKS:
        for body in BODIES:
            seqs = [tuple(EVENTS[index] for index in episode.events) for episode in episodes if episode.task == task and episode.body == body]
            modes[(task, body)] = ">".join(Counter(seqs).most_common(1)[0][0])
    rows = []
    for task in TASKS:
        shared = len({modes[(task, body)] for body in BODIES}) == 1
        sequences = [modes[(task, body)].split(">") for body in BODIES]
        prefix_compatible = all(
            left[: min(len(left), len(right))] == right[: min(len(left), len(right))]
            for left in sequences
            for right in sequences
        )
        for body in BODIES:
            body_eps = [episode for episode in episodes if episode.task == task and episode.body == body]
            mode_indices = [EVENTS.index(event) for event in modes[(task, body)].split(">")]
            canonical_rate = float(
                np.mean(
                    [
                        list(episode.events) == sorted(episode.events)
                        for episode in body_eps
                    ]
                )
            )
            for event_index, event in enumerate(EVENTS):
                previous = event_index - 1
                eligible = len(body_eps) if event_index == 0 else sum(previous in episode.events for episode in body_eps)
                reached = (
                    len(body_eps)
                    if event_index == 0
                    else sum(
                        event_index in episode.events
                        and previous in episode.events
                        and episode.events[event_index] >= episode.events[previous]
                        for episode in body_eps
                    )
                )
                intervals = (
                    []
                    if event_index == 0
                    else [
                        episode.events[event_index] - episode.events[previous]
                        for episode in body_eps
                        if event_index in episode.events
                        and previous in episode.events
                        and episode.events[event_index] >= episode.events[previous]
                    ]
                )
                rows.append(
                    {
                        "task": task,
                        "embodiment": body,
                        "event": event,
                        "transition": "start" if event_index == 0 else f"{EVENTS[previous]}->{event}",
                        "n_rollouts": len(body_eps),
                        "mode_sequence": modes[(task, body)],
                        "mode_count": sum(
                            ">".join(EVENTS[index] for index in episode.events) == modes[(task, body)] for episode in body_eps
                        ),
                        "mode_is_canonical_order": int(mode_indices == sorted(mode_indices)),
                        "all_rollout_canonical_order_rate": canonical_rate,
                        "cross_body_mode_consistent": int(shared),
                        "cross_body_mode_prefix_compatible": int(prefix_compatible),
                        "eligible": eligible,
                        "reached": reached,
                        "order_violations": 0
                        if event_index == 0
                        else sum(
                            event_index in episode.events
                            and previous in episode.events
                            and episode.events[event_index] < episode.events[previous]
                            for episode in body_eps
                        ),
                        "reach_rate": reached / eligible if eligible else math.nan,
                        "interval_n": len(intervals),
                        "interval_mean": float(np.mean(intervals)) if intervals else math.nan,
                        "interval_std": float(np.std(intervals)) if intervals else math.nan,
                        "interval_q25": float(np.quantile(intervals, 0.25)) if intervals else math.nan,
                        "interval_median": float(np.median(intervals)) if intervals else math.nan,
                        "interval_q75": float(np.quantile(intervals, 0.75)) if intervals else math.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / "results/M1_event_consistency.csv", index=False)
    return frame


def estimate_parameters(episodes: list[Episode], count: int) -> tuple[float, np.ndarray, list[int]]:
    selected = sorted(episodes, key=lambda episode: episode.index)[:count]
    rho = np.ones(len(EVENTS), dtype=np.float32)
    intervals: list[int] = []
    for event_index in range(1, len(EVENTS)):
        eligible = sum(event_index - 1 in episode.events for episode in selected)
        reached = sum(
            event_index in episode.events
            and event_index - 1 in episode.events
            and episode.events[event_index] >= episode.events[event_index - 1]
            for episode in selected
        )
        rho[event_index] = (reached + 1) / (eligible + 2)
        intervals.extend(
            episode.events[event_index] - episode.events[event_index - 1]
            for episode in selected
            if event_index in episode.events
            and event_index - 1 in episode.events
            and episode.events[event_index] >= episode.events[event_index - 1]
        )
    lam = float(np.mean(np.power(GAMMA, intervals))) if intervals else math.nan
    return lam, rho, intervals


def write_parameters(episodes: list[Episode]) -> pd.DataFrame:
    rows = []
    for body in BODIES:
        for count in [5, 10, 15, 20]:
            groups = [(task, [episode for episode in episodes if episode.task == task and episode.body == body]) for task in TASKS]
            groups.append(("__all__", [episode for episode in episodes if episode.body == body]))
            for task, group in groups:
                if task == "__all__":
                    selected = []
                    for name in TASKS:
                        selected.extend(sorted((episode for episode in group if episode.task == name), key=lambda episode: episode.index)[:count])
                    lam, rho, intervals = estimate_parameters(selected, len(selected))
                else:
                    lam, rho, intervals = estimate_parameters(group, count)
                rows.append(
                    {
                        "task": task,
                        "embodiment": body,
                        "N": count,
                        "parameter": "lambda",
                        "event": "all_intervals",
                        "estimate": lam,
                        "posterior_a": math.nan,
                        "posterior_b": math.nan,
                        "posterior_ci_low": math.nan,
                        "posterior_ci_high": math.nan,
                        "probability_above_deploy": math.nan,
                        "probability_below_fail": math.nan,
                        "reached": len(intervals),
                        "eligible": len(intervals),
                    }
                )
                if task == "__all__":
                    selected_group = []
                    for name in TASKS:
                        selected_group.extend(sorted((episode for episode in group if episode.task == name), key=lambda episode: episode.index)[:count])
                else:
                    selected_group = sorted(group, key=lambda episode: episode.index)[:count]
                for event_index in range(1, len(EVENTS)):
                    eligible = sum(event_index - 1 in episode.events for episode in selected_group)
                    reached = sum(
                        event_index in episode.events
                        and event_index - 1 in episode.events
                        and episode.events[event_index] >= episode.events[event_index - 1]
                        for episode in selected_group
                    )
                    posterior_a = reached + 1
                    posterior_b = eligible - reached + 1
                    rows.append(
                        {
                            "task": task,
                            "embodiment": body,
                            "N": count,
                            "parameter": "rho",
                            "event": EVENTS[event_index],
                            "estimate": posterior_a / (posterior_a + posterior_b),
                            "posterior_a": posterior_a,
                            "posterior_b": posterior_b,
                            "posterior_ci_low": beta_distribution.ppf(0.025, posterior_a, posterior_b),
                            "posterior_ci_high": beta_distribution.ppf(0.975, posterior_a, posterior_b),
                            "probability_above_deploy": beta_distribution.sf(RHO_DEPLOY, posterior_a, posterior_b),
                            "probability_below_fail": beta_distribution.cdf(RHO_FAIL, posterior_a, posterior_b),
                            "reached": reached,
                            "eligible": eligible,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/lambda_rho_estimates.csv", index=False)
    return frame


def extract_features(episodes: list[Episode]) -> None:
    pending = [episode for episode in episodes if not episode.feature_path.exists()]
    if not pending:
        return
    device = torch.device("cuda:0")
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=224).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.time()
    for number, episode in enumerate(pending, 1):
        view_features = []
        with h5py.File(episode.image_path, "r") as handle:
            for camera in CAMERAS:
                dataset = handle[f"observation/{camera}/rgb"] if episode.body in SOURCE_BODIES else handle[f"images/{camera}"]
                outputs = []
                for start in range(0, len(dataset), 128):
                    images = []
                    for payload in dataset[start : start + 128]:
                        encoded = np.frombuffer(payload.tobytes(), dtype=np.uint8) if episode.body in SOURCE_BODIES else np.asarray(payload, dtype=np.uint8)
                        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    tensor = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).float().div_(255)
                    tensor = F.interpolate(tensor, size=(224, 224), mode="bicubic", align_corners=False)
                    mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                    std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs.append(model((tensor - mean) / std).float().cpu().numpy().astype(np.float16))
                view_features.append(np.concatenate(outputs))
        if {array.shape[0] for array in view_features} != {episode.steps}:
            raise RuntimeError(f"feature/pose frame mismatch: {episode.task}/{episode.body}/{episode.index}")
        episode.feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(episode.feature_path, features=np.concatenate(view_features, axis=1))
        print(
            f"FEATURE={number}/{len(pending)} task={episode.task} body={episode.body} episode={episode.index} "
            f"frames={episode.steps} elapsed={time.time() - started:.1f}",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()


def load_feature_cache(episodes: list[Episode]) -> tuple[dict[tuple[str, str, int], np.ndarray], np.ndarray, np.ndarray]:
    cache = {}
    total = 0
    sum_x = None
    sum_x2 = None
    for episode in episodes:
        with np.load(episode.feature_path) as data:
            feature = data["features"].astype(np.float32)
        cache[(episode.task, episode.body, episode.index)] = feature
        if episode.body in SOURCE_BODIES:
            if sum_x is None:
                sum_x = np.zeros(feature.shape[1], dtype=np.float64)
                sum_x2 = np.zeros(feature.shape[1], dtype=np.float64)
            sum_x += feature.sum(0)
            sum_x2 += np.square(feature).sum(0)
            total += len(feature)
    assert sum_x is not None and sum_x2 is not None
    mean = (sum_x / total).astype(np.float32)
    std = np.sqrt(np.maximum(sum_x2 / total - np.square(mean), 1e-8)).astype(np.float32)
    for key in cache:
        cache[key] = ((cache[key] - mean) / std).astype(np.float16)
    return cache, mean, std


def make_input(feature: torch.Tensor, task: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(task.long(), num_classes=len(TASKS)).float()
    return torch.cat([feature.float(), one_hot, condition[:, None].float()], dim=1)


def train_etsf(
    episodes: list[Episode],
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[Head, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    current_features = []
    next_features = []
    task_ids = []
    phis = []
    dones = []
    for episode in episodes:
        feature = features[(episode.task, episode.body, episode.index)]
        occurrences = list(episode.events.items())
        for index, (event, frame_index) in enumerate(occurrences):
            next_index = occurrences[index + 1][1] if index + 1 < len(occurrences) else frame_index
            current_features.append(feature[frame_index])
            next_features.append(feature[next_index])
            task_ids.append(TASKS.index(episode.task))
            phi = np.zeros(len(EVENTS), dtype=np.float32)
            phi[event] = 1.0
            phis.append(phi)
            dones.append(index + 1 == len(occurrences))
    current = torch.from_numpy(np.stack(current_features)).to(device)
    following = torch.from_numpy(np.stack(next_features)).to(device)
    tasks = torch.tensor(task_ids, device=device)
    phi = torch.from_numpy(np.stack(phis)).to(device)
    done = torch.tensor(dones, dtype=torch.float32, device=device)
    model = Head(current.shape[1] + len(TASKS) + 1).to(device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    loss_value = math.nan
    for step in range(steps):
        selection = torch.randint(len(current), (min(512, len(current)),), generator=generator, device=device)
        lam = torch.from_numpy(np.random.choice(LAMBDAS, size=len(selection))).to(device)
        prediction = model(make_input(current[selection], tasks[selection], lam))
        with torch.no_grad():
            next_prediction = target(make_input(following[selection], tasks[selection], lam))
            desired = phi[selection] + lam[:, None] * (1.0 - done[selection, None]) * next_prediction
        loss = F.huber_loss(prediction, desired)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss)
        if (step + 1) % 200 == 0:
            target.load_state_dict(model.state_dict())
    return model.eval(), loss_value


def train_controls(
    episodes: list[Episode],
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> tuple[Head, Head, float, float]:
    all_features = []
    all_tasks = []
    remaining_targets = []
    progress_targets = []
    for episode in episodes:
        feature = features[(episode.task, episode.body, episode.index)]
        for frame_index in range(episode.steps):
            target = np.full(len(EVENTS), -1.0, dtype=np.float32)
            for event, event_time in episode.events.items():
                if event_time >= frame_index:
                    target[event] = -(event_time - frame_index) / STEP_SCALE
            remaining_targets.append(target)
            progress_targets.append(np.full(len(EVENTS), frame_index / max(episode.steps - 1, 1), dtype=np.float32))
        all_features.append(feature)
        all_tasks.append(np.full(episode.steps, TASKS.index(episode.task), dtype=np.int64))
    feature_tensor = torch.from_numpy(np.concatenate(all_features)).to(device)
    task_tensor = torch.from_numpy(np.concatenate(all_tasks)).to(device)
    remaining = torch.from_numpy(np.stack(remaining_targets)).to(device)
    progress = torch.from_numpy(np.stack(progress_targets)).to(device)
    models = []
    losses = []
    for seed, target in [(211, remaining), (307, progress)]:
        torch.manual_seed(seed)
        model = Head(feature_tensor.shape[1] + len(TASKS) + 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        loss_value = math.nan
        for _ in range(6000):
            selection = torch.randint(len(feature_tensor), (512,), generator=generator, device=device)
            condition = torch.zeros(len(selection), device=device)
            prediction = model(make_input(feature_tensor[selection], task_tensor[selection], condition))
            loss = F.huber_loss(prediction, target[selection])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss)
        models.append(model.eval())
        losses.append(loss_value)
    return models[0], models[1], losses[0], losses[1]


def predict(
    model: Head,
    episode: Episode,
    frame_indices: list[int],
    conditions: np.ndarray,
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> np.ndarray:
    feature = torch.from_numpy(features[(episode.task, episode.body, episode.index)][frame_indices]).to(device)
    tasks = torch.full((len(frame_indices),), TASKS.index(episode.task), device=device)
    condition = torch.from_numpy(conditions.astype(np.float32)).to(device)
    with torch.inference_mode():
        return model(make_input(feature, tasks, condition)).cpu().numpy()


def gates(current_event: int, rho: np.ndarray) -> np.ndarray:
    output = np.zeros(len(EVENTS), dtype=np.float32)
    output[current_event] = 1.0
    probability = 1.0
    for event in range(current_event + 1, len(EVENTS)):
        probability *= float(rho[event])
        output[event] = probability
    return output


def body_global_parameters(episodes: list[Episode], body: str, count: int = 20) -> tuple[float, dict[str, np.ndarray]]:
    selected = []
    rhos = {}
    for task in TASKS:
        group = [episode for episode in episodes if episode.task == task and episode.body == body]
        selected.extend(sorted(group, key=lambda episode: episode.index)[:count])
        _, rhos[task], _ = estimate_parameters(group, count)
    lam, _, _ = estimate_parameters(selected, len(selected))
    return lam, rhos


def m2_results(
    episodes: list[Episode],
    model: Head,
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[int, float]]:
    rows = []
    correlations = {}
    rng = np.random.default_rng(20260825)
    for count in [5, 10, 15, 20]:
        for body in TARGET_BODIES:
            global_lam, _ = body_global_parameters(episodes, body, count)
            for task in TASKS:
                group = sorted(
                    [episode for episode in episodes if episode.task == task and episode.body == body],
                    key=lambda episode: episode.index,
                )
                _, rho, _ = estimate_parameters(group, count)
                lam = global_lam
                selected = group[:count]
                posterior_a = np.ones(len(EVENTS))
                posterior_b = np.ones(len(EVENTS))
                for event_index in range(1, len(EVENTS)):
                    eligible = sum(event_index - 1 in episode.events for episode in selected)
                    reached = sum(
                        event_index in episode.events
                        and event_index - 1 in episode.events
                        and episode.events[event_index] >= episode.events[event_index - 1]
                        for episode in selected
                    )
                    posterior_a[event_index] += reached
                    posterior_b[event_index] += eligible - reached
                rho_samples = rng.beta(posterior_a, posterior_b, size=(20000, len(EVENTS)))
                shared = []
                for episode in selected:
                    value = predict(model, episode, [0], np.asarray([lam]), features, device)[0, -1]
                    shared.append(np.clip(value / max(lam ** 5, 1e-4), 0.0, 1.0))
                prediction_samples = float(np.mean(shared)) * np.prod(rho_samples[:, 1:], axis=1)
                successes = sum(episode.success for episode in selected)
                direct_a = successes + 1
                direct_b = count - successes + 1
                prediction = float(np.mean(prediction_samples))
                rows.append(
                    {
                        "analysis_stream": "stage1_target_exploratory",
                        "task": task,
                        "embodiment": body,
                        "N": count,
                        "lambda": lam,
                        "shared_head_feasibility": float(np.mean(shared)),
                        "rho_product": float(np.prod(rho[1:])),
                        "rho_product_posterior_mean": float(np.mean(np.prod(rho_samples[:, 1:], axis=1))),
                        "rho_posterior_a": json.dumps(posterior_a[1:].astype(int).tolist()),
                        "rho_posterior_b": json.dumps(posterior_b[1:].astype(int).tolist()),
                        "predicted_success_rate": prediction,
                        "predicted_success_ci_low": float(np.quantile(prediction_samples, 0.025)),
                        "predicted_success_ci_high": float(np.quantile(prediction_samples, 0.975)),
                        "prediction_approval_probability": float(np.mean(prediction_samples >= RHO_DEPLOY)),
                        "prediction_below_fail_probability": float(np.mean(prediction_samples < RHO_FAIL)),
                        "direct_posterior_a": direct_a,
                        "direct_posterior_b": direct_b,
                        "direct_posterior_mean": direct_a / (direct_a + direct_b),
                        "direct_success_ci_low": beta_distribution.ppf(0.025, direct_a, direct_b),
                        "direct_success_ci_high": beta_distribution.ppf(0.975, direct_a, direct_b),
                        "direct_approval_probability": beta_distribution.sf(RHO_DEPLOY, direct_a, direct_b),
                        "direct_below_fail_probability": beta_distribution.cdf(RHO_FAIL, direct_a, direct_b),
                        "robogate_boundary_probability": beta_distribution.cdf(RHO_DEPLOY, direct_a, direct_b) - beta_distribution.cdf(RHO_FAIL, direct_a, direct_b),
                        "robogate_priority": int(RHO_FAIL <= direct_a / (direct_a + direct_b) <= RHO_DEPLOY),
                        "true_success_rate": float(np.mean([episode.success for episode in group])),
                        "absolute_error": abs(prediction - float(np.mean([episode.success for episode in group]))),
                        "e_process_fail": math.nan,
                        "e_process_deploy": math.nan,
                        "anytime_p_value": math.nan,
                        "sequential_decision": "exploratory_only",
                        "decision_rollouts": math.nan,
                    }
                )
        subset = rows[-len(TARGET_BODIES) * len(TASKS) :]
        predicted = np.asarray([row["predicted_success_rate"] for row in subset])
        truth = np.asarray([row["true_success_rate"] for row in subset])
        correlations[count] = float(np.corrcoef(predicted, truth)[0, 1])
    fresh_paths = sorted((ROOT / "m2_fresh/adjust_bottle/piper").glob("episode_*.hdf5"))
    e_fail = 1.0
    e_deploy = 1.0
    max_evidence = 1.0
    decision = "continue"
    decision_rollouts = math.nan
    successes = 0
    for index, path in enumerate(fresh_paths, 1):
        with h5py.File(path, "r") as handle:
            outcome = int(handle.attrs["success"])
        successes += outcome
        e_fail *= 1.0 + RHO_FAIL - outcome
        e_deploy *= 1.0 + outcome - RHO_DEPLOY
        max_evidence = max(max_evidence, e_fail, e_deploy)
        if decision == "continue" and e_fail >= 1.0 / SEQUENTIAL_ALPHA:
            decision = "reject_rho_at_least_0.30"
            decision_rollouts = index
        elif decision == "continue" and e_deploy >= 1.0 / SEQUENTIAL_ALPHA:
            decision = "approve_rho_above_0.70"
            decision_rollouts = index
        direct_a = successes + 1
        direct_b = index - successes + 1
        rows.append(
            {
                "analysis_stream": "formal_fresh_sequential",
                "task": "adjust_bottle",
                "embodiment": "piper",
                "N": index,
                "direct_posterior_a": direct_a,
                "direct_posterior_b": direct_b,
                "direct_posterior_mean": direct_a / (direct_a + direct_b),
                "direct_success_ci_low": beta_distribution.ppf(0.025, direct_a, direct_b),
                "direct_success_ci_high": beta_distribution.ppf(0.975, direct_a, direct_b),
                "direct_approval_probability": beta_distribution.sf(RHO_DEPLOY, direct_a, direct_b),
                "direct_below_fail_probability": beta_distribution.cdf(RHO_FAIL, direct_a, direct_b),
                "robogate_boundary_probability": beta_distribution.cdf(RHO_DEPLOY, direct_a, direct_b) - beta_distribution.cdf(RHO_FAIL, direct_a, direct_b),
                "robogate_priority": int(RHO_FAIL <= direct_a / (direct_a + direct_b) <= RHO_DEPLOY),
                "true_success_rate": successes / index,
                "e_process_fail": e_fail,
                "e_process_deploy": e_deploy,
                "anytime_p_value": min(1.0, 1.0 / max_evidence),
                "sequential_decision": decision,
                "decision_rollouts": decision_rollouts,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/M2_feasibility_prediction.csv", index=False)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    figure, axis = plt.subplots(figsize=(7, 6))
    markers = {5: "o", 10: "s", 15: "^", 20: "D"}
    for count in markers:
        subset = frame[(frame.analysis_stream == "stage1_target_exploratory") & (frame.N == count)]
        axis.scatter(subset.true_success_rate, subset.predicted_success_rate, label=f"N={count}, r={correlations[count]:.3f}", marker=markers[count], alpha=0.75)
    key = frame[(frame.analysis_stream == "stage1_target_exploratory") & (frame.N == 20) & (frame.task == "adjust_bottle") & (frame.embodiment == "piper")].iloc[0]
    axis.scatter([key.true_success_rate], [key.predicted_success_rate], color="red", s=140, facecolors="none", linewidths=2)
    axis.annotate("Piper / adjust_bottle", (key.true_success_rate, key.predicted_success_rate), xytext=(8, 8), textcoords="offset points")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="真实成功率", ylabel="冻结头 + ρ 的预测成功率", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axis.legend()
    figure.tight_layout()
    figure.savefig(ROOT / "results/M2_feasibility_prediction.png", dpi=180)
    plt.close(figure)
    return frame, correlations


def control_occupancy(output: np.ndarray) -> np.ndarray:
    delta = np.clip(-output * STEP_SCALE, 0, STEP_SCALE)
    occupancy = np.power(GAMMA, delta)
    occupancy[output < -0.9] = 0.0
    return occupancy


def pca_metrics(residual: np.ndarray) -> tuple[float, int, float, float, float]:
    centered = residual - residual.mean(0, keepdims=True)
    eigenvalues = np.square(np.linalg.svd(centered, full_matrices=False, compute_uv=False)) / max(len(centered) - 1, 1)
    if eigenvalues.sum() <= 1e-12:
        return 0.0, 0, 0.0, 0.0, 0.0
    ratios = eigenvalues / eigenvalues.sum()
    participation = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    effective95 = int(np.searchsorted(np.cumsum(ratios), 0.95) + 1)
    return participation, effective95, float(ratios[0]), float(ratios[:2].sum()), float(ratios[:3].sum())


def m3_results(
    episodes: list[Episode],
    etsf: Head,
    control: Head,
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    body_parameters = {body: body_global_parameters(episodes, body) for body in TARGET_BODIES}
    rows = []
    rng = np.random.default_rng(20260825)
    for body in TARGET_BODIES:
        other = TARGET_BODIES[1 - TARGET_BODIES.index(body)]
        lam, task_rhos = body_parameters[body]
        mismatch_lam, mismatch_rhos = body_parameters[other]
        for task in TASKS:
            residuals = {name: [] for name in ["ETSF_matched", "control_time_baseline", "mismatched_embodiment", "random_lambda"]}
            group = [episode for episode in episodes if episode.task == task and episode.body == body]
            for episode in group:
                occurrences = list(episode.events.items())
                frames = [frame for _, frame in occurrences]
                current_events = [event for event, _ in occurrences]
                random_lambdas = rng.choice(LAMBDAS, size=len(frames)).astype(np.float32)
                matched = predict(etsf, episode, frames, np.full(len(frames), lam), features, device)
                mismatched = predict(etsf, episode, frames, np.full(len(frames), mismatch_lam), features, device)
                randomized = predict(etsf, episode, frames, random_lambdas, features, device)
                baseline = control_occupancy(predict(control, episode, frames, np.zeros(len(frames)), features, device))
                for row_index, (event, frame_index) in enumerate(occurrences):
                    monte_carlo = np.zeros(len(EVENTS), dtype=np.float32)
                    for future_event, future_time in episode.events.items():
                        if future_time >= frame_index:
                            monte_carlo[future_event] = GAMMA ** (future_time - frame_index)
                    residuals["ETSF_matched"].append(monte_carlo - matched[row_index] * gates(event, task_rhos[task]))
                    residuals["control_time_baseline"].append(monte_carlo - baseline[row_index])
                    residuals["mismatched_embodiment"].append(monte_carlo - mismatched[row_index] * gates(event, mismatch_rhos[task]))
                    residuals["random_lambda"].append(monte_carlo - randomized[row_index] * gates(event, task_rhos[task]))
            for condition, values in residuals.items():
                residual = np.stack(values)
                participation, effective95, variance1, variance2, variance3 = pca_metrics(residual)
                rows.append(
                    {
                        "task": task,
                        "embodiment": body,
                        "condition": condition,
                        "n_samples": len(residual),
                        "participation_ratio": participation,
                        "effective_dimension_95pct": effective95,
                        "explained_variance_pc1": variance1,
                        "explained_variance_pc1_2": variance2,
                        "explained_variance_pc1_3": variance3,
                        "residual_mse": float(np.square(residual).mean()),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/M3_residual_dimension.csv", index=False)
    return frame


def rank_auc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return math.nan
    return float(np.mean([1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positive for neg in negative]))


def event_bellman(
    model: Head,
    group: list[Episode],
    lam: float,
    task_rhos: dict[str, np.ndarray] | None,
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> tuple[float, list[float]]:
    errors = []
    scores = []
    for episode in group:
        occurrences = list(episode.events.items())
        frames = [frame for _, frame in occurrences]
        output = predict(model, episode, frames, np.full(len(frames), lam), features, device)
        rho = task_rhos[episode.task] if task_rhos is not None else np.ones(len(EVENTS), dtype=np.float32)
        adjusted = np.stack([value * gates(event, rho) for value, (event, _) in zip(output, occurrences)])
        for index, (event, _) in enumerate(occurrences):
            phi = np.zeros(len(EVENTS), dtype=np.float32)
            phi[event] = 1.0
            if index + 1 < len(occurrences):
                target = phi + lam * gates(event, rho)[occurrences[index + 1][0]] * adjusted[index + 1]
            else:
                target = phi
            errors.append(float(np.square(adjusted[index] - target).mean()))
        scores.append(float(adjusted[0, -1]))
    return float(np.mean(errors)), scores


def control_bellman(
    model: Head,
    group: list[Episode],
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> tuple[float, list[float]]:
    errors = []
    scores = []
    for episode in group:
        frames = list(range(episode.steps))
        occupancy = control_occupancy(predict(model, episode, frames, np.zeros(len(frames)), features, device))
        by_time: dict[int, list[int]] = {}
        for event, time_index in episode.events.items():
            by_time.setdefault(time_index, []).append(event)
        for time_index in range(episode.steps):
            phi = np.zeros(len(EVENTS), dtype=np.float32)
            for event in by_time.get(time_index, []):
                phi[event] = 1.0
            target = phi if time_index + 1 == episode.steps else phi + GAMMA * occupancy[time_index + 1]
            errors.append(float(np.square(occupancy[time_index] - target).mean()))
        scores.append(float(occupancy[0, -1]))
    return float(np.mean(errors)), scores


def m4_results(
    episodes: list[Episode],
    etsf: Head,
    control: Head,
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Head]]:
    rows = []
    scratch_models = {}
    for body_index, body in enumerate(TARGET_BODIES):
        train = []
        evaluation = []
        for task in TASKS:
            group = sorted([episode for episode in episodes if episode.task == task and episode.body == body], key=lambda episode: episode.index)
            train.extend(group[:15])
            evaluation.extend(group[15:20])
        scratch, scratch_loss = train_etsf(train, features, device, 5000, 601 + body_index)
        scratch_models[body] = scratch
        lam, task_rhos = body_global_parameters(episodes, body)
        for task in TASKS + ["__all__"]:
            group = evaluation if task == "__all__" else [episode for episode in evaluation if episode.task == task]
            labels = [int(episode.success) for episode in group]
            frozen_mse, frozen_scores = event_bellman(etsf, group, lam, task_rhos, features, device)
            control_mse, control_scores = control_bellman(control, group, features, device)
            scratch_mse, scratch_scores = event_bellman(scratch, group, lam, None, features, device)
            for name, mse, scores, unit in [
                ("ETSF_frozen_estimated_params", frozen_mse, frozen_scores, "event_boundary"),
                ("control_time_frozen", control_mse, control_scores, "control_step"),
                ("target_from_scratch", scratch_mse, scratch_scores, "event_boundary"),
            ]:
                rows.append(
                    {
                        "task": task,
                        "embodiment": body,
                        "model": name,
                        "n_eval_rollouts": len(group),
                        "transition_unit": unit,
                        "bellman_mse": mse,
                        "value_ranking_auc": rank_auc(labels, scores),
                        "target_scratch_train_loss": scratch_loss if name == "target_from_scratch" else math.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/M4_bellman_comparison.csv", index=False)
    return frame, scratch_models


def m5_results(
    episodes: list[Episode],
    source_models: dict[str, Head],
    features: dict[tuple[str, str, int], np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for source, target in [("ARX-X5", "piper"), ("aloha-agilex", "ARX-X5"), ("aloha-agilex", "ur5-wsg")]:
        source_lam, source_rhos = body_global_parameters(episodes, source)
        target_lam, target_rhos = body_global_parameters(episodes, target)
        model = source_models[source]
        for ablation in ["lambda_only", "rho_only", "lambda_and_rho"]:
            lam = target_lam if ablation in {"lambda_only", "lambda_and_rho"} else source_lam
            rhos = target_rhos if ablation in {"rho_only", "lambda_and_rho"} else source_rhos
            predictions = []
            truths = []
            errors = []
            labels = []
            scores = []
            for task in TASKS:
                group = sorted([episode for episode in episodes if episode.task == task and episode.body == target], key=lambda episode: episode.index)
                shared = []
                for episode in group:
                    value = predict(model, episode, [0], np.asarray([lam]), features, device)[0, -1]
                    shared.append(np.clip(value / max(lam ** 5, 1e-4), 0.0, 1.0))
                predictions.append(float(np.mean(shared) * np.prod(rhos[task][1:])))
                truths.append(float(np.mean([episode.success for episode in group])))
                mse, task_scores = event_bellman(model, group, lam, rhos, features, device)
                errors.append(mse)
                labels.extend(int(episode.success) for episode in group)
                scores.extend(task_scores)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "mechanism": "pure_rho" if target == "piper" else "pure_lambda" if target == "ARX-X5" else "combined",
                    "ablation": ablation,
                    "lambda_used": lam,
                    "mean_success_prediction_mae": float(np.mean(np.abs(np.asarray(predictions) - truths))),
                    "mean_event_bellman_mse": float(np.mean(errors)),
                    "value_ranking_auc": rank_auc(labels, scores),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/M5_dissociation_table.csv", index=False)
    return frame


def write_raw_metrics(
    episodes: list[Episode],
    calibration: dict[str, dict[str, object]],
    losses: dict[str, float],
    correlations: dict[int, float],
) -> pd.DataFrame:
    rows = []
    for task, config in calibration.items():
        for metric in ["tau_v", "delta_z", "tau_d", "k"]:
            rows.append({"category": "calibration", "task": task, "embodiment": "all", "episode": -1, "metric": metric, "value": config[metric], "detail": str(config["moving"])})
        rows.append({"category": "calibration", "task": task, "embodiment": "all", "episode": -1, "metric": "target_mode", "value": math.nan, "detail": str(config["anchor"]) or f"centroids={len(np.asarray(config['centers']))}"})
        target_values = np.asarray(config["offset"])[None, :] if config["anchor"] else np.asarray(config["centers"])
        for target_index, target in enumerate(target_values):
            for axis, value in zip("xyz", target):
                rows.append({"category": "calibration", "task": task, "embodiment": "all", "episode": -1, "metric": f"target_{target_index}_{axis}", "value": value, "detail": "relative_offset" if config["anchor"] else "absolute_centroid"})
    for episode in episodes:
        rows.append({"category": "episode", "task": episode.task, "embodiment": episode.body, "episode": episode.index, "metric": "success", "value": int(episode.success), "detail": ""})
        rows.append({"category": "episode", "task": episode.task, "embodiment": episode.body, "episode": episode.index, "metric": "total_steps", "value": episode.steps, "detail": ""})
        for event, frame_index in episode.events.items():
            rows.append({"category": "event", "task": episode.task, "embodiment": episode.body, "episode": episode.index, "metric": EVENTS[event], "value": frame_index, "detail": ""})
    for name, value in losses.items():
        rows.append({"category": "training", "task": "all", "embodiment": "source", "episode": -1, "metric": name, "value": value, "detail": ""})
    for count, value in correlations.items():
        rows.append({"category": "M2", "task": "all", "embodiment": "target", "episode": -1, "metric": f"pearson_r_N{count}", "value": value, "detail": ""})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/raw_metrics.csv", index=False)
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def write_report(
    episodes: list[Episode],
    m1: pd.DataFrame,
    m2: pd.DataFrame,
    correlations: dict[int, float],
    m3: pd.DataFrame,
    m4: pd.DataFrame,
    m5: pd.DataFrame,
    losses: dict[str, float],
) -> None:
    target_summary = (
        pd.DataFrame([{"task": episode.task, "embodiment": episode.body, "success": int(episode.success), "steps": episode.steps} for episode in episodes if episode.body in TARGET_BODIES])
        .groupby(["task", "embodiment"])
        .agg(rollouts=("success", "size"), success_rate=("success", "mean"), mean_steps=("steps", "mean"))
        .reset_index()
    )
    consistency = m1.groupby("task").cross_body_mode_consistent.max().mean()
    prefix_consistency = m1.groupby("task").cross_body_mode_prefix_compatible.max().mean()
    key = m2[(m2.analysis_stream == "stage1_target_exploratory") & (m2.N == 20) & (m2.task == "adjust_bottle") & (m2.embodiment == "piper")].iloc[0]
    fresh = m2[m2.analysis_stream == "formal_fresh_sequential"].iloc[-1]
    m3_summary = m3.groupby("condition").agg(participation_ratio=("participation_ratio", "mean"), residual_mse=("residual_mse", "mean")).reset_index()
    m4_summary = m4[m4.task == "__all__"][["embodiment", "model", "bellman_mse", "value_ranking_auc"]]
    report = f"""# ETSF 阶段 1 报告

## 先报与预期不一致的结果

- 六个任务中，完整事件序列众数在四本体间严格一致的比例为 **{consistency:.3f}**；把失败造成的前缀截断交给 rho 表达后，前缀兼容比例为 **{prefix_consistency:.3f}**。真正的偏序冲突是 `beat_block_hammer` 的 e1/e2 反转。
- N=20 时，12 个目标「任务×本体」格子的可行性预测 Pearson r 为 **{correlations[20]:.4f}**。
- Piper / adjust_bottle 的真实成功率为 **{key.true_success_rate:.4f}**，冻结头加 rho 后验的预测均值为 **{key.predicted_success_rate:.4f}**，95% 区间为 **[{key.predicted_success_ci_low:.4f}, {key.predicted_success_ci_high:.4f}]**，`P(成功率 < 0.30)` 为 **{key.prediction_below_fail_probability:.6f}**。
- 独立 fresh 流在第 **{fresh.decision_rollouts:.0f}** 条有效 rollout 停止；后验 `P(rho < 0.30)` 为 **{fresh.direct_below_fail_probability:.6f}**，anytime p 为 **{fresh.anytime_p_value:.6f}**。

## 数据与不可妥协约束

- 源本体 Aloha、ARX-X5：每任务各 50 条官方成功轨迹；目标 Piper、UR5-WSG：每任务各 20 条未筛选 rollout。
- 正式序贯证据另用 seed 1000 起的 fresh 流；三个未产生观测的 `UnStableError` seed 不计样本。当前 MPlib/FCL 初始化会原生崩溃，因此 fresh 流仅跳过 adjust_bottle 专家路径未使用的 TOPP 初始化；旧 seed 0 对照仍为 61 步失败。
- 事件只使用物体位姿与仿真器成功标志。每任务 `tau_v` 为合并速度的 85% 分位数（下限 0.0005），`delta_z` 为成功轨迹中位最大抬升的 20%，`tau_d` 为成功终点距离 95% 分位数加 0.005；均在四本体合并数据上只标定一次。
- 冻结骨干为 `{BACKBONE}`，三视角特征写入磁盘缓存。共享 ETSF 头只用源本体训练。

{markdown_table(target_summary)}

## M1：事件图共享性

`M1_event_consistency.csv` 给出每个「任务×本体×事件转移」的众数序列、到达率与 D_j 分布。任务级完整众数一致率为 **{consistency:.3f}**，前缀兼容率为 **{prefix_consistency:.3f}**；两者分开报告，避免把 rho 的可达性差异误判成共享偏序通过。

## M2：零样本可行性预测

冻结共享头与各事件 rho 的 Beta(1,1) 后验组合，报告预测分布而非点平滑；收敛相关系数为：N=5 **{correlations[5]:.4f}**，N=10 **{correlations[10]:.4f}**，N=15 **{correlations[15]:.4f}**，N=20 **{correlations[20]:.4f}**。

部署口径固定为失败阈值 0.30、批准阈值 0.70。`direct_approval_probability` 是后验 `P(rho >= 0.70)`；额外 rollout 的 `robogate_priority` 只用于把预算排到后验均值仍在 0.30–0.70 的格子，不作为停止证据。正式 fresh 流使用双向 e-process，阈值为 1/0.05=20；旧 0/10 与固定 20 条只作探索性结果。

## M3：残差有效维度

{markdown_table(m3_summary)}

参与比与前三主成分解释率逐任务、逐本体列在 `M3_residual_dimension.csv`；错配本体和随机 lambda 均未丢弃。

## M4：目标域 Bellman 残差

{markdown_table(m4_summary)}

ETSF 与从零训练按事件边界计算；RECAP 式控制时间基线按控制步计算，`transition_unit` 列明确记录该差异。AUC 只在同时含成败样本时定义。

## M5：机制解离

{markdown_table(m5)}

## 训练末步损失

{markdown_table(pd.DataFrame([{'model': name, 'loss': value} for name, value in losses.items()]))}

## 口径

- ETSF：6 维事件后继头，在事件边界做 TD；lambda 从 `{LAMBDAS.tolist()}` 采样。
- 控制时间基线：同一 256×256 MLP、6 维逐事件剩余控制步回归，其中 eK 分量就是 RECAP 式剩余步数。
- 归一化进度：同容量 6 维头，六维重复同一归一化进度目标，仅作为下界。
- lambda_b 是跨任务全局标量；rho_b 按任务和事件转移估计，全部报告 Beta(1,1) 后验、95% 区间与批准/失败侧概率。
"""
    (ROOT / "stage1_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after-m2", action="store_true")
    args = parser.parse_args()
    random.seed(20260825)
    np.random.seed(20260825)
    torch.manual_seed(20260825)
    episodes = load_episodes()
    counts = Counter((episode.task, episode.body) for episode in episodes)
    expected = {(task, body): 50 if body in SOURCE_BODIES else 20 for task in TASKS for body in BODIES}
    if counts != Counter(expected):
        raise RuntimeError(f"incomplete data cells: actual={counts}, expected={expected}")
    calibration = event_calibration(episodes)
    assign_events(episodes, calibration)
    m1 = write_m1(episodes)
    write_parameters(episodes)
    extract_features(episodes)
    features, _, _ = load_feature_cache(episodes)
    device = torch.device("cuda:0")
    source = [episode for episode in episodes if episode.body in SOURCE_BODIES]
    etsf, etsf_loss = train_etsf(source, features, device, 7000, 101)
    m2, correlations = m2_results(episodes, etsf, features, device)
    if args.stop_after_m2:
        print("M1_M2_COMPLETE=" + json.dumps({"episodes": len(episodes), "pearson_r_N20": correlations[20]}), flush=True)
        return
    control, progress, control_loss, progress_loss = train_controls(source, features, device)
    del progress
    source_models = {}
    source_losses = {}
    for offset, body in enumerate(["aloha-agilex", "ARX-X5"]):
        model, loss = train_etsf([episode for episode in source if episode.body == body], features, device, 5000, 401 + offset)
        source_models[body] = model
        source_losses[f"etsf_{body}"] = loss
    m3 = m3_results(episodes, etsf, control, features, device)
    m4, _ = m4_results(episodes, etsf, control, features, device)
    m5 = m5_results(episodes, source_models, features, device)
    losses = {"etsf_shared": etsf_loss, "remaining_steps": control_loss, "normalized_progress": progress_loss, **source_losses}
    write_raw_metrics(episodes, calibration, losses, correlations)
    write_report(episodes, m1, m2, correlations, m3, m4, m5, losses)
    print("STAGE1_COMPLETE=" + json.dumps({"episodes": len(episodes), "results": 8, "losses": losses}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
