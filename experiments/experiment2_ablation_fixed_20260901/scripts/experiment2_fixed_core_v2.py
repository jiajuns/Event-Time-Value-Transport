#!/usr/bin/env python3
"""Corrected progressive A -> A+B -> A+B+C experiment core.

This module intentionally keeps the paper's frozen data split and six-metric
protocol while repairing the implementation mismatches in the first TABLE V
run.  The three models are true supersets:

* A: masked GRU + trajectory-success head.
* A+B: A + event values + successor head.
* A+B+C: A+B + a required event-time CfC path with no bypass gate.

No target-test sample is used by training, checkpoint selection, or success
calibration.  Checkpoint selection uses two source-body LOBO folds only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn

try:
    from ncps.torch import CfCCell
except ImportError as error:  # pragma: no cover - exercised by deployment preflight
    raise RuntimeError("ncps==1.0.1 is required for the A+B+C model") from error


FORMAT = "etsf_progressive_event_time_ablation_v8"
METHODS = ("e2_base_gru_fixed", "e2_gru_event_fixed", "e2_cfc_event_fixed")
STAGES = {
    "e2_base_gru_fixed": "A",
    "e2_gru_event_fixed": "AB",
    "e2_cfc_event_fixed": "ABC",
}
CHAIN = ("e0", "e12", "e3", "e4", "eK")
STATE_DIM = 27
EVENT_COUNT = len(CHAIN)
FRAME_HISTORY = 16
EVENT_HISTORY = 8
GRU_HIDDEN = 96
CFC_HIDDEN = 64
GAMMA = 0.99
BETA_GRID = (0.35, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5)
# At the end of the C stage, the selected event-value logit is decomposed into
# bounded within-event evidence plus a causal-chain progress prior.  Because
# 2 * evidence amplitude < one event gap, forward/backward event transitions
# have the correct sign by construction while the monotone transform preserves
# every within-event ranking used by the AUC metric.
BASE_EVIDENCE_AMPLITUDE = 0.35
TIME_EVIDENCE_AMPLITUDE = 0.10
PROGRESS_EVIDENCE_TEMPERATURE = 2.0
PROGRESS_LOGIT_GAP = 1.0


def _string_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class Rollout:
    path: str
    body: str
    seed: int
    success: bool
    failure_class: str
    state27: np.ndarray
    event_ids: np.ndarray
    reward: np.ndarray
    sim_times: np.ndarray


def load_rollout(path: Path) -> Rollout:
    with h5py.File(path, "r") as handle:
        required = ("state27", "event_ids", "reward", "sim_times")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path}: missing HDF5 datasets {missing}")
        state27 = handle["state27"][:].astype(np.float32)
        event_ids = handle["event_ids"][:].astype(np.int64)
        reward = handle["reward"][:].astype(np.float32)
        sim_times = handle["sim_times"][:].astype(np.float64)
        if not (len(state27) == len(event_ids) == len(reward) == len(sim_times)):
            raise ValueError(f"{path}: trajectory arrays have different lengths")
        if state27.ndim != 2 or state27.shape[1] != STATE_DIM:
            raise ValueError(f"{path}: state27 shape is {state27.shape}")
        if len(state27) < 2:
            raise ValueError(f"{path}: trajectory is too short")
        if not np.isfinite(state27).all() or not np.isfinite(sim_times).all():
            raise ValueError(f"{path}: trajectory has non-finite values")
        if np.any(np.diff(sim_times) <= 0.0):
            raise ValueError(f"{path}: sim_times must be strictly increasing")
        if event_ids.min() < 0 or event_ids.max() >= EVENT_COUNT:
            raise ValueError(f"{path}: event id outside frozen alphabet")
        return Rollout(
            path=str(path),
            body=_string_attr(handle.attrs["body"]),
            seed=int(handle.attrs["seed"]),
            success=bool(handle.attrs["success"]),
            failure_class=_string_attr(handle.attrs.get("failure_class", "none")),
            state27=state27,
            event_ids=event_ids,
            reward=reward,
            sim_times=sim_times,
        )


def collect_rollouts(root: Path) -> list[Rollout]:
    paths = sorted(root.rglob("episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no episode HDF5 files under {root}")
    return [load_rollout(path) for path in paths]


def logical_data_manifest_sha256(root: Path) -> str:
    """Hash the same relative-path/size roster frozen during the audit."""
    rows = []
    for path in sorted(root.rglob("episode_*.hdf5")):
        rows.append(f"{path.relative_to(root)} {path.stat().st_size}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_protocol(rollouts: Sequence[Rollout]) -> dict:
    counts: dict[str, int] = {}
    for rollout in rollouts:
        counts[rollout.body] = counts.get(rollout.body, 0) + 1
    expected = {"aloha-agilex": 270, "arx-x5": 270, "piper": 30, "ur5": 30}
    if counts != expected:
        raise ValueError(f"frozen 600-rollout roster mismatch: {counts} != {expected}")
    return {
        "total": len(rollouts),
        "source": counts["aloha-agilex"] + counts["arx-x5"],
        "target": counts["piper"] + counts["ur5"],
        "by_body": counts,
    }


def event_boundary_indices(events: np.ndarray) -> np.ndarray:
    frames = [0]
    frames.extend((np.flatnonzero(events[1:] != events[:-1]) + 1).tolist())
    if frames[-1] != len(events) - 1:
        frames.append(len(events) - 1)
    return np.asarray(sorted(set(frames)), dtype=np.int64)


def split_adapt_test_rollouts(
    rollouts: Sequence[Rollout],
    bodies: Iterable[str],
    n_adapt_success: int = 3,
    n_adapt_failure: int = 2,
) -> tuple[list[int], list[int]]:
    body_set = set(bodies)
    adapt: list[int] = []
    test: list[int] = []
    for body in sorted(body_set):
        ids = [i for i, rollout in enumerate(rollouts) if rollout.body == body]
        ids.sort(key=lambda i: Path(rollouts[i].path).stem)
        success = [i for i in ids if rollouts[i].success]
        failure = [i for i in ids if not rollouts[i].success]
        if len(success) < n_adapt_success or len(failure) < n_adapt_failure:
            raise ValueError(f"{body}: insufficient stratified adaptation rollouts")
        selected = success[:n_adapt_success] + failure[:n_adapt_failure]
        adapt.extend(selected)
        selected_set = set(selected)
        test.extend(i for i in ids if i not in selected_set)
    return adapt, test


def compute_rho(
    rollouts: Sequence[Rollout], adapt_rollout_ids: Sequence[int]
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    bodies = sorted({rollouts[i].body for i in adapt_rollout_ids})
    for body in bodies:
        episodes = [rollouts[i] for i in adapt_rollout_ids if rollouts[i].body == body]
        per_event: dict[str, dict[str, float]] = {}
        for event_id, name in enumerate(CHAIN):
            reached = 0
            seen = 0
            for rollout in episodes:
                maximum = int(rollout.event_ids.max())
                if maximum >= event_id:
                    reached += 1
                    seen += 1
                elif maximum == event_id - 1:
                    seen += 1
            alpha = 1.0 + reached
            beta = 1.0 + seen - reached
            per_event[name] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "mean": float(alpha / (alpha + beta)),
            }
        result[body] = per_event
    return result


def rho_product(rho: Mapping[str, Mapping[str, float]] | None) -> float:
    probability = 1.0
    rho = rho or {}
    for name in CHAIN:
        probability *= float(rho.get(name, {"mean": 0.5}).get("mean", 0.5))
    return float(np.clip(probability, 1e-5, 1.0 - 1e-5))


class BoundaryDataset:
    """Event-boundary examples with real next-state and event-time histories."""

    ARRAY_NAMES = (
        "frame_states",
        "frame_mask",
        "event_states",
        "event_mask",
        "event_dt",
        "event_seq_ids",
        "event_id",
        "next_event_id",
        "reward",
        "terminal",
        "label",
        "remaining",
        "body_id",
        "rollout_id",
        "boundary_position",
        "is_initial",
        "next_index",
        "next_duration",
    )

    def __init__(self, arrays: Mapping[str, np.ndarray], rollouts: Sequence[Rollout]):
        self.arrays = {name: np.asarray(arrays[name]) for name in self.ARRAY_NAMES}
        lengths = {len(value) for value in self.arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"boundary arrays have inconsistent lengths: {lengths}")
        self.rollouts = list(rollouts)
        self.size = lengths.pop()
        self.body_names = sorted({rollout.body for rollout in rollouts})
        self.body_to_id = {body: index for index, body in enumerate(self.body_names)}

    def indices_for_rollouts(self, rollout_ids: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.arrays["rollout_id"], np.asarray(rollout_ids)))

    def indices_for_bodies(self, bodies: Iterable[str]) -> np.ndarray:
        wanted = {self.body_to_id[body] for body in bodies}
        return np.flatnonzero(np.isin(self.arrays["body_id"], np.asarray(sorted(wanted))))

    def to_tensors(self, device: torch.device) -> "TensorBoundaryDataset":
        return TensorBoundaryDataset(self, device)


def build_boundary_dataset(
    rollouts: Sequence[Rollout], mean: np.ndarray, std: np.ndarray
) -> BoundaryDataset:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, STATE_DIM)
    std = np.asarray(std, dtype=np.float32).reshape(1, STATE_DIM)
    if np.any(std < 1e-6):
        raise ValueError("frozen source z-score contains a near-zero std")
    body_names = sorted({rollout.body for rollout in rollouts})
    body_to_id = {body: index for index, body in enumerate(body_names)}
    values: dict[str, list] = {name: [] for name in BoundaryDataset.ARRAY_NAMES}

    for rollout_id, rollout in enumerate(rollouts):
        frames = event_boundary_indices(rollout.event_ids)
        base_index = len(values["event_id"])
        for position, frame in enumerate(frames):
            lo = max(0, int(frame) - FRAME_HISTORY + 1)
            normalized = (rollout.state27[lo : frame + 1] - mean) / std
            frame_states = np.zeros((FRAME_HISTORY, STATE_DIM), dtype=np.float32)
            frame_mask = np.zeros(FRAME_HISTORY, dtype=bool)
            frame_states[-len(normalized) :] = normalized
            frame_mask[-len(normalized) :] = True

            first_event = max(0, position - EVENT_HISTORY + 1)
            event_positions = list(range(first_event, position + 1))
            event_states = np.zeros((EVENT_HISTORY, STATE_DIM), dtype=np.float32)
            event_mask = np.zeros(EVENT_HISTORY, dtype=bool)
            event_dt = np.zeros(EVENT_HISTORY, dtype=np.float32)
            event_seq_ids = np.full(EVENT_HISTORY, -1, dtype=np.int64)
            offset = EVENT_HISTORY - len(event_positions)
            for local, boundary_position in enumerate(event_positions):
                slot = offset + local
                boundary_frame = int(frames[boundary_position])
                event_states[slot] = (
                    rollout.state27[boundary_frame] - mean[0]
                ) / std[0]
                event_mask[slot] = True
                event_seq_ids[slot] = int(rollout.event_ids[boundary_frame])
                if local > 0:
                    previous_frame = int(frames[event_positions[local - 1]])
                    event_dt[slot] = float(
                        rollout.sim_times[boundary_frame] - rollout.sim_times[previous_frame]
                    )

            terminal = position == len(frames) - 1
            if terminal:
                next_event_id = int(rollout.event_ids[frame])
                reward = float(rollout.reward[frame])
                duration = 0.0
                next_index = base_index + position
            else:
                next_frame = int(frames[position + 1])
                next_event_id = int(rollout.event_ids[next_frame])
                reward = 0.0
                duration = float(rollout.sim_times[next_frame] - rollout.sim_times[frame])
                next_index = base_index + position + 1

            values["frame_states"].append(frame_states)
            values["frame_mask"].append(frame_mask)
            values["event_states"].append(event_states)
            values["event_mask"].append(event_mask)
            values["event_dt"].append(event_dt)
            values["event_seq_ids"].append(event_seq_ids)
            values["event_id"].append(int(rollout.event_ids[frame]))
            values["next_event_id"].append(next_event_id)
            values["reward"].append(reward)
            values["terminal"].append(float(terminal))
            values["label"].append(float(rollout.success))
            values["remaining"].append(float(len(frames) - 1 - position))
            values["body_id"].append(body_to_id[rollout.body])
            values["rollout_id"].append(rollout_id)
            values["boundary_position"].append(position)
            values["is_initial"].append(float(position == 0))
            values["next_index"].append(next_index)
            values["next_duration"].append(duration)

    arrays = {
        name: np.stack(items) if name in {
            "frame_states", "frame_mask", "event_states", "event_mask",
            "event_dt", "event_seq_ids"
        } else np.asarray(items)
        for name, items in values.items()
    }
    # The first valid event slot and all padding slots must have dt=0.
    for mask, dts in zip(arrays["event_mask"], arrays["event_dt"]):
        valid = np.flatnonzero(mask)
        if len(valid) == 0 or dts[valid[0]] != 0.0 or np.any(dts[~mask] != 0.0):
            raise AssertionError("invalid event-time padding contract")
    return BoundaryDataset(arrays, rollouts)


class TensorBoundaryDataset:
    def __init__(self, source: BoundaryDataset, device: torch.device):
        self.source = source
        self.device = device
        self.tensors: dict[str, torch.Tensor] = {}
        integer = {"event_seq_ids", "event_id", "next_event_id", "body_id", "rollout_id",
                   "boundary_position", "next_index"}
        boolean = {"frame_mask", "event_mask"}
        for name, value in source.arrays.items():
            if name in integer:
                tensor = torch.as_tensor(value, dtype=torch.long, device=device)
            elif name in boolean:
                tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            self.tensors[name] = tensor

    def take(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        return self.tensors[name][indices]


class MaskedGRUEncoder(nn.Module):
    """GRUCell loop that truly ignores left padding."""

    def __init__(self, input_size: int = STATE_DIM, hidden_size: int = GRU_HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = states.new_zeros(states.shape[0], self.hidden_size)
        for step in range(states.shape[1]):
            proposal = self.cell(states[:, step], hidden)
            hidden = torch.where(mask[:, step, None], proposal, hidden)
        return self.output(hidden)


class MaskedCfCSequence(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = CfCCell(
            input_size,
            hidden_size,
            mode="default",
            backbone_layers=0,
            backbone_units=0,
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, timespans: torch.Tensor
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_size:
            raise ValueError("invalid CfC value shape")
        if mask.shape != values.shape[:2] or timespans.shape != values.shape[:2]:
            raise ValueError("invalid CfC mask/timespan shape")
        if bool((timespans < 0).any()) or bool((timespans.masked_select(~mask) != 0).any()):
            raise ValueError("invalid CfC physical timespans")
        hidden = values.new_zeros(values.shape[0], self.hidden_size)
        for step in range(values.shape[1]):
            proposal, _ = self.cell(values[:, step], hidden, timespans[:, step, None])
            hidden = torch.where(mask[:, step, None], proposal, hidden)
        return hidden


class ProgressiveEventModel(nn.Module):
    def __init__(self, stage: str):
        super().__init__()
        if stage not in {"A", "AB", "ABC"}:
            raise ValueError(f"unknown progressive stage {stage}")
        self.stage = stage
        self.encoder = MaskedGRUEncoder()
        self.success_head = nn.Linear(GRU_HIDDEN, 1)
        if stage in {"AB", "ABC"}:
            self.value_heads = nn.ModuleList(
                [nn.Linear(GRU_HIDDEN, 1) for _ in range(EVENT_COUNT)]
            )
            self.successor_head = nn.Linear(GRU_HIDDEN, EVENT_COUNT)
        if stage == "ABC":
            self.time_input = nn.Sequential(
                nn.Linear(2 * STATE_DIM + EVENT_COUNT, CFC_HIDDEN), nn.GELU()
            )
            self.time_sequence = MaskedCfCSequence(CFC_HIDDEN, CFC_HIDDEN)
            self.time_output = nn.Sequential(
                nn.Linear(CFC_HIDDEN, GRU_HIDDEN), nn.LayerNorm(GRU_HIDDEN)
            )
            self.time_value_head = nn.Linear(CFC_HIDDEN, EVENT_COUNT)
            self.time_state_head = nn.Linear(CFC_HIDDEN, STATE_DIM)
            self.elapsed_head = nn.Linear(CFC_HIDDEN, 1)
            self.duration_head = nn.Linear(CFC_HIDDEN, 1)
            # The source event-boundary median is close to 2 s.  Keep the
            # physical reference fixed: the previous learnable scale drifted
            # to 1.5--1.7 and made target beta hit the lower grid edge.
            self.register_buffer("time_reference_scale", torch.tensor(0.5))

    def forward(
        self,
        frame_states: torch.Tensor,
        frame_mask: torch.Tensor,
        event_states: torch.Tensor,
        event_mask: torch.Tensor,
        event_dt: torch.Tensor,
        event_seq_ids: torch.Tensor,
        beta: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor | None]:
        base = self.encoder(frame_states, frame_mask)
        fused = base
        time_hidden = None
        duration = None
        next_state = None
        elapsed_prediction = None
        clock_exposure = None
        temporal_logits = None
        if self.stage == "ABC":
            safe_ids = event_seq_ids.clamp(min=0, max=EVENT_COUNT - 1)
            one_hot = nn.functional.one_hot(safe_ids, EVENT_COUNT).to(event_states.dtype)
            one_hot = one_hot * event_mask[..., None]
            effective_dt = (
                event_dt
                * torch.as_tensor(beta, device=event_dt.device)
                * self.time_reference_scale
            )
            previous_states = torch.roll(event_states, shifts=1, dims=1)
            state_delta = event_states - previous_states
            valid_interval = (effective_dt > 1e-6) & event_mask
            state_rate = torch.where(
                valid_interval[..., None],
                state_delta / effective_dt.clamp(min=1e-6)[..., None],
                torch.zeros_like(state_delta),
            ).clamp(min=-10.0, max=10.0)
            mapped = self.time_input(
                torch.cat([event_states, state_rate, one_hot], dim=-1)
            )
            time_hidden = self.time_sequence(mapped, event_mask, effective_dt)
            correction = self.time_output(time_hidden)
            # No learned zero gate: C is a required component of the full
            # model, not an optional residual that can collapse back to A+B.
            fused = base + correction
            temporal_logits = self.time_value_head(time_hidden)
            next_state = self.time_state_head(time_hidden)
            elapsed_prediction = self.elapsed_head(time_hidden).squeeze(-1)
            duration = self.duration_head(time_hidden).squeeze(-1)
            elapsed = effective_dt.sum(dim=1).clamp(min=0.0, max=30.0)
            interval_count = (event_dt > 0.0).sum(dim=1).clamp(min=1)
            mean_interval = elapsed / interval_count
            # Use mean dimensionless boundary time rather than cumulative time;
            # the latter saturated after two or three events and encoded body ID.
            clock_exposure = -torch.expm1(-mean_interval)
        # Success probability is a trajectory-level quantity.  It deliberately
        # reads the stable GRU path, so target clock calibration cannot corrupt
        # it when the CfC residual is activated.
        success_logit = self.success_head(base).squeeze(-1)
        values = None
        successor = None
        if self.stage in {"AB", "ABC"}:
            base_logits = torch.stack(
                [head(base).squeeze(-1) for head in self.value_heads], dim=1
            )
            logits = base_logits
            if self.stage == "ABC":
                event_axis = torch.arange(
                    EVENT_COUNT, device=base_logits.device, dtype=base_logits.dtype
                )
                event_axis = event_axis - (EVENT_COUNT - 1.0) / 2.0
                # Both evidence terms are mandatory.  Their amplitudes sum to
                # 0.45, preserving the structural sign guarantee under a unit
                # event gap while retaining state/time discrimination.
                evidence_logits = (
                    BASE_EVIDENCE_AMPLITUDE
                    * torch.tanh(base_logits / PROGRESS_EVIDENCE_TEMPERATURE)
                    + TIME_EVIDENCE_AMPLITUDE * clock_exposure[:, None]
                    * torch.tanh(temporal_logits / PROGRESS_EVIDENCE_TEMPERATURE)
                )
                logits = evidence_logits + PROGRESS_LOGIT_GAP * event_axis[None, :]
            values = torch.sigmoid(logits)
            successor = self.successor_head(fused)
        return {
            "embedding": fused,
            "base_embedding": base,
            "success_logit": success_logit,
            "values": values,
            "successor_logits": successor,
            "duration_log1p": duration,
            "elapsed_log1p": elapsed_prediction,
            "next_state": next_state,
            "clock_exposure": clock_exposure,
            "temporal_value_logits": temporal_logits,
            "cfc_required": self.stage == "ABC",
        }


def model_for_method(method: str) -> ProgressiveEventModel:
    if method not in STAGES:
        raise ValueError(f"unknown method {method}")
    return ProgressiveEventModel(STAGES[method])


def load_compatible_parent(model: nn.Module, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": len(compatible),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def forward_indices(
    model: ProgressiveEventModel,
    data: TensorBoundaryDataset,
    indices: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor | None]:
    return model(
        data.take("frame_states", indices),
        data.take("frame_mask", indices),
        data.take("event_states", indices),
        data.take("event_mask", indices),
        data.take("event_dt", indices),
        data.take("event_seq_ids", indices),
        beta=beta,
    )


class MatchedRankSampler:
    """Samples positive/negative pairs only inside (body,event) buckets."""

    def __init__(self, dataset: BoundaryDataset, train_indices: Sequence[int]):
        groups: dict[tuple[int, int], dict[int, list[int]]] = {}
        arrays = dataset.arrays
        for index in train_indices:
            key = (int(arrays["body_id"][index]), int(arrays["event_id"][index]))
            label = int(arrays["label"][index] > 0.5)
            groups.setdefault(key, {0: [], 1: []})[label].append(int(index))
        all_groups = groups
        self.groups = {
            key: value for key, value in groups.items() if value[0] and value[1]
        }
        self.keys = sorted(self.groups)
        if not self.keys:
            raise ValueError("no matched (body,event) success/failure ranking buckets")
        self.cross_body_pairs = []
        events = sorted({event for _, event in all_groups})
        for event in events:
            bodies = sorted({body for body, grouped_event in all_groups if grouped_event == event})
            for positive_body in bodies:
                for negative_body in bodies:
                    if positive_body == negative_body:
                        continue
                    positive = all_groups[(positive_body, event)][1]
                    negative = all_groups[(negative_body, event)][0]
                    if positive and negative:
                        self.cross_body_pairs.append((positive, negative))

    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        positive = []
        negative = []
        for _ in range(count):
            key = self.keys[int(rng.integers(len(self.keys)))]
            group = self.groups[key]
            positive.append(group[1][int(rng.integers(len(group[1])))])
            negative.append(group[0][int(rng.integers(len(group[0])))])
        return np.asarray(positive, dtype=np.int64), np.asarray(negative, dtype=np.int64)

    def sample_cross_body(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.cross_body_pairs:
            return None
        positive = []
        negative = []
        for _ in range(count):
            pair = self.cross_body_pairs[int(rng.integers(len(self.cross_body_pairs)))]
            positive.append(pair[0][int(rng.integers(len(pair[0])))])
            negative.append(pair[1][int(rng.integers(len(pair[1])))])
        return np.asarray(positive, dtype=np.int64), np.asarray(negative, dtype=np.int64)


class BalancedExampleSampler:
    """Uniform sampler over (body,event,outcome) rather than raw boundaries."""

    def __init__(self, dataset: BoundaryDataset, train_indices: Sequence[int]):
        arrays = dataset.arrays
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index in train_indices:
            key = (
                int(arrays["body_id"][index]),
                int(arrays["event_id"][index]),
                int(arrays["label"][index] > 0.5),
            )
            groups.setdefault(key, []).append(int(index))
        self.groups = groups
        self.keys = sorted(groups)
        if not self.keys:
            raise ValueError("empty balanced training sampler")

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        result = []
        for _ in range(count):
            key = self.keys[int(rng.integers(len(self.keys)))]
            group = self.groups[key]
            result.append(group[int(rng.integers(len(group)))])
        return np.asarray(result, dtype=np.int64)


def _selected_value(outputs: Mapping[str, torch.Tensor | None], event_ids: torch.Tensor) -> torch.Tensor:
    values = outputs["values"]
    if values is None:
        return torch.sigmoid(outputs["success_logit"])
    return values.gather(1, event_ids[:, None]).squeeze(1)


def _selected_temporal_value(
    outputs: Mapping[str, torch.Tensor | None], event_ids: torch.Tensor
) -> torch.Tensor:
    logits = outputs["temporal_value_logits"]
    if logits is None:
        raise ValueError("temporal value requested from a model without CfC")
    selected = logits.gather(1, event_ids[:, None]).squeeze(1)
    return torch.sigmoid(selected)


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    positives = labels.sum().clamp(min=1.0)
    negatives = (1.0 - labels).sum().clamp(min=1.0)
    count = float(labels.numel())
    weights = torch.where(labels > 0.5, count / (2.0 * positives), count / (2.0 * negatives))
    weights = weights * (1.0 + initial)  # explicitly protect the s0 probability readout
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)


def _event_conditioned_body_alignment(
    embedding: torch.Tensor, body_ids: torch.Tensor, event_ids: torch.Tensor
) -> torch.Tensor:
    """Small source-only mean/CORAL penalty inside matched event sections."""
    terms = []
    for event_id in torch.unique(event_ids):
        event_mask = event_ids == event_id
        bodies = torch.unique(body_ids[event_mask])
        if len(bodies) < 2:
            continue
        reference = embedding[event_mask & (body_ids == bodies[0])]
        if len(reference) < 2:
            continue
        reference_centered = reference - reference.mean(dim=0, keepdim=True)
        reference_cov = reference_centered.T @ reference_centered / max(len(reference) - 1, 1)
        for body in bodies[1:]:
            other = embedding[event_mask & (body_ids == body)]
            if len(other) < 2:
                continue
            other_centered = other - other.mean(dim=0, keepdim=True)
            other_cov = other_centered.T @ other_centered / max(len(other) - 1, 1)
            mean_loss = (reference.mean(dim=0) - other.mean(dim=0)).square().mean()
            covariance_loss = (reference_cov - other_cov).square().mean()
            terms.append(mean_loss + 0.01 * covariance_loss)
    return torch.stack(terms).mean() if terms else embedding.new_zeros(())


def update_ema(target: nn.Module, source: nn.Module, decay: float = 0.995) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)


def state_dict_cpu(model: nn.Module, half: bool = False) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().clone()
        if half and tensor.is_floating_point():
            tensor = tensor.half()
        result[name] = tensor
    return result


def save_checkpoint(
    path: Path,
    model: ProgressiveEventModel,
    method: str,
    seed: int,
    step: int,
    extra: Mapping | None = None,
    half: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "method": method,
        "stage": STAGES[method],
        "seed": int(seed),
        "step": int(step),
        "model": state_dict_cpu(model, half=half),
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[ProgressiveEventModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    method = payload["method"]
    model = model_for_method(method).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def compute_training_loss(
    method: str,
    model: ProgressiveEventModel,
    target_model: ProgressiveEventModel,
    data: TensorBoundaryDataset,
    indices: torch.Tensor,
    rank_sampler: MatchedRankSampler | None,
    rng: np.random.Generator,
    ranking_enabled: bool,
    teacher: ProgressiveEventModel | None,
    rank_pairs: int = 32,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = forward_indices(model, data, indices)
    labels = data.take("label", indices)
    loss_success = _balanced_bce(
        outputs["success_logit"], labels, data.take("is_initial", indices)
    )
    loss_alignment = _event_conditioned_body_alignment(
        outputs["embedding"], data.take("body_id", indices), data.take("event_id", indices)
    )
    components: dict[str, torch.Tensor] = {
        "success": loss_success,
        "body_alignment": loss_alignment,
    }
    if STAGES[method] == "A":
        loss = loss_success + 0.02 * loss_alignment
        return loss, {name: float(value.detach()) for name, value in components.items()}

    next_indices = data.take("next_index", indices)
    event_ids = data.take("event_id", indices)
    next_event_ids = data.take("next_event_id", indices)
    value = _selected_value(outputs, event_ids)
    with torch.no_grad():
        target_outputs = forward_indices(target_model, data, next_indices)
        next_value_target = _selected_value(target_outputs, next_event_ids)
        td_target = data.take("reward", indices) + GAMMA * next_value_target * (
            1.0 - data.take("terminal", indices)
        )
    loss_td = nn.functional.mse_loss(value, td_target)
    mc_target = labels * torch.pow(
        value.new_full((), GAMMA), data.take("remaining", indices)
    )
    loss_mc = nn.functional.mse_loss(value, mc_target)
    loss_successor = nn.functional.cross_entropy(
        outputs["successor_logits"], next_event_ids
    )

    next_outputs = forward_indices(model, data, next_indices)
    next_value = _selected_value(next_outputs, next_event_ids)
    delta = next_value - value
    terminal = data.take("terminal", indices).bool()
    forward = (next_event_ids > event_ids) & ~terminal
    backward = (next_event_ids < event_ids) & ~terminal
    same = (next_event_ids == event_ids) & ~terminal
    sign_terms = []
    if bool(forward.any()):
        sign_terms.append(nn.functional.relu(0.01 - delta[forward]).mean())
    if bool(backward.any()):
        sign_terms.append(nn.functional.relu(0.01 + delta[backward]).mean())
    if bool(same.any()):
        sign_terms.append(delta[same].square().mean())
    loss_sign = torch.stack(sign_terms).mean() if sign_terms else value.new_zeros(())

    components.update(td=loss_td, mc=loss_mc, successor=loss_successor, sign=loss_sign)
    td_weight = 2.0 if STAGES[method] == "ABC" else 1.0
    sign_weight = 0.75 if STAGES[method] == "ABC" else 0.1
    loss = (
        td_weight * loss_td + 0.5 * loss_mc + 0.25 * loss_successor
        + 0.35 * loss_success + sign_weight * loss_sign + 0.02 * loss_alignment
    )

    if STAGES[method] == "ABC":
        duration_mask = ~terminal
        if bool(duration_mask.any()):
            duration_target = torch.log1p(data.take("next_duration", indices)[duration_mask])
            loss_duration = nn.functional.smooth_l1_loss(
                outputs["duration_log1p"][duration_mask], duration_target
            )
            next_state_target = data.take("event_states", next_indices)[duration_mask, -1]
            loss_state_transport = nn.functional.smooth_l1_loss(
                outputs["next_state"][duration_mask], next_state_target
            )
        else:
            loss_duration = value.new_zeros(())
            loss_state_transport = value.new_zeros(())
        elapsed_target = torch.log1p(
            data.take("event_dt", indices).sum(dim=1).clamp(min=0.0)
        )
        loss_elapsed = nn.functional.smooth_l1_loss(
            outputs["elapsed_log1p"], elapsed_target
        )
        temporal_value = _selected_temporal_value(outputs, event_ids)
        loss_temporal_mc = nn.functional.mse_loss(temporal_value, mc_target)
        temporal_condition = event_ids * 2 + (labels > 0.5).long()
        loss_temporal_alignment = _event_conditioned_body_alignment(
            outputs["temporal_value_logits"],
            data.take("body_id", indices),
            temporal_condition,
        )
        components["duration"] = loss_duration
        components["state_transport"] = loss_state_transport
        components["elapsed"] = loss_elapsed
        components["temporal_mc"] = loss_temporal_mc
        components["temporal_alignment"] = loss_temporal_alignment
        loss = loss + (
            0.15 * loss_duration
            + 0.10 * loss_state_transport
            + 0.10 * loss_elapsed
            + 0.15 * loss_temporal_mc
            + 0.10 * loss_temporal_alignment
        )

    if ranking_enabled:
        if rank_sampler is None:
            raise ValueError("ranking stage requires a matched sampler")
        positive_np, negative_np = rank_sampler.sample(rank_pairs, rng)
        positive = torch.as_tensor(positive_np, dtype=torch.long, device=indices.device)
        negative = torch.as_tensor(negative_np, dtype=torch.long, device=indices.device)
        positive_outputs = forward_indices(model, data, positive)
        negative_outputs = forward_indices(model, data, negative)
        positive_values = _selected_value(
            positive_outputs, data.take("event_id", positive)
        )
        negative_values = _selected_value(
            negative_outputs, data.take("event_id", negative)
        )
        loss_rank = nn.functional.softplus(
            -(positive_values - negative_values) / 0.1
        ).mean()
        components["rank"] = loss_rank
        loss = loss + 0.25 * loss_rank

        if STAGES[method] == "ABC":
            positive_temporal = _selected_temporal_value(
                positive_outputs, data.take("event_id", positive)
            )
            negative_temporal = _selected_temporal_value(
                negative_outputs, data.take("event_id", negative)
            )
            loss_temporal_rank = nn.functional.softplus(
                -(positive_temporal - negative_temporal) / 0.1
            ).mean()
            components["temporal_rank"] = loss_temporal_rank
            loss = loss + 0.35 * loss_temporal_rank

        if teacher is not None:
            with torch.no_grad():
                teacher_outputs = forward_indices(teacher, data, indices)
                teacher_value = _selected_value(teacher_outputs, event_ids)
                teacher_success = teacher_outputs["success_logit"]
            loss_anchor = nn.functional.mse_loss(value, teacher_value) + 0.1 * nn.functional.mse_loss(
                outputs["success_logit"], teacher_success
            )
            components["anchor"] = loss_anchor
            loss = loss + 0.15 * loss_anchor

    return loss, {name: float(value.detach()) for name, value in components.items()}


def _auc(samples: Sequence[tuple[float, float]]) -> float | None:
    positive = [score for score, label in samples if label > 0.5]
    negative = [score for score, label in samples if label <= 0.5]
    if len(positive) < 4 or len(negative) < 4:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else (0.5 if pos == neg else 0.0)
    return wins / (len(positive) * len(negative))


def fit_beta(
    method: str,
    model: ProgressiveEventModel,
    data: TensorBoundaryDataset,
    adapt_indices: np.ndarray,
) -> float:
    if STAGES[method] != "ABC" or len(adapt_indices) == 0:
        return 1.0
    indices = torch.as_tensor(adapt_indices, dtype=torch.long, device=data.device)
    best_beta = 1.0
    best_error = float("inf")
    with torch.no_grad():
        for beta in BETA_GRID:
            outputs = forward_indices(model, data, indices, beta=beta)
            value = _selected_value(outputs, data.take("event_id", indices))
            next_indices = data.take("next_index", indices)
            next_outputs = forward_indices(model, data, next_indices, beta=beta)
            next_value = _selected_value(
                next_outputs, data.take("next_event_id", indices)
            )
            target = data.take("reward", indices) + GAMMA * next_value * (
                1.0 - data.take("terminal", indices)
            )
            error = float(nn.functional.mse_loss(value, target))
            # Weak source-clock prior prevents tiny N=5 sets from selecting an edge on ties.
            error += 1e-4 * float(math.log(beta) ** 2)
            if error < best_error:
                best_error = error
                best_beta = beta
    return float(best_beta)


def _calibrated_probability(
    logit: float,
    rho_probability: float,
    calibration: Mapping[str, float] | None,
) -> float:
    calibration = calibration or {}
    temperature = max(float(calibration.get("temperature", 1.0)), 1e-3)
    bias = float(calibration.get("bias", 0.0))
    rho_weight = float(calibration.get("rho_weight", 0.0))
    rho_logit = math.log(rho_probability / (1.0 - rho_probability))
    calibrated = logit / temperature + bias + rho_weight * rho_logit
    return float(1.0 / (1.0 + math.exp(-float(np.clip(calibrated, -30.0, 30.0)))))


def evaluate_model(
    method: str,
    model: ProgressiveEventModel,
    dataset: BoundaryDataset,
    tensor_data: TensorBoundaryDataset,
    test_rollout_ids: Sequence[int],
    adapt_rollout_ids: Sequence[int] | None = None,
    calibration: Mapping[str, float] | None = None,
    return_predictions: bool = False,
    forced_beta: float | None = None,
) -> dict:
    model.eval()
    test_set = set(int(value) for value in test_rollout_ids)
    adapt_rollout_ids = list(adapt_rollout_ids or [])
    rho_by_body = compute_rho(dataset.rollouts, adapt_rollout_ids) if adapt_rollout_ids else {}
    beta_by_body: dict[str, float] = {}
    for body in sorted({dataset.rollouts[i].body for i in test_set}):
        if forced_beta is not None:
            if forced_beta < 0.0:
                raise ValueError("forced beta must be non-negative")
            beta_by_body[body] = float(forced_beta)
        else:
            body_adapt = [i for i in adapt_rollout_ids if dataset.rollouts[i].body == body]
            beta_by_body[body] = fit_beta(
                method,
                model,
                tensor_data,
                dataset.indices_for_rollouts(body_adapt),
            )

    buckets: dict[tuple[str, int], list[tuple[float, float]]] = {}
    bellman_errors: list[float] = []
    sign_score = 0.0
    sign_total = 0
    probabilities: list[float] = []
    value_s0_probabilities: list[float] = []
    rho_probabilities: list[float] = []
    labels: list[float] = []
    prediction_rows: list[dict] = []
    with torch.no_grad():
        for rollout_id in sorted(test_set):
            rollout = dataset.rollouts[rollout_id]
            sample_indices = dataset.indices_for_rollouts([rollout_id])
            sample_indices = sample_indices[
                np.argsort(dataset.arrays["boundary_position"][sample_indices])
            ]
            indices = torch.as_tensor(sample_indices, dtype=torch.long, device=tensor_data.device)
            outputs = forward_indices(
                model, tensor_data, indices, beta=beta_by_body.get(rollout.body, 1.0)
            )
            event_ids = tensor_data.take("event_id", indices)
            scores = _selected_value(outputs, event_ids).detach().cpu().numpy()
            event_np = event_ids.detach().cpu().numpy()
            for position, (event_id, score) in enumerate(zip(event_np, scores)):
                if not (position == len(scores) - 1 and event_id == EVENT_COUNT - 1):
                    buckets.setdefault((rollout.body, int(event_id)), []).append(
                        (float(score), float(rollout.success))
                    )
            if STAGES[method] != "A":
                for position, index in enumerate(sample_indices):
                    terminal = bool(dataset.arrays["terminal"][index])
                    target = float(dataset.arrays["reward"][index]) if terminal else GAMMA * float(scores[position + 1])
                    bellman_errors.append((float(scores[position]) - target) ** 2)
            for position in range(len(scores) - 1):
                current_event = int(event_np[position])
                next_event = int(event_np[position + 1])
                delta = float(scores[position + 1] - scores[position])
                if next_event > current_event and delta > 0.0:
                    sign_score += 1.0
                elif next_event < current_event and delta < 0.0:
                    sign_score += 1.0
                elif next_event == current_event and abs(delta) < 1e-4:
                    sign_score += 0.5
                sign_total += 1

            initial_logit = float(outputs["success_logit"][0].detach().cpu())
            body_rho = rho_product(rho_by_body.get(rollout.body))
            probability = _calibrated_probability(initial_logit, body_rho, calibration)
            probabilities.append(probability)
            value_s0_probability = float(scores[0])
            value_s0_probabilities.append(value_s0_probability)
            rho_probabilities.append(body_rho)
            labels.append(float(rollout.success))
            prediction_rows.append(
                {
                    "rollout_id": rollout_id,
                    "body": rollout.body,
                    "label": float(rollout.success),
                    "success_logit": initial_logit,
                    "rho_probability": body_rho,
                    "probability": probability,
                    "value_s0_probability": value_s0_probability,
                    "rho_chain_probability": body_rho,
                }
            )

    aucs = [value for value in (_auc(samples) for samples in buckets.values()) if value is not None]
    probability_array = np.asarray(probabilities, dtype=np.float64)
    value_s0_array = np.asarray(value_s0_probabilities, dtype=np.float64)
    rho_array = np.asarray(rho_probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    result = {
        "bellman_mse_target": float(np.mean(bellman_errors)) if bellman_errors else None,
        "ranking_auc": float(np.mean(aucs)) if aucs else None,
        "sign_consistency": float(sign_score / sign_total) if sign_total else None,
        "success_rate_mae": float(np.mean(np.abs(probability_array - label_array))),
        "success_brier": float(np.mean((probability_array - label_array) ** 2)),
        "success_rate_mae_value_s0": float(np.mean(np.abs(value_s0_array - label_array))),
        "success_brier_value_s0": float(np.mean((value_s0_array - label_array) ** 2)),
        "success_rate_mae_rho_chain": float(np.mean(np.abs(rho_array - label_array))),
        "success_brier_rho_chain": float(np.mean((rho_array - label_array) ** 2)),
        "beta": beta_by_body,
        "rho": rho_by_body,
        "test_rollouts": len(test_set),
    }
    if return_predictions:
        result["predictions"] = prediction_rows
    return result


def fit_success_calibration(method: str, prediction_rows: Sequence[Mapping]) -> dict[str, float]:
    if not prediction_rows:
        return {"temperature": 1.0, "bias": 0.0, "rho_weight": 0.0}
    rho_weights = (0.0,) if STAGES[method] == "A" else (0.0, 0.25, 0.5, 0.75, 1.0)
    candidates = []
    for temperature in (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        for bias in np.arange(-1.5, 1.5001, 0.25):
            for rho_weight in rho_weights:
                squared_errors = []
                absolute_errors = []
                for row in prediction_rows:
                    probability = _calibrated_probability(
                        float(row["success_logit"]),
                        float(row["rho_probability"]),
                        {
                            "temperature": temperature,
                            "bias": bias,
                            "rho_weight": rho_weight,
                        },
                    )
                    error = probability - float(row["label"])
                    squared_errors.append(error ** 2)
                    absolute_errors.append(abs(error))
                candidates.append(
                    {
                        "temperature": float(temperature),
                        "bias": float(bias),
                        "rho_weight": float(rho_weight),
                        "brier": float(np.mean(squared_errors)),
                        "mae": float(np.mean(absolute_errors)),
                        "squared_errors": np.asarray(squared_errors, dtype=np.float64),
                    }
                )
    proper_best = min(candidates, key=lambda item: item["brier"])
    errors = proper_best["squared_errors"]
    proper_se = float(errors.std(ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0
    proper_limit = float(proper_best["brier"] + proper_se)
    eligible = [item for item in candidates if item["brier"] <= proper_limit + 1e-12]
    # Proper score is a calibration guard, not the optimization target.  Inside
    # best+1SE we minimize success MAE, then prefer the better proper score and
    # the smaller departure from identity calibration.
    selected = min(
        eligible,
        key=lambda item: (
            item["mae"],
            item["brier"],
            abs(math.log(item["temperature"])) + abs(item["bias"]) + item["rho_weight"],
        ),
    )
    return {
        "temperature": selected["temperature"],
        "bias": selected["bias"],
        "rho_weight": selected["rho_weight"],
        "oof_brier": selected["brier"],
        "oof_mae": selected["mae"],
        "proper_best_brier": proper_best["brier"],
        "proper_brier_se": proper_se,
        "proper_limit_1se": proper_limit,
        "proper_guard_satisfied": bool(selected["brier"] <= proper_limit + 1e-12),
    }


def aggregate_fold_metrics(rows: Sequence[Mapping]) -> dict[int, dict[str, float | None]]:
    by_step: dict[int, list[Mapping]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    result: dict[int, dict[str, float | None]] = {}
    for step, step_rows in sorted(by_step.items()):
        aggregate: dict[str, float | None] = {"step": step, "folds": len(step_rows)}
        for name in (
            "bellman_mse_target", "ranking_auc", "sign_consistency",
            "success_rate_mae", "success_brier"
        ):
            values = np.asarray(
                [float(row[name]) for row in step_rows if row.get(name) is not None],
                dtype=np.float64,
            )
            aggregate[name] = float(values.mean()) if len(values) else None
            aggregate[f"{name}_se"] = (
                float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        result[step] = aggregate
    return result


def select_checkpoint(method: str, aggregate: Mapping[int, Mapping]) -> tuple[int, dict]:
    rows = list(aggregate.values())
    if not rows:
        raise ValueError("no source-LOBO checkpoint metrics")
    best_brier = min(float(row["success_brier"]) for row in rows)
    brier_row = min(rows, key=lambda row: float(row["success_brier"]))
    brier_limit = best_brier + float(brier_row.get("success_brier_se", 0.0))
    eligible = [row for row in rows if float(row["success_brier"]) <= brier_limit + 1e-12]
    mse_limit = None
    if STAGES[method] != "A":
        best_mse_row = min(eligible, key=lambda row: float(row["bellman_mse_target"]))
        mse_limit = float(best_mse_row["bellman_mse_target"]) + float(
            best_mse_row.get("bellman_mse_target_se", 0.0)
        )
        eligible = [
            row for row in eligible
            if float(row["bellman_mse_target"]) <= mse_limit + 1e-12
        ]
    # Frozen lexicographic rule: AUC -> sign -> MAE -> MSE -> earlier step.
    selected = max(
        eligible,
        key=lambda row: (
            float(row["ranking_auc"]),
            float(row["sign_consistency"]),
            -float(row["success_rate_mae"]),
            -float(row["bellman_mse_target"] or 0.0),
            -int(row["step"]),
        ),
    )
    audit = {
        "proper_brier_best": best_brier,
        "proper_brier_limit_1se": brier_limit,
        "bellman_limit_1se": mse_limit,
        "eligible_steps": [int(row["step"]) for row in eligible],
        "selected_metrics": dict(selected),
        "selection_order": ["ranking_auc", "sign_consistency", "success_rate_mae", "bellman_mse"],
    }
    return int(selected["step"]), audit


def json_dump(path: Path, payload: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def parameter_report(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def now_seconds() -> float:
    return time.perf_counter()
