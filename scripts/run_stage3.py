#!/usr/bin/env python3
"""Run the factorized ETSF Stage-3 semantic/clock transport study.

Stage 3 keeps embodiment-invariant event semantics separate from the target
clock.  The shared semantic encoder and successor head never receive beta.
An independent liquid clock head uses beta to predict event-duration
distributions, and an explicit semi-Markov transport operator combines the two.

The current source dataset contains successful demonstrations only.  Ranking
loss support is implemented, but the run report records zero usable source
success/failure pairs instead of pretending that ranking was supervised.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import run_stage2 as s2


MIN_CLOCK_STEPS = 5.0
SEMANTIC_BINS = 51
SEMANTIC_HIDDEN = 96
CLOCK_HIDDEN = 64
BETA_GRID = np.linspace(-2.0, 2.0, 161, dtype=np.float32)
BETA_PRIOR_STD = 0.75
SYNTHETIC_WARP_LIMIT = 0.70
POSTERIOR_STD_FALLBACK = 0.75
POSTERIOR_EDGE_FALLBACK = 0.25
MIN_POSTERIOR_OBSERVATIONS = 12
MIN_DECISION_PAIRS = 100
GAUSS_HERMITE_NODES, GAUSS_HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(7)


@dataclass
class Posterior:
    grid: np.ndarray
    weights: np.ndarray
    mean: float
    std: float
    map_value: float
    edge_mass: float
    n_observations: int
    fallback: bool
    fallback_reasons: list[str]


def pack_stage3(records: list[s2.BoundaryRecord], device: torch.device) -> dict[str, torch.Tensor]:
    """Pack records and add causal semantic, MC-return, and clock targets."""
    batch = s2.pack_records(records, device)
    count, length = batch["mask"].shape
    events = len(s2.MODEL_EVENTS)
    semantic_targets = np.zeros((count, length, events), dtype=np.float32)
    mc_targets = np.zeros((count, length, events), dtype=np.float32)
    clock_mask = np.zeros((count, length), dtype=bool)
    censor_mask = np.zeros((count, length), dtype=bool)
    censor_durations = np.zeros((count, length), dtype=np.float32)
    task_ids = np.zeros(count, dtype=np.int64)

    for row, record in enumerate(records):
        frames = np.asarray([frame for _, frame in record.episode.events], dtype=np.float32)
        size = len(record.event_ids)
        task_ids[row] = s2.TASKS.index(record.episode.task)
        for current in range(size):
            for future in range(current, size):
                event_id = int(record.event_ids[future])
                semantic_targets[row, current, event_id] = 1.0
                elapsed = float(frames[future] - frames[current])
                mc_targets[row, current, event_id] = s2.GAMMA**elapsed
            if current < len(record.durations):
                duration = float(record.durations[current])
                clock_mask[row, current] = math.isfinite(duration) and duration >= MIN_CLOCK_STEPS

        if (
            size
            and not record.episode.success
            and int(record.event_ids[-1]) != s2.MODEL_EVENTS.index("eK")
        ):
            censor_duration = float(record.episode.steps - 1 - frames[-1])
            if censor_duration >= MIN_CLOCK_STEPS:
                censor_mask[row, size - 1] = True
                censor_durations[row, size - 1] = censor_duration

    batch.update(
        {
            "semantic_targets": torch.from_numpy(semantic_targets).to(device),
            "mc_targets": torch.from_numpy(mc_targets).to(device),
            "clock_mask": torch.from_numpy(clock_mask).to(device),
            "censor_mask": torch.from_numpy(censor_mask).to(device),
            "censor_durations": torch.from_numpy(censor_durations).to(device),
            "task_ids": torch.from_numpy(task_ids).to(device),
        }
    )
    return batch


class SharedSemanticEncoder(nn.Module):
    """Embodiment-invariant event encoder; it has no time or beta input."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_map = nn.Sequential(
            nn.Linear(input_dim, SEMANTIC_HIDDEN),
            nn.GELU(),
            nn.Linear(SEMANTIC_HIDDEN, SEMANTIC_HIDDEN),
            nn.LayerNorm(SEMANTIC_HIDDEN),
        )
        self.cell = nn.GRUCell(SEMANTIC_HIDDEN, SEMANTIC_HIDDEN)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = inputs.new_zeros(inputs.shape[0], SEMANTIC_HIDDEN)
        outputs = []
        for index in range(inputs.shape[1]):
            proposed = self.cell(self.input_map(inputs[:, index]), hidden)
            hidden = torch.where(mask[:, index, None], proposed, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, 1)


class ClockLiquidCell(nn.Module):
    """Low-rank continuous-time cell used only inside the clock branch."""

    def __init__(self) -> None:
        super().__init__()
        width = SEMANTIC_HIDDEN + CLOCK_HIDDEN
        self.candidate = nn.Linear(width, CLOCK_HIDDEN)
        self.base_tau = nn.Linear(width, CLOCK_HIDDEN)
        self.beta_shape = nn.Linear(width, CLOCK_HIDDEN)

    def forward(
        self,
        semantic: torch.Tensor,
        hidden: torch.Tensor,
        timespan: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([semantic, hidden], dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        base = math.log(10.0) + 1.5 * torch.tanh(self.base_tau(joined))
        shape = torch.tanh(self.beta_shape(joined))
        shape = shape - shape.mean(-1, keepdim=True)
        shape = shape / torch.sqrt(torch.square(shape).mean(-1, keepdim=True) + 1e-6)
        log_tau = torch.clamp(base + 0.5 * beta[:, None] * shape, -3.0, 7.0)
        decay = torch.exp(-timespan[:, None] / torch.exp(log_tau))
        return decay * hidden + (1.0 - decay) * candidate, log_tau


class FactorizedEventTransport(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.semantic = SharedSemanticEncoder(input_dim)
        self.successor = nn.Linear(
            SEMANTIC_HIDDEN, len(s2.MODEL_EVENTS) * SEMANTIC_BINS
        )
        self.success = nn.Linear(SEMANTIC_HIDDEN, 1)
        self.clock_cell = ClockLiquidCell()
        self.duration_mean = nn.Linear(CLOCK_HIDDEN, len(s2.MODEL_EVENTS))
        self.duration_scale = nn.Linear(CLOCK_HIDDEN, len(s2.MODEL_EVENTS))
        self.beta_arx = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        inputs: torch.Tensor,
        dts: torch.Tensor,
        mask: torch.Tensor,
        beta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        semantic = self.semantic(inputs, mask)
        successor_logits = self.successor(semantic).view(
            *semantic.shape[:2], len(s2.MODEL_EVENTS), SEMANTIC_BINS
        )
        success_logits = self.success(semantic).squeeze(-1)

        # Stop-gradient is deliberate: duration supervision cannot rewrite the
        # embodiment-invariant semantic geometry.
        clock_hidden = inputs.new_zeros(inputs.shape[0], CLOCK_HIDDEN)
        means = []
        log_scales = []
        log_taus = []
        for index in range(inputs.shape[1]):
            proposed, log_tau = self.clock_cell(
                semantic[:, index].detach(), clock_hidden, dts[:, index], beta
            )
            clock_hidden = torch.where(mask[:, index, None], proposed, clock_hidden)
            means.append(F.softplus(self.duration_mean(clock_hidden)))
            log_scales.append(torch.clamp(self.duration_scale(clock_hidden), -3.0, 1.5))
            log_taus.append(log_tau)
        return {
            "semantic": semantic,
            "successor_logits": successor_logits,
            "success_logits": success_logits,
            "duration_log_mean": torch.stack(means, 1),
            "duration_log_scale": torch.stack(log_scales, 1),
            "clock_log_tau": torch.stack(log_taus, 1),
        }


def semantic_support(device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, SEMANTIC_BINS, device=device)


def decode_semantic(logits: torch.Tensor) -> torch.Tensor:
    support = semantic_support(logits.device)
    return (torch.softmax(logits, dim=-1) * support).sum(-1)


def fixed_hl_gauss(target: torch.Tensor) -> torch.Tensor:
    support = semantic_support(target.device)
    width = support[1] - support[0]
    distance = (support - target[..., None]) / (0.75 * width)
    weights = torch.exp(-0.5 * torch.square(distance))
    return weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)


def select_event(values: torch.Tensor, event_ids: torch.Tensor) -> torch.Tensor:
    return values.gather(-1, event_ids[..., None]).squeeze(-1)


def normal_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    scale = torch.exp(log_scale).clamp(min=1e-3)
    return 0.5 * torch.square((target - mean) / scale) + log_scale + 0.5 * math.log(2.0 * math.pi)


def normal_censor_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    lower_bound: torch.Tensor,
) -> torch.Tensor:
    z = (lower_bound - mean) / torch.exp(log_scale).clamp(min=1e-3)
    survival = (1.0 - torch.special.ndtr(z)).clamp(min=1e-8)
    return -torch.log(survival)


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    pieces = []
    for task in range(len(s2.TASKS)):
        positive = scores[(task_ids == task) & (labels == 1)]
        negative = scores[(task_ids == task) & (labels == 0)]
        if len(positive) and len(negative):
            differences = positive[:, None] - negative[None, :]
            pieces.append(F.softplus(0.10 - differences).flatten())
    if not pieces:
        return scores.sum() * 0.0, 0
    joined = torch.cat(pieces)
    return joined.mean(), int(joined.numel())


def compute_loss(
    model: FactorizedEventTransport,
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    warps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected = {key: value[indices] for key, value in batch.items()}
    beta = selected["body_is_arx"] * model.beta_arx + warps
    warped_dts = selected["dts"] * torch.exp(warps[:, None])
    output = model(selected["inputs"], warped_dts, selected["mask"], beta)

    targets = fixed_hl_gauss(selected["semantic_targets"])
    semantic_ce = -(
        targets * torch.log_softmax(output["successor_logits"], dim=-1)
    ).sum(-1).mean(-1)
    semantic_loss = semantic_ce[selected["mask"]].mean()

    initial_success = output["success_logits"][:, 0]
    has_both_outcomes = bool(
        (selected["success"] == 0).any() and (selected["success"] == 1).any()
    )
    success_loss = (
        F.binary_cross_entropy_with_logits(initial_success, selected["success"].float())
        if has_both_outcomes
        else initial_success.sum() * 0.0
    )
    ranking, ranking_pairs = pairwise_ranking_loss(
        torch.sigmoid(initial_success), selected["success"], selected["task_ids"]
    )

    mean = select_event(output["duration_log_mean"], selected["event_ids"])
    log_scale = select_event(output["duration_log_scale"], selected["event_ids"])
    scaled_duration = selected["durations"] * torch.exp(warps[:, None])
    duration_target = torch.log1p(scaled_duration.clamp(min=0.0))
    observed_error = normal_nll(mean, log_scale, duration_target)
    observed_mask = selected["clock_mask"]
    observed_loss = (
        observed_error[observed_mask].mean()
        if observed_mask.any()
        else semantic_loss * 0.0
    )

    censor_target = torch.log1p(
        selected["censor_durations"] * torch.exp(warps[:, None])
    )
    censor_error = normal_censor_nll(mean, log_scale, censor_target)
    censor_mask = selected["censor_mask"]
    censor_loss = (
        censor_error[censor_mask].mean() if censor_mask.any() else semantic_loss * 0.0
    )
    clock_loss = observed_loss + 0.25 * censor_loss
    total = semantic_loss + 0.10 * success_loss + 0.25 * ranking + clock_loss
    metrics = {
        "total": float(total.detach()),
        "semantic": float(semantic_loss.detach()),
        "success": float(success_loss.detach()),
        "success_supervision_available": has_both_outcomes,
        "ranking": float(ranking.detach()),
        "ranking_pairs": ranking_pairs,
        "clock": float(clock_loss.detach()),
        "clock_observed": float(observed_loss.detach()),
        "clock_censored": float(censor_loss.detach()),
    }
    return total, metrics


def train_model(
    train_records: list[s2.BoundaryRecord],
    validation_records: list[s2.BoundaryRecord],
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[FactorizedEventTransport, list[dict[str, float]]]:
    torch.manual_seed(seed)
    train = pack_stage3(train_records, device)
    validation = pack_stage3(validation_records, device)
    model = FactorizedEventTransport(train["inputs"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    history = []
    started = time.time()

    for step in range(steps):
        size = min(64, len(train_records))
        indices = torch.randint(len(train_records), (size,), generator=generator, device=device)
        warps = (
            torch.rand(size, generator=generator, device=device) * 2.0 - 1.0
        ) * SYNTHETIC_WARP_LIMIT
        loss, metrics = compute_loss(model, train, indices, warps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            model.beta_arx.clamp_(-1.5, 1.5)

        if (step + 1) % 500 == 0 or step + 1 == steps:
            with torch.no_grad():
                val_indices = torch.arange(len(validation_records), device=device)
                val_warps = torch.zeros(len(validation_records), device=device)
                _, val_metrics = compute_loss(model, validation, val_indices, val_warps)
            row = {
                "step": step + 1,
                "seconds": time.time() - started,
                "beta_arx": float(model.beta_arx.detach()),
                **{f"train_{key}": value for key, value in metrics.items()},
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            print(f"TRAIN_STAGE3={seed}:{step + 1}/{steps} " + json.dumps(row, sort_keys=True), flush=True)
    return model.eval(), history


def clock_log_likelihood(
    model: FactorizedEventTransport,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> tuple[torch.Tensor, int]:
    beta_tensor = torch.full((len(batch["inputs"]),), beta, device=batch["inputs"].device)
    output = model(batch["inputs"], batch["dts"], batch["mask"], beta_tensor)
    mean = select_event(output["duration_log_mean"], batch["event_ids"])
    log_scale = select_event(output["duration_log_scale"], batch["event_ids"])
    observed = normal_nll(mean, log_scale, torch.log1p(batch["durations"].clamp(min=0.0)))
    observed_mask = batch["clock_mask"]
    censor = normal_censor_nll(
        mean, log_scale, torch.log1p(batch["censor_durations"].clamp(min=0.0))
    )
    censor_mask = batch["censor_mask"]
    # Event segments in one rollout are correlated.  Average within each
    # rollout before summing across rollouts so long chains cannot dominate the
    # body-level beta posterior.
    per_row_count = observed_mask.sum(1) + censor_mask.sum(1)
    per_row_loss = (
        (observed * observed_mask).sum(1) + (censor * censor_mask).sum(1)
    ) / per_row_count.clamp(min=1)
    total = per_row_loss[per_row_count > 0].sum()
    count = int(observed_mask.sum() + censor_mask.sum())
    return -total, count


def infer_beta_posterior(
    model: FactorizedEventTransport,
    records: list[s2.BoundaryRecord],
    device: torch.device,
) -> Posterior:
    batch = pack_stage3(records, device)
    snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
    log_weights = []
    observation_count = 0
    with torch.no_grad():
        for beta in BETA_GRID:
            likelihood, observation_count = clock_log_likelihood(model, batch, float(beta))
            prior = -0.5 * (float(beta) / BETA_PRIOR_STD) ** 2
            log_weights.append(float(likelihood) + prior)
    normalized = np.asarray(log_weights, dtype=np.float64)
    normalized -= normalized.max()
    weights = np.exp(normalized)
    weights /= weights.sum()
    mean = float(np.sum(BETA_GRID * weights))
    std = float(np.sqrt(np.sum(np.square(BETA_GRID - mean) * weights)))
    map_value = float(BETA_GRID[int(weights.argmax())])
    edge_mass = float(weights[:5].sum() + weights[-5:].sum())
    reasons = []
    if observation_count < MIN_POSTERIOR_OBSERVATIONS:
        reasons.append("insufficient_clock_observations")
    if std > POSTERIOR_STD_FALLBACK:
        reasons.append("posterior_too_wide")
    if edge_mass > POSTERIOR_EDGE_FALLBACK:
        reasons.append("posterior_at_support_edge")
    for name, value in model.state_dict().items():
        if not torch.equal(value, snapshot[name]):
            raise RuntimeError(f"shared parameter changed during posterior inference: {name}")
    return Posterior(
        grid=BETA_GRID.copy(),
        weights=weights.astype(np.float64),
        mean=mean,
        std=std,
        map_value=map_value,
        edge_mass=edge_mass,
        n_observations=observation_count,
        fallback=bool(reasons),
        fallback_reasons=reasons,
    )


def duration_statistics(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the LogNormal median, which is optimal for absolute error."""
    return torch.expm1(output["duration_log_mean"]).clamp(min=0.0, max=500.0)


def discount_moments(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute E[gamma**D] for log1p(D) Normal using Gauss-Hermite quadrature."""
    mean = output["duration_log_mean"]
    scale = torch.exp(output["duration_log_scale"])
    nodes = mean.new_tensor(GAUSS_HERMITE_NODES)
    weights = mean.new_tensor(GAUSS_HERMITE_WEIGHTS / math.sqrt(math.pi))
    log_duration = mean[..., None] + math.sqrt(2.0) * scale[..., None] * nodes
    durations = torch.expm1(log_duration).clamp(min=0.0, max=1000.0)
    discounts = torch.pow(durations.new_tensor(s2.GAMMA), durations)
    return (discounts * weights).sum(-1).clamp(min=0.0, max=1.0)


def transport_values(
    semantic_values: torch.Tensor,
    segment_discounts: torch.Tensor,
    gates: torch.Tensor,
    records: list[s2.BoundaryRecord],
    specs: dict[str, dict[str, object]],
) -> torch.Tensor:
    discounts = torch.zeros_like(semantic_values)
    for row, record in enumerate(records):
        chain = list(specs[record.episode.task]["chain"])
        for column, event_id in enumerate(record.event_ids):
            event = s2.MODEL_EVENTS[int(event_id)]
            current = chain.index(event)
            discounts[row, column, int(event_id)] = 1.0
            cumulative = segment_discounts.new_ones(())
            for position in range(current + 1, len(chain)):
                segment_id = s2.MODEL_EVENTS.index(chain[position - 1])
                cumulative = cumulative * segment_discounts[row, column, segment_id]
                target_id = s2.MODEL_EVENTS.index(chain[position])
                discounts[row, column, target_id] = cumulative
    return semantic_values * gates * discounts


def posterior_outputs(
    model: FactorizedEventTransport,
    records: list[s2.BoundaryRecord],
    adaptation: list[s2.BoundaryRecord],
    specs: dict[str, dict[str, object]],
    posterior: Posterior,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = pack_stage3(records, device)
    gates, _ = s2.gate_tensor(records, specs, adaptation, device, batch["inputs"].shape[1])
    if mode == "beta0" or posterior.fallback:
        betas = np.asarray([0.0], dtype=np.float32)
        weights = np.asarray([1.0], dtype=np.float64)
    elif mode == "map":
        betas = np.asarray([posterior.map_value], dtype=np.float32)
        weights = np.asarray([1.0], dtype=np.float64)
    else:
        selected = posterior.weights > 1e-5
        betas = posterior.grid[selected]
        weights = posterior.weights[selected]
        weights = weights / weights.sum()

    aggregate_values = None
    aggregate_durations = None
    semantic_values = None
    with torch.no_grad():
        for beta, weight in zip(betas, weights):
            beta_tensor = torch.full((len(records),), float(beta), device=device)
            output = model(batch["inputs"], batch["dts"], batch["mask"], beta_tensor)
            current_semantic = decode_semantic(output["successor_logits"])
            durations = duration_statistics(output)
            segment_discounts = discount_moments(output)
            values = transport_values(
                current_semantic, segment_discounts, gates, records, specs
            )
            if aggregate_values is None:
                aggregate_values = values * float(weight)
                aggregate_durations = durations * float(weight)
                semantic_values = current_semantic
            else:
                aggregate_values += values * float(weight)
                aggregate_durations += durations * float(weight)
    assert aggregate_values is not None and aggregate_durations is not None and semantic_values is not None
    return aggregate_values, aggregate_durations, semantic_values


def stratified_auc(
    records: list[s2.BoundaryRecord],
    labels: list[int],
    scores: list[float],
) -> tuple[float, float, dict[str, float]]:
    correct = 0.0
    pairs = 0
    task_values = {}
    for task in s2.TASKS:
        indices = [index for index, record in enumerate(records) if record.episode.task == task]
        task_labels = [labels[index] for index in indices]
        task_scores = [scores[index] for index in indices]
        auc = s2.rank_auc(task_labels, task_scores)
        task_values[task] = auc
        positives = [task_scores[i] for i, label in enumerate(task_labels) if label == 1]
        negatives = [task_scores[i] for i, label in enumerate(task_labels) if label == 0]
        for positive in positives:
            for negative in negatives:
                correct += float(positive > negative) + 0.5 * float(positive == negative)
                pairs += 1
    defined = [value for value in task_values.values() if math.isfinite(value)]
    macro = float(np.mean(defined)) if defined else math.nan
    micro = correct / pairs if pairs else math.nan
    return micro, macro, task_values


def grouped_auc(groups: list[tuple[list[int], list[float]]]) -> tuple[float, float, int, int]:
    correct = 0.0
    pairs = 0
    aucs = []
    for labels, scores in groups:
        auc = s2.rank_auc(labels, scores)
        if math.isfinite(auc):
            aucs.append(auc)
        positives = [score for label, score in zip(labels, scores) if label == 1]
        negatives = [score for label, score in zip(labels, scores) if label == 0]
        for positive in positives:
            for negative in negatives:
                correct += float(positive > negative) + 0.5 * float(positive == negative)
                pairs += 1
    return (
        correct / pairs if pairs else math.nan,
        float(np.mean(aucs)) if aucs else math.nan,
        pairs,
        len(aucs),
    )


def boundary_diagnostics(
    records: list[s2.BoundaryRecord],
    values: torch.Tensor,
    semantic: torch.Tensor,
    labels: np.ndarray,
) -> dict[str, object]:
    """Evaluate prefix value without treating terminal event identity as a decision gate."""
    goal = s2.MODEL_EVENTS.index("eK")

    def position_auc(position: str) -> tuple[float, float, int, int]:
        groups = []
        for task in s2.TASKS:
            task_labels = []
            task_scores = []
            for row, record in enumerate(records):
                if record.episode.task != task:
                    continue
                if position == "start":
                    column = 0
                elif position == "penultimate":
                    column = max(0, len(record.event_ids) - 2)
                else:
                    column = len(record.event_ids) - 1
                task_labels.append(int(labels[row]))
                task_scores.append(float(values[row, column, goal]))
            groups.append((task_labels, task_scores))
        return grouped_auc(groups)

    start_micro, start_macro, start_pairs, start_groups = position_auc("start")
    penultimate_micro, penultimate_macro, _, _ = position_auc("penultimate")
    terminal_micro, terminal_macro, _, _ = position_auc("terminal")

    # Compare success and failure only after matching both task and observed
    # event.  eK is excluded because reaching it directly reveals completion.
    conditional_groups = []
    for task in s2.TASKS:
        for event_id, event in enumerate(s2.MODEL_EVENTS):
            if event == "eK":
                continue
            event_labels = []
            event_scores = []
            for row, record in enumerate(records):
                if record.episode.task != task:
                    continue
                columns = np.flatnonzero(record.event_ids == event_id)
                if len(columns):
                    event_labels.append(int(labels[row]))
                    event_scores.append(float(values[row, int(columns[0]), goal]))
            conditional_groups.append((event_labels, event_scores))
    conditional_micro, conditional_macro, conditional_pairs, conditional_defined = grouped_auc(
        conditional_groups
    )

    transported_correct = 0.0
    semantic_correct = 0.0
    progress_pairs = 0
    for row, record in enumerate(records):
        if not record.episode.success:
            continue
        for earlier in range(len(record.event_ids)):
            for later in range(earlier + 1, len(record.event_ids)):
                transported_correct += float(
                    values[row, later, goal] > values[row, earlier, goal]
                ) + 0.5 * float(values[row, later, goal] == values[row, earlier, goal])
                semantic_correct += float(
                    semantic[row, later, goal] > semantic[row, earlier, goal]
                ) + 0.5 * float(semantic[row, later, goal] == semantic[row, earlier, goal])
                progress_pairs += 1

    success_last_goal = []
    failure_last_goal = []
    for record in records:
        last_is_goal = int(int(record.event_ids[-1]) == goal)
        (success_last_goal if record.episode.success else failure_last_goal).append(last_is_goal)

    return {
        "start_micro_auc": start_micro,
        "start_macro_auc": start_macro,
        "start_auc_pairs": start_pairs,
        "start_auc_defined_groups": start_groups,
        "penultimate_micro_auc_diagnostic": penultimate_micro,
        "penultimate_macro_auc_diagnostic": penultimate_macro,
        "terminal_micro_auc_leaky": terminal_micro,
        "terminal_macro_auc_leaky": terminal_macro,
        "same_event_micro_auc": conditional_micro,
        "same_event_macro_auc": conditional_macro,
        "same_event_auc_pairs": conditional_pairs,
        "same_event_auc_defined_groups": conditional_defined,
        "success_progress_pair_accuracy": (
            transported_correct / progress_pairs if progress_pairs else math.nan
        ),
        "semantic_progress_pair_accuracy": (
            semantic_correct / progress_pairs if progress_pairs else math.nan
        ),
        "event_index_progress_baseline": 1.0 if progress_pairs else math.nan,
        "progress_pairs": progress_pairs,
        "success_last_event_goal_rate": (
            float(np.mean(success_last_goal)) if success_last_goal else math.nan
        ),
        "failure_last_event_goal_rate": (
            float(np.mean(failure_last_goal)) if failure_last_goal else math.nan
        ),
        "terminal_auc_has_event_label_leakage": bool(
            success_last_goal
            and failure_last_goal
            and np.mean(success_last_goal) > np.mean(failure_last_goal)
        ),
    }


def evaluate(
    model: FactorizedEventTransport,
    records: list[s2.BoundaryRecord],
    adaptation: list[s2.BoundaryRecord],
    source_records: list[s2.BoundaryRecord],
    specs: dict[str, dict[str, object]],
    posterior: Posterior,
    device: torch.device,
    mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    batch = pack_stage3(records, device)
    values, duration_means, semantic = posterior_outputs(
        model, records, adaptation, specs, posterior, device, mode
    )
    goal = s2.MODEL_EVENTS.index("eK")
    scores = values[:, 0, goal].cpu().numpy()
    semantic_scores = semantic[:, 0, goal].cpu().numpy()
    labels = batch["success"].cpu().numpy()
    mc_targets = batch["mc_targets"]
    mask = batch["mask"]
    clock_prediction = select_event(duration_means, batch["event_ids"])
    clock_mask = batch["clock_mask"]
    micro_auc, macro_auc, task_aucs = stratified_auc(
        records, labels.tolist(), scores.tolist()
    )
    diagnostics = boundary_diagnostics(records, values, semantic, labels)

    rows = []
    for task in [*s2.TASKS, "__all__"]:
        indices = [
            index
            for index, record in enumerate(records)
            if task == "__all__" or record.episode.task == task
        ]
        selected_mask = mask[indices]
        error = torch.square(values[indices] - mc_targets[indices]).mean(-1)
        mc_mse = float(error[selected_mask].mean())
        goal_error = torch.square(values[indices, :, goal] - mc_targets[indices, :, goal])
        goal_mse = float(goal_error[selected_mask].mean())
        local_clock_mask = clock_mask[indices]
        duration_mae = (
            float(
                torch.abs(clock_prediction[indices] - batch["durations"][indices])[
                    local_clock_mask
                ].mean()
            )
            if local_clock_mask.any()
            else math.nan
        )
        task_labels = labels[indices].tolist()
        task_scores = scores[indices].tolist()
        row = {
                "mode": mode,
                "task": task,
                "embodiment": records[0].episode.body,
                "n_test": len(indices),
                "event_mc_mse": mc_mse,
                "goal_mc_mse": goal_mse,
                "pooled_auc": s2.rank_auc(task_labels, task_scores),
                "within_task_micro_auc": micro_auc if task == "__all__" else task_aucs[task],
                "within_task_macro_auc": macro_auc if task == "__all__" else task_aucs[task],
                "duration_mae": duration_mae,
                "beta_mean": posterior.mean,
                "beta_std": posterior.std,
                "beta_map": posterior.map_value,
                "posterior_fallback": posterior.fallback,
            }
        if task == "__all__":
            row.update(diagnostics)
        rows.append(row)

    episode_rows = []
    for index, record in enumerate(records):
        episode_rows.append(
            {
                "mode": mode,
                "task": record.episode.task,
                "embodiment": record.episode.body,
                "episode_index": record.episode.index,
                "success": int(record.episode.success),
                "transported_goal_score": float(scores[index]),
                "semantic_goal_score": float(semantic_scores[index]),
                "actual_goal_return": float(mc_targets[index, 0, goal]),
                "penultimate_goal_score": float(
                    values[index, max(0, len(record.event_ids) - 2), goal]
                ),
                "terminal_goal_score": float(
                    values[index, len(record.event_ids) - 1, goal]
                ),
                "last_event": s2.MODEL_EVENTS[int(record.event_ids[-1])],
            }
        )
    arrays = {
        "values": values.cpu().numpy(),
        "durations": duration_means.cpu().numpy(),
        "semantic": semantic.cpu().numpy(),
    }
    return rows, episode_rows, arrays


def self_test(device: torch.device) -> list[dict[str, object]]:
    torch.manual_seed(s2.SEED)
    input_dim = 27
    model = FactorizedEventTransport(input_dim).to(device).eval()
    inputs = torch.randn(2, 4, input_dim, device=device)
    dts = torch.ones(2, 4, device=device)
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    with torch.no_grad():
        zero = model(inputs, dts, mask, torch.zeros(2, device=device))
        one = model(inputs, dts, mask, torch.ones(2, device=device))
    rows = [
        {
            "test": "beta_absent_from_semantics",
            "passed": bool(torch.equal(zero["successor_logits"], one["successor_logits"])),
        },
        {
            "test": "beta_changes_clock_only",
            "passed": bool(
                not torch.equal(zero["duration_log_mean"], one["duration_log_mean"])
            ),
        },
        {
            "test": "fixed_semantic_support",
            "passed": bool(
                torch.equal(semantic_support(device), semantic_support(device))
                and float(semantic_support(device)[0]) == 0.0
                and float(semantic_support(device)[-1]) == 1.0
            ),
        },
    ]
    changed = inputs.clone()
    changed[:, 3] += 100.0
    with torch.no_grad():
        original = model(inputs, dts, mask, torch.zeros(2, device=device))
        perturbed = model(changed, dts, mask, torch.zeros(2, device=device))
    rows.append(
        {
            "test": "future_causality",
            "passed": bool(
                torch.equal(
                    original["successor_logits"][:, :3],
                    perturbed["successor_logits"][:, :3],
                )
                and torch.equal(
                    original["duration_log_mean"][:, :3],
                    perturbed["duration_log_mean"][:, :3],
                )
            ),
        }
    )
    durations = torch.tensor([1.0, 4.0, 5.0, 10.0], device=device)
    clean = torch.isfinite(durations) & (durations >= MIN_CLOCK_STEPS)
    rows.append(
        {
            "test": "clock_mask_excludes_short_intervals",
            "passed": bool(torch.equal(clean, torch.tensor([False, False, True, True], device=device))),
        }
    )
    target = torch.tensor([0.0, 0.5, 1.0], device=device)
    distribution = fixed_hl_gauss(target)
    rows.append(
        {
            "test": "hl_gauss_normalized",
            "passed": bool(
                torch.allclose(distribution.sum(-1), torch.ones(3, device=device))
            ),
        }
    )
    slow_output = {
        "duration_log_mean": torch.tensor([[[1.0], [2.0]]], device=device),
        "duration_log_scale": torch.full((1, 2, 1), -2.0, device=device),
    }
    moments = discount_moments(slow_output)
    rows.append(
        {
            "test": "discount_moment_decreases_with_duration",
            "passed": bool(moments[0, 1, 0] < moments[0, 0, 0]),
        }
    )
    micro, macro, pairs, groups = grouped_auc([([0, 1], [0.0, 1.0])])
    rows.append(
        {
            "test": "grouped_auc_pair_accounting",
            "passed": bool(micro == 1.0 and macro == 1.0 and pairs == 1 and groups == 1),
        }
    )
    if not all(row["passed"] for row in rows):
        raise RuntimeError(f"Stage-3 self test failed: {rows}")
    return rows


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not math.isfinite(float(value)) else f"{float(value):.6f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def load_records(
    data_root: Path,
) -> tuple[
    list[s2.BoundaryRecord],
    dict[str, dict[str, object]],
    np.ndarray,
    np.ndarray,
]:
    episodes = s2.load_episodes(data_root)
    calibration = s2.calibrate_events(episodes)
    s2.assign_raw_events(episodes, calibration)
    specs = s2.derive_chains(episodes)
    s2.apply_chains(episodes, specs)
    records, mean, std = s2.build_records(episodes, calibration, specs)
    return records, specs, mean, std


def calculate_development_checks(
    comparison: pd.DataFrame,
    source_failures: int,
) -> tuple[dict[str, dict[str, object]], bool, str, bool | None]:
    development_checks = {}
    for body in s2.TARGET_BODIES:
        body_rows = comparison[
            (comparison.task == "__all__") & (comparison.embodiment == body)
        ]
        pivot = body_rows.pivot(index="seed", columns="mode")
        event_delta = (
            pivot["event_mc_mse"]["posterior_predictive"]
            - pivot["event_mc_mse"]["beta0"]
        )
        duration_delta = (
            pivot["duration_mae"]["posterior_predictive"]
            - pivot["duration_mae"]["beta0"]
        )
        start_macro_auc = float(pivot["start_macro_auc"]["posterior_predictive"].mean())
        conditional_macro_auc = float(
            pivot["same_event_macro_auc"]["posterior_predictive"].mean()
        )
        conditional_pairs = int(
            pivot["same_event_auc_pairs"]["posterior_predictive"].min()
        )
        development_checks[body] = {
            "event_mc_better_all_seeds": bool((event_delta < 0.0).all()),
            "duration_mae_better_all_seeds": bool((duration_delta < 0.0).all()),
            "start_macro_auc_diagnostic": start_macro_auc,
            "same_event_macro_auc": conditional_macro_auc,
            "same_event_pairs": conditional_pairs,
            "same_event_pairs_sufficient": bool(conditional_pairs >= MIN_DECISION_PAIRS),
            "success_progress_pair_accuracy": float(
                pivot["success_progress_pair_accuracy"]["posterior_predictive"].mean()
            ),
            "event_index_progress_baseline": 1.0,
            "terminal_auc_excluded_for_event_label_leakage": bool(
                pivot["terminal_auc_has_event_label_leakage"]["posterior_predictive"].all()
            ),
        }
    mechanism_passed = all(
        checks["event_mc_better_all_seeds"]
        and checks["duration_mae_better_all_seeds"]
        for checks in development_checks.values()
    )
    if source_failures == 0:
        decision_status = "inconclusive_missing_source_failure_supervision"
        decision_passed = None
    elif not all(
        checks["same_event_pairs_sufficient"] for checks in development_checks.values()
    ):
        decision_status = "inconclusive_insufficient_matched_event_pairs"
        decision_passed = None
    elif all(checks["same_event_macro_auc"] > 0.5 for checks in development_checks.values()):
        decision_status = "exploratory_pass_requires_fresh_confirmatory_data"
        decision_passed = None
    else:
        decision_status = "exploratory_failure_branch_discrimination_not_demonstrated"
        decision_passed = False
    return development_checks, mechanism_passed, decision_status, decision_passed


def refresh_development_gate(output_root: Path) -> dict[str, object]:
    comparison = pd.read_csv(output_root / "results/stage3_comparison.csv")
    summary_path = output_root / "stage3_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks, mechanism_passed, decision_status, decision_passed = calculate_development_checks(
        comparison, int(summary["source_failures"])
    )
    gate = {
        "development_checks": checks,
        "mechanism_gate_passed": mechanism_passed,
        "decision_gate_passed": decision_passed,
        "decision_gate_status": decision_status,
        "stop_before_critic_integration": decision_passed is not True,
    }
    (output_root / "development_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    summary.update(gate)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = output_root / "stage3_report.md"
    report = report_path.read_text(encoding="utf-8")
    if "## Development gates" not in report:
        report += f"""

## Development gates

- Mechanism gate passed: **{mechanism_passed}**
- Decision gate passed: **{decision_passed}**
- Decision status: **{decision_status}**
- Stop before action-conditioned critic integration: **{decision_passed is not True}**
"""
        report_path.write_text(report, encoding="utf-8")
    return gate


def run_main(
    data_root: Path,
    output_root: Path,
    steps: int,
    adaptation_count: int,
    seeds: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tests = self_test(device)
    pd.DataFrame(tests).to_csv(output_root / "results/self_test.csv", index=False)

    records, specs, mean, std = load_records(data_root)
    source_train = [
        record
        for record in records
        if record.episode.body in s2.SOURCE_BODIES and record.episode.index < 40
    ]
    source_validation = [
        record
        for record in records
        if record.episode.body in s2.SOURCE_BODIES and record.episode.index >= 40
    ]
    source_successes = sum(record.episode.success for record in source_train)
    source_failures = len(source_train) - source_successes
    clock_audit = []
    for split, selected in [("train", source_train), ("validation", source_validation)]:
        packed = pack_stage3(selected, device)
        finite = int(packed["duration_mask"].sum())
        clean = int(packed["clock_mask"].sum())
        clock_audit.append(
            {
                "split": split,
                "finite_intervals": finite,
                "clean_intervals": clean,
                "excluded_short_intervals": finite - clean,
                "excluded_fraction": (finite - clean) / max(finite, 1),
                "right_censored_intervals": int(packed["censor_mask"].sum()),
            }
        )
    pd.DataFrame(clock_audit).to_csv(
        output_root / "results/clock_mask_audit.csv", index=False
    )

    training_rows = []
    comparison_rows = []
    episode_rows = []
    posterior_rows = []
    model_states = {}
    posterior_summary: dict[str, list[dict[str, object]]] = {body: [] for body in s2.TARGET_BODIES}

    for seed_index in range(seeds):
        seed = s2.SEED + 100 * seed_index
        model, history = train_model(
            source_train, source_validation, device, steps, seed
        )
        for row in history:
            training_rows.append({"seed": seed, **row})
        model_states[str(seed)] = copy.deepcopy(model.state_dict())

        for body in s2.TARGET_BODIES:
            adaptation = [
                record
                for record in records
                if record.episode.body == body and record.episode.index < adaptation_count
            ]
            test = [
                record
                for record in records
                if record.episode.body == body and record.episode.index >= 15
            ]
            posterior = infer_beta_posterior(model, adaptation, device)
            posterior_summary[body].append(
                {
                    "seed": seed,
                    "mean": posterior.mean,
                    "std": posterior.std,
                    "map": posterior.map_value,
                    "edge_mass": posterior.edge_mass,
                    "n_observations": posterior.n_observations,
                    "fallback": posterior.fallback,
                    "fallback_reasons": posterior.fallback_reasons,
                }
            )
            for beta, weight in zip(posterior.grid, posterior.weights):
                posterior_rows.append(
                    {
                        "seed": seed,
                        "embodiment": body,
                        "beta": float(beta),
                        "weight": float(weight),
                    }
                )
            for mode in ["beta0", "map", "posterior_predictive"]:
                rows, predictions, _ = evaluate(
                    model,
                    test,
                    adaptation,
                    source_train,
                    specs,
                    posterior,
                    device,
                    mode,
                )
                comparison_rows.extend({"seed": seed, **row} for row in rows)
                episode_rows.extend({"seed": seed, **row} for row in predictions)

    pd.DataFrame(training_rows).to_csv(
        output_root / "results/training_history.csv", index=False
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_root / "results/stage3_comparison.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(
        output_root / "results/per_episode_predictions.csv", index=False
    )
    pd.DataFrame(posterior_rows).to_csv(
        output_root / "results/beta_posterior.csv", index=False
    )
    np.savez(output_root / "feature_normalization.npz", mean=mean, std=std)
    torch.save(
        {
            "models": model_states,
            "input_dim": source_train[0].inputs.shape[1],
            "semantic_bins": SEMANTIC_BINS,
            "min_clock_steps": MIN_CLOCK_STEPS,
        },
        output_root / "stage3_models.pt",
    )

    aggregate = comparison[comparison.task == "__all__"].groupby(
        ["embodiment", "mode"], as_index=False
    ).agg(
        event_mc_mse_mean=("event_mc_mse", "mean"),
        event_mc_mse_std=("event_mc_mse", "std"),
        goal_mc_mse_mean=("goal_mc_mse", "mean"),
        start_micro_auc_diagnostic_mean=("start_micro_auc", "mean"),
        start_macro_auc_diagnostic_mean=("start_macro_auc", "mean"),
        same_event_micro_auc_mean=("same_event_micro_auc", "mean"),
        same_event_macro_auc_mean=("same_event_macro_auc", "mean"),
        same_event_auc_pairs_min=("same_event_auc_pairs", "min"),
        success_progress_pair_accuracy_mean=("success_progress_pair_accuracy", "mean"),
        event_index_progress_baseline=("event_index_progress_baseline", "mean"),
        terminal_macro_auc_leaky_mean=("terminal_macro_auc_leaky", "mean"),
        duration_mae_mean=("duration_mae", "mean"),
    )
    aggregate.to_csv(output_root / "results/stage3_aggregate.csv", index=False)
    development_checks, mechanism_passed, decision_status, decision_passed = (
        calculate_development_checks(comparison, int(source_failures))
    )
    summary = {
        "status": "exploratory_development_only",
        "architecture": "shared_semantic_encoder_plus_isolated_clock_liquid_head",
        "steps": steps,
        "seeds": seeds,
        "adaptation_count_per_task": adaptation_count,
        "adaptation_rollouts_per_embodiment": adaptation_count * len(s2.TASKS),
        "source_train_episodes": len(source_train),
        "source_validation_episodes": len(source_validation),
        "source_successes": int(source_successes),
        "source_failures": int(source_failures),
        "ranking_supervision_available": bool(source_failures),
        "clock_mask_audit": clock_audit,
        "posterior": posterior_summary,
        "aggregate": aggregate.to_dict(orient="records"),
        "development_checks": development_checks,
        "mechanism_gate_passed": mechanism_passed,
        "decision_gate_passed": decision_passed,
        "decision_gate_status": decision_status,
        "stop_before_critic_integration": decision_passed is not True,
        "limitations": [
            "Piper and UR5-WSG are development sets already inspected in Stage 2.",
            "The source dataset has no failure trajectories, so ranking loss has zero usable pairs.",
            "Initial-boundary AUC is diagnostic only because one rollout outcome is a noisy label for expected state value.",
            "Terminal-boundary AUC is excluded from gating because eK event identity leaks completion.",
            "This is an event-value transport head, not yet an action-conditioned RoboTwin critic.",
        ],
    }
    (output_root / "stage3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    table = dataframe_markdown(aggregate)
    report = f"""# ETSF Stage 3 · Factorized Semantic/Clock Transport

## Status

This run is **exploratory development evidence**. Piper and UR5-WSG have already
been inspected and cannot serve as a new confirmatory holdout.

## Architecture

- Shared embodiment-invariant semantic encoder and fixed-support successor head.
- Isolated liquid clock head; beta cannot change semantic hidden states or logits.
- Clean clock mask requires duration >= {MIN_CLOCK_STEPS:.0f} steps.
- Target adaptation uses a one-dimensional beta posterior and event-reachability posteriors.
- Final discounted event values are produced by an explicit semi-Markov transport operator.

## Data limitation

- Source train successes: {source_successes}
- Source train failures: {source_failures}
- Usable source ranking supervision: {bool(source_failures)}
- N={adaptation_count} means {adaptation_count} rollouts per task, or {adaptation_count * len(s2.TASKS)} per target embodiment.

## Development results

{table}

## Development gates

- Mechanism gate passed: **{mechanism_passed}**
- Decision gate passed: **{decision_passed}**
- Decision status: **{decision_status}**
- Stop before action-conditioned critic integration: **{decision_passed is not True}**

## Interpretation boundary

The primary correctness metric is held-out event Monte Carlo return error. Bellman
self-consistency is not used as a primary gate. Initial-boundary AUC is diagnostic,
terminal AUC is excluded for event-label leakage, and failure-branch discrimination
matches task and event. A new embodiment or sealed fresh rollouts are required for
confirmatory claims, and source failures are required before claiming decision transfer.
"""
    (output_root / "stage3_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["self-test", "main", "gate"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/home/user/etsf_stage1"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/user/etsf_stage3"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--adaptation-count", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.stage == "self-test":
        rows = self_test(device)
        print("SELF_TEST_PASS=" + ",".join(row["test"] for row in rows), flush=True)
        return
    if args.stage == "gate":
        gate = refresh_development_gate(args.output_root)
        print("DEVELOPMENT_GATE=" + json.dumps(gate, sort_keys=True), flush=True)
        return
    summary = run_main(
        args.data_root,
        args.output_root,
        args.steps,
        args.adaptation_count,
        args.seeds,
    )
    print("STAGE3_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
