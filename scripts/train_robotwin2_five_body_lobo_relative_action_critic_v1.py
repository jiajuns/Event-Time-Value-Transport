#!/usr/bin/env python3
"""Train a matched VLA-ATTC-style Relative Action Critic on RoboTwin branches.

This is an independent baseline for the five-body LOBO study.  It reuses the
frozen primary/supplement bindings, source-only split, canonical state/action
pool and 5-member budget from the shared event head, but it does not import or
modify that model's parameters.  The held-out body remains manifest-visible
only: no held-out payload is opened before every checkpoint is frozen.

The official RAC representation independently embeds ``a_i``, ``a_j``,
``a_i-a_j`` and state/context before a lightweight Transformer and trains a
binary preference logit with focal loss.  RoboTwin supplies stronger real
branch outcomes than RAC's generated good/bad proxy, so preference labels are
the preregistered lexicographic order success -> terminal stage -> terminal
goal progress.  Exact/tolerance ties are excluded rather than assigned an
arbitrary label.  Every informative unordered pair is emitted in both orders.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_multibody_canonical_event_world_model as core
import train_robotwin2_five_body_lobo_shared_event_head_v1 as shared


FORMAT = "etsf_robotwin2_five_body_lobo_relative_action_critic_v1"
SUMMARY_FORMAT = "etsf_robotwin2_five_body_lobo_relative_action_critic_summary_v1"
MODEL_FAMILY = "vla_attc_matched_relative_action_critic_v1"
OFFICIAL_REFERENCE = {
    "paper": "VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model",
    "arxiv": "2605.01194v2",
    "input_equations": [10, 11, 12],
    "training_loss": "binary_focal_loss",
    "official_inference": "single_elimination_tournament",
    "matched_inference": "all_unordered_pairs_soft_copeland_no_bracket",
}
BODIES = shared.BODIES
CONDITIONS = shared.CONDITIONS
CANDIDATE_COUNT = shared.CANDIDATE_COUNT
ENSEMBLE_SIZE = 5
DEFAULT_STEPS = 3000
DEFAULT_EVAL_EVERY = 100
DEFAULT_ENSEMBLE_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
FOCAL_GAMMA = 2.0
GOAL_EQUALITY_TOLERANCE = shared.DENSE_RANK_LABEL_EQUALITY_TOLERANCE
SUPPLEMENT_PAIR_LOSS_WEIGHT = shared.SUPPLEMENT_RANK_LOSS_WEIGHT
CROSS_BODY_STANDARDIZED_INPUT_CLIP = shared.CROSS_BODY_STANDARDIZED_INPUT_CLIP
PAIR_TIERS = ("success", "stage", "goal")
SUPPORTED_RUNTIME_CANDIDATE_COUNTS = (4, 8)


class RelativeActionCriticError(RuntimeError):
    """Raised when the matched RAC data/model protocol is violated."""


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def rac_contract() -> dict[str, Any]:
    return {
        "format": "etsf_matched_relative_action_critic_contract_v1",
        "official_reference": dict(OFFICIAL_REFERENCE),
        "inputs": [
            "independent_temporal_encoder_a_i",
            "independent_temporal_encoder_a_j",
            "independent_temporal_encoder_a_i_minus_a_j",
            "canonical_state_27d",
            "event_dt_age_remaining_budget_context",
        ],
        "body_or_condition_identity_input": False,
        "action_schema": shared.CANONICAL_ACTION_SCHEMA,
        "state_schema": shared.CANONICAL_STATE_SCHEMA,
        "preference": "strict_lexicographic_success_then_stage_then_goal",
        "goal_equality_tolerance": GOAL_EQUALITY_TOLERANCE,
        "tie_policy": "exclude_unordered_pair_no_synthetic_label",
        "orientation_augmentation": "both_orders_inverse_labels",
        "objective": "symmetric_binary_focal_bce_with_logits",
        "focal_gamma": FOCAL_GAMMA,
        "gamma_zero_ablation": "exact_weighted_binary_cross_entropy",
        "inference": "all_unordered_pairs_soft_copeland_mean_win_probability",
        "bracket_or_candidate_order_dependence": False,
        "runtime_candidate_counts": list(SUPPORTED_RUNTIME_CANDIDATE_COUNTS),
        "ensemble_members": ENSEMBLE_SIZE,
        "steps_per_member_default": DEFAULT_STEPS,
        "heldout_payload_access_before_checkpoint_freeze": False,
    }


@dataclasses.dataclass(frozen=True)
class RACConfig:
    action_dim: int = core.ACTION_DIM
    state_dim: int = core.STATE_DIM
    model_dim: int = 128
    transformer_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.1
    normalization_clip: float = CROSS_BODY_STANDARDIZED_INPUT_CLIP


class MaskedTemporalActionEncoder(nn.Module):
    """One dedicated MLP+GRU action-chunk stem."""

    def __init__(self, action_dim: int, model_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.input = nn.Sequential(
            nn.Linear(action_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )
        self.cell = nn.GRUCell(model_dim, model_dim)
        self.output = nn.LayerNorm(model_dim)

    def forward(self, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise RelativeActionCriticError("RAC action chunk shape is invalid")
        if mask.shape != actions.shape[:2] or mask.dtype != torch.bool:
            raise RelativeActionCriticError("RAC action mask is invalid")
        if not bool(mask.any(dim=1).all()):
            raise RelativeActionCriticError("RAC action chunk has an empty prefix")
        hidden = actions.new_zeros(actions.shape[0], self.model_dim)
        projected = self.input(actions)
        for step in range(actions.shape[1]):
            proposal = self.cell(projected[:, step], hidden)
            hidden = torch.where(mask[:, step, None], proposal, hidden)
        return self.output(hidden)


class MatchedRelativeActionCritic(nn.Module):
    """Lightweight RAC with official four-way action/state representation.

    The ordered score is antisymmetrized at the logit level.  At evaluation
    this guarantees ``logit(a_i,a_j) == -logit(a_j,a_i)`` and removes a
    remaining positional shortcut beyond the official inverse-order
    augmentation.
    """

    def __init__(self, config: RACConfig | None = None) -> None:
        super().__init__()
        self.config = config or RACConfig()
        if (
            self.config.model_dim <= 0
            or self.config.attention_heads <= 0
            or self.config.model_dim % self.config.attention_heads != 0
            or self.config.transformer_layers <= 0
            or not 0.0 <= self.config.dropout < 1.0
        ):
            raise RelativeActionCriticError("RAC model configuration is invalid")
        d = self.config.model_dim
        # These modules are intentionally not shared: matching Eq. 10/11.
        self.action_i_encoder = MaskedTemporalActionEncoder(core.ACTION_DIM, d)
        self.action_j_encoder = MaskedTemporalActionEncoder(core.ACTION_DIM, d)
        self.action_difference_encoder = MaskedTemporalActionEncoder(
            core.ACTION_DIM, d
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(core.STATE_DIM, d), nn.GELU(), nn.Linear(d, d)
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(4, d), nn.GELU(), nn.Linear(d, d)
        )
        self.token_type = nn.Parameter(torch.zeros(5, d))
        nn.init.normal_(self.token_type, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.attention_heads,
            dim_feedforward=4 * d,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(d),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d, 1),
        )
        self.register_buffer("action_mean", torch.zeros(core.ACTION_DIM))
        self.register_buffer("action_std", torch.ones(core.ACTION_DIM))
        self.register_buffer("state_mean", torch.zeros(core.STATE_DIM))
        self.register_buffer("state_std", torch.ones(core.STATE_DIM))

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        expected_action = (core.ACTION_DIM,)
        expected_state = (core.STATE_DIM,)
        if (
            action_mean.shape != expected_action
            or action_std.shape != expected_action
            or state_mean.shape != expected_state
            or state_std.shape != expected_state
            or not bool(torch.isfinite(action_mean).all())
            or not bool(torch.isfinite(action_std).all())
            or not bool(torch.isfinite(state_mean).all())
            or not bool(torch.isfinite(state_std).all())
            or bool((action_std < 1e-4).any())
            or bool((state_std < 1e-4).any())
        ):
            raise RelativeActionCriticError("RAC normalization is invalid")
        self.action_mean.copy_(action_mean.to(self.action_mean))
        self.action_std.copy_(action_std.to(self.action_std))
        self.state_mean.copy_(state_mean.to(self.state_mean))
        self.state_std.copy_(state_std.to(self.state_std))

    def _validate_pair_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        required = {
            "state",
            "action_i",
            "action_j",
            "action_i_mask",
            "action_j_mask",
            "dt",
            "current_event_id",
            "event_age_seconds",
            "remaining_action_budget",
        }
        if not required <= set(batch):
            raise RelativeActionCriticError(
                f"RAC pair batch lacks {sorted(required-set(batch))}"
            )
        state = batch["state"]
        action_i = batch["action_i"]
        action_j = batch["action_j"]
        size = state.shape[0] if state.ndim == 2 else -1
        if (
            state.shape != (size, core.STATE_DIM)
            or action_i.ndim != 3
            or action_j.shape != action_i.shape
            or action_i.shape[0] != size
            or action_i.shape[-1] != core.ACTION_DIM
            or batch["action_i_mask"].shape != action_i.shape[:2]
            or batch["action_j_mask"].shape != action_i.shape[:2]
            or batch["action_i_mask"].dtype != torch.bool
            or batch["action_j_mask"].dtype != torch.bool
        ):
            raise RelativeActionCriticError("RAC pair tensors have invalid shapes")
        for name in (
            "dt", "current_event_id", "event_age_seconds", "remaining_action_budget"
        ):
            if batch[name].shape != (size,):
                raise RelativeActionCriticError(f"RAC {name} shape is invalid")
        floating = (state, action_i, action_j, batch["dt"],
                    batch["event_age_seconds"], batch["remaining_action_budget"])
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise RelativeActionCriticError("RAC pair batch contains non-finite values")
        if (
            bool((batch["dt"] <= 0.0).any())
            or bool((batch["event_age_seconds"] < 0.0).any())
            or bool((batch["remaining_action_budget"] <= 0.0).any())
            or bool((batch["current_event_id"] < 0).any())
            or bool((batch["current_event_id"] >= len(core.CANONICAL_EVENTS)).any())
        ):
            raise RelativeActionCriticError("RAC causal context is outside its support")

    def _ordered_logit(
        self,
        *,
        action_i: torch.Tensor,
        action_j: torch.Tensor,
        action_i_mask: torch.Tensor,
        action_j_mask: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        clip = float(self.config.normalization_clip)
        normalized_i = ((action_i - self.action_mean) / self.action_std).clamp(
            -clip, clip
        )
        normalized_j = ((action_j - self.action_mean) / self.action_std).clamp(
            -clip, clip
        )
        normalized_difference = ((action_i - action_j) / self.action_std).clamp(
            -clip, clip
        )
        difference_mask = action_i_mask & action_j_mask
        if not bool(difference_mask.any(dim=1).all()):
            raise RelativeActionCriticError("RAC actions have no overlapping prefix")
        normalized_state = ((state - self.state_mean) / self.state_std).clamp(
            -clip, clip
        )
        tokens = torch.stack(
            (
                self.action_i_encoder(normalized_i, action_i_mask),
                self.action_j_encoder(normalized_j, action_j_mask),
                self.action_difference_encoder(
                    normalized_difference, difference_mask
                ),
                self.state_encoder(normalized_state),
                self.context_encoder(context),
            ),
            dim=1,
        )
        hidden = self.transformer(tokens + self.token_type[None])
        return self.classifier(hidden.mean(dim=1)).squeeze(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self._validate_pair_batch(batch)
        dt = batch["dt"].float()
        event_age = batch["event_age_seconds"].float()
        remaining = batch["remaining_action_budget"].float()
        context = torch.stack(
            (
                (dt * shared.SOURCE_EVENT_SAMPLING_HZ).clamp(0.0, 100.0),
                torch.log1p(event_age).clamp(0.0, 10.0) / 10.0,
                batch["current_event_id"].float()
                / float(len(core.CANONICAL_EVENTS) - 1),
                torch.log1p(remaining).clamp(0.0, 10.0) / 10.0,
            ),
            dim=-1,
        )
        forward = self._ordered_logit(
            action_i=batch["action_i"],
            action_j=batch["action_j"],
            action_i_mask=batch["action_i_mask"],
            action_j_mask=batch["action_j_mask"],
            state=batch["state"],
            context=context,
        )
        reverse = self._ordered_logit(
            action_i=batch["action_j"],
            action_j=batch["action_i"],
            action_i_mask=batch["action_j_mask"],
            action_j_mask=batch["action_i_mask"],
            state=batch["state"],
            context=context,
        )
        return 0.5 * (forward - reverse)


@dataclasses.dataclass(frozen=True)
class PreferencePair:
    logical_group: str
    body: str
    left_candidate: int
    right_candidate: int
    label: float
    tier: str
    left: Mapping[str, Any]
    right: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class PreferencePairCollection:
    examples: tuple[PreferencePair, ...]
    group_pair_counts: Mapping[str, int]
    best_candidates: Mapping[str, tuple[int, ...]]
    audit: Mapping[str, Any]


def _binary_scalar(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (bool, np.bool_)):
        value = float(value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise RelativeActionCriticError(f"RAC {key} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result not in (0.0, 1.0):
        raise RelativeActionCriticError(f"RAC {key} is not strictly binary")
    return result


def _finite_scalar(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise RelativeActionCriticError(f"RAC {key} is not finite numeric")
    return float(value)


def _preference_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    if (
        _binary_scalar(row, "success_mask") != 1.0
        or _binary_scalar(row, "terminal_event_mask") != 1.0
        or _binary_scalar(row, "terminal_goal_progress_mask") != 1.0
    ):
        raise RelativeActionCriticError(
            "RAC lexicographic preference lacks complete real branch supervision"
        )
    success = _binary_scalar(row, "success")
    stage = _finite_scalar(row, "terminal_max_event_id")
    goal = _finite_scalar(row, "terminal_goal_progress")
    if stage != float(int(stage)) or not 0 <= stage < len(core.CANONICAL_EVENTS):
        raise RelativeActionCriticError("RAC terminal stage is invalid")
    return success, stage, goal


def lexicographic_preference(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[int | None, str | None]:
    """Return 1/0 for left/right preference, or None for a real tie."""

    left_key = _preference_key(left)
    right_key = _preference_key(right)
    if left_key[0] != right_key[0]:
        return int(left_key[0] > right_key[0]), "success"
    if left_key[1] != right_key[1]:
        return int(left_key[1] > right_key[1]), "stage"
    difference = left_key[2] - right_key[2]
    if abs(difference) > GOAL_EQUALITY_TOLERANCE:
        return int(difference > 0.0), "goal"
    return None, None


def _validate_complete_decision(
    members: Sequence[Mapping[str, Any]],
    *,
    logical_group: str,
    allowed_source_bodies: set[str],
) -> list[Mapping[str, Any]]:
    if len(members) != CANDIDATE_COUNT:
        raise RelativeActionCriticError(
            f"RAC decision {logical_group} is not a complete candidate set"
        )
    ordered = sorted(members, key=lambda row: int(row.get("candidate_index", -1)))
    if [int(row.get("candidate_index", -1)) for row in ordered] != list(
        range(CANDIDATE_COUNT)
    ):
        raise RelativeActionCriticError(
            f"RAC decision {logical_group} candidate identities changed"
        )
    bodies = {str(row.get("body", "")) for row in ordered}
    if len(bodies) != 1 or not bodies <= allowed_source_bodies:
        raise RelativeActionCriticError(
            f"RAC decision {logical_group} reached a non-source body"
        )
    for row in ordered:
        if _binary_scalar(row, "action_available") != 1.0:
            raise RelativeActionCriticError("RAC cannot compare an unavailable action")
        schema = row.get("action_schema_id")
        event = row.get("current_event_id")
        if (
            isinstance(schema, (bool, np.bool_))
            or not isinstance(schema, (int, np.integer))
            or int(schema) != 0
            or isinstance(event, (bool, np.bool_))
            or not isinstance(event, (int, np.integer))
            or not 0 <= int(event) < len(core.CANONICAL_EVENTS)
        ):
            raise RelativeActionCriticError(
                "RAC decision is outside canonical schema/event support"
            )
        actions = np.asarray(row.get("actions"))
        mask = np.asarray(row.get("action_mask"))
        state = np.asarray(row.get("state"))
        if (
            actions.ndim != 2
            or actions.shape[-1] != core.ACTION_DIM
            or mask.shape != actions.shape[:1]
            or state.shape != (core.STATE_DIM,)
            or not np.isfinite(actions).all()
            or not np.isfinite(state).all()
            or not np.all(np.isin(mask, [0, 1, False, True]))
            or not bool(mask.astype(bool).any())
        ):
            raise RelativeActionCriticError("RAC decision contains invalid canonical input")
        event_onehot = np.zeros(len(core.CANONICAL_EVENTS), dtype=np.float32)
        event_onehot[int(event)] = 1.0
        if not np.array_equal(state[18:23].astype(np.float32), event_onehot):
            raise RelativeActionCriticError(
                "RAC state event onehot disagrees with current event"
            )
        dt = _finite_scalar(row, "dt")
        event_age = _finite_scalar(row, "event_age_seconds")
        remaining = _finite_scalar(row, "remaining_action_budget")
        if dt <= 0.0 or event_age < 0.0 or remaining <= 0.0:
            raise RelativeActionCriticError("RAC causal context is outside support")
        _preference_key(row)
    reference = ordered[0]
    for row in ordered[1:]:
        if (
            not np.array_equal(np.asarray(row["state"]), np.asarray(reference["state"]))
            or int(row["current_event_id"]) != int(reference["current_event_id"])
            or not math.isclose(float(row["dt"]), float(reference["dt"]), rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(
                float(row["event_age_seconds"]),
                float(reference["event_age_seconds"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(row["remaining_action_budget"]),
                float(reference["remaining_action_budget"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise RelativeActionCriticError(
                "RAC candidates do not share one pre-action state/context"
            )
    return ordered


def build_preference_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_bodies: Sequence[str],
    stream: str,
) -> PreferencePairCollection:
    """Build symmetric ordered RAC pairs from complete real branch decisions."""

    declared_sources = tuple(str(body) for body in source_bodies)
    allowed = set(declared_sources)
    if (
        not rows
        or not allowed
        or any(body not in BODIES for body in allowed)
        or len(allowed) != len(declared_sources)
        or not isinstance(stream, str)
        or not stream
    ):
        raise RelativeActionCriticError("RAC source pair contract is invalid")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get("logical_group")
        if not isinstance(group, str) or not group:
            raise RelativeActionCriticError("RAC row lacks a logical group")
        grouped[group].append(row)
    examples: list[PreferencePair] = []
    group_pair_counts: dict[str, int] = {}
    best_candidates: dict[str, tuple[int, ...]] = {}
    tier_counts = {tier: 0 for tier in PAIR_TIERS}
    tied_unordered = 0
    observed_bodies: set[str] = set()
    identity: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items()):
        ordered = _validate_complete_decision(
            members, logical_group=group, allowed_source_bodies=allowed
        )
        body = str(ordered[0]["body"])
        observed_bodies.add(body)
        keys = [_preference_key(row) for row in ordered]
        maximum_success = max(key[0] for key in keys)
        candidates = [index for index, key in enumerate(keys) if key[0] == maximum_success]
        maximum_stage = max(keys[index][1] for index in candidates)
        candidates = [index for index in candidates if keys[index][1] == maximum_stage]
        maximum_goal = max(keys[index][2] for index in candidates)
        best = tuple(
            index
            for index in candidates
            if maximum_goal - keys[index][2] <= GOAL_EQUALITY_TOLERANCE
        )
        if not best:
            raise RelativeActionCriticError("RAC decision has no lexicographic best")
        best_candidates[group] = best
        before = len(examples)
        for left_index in range(CANDIDATE_COUNT):
            for right_index in range(left_index + 1, CANDIDATE_COUNT):
                label, tier = lexicographic_preference(
                    ordered[left_index], ordered[right_index]
                )
                if label is None:
                    tied_unordered += 1
                    continue
                assert tier is not None
                tier_counts[tier] += 1
                examples.extend(
                    (
                        PreferencePair(
                            group, body, left_index, right_index, float(label), tier,
                            ordered[left_index], ordered[right_index],
                        ),
                        PreferencePair(
                            group, body, right_index, left_index, float(1-label), tier,
                            ordered[right_index], ordered[left_index],
                        ),
                    )
                )
                identity.append(
                    {
                        "logical_group": group,
                        "unordered_candidates": [left_index, right_index],
                        "preferred_candidate": left_index if label else right_index,
                        "tier": tier,
                    }
                )
        count = len(examples) - before
        if count:
            group_pair_counts[group] = count
    if observed_bodies != allowed:
        raise RelativeActionCriticError(
            "RAC pair stream does not contain exactly the declared source bodies"
        )
    if not examples or not group_pair_counts:
        raise RelativeActionCriticError("RAC pair stream has no real preference")
    positive = sum(pair.label == 1.0 for pair in examples)
    negative = len(examples) - positive
    if positive != negative:
        raise RelativeActionCriticError("RAC symmetric pair labels lost class balance")
    audit = {
        "format": "etsf_rac_real_branch_preference_pairs_v1",
        "stream": stream,
        "source_bodies": sorted(allowed),
        "decision_groups_total": len(grouped),
        "decision_groups_with_real_preference": len(group_pair_counts),
        "decision_groups_all_tied": len(grouped) - len(group_pair_counts),
        "candidate_rows": len(rows),
        "possible_unordered_pairs": len(grouped) * math.comb(CANDIDATE_COUNT, 2),
        "labeled_unordered_pairs": len(examples) // 2,
        "tied_unordered_pairs_excluded": tied_unordered,
        "ordered_training_pairs": len(examples),
        "positive_ordered_pairs": positive,
        "negative_ordered_pairs": negative,
        "unordered_pair_tier_counts": tier_counts,
        "preference_identity_sha256": canonical_sha256(identity),
        "synthetic_or_tie_labels": 0,
        "heldout_rows_used": 0,
    }
    return PreferencePairCollection(
        tuple(examples), group_pair_counts, best_candidates, audit
    )


class PreferencePairDataset(Dataset[PreferencePair]):
    def __init__(self, collection: PreferencePairCollection) -> None:
        self.collection = collection

    def __len__(self) -> int:
        return len(self.collection.examples)

    def __getitem__(self, index: int) -> PreferencePair:
        return self.collection.examples[index]


class CompletePreferenceGroupBatchSampler:
    """Shuffle decisions while keeping all informative ordered pairs together."""

    def __init__(
        self,
        collection: PreferencePairCollection,
        *,
        batch_size_pairs: int,
        seed: int,
    ) -> None:
        if batch_size_pairs < 2 * math.comb(CANDIDATE_COUNT, 2):
            raise RelativeActionCriticError(
                "RAC pair batch must fit every ordered pair of one decision"
            )
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(collection.examples):
            grouped[example.logical_group].append(index)
        if set(grouped) != set(collection.group_pair_counts):
            raise RelativeActionCriticError("RAC batch groups changed")
        self.groups = [(group, tuple(grouped[group])) for group in sorted(grouped)]
        self.batch_size_pairs = int(batch_size_pairs)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = list(self.groups)
        generator.shuffle(groups)
        batch: list[int] = []
        for _group, indices in groups:
            if batch and len(batch) + len(indices) > self.batch_size_pairs:
                yield batch
                batch = []
            batch.extend(indices)
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(len(indices) for _group, indices in self.groups)
        return max(1, math.ceil(total / self.batch_size_pairs))


def collate_preference_pairs(examples: Sequence[PreferencePair]) -> dict[str, Any]:
    if not examples:
        raise RelativeActionCriticError("cannot collate an empty RAC pair batch")
    horizon = max(
        max(np.asarray(item.left["actions"]).shape[0],
            np.asarray(item.right["actions"]).shape[0])
        for item in examples
    )
    action_i = np.zeros((len(examples), horizon, core.ACTION_DIM), dtype=np.float32)
    action_j = np.zeros_like(action_i)
    mask_i = np.zeros((len(examples), horizon), dtype=bool)
    mask_j = np.zeros_like(mask_i)
    for index, example in enumerate(examples):
        left_actions = np.asarray(example.left["actions"], dtype=np.float32)
        right_actions = np.asarray(example.right["actions"], dtype=np.float32)
        left_mask = np.asarray(example.left["action_mask"], dtype=bool)
        right_mask = np.asarray(example.right["action_mask"], dtype=bool)
        action_i[index, : len(left_actions)] = left_actions
        action_j[index, : len(right_actions)] = right_actions
        mask_i[index, : len(left_mask)] = left_mask
        mask_j[index, : len(right_mask)] = right_mask
    return {
        "state": torch.as_tensor(
            np.stack([np.asarray(item.left["state"], dtype=np.float32) for item in examples])
        ),
        "action_i": torch.as_tensor(action_i),
        "action_j": torch.as_tensor(action_j),
        "action_i_mask": torch.as_tensor(mask_i),
        "action_j_mask": torch.as_tensor(mask_j),
        "dt": torch.as_tensor([float(item.left["dt"]) for item in examples], dtype=torch.float32),
        "current_event_id": torch.as_tensor(
            [int(item.left["current_event_id"]) for item in examples], dtype=torch.long
        ),
        "event_age_seconds": torch.as_tensor(
            [float(item.left["event_age_seconds"]) for item in examples], dtype=torch.float32
        ),
        "remaining_action_budget": torch.as_tensor(
            [float(item.left["remaining_action_budget"]) for item in examples], dtype=torch.float32
        ),
        "label": torch.as_tensor([item.label for item in examples], dtype=torch.float32),
        "logical_group": [item.logical_group for item in examples],
        "left_candidate": torch.as_tensor(
            [item.left_candidate for item in examples], dtype=torch.long
        ),
        "right_candidate": torch.as_tensor(
            [item.right_candidate for item in examples], dtype=torch.long
        ),
        "tier": [item.tier for item in examples],
    }


def move_pair_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def pairwise_focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(labels).all())
        or not bool(((labels == 0.0) | (labels == 1.0)).all())
        or not math.isfinite(float(gamma))
        or gamma < 0.0
    ):
        raise RelativeActionCriticError("RAC focal BCE inputs are invalid")
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = bce * (1.0 - torch.exp(-bce)).pow(float(gamma))
    if sample_weight is None:
        return loss.mean()
    if (
        sample_weight.shape != logits.shape
        or not bool(torch.isfinite(sample_weight).all())
        or bool((sample_weight < 0.0).any())
    ):
        raise RelativeActionCriticError("RAC sample weights are invalid")
    if not bool((sample_weight > 0.0).any()):
        return loss.sum() * 0.0
    return (loss * sample_weight).sum() / sample_weight.sum()


def group_bootstrap_weights(
    collection: PreferencePairCollection,
    *,
    members: int = ENSEMBLE_SIZE,
    seed: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    groups = sorted(collection.group_pair_counts)
    if members != ENSEMBLE_SIZE or not groups:
        raise RelativeActionCriticError("RAC bootstrap contract is invalid")
    weights = core.logical_group_bootstrap_weights(
        groups, members=members, seed=seed
    ).astype(np.float32, copy=False)
    audits = []
    for member in range(members):
        repaired = None
        if not np.any(weights[member] > 0.0):
            selector = int.from_bytes(
                hashlib.sha256(f"{seed}|rac|{member}".encode()).digest()[:8], "big"
            )
            index = selector % len(groups)
            weights[member, index] = 1.0
            repaired = groups[index]
        audits.append(
            {
                "member": member,
                "real_preference_groups_total": len(groups),
                "real_preference_groups_nonzero": int((weights[member] > 0.0).sum()),
                "repaired_group": repaired,
                "synthetic_groups_or_labels": 0,
                "bootstrap_seed": seed,
            }
        )
    return (
        {group: weights[:, index].astype(float).tolist() for index, group in enumerate(groups)},
        audits,
    )


def pair_sample_weights(
    batch: Mapping[str, Any],
    *,
    member: int,
    group_weights: Mapping[str, Sequence[float]],
    group_pair_counts: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor:
    values = []
    for group in batch["logical_group"]:
        if group not in group_weights or group not in group_pair_counts:
            raise RelativeActionCriticError("RAC batch group lacks a frozen weight")
        weight = float(group_weights[group][member]) / int(group_pair_counts[group])
        values.append(weight)
    result = torch.as_tensor(values, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(result).all()) or bool((result < 0.0).any()):
        raise RelativeActionCriticError("RAC pair weight is invalid")
    return result


def _pair_loader(
    collection: PreferencePairCollection,
    *,
    batch_size_pairs: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        PreferencePairDataset(collection),
        batch_sampler=CompletePreferenceGroupBatchSampler(
            collection, batch_size_pairs=batch_size_pairs, seed=seed
        ),
        collate_fn=collate_preference_pairs,
    )


@torch.no_grad()
def predict_pair_logits(
    model: MatchedRelativeActionCritic,
    collection: PreferencePairCollection,
    *,
    batch_size_pairs: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = _pair_loader(
        collection, batch_size_pairs=batch_size_pairs, seed=0
    )
    by_identity: dict[tuple[str, int, int], float] = {}
    for raw in loader:
        batch = move_pair_batch(raw, device)
        logits = model(batch).detach().cpu().numpy()
        for group, left, right, logit in zip(
            raw["logical_group"],
            raw["left_candidate"].tolist(),
            raw["right_candidate"].tolist(),
            logits.tolist(),
        ):
            identity = (str(group), int(left), int(right))
            if identity in by_identity:
                raise RelativeActionCriticError("RAC validation pair repeated")
            by_identity[identity] = float(logit)
    expected = [
        (example.logical_group, example.left_candidate, example.right_candidate)
        for example in collection.examples
    ]
    if set(by_identity) != set(expected):
        raise RelativeActionCriticError("RAC validation pair roster changed")
    result = np.asarray([by_identity[identity] for identity in expected], dtype=np.float64)
    if result.shape != (len(collection.examples),) or not np.isfinite(result).all():
        raise RelativeActionCriticError("RAC validation logits are invalid")
    return result


def preference_metrics_from_member_logits(
    member_logits: np.ndarray,
    collection: PreferencePairCollection,
    *,
    gamma: float = FOCAL_GAMMA,
) -> dict[str, Any]:
    logits = np.asarray(member_logits, dtype=np.float64)
    if (
        logits.ndim != 2
        or logits.shape[1] != len(collection.examples)
        or logits.shape[0] < 1
        or not np.isfinite(logits).all()
    ):
        raise RelativeActionCriticError("RAC metric member logits are invalid")
    ensemble_logits = logits.mean(axis=0)
    labels = np.asarray([item.label for item in collection.examples], dtype=np.float64)
    # Stable BCE and focal values without relying on a device.
    bce = np.maximum(ensemble_logits, 0.0) - ensemble_logits * labels + np.log1p(
        np.exp(-np.abs(ensemble_logits))
    )
    focal = bce * np.power(1.0 - np.exp(-bce), float(gamma))
    predicted = ensemble_logits >= 0.0
    correct = predicted == (labels > 0.5)
    group_indices: dict[str, list[int]] = defaultdict(list)
    tier_indices: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(collection.examples):
        group_indices[example.logical_group].append(index)
        tier_indices[example.tier].append(index)
    group_focal = [float(focal[indices].mean()) for indices in group_indices.values()]
    group_bce = [float(bce[indices].mean()) for indices in group_indices.values()]

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(ensemble_logits, -60.0, 60.0)))
    selected_correct = []
    for group, indices in sorted(group_indices.items()):
        wins = np.zeros(CANDIDATE_COUNT, dtype=np.float64)
        counts = np.zeros(CANDIDATE_COUNT, dtype=np.int64)
        seen_unordered: set[tuple[int, int]] = set()
        for index in indices:
            example = collection.examples[index]
            unordered = tuple(sorted((example.left_candidate, example.right_candidate)))
            if unordered in seen_unordered:
                continue
            seen_unordered.add(unordered)
            probability = probabilities[index]
            wins[example.left_candidate] += probability
            wins[example.right_candidate] += 1.0 - probability
            counts[example.left_candidate] += 1
            counts[example.right_candidate] += 1
        # A tied candidate can have fewer informative comparisons.  The mean
        # over only real, labeled comparisons preserves the no-fabricated-label
        # contract; completely unobserved candidates are not selectable.
        scores = np.divide(
            wins,
            counts,
            out=np.full(CANDIDATE_COUNT, -np.inf, dtype=np.float64),
            where=counts > 0,
        )
        selected = int(np.argmax(scores))
        selected_correct.append(selected in collection.best_candidates[group])
    return {
        "pair_bce": float(bce.mean()),
        "pair_focal": float(focal.mean()),
        "macro_decision_pair_bce": float(np.mean(group_bce)),
        "macro_decision_pair_focal": float(np.mean(group_focal)),
        "pair_accuracy": float(correct.mean()),
        "decision_best_set_accuracy": float(np.mean(selected_correct)),
        "decision_groups": len(group_indices),
        "ordered_pairs": len(collection.examples),
        "ensemble_members": int(logits.shape[0]),
        "tier_accuracy": {
            tier: (
                float(correct[indices].mean()) if indices else None
            )
            for tier, indices in ((tier, tier_indices.get(tier, [])) for tier in PAIR_TIERS)
        },
        "heldout_rows_used": 0,
    }


@torch.no_grad()
def evaluate_preference_ensemble(
    models: Sequence[MatchedRelativeActionCritic],
    collection: PreferencePairCollection,
    *,
    batch_size_pairs: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    if len(models) != ENSEMBLE_SIZE:
        raise RelativeActionCriticError("RAC evaluation requires five members")
    logits = np.stack(
        [
            predict_pair_logits(
                model,
                collection,
                batch_size_pairs=batch_size_pairs,
                device=device,
            )
            for model in models
        ]
    )
    return preference_metrics_from_member_logits(logits, collection), logits


def fit_source_normalization(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not rows:
        raise RelativeActionCriticError("RAC normalization source is empty")
    source_normalization = core.fit_train_action_normalization(
        rows, required_schema_ids=(0,)
    )
    action = dict(source_normalization["schemas"]["aloha"])
    action_mean = np.asarray(action["mean"], dtype=np.float32)
    action_std = np.asarray(action["std"], dtype=np.float32)
    states = np.stack([np.asarray(row["state"], dtype=np.float32) for row in rows])
    state_mean = np.zeros(core.STATE_DIM, dtype=np.float32)
    state_std = np.ones(core.STATE_DIM, dtype=np.float32)
    state_mean[:18] = states[:, :18].mean(axis=0)
    state_std[:18] = np.maximum(states[:, :18].std(axis=0), 1e-4)
    audit = {
        "format": "etsf_rac_source_train_only_normalization_v1",
        "action": {
            "mean": action_mean.astype(float).tolist(),
            "std": action_std.astype(float).tolist(),
            "canonical_action_schema_id": 0,
        },
        "state": {
            "mean": state_mean.astype(float).tolist(),
            "std": state_std.astype(float).tolist(),
            "continuous_channels": list(range(18)),
            "binary_channels_unchanged": list(range(18, core.STATE_DIM)),
        },
        "source_rows": len(rows),
        "supplement_rows_used": 0,
        "heldout_rows_used": 0,
    }
    audit["logical_sha256"] = canonical_sha256(audit)
    return action_mean, action_std, state_mean, state_std, audit


def _new_model_with_normalization(
    *,
    config: RACConfig,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> MatchedRelativeActionCritic:
    model = MatchedRelativeActionCritic(config).to(device)
    model.set_normalization(
        action_mean=torch.as_tensor(action_mean, device=device),
        action_std=torch.as_tensor(action_std, device=device),
        state_mean=torch.as_tensor(state_mean, device=device),
        state_std=torch.as_tensor(state_std, device=device),
    )
    return model


def _next_batch(
    loader: DataLoader,
    iterator: Iterator[dict[str, Any]],
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def build_preflight_receipt(
    audit: Mapping[str, Any],
    *,
    held_out_body: str,
    split_seed: int,
    supplement_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parent = shared.build_preflight_receipt(
        audit,
        held_out_body=held_out_body,
        split_seed=split_seed,
        supplement_audit=supplement_audit,
    )
    result = {
        "format": FORMAT,
        "status": "rac_preflight_passed_payloads_still_unopened",
        "model_family": MODEL_FAMILY,
        "rac_contract": rac_contract(),
        "held_out_body": held_out_body,
        "source_bodies": parent["source_bodies"],
        "split_seed": split_seed,
        "primary_preflight_logical_sha256": parent["logical_sha256"],
        "primary_binding_file_sha256": parent["binding_file_sha256"],
        "actor_execution_protocol": parent["actor_execution_protocol"],
        "actor_execution_protocol_binding": parent[
            "actor_execution_protocol_binding"
        ],
        "actor_execution_protocol_file_sha256": parent[
            "actor_execution_protocol_file_sha256"
        ],
        "source_train_groups": parent["source_train_groups"],
        "source_validation_groups": parent["source_validation_groups"],
        "supplement": parent["supplement"],
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_labels_used_for_training_or_selection": False,
        "body_or_condition_trainable_adapter": False,
        "actor_frozen": True,
    }
    result["logical_sha256"] = canonical_sha256(result)
    return result


def _train_fold(
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    *,
    supplement_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("RAC training output must be a new path")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RelativeActionCriticError("real RAC training is remote CUDA-only")
    if "4090" not in torch.cuda.get_device_name(0):
        raise RelativeActionCriticError("real RAC training requires the authorized RTX 4090")
    shared.validate_ensemble_seeds(args.ensemble_seeds)
    primary_train_groups, primary_validation_groups, _heldout = shared.source_group_split(
        audit, held_out_body=args.held_out_body, split_seed=args.split_seed
    )
    primary_preflight = shared.build_preflight_receipt(
        audit,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
        supplement_audit=supplement_audit,
    )
    preflight = build_preflight_receipt(
        audit,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
        supplement_audit=supplement_audit,
    )
    output.mkdir(parents=True)
    core.atomic_json(output / "preflight_receipt.json", preflight)
    source_bodies = tuple(primary_preflight["source_bodies"])
    primary_train_rows = shared.materialize_source_rows(
        primary_train_groups, held_out_body=args.held_out_body
    )
    primary_validation_rows = shared.materialize_source_rows(
        primary_validation_groups, held_out_body=args.held_out_body
    )
    if supplement_audit is None:
        supplement_train_rows: list[dict[str, Any]] = []
        supplement_validation_rows: list[dict[str, Any]] = []
    else:
        supplement_train_groups, supplement_validation_groups, _supplement_heldout = (
            shared.supplement_source_train_split(
                supplement_audit,
                held_out_body=args.held_out_body,
                split_seed=args.split_seed,
            )
        )
        supplement_train_rows = shared.materialize_supplement_rows(
            supplement_train_groups, held_out_body=args.held_out_body
        )
        supplement_validation_rows = shared.materialize_supplement_rows(
            supplement_validation_groups, held_out_body=args.held_out_body
        )
    primary_train = build_preference_pairs(
        primary_train_rows,
        source_bodies=source_bodies,
        stream="primary_source_train",
    )
    primary_validation = build_preference_pairs(
        primary_validation_rows,
        source_bodies=source_bodies,
        stream="primary_source_validation",
    )
    supplement_train = (
        build_preference_pairs(
            supplement_train_rows,
            source_bodies=tuple(sorted({str(row["body"]) for row in supplement_train_rows})),
            stream="supplement_source_train",
        )
        if supplement_train_rows
        else None
    )
    supplement_validation = (
        build_preference_pairs(
            supplement_validation_rows,
            source_bodies=tuple(
                sorted({str(row["body"]) for row in supplement_validation_rows})
            ),
            stream="supplement_source_validation_diagnostic_only",
        )
        if supplement_validation_rows
        else None
    )
    action_mean, action_std, state_mean, state_std, normalization = (
        fit_source_normalization(primary_train_rows)
    )
    primary_bootstrap, primary_bootstrap_audit = group_bootstrap_weights(
        primary_train, members=ENSEMBLE_SIZE, seed=args.split_seed + 101
    )
    if supplement_train is not None:
        supplement_bootstrap, supplement_bootstrap_audit = group_bootstrap_weights(
            supplement_train, members=ENSEMBLE_SIZE, seed=args.split_seed + 211
        )
    else:
        supplement_bootstrap = {}
        supplement_bootstrap_audit = []
    config = RACConfig(
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    snapshots_root = output / "source_validation_common_step_snapshots"
    snapshots_root.mkdir()
    snapshot_paths: list[dict[int, Path]] = []
    member_validation: list[dict[int, dict[str, Any]]] = []
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = _new_model_with_normalization(
            config=config,
            action_mean=action_mean,
            action_std=action_std,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        primary_loader = _pair_loader(
            primary_train, batch_size_pairs=args.batch_size_pairs, seed=seed
        )
        primary_iterator = iter(primary_loader)
        supplement_loader = (
            _pair_loader(
                supplement_train,
                batch_size_pairs=args.batch_size_pairs,
                seed=seed + 1000,
            )
            if supplement_train is not None
            else None
        )
        supplement_iterator = iter(supplement_loader) if supplement_loader else None
        paths: dict[int, Path] = {}
        records: dict[int, dict[str, Any]] = {}
        last_train: dict[str, float] = {}
        for step in range(1, args.steps + 1):
            primary_raw, primary_iterator = _next_batch(
                primary_loader, primary_iterator
            )
            primary_batch = move_pair_batch(primary_raw, device)
            primary_logits = model(primary_batch)
            primary_weight = pair_sample_weights(
                primary_raw,
                member=member,
                group_weights=primary_bootstrap,
                group_pair_counts=primary_train.group_pair_counts,
                device=device,
            )
            primary_loss = pairwise_focal_bce_with_logits(
                primary_logits,
                primary_batch["label"],
                sample_weight=primary_weight,
                gamma=args.focal_gamma,
            )
            if supplement_loader is None:
                supplement_loss = primary_loss.new_zeros(())
            else:
                assert supplement_iterator is not None and supplement_train is not None
                supplement_raw, supplement_iterator = _next_batch(
                    supplement_loader, supplement_iterator
                )
                supplement_batch = move_pair_batch(supplement_raw, device)
                supplement_weight = pair_sample_weights(
                    supplement_raw,
                    member=member,
                    group_weights=supplement_bootstrap,
                    group_pair_counts=supplement_train.group_pair_counts,
                    device=device,
                )
                supplement_loss = pairwise_focal_bce_with_logits(
                    model(supplement_batch),
                    supplement_batch["label"],
                    sample_weight=supplement_weight,
                    gamma=args.focal_gamma,
                )
            loss = primary_loss + SUPPLEMENT_PAIR_LOSS_WEIGHT * supplement_loss
            if not bool(torch.isfinite(loss)):
                raise RelativeActionCriticError("RAC training loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_train = {
                "total": float(loss.detach()),
                "primary_pair_focal": float(primary_loss.detach()),
                "supplement_pair_focal_unweighted": float(supplement_loss.detach()),
                "supplement_pair_fixed_lambda": (
                    SUPPLEMENT_PAIR_LOSS_WEIGHT if supplement_loader else 0.0
                ),
            }
            if step % args.eval_every != 0 and step != args.steps:
                continue
            validation_logits = predict_pair_logits(
                model,
                primary_validation,
                batch_size_pairs=args.batch_size_pairs,
                device=device,
            )[None]
            metrics = preference_metrics_from_member_logits(
                validation_logits, primary_validation, gamma=args.focal_gamma
            )
            metrics["train_objective_last"] = last_train
            snapshot = snapshots_root / f"member_{member:02d}_seed_{seed}_step_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "member": member,
                    "seed": seed,
                    "step": step,
                },
                snapshot,
            )
            paths[step] = snapshot
            records[step] = metrics
            model.train()
        if not records:
            raise RelativeActionCriticError("RAC member produced no validation snapshot")
        snapshot_paths.append(paths)
        member_validation.append(records)
        del optimizer, model
        torch.cuda.empty_cache()
    common_steps = sorted(set.intersection(*(set(paths) for paths in snapshot_paths)))
    expected_steps = list(range(args.eval_every, args.steps + 1, args.eval_every))
    if not expected_steps or expected_steps[-1] != args.steps:
        expected_steps.append(args.steps)
    if common_steps != expected_steps:
        raise RelativeActionCriticError("RAC ensemble common-step roster changed")
    selection_models = [
        _new_model_with_normalization(
            config=config,
            action_mean=action_mean,
            action_std=action_std,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
        )
        for _ in range(ENSEMBLE_SIZE)
    ]
    selection_audit = []
    best_key: tuple[float, float, float, int] | None = None
    best_step = 0
    best_metrics: dict[str, Any] | None = None
    for step in common_steps:
        for member, model in enumerate(selection_models):
            snapshot = torch.load(
                snapshot_paths[member][step], map_location=device, weights_only=True
            )
            model.load_state_dict(snapshot["model"], strict=True)
        metrics, _member_logits = evaluate_preference_ensemble(
            selection_models,
            primary_validation,
            batch_size_pairs=args.batch_size_pairs,
            device=device,
        )
        key = (
            float(metrics["macro_decision_pair_focal"]),
            -float(metrics["decision_best_set_accuracy"]),
            -float(metrics["pair_accuracy"]),
            int(step),
        )
        selection_audit.append({"step": step, "selection_key": list(key), "metrics": metrics})
        if best_key is None or key < best_key:
            best_key, best_step, best_metrics = key, step, metrics
    if best_metrics is None or best_step <= 0:
        raise RelativeActionCriticError("RAC source-only checkpoint selection failed")
    for member, model in enumerate(selection_models):
        snapshot = torch.load(
            snapshot_paths[member][best_step], map_location=device, weights_only=True
        )
        model.load_state_dict(snapshot["model"], strict=True)
    supplement_validation_metrics = (
        evaluate_preference_ensemble(
            selection_models,
            supplement_validation,
            batch_size_pairs=args.batch_size_pairs,
            device=device,
        )[0]
        if supplement_validation is not None
        else None
    )
    trainer_sha = sha256_file(Path(__file__).resolve())
    members = []
    for member, seed in enumerate(args.ensemble_seeds):
        snapshot = torch.load(
            snapshot_paths[member][best_step], map_location="cpu", weights_only=True
        )
        checkpoint_path = output / f"member_{member:02d}_seed_{seed}_best.pt"
        checkpoint = {
            "format": FORMAT,
            "model_family": MODEL_FAMILY,
            "model": snapshot["model"],
            "config": dataclasses.asdict(config),
            "member": member,
            "seed": seed,
            "step": best_step,
            "ensemble_common_selection_step": best_step,
            "held_out_body": args.held_out_body,
            "source_bodies": list(source_bodies),
            "body_adapter": "none_body_identity_not_an_input",
            "rac_contract": rac_contract(),
            "canonical_state_schema": shared.CANONICAL_STATE_SCHEMA,
            "canonical_action_schema": shared.CANONICAL_ACTION_SCHEMA,
            "state_action_frame_contract": shared.state_action_frame_contract(),
            "actor_execution_protocol": primary_preflight[
                "actor_execution_protocol"
            ],
            "actor_execution_protocol_binding": primary_preflight[
                "actor_execution_protocol_binding"
            ],
            "actor_execution_protocol_file_sha256": primary_preflight[
                "actor_execution_protocol_file_sha256"
            ],
            "normalization": normalization,
            "primary_train_pairs": primary_train.audit,
            "primary_validation_pairs": primary_validation.audit,
            "supplement_train_pairs": (
                supplement_train.audit if supplement_train is not None else None
            ),
            "primary_bootstrap": primary_bootstrap_audit[member],
            "supplement_bootstrap": (
                supplement_bootstrap_audit[member]
                if supplement_bootstrap_audit
                else None
            ),
            "source_validation": member_validation[member][best_step],
            "heldout_rows_used_for_training_normalization_or_selection": 0,
            "synthetic_or_tie_labels": 0,
            "trainer_file_sha256": trainer_sha,
        }
        torch.save(checkpoint, checkpoint_path)
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": best_step,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_validation": member_validation[member][best_step],
            }
        )
    del selection_models
    torch.cuda.empty_cache()
    for paths in snapshot_paths:
        for snapshot_path in paths.values():
            snapshot_path.unlink()
    snapshots_root.rmdir()
    summary = {
        "format": SUMMARY_FORMAT,
        "status": "source_only_rac_checkpoint_selection_complete",
        "model_family": MODEL_FAMILY,
        "held_out_body": args.held_out_body,
        "source_bodies": list(source_bodies),
        "rac_contract": rac_contract(),
        "canonical_state_schema": shared.CANONICAL_STATE_SCHEMA,
        "canonical_action_schema": shared.CANONICAL_ACTION_SCHEMA,
        "state_action_frame_contract": shared.state_action_frame_contract(),
        "actor_execution_protocol": primary_preflight["actor_execution_protocol"],
        "actor_execution_protocol_binding": primary_preflight[
            "actor_execution_protocol_binding"
        ],
        "actor_execution_protocol_file_sha256": primary_preflight[
            "actor_execution_protocol_file_sha256"
        ],
        "normalization": normalization,
        "training_budget": {
            "steps_per_member": args.steps,
            "eval_every_steps": args.eval_every,
            "ensemble_members": len(args.ensemble_seeds),
            "batch_size_pairs": args.batch_size_pairs,
            "learning_rate": args.learning_rate,
            "focal_gamma": args.focal_gamma,
            "supplement_pair_loss_weight": (
                SUPPLEMENT_PAIR_LOSS_WEIGHT if supplement_train is not None else 0.0
            ),
        },
        "primary_train_pairs": primary_train.audit,
        "primary_validation_pairs": primary_validation.audit,
        "supplement_train_pairs": (
            supplement_train.audit if supplement_train is not None else None
        ),
        "supplement_validation_pairs": (
            supplement_validation.audit if supplement_validation is not None else None
        ),
        "supplement_validation_role": "diagnostic_only_not_checkpoint_selection",
        "checkpoint_selection": {
            "scope": "primary_source_validation_only",
            "selection_key": (
                "min_macro_decision_focal_then_max_best_set_accuracy_then_"
                "max_pair_accuracy_then_earliest_step"
            ),
            "selected_step": best_step,
            "selected_metrics": best_metrics,
            "common_step_audit": selection_audit,
            "supplement_rows_used": 0,
            "heldout_rows_used": 0,
        },
        "supplement_validation_diagnostic": supplement_validation_metrics,
        "members": members,
        "trainer_file_sha256": trainer_sha,
        "heldout_rows_used_for_training_normalization_or_selection": 0,
        "all_checkpoints_selected_before_any_heldout_payload_open": True,
    }
    summary["logical_sha256"] = canonical_sha256(summary)
    core.atomic_json(output / "training_summary.json", summary)
    return summary


def load_checkpoint_model(
    checkpoint: Mapping[str, Any], *, device: torch.device
) -> MatchedRelativeActionCritic:
    if (
        checkpoint.get("format") != FORMAT
        or checkpoint.get("model_family") != MODEL_FAMILY
        or checkpoint.get("rac_contract") != rac_contract()
        or checkpoint.get("canonical_state_schema") != shared.CANONICAL_STATE_SCHEMA
        or checkpoint.get("canonical_action_schema") != shared.CANONICAL_ACTION_SCHEMA
        or checkpoint.get("heldout_rows_used_for_training_normalization_or_selection") != 0
        or checkpoint.get("synthetic_or_tie_labels") != 0
    ):
        raise RelativeActionCriticError("RAC checkpoint contract changed")
    config_raw = checkpoint.get("config")
    normalization = checkpoint.get("normalization")
    if not isinstance(config_raw, Mapping) or not isinstance(normalization, Mapping):
        raise RelativeActionCriticError("RAC checkpoint lacks config/normalization")
    try:
        config = RACConfig(**dict(config_raw))
        action_mean = np.asarray(normalization["action"]["mean"], dtype=np.float32)
        action_std = np.asarray(normalization["action"]["std"], dtype=np.float32)
        state_mean = np.asarray(normalization["state"]["mean"], dtype=np.float32)
        state_std = np.asarray(normalization["state"]["std"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise RelativeActionCriticError("RAC checkpoint normalization is invalid") from error
    model = _new_model_with_normalization(
        config=config,
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise RelativeActionCriticError("RAC checkpoint lacks model state")
    model.load_state_dict(state, strict=True)
    return model.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train-fold"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--supplement-binding", type=Path)
    parser.add_argument("--supplement-binding-sha256")
    parser.add_argument("--held-out-body", choices=BODIES, required=True)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--batch-size-pairs", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ensemble-seeds", nargs=5, type=int, default=list(DEFAULT_ENSEMBLE_SEEDS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared.validate_ensemble_seeds(args.ensemble_seeds)
    if (
        args.steps <= 0
        or args.eval_every <= 0
        or args.batch_size_pairs < 12
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not math.isfinite(args.focal_gamma)
        or args.focal_gamma < 0.0
    ):
        raise RelativeActionCriticError("RAC training hyperparameters are invalid")
    audit = shared.load_binding(args.binding, args.binding_sha256)
    if (args.supplement_binding is None) != (
        args.supplement_binding_sha256 is None
    ):
        raise RelativeActionCriticError(
            "RAC supplement binding path and SHA must be supplied together"
        )
    supplement_audit = (
        shared.load_supplement_binding(
            args.supplement_binding,
            args.supplement_binding_sha256,
            primary_audit=audit,
            held_out_body=args.held_out_body,
        )
        if args.supplement_binding is not None
        else None
    )
    if args.mode == "preflight":
        receipt = build_preflight_receipt(
            audit,
            held_out_body=args.held_out_body,
            split_seed=args.split_seed,
            supplement_audit=supplement_audit,
        )
        print("PREFLIGHT=" + json.dumps(receipt, sort_keys=True))
        return
    if args.output is None:
        raise RelativeActionCriticError("RAC train-fold requires --output")
    summary = _train_fold(args, audit, supplement_audit=supplement_audit)
    print("TRAINING=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BODIES",
    "DEFAULT_ENSEMBLE_SEEDS",
    "DEFAULT_STEPS",
    "ENSEMBLE_SIZE",
    "FOCAL_GAMMA",
    "FORMAT",
    "GOAL_EQUALITY_TOLERANCE",
    "MODEL_FAMILY",
    "MatchedRelativeActionCritic",
    "PreferencePair",
    "PreferencePairCollection",
    "RACConfig",
    "RelativeActionCriticError",
    "SUMMARY_FORMAT",
    "SUPPORTED_RUNTIME_CANDIDATE_COUNTS",
    "build_preference_pairs",
    "build_preflight_receipt",
    "canonical_sha256",
    "collate_preference_pairs",
    "evaluate_preference_ensemble",
    "fit_source_normalization",
    "group_bootstrap_weights",
    "lexicographic_preference",
    "load_checkpoint_model",
    "pair_sample_weights",
    "pairwise_focal_bce_with_logits",
    "preference_metrics_from_member_logits",
    "rac_contract",
    "sha256_file",
]
