#!/usr/bin/env python3
"""Load and score the matched RAC baseline for RoboTwin N4/N8 runners.

The public scorer intentionally mirrors the rank-audit fields consumed by the
existing paired-success and nested N4/N8 verification code.  Unlike the
official single-elimination RAC tournament, every unordered candidate pair is
evaluated once per ensemble member and each candidate receives its mean soft
win probability against all opponents (soft Copeland/Borda score).
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import train_robotwin2_five_body_lobo_relative_action_critic_v1 as rac
import train_robotwin2_five_body_lobo_shared_event_head_v1 as shared


FORMAT = "etsf_robotwin2_relative_action_critic_runtime_adapter_v1"
RUNTIME_CANDIDATE_COUNTS = rac.SUPPORTED_RUNTIME_CANDIDATE_COUNTS


class RelativeActionCriticAdapterError(RuntimeError):
    pass


def runtime_contract(candidate_count: int) -> dict[str, Any]:
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count not in RUNTIME_CANDIDATE_COUNTS
    ):
        raise RelativeActionCriticAdapterError(
            "RAC runtime candidate count must be N4 or N8"
        )
    return {
        "format": FORMAT,
        "candidate_count": candidate_count,
        "unordered_pairs_per_member": math.comb(candidate_count, 2),
        "pair_probability": "sigmoid_antisymmetric_preference_logit",
        "candidate_score": "mean_soft_win_probability_against_all_other_candidates",
        "all_pairs_evaluated": True,
        "single_elimination_bracket_used": False,
        "candidate_order_tie_break": "lowest_index_only_after_equal_aggregate_score",
        "member_axis": rac.ENSEMBLE_SIZE,
        "ensemble_aggregation": shared.risk_adjusted_rank_ensemble_contract()[
            "aggregation"
        ],
        "heldout_labels_or_outcomes_read": False,
    }


def _validate_runtime_candidate_batch(
    batch: Mapping[str, torch.Tensor], *, candidate_count: int
) -> None:
    runtime_contract(candidate_count)
    required = {
        "state",
        "actions",
        "action_mask",
        "action_available",
        "action_schema_id",
        "dt",
        "current_event_id",
        "event_age_seconds",
        "remaining_action_budget",
    }
    if not required <= set(batch):
        raise RelativeActionCriticAdapterError(
            f"RAC runtime batch lacks {sorted(required-set(batch))}"
        )
    state = batch["state"]
    actions = batch["actions"]
    mask = batch["action_mask"]
    if (
        state.shape != (candidate_count, rac.core.STATE_DIM)
        or actions.ndim != 3
        or actions.shape[0] != candidate_count
        or actions.shape[-1] != rac.core.ACTION_DIM
        or mask.shape != actions.shape[:2]
        or mask.dtype != torch.bool
        or batch["action_available"].shape != (candidate_count,)
        or batch["action_schema_id"].shape != (candidate_count,)
        or not bool(batch["action_available"].all())
        or not bool((batch["action_schema_id"] == 0).all())
        or not bool(mask.any(dim=1).all())
    ):
        raise RelativeActionCriticAdapterError("RAC runtime candidate tensors changed")
    for name in ("dt", "current_event_id", "event_age_seconds", "remaining_action_budget"):
        value = batch[name]
        if value.shape != (candidate_count,):
            raise RelativeActionCriticAdapterError(f"RAC runtime {name} shape changed")
        if name != "current_event_id" and not bool(torch.isfinite(value).all()):
            raise RelativeActionCriticAdapterError(f"RAC runtime {name} is non-finite")
    floating = (state, actions)
    if not all(bool(torch.isfinite(value).all()) for value in floating):
        raise RelativeActionCriticAdapterError("RAC runtime input is non-finite")
    # One candidate pool must share one exact pre-action causal context.
    if not all(torch.equal(state[0], state[index]) for index in range(1, candidate_count)):
        raise RelativeActionCriticAdapterError("RAC candidates do not share one state")
    for name in ("dt", "current_event_id", "event_age_seconds", "remaining_action_budget"):
        value = batch[name]
        if not all(bool(value[index] == value[0]) for index in range(1, candidate_count)):
            raise RelativeActionCriticAdapterError(
                f"RAC candidates do not share one {name}"
            )
    current = int(batch["current_event_id"][0])
    if not 0 <= current < len(rac.core.CANONICAL_EVENTS):
        raise RelativeActionCriticAdapterError("RAC current event is invalid")
    expected = torch.zeros(
        len(rac.core.CANONICAL_EVENTS), dtype=state.dtype, device=state.device
    )
    expected[current] = 1.0
    if not torch.equal(state[0, 18:23], expected):
        raise RelativeActionCriticAdapterError(
            "RAC state event onehot disagrees with current event"
        )


def all_pair_batch(
    batch: Mapping[str, torch.Tensor], *, candidate_count: int
) -> tuple[dict[str, torch.Tensor], tuple[tuple[int, int], ...]]:
    _validate_runtime_candidate_batch(batch, candidate_count=candidate_count)
    pairs = tuple(combinations(range(candidate_count), 2))
    left = torch.as_tensor(
        [pair[0] for pair in pairs], dtype=torch.long, device=batch["state"].device
    )
    right = torch.as_tensor(
        [pair[1] for pair in pairs], dtype=torch.long, device=batch["state"].device
    )
    pair_batch = {
        "state": batch["state"].index_select(0, left),
        "action_i": batch["actions"].index_select(0, left),
        "action_j": batch["actions"].index_select(0, right),
        "action_i_mask": batch["action_mask"].index_select(0, left),
        "action_j_mask": batch["action_mask"].index_select(0, right),
        "dt": batch["dt"].index_select(0, left),
        "current_event_id": batch["current_event_id"].index_select(0, left),
        "event_age_seconds": batch["event_age_seconds"].index_select(0, left),
        "remaining_action_budget": batch["remaining_action_budget"].index_select(
            0, left
        ),
    }
    if len(pairs) != math.comb(candidate_count, 2):
        raise RelativeActionCriticAdapterError("RAC all-pair roster is incomplete")
    return pair_batch, pairs


def aggregate_member_candidate_scores(member_scores: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(member_scores, torch.Tensor)
        or member_scores.ndim != 2
        or member_scores.shape[0] != rac.ENSEMBLE_SIZE
        or int(member_scores.shape[1]) not in RUNTIME_CANDIDATE_COUNTS
        or not bool(torch.isfinite(member_scores).all())
    ):
        raise RelativeActionCriticAdapterError("RAC member candidate scores are invalid")
    return member_scores.mean(dim=0) - float(
        shared.EPISTEMIC_RANK_RISK_WEIGHT
    ) * member_scores.std(dim=0, correction=0)


@torch.no_grad()
def score_candidates(
    models: Sequence[rac.MatchedRelativeActionCritic],
    batch: Mapping[str, torch.Tensor],
    *,
    candidate_count: int,
) -> dict[str, Any]:
    if len(models) != rac.ENSEMBLE_SIZE:
        raise RelativeActionCriticAdapterError("RAC scoring requires five members")
    pair_batch, pairs = all_pair_batch(batch, candidate_count=candidate_count)
    member_candidate_scores = []
    member_pair_probabilities = []
    for model in models:
        model.eval()
        probability = torch.sigmoid(model(pair_batch))
        if probability.shape != (len(pairs),) or not bool(torch.isfinite(probability).all()):
            raise RelativeActionCriticAdapterError("RAC pair probability is invalid")
        wins = probability.new_zeros(candidate_count)
        counts = probability.new_zeros(candidate_count)
        matrix = probability.new_full((candidate_count, candidate_count), 0.5)
        for pair_index, (left, right) in enumerate(pairs):
            value = probability[pair_index]
            wins[left] += value
            wins[right] += 1.0 - value
            counts[left] += 1.0
            counts[right] += 1.0
            matrix[left, right] = value
            matrix[right, left] = 1.0 - value
        if not bool((counts == candidate_count - 1).all()):
            raise RelativeActionCriticAdapterError("RAC candidate missed an opponent")
        member_candidate_scores.append(wins / counts)
        member_pair_probabilities.append(matrix)
    scores = torch.stack(member_candidate_scores)
    matrices = torch.stack(member_pair_probabilities)
    risk_adjusted = aggregate_member_candidate_scores(scores)
    selected = int(torch.argmax(risk_adjusted))
    scores_np = scores.detach().cpu().numpy().astype(float)
    risk_np = risk_adjusted.detach().cpu().numpy().astype(float)
    matrices_np = matrices.detach().cpu().numpy().astype(float)
    return {
        "selected_candidate_index": selected,
        # Existing N4/N8 audit-compatible fields.
        "candidate_rank_score_epistemic_lcb_ensemble": risk_np.tolist(),
        "candidate_rank_score_mean": scores_np.mean(axis=0).tolist(),
        "candidate_rank_score_raw_candidate_population_std": scores_np.std(
            axis=0, ddof=0
        ).tolist(),
        "candidate_rank_score_raw_member_candidate_mean": scores_np.mean(
            axis=1
        ).tolist(),
        "candidate_rank_score_raw_member_candidate_population_std": scores_np.std(
            axis=1, ddof=0
        ).tolist(),
        "candidate_rank_score_members": scores_np.tolist(),
        # RAC-specific replay evidence.
        "relative_action_critic_runtime_contract": runtime_contract(candidate_count),
        "rac_unordered_pair_indices": [list(pair) for pair in pairs],
        "rac_unordered_pairs_evaluated_per_member": len(pairs),
        "rac_pair_probability_matrix_members": matrices_np.tolist(),
        "rac_pair_probability_matrix_mean": matrices_np.mean(axis=0).tolist(),
        "rac_all_pair_candidate_score_definition": (
            "mean_probability_candidate_i_preferred_to_each_j_not_equal_i"
        ),
    }


def score_n4_candidates(
    models: Sequence[rac.MatchedRelativeActionCritic],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Drop-in score signature for the formal N4 runner."""

    return score_candidates(models, batch, candidate_count=4)


def score_n8_candidates(
    models: Sequence[rac.MatchedRelativeActionCritic],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Drop-in score signature for the nested/postformal N8 runner."""

    return score_candidates(models, batch, candidate_count=8)


def _read_summary(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RelativeActionCriticAdapterError("RAC summary is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RelativeActionCriticAdapterError("RAC summary is invalid JSON") from error
    if not isinstance(value, dict):
        raise RelativeActionCriticAdapterError("RAC summary must be an object")
    logical = value.get("logical_sha256")
    unsigned = {key: item for key, item in value.items() if key != "logical_sha256"}
    if logical != rac.canonical_sha256(unsigned):
        raise RelativeActionCriticAdapterError("RAC summary logical SHA changed")
    return value


def load_fold_ensemble(
    summary_path: Path,
    *,
    device: torch.device,
    expected_held_out_body: str | None = None,
) -> tuple[list[rac.MatchedRelativeActionCritic], dict[str, Any]]:
    summary_path = summary_path.expanduser().resolve()
    summary = _read_summary(summary_path)
    heldout = summary.get("held_out_body")
    members = summary.get("members")
    selection = summary.get("checkpoint_selection")
    if (
        summary.get("format") != rac.SUMMARY_FORMAT
        or summary.get("status") != "source_only_rac_checkpoint_selection_complete"
        or summary.get("model_family") != rac.MODEL_FAMILY
        or summary.get("rac_contract") != rac.rac_contract()
        or heldout not in rac.BODIES
        or (expected_held_out_body is not None and heldout != expected_held_out_body)
        or summary.get("heldout_rows_used_for_training_normalization_or_selection") != 0
        or summary.get("all_checkpoints_selected_before_any_heldout_payload_open") is not True
        or not isinstance(members, list)
        or len(members) != rac.ENSEMBLE_SIZE
        or not isinstance(selection, Mapping)
        or selection.get("heldout_rows_used") != 0
    ):
        raise RelativeActionCriticAdapterError("RAC summary contract changed")
    source_bodies = summary.get("source_bodies")
    if source_bodies != [body for body in rac.BODIES if body != heldout]:
        raise RelativeActionCriticAdapterError("RAC LOBO source-body roster changed")
    try:
        frozen_seeds = shared.validate_ensemble_seeds(
            [item.get("seed") for item in members if isinstance(item, Mapping)]
        )
    except shared.FiveBodyContractError as error:
        raise RelativeActionCriticAdapterError("RAC ensemble seeds changed") from error
    root = summary_path.parent
    models = []
    checkpoint_audit = []
    selected_step = selection.get("selected_step")
    for expected_member, item in enumerate(members):
        if (
            not isinstance(item, Mapping)
            or item.get("member") != expected_member
            or item.get("seed") != frozen_seeds[expected_member]
            or item.get("best_step") != selected_step
        ):
            raise RelativeActionCriticAdapterError("RAC ensemble member roster changed")
        checkpoint_path = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        try:
            checkpoint_path.relative_to(root)
        except ValueError as error:
            raise RelativeActionCriticAdapterError("RAC checkpoint escapes fold root") from error
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.is_symlink()
            or rac.sha256_file(checkpoint_path) != item.get("checkpoint_sha256")
        ):
            raise RelativeActionCriticAdapterError("RAC checkpoint is missing or changed")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("member") != expected_member
            or checkpoint.get("seed") != item["seed"]
            or checkpoint.get("step") != selected_step
            or checkpoint.get("held_out_body") != heldout
            or checkpoint.get("source_bodies") != source_bodies
            or checkpoint.get("state_action_frame_contract")
            != shared.state_action_frame_contract()
            or checkpoint.get("actor_execution_protocol")
            != summary.get("actor_execution_protocol")
            or checkpoint.get("actor_execution_protocol_binding")
            != summary.get("actor_execution_protocol_binding")
            or checkpoint.get("actor_execution_protocol_file_sha256")
            != summary.get("actor_execution_protocol_file_sha256")
        ):
            raise RelativeActionCriticAdapterError("RAC checkpoint binding changed")
        models.append(rac.load_checkpoint_model(checkpoint, device=device))
        checkpoint_audit.append(
            {
                "member": expected_member,
                "seed": item["seed"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
        )
    receipt = {
        "format": "etsf_rac_fold_ensemble_load_receipt_v1",
        "held_out_body": heldout,
        "source_bodies": source_bodies,
        "selected_step": selected_step,
        "members": checkpoint_audit,
        "heldout_payloads_or_labels_opened": 0,
    }
    receipt["logical_sha256"] = rac.canonical_sha256(receipt)
    return models, receipt


__all__ = [
    "FORMAT",
    "RUNTIME_CANDIDATE_COUNTS",
    "RelativeActionCriticAdapterError",
    "aggregate_member_candidate_scores",
    "all_pair_batch",
    "load_fold_ensemble",
    "runtime_contract",
    "score_candidates",
    "score_n4_candidates",
    "score_n8_candidates",
]
