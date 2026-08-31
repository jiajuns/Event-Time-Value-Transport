#!/usr/bin/env python3
"""Train one matched WCM-style source-only RoboTwin2 LOBO fold.

The trainer deliberately reuses the frozen v13 input authority, manifest
validation, label-blind split, canonical payload materializer, actor protocol,
source-only normalization and causal proper-loss balance.  It writes a
separate model family and never changes or masquerades as the ETSF shared head.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import robotwin2_wcm_future_latent_baseline_v1 as wcm
import train_multibody_canonical_event_world_model as core
import train_robotwin2_five_body_lobo_shared_event_head_v1 as source


FORMAT = "etsf_robotwin2_five_body_lobo_wcm_future_latent_trainer_v1"
SUMMARY_FORMAT = "etsf_robotwin2_five_body_lobo_wcm_future_latent_summary_v1"
DEFAULT_SPLIT_SEED = 20260901
DEFAULT_ENSEMBLE_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
DEFAULT_STEPS = 3000
DEFAULT_EVAL_EVERY = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
SUPPLEMENT_LOSS_WEIGHT = source.SUPPLEMENT_PROPER_LOSS_WEIGHT
TRAINING_CONTRACT = {
    "format": "etsf_wcm_style_matched_training_budget_v1",
    "ensemble_members": 5,
    "steps_per_member": DEFAULT_STEPS,
    "eval_every_steps": DEFAULT_EVAL_EVERY,
    "batch_size_rows": DEFAULT_BATCH_SIZE,
    "candidate_rows_per_decision": source.CANDIDATE_COUNT,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "primary_balance": source.DEFAULT_PROPER_BALANCE_MODE,
    "source_group_bootstrap": "outcome_preserving_group_constant_poisson_five_member",
    "supplement_train_weight": SUPPLEMENT_LOSS_WEIGHT,
    "supplement_validation_weight": SUPPLEMENT_LOSS_WEIGHT,
    "common_source_validation_step_for_all_members": True,
    "heldout_payload_opened": False,
}


class WCMTrainingError(RuntimeError):
    """A matched training, source-only, or artifact contract failed closed."""


def _conditioned_rows(
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    *,
    supplement: bool,
) -> list[dict[str, Any]]:
    """Attach manifest-visible condition after the frozen payload loader.

    The reused v13 materializer intentionally places condition in the logical
    group identity but currently omits a standalone row field.  Causal balance
    needs that manifest-visible, label-free field.  This wrapper restores it
    in the new baseline only; it never opens another payload or changes v13.
    """

    condition_by_identity: dict[str, str] = {}
    for group in groups:
        body = str(group.get("body", ""))
        condition = str(group.get("condition", ""))
        group_id = str(group.get("group_id", ""))
        if body not in source.BODIES or condition not in source.CONDITIONS or not group_id:
            raise WCMTrainingError("source group lacks body/condition/group identity")
        identity = (
            f"{body}|{condition}|proper-world-supplement|{group_id}"
            if supplement
            else f"{body}|{condition}|{group_id}"
        )
        if identity in condition_by_identity:
            raise WCMTrainingError("duplicate source logical group identity")
        condition_by_identity[identity] = condition
    normalized = []
    for raw in rows:
        row = dict(raw)
        identity = str(row.get("logical_group", ""))
        if identity not in condition_by_identity:
            raise WCMTrainingError("materialized row lacks its manifest condition")
        row["condition"] = condition_by_identity[identity]
        normalized.append(row)
    if set(condition_by_identity) != {
        str(row["logical_group"]) for row in normalized
    }:
        raise WCMTrainingError("one or more source groups produced no rows")
    return normalized


def materialize_primary_rows(
    groups: Sequence[Mapping[str, Any]], *, held_out_body: str
) -> list[dict[str, Any]]:
    rows = source.materialize_source_rows(groups, held_out_body=held_out_body)
    return _conditioned_rows(rows, groups, supplement=False)


def materialize_supplement_rows(
    groups: Sequence[Mapping[str, Any]], *, held_out_body: str
) -> list[dict[str, Any]]:
    rows = source.materialize_supplement_rows(groups, held_out_body=held_out_body)
    return _conditioned_rows(rows, groups, supplement=True)


def fit_primary_source_normalization(
    rows: Sequence[Mapping[str, Any]], *, held_out_body: str
) -> dict[str, Any]:
    if not rows or any(str(row.get("body")) == held_out_body for row in rows):
        raise WCMTrainingError("normalization received empty or held-out rows")
    action = core.fit_train_action_normalization(rows, required_schema_ids=(0,))
    action_schema = action["schemas"]["aloha"]
    action_mean = np.asarray(action_schema["mean"], dtype=np.float32)
    action_std = np.asarray(action_schema["std"], dtype=np.float32)
    states = np.stack([np.asarray(row["state"], dtype=np.float32) for row in rows])
    state_mean = np.zeros(wcm.STATE_DIM, dtype=np.float32)
    state_std = np.ones(wcm.STATE_DIM, dtype=np.float32)
    state_mean[:18] = states[:, :18].mean(axis=0)
    state_std[:18] = np.maximum(states[:, :18].std(axis=0), 1e-4)
    if (
        action_mean.shape != (wcm.ACTION_DIM,)
        or action_std.shape != (wcm.ACTION_DIM,)
        or not np.isfinite(action_mean).all()
        or not np.isfinite(action_std).all()
    ):
        raise WCMTrainingError("primary source normalization is invalid")
    receipt = {
        "format": "etsf_wcm_matched_primary_source_normalization_v1",
        "canonical_state_schema": wcm.STATE_SCHEMA,
        "canonical_action_schema": wcm.ACTION_SCHEMA,
        "state_continuous_channels": list(range(18)),
        "state_binary_channels_unchanged": list(range(18, wcm.STATE_DIM)),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "primary_source_train_rows": len(rows),
        "supplement_rows_used": 0,
        "validation_rows_used": 0,
        "heldout_rows_used": 0,
    }
    receipt["logical_sha256"] = wcm.canonical_sha256(receipt)
    return receipt


def apply_normalization(
    model: wcm.WCMFutureLatentBaseline,
    receipt: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    unsigned = dict(receipt)
    digest = unsigned.pop("logical_sha256", None)
    if (
        digest != wcm.canonical_sha256(unsigned)
        or receipt.get("supplement_rows_used") != 0
        or receipt.get("validation_rows_used") != 0
        or receipt.get("heldout_rows_used") != 0
    ):
        raise WCMTrainingError("normalization receipt changed")
    model.set_input_normalization(
        state_mean=torch.tensor(receipt["state_mean"], device=device),
        state_std=torch.tensor(receipt["state_std"], device=device),
        action_mean=torch.tensor(receipt["action_mean"], device=device),
        action_std=torch.tensor(receipt["action_std"], device=device),
    )


def baseline_preflight(
    audit: Mapping[str, Any],
    *,
    held_out_body: str,
    split_seed: int,
    supplement_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    inherited = source.build_preflight_receipt(
        audit,
        held_out_body=held_out_body,
        split_seed=split_seed,
        supplement_audit=supplement_audit,
    )
    if (
        inherited.get("heldout_group_npz_opened") != 0
        or inherited.get("heldout_group_payload_bytes_read") != 0
        or inherited.get("heldout_group_payload_deserialized") != 0
        or inherited.get("heldout_labels_used_for_normalization_training_or_selection")
        is not False
    ):
        raise WCMTrainingError("inherited preflight does not prove heldout zero-open")
    if (
        inherited.get("state_action_frame_contract")
        != wcm.STATE_ACTION_FRAME_CONTRACT
        or inherited.get("event_spec_sha256") != wcm.EVENT_SPEC_SHA256
    ):
        raise WCMTrainingError("shared-head and matched-WCM canonical ABI disagree")
    parameter_budget = wcm.parameter_budget_receipt(
        wcm.WCMFutureLatentBaseline()
    )
    base = {
        "format": FORMAT,
        "status": "matched_wcm_preflight_passed_payloads_still_unopened",
        "model_family": wcm.MODEL_FAMILY,
        "not_official_wcm_architecture_or_weights": True,
        "future_target": "canonical_terminal_consequence_and_object_effect_latent",
        "raw_future_observation_or_image_available": False,
        "primary_preflight": inherited,
        "primary_preflight_logical_sha256": inherited["logical_sha256"],
        "parameter_budget": parameter_budget,
        "training_contract": dict(TRAINING_CONTRACT),
        "state_action_frame_contract": source.state_action_frame_contract(),
        "event_spec_sha256": source.EVENT_SPEC_SHA256,
        "held_out_body": held_out_body,
        "source_bodies": [body for body in source.BODIES if body != held_out_body],
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "body_or_condition_trainable_input": False,
        "actor_frozen": True,
    }
    return {**base, "logical_sha256": wcm.canonical_sha256(base)}


def _group_constant_map(
    rows: Sequence[Mapping[str, Any]], weights: np.ndarray, *, member: int
) -> dict[str, float]:
    if weights.shape != (5, len(rows)):
        raise WCMTrainingError("bootstrap row-weight shape changed")
    result: dict[str, float] = {}
    for index, row in enumerate(rows):
        group = str(row["logical_group"])
        value = float(weights[member, index])
        previous = result.setdefault(group, value)
        if previous != value:
            raise WCMTrainingError("bootstrap weight is not constant within decision")
    return result


def _balance_group_map(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, float]
) -> dict[str, float]:
    groups = {str(row["logical_group"]) for row in rows}
    if set(weights) != groups:
        raise WCMTrainingError("causal balance group support changed")
    result = {group: float(weights[group]) for group in sorted(groups)}
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        raise WCMTrainingError("causal balance group weight is invalid")
    return result


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_model(
    model: wcm.WCMFutureLatentBaseline,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    diagnostic_totals: dict[str, float] = defaultdict(float)
    proper_totals: dict[str, float] = defaultdict(float)
    proper_support: dict[str, float] = defaultdict(float)
    rows = 0
    brier_total = 0.0
    stage_absolute_total = 0.0
    goal_absolute_total = 0.0
    brier_support = 0.0
    stage_support = 0.0
    goal_support = 0.0
    for raw in loader:
        batch = _move_batch(raw, device)
        output = model(batch)
        _loss, pieces = wcm.compute_wcm_loss(model, output, batch)
        count = int(batch["state"].shape[0])
        rows += count
        for name in ("sigreg", "variance_covariance"):
            diagnostic_totals[name] += float(pieces[name].detach()) * count
        complete_mask = (
            batch["success_mask"]
            * batch["terminal_event_mask"]
            * batch["terminal_goal_progress_mask"]
            * batch["object_delta_mask"]
        ).to(output["predicted_future_latent"])
        latent_rows = (
            output["predicted_future_latent"]
            - output["target_future_latent"].detach()
        ).square().mean(dim=-1)
        proper_totals["latent_mse"] += float((latent_rows * complete_mask).sum())
        proper_support["latent_mse"] += float(complete_mask.sum())

        success_mask = batch["success_mask"].to(output["success_logit"])
        success_rows = torch.nn.functional.binary_cross_entropy_with_logits(
            output["success_logit"],
            batch["success"].to(output["success_logit"]),
            reduction="none",
        )
        proper_totals["success_binary_nll"] += float(
            (success_rows * success_mask).sum()
        )
        proper_support["success_binary_nll"] += float(success_mask.sum())

        value_target = torch.stack(
            (
                batch["terminal_stage_progress"].to(output["value_mean"]),
                torch.tanh(
                    batch["terminal_goal_progress"].to(output["value_mean"])
                    / wcm.GOAL_PROGRESS_SCALE_METERS
                ),
            ),
            dim=-1,
        )
        value_mask = torch.stack(
            (
                batch["terminal_event_mask"].to(output["value_mean"]),
                batch["terminal_goal_progress_mask"].to(output["value_mean"]),
            ),
            dim=-1,
        )
        value_nll = (
            0.5
            * (
                (value_target - output["value_mean"])
                / output["value_log_scale"].exp()
            ).square()
            + output["value_log_scale"]
        )
        value_row_support = (value_mask.sum(dim=-1) > 0).to(value_nll)
        value_rows = (value_nll * value_mask).sum(dim=-1) / value_mask.sum(
            dim=-1
        ).clamp_min(1.0)
        proper_totals["value_diagonal_gaussian_nll"] += float(
            (value_rows * value_row_support).sum()
        )
        proper_support["value_diagonal_gaussian_nll"] += float(
            value_row_support.sum()
        )

        event_mask = batch["terminal_event_mask"].to(
            output["terminal_event_logits"]
        )
        event_rows = torch.nn.functional.cross_entropy(
            output["terminal_event_logits"],
            batch["terminal_max_event_id"].long(),
            reduction="none",
        )
        proper_totals["terminal_event_categorical_nll"] += float(
            (event_rows * event_mask).sum()
        )
        proper_support["terminal_event_categorical_nll"] += float(event_mask.sum())

        effect_mask = batch["object_delta_mask"].to(output["object_effect_mean"])
        effect_target = torch.tanh(
            batch["object_delta"].to(output["object_effect_mean"])
            / model.object_effect_scales.to(output["object_effect_mean"])
        )
        effect_rows = (
            0.5
            * (
                (effect_target - output["object_effect_mean"])
                / output["object_effect_log_scale"].exp()
            ).square()
            + output["object_effect_log_scale"]
        ).mean(dim=-1)
        proper_totals["object_effect_diagonal_gaussian_nll"] += float(
            (effect_rows * effect_mask).sum()
        )
        proper_support["object_effect_diagonal_gaussian_nll"] += float(
            effect_mask.sum()
        )
        probability = output["success_probability"]
        brier_total += float(
            (
                (probability - batch["success"].to(probability)).square()
                * success_mask
            ).sum()
        )
        brier_support += float(success_mask.sum())
        stage_absolute_total += float(
            (
                output["value_mean"][:, 0]
                - batch["terminal_stage_progress"].to(output["value_mean"])
            ).abs().mul(event_mask).sum()
        )
        stage_support += float(event_mask.sum())
        bounded_goal = torch.tanh(
            batch["terminal_goal_progress"].to(output["value_mean"])
            / wcm.GOAL_PROGRESS_SCALE_METERS
        )
        goal_absolute_total += float(
            (output["value_mean"][:, 1] - bounded_goal)
            .abs()
            .mul(value_mask[:, 1])
            .sum()
        )
        goal_support += float(value_mask[:, 1].sum())
    if rows == 0:
        raise WCMTrainingError("source validation loader is empty")
    required = {
        "latent_mse",
        "success_binary_nll",
        "value_diagonal_gaussian_nll",
        "terminal_event_categorical_nll",
        "object_effect_diagonal_gaussian_nll",
    }
    if any(proper_support[name] <= 0.0 for name in required):
        raise WCMTrainingError("source validation lacks required real branch labels")
    averaged = {
        name: proper_totals[name] / proper_support[name] for name in required
    }
    proper_score = (
        averaged["success_binary_nll"]
        + averaged["value_diagonal_gaussian_nll"]
        + 0.5 * averaged["terminal_event_categorical_nll"]
        + 0.5 * averaged["object_effect_diagonal_gaussian_nll"]
    )
    return {
        "row_count": rows,
        "strict_proper_selection_score": proper_score,
        "success_brier": brier_total / brier_support,
        "terminal_stage_mae": stage_absolute_total / stage_support,
        "bounded_terminal_goal_progress_mae": goal_absolute_total / goal_support,
        "latent_mse_diagnostic_only": averaged["latent_mse"],
        "sigreg_diagnostic_only": diagnostic_totals["sigreg"] / rows,
        "variance_covariance_diagnostic_only": (
            diagnostic_totals["variance_covariance"] / rows
        ),
        "real_label_support": dict(sorted(proper_support.items())),
        "proper_components": {
            name: averaged[name]
            for name in (
                "success_binary_nll",
                "value_diagonal_gaussian_nll",
                "terminal_event_categorical_nll",
                "object_effect_diagonal_gaussian_nll",
            )
        },
        "labels_or_outcomes_used": True,
        "source_validation_only": True,
    }


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[Mapping[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _training_rows_and_receipts(
    audit: Mapping[str, Any],
    supplement_audit: Mapping[str, Any] | None,
    *,
    held_out_body: str,
    split_seed: int,
) -> dict[str, Any]:
    train_groups, validation_groups, _heldout = source.source_group_split(
        audit, held_out_body=held_out_body, split_seed=split_seed
    )
    train_rows = materialize_primary_rows(
        train_groups, held_out_body=held_out_body
    )
    validation_rows = materialize_primary_rows(
        validation_groups, held_out_body=held_out_body
    )
    if supplement_audit is None:
        supplement_train_groups: list[dict[str, Any]] = []
        supplement_validation_groups: list[dict[str, Any]] = []
        supplement_train_rows: list[dict[str, Any]] = []
        supplement_validation_rows: list[dict[str, Any]] = []
    else:
        (
            supplement_train_groups,
            supplement_validation_groups,
            _heldout_supplement,
        ) = source.supplement_source_train_split(
            supplement_audit,
            held_out_body=held_out_body,
            split_seed=split_seed,
        )
        supplement_train_rows = materialize_supplement_rows(
            supplement_train_groups, held_out_body=held_out_body
        )
        supplement_validation_rows = materialize_supplement_rows(
            supplement_validation_groups, held_out_body=held_out_body
        )
    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "supplement_train_rows": supplement_train_rows,
        "supplement_validation_rows": supplement_validation_rows,
        "primary_train_groups": len(train_groups),
        "primary_validation_groups": len(validation_groups),
        "supplement_train_groups": len(supplement_train_groups),
        "supplement_validation_groups": len(supplement_validation_groups),
    }


def train_fold(
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    supplement_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("WCM training output must be one new path")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise WCMTrainingError("real WCM fold training is remote CUDA-only")
    if "4090" not in torch.cuda.get_device_name(0):
        raise WCMTrainingError("real WCM fold training requires the authorized RTX 4090")
    if (
        args.steps <= 0
        or args.eval_every <= 0
        or args.batch_size <= 1
        or args.batch_size % source.CANDIDATE_COUNT
        or args.learning_rate <= 0.0
    ):
        raise WCMTrainingError("matched WCM training budget is invalid")
    source.validate_ensemble_seeds(args.ensemble_seeds)
    preflight = baseline_preflight(
        audit,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
        supplement_audit=supplement_audit,
    )
    materialized = _training_rows_and_receipts(
        audit,
        supplement_audit,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
    )
    train_rows = materialized["train_rows"]
    validation_rows = materialized["validation_rows"]
    supplement_rows = materialized["supplement_train_rows"]
    supplement_validation_rows = materialized["supplement_validation_rows"]
    normalization = fit_primary_source_normalization(
        train_rows, held_out_body=args.held_out_body
    )
    causal_weights, causal_balance_audit = source.source_causal_stratum_proper_weights(
        train_rows,
        source_bodies=preflight["source_bodies"],
        mode=source.DEFAULT_PROPER_BALANCE_MODE,
    )
    causal_group_weight = _balance_group_map(train_rows, causal_weights)
    primary_bootstrap, primary_bootstrap_audit = (
        source.proper_outcome_preserving_group_bootstrap_weights(
            train_rows, members=5, seed=args.split_seed
        )
    )
    if supplement_rows:
        supplement_bootstrap, supplement_bootstrap_audit, supplement_bootstrap_seed = (
            source.supplement_group_bootstrap_weights(
                supplement_rows, members=5, seed=args.split_seed
            )
        )
    else:
        supplement_bootstrap = np.zeros((5, 0), dtype=np.float32)
        supplement_bootstrap_audit = []
        supplement_bootstrap_seed = None
    body_to_id = {body: 0 for body in preflight["source_bodies"]}
    train_dataset = core.TransitionDataset(train_rows, body_to_id)
    validation_loader = DataLoader(
        core.TransitionDataset(validation_rows, body_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    supplement_dataset = (
        core.TransitionDataset(supplement_rows, body_to_id)
        if supplement_rows
        else None
    )
    supplement_validation_loader = (
        DataLoader(
            core.TransitionDataset(supplement_validation_rows, body_to_id),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=core.collate_rows,
        )
        if supplement_validation_rows
        else None
    )
    output.mkdir(parents=True)
    core.atomic_json(output / "preflight_receipt.json", preflight)
    device = torch.device("cuda:0")
    trainer_file_sha256 = wcm.sha256_file(Path(__file__).resolve())
    snapshots = output / "source_validation_common_step_snapshots"
    snapshots.mkdir()
    member_records: list[dict[int, dict[str, Any]]] = []
    member_snapshots: list[dict[int, Path]] = []
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = wcm.WCMFutureLatentBaseline().to(device)
        apply_normalization(model, normalization, device=device)
        budget = wcm.parameter_budget_receipt(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=source.CompleteDecisionBatchSampler(
                train_rows, batch_size=args.batch_size, seed=seed
            ),
            collate_fn=core.collate_rows,
        )
        supplement_loader = (
            DataLoader(
                supplement_dataset,
                batch_sampler=source.CompleteDecisionBatchSampler(
                    supplement_rows, batch_size=args.batch_size, seed=seed
                ),
                collate_fn=core.collate_rows,
            )
            if supplement_dataset is not None
            else None
        )
        primary_group_bootstrap = _group_constant_map(
            train_rows, primary_bootstrap, member=member
        )
        supplement_group_bootstrap = (
            _group_constant_map(supplement_rows, supplement_bootstrap, member=member)
            if supplement_rows
            else {}
        )
        iterator = iter(loader)
        supplement_iterator = iter(supplement_loader) if supplement_loader else None
        records: dict[int, dict[str, Any]] = {}
        paths: dict[int, Path] = {}
        for step in range(1, args.steps + 1):
            primary_raw, iterator = _next_batch(iterator, loader)
            primary_batch = _move_batch(primary_raw, device)
            primary_weight = torch.tensor(
                [
                    primary_group_bootstrap[group]
                    * causal_group_weight[group]
                    for group in primary_raw["logical_group"]
                ],
                device=device,
            )
            primary_output = model(primary_batch)
            primary_loss, primary_pieces = wcm.compute_wcm_loss(
                model,
                primary_output,
                primary_batch,
                sample_weight=primary_weight,
            )
            if supplement_loader is None:
                supplement_loss = primary_loss.new_zeros(())
                supplement_pieces: dict[str, torch.Tensor] = {}
            else:
                assert supplement_iterator is not None
                supplement_raw, supplement_iterator = _next_batch(
                    supplement_iterator, supplement_loader
                )
                supplement_batch = _move_batch(supplement_raw, device)
                supplement_weight = torch.tensor(
                    [
                        supplement_group_bootstrap[group]
                        for group in supplement_raw["logical_group"]
                    ],
                    device=device,
                )
                supplement_output = model(supplement_batch)
                supplement_unweighted, supplement_pieces = wcm.compute_wcm_loss(
                    model,
                    supplement_output,
                    supplement_batch,
                    sample_weight=supplement_weight,
                )
                supplement_loss = SUPPLEMENT_LOSS_WEIGHT * supplement_unweighted
            loss = primary_loss + supplement_loss
            if not bool(torch.isfinite(loss)):
                raise WCMTrainingError("matched WCM training loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if step % args.eval_every and step != args.steps:
                continue
            primary_validation = evaluate_model(model, validation_loader, device)
            supplement_validation = (
                evaluate_model(model, supplement_validation_loader, device)
                if supplement_validation_loader is not None
                else None
            )
            selection_score = primary_validation["strict_proper_selection_score"]
            if supplement_validation is not None:
                selection_score += SUPPLEMENT_LOSS_WEIGHT * supplement_validation[
                    "strict_proper_selection_score"
                ]
            record = {
                "step": step,
                "source_validation": primary_validation,
                "supplement_source_validation": supplement_validation,
                "checkpoint_selection_score": selection_score,
                "checkpoint_selection_uses_only_source_strict_proper": True,
                "train_objective_last": {
                    "total": float(loss.detach()),
                    "primary": {
                        name: float(value.detach())
                        for name, value in primary_pieces.items()
                    },
                    "supplement_fixed_weight": SUPPLEMENT_LOSS_WEIGHT,
                    "supplement": {
                        name: float(value.detach())
                        for name, value in supplement_pieces.items()
                    },
                },
            }
            snapshot = snapshots / f"member_{member:02d}_step_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "member": member,
                    "seed": seed,
                    "step": step,
                    "trainer_file_sha256": trainer_file_sha256,
                },
                snapshot,
            )
            records[step] = record
            paths[step] = snapshot
            model.train()
        if not records:
            raise WCMTrainingError("WCM member produced no validation snapshot")
        member_records.append(records)
        member_snapshots.append(paths)
        del optimizer, model
        torch.cuda.empty_cache()

    common_steps = sorted(set.intersection(*(set(item) for item in member_records)))
    expected_steps = list(range(args.eval_every, args.steps + 1, args.eval_every))
    if not expected_steps or expected_steps[-1] != args.steps:
        expected_steps.append(args.steps)
    if common_steps != expected_steps:
        raise WCMTrainingError("five members lack the exact common validation steps")
    common_selection = []
    for step in common_steps:
        score = float(
            np.mean(
                [records[step]["checkpoint_selection_score"] for records in member_records]
            )
        )
        common_selection.append(
            {
                "step": step,
                "mean_member_source_strict_proper_score": score,
            }
        )
    selected = min(
        common_selection,
        key=lambda row: (
            row["mean_member_source_strict_proper_score"],
            row["step"],
        ),
    )
    selected_step = int(selected["step"])
    members = []
    for member, seed in enumerate(args.ensemble_seeds):
        snapshot = torch.load(
            member_snapshots[member][selected_step],
            map_location="cpu",
            weights_only=True,
        )
        if (
            snapshot.get("member") != member
            or snapshot.get("seed") != seed
            or snapshot.get("step") != selected_step
            or snapshot.get("trainer_file_sha256") != trainer_file_sha256
        ):
            raise WCMTrainingError("selected WCM snapshot binding changed")
        model_for_count = wcm.WCMFutureLatentBaseline()
        parameter_count = wcm.count_trainable_parameters(model_for_count)
        checkpoint_value = {
            "format": wcm.CHECKPOINT_FORMAT,
            "model_family": wcm.MODEL_FAMILY,
            "config": dataclasses.asdict(model_for_count.config),
            "model": snapshot["model"],
            "member": member,
            "seed": seed,
            "step": selected_step,
            "held_out_body": args.held_out_body,
            "source_bodies": preflight["source_bodies"],
            "canonical_state_schema": wcm.STATE_SCHEMA,
            "canonical_action_schema": wcm.ACTION_SCHEMA,
            "state_action_frame_contract": source.state_action_frame_contract(),
            "event_spec_sha256": source.EVENT_SPEC_SHA256,
            "actor_execution_protocol": preflight["primary_preflight"][
                "actor_execution_protocol"
            ],
            "actor_execution_protocol_binding": preflight["primary_preflight"][
                "actor_execution_protocol_binding"
            ],
            "actor_execution_protocol_file_sha256": preflight["primary_preflight"][
                "actor_execution_protocol_file_sha256"
            ],
            "primary_binding_file_sha256": audit["binding_file_sha256"],
            "supplement_binding_file_sha256": (
                supplement_audit["binding_file_sha256"]
                if supplement_audit is not None
                else None
            ),
            "heldout_rows_used_for_training_normalization_or_selection": 0,
            "trainable_parameter_count": parameter_count,
            "rank_score_contract": dict(wcm.RANK_SCORE_CONTRACT),
            "trainer_file_sha256": trainer_file_sha256,
            "preflight_logical_sha256": preflight["logical_sha256"],
            "normalization": normalization,
            "validation": member_records[member][selected_step],
        }
        checkpoint_path = output / f"member_{member:02d}_seed_{seed}_best.pt"
        torch.save(checkpoint_value, checkpoint_path)
        # Load through the independent production loader before authority is
        # written, so an unusable checkpoint can never enter the summary.
        wcm.load_member_checkpoint(checkpoint_path, map_location="cpu")
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": selected_step,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": wcm.sha256_file(checkpoint_path),
                "source_validation": member_records[member][selected_step],
            }
        )
    for paths in member_snapshots:
        for path in paths.values():
            path.unlink()
    snapshots.rmdir()
    summary = {
        "format": SUMMARY_FORMAT,
        "status": "five_member_source_only_common_step_complete",
        "model_family": wcm.MODEL_FAMILY,
        "not_official_wcm_architecture_or_weights": True,
        "held_out_body": args.held_out_body,
        "source_bodies": preflight["source_bodies"],
        "canonical_state_schema": wcm.STATE_SCHEMA,
        "canonical_action_schema": wcm.ACTION_SCHEMA,
        "state_action_frame_contract": source.state_action_frame_contract(),
        "event_spec_sha256": source.EVENT_SPEC_SHA256,
        "actor_execution_protocol": preflight["primary_preflight"][
            "actor_execution_protocol"
        ],
        "actor_execution_protocol_binding": preflight["primary_preflight"][
            "actor_execution_protocol_binding"
        ],
        "actor_execution_protocol_file_sha256": preflight["primary_preflight"][
            "actor_execution_protocol_file_sha256"
        ],
        "primary_binding_file_sha256": audit["binding_file_sha256"],
        "supplement_binding_file_sha256": (
            supplement_audit["binding_file_sha256"]
            if supplement_audit is not None
            else None
        ),
        "training_budget": {
            "steps_per_member": args.steps,
            "eval_every_steps": args.eval_every,
            "batch_size_rows": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "ensemble_members": len(args.ensemble_seeds),
        },
        "parameter_budget": preflight["parameter_budget"],
        "normalization": normalization,
        "normalization_fit": "primary_source_train_only",
        "causal_source_proper_balance": causal_balance_audit,
        "primary_group_bootstrap": primary_bootstrap_audit,
        "supplement_group_bootstrap": supplement_bootstrap_audit,
        "supplement_group_bootstrap_seed": supplement_bootstrap_seed,
        "supplement": {
            "enabled": supplement_audit is not None,
            "source_train_rows": len(supplement_rows),
            "source_validation_rows": len(supplement_validation_rows),
            "train_loss_weight": SUPPLEMENT_LOSS_WEIGHT if supplement_rows else 0.0,
            "validation_selection_weight": (
                SUPPLEMENT_LOSS_WEIGHT if supplement_validation_rows else 0.0
            ),
            "normalization_rows_used": 0,
            "heldout_manifest_or_payload_opened": 0,
        },
        "future_target_schema": list(wcm.FUTURE_TARGET_SCHEMA),
        "raw_future_observation_or_image_available": False,
        "future_target_claim": "canonical_terminal_consequence_and_object_effect_latent",
        "loss_contract": {
            "prediction": "future_latent_mse_to_detached_live_target_encoder",
            "proper": "binary_success_plus_diagonal_gaussian_value_plus_event_and_effect",
            "anti_collapse": "batch_SIGReg_plus_variance_covariance",
        },
        "rank_score_contract": dict(wcm.RANK_SCORE_CONTRACT),
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "selected_step": selected_step,
            "selected_mean_member_source_strict_proper_score": selected[
                "mean_member_source_strict_proper_score"
            ],
            "evaluated_common_steps": common_selection,
            "heldout_rows_used": 0,
        },
        "members": members,
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "body_or_condition_trainable_input": False,
        "actor_frozen": True,
        "task_success_evaluation_authorized": False,
        "trainer_file_sha256": trainer_file_sha256,
        "preflight": preflight,
    }
    summary["logical_sha256"] = wcm.canonical_sha256(summary)
    core.atomic_json(output / "training_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train-fold"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--supplement-binding", type=Path)
    parser.add_argument("--supplement-binding-sha256")
    parser.add_argument("--held-out-body", choices=source.BODIES, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--ensemble-seeds", nargs=5, type=int, default=list(DEFAULT_ENSEMBLE_SEEDS)
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source.validate_ensemble_seeds(args.ensemble_seeds)
    audit = source.load_binding(args.binding, args.binding_sha256)
    if (args.supplement_binding is None) != (
        args.supplement_binding_sha256 is None
    ):
        raise WCMTrainingError(
            "supplement binding path and SHA-256 must be supplied together"
        )
    supplement_audit = (
        source.load_supplement_binding(
            args.supplement_binding,
            args.supplement_binding_sha256,
            primary_audit=audit,
            held_out_body=args.held_out_body,
        )
        if args.supplement_binding is not None
        else None
    )
    if args.mode == "preflight":
        receipt = baseline_preflight(
            audit,
            held_out_body=args.held_out_body,
            split_seed=args.split_seed,
            supplement_audit=supplement_audit,
        )
        print("WCM_PREFLIGHT=" + json.dumps(receipt, sort_keys=True))
        return 0
    if args.output is None:
        raise WCMTrainingError("train-fold requires --output")
    result = train_fold(args, audit, supplement_audit)
    print("WCM_TRAINING=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENSEMBLE_SEEDS",
    "DEFAULT_EVAL_EVERY",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_STEPS",
    "FORMAT",
    "SUMMARY_FORMAT",
    "TRAINING_CONTRACT",
    "WCMTrainingError",
    "apply_normalization",
    "baseline_preflight",
    "evaluate_model",
    "fit_primary_source_normalization",
    "main",
    "materialize_primary_rows",
    "materialize_supplement_rows",
    "parse_args",
    "train_fold",
]
