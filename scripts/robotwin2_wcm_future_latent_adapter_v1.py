#!/usr/bin/env python3
"""Signed fold/runtime adapter for the matched WCM-style baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import robotwin2_wcm_future_latent_baseline_v1 as wcm
import train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1 as trainer


FORMAT = "etsf_robotwin2_wcm_future_latent_adapter_v1"
LOAD_RECEIPT_FORMAT = "etsf_robotwin2_wcm_fold_ensemble_load_receipt_v1"
RANK_RECEIPT_FORMAT = "etsf_robotwin2_nested_wcm_rank_receipt_v1"
SOURCE_ACTION_NORMALIZER_FORMAT = (
    "etsf_robotwin2_fold_source_action_normalizer_binding_v1"
)
ENSEMBLE_SIZE = 5
METHOD_CANDIDATE_COUNTS = {
    "etsf_nested_best_of_4_from_raw16": 4,
    "etsf_nested_best_of_8_from_raw16": 8,
}


class WCMAdapterError(RuntimeError):
    pass


def _read(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise WCMAdapterError(f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WCMAdapterError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise WCMAdapterError(f"{label} must be an object")
    unsigned = {key: item for key, item in value.items() if key != "logical_sha256"}
    if value.get("logical_sha256") != wcm.canonical_sha256(unsigned):
        raise WCMAdapterError(f"{label} logical SHA changed")
    return value


def _sha(path: Path) -> str:
    try:
        return wcm.sha256_file(path)
    except wcm.WCMBaselineError as error:
        raise WCMAdapterError(str(error)) from error


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(value)
    if "logical_sha256" in base:
        raise WCMAdapterError("cannot sign an already signed value")
    return {**base, "logical_sha256": wcm.canonical_sha256(base)}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def inspect_fold(
    body: str,
    fold_root: Path,
    *,
    expected_actor_protocol_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if body not in wcm.BODIES:
        raise WCMAdapterError("unknown WCM held-out body")
    root = fold_root.expanduser().resolve()
    summary_path = root / "training_summary.json"
    preflight_path = root / "preflight_receipt.json"
    summary = _read(summary_path, f"{body} WCM summary")
    preflight = _read(preflight_path, f"{body} WCM preflight")
    source_bodies = [candidate for candidate in wcm.BODIES if candidate != body]
    members = summary.get("members")
    selection = summary.get("ensemble_checkpoint_selection")
    budget = summary.get("training_budget")
    if (
        summary.get("format") != trainer.SUMMARY_FORMAT
        or summary.get("status") != "five_member_source_only_common_step_complete"
        or summary.get("model_family") != wcm.MODEL_FAMILY
        or summary.get("held_out_body") != body
        or summary.get("source_bodies") != source_bodies
        or summary.get("actor_execution_protocol_binding")
        != expected_actor_protocol_binding
        or summary.get("actor_execution_protocol")
        != expected_actor_protocol_binding.get("protocol")
        or summary.get("actor_execution_protocol_file_sha256")
        != expected_actor_protocol_binding.get("file_sha256")
        or summary.get("state_action_frame_contract")
        != wcm.STATE_ACTION_FRAME_CONTRACT
        or summary.get("event_spec_sha256") != wcm.EVENT_SPEC_SHA256
        or summary.get("trainer_file_sha256")
        != _sha(Path(trainer.__file__).resolve())
        or summary.get("heldout_group_npz_opened") != 0
        or summary.get("heldout_group_payload_bytes_read") != 0
        or summary.get("heldout_group_payload_deserialized") != 0
        or summary.get(
            "heldout_labels_used_for_normalization_training_or_selection"
        )
        is not False
        or not isinstance(selection, Mapping)
        or selection.get("common_step_required_for_all_five_members") is not True
        or selection.get("heldout_rows_used") != 0
        or type(selection.get("selected_step")) is not int
        or not isinstance(budget, Mapping)
        or budget.get("steps_per_member") != trainer.DEFAULT_STEPS
        or budget.get("ensemble_members") != ENSEMBLE_SIZE
        or not isinstance(members, list)
        or len(members) != ENSEMBLE_SIZE
        or preflight.get("format") != trainer.FORMAT
        or preflight.get("status")
        != "matched_wcm_preflight_passed_payloads_still_unopened"
        or preflight.get("held_out_body") != body
        or preflight.get("source_bodies") != source_bodies
        or preflight.get("heldout_group_npz_opened") != 0
        or preflight.get("heldout_group_payload_bytes_read") != 0
    ):
        raise WCMAdapterError(f"{body} WCM fold contract changed")
    normalized = []
    paths = []
    for index, item in enumerate(members):
        if not isinstance(item, Mapping):
            raise WCMAdapterError("WCM member is invalid")
        path = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise WCMAdapterError("WCM checkpoint escapes fold root") from error
        if (
            item.get("member") != index
            or item.get("seed") != trainer.DEFAULT_ENSEMBLE_SEEDS[index]
            or item.get("best_step") != selection["selected_step"]
            or not path.is_file()
            or path.is_symlink()
            or _sha(path) != item.get("checkpoint_sha256")
        ):
            raise WCMAdapterError("WCM member binding changed")
        paths.append(path)
        normalized.append(
            {
                "member": index,
                "seed": item["seed"],
                "checkpoint": str(path),
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
        )
    return {
        "critic_kind": "wcm",
        "heldout_body": body,
        "source_bodies": source_bodies,
        "body_adapter": "none_body_identity_not_an_input",
        "state_action_frame_contract": dict(wcm.STATE_ACTION_FRAME_CONTRACT),
        "actor_execution_protocol": expected_actor_protocol_binding["protocol"],
        "actor_execution_protocol_binding": dict(expected_actor_protocol_binding),
        "actor_execution_protocol_file_sha256": expected_actor_protocol_binding[
            "file_sha256"
        ],
        "fold_root": str(root),
        "training_summary": str(summary_path),
        "training_summary_sha256": _sha(summary_path),
        "training_summary_logical_sha256": summary["logical_sha256"],
        "preflight_receipt": str(preflight_path),
        "preflight_receipt_sha256": _sha(preflight_path),
        "preflight_receipt_logical_sha256": preflight["logical_sha256"],
        "ensemble_common_selection_step": selection["selected_step"],
        "primary_binding_file_sha256": summary["primary_binding_file_sha256"],
        "supplement_binding_file_sha256": summary[
            "supplement_binding_file_sha256"
        ],
        "members": normalized,
    }


def inspect_source_action_normalizer(fold: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in fold.get("members", []):
        try:
            model, checkpoint = wcm.load_member_checkpoint(
                Path(str(item["checkpoint"])), map_location="cpu"
            )
        except (wcm.WCMBaselineError, RuntimeError, ValueError) as error:
            raise WCMAdapterError(str(error)) from error
        normalization = checkpoint["normalization"]
        mean = np.asarray(normalization["action_mean"], dtype=np.float32)
        std = np.asarray(normalization["action_std"], dtype=np.float32)
        if (
            mean.shape != (wcm.ACTION_DIM,)
            or std.shape != (wcm.ACTION_DIM,)
            or not np.array_equal(model.action_mean.detach().cpu().numpy(), mean)
            or not np.array_equal(model.action_std.detach().cpu().numpy(), std)
        ):
            raise WCMAdapterError("WCM model/action normalization disagrees")
        rows.append(
            {
                "member": int(item["member"]),
                "normalization_sha256": normalization["logical_sha256"],
                "action_mean": mean.astype(float).tolist(),
                "action_std": std.astype(float).tolist(),
            }
        )
    if len(rows) != ENSEMBLE_SIZE or any(
        row != {**rows[0], "member": row["member"]} for row in rows
    ):
        raise WCMAdapterError("five WCM members do not share one normalizer")
    base = {
        "format": SOURCE_ACTION_NORMALIZER_FORMAT,
        "heldout_body": fold["heldout_body"],
        "canonical_action_schema": wcm.ACTION_SCHEMA,
        "normalization_fit_scope": "four_source_bodies_train_only",
        "heldout_rows_used": 0,
        "checkpoint_action_normalization_sha256": rows[0][
            "normalization_sha256"
        ],
        "action_mean": rows[0]["action_mean"],
        "action_std": rows[0]["action_std"],
        "normalization_clip": wcm.STANDARDIZED_INPUT_CLIP,
        "five_member_normalizers_bit_exact_equal": True,
        "selection_reads_heldout_labels_outcomes_or_critic_scores": False,
    }
    return signed(base)


def inspect_training_regime(
    folds: Mapping[str, Mapping[str, Any]],
    *,
    required_supplement_binding_sha256: str | None,
) -> dict[str, Any]:
    if set(folds) != set(wcm.BODIES):
        raise WCMAdapterError("WCM regime requires five folds")
    rows = {}
    regimes = set()
    for body in wcm.BODIES:
        summary = _read(
            Path(str(folds[body]["training_summary"])), f"{body} WCM summary"
        )
        supplement = summary.get("supplement")
        selection = summary.get("ensemble_checkpoint_selection")
        if not isinstance(supplement, Mapping) or not isinstance(selection, Mapping):
            raise WCMAdapterError("WCM supplement/selection contract is missing")
        enabled = supplement.get("enabled") is True
        sha = summary.get("supplement_binding_file_sha256") if enabled else None
        if (
            selection.get("heldout_rows_used") != 0
            or supplement.get("normalization_rows_used") != 0
            or supplement.get("heldout_manifest_or_payload_opened") != 0
        ):
            raise WCMAdapterError("WCM supplement regime changed")
        regimes.add((enabled, sha))
        rows[body] = {
            "supplement_enabled": enabled,
            "supplement_binding_file_sha256": sha,
            "selected_step": selection["selected_step"],
            "heldout_rows_used": 0,
        }
    if len(regimes) != 1:
        raise WCMAdapterError("five WCM folds mix training regimes")
    enabled, sha = next(iter(regimes))
    if required_supplement_binding_sha256 is not None and (
        not enabled or sha != required_supplement_binding_sha256
    ):
        raise WCMAdapterError("WCM folds do not use required supplement")
    return signed(
        {
            "name": "wcm_c_plus_expert_root_supplement" if enabled else "wcm_c_only",
            "critic_kind": "wcm",
            "supplement_enabled": enabled,
            "supplement_binding_file_sha256": sha,
            "required_supplement_binding_sha256": required_supplement_binding_sha256,
            "checkpoint_selection_scope": "source_validation_only",
            "folds": rows,
        }
    )


def load_fold_ensemble(
    fold: Mapping[str, Any], *, device: torch.device
) -> tuple[wcm.WCMFutureLatentEnsemble, dict[str, Any]]:
    paths = [Path(str(item["checkpoint"])) for item in fold["members"]]
    try:
        ensemble, checkpoints = wcm.load_five_member_ensemble(
            paths, map_location=device
        )
    except (wcm.WCMBaselineError, RuntimeError, ValueError) as error:
        raise WCMAdapterError(str(error)) from error
    members = [
        {
            "member": checkpoint["member"],
            "seed": checkpoint["seed"],
            "step": checkpoint["step"],
            "checkpoint": str(paths[index].resolve()),
            "checkpoint_sha256": fold["members"][index]["checkpoint_sha256"],
            "normalization_logical_sha256": checkpoint["normalization"][
                "logical_sha256"
            ],
        }
        for index, checkpoint in enumerate(checkpoints)
    ]
    receipt = signed(
        {
            "format": LOAD_RECEIPT_FORMAT,
            "critic_kind": "wcm",
            "model_family": wcm.MODEL_FAMILY,
            "held_out_body": fold["heldout_body"],
            "source_bodies": fold["source_bodies"],
            "common_selection_step": fold["ensemble_common_selection_step"],
            "training_summary": fold["training_summary"],
            "training_summary_sha256": fold["training_summary_sha256"],
            "training_summary_logical_sha256": fold[
                "training_summary_logical_sha256"
            ],
            "members": members,
            "rank_score_contract": dict(wcm.RANK_SCORE_CONTRACT),
            "heldout_payloads_or_labels_opened": 0,
        }
    )
    return ensemble, receipt


def validate_runtime_action_normalizer(
    ensemble: wcm.WCMFutureLatentEnsemble,
    binding: Mapping[str, Any],
) -> None:
    mean = np.asarray(binding.get("action_mean"), dtype=np.float32)
    std = np.asarray(binding.get("action_std"), dtype=np.float32)
    unsigned = {key: item for key, item in binding.items() if key != "logical_sha256"}
    if (
        binding.get("format") != SOURCE_ACTION_NORMALIZER_FORMAT
        or binding.get("logical_sha256") != wcm.canonical_sha256(unsigned)
        or mean.shape != (wcm.ACTION_DIM,)
        or std.shape != (wcm.ACTION_DIM,)
        or float(binding.get("normalization_clip", -1))
        != wcm.STANDARDIZED_INPUT_CLIP
        or len(ensemble.models) != ENSEMBLE_SIZE
    ):
        raise WCMAdapterError("WCM runtime normalizer binding changed")
    for model in ensemble.models:
        if (
            not np.array_equal(model.action_mean.detach().cpu().numpy(), mean)
            or not np.array_equal(model.action_std.detach().cpu().numpy(), std)
        ):
            raise WCMAdapterError("WCM runtime normalizer differs from checkpoint")


def runtime_contract(candidate_count: int) -> dict[str, Any]:
    if candidate_count not in wcm.CANDIDATE_COUNTS:
        raise WCMAdapterError("WCM runtime candidate count must be 4 or 8")
    return {
        "format": "etsf_robotwin2_wcm_future_latent_runtime_v1",
        "candidate_count": candidate_count,
        "rank_score_contract": dict(wcm.RANK_SCORE_CONTRACT),
        "ensemble_size": ENSEMBLE_SIZE,
        "candidate_outcomes_read": False,
    }


def score_candidates(
    ensemble: wcm.WCMFutureLatentEnsemble,
    batch: Mapping[str, Any],
    *,
    candidate_count: int,
    method: str,
    load_receipt: Mapping[str, Any],
    candidate_roster_sha256: str,
) -> dict[str, Any]:
    unsigned_load = {
        key: item for key, item in load_receipt.items() if key != "logical_sha256"
    }
    if (
        METHOD_CANDIDATE_COUNTS.get(method) != candidate_count
        or
        load_receipt.get("format") != LOAD_RECEIPT_FORMAT
        or load_receipt.get("logical_sha256") != wcm.canonical_sha256(unsigned_load)
        or load_receipt.get("heldout_payloads_or_labels_opened") != 0
        or load_receipt.get("critic_kind") != "wcm"
        or load_receipt.get("model_family") != wcm.MODEL_FAMILY
        or len(load_receipt.get("members", [])) != ENSEMBLE_SIZE
        or not _is_sha256(candidate_roster_sha256)
    ):
        raise WCMAdapterError("WCM load receipt changed")
    runtime = dict(batch)
    reference = runtime.get("state")
    if not isinstance(reference, torch.Tensor):
        raise WCMAdapterError("WCM runtime batch lacks state")
    runtime["candidate_index"] = torch.arange(
        candidate_count, device=reference.device
    )
    runtime["logical_group"] = ["label-free-runtime-root"] * candidate_count
    try:
        raw = wcm.score_candidate_pool(
            ensemble, runtime, candidate_count=candidate_count
        )
    except wcm.WCMBaselineError as error:
        raise WCMAdapterError(str(error)) from error
    members = raw["candidate_rank_score_members"].numpy()
    aggregate = raw["candidate_rank_score_epistemic_lcb_ensemble"].numpy()
    result = {
        "critic_kind": "wcm",
        "candidate_rank_score_members": members.astype(float).tolist(),
        "candidate_rank_score_epistemic_lcb_ensemble": aggregate.astype(float).tolist(),
        "candidate_rank_score_mean": members.mean(axis=0).astype(float).tolist(),
        "candidate_rank_score_population_std": members.std(axis=0).astype(float).tolist(),
        "candidate_rank_score_raw_member_candidate_mean": members.mean(axis=1).astype(float).tolist(),
        "candidate_rank_score_raw_member_candidate_population_std": members.std(axis=1).astype(float).tolist(),
        "selected_candidate_index": int(raw["selected_candidate_index"]),
        "candidate_roster_sha256": candidate_roster_sha256,
        "wcm_runtime_contract": runtime_contract(candidate_count),
    }
    receipt = signed(
        {
            "format": RANK_RECEIPT_FORMAT,
            "critic_kind": "wcm",
            "method": method,
            "candidate_count": candidate_count,
            "wcm_runtime_contract": runtime_contract(candidate_count),
            "critic_ensemble_load_receipt": dict(load_receipt),
            "critic_ensemble_load_receipt_logical_sha256": load_receipt[
                "logical_sha256"
            ],
            "candidate_roster_sha256": candidate_roster_sha256,
            "candidate_rank_score_members_sha256": wcm.canonical_sha256(
                result["candidate_rank_score_members"]
            ),
            "risk_adjusted_candidate_score_sha256": wcm.canonical_sha256(
                result["candidate_rank_score_epistemic_lcb_ensemble"]
            ),
            "candidate_rank_score_mean_sha256": wcm.canonical_sha256(
                result["candidate_rank_score_mean"]
            ),
            "candidate_rank_score_population_std_sha256": wcm.canonical_sha256(
                result["candidate_rank_score_population_std"]
            ),
            "selected_candidate_index": result["selected_candidate_index"],
            "heldout_labels_or_outcomes_read_for_ranking": False,
        }
    )
    return {**result, "rank_receipt": receipt}


def validate_rank_receipt(
    scores: Mapping[str, Any],
    *,
    method: str,
    candidate_count: int,
    expected_heldout_body: str,
    expected_candidate_roster_sha256: str,
) -> None:
    receipt = scores.get("rank_receipt")
    if not isinstance(receipt, Mapping):
        raise WCMAdapterError("WCM rank receipt is missing")
    unsigned = {key: item for key, item in receipt.items() if key != "logical_sha256"}
    load = receipt.get("critic_ensemble_load_receipt")
    if not isinstance(load, Mapping):
        raise WCMAdapterError("WCM rank receipt lacks load receipt")
    load_unsigned = {key: item for key, item in load.items() if key != "logical_sha256"}
    try:
        members = torch.as_tensor(scores.get("candidate_rank_score_members"))
        aggregate = torch.as_tensor(
            scores.get("candidate_rank_score_epistemic_lcb_ensemble")
        )
        recorded_mean = torch.as_tensor(scores.get("candidate_rank_score_mean"))
        recorded_std = torch.as_tensor(
            scores.get("candidate_rank_score_population_std")
        )
        replayed = wcm.aggregate_epistemic_lcb(members)
    except (TypeError, ValueError, wcm.WCMBaselineError) as error:
        raise WCMAdapterError(str(error)) from error
    if (
        METHOD_CANDIDATE_COUNTS.get(method) != candidate_count
        or receipt.get("format") != RANK_RECEIPT_FORMAT
        or receipt.get("logical_sha256") != wcm.canonical_sha256(unsigned)
        or receipt.get("critic_kind") != "wcm"
        or receipt.get("method") != method
        or receipt.get("candidate_count") != candidate_count
        or receipt.get("wcm_runtime_contract") != runtime_contract(candidate_count)
        or scores.get("critic_kind") != "wcm"
        or scores.get("wcm_runtime_contract") != runtime_contract(candidate_count)
        or load.get("logical_sha256") != wcm.canonical_sha256(load_unsigned)
        or load.get("format") != LOAD_RECEIPT_FORMAT
        or load.get("critic_kind") != "wcm"
        or load.get("model_family") != wcm.MODEL_FAMILY
        or len(load.get("members", [])) != ENSEMBLE_SIZE
        or load.get("held_out_body") != expected_heldout_body
        or load.get("heldout_payloads_or_labels_opened") != 0
        or receipt.get("critic_ensemble_load_receipt_logical_sha256")
        != load.get("logical_sha256")
        or receipt.get("candidate_roster_sha256")
        != expected_candidate_roster_sha256
        or scores.get("candidate_roster_sha256")
        != expected_candidate_roster_sha256
        or not _is_sha256(expected_candidate_roster_sha256)
        or receipt.get("candidate_rank_score_members_sha256")
        != wcm.canonical_sha256(scores.get("candidate_rank_score_members"))
        or receipt.get("risk_adjusted_candidate_score_sha256")
        != wcm.canonical_sha256(
            scores.get("candidate_rank_score_epistemic_lcb_ensemble")
        )
        or receipt.get("candidate_rank_score_mean_sha256")
        != wcm.canonical_sha256(scores.get("candidate_rank_score_mean"))
        or receipt.get("candidate_rank_score_population_std_sha256")
        != wcm.canonical_sha256(
            scores.get("candidate_rank_score_population_std")
        )
        or receipt.get("selected_candidate_index")
        != scores.get("selected_candidate_index")
        or receipt.get("heldout_labels_or_outcomes_read_for_ranking") is not False
        or members.shape != (ENSEMBLE_SIZE, candidate_count)
        or aggregate.shape != (candidate_count,)
        or recorded_mean.shape != (candidate_count,)
        or recorded_std.shape != (candidate_count,)
        or not torch.allclose(aggregate, replayed, atol=1e-6, rtol=0.0)
        or not torch.allclose(
            recorded_mean, members.mean(dim=0), atol=1e-6, rtol=0.0
        )
        or not torch.allclose(
            recorded_std,
            members.std(dim=0, correction=0),
            atol=1e-6,
            rtol=0.0,
        )
        or int(torch.argmax(replayed)) != scores.get("selected_candidate_index")
    ):
        raise WCMAdapterError("WCM rank receipt cannot be replayed")


__all__ = [
    "ENSEMBLE_SIZE",
    "FORMAT",
    "LOAD_RECEIPT_FORMAT",
    "RANK_RECEIPT_FORMAT",
    "SOURCE_ACTION_NORMALIZER_FORMAT",
    "WCMAdapterError",
    "inspect_fold",
    "inspect_source_action_normalizer",
    "inspect_training_regime",
    "load_fold_ensemble",
    "runtime_contract",
    "score_candidates",
    "validate_rank_receipt",
    "validate_runtime_action_normalizer",
]
