#!/usr/bin/env python3
"""Evaluate a frozen ETSF counterfactual ensemble exactly once on sealed data.

The evaluator has no threshold/scoring override flags.  It loads the validation-
frozen temperature, score coefficients and guard from ``ensemble_manifest.json``.
The output marker is reserved with O_EXCL before any sealed manifest, file hash,
or label dataset is read; a completed or interrupted evaluation therefore cannot
silently be repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from openvla_etsf_event_critic_plugin import EventCriticPlugin
from openvla_etsf_oof_final_contract import (
    OOF_TEST_POLICY,
    validate_authorized_oof_final,
)
from train_openvla_etsf_counterfactual import (
    BranchGroup,
    GroupDescriptor,
    candidate_rank_score,
    collate_groups,
    forward_model,
    load_descriptor_groups,
    lognormal_nll_per_item,
    move_batch,
    scan_group_descriptors,
)


FORMAT = "etsf_counterfactual_ensemble_v1"
EVALUATION_FORMAT = "etsf_counterfactual_sealed_evaluation_v1"
SCHEMA_VERSION = 5
LANGUAGE_CONTRACT = "same_instruction_for_initial_query_and_all_candidate_branches"
INTERVENTION = "candidate_first_chunk_then_deterministic_actor"
POST_QUERY_ACTION_CONTRACT = "executed_as_next_query_when_nonterminal"
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_REPLICATES = 10000
LEGACY_TEST_POLICY = (
    "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_is_within(path: Path, root: Path) -> bool:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def validate_pre_reservation_paths(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sealed_root: Path,
) -> None:
    """Prevent checkpoint provenance from aliasing the sealed collection."""

    artifacts: list[tuple[str, str]] = []
    aggregate = manifest.get("ensemble_checkpoint")
    if isinstance(aggregate, Mapping) and aggregate.get("path"):
        artifacts.append(("aggregate checkpoint", str(aggregate["path"])))
    members = manifest.get("members")
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        for index, member in enumerate(members):
            if isinstance(member, Mapping) and member.get("path"):
                artifacts.append((f"member checkpoint {index}", str(member["path"])))
    contract = manifest.get("contract")
    if isinstance(contract, Mapping):
        for key, name in (
            ("oof_selection", "OOF selection"),
            ("oof_manifest", "OOF preregistration manifest"),
        ):
            if contract.get(key):
                artifacts.append((name, str(contract[key])))
    for name, recorded_value in artifacts:
        recorded = Path(recorded_value).expanduser()
        portable = manifest_path.parent / recorded.name
        selected = recorded if recorded.is_file() else portable
        if path_is_within(selected, sealed_root):
            raise RuntimeError(
                f"{name} must be outside the sealed data root before one-shot reservation"
            )


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reserve_evaluated_once(path: Path, request: Mapping[str, Any]) -> None:
    """Atomically reserve the only allowed sealed-label access attempt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "format": EVALUATION_FORMAT,
            "status": "sealed_label_access_reserved",
            "request": dict(request),
            "recovery_policy": "automatic_rerun_forbidden_even_after_interruption",
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"sealed evaluation already reserved or completed: {path}; refusing overwrite"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _as_string_set(value: Any, name: str) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"ensemble contract {name} must be a sequence")
    result = {str(item) for item in value}
    if len(result) != len(value):
        raise RuntimeError(f"ensemble contract {name} contains duplicates")
    return result


def _fresh_seed_lists(manifest: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    requested = manifest.get(
        "selected_requested_seeds", manifest.get("requested_seeds")
    )
    resolved = manifest.get(
        "selected_resolved_seeds", manifest.get("resolved_seeds")
    )
    selected = manifest.get("selected_seeds", manifest.get("test"))
    if selected is not None:
        if not isinstance(selected, list):
            raise RuntimeError("fresh selected_seeds must be a list")
        requested = [
            int(item.get("requested_seed", item.get("seed")))
            if isinstance(item, Mapping)
            else int(item)
            for item in selected
        ]
        resolved = [
            int(item.get("resolved_seed", item.get("seed")))
            if isinstance(item, Mapping)
            else int(item)
            for item in selected
        ]
    if not isinstance(requested, list) or not isinstance(resolved, list):
        raise RuntimeError(
            "frozen fresh manifest lacks selected requested/resolved seed lists"
        )
    requested_values = [int(value) for value in requested]
    resolved_values = [int(value) for value in resolved]
    if (
        len(requested_values) != 50
        or len(resolved_values) != 50
        or len(set(requested_values)) != 50
        or len(set(resolved_values)) != 50
    ):
        raise RuntimeError("confirmatory fresh manifest must freeze exactly 50 unique scenes")
    return requested_values, resolved_values


def classify_evaluation_protocol(
    root_manifest: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any] | None,
    fresh_manifest_sha256: str | None,
) -> dict[str, Any]:
    registry = str(root_manifest.get("seed_registry", ""))
    if registry == "official_150":
        if fresh_manifest is not None:
            raise RuntimeError(
                "a fresh seed manifest cannot upgrade official test data to confirmatory"
            )
        if root_manifest.get("fresh_seed_manifest_sha256") not in (None, ""):
            raise RuntimeError("official seed registry unexpectedly binds a fresh manifest")
        return {
            "seed_registry": registry,
            "evidence_tier": "development_holdout",
            "confirmatory": False,
            "reason": "original_official_test_is_not_untouched_after_protocol_incident",
            "fresh_seed_manifest": None,
        }
    if registry != "explicit_fresh_confirmation":
        raise RuntimeError(
            "sealed collection must declare seed_registry as official_150 or "
            "explicit_fresh_confirmation"
        )
    if fresh_manifest is None or fresh_manifest_sha256 is None:
        raise RuntimeError(
            "explicit fresh confirmation requires --fresh-seed-manifest"
        )
    status = str(fresh_manifest.get("status", "")).lower()
    if status != "fresh_confirmation_preregistered_resolved":
        raise RuntimeError("fresh confirmation manifest is not resolved and frozen")
    if str(fresh_manifest.get("task", "")) != str(root_manifest.get("task", "")):
        raise RuntimeError("fresh manifest task differs from sealed collection")
    requested, resolved = _fresh_seed_lists(fresh_manifest)
    root_requested = [int(value) for value in root_manifest.get("requested_seeds", [])]
    root_resolved = [int(value) for value in root_manifest.get("resolved_seeds", [])]
    if root_requested != requested or root_resolved != resolved:
        raise RuntimeError(
            "sealed collection seeds differ from the frozen fresh-50 manifest"
        )
    if str(root_manifest.get("fresh_seed_manifest_sha256", "")) != fresh_manifest_sha256:
        raise RuntimeError(
            "collector did not bind outcomes to the frozen fresh seed manifest SHA256"
        )
    if not str(root_manifest.get("fresh_seed_manifest", "")):
        raise RuntimeError("collector did not record the frozen fresh manifest path")
    return {
        "seed_registry": registry,
        "evidence_tier": "fresh_confirmatory",
        "confirmatory": True,
        "reason": "exact_frozen_fresh50_seed_contract",
        "fresh_seed_manifest": {
            "sha256": fresh_manifest_sha256,
            "status": fresh_manifest.get("status"),
            "requested_seeds": requested,
            "resolved_seeds": resolved,
        },
    }


def validate_frozen_split_contract(
    contract: Mapping[str, Any],
    descriptors: Sequence[GroupDescriptor],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    oof_final = contract.get("development_protocol") == (
        "etsf_counterfactual_five_fold_oof_v1"
    )
    expected_access = (
        "fresh50_absent_not_read_one_shot_only_after_oof_authorization"
        if oof_final
        else "identity_attrs_and_raw_file_sha256_only_no_label_datasets"
    )
    if contract.get("sealed_test_access") != expected_access:
        raise RuntimeError("training contract does not prove sealed labels were held out")
    train = _as_string_set(contract.get("train_groups"), "train_groups")
    validation = _as_string_set(
        contract.get("validation_groups"), "validation_groups"
    )
    sealed = _as_string_set(
        contract.get("sealed_test_groups"), "sealed_test_groups"
    )
    if not train or (not oof_final and (not validation or not sealed)):
        raise RuntimeError("frozen train/validation/sealed splits must all be non-empty")
    if oof_final and (validation or sealed or not bool(protocol["confirmatory"])):
        raise RuntimeError("OOF final is authorized only for fresh confirmatory data")
    if train & validation or train & sealed or validation & sealed:
        raise RuntimeError("logical-key leakage in frozen ensemble split contract")
    actual = {descriptor.logical_key for descriptor in descriptors}
    if bool(protocol["confirmatory"]):
        if actual & train or actual & validation or actual & sealed:
            raise RuntimeError(
                "fresh confirmation groups overlap a model-development split"
            )
    elif actual != sealed:
        raise RuntimeError(
            "official development holdout does not exactly match frozen test keys: "
            f"missing={sorted(sealed - actual)}, unexpected={sorted(actual - sealed)}"
        )
    train_seeds = {key.rsplit("|", 1)[-1] for key in train}
    validation_seeds = {key.rsplit("|", 1)[-1] for key in validation}
    sealed_seeds = {str(descriptor.seed) for descriptor in descriptors}
    old_sealed_seeds = {key.rsplit("|", 1)[-1] for key in sealed}
    if (
        train_seeds & validation_seeds
        or train_seeds & old_sealed_seeds
        or validation_seeds & old_sealed_seeds
        or train_seeds & sealed_seeds
        or validation_seeds & sealed_seeds
        or (bool(protocol["confirmatory"]) and old_sealed_seeds & sealed_seeds)
    ):
        raise RuntimeError("resolved seed leakage across train/validation/sealed splits")
    if any(descriptor.schema_version != SCHEMA_VERSION for descriptor in descriptors):
        raise RuntimeError("formal sealed evaluation requires schema-v5 groups only")
    return {
        "train_logical_groups": len(train),
        "validation_logical_groups": len(validation),
        "original_frozen_test_logical_groups": len(sealed),
        "evaluated_logical_groups": len(actual),
        "train_resolved_seeds": sorted(train_seeds),
        "validation_resolved_seeds": sorted(validation_seeds),
        "sealed_resolved_seeds": sorted(sealed_seeds),
        "original_frozen_test_resolved_seeds": sorted(old_sealed_seeds),
        "evaluation_evidence_tier": protocol["evidence_tier"],
        "overlap_checks_passed": True,
    }


def validate_sealed_collection_contract(
    root: Path,
    descriptors: Sequence[GroupDescriptor],
    ensemble_contract: Mapping[str, Any],
    event_spec_sha256: str,
    model_config: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit collection/language/file provenance after access is reserved."""

    root_manifest_path = root / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    exact = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "intervention": INTERVENTION,
        "language_contract": LANGUAGE_CONTRACT,
        "event_spec_sha256": event_spec_sha256,
    }
    for key, expected in exact.items():
        if root_manifest.get(key) != expected:
            raise RuntimeError(
                f"sealed collection contract mismatch for {key}: "
                f"{root_manifest.get(key)!r} != {expected!r}"
            )
    if list(root_manifest.get("event_vocab", [])) != list(model_config.event_names):
        raise RuntimeError("sealed event vocabulary differs from ensemble config")
    if int(root_manifest.get("hidden_dim", -1)) != int(model_config.state_input_dim):
        raise RuntimeError("sealed hidden dimension differs from ensemble config")
    if int(root_manifest.get("action_dim", -1)) != int(model_config.action_dim):
        raise RuntimeError("sealed action dimension differs from ensemble config")
    action_chunk = int(root_manifest.get("action_chunk", 0))
    if action_chunk <= 0:
        raise RuntimeError("sealed action chunk contract is invalid")
    trajectory_contract = root_manifest.get("trajectory_contract")
    continuation_contract = root_manifest.get("continuation_query_contract")
    if not isinstance(trajectory_contract, Mapping) or trajectory_contract.get(
        "purpose"
    ) != "dynamic_predicates_failure_and_recovery_labels":
        raise RuntimeError("sealed trajectory contract is invalid")
    if not isinstance(continuation_contract, Mapping) or (
        continuation_contract.get("post_query_action") != POST_QUERY_ACTION_CONTRACT
        or continuation_contract.get("query_action_mask")
        != "contiguous_executed_prefix"
        or continuation_contract.get("purpose")
        != "late_event_action_conditioned_auxiliary_transitions"
    ):
        raise RuntimeError("sealed continuation-query contract is invalid")
    candidate_count = int(root_manifest.get("candidate_count", 0))
    if candidate_count < 2:
        raise RuntimeError("sealed collection must contain at least two candidates")
    if int(root_manifest.get("completed", -1)) != len(descriptors):
        raise RuntimeError("sealed collection completion count mismatch")
    requested = root_manifest.get("requested_seeds")
    resolved = root_manifest.get("resolved_seeds")
    if not isinstance(requested, list) or not isinstance(resolved, list):
        raise RuntimeError("sealed collection lacks requested/resolved seed proof")
    if len(requested) != len(descriptors) or sorted(map(int, resolved)) != sorted(
        descriptor.seed for descriptor in descriptors
    ):
        raise RuntimeError("sealed collection seed manifest mismatch")

    expected_files = ensemble_contract.get("sealed_test_files")
    if not isinstance(expected_files, Sequence) or isinstance(expected_files, (str, bytes)):
        raise RuntimeError("ensemble contract lacks sealed_test_files provenance")
    expected_by_key = {
        str(item["logical_key"]): item
        for item in expected_files
        if isinstance(item, Mapping)
    }
    if len(expected_by_key) != len(expected_files):
        raise RuntimeError("sealed file provenance contains duplicates or invalid rows")
    descriptor_keys = {descriptor.logical_key for descriptor in descriptors}
    if not bool(protocol["confirmatory"]) and set(expected_by_key) != descriptor_keys:
        raise RuntimeError("sealed file provenance keys do not match official holdout root")
    if bool(protocol["confirmatory"]) and set(expected_by_key) & descriptor_keys:
        raise RuntimeError("fresh confirmation files unexpectedly appear in development provenance")
    all_files = ensemble_contract.get("group_files")
    if not isinstance(all_files, Sequence) or isinstance(all_files, (str, bytes)):
        raise RuntimeError("ensemble contract lacks group_files provenance")
    all_by_key = {
        str(item["logical_key"]): item
        for item in all_files
        if isinstance(item, Mapping)
    }
    if len(all_by_key) != len(all_files):
        raise RuntimeError("global group provenance contains duplicates or invalid rows")
    all_split_keys = set().union(
        _as_string_set(ensemble_contract.get("train_groups"), "train_groups"),
        _as_string_set(
            ensemble_contract.get("validation_groups"), "validation_groups"
        ),
        _as_string_set(
            ensemble_contract.get("sealed_test_groups"), "sealed_test_groups"
        ),
    )
    if set(all_by_key) != all_split_keys:
        raise RuntimeError("global group provenance does not cover the frozen split exactly")

    root_items = root_manifest.get("groups")
    if not isinstance(root_items, list) or len(root_items) != len(descriptors):
        raise RuntimeError("sealed collection group manifest is incomplete")
    root_by_seed = {int(item["resolved_seed"]): item for item in root_items}
    if len(root_by_seed) != len(root_items):
        raise RuntimeError("sealed collection manifest contains duplicate resolved seeds")
    candidate_names: tuple[str, ...] | None = None
    file_rows = []
    for descriptor in descriptors:
        path = Path(descriptor.path)
        digest = sha256(path)
        expected = expected_by_key.get(descriptor.logical_key)
        global_expected = all_by_key.get(descriptor.logical_key)
        if bool(protocol["confirmatory"]):
            if expected is not None or global_expected is not None:
                raise RuntimeError("fresh confirmation group leaked into training provenance")
        else:
            assert expected is not None
            if int(expected.get("schema_version", -1)) != SCHEMA_VERSION:
                raise RuntimeError("frozen sealed provenance is not schema-v5")
            if digest != str(expected.get("sha256", "")):
                raise RuntimeError(f"sealed group SHA256 mismatch: {descriptor.logical_key}")
            if global_expected is None or digest != str(global_expected.get("sha256", "")):
                raise RuntimeError(f"global group provenance mismatch: {descriptor.logical_key}")
        with h5py.File(path, "r") as handle:
            attrs = handle.attrs
            if int(attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise RuntimeError(f"non-v5 sealed group: {path}")
            if str(attrs.get("language_contract", "")) != LANGUAGE_CONTRACT:
                raise RuntimeError(f"candidate language contract mismatch: {path}")
            if not bool(attrs.get("branch_instruction_consistent", False)):
                raise RuntimeError(f"candidate branch language changed: {path}")
            if str(attrs.get("intervention", "")) != INTERVENTION:
                raise RuntimeError(f"candidate intervention mismatch: {path}")
            if str(attrs.get("post_query_action_contract", "")) != POST_QUERY_ACTION_CONTRACT:
                raise RuntimeError(f"post-query action contract mismatch: {path}")
            if int(attrs.get("candidate_count", -1)) != candidate_count:
                raise RuntimeError(f"candidate count mismatch: {path}")
            names = tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["candidate_names"][:]
            )
            if len(names) != candidate_count or names.count("deterministic") != 1:
                raise RuntimeError(f"sealed group lacks unique deterministic baseline: {path}")
            if candidate_names is None:
                candidate_names = names
            elif names != candidate_names:
                raise RuntimeError("candidate names/order differ across sealed groups")
        root_item = root_by_seed.get(descriptor.seed)
        if root_item is None or tuple(map(str, root_item.get("candidate_names", []))) != names:
            raise RuntimeError("root/group candidate-name contract mismatch")
        file_rows.append(
            {
                "logical_key": descriptor.logical_key,
                "schema_version": descriptor.schema_version,
                "sha256": digest,
            }
        )
    return {
        "root": str(root),
        "root_manifest_sha256": sha256(root_manifest_path),
        "schema_version": SCHEMA_VERSION,
        "seed_registry": protocol["seed_registry"],
        "evidence_tier": protocol["evidence_tier"],
        "candidate_count": candidate_count,
        "hidden_dim": int(model_config.state_input_dim),
        "action_dim": int(model_config.action_dim),
        "action_chunk": action_chunk,
        "candidate_names": list(candidate_names or ()),
        "language_contract": LANGUAGE_CONTRACT,
        "intervention": INTERVENTION,
        "post_query_action_contract": POST_QUERY_ACTION_CONTRACT,
        "group_files": file_rows,
    }


@torch.inference_mode()
def score_groups_frozen(
    plugin: EventCriticPlugin,
    groups: Sequence[BranchGroup],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Strict offline equivalent of validation-time ensemble scoring."""

    deployment = plugin.counterfactual_deployment
    if deployment is None:
        raise RuntimeError("manifest did not load a frozen counterfactual deployment")
    object_mean = np.asarray(
        plugin.normalization.get("object_delta_mean"), dtype=np.float32
    )
    object_std = np.asarray(
        plugin.normalization.get("object_delta_std"), dtype=np.float32
    )
    if (
        object_mean.shape != (plugin.config.object_delta_dim,)
        or object_std.shape != object_mean.shape
        or not np.isfinite(object_mean).all()
        or not np.isfinite(object_std).all()
        or bool((object_std <= 0.0).any())
    ):
        raise RuntimeError("ensemble object normalization shape mismatch")
    if not plugin.models:
        raise RuntimeError("frozen scoring requires a non-empty ensemble")
    event_values = torch.tensor(deployment.event_values, device=device)
    rows: list[dict[str, Any]] = []
    maximum_formula_error = 0.0
    maximum_probability_error = 0.0
    for group in groups:
        batch = move_batch(
            collate_groups(
                [group], object_mean, object_std, include_auxiliary=False
            ),
            device,
        )
        member_logits = []
        member_scores = []
        member_aleatoric = []
        for model in plugin.models:
            output = forward_model(model, batch)
            logits = output["success_logit"]
            score = candidate_rank_score(
                output,
                event_values,
                deployment.duration_scale,
                success_temperature=deployment.success_temperature,
                event_weight=deployment.event_weight,
                duration_weight=deployment.duration_weight,
            )
            probability = torch.softmax(output["next_event_logits"], dim=-1)
            event_progress = (probability * event_values.to(probability)).sum(-1)
            duration = torch.expm1(
                output["duration_selected_log_mean"].clamp(0.0, 12.0)
            )
            explicit = (
                logits / deployment.success_temperature
                + deployment.event_weight * event_progress
                - deployment.duration_weight * duration / deployment.duration_scale
            )
            maximum_formula_error = max(
                maximum_formula_error,
                float((score - explicit).abs().max().cpu()),
            )
            member_logits.append(logits.cpu().numpy())
            member_scores.append(score.cpu().numpy())
            member_aleatoric.append(
                output["aleatoric_uncertainty"].cpu().numpy()
            )
        logits = np.stack(member_logits)
        scaled_logits = logits / deployment.success_temperature
        calibrated_probability = np.empty_like(scaled_logits)
        positive_logits = scaled_logits >= 0
        calibrated_probability[positive_logits] = 1.0 / (
            1.0 + np.exp(-scaled_logits[positive_logits])
        )
        negative_exponential = np.exp(scaled_logits[~positive_logits])
        calibrated_probability[~positive_logits] = negative_exponential / (
            1.0 + negative_exponential
        )
        torch_probability = torch.sigmoid(
            torch.from_numpy(logits) / deployment.success_temperature
        ).numpy()
        maximum_probability_error = max(
            maximum_probability_error,
            float(np.max(np.abs(calibrated_probability - torch_probability))),
        )
        member_scores_array = np.stack(member_scores)
        score = (
            member_scores_array.mean(0)
            - deployment.candidate_distance_weight * group.candidate_distance
        )
        uncertainty = calibrated_probability.std(0) + np.stack(
            member_aleatoric
        ).mean(0)
        baseline = [
            index
            for index, name in enumerate(group.candidate_names)
            if name == "deterministic"
        ]
        if len(baseline) != 1:
            raise RuntimeError(f"group lacks unique deterministic baseline: {group.logical_key}")
        if not np.isfinite(score).all() or not np.isfinite(uncertainty).all():
            raise RuntimeError(f"non-finite ensemble prediction: {group.logical_key}")
        rows.append(
            {
                "logical_key": group.logical_key,
                "schema_version": group.schema_version,
                "success": group.success.copy(),
                "steps": group.steps.copy(),
                "baseline_index": baseline[0],
                "mean_score": score,
                "mean_success_probability": calibrated_probability.mean(0),
                "uncertainty": uncertainty,
            }
        )
    if maximum_formula_error > 1e-6 or maximum_probability_error > 1e-6:
        raise RuntimeError("offline scoring drifted from the frozen training formula")
    return rows, {
        "maximum_score_formula_abs_error": maximum_formula_error,
        "maximum_temperature_probability_abs_error": maximum_probability_error,
    }


def apply_frozen_guard(
    rows: Sequence[Mapping[str, Any]], plugin: EventCriticPlugin
) -> list[dict[str, Any]]:
    deployment = plugin.counterfactual_deployment
    if deployment is None:
        raise RuntimeError("counterfactual deployment is absent")
    decisions = []
    for row in rows:
        scores = np.asarray(row["mean_score"], dtype=np.float64)
        uncertainty = np.asarray(row["uncertainty"], dtype=np.float64)
        baseline = int(row["baseline_index"])
        proposed = int(np.argmax(scores))
        margin = float(scores[proposed] - scores[baseline])
        reasons = []
        if proposed != baseline:
            gain_margin = deployment.gain_margin or 0.0
            uncertainty_threshold = (
                deployment.uncertainty_threshold
                if deployment.uncertainty_threshold is not None
                else 0.0
            )
            if not deployment.guard_enabled:
                reasons.append("manifest_guard_disabled")
            if not math.isfinite(scores[proposed]) or not math.isfinite(scores[baseline]):
                reasons.append("nonfinite_candidate_score")
            if margin < gain_margin:
                reasons.append("score_margin_below_guard")
            proposed_uncertainty = float(uncertainty[proposed])
            if not math.isfinite(proposed_uncertainty):
                reasons.append("nonfinite_uncertainty")
            elif (
                proposed_uncertainty > uncertainty_threshold
            ):
                reasons.append("uncertainty_above_guard")
        selected = baseline if reasons else proposed
        decisions.append(
            {
                "logical_key": str(row["logical_key"]),
                "baseline_index": baseline,
                "proposed_index": proposed,
                "selected_index": selected,
                "score_margin": margin,
                "proposed_uncertainty": float(uncertainty[proposed]),
                "guard_fallback": bool(proposed != baseline and selected == baseline),
                "changed_from_actor": bool(selected != baseline),
                "fallback_reasons": reasons,
            }
        )
    return decisions


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware Mann-Whitney AUC using O(n) memory."""

    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("AUC labels and scores must be aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("AUC scores must be finite")
    positive_count = int((labels > 0.5).sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        # Ranks are one-based; ties receive their average rank.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = float(ranks[labels > 0.5].sum())
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(statistic / (positive_count * negative_count))


def multiclass_prediction_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    """Return support-aware metrics without hiding absent event classes."""

    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("multiclass targets and predictions must be aligned vectors")
    if not len(target):
        return {
            "count": 0,
            "accuracy": None,
            "balanced_accuracy_supported": None,
            "macro_f1_supported": None,
            "confusion_matrix_rows_true_columns_predicted": [],
            "per_class": {},
        }
    classes = len(names)
    if classes < 1 or bool(
        ((target < 0) | (target >= classes) | (prediction < 0) | (prediction >= classes)).any()
    ):
        raise ValueError("multiclass ids fall outside the declared vocabulary")
    per_class: dict[str, Any] = {}
    supported_f1 = []
    supported_recall = []
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    for index, name in enumerate(names):
        true_positive = int(((target == index) & (prediction == index)).sum())
        false_positive = int(((target != index) & (prediction == index)).sum())
        false_negative = int(((target == index) & (prediction != index)).sum())
        support = int((target == index).sum())
        predicted = int((prediction == index).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            supported_f1.append(f1)
            supported_recall.append(recall)
        per_class[str(name)] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "count": int(len(target)),
        "accuracy": float((target == prediction).mean()),
        "balanced_accuracy_supported": (
            float(np.mean(supported_recall)) if supported_recall else None
        ),
        "macro_f1_supported": (
            float(np.mean(supported_f1)) if supported_f1 else None
        ),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "per_class": per_class,
    }


def binary_probability_metrics(
    target: np.ndarray, probability: np.ndarray, *, bins: int = 10
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if target.shape != probability.shape or target.ndim != 1:
        raise ValueError("binary targets and probabilities must be aligned vectors")
    if not len(target):
        return {
            "count": 0,
            "positive_support": 0,
            "auc": None,
            "average_precision": None,
            "accuracy_at_0_5": None,
            "f1_at_0_5": None,
            "mixture_log_loss": None,
            "brier": None,
            "ece": None,
        }
    if not np.isfinite(probability).all() or bool(
        ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("binary probabilities must be finite values in [0,1]")
    if bool(((target != 0.0) & (target != 1.0)).any()):
        raise ValueError("binary targets must contain only zero or one")
    hard = probability >= 0.5
    positive = target > 0.5
    true_positive = int((hard & positive).sum())
    false_positive = int((hard & ~positive).sum())
    false_negative = int((~hard & positive).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(target[mask].mean())
            )
    order = np.argsort(-probability, kind="stable")
    ordered_probability = probability[order]
    ordered_positive = positive[order]
    if ordered_positive.any():
        # Threshold-block AP is invariant to row order inside equal-score ties.
        true_positives = 0
        seen = 0
        average_precision = 0.0
        start = 0
        positive_count = int(ordered_positive.sum())
        while start < len(ordered_positive):
            stop = start + 1
            while (
                stop < len(ordered_positive)
                and ordered_probability[stop] == ordered_probability[start]
            ):
                stop += 1
            block_positive = int(ordered_positive[start:stop].sum())
            true_positives += block_positive
            seen = stop
            average_precision += (
                block_positive / positive_count
            ) * (true_positives / seen)
            start = stop
        average_precision = float(average_precision)
    else:
        average_precision = None
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    log_loss = -float(
        (target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)).mean()
    )
    return {
        "count": int(len(target)),
        "positive_support": int(positive.sum()),
        "auc": binary_auc(target, probability),
        "average_precision": average_precision,
        "accuracy_at_0_5": float((hard == positive).mean()),
        "f1_at_0_5": f1,
        "mixture_log_loss": log_loss,
        "brier": float(np.square(probability - target).mean()),
        "ece": ece,
    }


def regression_prediction_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("regression targets and predictions must be aligned vectors")
    if not len(target):
        return {"count": 0, "mae": None, "rmse": None, "median_ae": None}
    error = prediction - target
    return {
        "count": int(len(target)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "median_ae": float(np.median(np.abs(error))),
    }


def uncertainty_risk_coverage_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, Any]:
    """Classification risk when accepting predictions from low uncertainty up."""

    target = np.asarray(target, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if (
        target.shape != probability.shape
        or target.shape != uncertainty.shape
        or target.ndim != 1
    ):
        raise ValueError("risk-coverage inputs must be aligned vectors")
    if not len(target):
        return {"count": 0, "aurc": None, "risk_at_coverage": {}}
    if (
        not np.isfinite(uncertainty).all()
        or not np.isfinite(probability).all()
        or bool(((probability < 0.0) | (probability > 1.0)).any())
    ):
        raise ValueError("probability/uncertainty must be finite and probability in [0,1]")
    if bool(((target != 0.0) & (target != 1.0)).any()):
        raise ValueError("risk-coverage targets must contain only zero or one")
    error = ((probability >= 0.5) != (target > 0.5)).astype(np.float64)
    order = np.argsort(uncertainty, kind="stable")
    ordered_uncertainty = uncertainty[order]
    ordered_error = error[order]
    # Equal uncertainty does not authorize arbitrary cherry-picking.  Use the
    # expected prefix error for a random ordering within every tied block.
    expected_cumulative_error = np.empty(len(error), dtype=np.float64)
    errors_before = 0.0
    start = 0
    while start < len(error):
        stop = start + 1
        while (
            stop < len(error)
            and ordered_uncertainty[stop] == ordered_uncertainty[start]
        ):
            stop += 1
        block_errors = float(ordered_error[start:stop].sum())
        block_size = stop - start
        for offset in range(1, block_size + 1):
            expected_cumulative_error[start + offset - 1] = (
                errors_before + offset * block_errors / block_size
            )
        errors_before += block_errors
        start = stop
    cumulative_risk = expected_cumulative_error / np.arange(1, len(error) + 1)
    risks = {}
    for coverage in (0.25, 0.5, 0.75, 1.0):
        accepted = max(1, int(math.ceil(coverage * len(error))))
        risks[f"{coverage:.2f}"] = float(cumulative_risk[accepted - 1])
    return {
        "count": int(len(error)),
        "aurc": float(cumulative_risk.mean()),
        "risk_at_coverage": risks,
        "lowest_uncertainty": float(uncertainty[order[0]]),
        "highest_uncertainty": float(uncertainty[order[-1]]),
        "risk": "binary_error_at_probability_0.5",
        "tie_policy": "expected_random_order_within_equal_uncertainty",
    }


def equal_weight_mixture_nll(member_nll: torch.Tensor) -> torch.Tensor:
    """Exact negative log density of an equal-weight member mixture."""

    if member_nll.ndim < 1 or member_nll.shape[0] < 1:
        raise ValueError("member NLL tensor must have a non-empty member axis")
    if not torch.isfinite(member_nll).all():
        raise ValueError("member NLL tensor contains non-finite values")
    return -torch.logsumexp(-member_nll, dim=0) + math.log(member_nll.shape[0])


def diagonal_gaussian_joint_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
) -> torch.Tensor:
    """Joint diagonal-Gaussian NLL, retaining every leading batch axis."""

    if target.shape != mean.shape or target.shape != log_scale.shape:
        raise ValueError("Gaussian target/mean/log-scale shapes must match")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("Gaussian tensors need a non-empty feature axis")
    scale = torch.exp(log_scale.clamp(-5.0, 3.0)).clamp_min(1e-4)
    return (
        0.5 * torch.square((target - mean) / scale)
        + torch.log(scale)
        + 0.5 * math.log(2.0 * math.pi)
    ).sum(-1)


@torch.inference_mode()
def structured_prediction_metrics_frozen(
    plugin: EventCriticPlugin,
    groups: Sequence[BranchGroup],
    device: torch.device,
) -> dict[str, Any]:
    """Audit frozen event predictions on initial and continuation queries.

    This function never selects thresholds or alters the deployed score.  It is
    called only after the one-shot sealed-access marker has been reserved.
    """

    if not plugin.config.structured_events:
        raise RuntimeError("sealed prediction audit requires a structured checkpoint")
    deployment = plugin.counterfactual_deployment
    if deployment is None:
        raise RuntimeError("prediction audit requires a frozen deployment manifest")
    object_mean = np.asarray(
        plugin.normalization.get("object_delta_mean"), dtype=np.float32
    )
    object_std = np.asarray(
        plugin.normalization.get("object_delta_std"), dtype=np.float32
    )
    if (
        object_mean.shape != (plugin.config.object_delta_dim,)
        or object_std.shape != object_mean.shape
        or not np.isfinite(object_mean).all()
        or not np.isfinite(object_std).all()
        or bool((object_std <= 0.0).any())
    ):
        raise RuntimeError("ensemble object normalization shape mismatch")
    if not plugin.models:
        raise RuntimeError("prediction audit requires a non-empty ensemble")

    records: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "initial_mask",
            "structured_mask",
            "dense_mask",
            "current_event",
            "event_target",
            "event_prediction",
            "event_mixture_nll",
            "relative_target",
            "relative_prediction",
            "relative_mixture_nll",
            "destination_target",
            "destination_prediction",
            "destination_mixture_nll",
            "post_predicate_target",
            "post_predicate_probability",
            "reach_target",
            "reach_probability",
            "duration_target",
            "duration_prediction",
            "duration_censored_nll",
            "object_target",
            "object_prediction",
            "object_gaussian_nll",
            "latent_cosine",
            "latent_member_nll",
            "success_target",
            "success_probability",
            "success_uncertainty",
            "outcome_target",
            "outcome_prediction",
        )
    }
    for group in groups:
        batch = move_batch(
            collate_groups(
                [group], object_mean, object_std, include_auxiliary=True
            ),
            device,
        )
        outputs = [forward_model(model, batch) for model in plugin.models]
        mean_event = torch.stack(
            [torch.softmax(output["next_event_logits"], -1) for output in outputs]
        ).mean(0)
        mean_relative = torch.stack(
            [torch.softmax(output["relative_transition_logits"], -1) for output in outputs]
        ).mean(0)
        mean_destination = torch.stack(
            [torch.softmax(output["next_reached_event_logits"], -1) for output in outputs]
        ).mean(0)
        mean_predicate = torch.stack(
            [torch.sigmoid(output["post_predicate_logits"]) for output in outputs]
        ).mean(0)
        mean_reach = torch.stack(
            [torch.sigmoid(output["reach_logit"]) for output in outputs]
        ).mean(0)
        member_success_probability = torch.stack(
            [
                torch.sigmoid(
                    output["success_logit"] / deployment.success_temperature
                )
                for output in outputs
            ]
        )
        mean_success = member_success_probability.mean(0)
        success_uncertainty = member_success_probability.std(0, unbiased=False) + torch.stack(
            [output["aleatoric_uncertainty"] for output in outputs]
        ).mean(0)
        mean_outcome = torch.stack(
            [torch.softmax(output["outcome_logits"], -1) for output in outputs]
        ).mean(0)
        mean_duration = torch.stack(
            [
                torch.expm1(
                    output["duration_selected_log_mean"].clamp(0.0, 12.0)
                )
                for output in outputs
            ]
        ).mean(0)
        member_duration_nll = torch.stack(
            [
                lognormal_nll_per_item(
                    output["duration_selected_log_mean"],
                    output["duration_selected_log_scale"],
                    batch["duration"],
                    batch["duration_observed"],
                )
                for output in outputs
            ]
        )
        duration_mixture_nll = equal_weight_mixture_nll(member_duration_nll)
        mean_object = torch.stack(
            [output["object_delta_mean"] for output in outputs]
        ).mean(0)
        member_object_joint_nll = []
        for output in outputs:
            member_object_joint_nll.append(
                diagonal_gaussian_joint_nll(
                    batch["object_delta"],
                    output["object_delta_mean"],
                    output["object_delta_log_scale"],
                )
            )
        # Form each diagonal-Gaussian joint density before mixing members.
        # Averaging dimensions before logsumexp would define a different and
        # over-optimistic distribution.
        object_mixture_nll = equal_weight_mixture_nll(
            torch.stack(member_object_joint_nll)
        )
        latent_cosines = []
        latent_member_nll = []
        for model, output in zip(plugin.models, outputs):
            target_semantic = model.encode_state(batch["post_hidden"])
            latent_cosines.append(
                torch.nn.functional.cosine_similarity(
                    output["future_latent_mean"], target_semantic, dim=-1
                )
            )
            latent_scale = torch.exp(
                output["future_latent_log_scale"].clamp(-5.0, 2.0)
            ).clamp_min(1e-4)
            latent_member_nll.append(
                (
                    0.5
                    * torch.square(
                        (target_semantic - output["future_latent_mean"])
                        / latent_scale
                    )
                    + torch.log(latent_scale)
                    + 0.5 * math.log(2.0 * math.pi)
                ).mean(-1)
            )
        mean_latent_cosine = torch.stack(latent_cosines).mean(0)
        mean_latent_member_nll = torch.stack(latent_member_nll).mean(0)

        def categorical_nll(
            probability: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            selected = probability.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)
            return -torch.log(selected.clamp_min(1e-12))

        normalized_target = batch["object_delta"].detach().cpu().numpy()
        normalized_prediction = mean_object.detach().cpu().numpy()
        records["initial_mask"].append(
            batch["terminal_mask"].bool().detach().cpu().numpy()
        )
        records["structured_mask"].append(
            batch["structured_mask"].bool().detach().cpu().numpy()
        )
        records["dense_mask"].append(
            batch["dense_mask"].bool().detach().cpu().numpy()
        )
        records["current_event"].append(batch["current_event_id"].cpu().numpy())
        records["event_target"].append(batch["next_event_id"].cpu().numpy())
        records["event_prediction"].append(mean_event.argmax(-1).cpu().numpy())
        records["event_mixture_nll"].append(
            categorical_nll(mean_event, batch["next_event_id"]).cpu().numpy()
        )
        records["relative_target"].append(
            batch["relative_transition_id"].cpu().numpy()
        )
        records["relative_prediction"].append(
            mean_relative.argmax(-1).cpu().numpy()
        )
        records["relative_mixture_nll"].append(
            categorical_nll(
                mean_relative, batch["relative_transition_id"]
            ).cpu().numpy()
        )
        records["destination_target"].append(
            batch["next_reached_event_id"].cpu().numpy()
        )
        records["destination_prediction"].append(
            mean_destination.argmax(-1).cpu().numpy()
        )
        records["destination_mixture_nll"].append(
            categorical_nll(
                mean_destination, batch["next_reached_event_id"]
            ).cpu().numpy()
        )
        records["post_predicate_target"].append(
            batch["post_predicates"].cpu().numpy()
        )
        records["post_predicate_probability"].append(
            mean_predicate.cpu().numpy()
        )
        records["reach_target"].append(
            batch["duration_observed"].cpu().numpy()
        )
        records["reach_probability"].append(mean_reach.cpu().numpy())
        records["duration_target"].append(batch["duration"].cpu().numpy())
        records["duration_prediction"].append(mean_duration.cpu().numpy())
        records["duration_censored_nll"].append(
            duration_mixture_nll.cpu().numpy()
        )
        records["object_target"].append(
            normalized_target * object_std + object_mean
        )
        records["object_prediction"].append(
            normalized_prediction * object_std + object_mean
        )
        records["object_gaussian_nll"].append(object_mixture_nll.cpu().numpy())
        records["latent_cosine"].append(mean_latent_cosine.cpu().numpy())
        records["latent_member_nll"].append(
            mean_latent_member_nll.cpu().numpy()
        )
        records["success_target"].append(batch["success"].cpu().numpy())
        records["success_probability"].append(mean_success.cpu().numpy())
        records["success_uncertainty"].append(success_uncertainty.cpu().numpy())
        records["outcome_target"].append(batch["outcome_id"].cpu().numpy())
        records["outcome_prediction"].append(mean_outcome.argmax(-1).cpu().numpy())

    arrays = {name: np.concatenate(parts) for name, parts in records.items()}
    if not len(arrays["initial_mask"]):
        raise RuntimeError("sealed prediction audit received no query transitions")
    initial = arrays["initial_mask"].astype(bool)
    structured = arrays["structured_mask"].astype(bool)
    dense = arrays["dense_mask"].astype(bool)
    all_queries = np.ones(len(initial), dtype=bool)
    observed = arrays["reach_target"] > 0.5

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        structured_mask = mask & structured
        dense_mask = mask & dense
        duration_mask = dense_mask & observed
        predicate_metrics = {
            str(name): binary_probability_metrics(
                arrays["post_predicate_target"][structured_mask, index],
                arrays["post_predicate_probability"][structured_mask, index],
            )
            for index, name in enumerate(plugin.config.predicate_names)
        }
        object_error = (
            arrays["object_prediction"][dense_mask]
            - arrays["object_target"][dense_mask]
        )
        event_target = arrays["event_target"][structured_mask]
        frequency_event = (
            int(
                np.bincount(
                    event_target, minlength=plugin.config.num_events
                ).argmax()
            )
            if len(event_target)
            else 0
        )
        post_event = multiclass_prediction_metrics(
            event_target,
            arrays["event_prediction"][structured_mask],
            plugin.config.event_names,
        )
        post_event["mixture_nll"] = (
            float(arrays["event_mixture_nll"][structured_mask].mean())
            if structured_mask.any()
            else None
        )
        relative = multiclass_prediction_metrics(
            arrays["relative_target"][structured_mask],
            arrays["relative_prediction"][structured_mask],
            plugin.config.relative_transition_names,
        )
        relative["mixture_nll"] = (
            float(arrays["relative_mixture_nll"][structured_mask].mean())
            if structured_mask.any()
            else None
        )
        destination = multiclass_prediction_metrics(
            arrays["destination_target"][duration_mask],
            arrays["destination_prediction"][duration_mask],
            plugin.config.event_names,
        )
        destination["mixture_nll"] = (
            float(arrays["destination_mixture_nll"][duration_mask].mean())
            if duration_mask.any()
            else None
        )
        object_count = int(dense_mask.sum())
        return {
            "query_count": int(mask.sum()),
            "eligible_counts": {
                "structured": int(structured_mask.sum()),
                "dense": object_count,
                "duration_observed": int(duration_mask.sum()),
            },
            "post_event": post_event,
            "post_event_baselines": {
                "current_event_self_loop": multiclass_prediction_metrics(
                    event_target,
                    arrays["current_event"][structured_mask],
                    plugin.config.event_names,
                ),
                "sealed_frequency_diagnostic": {
                    "constant_event": plugin.config.event_names[frequency_event],
                    **multiclass_prediction_metrics(
                        event_target,
                        np.full_like(event_target, frequency_event),
                        plugin.config.event_names,
                    ),
                },
            },
            "relative_transition": relative,
            "next_reached_event_observed": destination,
            "post_predicates": predicate_metrics,
            "reach_probability": binary_probability_metrics(
                arrays["reach_target"][dense_mask],
                arrays["reach_probability"][dense_mask],
            ),
            "observed_duration_steps": {
                "point_prediction": (
                    "ensemble_mean_of_member_logtime_medians_expm1_mu"
                ),
                **regression_prediction_metrics(
                    arrays["duration_target"][duration_mask],
                    arrays["duration_prediction"][duration_mask],
                ),
            },
            "duration_censored_logtime_mixture_nll": {
                "count": object_count,
                "observed": int(duration_mask.sum()),
                "right_censored": int((dense_mask & ~observed).sum()),
                "mean": (
                    float(arrays["duration_censored_nll"][dense_mask].mean())
                    if object_count
                    else None
                ),
            },
            "object_xyz_delta_m": {
                "count": object_count,
                "mae": float(np.abs(object_error).mean()) if object_count else None,
                "rmse": (
                    float(np.sqrt(np.square(object_error).mean()))
                    if object_count
                    else None
                ),
                "per_dimension_mae": (
                    np.abs(object_error).mean(0).tolist() if object_count else None
                ),
                "normalized_gaussian_mixture_joint_nll": (
                    float(arrays["object_gaussian_nll"][dense_mask].mean())
                    if object_count
                    else None
                ),
                "normalized_gaussian_mixture_nll_per_dimension": (
                    float(arrays["object_gaussian_nll"][dense_mask].mean())
                    / plugin.config.object_delta_dim
                    if object_count
                    else None
                ),
            },
            "future_latent_cosine": {
                "count": object_count,
                "mean_member_cosine": (
                    float(arrays["latent_cosine"][dense_mask].mean())
                    if object_count
                    else None
                ),
                "median_member_cosine": (
                    float(np.median(arrays["latent_cosine"][dense_mask]))
                    if object_count
                    else None
                ),
                "mean_member_normalized_gaussian_nll_per_dimension": (
                    float(arrays["latent_member_nll"][dense_mask].mean())
                    if object_count
                    else None
                ),
                "aggregation_note": (
                    "member prediction is scored in its own semantic encoder space; "
                    "latent coordinates are not mixed across members"
                ),
            },
        }

    initial_summary = summarize(initial)
    initial_summary["success_probability"] = binary_probability_metrics(
        arrays["success_target"][initial],
        arrays["success_probability"][initial],
    )
    initial_summary["success_uncertainty_risk_coverage"] = (
        uncertainty_risk_coverage_metrics(
            arrays["success_target"][initial],
            arrays["success_probability"][initial],
            arrays["success_uncertainty"][initial],
        )
    )
    initial_summary["outcome"] = multiclass_prediction_metrics(
        arrays["outcome_target"][initial],
        arrays["outcome_prediction"][initial],
        plugin.config.outcome_names,
    )
    return {
        "contract": (
            "frozen_ensemble_no_threshold_or_weight_tuning_after_one_shot_reservation"
        ),
        "initial_candidates": initial_summary,
        "all_query_transitions": summarize(all_queries),
        "continuation_query_count": int((~initial).sum()),
        "mask_contract": (
            "event_relative_predicate=structured; "
            "reach_duration_object_future=dense; "
            "destination=dense_and_duration_observed; "
            "success_outcome_AURC=terminal_initial_candidates_only"
        ),
    }


def exact_two_sided_binomial_p(improved: int, harmed: int) -> float:
    discordant = improved + harmed
    if discordant == 0:
        return 1.0
    tail = min(improved, harmed)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    probability /= 2**discordant
    return float(min(1.0, 2.0 * probability))


def paired_policy_metrics(
    baseline: np.ndarray,
    selected: np.ndarray,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.float64)
    if baseline.shape != selected.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("paired policy arrays must be aligned and non-empty")
    delta = selected - baseline
    improved = int(np.sum(delta > 0))
    harmed = int(np.sum(delta < 0))
    generator = np.random.default_rng(bootstrap_seed)
    bootstrap = delta[
        generator.integers(
            0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta))
        )
    ].mean(1)
    exact = exact_two_sided_binomial_p(improved, harmed)
    return {
        "groups": int(len(delta)),
        "baseline_successes": int(baseline.sum()),
        "selected_successes": int(selected.sum()),
        "baseline_success_rate": float(baseline.mean()),
        "selected_success_rate": float(selected.mean()),
        "paired_success_delta": float(delta.mean()),
        "paired_delta_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "improved_groups": improved,
        "harmed_groups": harmed,
        "unchanged_groups": int(np.sum(delta == 0)),
        "exact_sign_test_two_sided_p": exact,
        "mcnemar_exact_two_sided_p": exact,
        "discordant_pairs": improved + harmed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(decisions) or not rows:
        raise RuntimeError("prediction and decision rows are not aligned")
    baseline_outcomes = []
    proposed_outcomes = []
    selected_outcomes = []
    oracle_outcomes = []
    candidate_labels = []
    candidate_scores = []
    candidate_probabilities = []
    pair_correct = pair_total = 0.0
    reason_counter: Counter[str] = Counter()
    per_group = []
    for row, decision in zip(rows, decisions):
        if str(row["logical_key"]) != str(decision["logical_key"]):
            raise RuntimeError("decision order differs from prediction rows")
        success = np.asarray(row["success"], dtype=np.float64)
        score = np.asarray(row["mean_score"], dtype=np.float64)
        probability = np.asarray(row["mean_success_probability"], dtype=np.float64)
        baseline = int(decision["baseline_index"])
        proposed = int(decision["proposed_index"])
        selected = int(decision["selected_index"])
        baseline_outcomes.append(success[baseline])
        proposed_outcomes.append(success[proposed])
        selected_outcomes.append(success[selected])
        oracle_outcomes.append(success.max())
        candidate_labels.append(success)
        candidate_scores.append(score)
        candidate_probabilities.append(probability)
        positive = score[success > 0.5]
        negative = score[success <= 0.5]
        if len(positive) and len(negative):
            delta = positive[:, None] - negative[None, :]
            pair_correct += float((delta > 0).sum() + 0.5 * (delta == 0).sum())
            pair_total += float(delta.size)
        reason_counter.update(map(str, decision["fallback_reasons"]))
        per_group.append(
            {
                **dict(decision),
                "baseline_success": int(success[baseline]),
                "proposed_success": int(success[proposed]),
                "selected_success": int(success[selected]),
                "oracle_success": int(success.max()),
                "candidate_scores": score.tolist(),
                "candidate_success_probabilities": probability.tolist(),
                "candidate_uncertainties": np.asarray(row["uncertainty"]).tolist(),
            }
        )
    baseline_array = np.asarray(baseline_outcomes)
    proposed_array = np.asarray(proposed_outcomes)
    selected_array = np.asarray(selected_outcomes)
    labels = np.concatenate(candidate_labels)
    scores = np.concatenate(candidate_scores)
    probabilities = np.concatenate(candidate_probabilities)
    guarded = paired_policy_metrics(
        baseline_array, selected_array, bootstrap_seed=BOOTSTRAP_SEED
    )
    unguarded = paired_policy_metrics(
        baseline_array, proposed_array, bootstrap_seed=BOOTSTRAP_SEED + 1
    )
    changed = sum(bool(decision["changed_from_actor"]) for decision in decisions)
    proposals = sum(
        int(decision["proposed_index"]) != int(decision["baseline_index"])
        for decision in decisions
    )
    fallbacks = sum(bool(decision["guard_fallback"]) for decision in decisions)
    guarded["changed_candidate_groups"] = changed
    unguarded["changed_candidate_groups"] = proposals
    return {
        "actor_baseline": {
            "successes": int(baseline_array.sum()),
            "success_rate": float(baseline_array.mean()),
        },
        "guarded_selected": guarded,
        "unguarded_diagnostic": unguarded,
        "oracle": {
            "successes": int(np.sum(oracle_outcomes)),
            "success_rate": float(np.mean(oracle_outcomes)),
        },
        "candidate_score_auc": binary_auc(labels, scores),
        "candidate_success_probability_auc": binary_auc(labels, probabilities),
        "within_group_pair_accuracy": pair_correct / pair_total if pair_total else None,
        "within_group_comparable_pairs": int(pair_total),
        "guard": {
            "proposed_nonbaseline_groups": proposals,
            "proposal_coverage": proposals / len(rows),
            "changed_groups": changed,
            "coverage": changed / len(rows),
            "guard_fallback_groups": fallbacks,
            "guard_fallback_rate": fallbacks / len(rows),
            "fallback_reason_histogram": dict(sorted(reason_counter.items())),
        },
        "per_group": per_group,
    }


def verify_member_files(manifest_path: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    members = manifest.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise RuntimeError("ensemble manifest lacks member provenance")
    rows = []
    for member in members:
        if not isinstance(member, Mapping) or not member.get("path"):
            raise RuntimeError("invalid ensemble member provenance")
        recorded = Path(str(member["path"])).expanduser()
        portable = manifest_path.parent / recorded.name
        path = recorded if recorded.is_file() else portable
        if not path.is_file():
            raise FileNotFoundError(f"ensemble member provenance file unavailable: {recorded}")
        digest = sha256(path)
        if digest != str(member.get("sha256", "")):
            raise RuntimeError(f"ensemble member SHA256 mismatch: {path}")
        rows.append(
            {"seed": int(member["seed"]), "path": str(path.resolve()), "sha256": digest}
        )
    return rows


def validate_evaluation_authorization(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    fresh_manifest_present: bool,
    sealed_root: Path,
) -> dict[str, Any]:
    """Fail closed before reserving or reading any sealed collection metadata."""

    policy = manifest.get("test_policy")
    if policy == LEGACY_TEST_POLICY:
        return {"mode": "legacy_frozen_split"}
    if policy != OOF_TEST_POLICY:
        raise RuntimeError("ensemble manifest test policy is unsupported")
    if not fresh_manifest_present:
        raise RuntimeError("OOF final may only consume one-shot fresh50 confirmation")
    return validate_authorized_oof_final(
        manifest_path, manifest, forbidden_root=sealed_root
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument(
        "--fresh-seed-manifest",
        type=Path,
        help=(
            "Frozen reset-only fresh-50 manifest; mandatory when the sealed "
            "collection declares explicit_fresh_confirmation."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest_path = args.ensemble_manifest.resolve()
    sealed_root = args.sealed_data.resolve()
    event_spec_path = args.event_spec.resolve()
    fresh_seed_manifest_path = (
        args.fresh_seed_manifest.resolve() if args.fresh_seed_manifest else None
    )
    output = args.output.resolve()
    if output == sealed_root or sealed_root in output.parents:
        raise RuntimeError("evaluation output must not be inside the sealed data root")
    for name, path in (
        ("ensemble manifest", manifest_path),
        ("event spec", event_spec_path),
        ("fresh seed manifest", fresh_seed_manifest_path),
    ):
        if path is not None and path_is_within(path, sealed_root):
            raise RuntimeError(
                f"{name} must be outside the sealed data root before one-shot reservation"
            )
    marker = output / "evaluated_once.json"
    if marker.exists():
        raise RuntimeError(
            f"sealed evaluation already reserved or completed: {marker}; refusing overwrite"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise RuntimeError("unsupported ensemble manifest format")
    validate_pre_reservation_paths(manifest_path, manifest, sealed_root)
    manifest_digest = sha256(manifest_path)
    event_spec_digest = sha256(event_spec_path)
    fresh_seed_manifest: Mapping[str, Any] | None = None
    fresh_seed_manifest_digest: str | None = None
    if fresh_seed_manifest_path is not None:
        fresh_seed_manifest = json.loads(
            fresh_seed_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(fresh_seed_manifest, Mapping):
            raise RuntimeError("fresh seed manifest must contain a JSON object")
        fresh_seed_manifest_digest = sha256(fresh_seed_manifest_path)
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("ensemble manifest lacks frozen contract")
    if str(contract.get("event_spec_sha256", "")) != event_spec_digest:
        raise RuntimeError("event-spec SHA256 differs from training contract")
    ensemble_authorization = validate_evaluation_authorization(
        manifest_path,
        manifest,
        fresh_manifest_present=fresh_seed_manifest_path is not None,
        sealed_root=sealed_root,
    )
    device = torch.device(args.device)
    plugin = EventCriticPlugin.from_manifest(
        manifest_path, device=device, verify_sha256=True
    )
    member_provenance = verify_member_files(manifest_path, manifest)

    reserve_evaluated_once(
        marker,
        {
            "ensemble_manifest": str(manifest_path),
            "ensemble_manifest_sha256": manifest_digest,
            "sealed_data": str(sealed_root),
            "event_spec": str(event_spec_path),
            "event_spec_sha256": event_spec_digest,
            "fresh_seed_manifest": (
                str(fresh_seed_manifest_path)
                if fresh_seed_manifest_path is not None
                else None
            ),
            "fresh_seed_manifest_sha256": fresh_seed_manifest_digest,
        },
    )
    # Everything below this line is the one authorized sealed access.  Root
    # manifests may themselves contain terminal outcomes, so even identity
    # scanning occurs only after the durable one-shot reservation.
    sealed_root_manifest = json.loads(
        (sealed_root / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(sealed_root_manifest, Mapping):
        raise RuntimeError("sealed collection manifest must contain a JSON object")
    protocol = classify_evaluation_protocol(
        sealed_root_manifest,
        fresh_seed_manifest,
        fresh_seed_manifest_digest,
    )
    if fresh_seed_manifest_path is not None:
        assert protocol["fresh_seed_manifest"] is not None
        protocol["fresh_seed_manifest"]["path"] = str(fresh_seed_manifest_path)
    descriptors = scan_group_descriptors([sealed_root])
    split_audit = validate_frozen_split_contract(contract, descriptors, protocol)
    collection_audit = validate_sealed_collection_contract(
        sealed_root,
        descriptors,
        contract,
        event_spec_digest,
        plugin.config,
        protocol,
    )
    event_spec = json.loads(event_spec_path.read_text(encoding="utf-8"))
    calibrations = event_spec.get("calibration")
    if not isinstance(calibrations, Mapping):
        raise RuntimeError("event spec lacks calibration mapping")
    object_names = contract.get("object_names")
    body_to_id = contract.get("body_to_id")
    policy_to_id = contract.get("policy_to_id")
    if not isinstance(object_names, Sequence) or isinstance(object_names, (str, bytes)):
        raise RuntimeError("ensemble contract lacks object_names")
    if not isinstance(body_to_id, Mapping) or not isinstance(policy_to_id, Mapping):
        raise RuntimeError("ensemble contract lacks body/policy id mappings")
    groups = load_descriptor_groups(
        descriptors,
        plugin.config,
        [str(name) for name in object_names],
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
    )
    rows, formula_audit = score_groups_frozen(plugin, groups, device)
    decisions = apply_frozen_guard(rows, plugin)
    metrics = summarize_rows(rows, decisions)
    prediction_metrics = structured_prediction_metrics_frozen(
        plugin, groups, device
    )
    deployment = plugin.counterfactual_deployment
    assert deployment is not None
    aggregate_checkpoint = plugin.checkpoint_paths[0]
    aggregate_checkpoint_digest = sha256(aggregate_checkpoint)
    if aggregate_checkpoint_digest != str(
        manifest["ensemble_checkpoint"]["sha256"]
    ):
        raise RuntimeError("aggregate checkpoint changed after manifest load")
    result = {
        "format": EVALUATION_FORMAT,
        "status": "complete",
        "evaluated_once": True,
        "sealed_labels_first_read_by": "this_evaluator_after_atomic_reservation",
        "ensemble_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
            "aggregate_checkpoint": {
                "path": str(aggregate_checkpoint),
                "sha256": aggregate_checkpoint_digest,
                "manifest_sha256_match": True,
            },
            "members": member_provenance,
        },
        "ensemble_authorization": ensemble_authorization,
        "event_spec": {"path": str(event_spec_path), "sha256": event_spec_digest},
        "evaluation_protocol": protocol,
        "frozen_deployment": {
            "temperature": deployment.success_temperature,
            "duration_scale": deployment.duration_scale,
            "event_values": list(deployment.event_values),
            "event_weight": deployment.event_weight,
            "duration_weight": deployment.duration_weight,
            "candidate_distance_weight": deployment.candidate_distance_weight,
            "guard_enabled": deployment.guard_enabled,
            "gain_margin": deployment.gain_margin,
            "uncertainty_threshold": deployment.uncertainty_threshold,
            "override_flags_available": False,
        },
        "split_audit": split_audit,
        "collection_audit": collection_audit,
        "scoring_formula_audit": formula_audit,
        "metrics": metrics,
        "prediction_metrics": prediction_metrics,
        "limitations": [
            "This is an offline same-state candidate-branch evaluation.",
            "Unguarded selection is diagnostic only; frozen guarded selection is primary.",
            "The evaluator does not retune temperature, scoring weights, or guard thresholds.",
        ],
    }
    atomic_json(marker, result)
    print(
        "SEALED_EVALUATION_COMPLETE="
        + json.dumps(
            {
                "groups": len(groups),
                "actor_success_rate": metrics["actor_baseline"]["success_rate"],
                "guarded_success_rate": metrics["guarded_selected"]["selected_success_rate"],
                "paired_delta": metrics["guarded_selected"]["paired_success_delta"],
                "result": str(marker),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
