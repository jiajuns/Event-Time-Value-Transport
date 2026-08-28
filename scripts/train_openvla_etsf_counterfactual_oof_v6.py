#!/usr/bin/env python3
"""Development-only nested OOF training for the frozen success-rank head.

The outer labels are not opened until all four inner models have produced
crossfit predictions, the guard has been selected, and the outer model has
been refit.  There is deliberately no final-refit or fresh-data subcommand.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluate_openvla_etsf_oof_prediction_diagnostics import (
    FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS,
    STRUCTURED_ROW_FORMAT,
    SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT,
    build_oof_prediction_diagnostics,
    validate_oof_prediction_diagnostics,
)
from openvla_etsf_counterfactual_oof import DEPLOYMENT_CANDIDATE_NAMES
from openvla_etsf_counterfactual_oof import canonical_sha256 as v5_canonical_sha256
from openvla_etsf_counterfactual_oof_v6 import (
    EXPECTED_GROUPS,
    FORMAL_TRAINABLE_PARAMETER_COUNT,
    FORMAT,
    OUTER_FOLDS,
    SELECTION_FORMAT,
    TRAINABLE_PARAMETER_NAMES,
    apply_outer_policy,
    canonical_sha256,
    evaluate_guard,
    make_nested_oof_manifest,
    select_inner_guard,
    validate_nested_oof_manifest,
)
from openvla_etsf_event_world_model import ActionConditionedEventWorldModel
from train_openvla_etsf_counterfactual import (
    atomic_json,
    atomic_torch_save,
    canonical_policy_mapping,
    collate_groups,
    configure_action_rank_training,
    counterfactual_aleatoric_uncertainty,
    forward_model,
    group_action_rank_residual,
    load_counterfactual_pretrained_state,
    load_descriptor_groups,
    load_pretrained,
    move_batch,
    scan_group_descriptors,
    sha256,
    train_member,
)


INNER_SEED_BASE = 20261000
OUTER_SEED_BASE = 20262000


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _refuse(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"v6 refuses overwrite/resume: {path}")


def _device() -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("v6 fold training requires CUDA bf16")
    return torch.device("cuda:0")


def _tensor_sha(state: Mapping[str, torch.Tensor], *, core_only: bool) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if core_only and name.startswith("action_rank_head."):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _training_args(args: argparse.Namespace) -> Namespace:
    # Only the two registered ranking losses are non-zero.  All factual heads
    # are frozen and ranking features are detached by the core trainer.
    return Namespace(
        freeze_factual_core=True,
        unfreeze_semantic=False,
        learning_rate=1e-3,
        weight_decay=0.1,
        amp="bf16",
        groups_per_batch=16,
        num_workers=args.num_workers,
        min_relative_support=5,
        success_weight=0.0,
        outcome_weight=0.0,
        pairwise_weight=1.0,
        listwise_weight=0.0,
        group_centered_weight=0.0,
        baseline_contrast_weight=1.0,
        event_weight=0.0,
        relative_weight=0.0,
        destination_weight=0.0,
        predicate_weight=0.0,
        reach_weight=0.0,
        duration_weight=0.0,
        object_weight=0.0,
        latent_weight=0.0,
        grad_clip=2.0,
        steps=100,
        eval_every=100,
        early_stopping_patience=0,
    )


def _event_context(event_spec: Path, contract: Mapping[str, Any]):
    spec = _json(event_spec)
    calibrations = spec.get("calibration")
    objects = contract.get("object_names")
    bodies = contract.get("body_to_id")
    policies = canonical_policy_mapping(contract.get("policy_to_id"))
    if not isinstance(calibrations, Mapping) or not isinstance(bodies, Mapping):
        raise RuntimeError("v6 factual/event context is incomplete")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise RuntimeError("v6 factual object registration is incomplete")
    if "openvla" not in policies:
        raise RuntimeError("v6 requires canonical OpenVLA policy registration")
    return calibrations, list(map(str, objects)), bodies, policies


def _normalization(checkpoint: Mapping[str, Any], config: Any):
    norm = checkpoint.get("normalization")
    if not isinstance(norm, Mapping):
        raise RuntimeError("v6 requires frozen factual normalization")
    mean = np.asarray(norm.get("object_delta_mean"), dtype=np.float32)
    std = np.asarray(norm.get("object_delta_std"), dtype=np.float32)
    if mean.shape != (config.object_delta_dim,) or std.shape != mean.shape:
        raise RuntimeError("v6 factual normalization shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("v6 factual normalization invalid")
    return mean, std


def _reject_fresh(data: Path, root: Mapping[str, Any]) -> None:
    # This is defense in depth.  No CLI option or stage accepts a fresh root.
    if "fresh" in data.name.lower() or root.get("seed_registry") == "explicit_fresh_confirmation":
        raise RuntimeError("fresh confirmation data are forbidden in v6")
    if root.get("fresh_seed_manifest_sha256") not in (None, ""):
        raise RuntimeError("fresh confirmation data are forbidden in v6")


def _source_files() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        here / "openvla_etsf_counterfactual_oof_v6.py",
        here / "openvla_etsf_event_world_model.py",
        here / "train_openvla_etsf_counterfactual.py",
        here / "evaluate_openvla_etsf_oof_prediction_diagnostics.py",
        here / "openvla_etsf_prediction_repair.py",
        here / "openvla_etsf_counterfactual_oof.py",
    ]


def preregister(args: argparse.Namespace) -> None:
    _refuse(args.output)
    data, pretrained_path, event_spec = (
        args.data.resolve(), args.pretrained.resolve(), args.event_spec.resolve()
    )
    root_path = data / "manifest.json"
    for path in (root_path, pretrained_path, event_spec):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = _json(root_path)
    _reject_fresh(data, root)
    groups = root.get("groups")
    if root.get("status") != "complete" or int(root.get("schema_version", -1)) != 5:
        raise RuntimeError("v6 requires a complete schema-v5 development collection")
    if not isinstance(groups, list) or len(groups) != EXPECTED_GROUPS or int(
        root.get("completed", -1)
    ) != EXPECTED_GROUPS:
        raise RuntimeError("v6 requires exactly 250 completed development groups")
    descriptors = scan_group_descriptors([data])
    if len(descriptors) != EXPECTED_GROUPS or any(d.schema_version != 5 for d in descriptors):
        raise RuntimeError("v6 requires exactly 250 schema-v5 group files")
    if {d.policy for d in descriptors} != {"openvla"}:
        raise RuntimeError("v6 is bound to OpenVLA candidate groups")
    checkpoint, factual_config = load_pretrained(pretrained_path)
    if (
        not factual_config.structured_events
        or factual_config.action_rank_residual
        or factual_config.action_rank_success_only
    ):
        raise RuntimeError("v6 requires a structured factual checkpoint without rank mode")
    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping) or contract.get("event_spec_sha256") != sha256(event_spec):
        raise RuntimeError("v6 factual/event-spec contract mismatch")
    source_contract = {
        "data_root": str(data),
        "collector_manifest": str(root_path),
        "collector_manifest_sha256": sha256(root_path),
        "event_spec": str(event_spec),
        "event_spec_sha256": sha256(event_spec),
        "pretrained": str(pretrained_path),
        "pretrained_sha256": sha256(pretrained_path),
        "factual_core_tensor_sha256": _tensor_sha(checkpoint["model"], core_only=True),
        "implementation_files": [
            {"path": str(path), "sha256": sha256(path)} for path in _source_files()
        ],
        "development_group_files": [
            {"logical_key": d.logical_key, "path": str(Path(d.path).resolve()),
             "sha256": sha256(Path(d.path).resolve()), "schema_version": d.schema_version}
            for d in descriptors
        ],
        "fresh_seed_manifest": None,
        "fresh_labels_read": False,
    }
    manifest = make_nested_oof_manifest(
        [d.logical_key for d in descriptors], source_contract=source_contract,
        semantic_dim=factual_config.semantic_dim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, manifest)


def _validate_source(args: argparse.Namespace, manifest: Mapping[str, Any]):
    data, pretrained_path, event_spec = (
        args.data.resolve(), args.pretrained.resolve(), args.event_spec.resolve()
    )
    root = _json(data / "manifest.json")
    _reject_fresh(data, root)
    descriptors = scan_group_descriptors([data])
    validate_nested_oof_manifest(manifest, [d.logical_key for d in descriptors])
    source = manifest.get("source_contract")
    if not isinstance(source, Mapping):
        raise RuntimeError("v6 source contract missing")
    expected = {
        data / "manifest.json": source.get("collector_manifest_sha256"),
        pretrained_path: source.get("pretrained_sha256"),
        event_spec: source.get("event_spec_sha256"),
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"v6 frozen source changed: {path}")
    for row in source.get("implementation_files", []):
        path = Path(str(row.get("path", ""))).resolve()
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"v6 implementation changed: {path}")
    by_record = {str(row["logical_key"]): row for row in source.get("development_group_files", [])}
    if set(by_record) != {d.logical_key for d in descriptors}:
        raise RuntimeError("v6 development identities changed")
    for d in descriptors:
        row = by_record[d.logical_key]
        path = Path(d.path).resolve()
        if path != Path(str(row["path"])).resolve() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"v6 development group changed: {d.logical_key}")
    checkpoint, factual_config = load_pretrained(pretrained_path)
    if _tensor_sha(checkpoint["model"], core_only=True) != source.get("factual_core_tensor_sha256"):
        raise RuntimeError("v6 factual tensor state changed")
    config = dataclasses.replace(
        factual_config, action_rank_residual=True, action_rank_success_only=True
    )
    return descriptors, checkpoint, config, checkpoint["contract"]


def _audit_checkpoint(path: Path, factual: Mapping[str, Any], config: Any) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(saved["model"], strict=True)
    optimization = configure_action_rank_training(model, freeze_factual_core=True)
    if tuple(optimization["trainable_parameter_names"]) != TRAINABLE_PARAMETER_NAMES:
        raise RuntimeError("v6 checkpoint exposed an unregistered parameter")
    if int(optimization["trainable_parameter_count"]) != FORMAL_TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("v6 checkpoint rank-head capacity changed")
    factual_sha = _tensor_sha(factual["model"], core_only=True)
    saved_sha = _tensor_sha(saved["model"], core_only=True)
    if saved_sha != factual_sha:
        raise RuntimeError("v6 factual core is not bit-exact after training")
    stored = saved.get("contract", {}).get("action_rank_optimization")
    if not isinstance(stored, Mapping) or stored.get("factual_core_trainable_parameters") != 0:
        raise RuntimeError("v6 checkpoint lacks frozen-core optimizer proof")
    return {
        "factual_core_before_sha256": factual_sha,
        "factual_core_after_sha256": saved_sha,
        "factual_core_bit_exact": True,
        **optimization,
    }


def _load_groups(keys, by_key, config, context, event_spec):
    calibrations, objects, bodies, policies = context
    return load_descriptor_groups(
        [by_key[key] for key in keys], config, objects, bodies, policies,
        calibrations=calibrations,
        expected_event_spec_sha256=sha256(event_spec.resolve()),
    )


def _candidate_count(group: Any) -> int:
    names = tuple(group.candidate_names)
    deployment = tuple(DEPLOYMENT_CANDIDATE_NAMES)
    if names[: len(deployment)] != deployment or len(names) not in (4, 5):
        raise RuntimeError("v6 group violates deployment candidate order")
    return len(deployment)


def _diagnostic_compatibility_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly expose v6 outer ownership to the unchanged v5 diagnostics.

    The prediction diagnostics consume only dimensions/owner folds.  This
    signed adapter does not alter v6 selection, training, or authorization.
    """
    value: dict[str, Any] = {
        "format": "etsf_counterfactual_five_fold_oof_v1",
        "status": "preregistered",
        "expected_groups": 250,
        "fold_count": 5,
        "groups_per_fold": 50,
        "training_steps": 250,
        "development_groups": list(manifest["development_groups"]),
        "source_contract": {
            "v6_preregistration_sha256": manifest["preregistration_sha256"],
            "purpose": "prediction_diagnostics_owner_fold_adapter_only",
        },
        "folds": [
            {
                "fold_id": int(fold["outer_fold_id"]),
                "training_groups": list(fold["training_groups"]),
                "oof_holdout_groups": list(fold["oof_holdout_groups"]),
                "training_group_count": 200,
                "oof_holdout_group_count": 50,
                "checkpoint_selection": "fixed_final_step_no_holdout_early_stop",
            }
            for fold in manifest["outer_folds"]
        ],
    }
    value["preregistration_sha256"] = v5_canonical_sha256(value)
    return value


@torch.inference_mode()
def _predict(path: Path, groups: Sequence[Any], config: Any, mean, std, device, fold_id: int):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    rows = []
    for group in groups:
        batch = move_batch(collate_groups([group], object_mean=mean, object_std=std), device)
        output = forward_model(model, batch)
        residual = group_action_rank_residual(
            model, output, batch["group_index"], batch["baseline_mask"], detach_features=True
        )
        count = _candidate_count(group)
        base = output["success_logit"][:count].float().cpu().numpy()
        rank = residual[:count].float().cpu().numpy()
        labels = np.asarray(group.success[:count], dtype=np.float32)
        full_candidate_count = len(group.candidate_names)
        full_success = np.asarray(group.success, dtype=np.float32)
        full_base = output["success_logit"][:full_candidate_count].float().cpu().numpy()
        full_aleatoric = counterfactual_aleatoric_uncertainty(
            model, output, batch
        )[:full_candidate_count].float().cpu().numpy()
        object_mean_tensor = torch.as_tensor(
            mean, device=device, dtype=output["object_delta_mean"].dtype
        )
        object_std_tensor = torch.as_tensor(
            std, device=device, dtype=output["object_delta_mean"].dtype
        )
        physical_object_mean = (
            output["object_delta_mean"] * object_std_tensor + object_mean_tensor
        )
        physical_object_log_scale = output["object_delta_log_scale"] + torch.log(
            object_std_tensor.clamp_min(1e-8)
        )
        physical_object_target = (
            batch["object_delta"].float().cpu().numpy()
            * np.asarray(std, dtype=np.float32)
            + np.asarray(mean, dtype=np.float32)
        )
        # The factual core is shared and bit-exact for all v6 heads.  The
        # unchanged diagnostics API has a frozen three-member axis, so the one
        # factual prediction is repeated exactly; means stay exact and
        # epistemic spread correctly remains zero for this single model.
        member = lambda value: np.repeat(value[None], 3, axis=0)
        rows.append({
            "logical_key": group.logical_key,
            "fold_id": fold_id,
            "candidate_names": list(group.candidate_names[:count]),
            "baseline_index": 0,
            "success": labels,
            "frozen_base_success_only": base,
            "residual_only": rank,
            "frozen_base_plus_residual": base + rank,
            # The selector consumes exactly this one preregistered score.
            "success_only_scores": base + rank,
            "diagnostic_candidate_names": list(group.candidate_names),
            "diagnostic_success": full_success,
            "member_success_logits": member(full_base),
            "member_aleatoric": member(full_aleatoric),
            "success_prediction_source": (
                "frozen_factual_success_logit_bit_exact_no_rank_residual"
            ),
            "success_head_training_contract": {
                "format": SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT,
                "status": FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS,
                "owner_fold_id": fold_id,
                "success_head_updated_on_owner_training_groups": False,
                "factual_core_bit_exact": True,
                "positive_weight": None,
            },
            "diagnostic_member_axis": (
                "single_frozen_factual_prediction_repeated_for_legacy_three_member_axis"
            ),
            "structured_predictions": {
                "format": STRUCTURED_ROW_FORMAT,
                "sample_names": list(batch["candidate_names"]),
                "terminal_mask": batch["terminal_mask"].bool().cpu().numpy(),
                "structured_mask": batch["structured_mask"].bool().cpu().numpy(),
                "dense_mask": batch["dense_mask"].bool().cpu().numpy(),
                "duration_observed": batch["duration_observed"].bool().cpu().numpy(),
                "current_event_id": batch["current_event_id"].long().cpu().numpy(),
                "clock_event_id": batch["clock_event_id"].long().cpu().numpy(),
                "next_event_id": batch["next_event_id"].long().cpu().numpy(),
                "next_reached_event_id": batch["next_reached_event_id"].long().cpu().numpy(),
                "body_id": batch["body_id"].long().cpu().numpy(),
                "policy_id": batch["policy_id"].long().cpu().numpy(),
                "duration": batch["duration"].float().cpu().numpy(),
                "success": batch["success"].float().cpu().numpy(),
                "outcome_id": batch["outcome_id"].long().cpu().numpy(),
                "trajectory_regress": batch["trajectory_regress"].bool().cpu().numpy(),
                "trajectory_recovery": batch["trajectory_recovery"].bool().cpu().numpy(),
                "object_delta": physical_object_target,
                "post_predicates": batch["post_predicates"].float().cpu().numpy(),
                "predicate_names": list(config.predicate_names),
                "recovery_supervised": bool(config.recovery_supervised),
                "member_next_event_logits": member(output["next_event_logits"].float().cpu().numpy()),
                "member_next_reached_event_logits": member(output["next_reached_event_logits"].float().cpu().numpy()),
                "member_post_predicate_logits": member(output["post_predicate_logits"].float().cpu().numpy()),
                "member_duration_log_mean": member(output["duration_selected_log_mean"].float().cpu().numpy()),
                "member_duration_log_scale": member(output["duration_selected_log_scale"].float().cpu().numpy()),
                "member_reach_logit": member(output["reach_logit"].float().cpu().numpy()),
                "member_object_delta_mean": member(physical_object_mean.float().cpu().numpy()),
                "member_object_delta_log_scale": member(physical_object_log_scale.float().cpu().numpy()),
                "member_outcome_logits": member(output["outcome_logits"].float().cpu().numpy()),
            },
        })
    return rows


def _train_one(seed, factual, config, groups, mean, std, output, device, args, contract):
    path = train_member(
        seed=seed, pretrained=factual, config=config, train_groups=groups,
        validation_groups=groups, object_mean=mean, object_std=std,
        output=output, device=device, args=_training_args(args), contract=contract,
    )
    return path, _audit_checkpoint(path, factual, config)


def run_fold(args: argparse.Namespace) -> None:
    _refuse(args.output)
    manifest = _json(args.oof_manifest.resolve())
    descriptors, factual, config, world_contract = _validate_source(args, manifest)
    if not 0 <= args.fold_id < OUTER_FOLDS:
        raise ValueError("fold-id must lie in [0,4]")
    fold = manifest["outer_folds"][args.fold_id]
    by_key = {d.logical_key: d for d in descriptors}
    context = _event_context(args.event_spec.resolve(), world_contract)
    mean, std = _normalization(factual, config)
    args.output.mkdir(parents=True, exist_ok=False)
    device = _device()
    inner_rows, inner_artifacts = [], []
    for child in fold["inner_folds"]:
        inner_id = int(child["inner_fold_id"])
        # At this point only this inner-training label set is opened.
        train_groups = _load_groups(child["training_groups"], by_key, config, context, args.event_spec)
        inner_root = args.output / "inner" / f"fold_{inner_id}"
        inner_root.mkdir(parents=True, exist_ok=False)
        seed = INNER_SEED_BASE + 10 * args.fold_id + inner_id
        checkpoint, audit = _train_one(
            seed, factual, config, train_groups, mean, std, inner_root, device, args,
            {"protocol": FORMAT, "role": "inner_crossfit", "outer_fold_id": args.fold_id,
             "inner_fold_id": inner_id, "training_groups": list(child["training_groups"]),
             "selection_holdout_groups": list(child["selection_holdout_groups"]),
             "outer_holdout_access": "forbidden"},
        )
        # Inner holdout labels are first opened after its checkpoint is fixed.
        holdout = _load_groups(child["selection_holdout_groups"], by_key, config, context, args.event_spec)
        predictions = _predict(checkpoint, holdout, config, mean, std, device, args.fold_id)
        inner_rows.extend(predictions)
        pred_path = inner_root / "crossfit_predictions.pt"
        atomic_torch_save(pred_path, {"format": FORMAT, "rows": predictions})
        inner_artifacts.append({
            "inner_fold_id": inner_id, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint), "predictions": str(pred_path.resolve()),
            "predictions_sha256": sha256(pred_path), "frozen_core_audit": audit,
            "training_group_count": 150, "holdout_group_count": 50,
            "holdout_loaded_after_checkpoint": True,
        })
    if len(inner_rows) != 200 or {r["logical_key"] for r in inner_rows} != set(fold["training_groups"]):
        raise RuntimeError("v6 inner crossfit coverage is incomplete")
    inner_selection = select_inner_guard(inner_rows)
    selection_path = args.output / "inner_selection.json"
    atomic_json(selection_path, {
        "format": FORMAT, "outer_fold_id": args.fold_id,
        "source": "four_inner_crossfit_holdouts_only", "outer_labels_read": False,
        "selection": inner_selection,
    })
    # Refit after selection, still without opening the outer holdout.
    outer_train = _load_groups(fold["training_groups"], by_key, config, context, args.event_spec)
    outer_root = args.output / "outer_refit"
    outer_root.mkdir(parents=True, exist_ok=False)
    outer_checkpoint, outer_audit = _train_one(
        OUTER_SEED_BASE + args.fold_id, factual, config, outer_train, mean, std,
        outer_root, device, args,
        {"protocol": FORMAT, "role": "outer_refit", "outer_fold_id": args.fold_id,
         "training_groups": list(fold["training_groups"]),
         "inner_selection": str(selection_path.resolve()),
         "outer_holdout_access": "after_checkpoint_only"},
    )
    # First and only outer-label access occurs here.
    outer_groups = _load_groups(fold["oof_holdout_groups"], by_key, config, context, args.event_spec)
    outer_rows = _predict(outer_checkpoint, outer_groups, config, mean, std, device, args.fold_id)
    decisions = apply_outer_policy(outer_rows, inner_selection)
    for row, decision in zip(outer_rows, decisions, strict=True):
        row["nested_policy_decision"] = decision
    predictions_path = args.output / "outer_oof_predictions.pt"
    atomic_torch_save(predictions_path, {
        "format": FORMAT, "fold_id": args.fold_id,
        "preregistration_sha256": manifest["preregistration_sha256"],
        "inner_selection_sha256": sha256(selection_path), "rows": outer_rows,
    })
    atomic_json(args.output / "fold_summary.json", {
        "format": FORMAT, "status": "complete", "fold_id": args.fold_id,
        "preregistration_sha256": manifest["preregistration_sha256"],
        "inner_crossfit_group_count": 200, "outer_training_group_count": 200,
        "outer_oof_group_count": 50, "inner_artifacts": inner_artifacts,
        "inner_selection": str(selection_path.resolve()),
        "inner_selection_sha256": sha256(selection_path),
        "outer_checkpoint": str(outer_checkpoint.resolve()),
        "outer_checkpoint_sha256": sha256(outer_checkpoint),
        "outer_frozen_core_audit": outer_audit,
        "outer_labels_first_loaded_after_inner_selection_and_outer_refit": True,
        "outer_predictions": str(predictions_path.resolve()),
        "outer_predictions_sha256": sha256(predictions_path),
        "fresh_confirmation_labels_read": False,
    })


def _score_report(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    decisions, pairs = [], []
    for row in rows:
        scores = np.asarray(row[field], dtype=float)
        labels = np.asarray(row["success"], dtype=float)
        selected = int(np.argmax(scores)); baseline = int(row["baseline_index"])
        delta = float(labels[selected] - labels[baseline])
        decisions.append(delta)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] != labels[j]:
                    pairs.append(float((scores[i] - scores[j]) * (labels[i] - labels[j]) > 0))
    delta = np.asarray(decisions)
    return {
        "groups": len(rows), "top1_success": float(np.mean([
            np.asarray(r["success"])[int(np.argmax(np.asarray(r[field])))] for r in rows
        ])),
        "baseline_success": float(np.mean([np.asarray(r["success"])[int(r["baseline_index"])] for r in rows])),
        "mean_paired_success_delta": float(delta.mean()),
        "helpful_changes": int((delta > 0).sum()), "harmful_changes": int((delta < 0).sum()),
        "success_pair_accuracy": float(np.mean(pairs)) if pairs else None,
    }


def _unconditional_inference(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (EXPECTED_GROUPS,):
        raise RuntimeError("v6 nested inference requires exactly 250 group deltas")
    generator = np.random.default_rng(20260903)
    means = np.empty(10_000, dtype=np.float64)
    for start in range(0, len(means), 1000):
        count = min(1000, len(means) - start)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    helpful, harmful = int((delta > 0).sum()), int((delta < 0).sum())
    nonzero = helpful + harmful
    tail = min(helpful, harmful)
    sign_p = (
        min(1.0, 2.0 * sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2.0**nonzero))
        if nonzero
        else 1.0
    )
    return {
        "estimand": "unconditional_equal_group_success_delta_including_zeros",
        "mean": float(delta.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260903,
        "exact_two_sided_sign_test_p": float(sign_p),
        "sign_test_nonzero_changes": nonzero,
    }


def select_oof(args: argparse.Namespace) -> None:
    _refuse(args.output)
    manifest = _json(args.oof_manifest.resolve())
    validate_nested_oof_manifest(manifest, manifest["development_groups"])
    rows, artifacts = [], []
    for fold_id in range(OUTER_FOLDS):
        root = args.fold_root.resolve() / f"fold_{fold_id}"
        summary_path = root / "fold_summary.json"
        summary = _json(summary_path)
        if summary.get("status") != "complete" or int(summary.get("fold_id", -1)) != fold_id:
            raise RuntimeError(f"v6 fold {fold_id} incomplete")
        if summary.get("preregistration_sha256") != manifest["preregistration_sha256"]:
            raise RuntimeError("v6 fold preregistration mismatch")
        if summary.get("outer_labels_first_loaded_after_inner_selection_and_outer_refit") is not True:
            raise RuntimeError("v6 fold lacks outer-access ordering proof")
        prediction = Path(str(summary["outer_predictions"]))
        if not prediction.is_file(): prediction = root / prediction.name
        if sha256(prediction) != summary["outer_predictions_sha256"]:
            raise RuntimeError("v6 outer prediction SHA mismatch")
        payload = torch.load(prediction, map_location="cpu", weights_only=False)
        fold_rows = payload.get("rows", [])
        if len(fold_rows) != 50 or {r["logical_key"] for r in fold_rows} != set(
            manifest["outer_folds"][fold_id]["oof_holdout_groups"]
        ):
            raise RuntimeError("v6 outer prediction identity mismatch")
        rows.extend(fold_rows)
        artifacts.append({"fold_id": fold_id, "summary": str(summary_path.resolve()),
                          "summary_sha256": sha256(summary_path),
                          "predictions": str(prediction.resolve()), "predictions_sha256": sha256(prediction)})
    if len(rows) != EXPECTED_GROUPS or len({r["logical_key"] for r in rows}) != EXPECTED_GROUPS:
        raise RuntimeError("v6 combined outer OOF coverage mismatch")
    nested_delta = np.asarray([r["nested_policy_decision"]["success_delta"] for r in rows], dtype=float)
    # The development gate is descriptive only and can never authorize fresh access.
    inference = _unconditional_inference(nested_delta)
    foldwise = {}
    for fold_id in range(OUTER_FOLDS):
        fold_rows = [r for r in rows if int(r["fold_id"]) == fold_id]
        fold_delta = np.asarray(
            [r["nested_policy_decision"]["success_delta"] for r in fold_rows],
            dtype=float,
        )
        foldwise[str(fold_id)] = {
            "groups": int(len(fold_delta)),
            "changed_groups": int(
                sum(r["nested_policy_decision"]["changed"] for r in fold_rows)
            ),
            "helpful_changes": int((fold_delta > 0).sum()),
            "harmful_changes": int((fold_delta < 0).sum()),
            "mean_unconditional_success_delta": float(fold_delta.mean()),
        }
    nested = {
        "groups": len(rows), "changed_groups": int(sum(r["nested_policy_decision"]["changed"] for r in rows)),
        "helpful_changes": int((nested_delta > 0).sum()), "harmful_changes": int((nested_delta < 0).sum()),
        "mean_paired_success_delta": float(nested_delta.mean()),
        "unconditional_inference": inference,
        "per_outer_fold": foldwise,
    }
    gate = (
        inference["bootstrap_95_ci"][0] > 0.0
        and inference["exact_two_sided_sign_test_p"] < 0.05
        and nested["helpful_changes"] > nested["harmful_changes"]
        and nested["changed_groups"] >= 10
    )
    selection = {
        "format": SELECTION_FORMAT, "status": "complete_development_only",
        "preregistration_sha256": manifest["preregistration_sha256"],
        "oof_prediction_groups": EXPECTED_GROUPS,
        "score_ablations": {field: _score_report(rows, field) for field in (
            "frozen_base_success_only", "residual_only", "frozen_base_plus_residual")},
        "nested_crossfit_policy": nested,
        "development_gate_pass": bool(gate),
        "development_gate_is_descriptive_only": True,
        "authorization": {"authorized": False, "fresh_confirmation_allowed": False,
                          "reason": "v6_development_only_fresh_forbidden_even_if_gate_passes"},
        "fold_artifacts": artifacts, "fresh_confirmation_labels_read": False,
    }
    selection["selection_sha256"] = canonical_sha256(selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compatibility = _diagnostic_compatibility_manifest(manifest)
    compatibility_path = args.output.with_name("oof_prediction_diagnostics_manifest.json")
    diagnostic_path = args.output.with_name("oof_prediction_diagnostics.json")
    _refuse(compatibility_path)
    _refuse(diagnostic_path)
    diagnostic_rows = []
    for original in rows:
        row = dict(original)
        row["candidate_names"] = list(row.pop("diagnostic_candidate_names"))
        row["success"] = np.asarray(row.pop("diagnostic_success"))
        diagnostic_rows.append(row)
    diagnostics = build_oof_prediction_diagnostics(diagnostic_rows, compatibility)
    diagnostics["v6_preregistration_sha256"] = manifest["preregistration_sha256"]
    diagnostics["single_factual_member_axis"] = (
        "repeated_exactly_three_times_for_unchanged_diagnostics_schema"
    )
    diagnostics.pop("diagnostics_sha256", None)
    diagnostics["diagnostics_sha256"] = v5_canonical_sha256(diagnostics)
    validate_oof_prediction_diagnostics(diagnostics, compatibility, require_structured=True)
    atomic_json(compatibility_path, compatibility)
    atomic_json(diagnostic_path, diagnostics)
    selection["prediction_diagnostics"] = {
        "path": str(diagnostic_path.resolve()), "sha256": sha256(diagnostic_path),
        "compatibility_manifest": str(compatibility_path.resolve()),
        "compatibility_manifest_sha256": sha256(compatibility_path),
        "descriptive_only_not_guard_input": True,
    }
    selection.pop("selection_sha256")
    selection["selection_sha256"] = canonical_sha256(selection)
    atomic_json(args.output, selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, required=True)
    common.add_argument("--pretrained", type=Path, required=True)
    common.add_argument("--event-spec", type=Path, required=True)
    pre = sub.add_parser("preregister", parents=[common]); pre.add_argument("--output", type=Path, required=True)
    fold = sub.add_parser("fold", parents=[common])
    fold.add_argument("--oof-manifest", type=Path, required=True); fold.add_argument("--fold-id", type=int, required=True)
    fold.add_argument("--output", type=Path, required=True); fold.add_argument("--num-workers", type=int, default=2)
    select = sub.add_parser("select")
    select.add_argument("--oof-manifest", type=Path, required=True); select.add_argument("--fold-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "preregister": preregister(args)
    elif args.stage == "fold": run_fold(args)
    elif args.stage == "select": select_oof(args)
    else: raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
