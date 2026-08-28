#!/usr/bin/env python3
"""Run the preregistered five-fold OOF counterfactual development protocol.

Stages are explicit and resumability is fail-closed:

* ``preregister`` scans only development identities and freezes five folds;
* ``fold`` trains three fixed-step members on 4/5 groups, then first loads the
  fold's heldout 1/5 and writes raw OOF predictions;
* ``select`` reduces the five raw artifacts into one frozen scoring/guard audit;
* ``final`` is permitted only after OOF authorization and refits three members
  on all frozen development groups using the already frozen constants.

No stage accepts a fresh-confirmation path or seed manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluate_openvla_etsf_oof_prediction_diagnostics import (
    STRUCTURED_ROW_FORMAT,
    build_oof_prediction_diagnostics,
    validate_oof_prediction_diagnostics,
)
from openvla_etsf_counterfactual_oof import (
    DEPLOYMENT_CANDIDATE_NAMES,
    FOLD_COUNT,
    FORMAT,
    MEMBER_SEEDS,
    SELECTION_FORMAT,
    canonical_sha256,
    make_oof_folds,
    oof_dimensions,
    oof_training_steps,
    reduce_oof_predictions,
    validate_oof_folds,
)
from openvla_etsf_event_world_model import ActionConditionedEventWorldModel
from train_openvla_etsf_counterfactual import (
    atomic_json,
    atomic_torch_save,
    canonical_policy_mapping,
    collate_groups,
    counterfactual_aleatoric_uncertainty,
    counterfactual_success_logit,
    forward_model,
    load_descriptor_groups,
    load_pretrained,
    move_batch,
    scan_group_descriptors,
    sha256,
    train_member,
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal OOF training requires CUDA bf16 support")
    return torch.device("cuda:0")


def _refuse_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"OOF output already exists; refusing overwrite: {path}")


def _validate_source(
    *,
    data: Path,
    pretrained: Path,
    event_spec: Path,
    fold_manifest: Mapping[str, Any],
) -> tuple[list[Any], Mapping[str, Any], Any, Mapping[str, Any]]:
    descriptors = scan_group_descriptors([data.resolve()])
    expected_groups, _, _ = oof_dimensions(fold_manifest)
    if len(descriptors) != expected_groups:
        raise RuntimeError(
            "OOF source count differs from the frozen fold manifest"
        )
    if any(descriptor.schema_version != 5 for descriptor in descriptors):
        raise RuntimeError("OOF development refuses non-schema-v5 groups")
    if {descriptor.policy for descriptor in descriptors} != {"openvla"}:
        raise RuntimeError("OOF protocol is bound to OpenVLA candidate groups")
    validate_oof_folds(fold_manifest, [row.logical_key for row in descriptors])
    source = fold_manifest.get("source_contract")
    if not isinstance(source, Mapping):
        raise RuntimeError("OOF preregistration lacks source provenance")
    if str(source.get("event_spec_sha256", "")) != sha256(event_spec.resolve()):
        raise RuntimeError("OOF event spec changed after preregistration")
    if str(source.get("pretrained_sha256", "")) != sha256(pretrained.resolve()):
        raise RuntimeError("OOF factual initialization changed after preregistration")
    collector_manifest = data.resolve() / "manifest.json"
    if not collector_manifest.is_file() or str(
        source.get("collector_manifest_sha256", "")
    ) != sha256(collector_manifest):
        raise RuntimeError("OOF collector manifest changed after preregistration")
    checkpoint, config = load_pretrained(pretrained.resolve())
    if not config.structured_events:
        raise RuntimeError("OOF requires a structured factual initialization")
    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("factual checkpoint lacks a contract")
    if str(contract.get("event_spec_sha256", "")) != sha256(event_spec.resolve()):
        raise RuntimeError("factual/event-spec contract mismatch")
    # The factual checkpoint intentionally predates the within-group action
    # residual.  Every formal OOF member adds that zero-initialized branch and
    # ``train_member`` permits only those newly missing checkpoint keys.
    config = dataclasses.replace(config, action_rank_residual=True)
    recorded_files = source.get("development_group_files")
    if not isinstance(recorded_files, Sequence) or isinstance(
        recorded_files, (str, bytes)
    ):
        raise RuntimeError("OOF preregistration lacks development file provenance")
    by_key = {
        str(row.get("logical_key")): row
        for row in recorded_files
        if isinstance(row, Mapping)
    }
    if set(by_key) != {descriptor.logical_key for descriptor in descriptors}:
        raise RuntimeError("OOF development file identities changed")
    for descriptor in descriptors:
        path = Path(descriptor.path).resolve()
        row = by_key[descriptor.logical_key]
        if Path(str(row.get("path", ""))).resolve() != path or str(
            row.get("sha256", "")
        ) != sha256(path):
            raise RuntimeError(f"OOF development HDF5 changed: {descriptor.logical_key}")
    return descriptors, checkpoint, config, contract


def _event_context(
    event_spec: Path, world_contract: Mapping[str, Any]
) -> tuple[Mapping[str, Mapping[str, Any]], list[str], Mapping[str, int], Mapping[str, int]]:
    value = _json(event_spec.resolve())
    calibrations = value.get("calibration")
    if not isinstance(calibrations, Mapping):
        raise RuntimeError("event spec lacks task calibration")
    object_names = world_contract.get("object_names")
    body_to_id = world_contract.get("body_to_id")
    policy_to_id = canonical_policy_mapping(world_contract.get("policy_to_id"))
    if not isinstance(object_names, Sequence) or isinstance(object_names, (str, bytes)):
        raise RuntimeError("factual checkpoint lacks object names")
    if not isinstance(body_to_id, Mapping):
        raise RuntimeError("factual checkpoint lacks body/policy registrations")
    if "openvla" not in policy_to_id:
        raise RuntimeError(
            "factual checkpoint policy mapping does not canonicalize to openvla"
        )
    return (
        calibrations,
        [str(name) for name in object_names],
        body_to_id,
        policy_to_id,
    )


def _normalization(
    checkpoint: Mapping[str, Any], config: Any
) -> tuple[np.ndarray, np.ndarray]:
    value = checkpoint.get("normalization")
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "OOF requires normalization frozen by factual training; fold-local "
            "normalization would change the prediction contract"
        )
    mean = np.asarray(value.get("object_delta_mean"), dtype=np.float32)
    std = np.asarray(value.get("object_delta_std"), dtype=np.float32)
    if mean.shape != (config.object_delta_dim,) or std.shape != mean.shape:
        raise RuntimeError("factual object normalization shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("factual object normalization is invalid")
    return mean, std


def _training_args(args: argparse.Namespace, training_steps: int) -> Namespace:
    # These values are part of make_oof_folds().hyperparameters.  The outer
    # holdout is evaluated exactly once because eval_every == steps and early
    # stopping is disabled.
    return Namespace(
        unfreeze_semantic=False,
        learning_rate=1e-4,
        weight_decay=1e-4,
        amp="bf16",
        groups_per_batch=8,
        num_workers=args.num_workers,
        min_relative_support=5,
        success_weight=1.0,
        outcome_weight=0.2,
        pairwise_weight=0.75,
        listwise_weight=0.5,
        group_centered_weight=1.0,
        baseline_contrast_weight=1.5,
        event_weight=1.0,
        relative_weight=0.5,
        destination_weight=0.5,
        predicate_weight=0.5,
        reach_weight=0.75,
        duration_weight=0.5,
        object_weight=0.5,
        latent_weight=0.5,
        grad_clip=2.0,
        steps=training_steps,
        eval_every=training_steps,
        early_stopping_patience=0,
    )


@torch.inference_mode()
def raw_oof_predictions(
    *,
    member_paths: Sequence[Path],
    groups: Sequence[Any],
    fold_id: int,
    config: Any,
    object_mean: np.ndarray,
    object_std: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    models = []
    duration_scales = []
    for path in member_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ActionConditionedEventWorldModel(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        models.append(model)
        duration_scales.append(float(checkpoint["duration_scale"]))
    if len(set(duration_scales)) != 1:
        raise RuntimeError("OOF fold members disagree on duration normalization")
    duration_scale = duration_scales[0]
    event_values = torch.linspace(0.0, 1.0, config.num_events, device=device)
    rows = []
    for group in groups:
        batch = move_batch(
            collate_groups(
                [group], object_mean, object_std, include_auxiliary=True
            ),
            device,
        )
        candidate_count = int(getattr(group, "candidate_count", len(group.candidate_names)))
        logits = []
        event_progress = []
        duration = []
        aleatoric = []
        structured_members: dict[str, list[np.ndarray]] = {
            "member_next_event_logits": [],
            "member_next_reached_event_logits": [],
            "member_post_predicate_logits": [],
            "member_duration_log_mean": [],
            "member_duration_log_scale": [],
            "member_reach_logit": [],
            "member_object_delta_mean": [],
            "member_object_delta_log_scale": [],
            "member_outcome_logits": [],
        }
        for model in models:
            output = forward_model(model, batch)
            probability = torch.softmax(output["next_event_logits"], -1)
            logits.append(
                counterfactual_success_logit(model, output, batch)[:candidate_count]
                .float()
                .cpu()
                .numpy()
            )
            event_progress.append(
                (probability * event_values.to(probability))
                .sum(-1)[:candidate_count]
                .float()
                .cpu()
                .numpy()
            )
            duration.append(
                (
                    torch.expm1(
                        output["duration_selected_log_mean"].clamp(0.0, 12.0)
                    )
                    / max(duration_scale, 1.0)
                )[:candidate_count]
                .float()
                .cpu()
                .numpy()
            )
            aleatoric.append(
                counterfactual_aleatoric_uncertainty(model, output, batch)[:candidate_count]
                .float()
                .cpu()
                .numpy()
            )
            # These optional arrays are diagnostic-only.  They are never read
            # by ``reduce_oof_predictions`` and therefore cannot change the
            # preregistered scoring/guard decision.  Schema-v5 formal groups
            # require every head below; the getattr guard only preserves small
            # legacy unit-test doubles that predate structured raw rows.
            if int(getattr(group, "schema_version", -1)) == 5:
                required = (
                    "next_event_logits",
                    "next_reached_event_logits",
                    "post_predicate_logits",
                    "duration_selected_log_mean",
                    "duration_selected_log_scale",
                    "reach_logit",
                    "object_delta_mean",
                    "object_delta_log_scale",
                    "outcome_logits",
                )
                missing = [name for name in required if name not in output]
                if missing:
                    raise RuntimeError(
                        f"structured OOF diagnostic heads missing: {missing}"
                    )
                object_mean_tensor = torch.as_tensor(
                    object_mean, device=device, dtype=output["object_delta_mean"].dtype
                )
                object_std_tensor = torch.as_tensor(
                    object_std, device=device, dtype=output["object_delta_mean"].dtype
                )
                physical_object_mean = (
                    output["object_delta_mean"] * object_std_tensor + object_mean_tensor
                )
                physical_object_log_scale = output["object_delta_log_scale"] + torch.log(
                    object_std_tensor.clamp_min(1e-8)
                )
                structured_members["member_next_event_logits"].append(
                    output["next_event_logits"].float().cpu().numpy()
                )
                structured_members["member_next_reached_event_logits"].append(
                    output["next_reached_event_logits"].float().cpu().numpy()
                )
                structured_members["member_post_predicate_logits"].append(
                    output["post_predicate_logits"].float().cpu().numpy()
                )
                structured_members["member_duration_log_mean"].append(
                    output["duration_selected_log_mean"].float().cpu().numpy()
                )
                structured_members["member_duration_log_scale"].append(
                    output["duration_selected_log_scale"].float().cpu().numpy()
                )
                structured_members["member_reach_logit"].append(
                    output["reach_logit"].float().cpu().numpy()
                )
                structured_members["member_object_delta_mean"].append(
                    physical_object_mean.float().cpu().numpy()
                )
                structured_members["member_object_delta_log_scale"].append(
                    physical_object_log_scale.float().cpu().numpy()
                )
                structured_members["member_outcome_logits"].append(
                    output["outcome_logits"].float().cpu().numpy()
                )
        baseline = [
            index
            for index, name in enumerate(group.candidate_names)
            if name == "deterministic"
        ]
        if len(baseline) != 1:
            raise RuntimeError("OOF group lacks a unique deterministic baseline")
        row = {
                "logical_key": group.logical_key,
                "fold_id": fold_id,
                "member_success_logits": np.stack(logits),
                "member_event_progress": np.stack(event_progress),
                "member_normalized_duration": np.stack(duration),
                "member_aleatoric": np.stack(aleatoric),
                "success": group.success.copy(),
                "steps": group.steps.copy(),
                "candidate_distance": group.candidate_distance.copy(),
                "baseline_index": baseline[0],
                "candidate_names": list(group.candidate_names),
            }
        if int(getattr(group, "schema_version", -1)) == 5:
            physical_object_target = (
                batch["object_delta"].float().cpu().numpy()
                * np.asarray(object_std, dtype=np.float32)
                + np.asarray(object_mean, dtype=np.float32)
            )
            row["structured_predictions"] = {
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
                **{
                    name: np.stack(member_values)
                    for name, member_values in structured_members.items()
                },
            }
        rows.append(row)
    return rows


def preregister(args: argparse.Namespace) -> None:
    _refuse_output(args.output)
    data = args.data.resolve()
    event_spec = args.event_spec.resolve()
    pretrained = args.pretrained.resolve()
    for path in (data / "manifest.json", event_spec, pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    root_manifest = _json(data / "manifest.json")
    if root_manifest.get("status") != "complete" or int(
        root_manifest.get("schema_version", -1)
    ) != 5:
        raise RuntimeError("OOF requires a complete schema-v5 development collection")
    if root_manifest.get("seed_registry") == "explicit_fresh_confirmation" or root_manifest.get(
        "fresh_seed_manifest_sha256"
    ) not in (None, ""):
        raise RuntimeError("fresh confirmation data cannot become OOF development data")
    root_groups = root_manifest.get("groups")
    if not isinstance(root_groups, list) or int(
        root_manifest.get("completed", len(root_groups))
    ) != len(root_groups):
        raise RuntimeError("OOF collector manifest group mirrors are incomplete")
    descriptors = scan_group_descriptors([data])
    if len(descriptors) not in (100, 250) or len(root_groups) != len(descriptors) or any(
        descriptor.schema_version != 5 for descriptor in descriptors
    ):
        raise RuntimeError(
            "OOF preregistration requires exactly 100 or 250 schema-v5 groups"
        )
    source_contract = {
        "data_root": str(data),
        "collector_manifest": str((data / "manifest.json").resolve()),
        "collector_manifest_sha256": sha256(data / "manifest.json"),
        "collector_seed_registry": root_manifest.get("seed_registry"),
        "development_group_count": len(descriptors),
        "event_spec": str(event_spec),
        "event_spec_sha256": sha256(event_spec),
        "pretrained": str(pretrained),
        "pretrained_sha256": sha256(pretrained),
        "trainer": str(Path(__file__).resolve()),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "development_group_files": [
            {
                "logical_key": descriptor.logical_key,
                "schema_version": descriptor.schema_version,
                "path": descriptor.path,
                "sha256": sha256(Path(descriptor.path)),
            }
            for descriptor in descriptors
        ],
        "fresh_seed_manifest": None,
        "fresh_labels_read": False,
    }
    manifest = make_oof_folds(
        [descriptor.logical_key for descriptor in descriptors],
        source_contract=source_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, manifest)


def run_fold(args: argparse.Namespace) -> None:
    _refuse_output(args.output)
    manifest = _json(args.oof_manifest.resolve())
    training_steps = oof_training_steps(manifest)
    descriptors, pretrained, config, world_contract = _validate_source(
        data=args.data,
        pretrained=args.pretrained,
        event_spec=args.event_spec,
        fold_manifest=manifest,
    )
    if not 0 <= args.fold_id < FOLD_COUNT:
        raise ValueError("fold-id must lie in [0,4]")
    fold = manifest["folds"][args.fold_id]
    by_key = {descriptor.logical_key: descriptor for descriptor in descriptors}
    calibrations, object_names, body_to_id, policy_to_id = _event_context(
        args.event_spec, world_contract
    )
    object_mean, object_std = _normalization(pretrained, config)
    # Only the frozen 4/5 training labels are loaded before fixed-step members are
    # complete.  The heldout descriptors remain identity-only at this point.
    training_groups = load_descriptor_groups(
        [by_key[key] for key in fold["training_groups"]],
        config,
        object_names,
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
        expected_event_spec_sha256=sha256(args.event_spec.resolve()),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        "trainer": "five_fold_oof_fixed_step_member_v1",
        "development_protocol": FORMAT,
        "oof_preregistration_sha256": manifest["preregistration_sha256"],
        "fold_id": args.fold_id,
        "training_groups": list(fold["training_groups"]),
        "oof_holdout_groups": list(fold["oof_holdout_groups"]),
        "holdout_label_access": "after_all_fixed_step_member_checkpoints_only",
        "checkpoint_selection": "fixed_final_step_no_holdout_early_stop",
        "training_steps": training_steps,
        "training_step_rationale": manifest["training_step_rationale"],
        "event_spec_sha256": sha256(args.event_spec.resolve()),
        "pretrained_sha256": sha256(args.pretrained.resolve()),
        "body_to_id": dict(body_to_id),
        "policy_to_id": dict(policy_to_id),
        "object_names": object_names,
        "fresh_confirmation_access": "forbidden",
    }
    training_args = _training_args(args, training_steps)
    device = _device()
    member_paths = [
        train_member(
            seed=seed,
            pretrained=pretrained,
            config=config,
            train_groups=training_groups,
            # This is a final-step train diagnostic, not a selector.  Passing
            # the train groups prevents any holdout label read before training.
            validation_groups=training_groups,
            object_mean=object_mean,
            object_std=object_std,
            output=args.output,
            device=device,
            args=training_args,
            contract=contract,
        )
        for seed in MEMBER_SEEDS
    ]
    # First heldout label access happens only here, after every member exists.
    holdout_groups = load_descriptor_groups(
        [by_key[key] for key in fold["oof_holdout_groups"]],
        config,
        object_names,
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
        expected_event_spec_sha256=sha256(args.event_spec.resolve()),
    )
    rows = raw_oof_predictions(
        member_paths=member_paths,
        groups=holdout_groups,
        fold_id=args.fold_id,
        config=config,
        object_mean=object_mean,
        object_std=object_std,
        device=device,
    )
    raw_path = args.output / "oof_predictions.pt"
    atomic_torch_save(
        raw_path,
        {
            "format": FORMAT,
            "fold_id": args.fold_id,
            "oof_preregistration_sha256": manifest["preregistration_sha256"],
            "member_seeds": list(MEMBER_SEEDS),
            "member_checkpoints": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in member_paths
            ],
            "rows": rows,
        },
    )
    atomic_json(
        args.output / "fold_summary.json",
        {
            "format": FORMAT,
            "status": "complete",
            "fold_id": args.fold_id,
            "training_group_count": len(training_groups),
            "oof_holdout_group_count": len(holdout_groups),
            "oof_preregistration_sha256": manifest["preregistration_sha256"],
            "checkpoint_selection": "fixed_final_step_no_holdout_early_stop",
            "holdout_labels_first_loaded_after_member_checkpoints": True,
            "raw_predictions": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path)},
            "fresh_confirmation_labels_read": False,
        },
    )


def select_oof(args: argparse.Namespace) -> None:
    _refuse_output(args.output)
    manifest = _json(args.oof_manifest.resolve())
    validate_oof_folds(manifest, manifest["development_groups"])
    rows = []
    fold_artifacts = []
    for fold_id in range(FOLD_COUNT):
        root = args.fold_root.resolve() / f"fold_{fold_id}"
        summary_path = root / "fold_summary.json"
        summary = _json(summary_path)
        if summary.get("status") != "complete" or int(
            summary.get("fold_id", -1)
        ) != fold_id:
            raise RuntimeError(f"OOF fold {fold_id} is incomplete")
        if summary.get("oof_preregistration_sha256") != manifest.get(
            "preregistration_sha256"
        ):
            raise RuntimeError("OOF fold used a different preregistration")
        raw_record = summary.get("raw_predictions")
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("OOF fold lacks raw prediction provenance")
        raw_path = Path(str(raw_record.get("path", "")))
        if not raw_path.is_file():
            raw_path = root / raw_path.name
        if not raw_path.is_file() or sha256(raw_path) != str(
            raw_record.get("sha256", "")
        ):
            raise RuntimeError("OOF raw prediction artifact SHA mismatch")
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        if int(payload.get("fold_id", -1)) != fold_id or payload.get(
            "oof_preregistration_sha256"
        ) != manifest.get("preregistration_sha256"):
            raise RuntimeError("OOF raw prediction contract mismatch")
        rows.extend(payload.get("rows", []))
        fold_artifacts.append(
            {
                "fold_id": fold_id,
                "summary": str(summary_path.resolve()),
                "summary_sha256": sha256(summary_path),
                "raw_predictions": str(raw_path.resolve()),
                "raw_predictions_sha256": sha256(raw_path),
            }
        )
    selection = reduce_oof_predictions(rows, manifest)
    selection["fold_artifacts"] = fold_artifacts
    # Re-sign after adding immutable fold artifact provenance.
    selection.pop("selection_sha256", None)
    selection["selection_sha256"] = canonical_sha256(selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Prediction-head quality is a separate descriptive artifact.  Keeping it
    # outside ``selection`` preserves the exact authorization input/schema and
    # prevents a metric from becoming an unregistered decision threshold.
    diagnostic_path = args.output.with_name("oof_prediction_diagnostics.json")
    _refuse_output(diagnostic_path)
    diagnostics = build_oof_prediction_diagnostics(rows, manifest)
    diagnostics["fold_artifacts"] = fold_artifacts
    diagnostics.pop("diagnostics_sha256", None)
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    atomic_json(diagnostic_path, diagnostics)
    atomic_json(args.output, selection)


def _validate_selection(
    path: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    selection = _json(path.resolve())
    if selection.get("format") != SELECTION_FORMAT or selection.get("status") != "complete":
        raise RuntimeError("OOF selection artifact is incomplete")
    unsigned = dict(selection)
    recorded = str(unsigned.pop("selection_sha256", ""))
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("OOF selection artifact changed")
    if selection.get("oof_preregistration_sha256") != manifest.get(
        "preregistration_sha256"
    ):
        raise RuntimeError("OOF selection used another preregistration")
    authorization = selection.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get("authorized") is not True:
        raise RuntimeError("OOF guard is not authorized; final refit/fresh50 are forbidden")
    if authorization.get("fresh_confirmation_allowed") is not True:
        raise RuntimeError("OOF selection does not authorize one-shot fresh confirmation")
    candidate_contract = selection.get("candidate_authorization_contract")
    if (
        not isinstance(candidate_contract, Mapping)
        or candidate_contract.get("deployment_candidate_names")
        != list(DEPLOYMENT_CANDIDATE_NAMES)
        or candidate_contract.get(
            "calibration_scoring_guard_use_deployment_candidates_only"
        )
        is not True
    ):
        raise RuntimeError("OOF selection candidate schedule is not deployment-matched")
    return selection


def final_refit(args: argparse.Namespace) -> None:
    _refuse_output(args.output)
    manifest = _json(args.oof_manifest.resolve())
    selection = _validate_selection(args.selection.resolve(), manifest)
    diagnostics_path = args.selection.resolve().with_name(
        "oof_prediction_diagnostics.json"
    )
    diagnostics = _json(diagnostics_path)
    validate_oof_prediction_diagnostics(
        diagnostics, manifest, require_structured=True
    )
    expected_groups, training_groups_per_fold, holdout_groups_per_fold = (
        oof_dimensions(manifest)
    )
    training_steps = oof_training_steps(manifest)
    descriptors, pretrained, config, world_contract = _validate_source(
        data=args.data,
        pretrained=args.pretrained,
        event_spec=args.event_spec,
        fold_manifest=manifest,
    )
    calibrations, object_names, body_to_id, policy_to_id = _event_context(
        args.event_spec, world_contract
    )
    object_mean, object_std = _normalization(pretrained, config)
    groups = load_descriptor_groups(
        descriptors,
        config,
        object_names,
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
        expected_event_spec_sha256=sha256(args.event_spec.resolve()),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    predicate_contract = world_contract.get("predicate_contract")
    if not isinstance(predicate_contract, Mapping):
        raise RuntimeError("factual checkpoint lacks structured predicate provenance")
    candidate_contract = {
        "baseline_candidate_name": "deterministic",
        "fallback_index": 0,
        **dict(selection["candidate_authorization_contract"]),
    }
    source_files = manifest["source_contract"]["development_group_files"]
    contract = {
        "trainer": f"five_fold_oof_authorized_refit_all{expected_groups}_v1",
        "development_protocol": FORMAT,
        "development_groups": list(manifest["development_groups"]),
        "oof_folds": manifest["folds"],
        "oof_preregistration_sha256": manifest["preregistration_sha256"],
        "oof_selection": str(args.selection.resolve()),
        "oof_selection_sha256": sha256(args.selection.resolve()),
        "oof_prediction_diagnostics": str(diagnostics_path),
        "oof_prediction_diagnostics_sha256": sha256(diagnostics_path),
        "oof_authorization": selection["authorization"],
        "training_groups": list(manifest["development_groups"]),
        "train_groups": list(manifest["development_groups"]),
        "validation_groups": [],
        "sealed_test_groups": [],
        "sealed_test_files": [],
        "sealed_test_access": "fresh50_absent_not_read_one_shot_only_after_oof_authorization",
        "group_files": list(source_files),
        "training_steps": training_steps,
        "training_step_rationale": manifest["training_step_rationale"],
        "checkpoint_selection": "fixed_final_step_no_development_metric_selection",
        "event_spec": str(args.event_spec.resolve()),
        "event_spec_sha256": sha256(args.event_spec.resolve()),
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": sha256(args.pretrained.resolve()),
        "object_names": object_names,
        "body_to_id": dict(body_to_id),
        "policy_to_id": dict(policy_to_id),
        "state_contracts": dict(world_contract.get("state_contracts", {})),
        "predicate_contract": dict(predicate_contract),
        "candidate_contract": candidate_contract,
        "scoring_selection_contract": {
            "selection_data": (
                f"five_fold_oof_all{expected_groups}_development_only"
            ),
            "training_groups_per_fold": training_groups_per_fold,
            "holdout_groups_per_fold": holdout_groups_per_fold,
            "oof_preregistration_sha256": manifest["preregistration_sha256"],
            "oof_selection_sha256": sha256(args.selection.resolve()),
            "fresh_confirmation_labels_read": False,
        },
        "fresh_confirmation": {
            "authorized": True,
            "required_registry": "explicit_fresh_confirmation",
            "required_groups": 50,
            "access": "not_read_during_development_or_refit",
            "one_shot": True,
        },
    }
    training_args = _training_args(args, training_steps)
    device = _device()
    member_paths = [
        train_member(
            seed=seed,
            pretrained=pretrained,
            config=config,
            train_groups=groups,
            validation_groups=groups,
            object_mean=object_mean,
            object_std=object_std,
            output=args.output,
            device=device,
            args=training_args,
            contract=contract,
        )
        for seed in MEMBER_SEEDS
    ]
    member_payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in member_paths
    ]
    duration_scales = {float(payload["duration_scale"]) for payload in member_payloads}
    if len(duration_scales) != 1:
        raise RuntimeError("final members disagree on duration scale")
    duration_scale = next(iter(duration_scales))
    scoring = dict(selection["scoring"])
    calibration = dict(selection["success_calibration"])
    guard = dict(selection["guard"])
    scoring_selection = dict(selection["scoring_selection"])
    aggregate_path = args.output / "counterfactual_ensemble.pt"
    payload = {
        "format": "etsf_counterfactual_ensemble_v1",
        "models": [payload["model"] for payload in member_payloads],
        "member_seeds": list(MEMBER_SEEDS),
        "config": dataclasses.asdict(config),
        "contract": contract,
        "predicate_contract": dict(predicate_contract),
        "candidate_contract": candidate_contract,
        "normalization": {
            "object_delta_mean": object_mean,
            "object_delta_std": object_std,
        },
        "duration_scale": duration_scale,
        "success_calibration": calibration,
        "guard": guard,
        "scoring": scoring,
        "scoring_selection": scoring_selection,
    }
    atomic_torch_save(aggregate_path, payload)
    ensemble_manifest = {
        **payload,
        "models": None,
        "ensemble_checkpoint": {
            "path": str(aggregate_path.resolve()),
            "sha256": sha256(aggregate_path),
        },
        "members": [
            {"path": str(path.resolve()), "sha256": sha256(path), "seed": seed}
            for path, seed in zip(member_paths, MEMBER_SEEDS)
        ],
        "normalization": {
            "object_delta_mean": object_mean.tolist(),
            "object_delta_std": object_std.tolist(),
        },
        "test_policy": "fresh50_one_shot_only_after_oof_authorization",
    }
    ensemble_manifest.pop("models")
    atomic_json(args.output / "ensemble_manifest.json", ensemble_manifest)
    atomic_json(
        args.output / "training_summary.json",
        {
            "format": FORMAT,
            "status": "complete",
            "development_groups": expected_groups,
            "member_seeds": list(MEMBER_SEEDS),
            "fixed_training_steps": training_steps,
            "oof_authorized": True,
            "fresh_confirmation_labels_read": False,
            "fresh_confirmation_next_action": "one_shot_fresh50_evaluator_only",
            "oof_prediction_diagnostics": str(diagnostics_path),
            "oof_prediction_diagnostics_sha256": sha256(diagnostics_path),
            "ensemble_manifest": str((args.output / "ensemble_manifest.json").resolve()),
            "ensemble_manifest_sha256": sha256(args.output / "ensemble_manifest.json"),
        },
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prereg = subparsers.add_parser("preregister")
    add_common(prereg)
    prereg.add_argument("--output", type=Path, required=True)

    fold = subparsers.add_parser("fold")
    add_common(fold)
    fold.add_argument("--oof-manifest", type=Path, required=True)
    fold.add_argument("--fold-id", type=int, required=True)
    fold.add_argument("--output", type=Path, required=True)
    fold.add_argument("--num-workers", type=int, default=2)

    select = subparsers.add_parser("select")
    select.add_argument("--oof-manifest", type=Path, required=True)
    select.add_argument("--fold-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("final")
    add_common(final)
    final.add_argument("--oof-manifest", type=Path, required=True)
    final.add_argument("--selection", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "preregister":
        preregister(args)
    elif args.stage == "fold":
        run_fold(args)
    elif args.stage == "select":
        select_oof(args)
    elif args.stage == "final":
        final_refit(args)
    else:  # pragma: no cover
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
