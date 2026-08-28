#!/usr/bin/env python3
"""Strict leave-one-body-out (LOBO) canonical event-world-model protocol.

This program intentionally separates two validation roles:

* source validation selects checkpoints and never contains the held-out body;
* frozen target development is opened only after every checkpoint is selected.

The ordinary test lane is sealed and never passed to a payload loader.  Target
groups assigned to the ordinary train lane are also sealed, rather than being
repurposed as adaptation data.  Thus this is a zero-target-label transfer
measurement, not joint multi-body training.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_multibody_canonical_event_world_model as core


FORMAT = "etsf_multibody_leave_one_body_out_v1"
SPLIT_FORMAT = "etsf_multibody_lobo_frozen_split_v1"
ALLOWED_HELD_OUT_BODIES = ("piper", "ur5-wsg")
VARIANTS = ("source_body_clock", "body_agnostic")


@dataclasses.dataclass(frozen=True)
class LoboSplit:
    source_train: tuple[core.GroupDescriptor, ...]
    source_validation: tuple[core.GroupDescriptor, ...]
    target_development: tuple[core.GroupDescriptor, ...]
    target_unused_train: tuple[core.GroupDescriptor, ...]
    sealed_test: tuple[core.GroupDescriptor, ...]

    def lanes(self) -> dict[str, tuple[core.GroupDescriptor, ...]]:
        return {
            "source_train": self.source_train,
            "source_validation": self.source_validation,
            "target_development": self.target_development,
            "target_unused_train": self.target_unused_train,
            "sealed_test": self.sealed_test,
        }


def _canonical_held_out_body(value: str) -> str:
    body = core.canonical_body_name(value)
    if body not in ALLOWED_HELD_OUT_BODIES:
        raise ValueError(
            f"held-out body must be one of {ALLOWED_HELD_OUT_BODIES}, got {body!r}"
        )
    return body


def strict_leave_one_body_out_split(
    descriptors: Sequence[core.GroupDescriptor],
    *,
    held_out_body: str,
    split_seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.10,
) -> LoboSplit:
    """Create label-free source-selection and target-development lanes.

    The base split is stratified before any HDF5 payload is opened.  Only the
    held-out body's preassigned validation lane becomes target development.
    Its train lane stays unused and the complete test lane stays sealed.
    """

    held_out = _canonical_held_out_body(held_out_body)
    base = core.strict_group_split(
        descriptors,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    split = LoboSplit(
        source_train=tuple(row for row in base["train"] if row.body != held_out),
        source_validation=tuple(
            row for row in base["validation"] if row.body != held_out
        ),
        target_development=tuple(
            row for row in base["validation"] if row.body == held_out
        ),
        target_unused_train=tuple(
            row for row in base["train"] if row.body == held_out
        ),
        sealed_test=tuple(base["test"]),
    )
    if not split.source_train or not split.source_validation:
        raise ValueError("leave-one-body-out split has an empty source lane")
    if not split.target_development or not split.target_unused_train:
        raise ValueError("leave-one-body-out split has insufficient target groups")
    if any(row.body == held_out for row in split.source_train):
        raise RuntimeError("held-out body leaked into source train")
    if any(row.body == held_out for row in split.source_validation):
        raise RuntimeError("held-out body leaked into source validation")
    memberships = {
        name: {row.logical_group for row in rows}
        for name, rows in split.lanes().items()
    }
    names = tuple(memberships)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if memberships[left] & memberships[right]:
                raise RuntimeError(f"LOBO lane leakage: {left}/{right}")
    expected = {row.logical_group for row in descriptors}
    if set().union(*memberships.values()) != expected:
        raise RuntimeError("LOBO lanes omitted one or more logical groups")
    return split


def _identity_payload(rows: Sequence[core.GroupDescriptor]) -> dict[str, Any]:
    identities = sorted(row.logical_group for row in rows)
    bodies = sorted({row.body for row in rows})
    policies = sorted({row.policy for row in rows})
    return {
        "groups": len(identities),
        "identity_sha256": core.canonical_json_sha256(identities),
        "identities": identities,
        "bodies": bodies,
        "policies": policies,
    }


def build_frozen_split_plan(
    split: LoboSplit,
    *,
    held_out_body: str,
    split_seed: int,
    binding_audit: Mapping[str, Any],
) -> dict[str, Any]:
    held_out = _canonical_held_out_body(held_out_body)
    plan: dict[str, Any] = {
        "format": SPLIT_FORMAT,
        "held_out_body": held_out,
        "split_seed": int(split_seed),
        "split_inputs": dict(binding_audit["input_sha256"]),
        "event_spec_sha256": str(binding_audit["event_spec_sha256"]),
        "split_unit": "body_policy_task_seed_logical_group",
        "labels_used_for_assignment": False,
        "checkpoint_selection_lane": "source_validation",
        "final_evaluation_lane": "target_development",
        "target_development_used_for_checkpoint_selection": False,
        "target_unused_train_payload_opened": 0,
        "sealed_test_group_hdf5_opened": 0,
        "lanes": {
            name: _identity_payload(rows) for name, rows in split.lanes().items()
        },
    }
    plan["sha256"] = core.canonical_json_sha256(plan)
    return plan


def verify_frozen_split_plan(
    path: Path,
    expected_file_sha256: str,
    recomputed: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = core.reject_forbidden_path(path, "LOBO split plan")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if len(expected_file_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in expected_file_sha256.lower()
    ):
        raise ValueError("LOBO split plan expected SHA-256 is malformed")
    actual_file_sha256 = core.sha256_file(resolved)
    if actual_file_sha256 != expected_file_sha256.lower():
        raise ValueError("LOBO split plan file SHA-256 mismatch")
    frozen = json.loads(resolved.read_text(encoding="utf-8"))
    unsigned = dict(frozen)
    internal = unsigned.pop("sha256", None)
    if internal != core.canonical_json_sha256(unsigned):
        raise ValueError("LOBO split plan internal SHA-256 mismatch")
    if frozen != dict(recomputed):
        raise ValueError("LOBO split plan differs from label-free recomputation")
    return {
        "path": str(resolved),
        "file_sha256": actual_file_sha256,
        "logical_sha256": internal,
        "verified_against_current_inputs": True,
    }


def fit_source_only_action_normalization(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit observed source schemas and freeze unseen schemas to identity."""

    observed = sorted(
        {
            int(row["action_schema_id"])
            for row in rows
            if bool(row["action_available"])
        }
    )
    if not observed:
        raise ValueError("source train has no observed action schema")
    receipt = core.fit_train_action_normalization(
        rows, required_schema_ids=observed
    )
    unsigned = dict(receipt)
    unsigned.pop("sha256", None)
    schemas = dict(unsigned["schemas"])
    for schema_id, name in sorted(core.ACTION_SCHEMA_NAMES.items()):
        if schema_id in observed:
            item = dict(schemas[name])
            item["transfer_status"] = "fitted_from_source_train_only"
            schemas[name] = item
        else:
            schemas[name] = {
                "schema_id": int(schema_id),
                "train_rows": 0,
                "train_logical_groups": 0,
                "valid_action_steps": 0,
                "mean": [0.0] * core.ACTION_DIM,
                "std": [1.0] * core.ACTION_DIM,
                "transfer_status": "unseen_source_schema_frozen_identity",
            }
    unsigned.update(
        {
            "format": "etsf_lobo_source_only_action_normalization_v1",
            "schemas": schemas,
            "observed_source_schema_ids": observed,
            "unseen_schema_ids": sorted(set(core.ACTION_SCHEMA_NAMES) - set(observed)),
            "held_out_rows_used": 0,
        }
    )
    unsigned["sha256"] = core.canonical_json_sha256(unsigned)
    return unsigned


def materialize_source_rows(
    split: LoboSplit,
    event_spec: Mapping[str, Any],
    held_out_body: str,
    *,
    loader: Callable[[Sequence[core.GroupDescriptor], Mapping[str, Any]], list[dict[str, Any]]] = core.load_rows,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    held_out = _canonical_held_out_body(held_out_body)
    selected = split.source_train + split.source_validation
    if any(row.body == held_out for row in selected):
        raise RuntimeError("held-out descriptor reached the source payload boundary")
    train_rows = loader(split.source_train, event_spec)
    validation_rows = loader(split.source_validation, event_spec)
    if any(str(row["body"]) == held_out for row in train_rows + validation_rows):
        raise RuntimeError("held-out payload row reached source fitting")
    return train_rows, validation_rows


def materialize_target_development_rows(
    split: LoboSplit,
    event_spec: Mapping[str, Any],
    held_out_body: str,
    *,
    checkpoints_selected: bool,
    loader: Callable[[Sequence[core.GroupDescriptor], Mapping[str, Any]], list[dict[str, Any]]] = core.load_rows,
) -> list[dict[str, Any]]:
    if not checkpoints_selected:
        raise RuntimeError("target development cannot open before checkpoint selection")
    held_out = _canonical_held_out_body(held_out_body)
    if any(row.body != held_out for row in split.target_development):
        raise RuntimeError("target development contains a source body")
    rows = loader(split.target_development, event_spec)
    if any(str(row["body"]) != held_out for row in rows):
        raise RuntimeError("target development loader returned a source body")
    return rows


def body_mapping(
    source_rows: Sequence[Mapping[str, Any]], held_out_body: str, variant: str
) -> tuple[dict[str, int], int | None]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown LOBO variant {variant!r}")
    held_out = _canonical_held_out_body(held_out_body)
    sources = sorted({str(row["body"]) for row in source_rows})
    if held_out in sources:
        raise RuntimeError("held-out body present while building body mapping")
    if variant == "body_agnostic":
        return {body: 0 for body in sources + [held_out]}, None
    mapping = {body: index for index, body in enumerate(sources)}
    mapping[held_out] = len(mapping)
    return mapping, mapping[held_out]


def reserve_zero_target_clock_row(
    model: core.MultibodyCanonicalEventWorldModel, target_body_id: int | None
) -> Any | None:
    """Keep the unseen target clock coefficient exactly zero during fitting."""

    if target_body_id is None:
        return None
    weight = model.clock.body_beta.weight
    if not 0 <= target_body_id < len(weight):
        raise ValueError("reserved target body id is out of range")
    with torch.no_grad():
        weight[target_body_id].zero_()

    def zero_reserved_gradient(gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        result[target_body_id].zero_()
        return result

    return weight.register_hook(zero_reserved_gradient)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if not len(labels):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1]
            if index + 1 < bins
            else probabilities <= edges[index + 1]
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(value)


def _multiclass_ece(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float | None:
    if not len(labels):
        return None
    predictions = probabilities.argmax(1)
    return _ece(
        (predictions == labels).astype(np.float64),
        probabilities.max(1),
        bins,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def evaluate_lobo_ensemble(
    models: Sequence[core.MultibodyCanonicalEventWorldModel],
    rows: Sequence[Mapping[str, Any]],
    body_to_id: Mapping[str, int],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if len(models) != 5:
        raise ValueError("LOBO epistemic evaluation requires exactly five members")
    loader = DataLoader(
        core.TransitionDataset(rows, body_to_id),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    gathered: dict[str, list[np.ndarray]] = defaultdict(list)
    for model in models:
        model.eval()
    for raw in loader:
        batch = core._move_batch(raw, device)
        outputs = [model(batch) for model in models]
        post_members = torch.stack(
            [torch.softmax(item["post_event_logits"], -1) for item in outputs]
        )
        next_members = torch.stack(
            [torch.softmax(item["next_event_logits"], -1) for item in outputs]
        )
        success_members = torch.stack(
            [torch.sigmoid(item["success_logit"]) for item in outputs]
        )
        duration_members = torch.stack(
            [
                torch.expm1(item["duration_selected_log_mean"]).clamp_min(0.0)
                for item in outputs
            ]
        )
        object_members = torch.stack(
            [item["object_delta_mean"] for item in outputs]
        )
        payload = {
            "post_label": batch["post_event_id"],
            "post_mask": batch["post_event_mask"],
            "post_probability": post_members.mean(0),
            "post_mutual_information": -(
                post_members.mean(0).clamp_min(1e-9)
                * post_members.mean(0).clamp_min(1e-9).log()
            ).sum(-1)
            - (
                -(post_members.clamp_min(1e-9) * post_members.clamp_min(1e-9).log())
                .sum(-1)
                .mean(0)
            ),
            "next_label": batch["next_event_id"],
            "next_mask": batch["next_event_mask"],
            "next_probability": next_members.mean(0),
            "next_mutual_information": -(
                next_members.mean(0).clamp_min(1e-9)
                * next_members.mean(0).clamp_min(1e-9).log()
            ).sum(-1)
            - (
                -(next_members.clamp_min(1e-9) * next_members.clamp_min(1e-9).log())
                .sum(-1)
                .mean(0)
            ),
            "duration": batch["duration"],
            "duration_observed": batch["duration_observed"] * batch["duration_mask"],
            "duration_prediction": duration_members.mean(0),
            "duration_epistemic_std": duration_members.std(0, correction=0),
            "success": batch["success"],
            "success_mask": batch["success_mask"],
            "success_probability": success_members.mean(0),
            "success_epistemic_std": success_members.std(0, correction=0),
            "object": batch["object_delta"],
            # Geometry exists even where the target action vector is unavailable.
            "object_prediction": object_members.mean(0),
            "object_epistemic_std": object_members.std(0, correction=0).mean(-1),
        }
        for key, tensor in payload.items():
            gathered[key].append(tensor.detach().cpu().numpy())
    values = {key: np.concatenate(parts) for key, parts in gathered.items()}
    post_mask = values["post_mask"] > 0.5
    next_mask = values["next_mask"] > 0.5
    duration_mask = values["duration_observed"] > 0.5
    success_mask = values["success_mask"] > 0.5
    post_labels = values["post_label"][post_mask].astype(np.int64)
    post_prob = values["post_probability"][post_mask]
    next_labels = values["next_label"][next_mask].astype(np.int64)
    next_prob = values["next_probability"][next_mask]
    success_labels = values["success"][success_mask]
    success_prob = values["success_probability"][success_mask]
    success_auc, success_auc_status = core._binary_auc(success_labels, success_prob)
    duration_error = np.abs(
        values["duration_prediction"][duration_mask]
        - values["duration"][duration_mask]
    )
    object_error_per_row = np.sqrt(
        np.mean(np.square(values["object_prediction"] - values["object"]), axis=1)
    )
    eps = 1e-9
    success_nll = (
        -np.mean(
            success_labels * np.log(success_prob.clip(eps, 1.0 - eps))
            + (1.0 - success_labels)
            * np.log((1.0 - success_prob).clip(eps, 1.0 - eps))
        )
        if len(success_labels)
        else None
    )
    return {
        "split": "frozen_target_development_only",
        "rows": len(rows),
        "post_event": {
            **core._event_metrics(post_labels, post_prob.argmax(1)),
            "ece_10bin": _multiclass_ece(post_labels, post_prob),
            "mean_epistemic_mutual_information": float(
                values["post_mutual_information"][post_mask].mean()
            )
            if post_mask.any()
            else None,
        },
        "next_event": {
            **core._event_metrics(next_labels, next_prob.argmax(1)),
            "ece_10bin": _multiclass_ece(next_labels, next_prob),
            "mean_epistemic_mutual_information": float(
                values["next_mutual_information"][next_mask].mean()
            )
            if next_mask.any()
            else None,
        },
        "observed_duration_mae": float(duration_error.mean())
        if len(duration_error)
        else None,
        "observed_duration_support": int(duration_mask.sum()),
        "duration_uncertainty": {
            "mean_epistemic_std": float(
                values["duration_epistemic_std"][duration_mask].mean()
            )
            if duration_mask.any()
            else None,
            "error_correlation": _safe_correlation(
                values["duration_epistemic_std"][duration_mask], duration_error
            ),
        },
        "success_brier": float(np.mean(np.square(success_prob - success_labels)))
        if len(success_labels)
        else None,
        "success_auroc": success_auc,
        "success_auroc_status": success_auc_status,
        "success_nll": float(success_nll) if success_nll is not None else None,
        "success_ece_10bin": _ece(success_labels, success_prob),
        "success_support": {
            "rows": int(len(success_labels)),
            "positive": int((success_labels > 0.5).sum()),
            "negative": int((success_labels <= 0.5).sum()),
        },
        "success_mean_epistemic_std": float(
            values["success_epistemic_std"][success_mask].mean()
        )
        if success_mask.any()
        else None,
        "object_rmse": float(
            np.sqrt(np.mean(np.square(values["object_prediction"] - values["object"])))
        ),
        "object_support": len(rows),
        "object_evaluation_contract": "geometry_delta_all_target_rows",
        "object_uncertainty": {
            "mean_epistemic_std": float(values["object_epistemic_std"].mean()),
            "error_correlation": _safe_correlation(
                values["object_epistemic_std"], object_error_per_row
            ),
        },
    }


def evaluate_source_only_baseline(
    baseline: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    post_labels = np.asarray(
        [int(row["post_event_id"]) for row in rows if bool(row["post_event_mask"])],
        dtype=np.int64,
    )
    next_labels = np.asarray(
        [int(row["next_event_id"]) for row in rows if bool(row["next_event_mask"])],
        dtype=np.int64,
    )
    post_prediction = np.full_like(post_labels, int(baseline["majority_post_event"]))
    next_prediction = np.full_like(next_labels, int(baseline["majority_next_event"]))
    durations: list[float] = []
    duration_predictions: list[float] = []
    successes: list[float] = []
    objects: list[np.ndarray] = []
    for row in rows:
        if bool(row["duration_mask"]) and bool(row["duration_observed"]):
            event = int(row["current_event_id"])
            durations.append(float(row["duration"]))
            # The target body key cannot exist in a source-only baseline.
            duration_predictions.append(
                float(
                    baseline["duration_median_by_event"].get(
                        str(event), baseline["duration_global_median"]
                    )
                )
            )
        if bool(row["success_mask"]):
            successes.append(float(row["success"]))
        objects.append(np.asarray(row["object_delta"], dtype=np.float64))
    success_array = np.asarray(successes, dtype=np.float64)
    success_scores = np.full_like(success_array, float(baseline["empirical_success"]))
    auc, auc_status = core._binary_auc(success_array, success_scores)
    object_array = np.stack(objects)
    return {
        "source": "source_train_only_statistical_baseline",
        "post_event": core._event_metrics(post_labels, post_prediction),
        "next_event": core._event_metrics(next_labels, next_prediction),
        "observed_duration_mae": float(
            np.mean(np.abs(np.asarray(duration_predictions) - np.asarray(durations)))
        )
        if durations
        else None,
        "observed_duration_support": len(durations),
        "success_brier": float(np.mean(np.square(success_scores - success_array)))
        if len(success_array)
        else None,
        "success_auroc": auc,
        "success_auroc_status": auc_status,
        "success_support": len(success_array),
        "object_rmse": float(np.sqrt(np.mean(np.square(object_array)))),
        "object_support": len(object_array),
        "target_rows_used_to_fit": 0,
    }


def _metric_delta(model: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {
        "post_event_macro_f1": (
            model["post_event"]["macro_f1"], reference["post_event"]["macro_f1"]
        ),
        "next_event_macro_f1": (
            model["next_event"]["macro_f1"], reference["next_event"]["macro_f1"]
        ),
        "observed_duration_mae": (
            model["observed_duration_mae"], reference["observed_duration_mae"]
        ),
        "success_brier": (model["success_brier"], reference["success_brier"]),
        "success_auroc": (model["success_auroc"], reference["success_auroc"]),
        "object_rmse": (model["object_rmse"], reference["object_rmse"]),
    }
    return {
        name: None if left is None or right is None else float(left) - float(right)
        for name, (left, right) in pairs.items()
    }


def grouped_target_metrics(
    models: Sequence[core.MultibodyCanonicalEventWorldModel],
    rows: Sequence[Mapping[str, Any]],
    body_to_id: Mapping[str, int],
    baseline: Mapping[str, Any],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "global": evaluate_lobo_ensemble(
            models, rows, body_to_id, device, batch_size=batch_size
        ),
        "by_body": {},
        "by_policy": {},
        "by_task": {},
    }
    for output_key, row_key in (
        ("by_body", "body"),
        ("by_policy", "policy"),
        ("by_task", "task"),
    ):
        values = sorted({str(row[row_key]) for row in rows})
        for value in values:
            subset = [row for row in rows if str(row[row_key]) == value]
            result[output_key][value] = evaluate_lobo_ensemble(
                models, subset, body_to_id, device, batch_size=batch_size
            )
    result["train_only_baseline"] = evaluate_source_only_baseline(baseline, rows)
    result["delta_vs_train_only_baseline"] = _metric_delta(
        result["global"], result["train_only_baseline"]
    )
    return result


def _binding_from_args(args: argparse.Namespace) -> core.InputBinding:
    return core.InputBinding(
        stage1_root=args.stage1_root,
        stage1_source_manifest=args.stage1_source_manifest,
        stage1_source_manifest_sha256=args.stage1_source_manifest_sha256,
        stage1_target_manifest=args.stage1_target_manifest,
        stage1_target_manifest_sha256=args.stage1_target_manifest_sha256,
        event_spec=args.event_spec,
        event_spec_sha256=args.event_spec_sha256,
        openvla_schema5_manifest=args.openvla_schema5_manifest,
        openvla_schema5_manifest_sha256=args.openvla_schema5_manifest_sha256,
    )


def _require_binding_args(args: argparse.Namespace) -> None:
    names = (
        "stage1_root",
        "stage1_source_manifest",
        "stage1_source_manifest_sha256",
        "stage1_target_manifest",
        "stage1_target_manifest_sha256",
        "event_spec",
        "event_spec_sha256",
        "openvla_schema5_manifest",
        "openvla_schema5_manifest_sha256",
        "held_out_body",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"mode {args.mode} requires binding arguments: {missing}")


def run_preflight(
    args: argparse.Namespace,
) -> tuple[core.InputBinding, dict[str, Any], LoboSplit, dict[str, Any]]:
    _require_binding_args(args)
    binding = _binding_from_args(args)
    audit = core.verify_input_bindings(binding)
    descriptors = core.scan_stage1_groups(binding) + core.scan_schema5_groups(binding)
    split = strict_leave_one_body_out_split(
        descriptors,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
    )
    plan = build_frozen_split_plan(
        split,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
        binding_audit=audit,
    )
    audit.update(
        {
            "protocol": FORMAT,
            "held_out_body": _canonical_held_out_body(args.held_out_body),
            "body_alias": core.body_alias_receipt(descriptors),
            "total_groups": len(descriptors),
            "split_plan_logical_sha256": plan["sha256"],
            "held_out_training_groups": 0,
            "held_out_source_validation_groups": 0,
            "target_development_used_for_checkpoint_selection": False,
            "target_unused_train_payload_opened": 0,
            "sealed_test_group_hdf5_opened": 0,
        }
    )
    return binding, audit, split, plan


def freeze_split(args: argparse.Namespace) -> dict[str, Any]:
    if args.split_plan_output is None:
        raise ValueError("freeze-split mode requires --split-plan-output")
    path = core.reject_forbidden_path(args.split_plan_output, "LOBO split plan output")
    if path.exists():
        raise FileExistsError("split plan output must be a new immutable path")
    _, _, _, plan = run_preflight(args)
    core.atomic_json(path, plan)
    return {
        "status": "frozen",
        "path": str(path),
        "file_sha256": core.sha256_file(path),
        "logical_sha256": plan["sha256"],
        "held_out_body": plan["held_out_body"],
        "payload_hdf5_opened": 0,
    }


def _variant_train(
    *,
    variant: str,
    args: argparse.Namespace,
    output: Path,
    source_train_rows: Sequence[Mapping[str, Any]],
    source_validation_rows: Sequence[Mapping[str, Any]],
    held_out_body: str,
    action_normalization: Mapping[str, Any],
    source_train_baseline: Mapping[str, Any],
    source_validation_baseline: Mapping[str, Any],
    protocol: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[Path], dict[str, int]]:
    mapping, reserved_target_id = body_mapping(
        source_train_rows, held_out_body, variant
    )
    train_dataset = core.TransitionDataset(source_train_rows, mapping)
    validation_dataset = core.TransitionDataset(source_validation_rows, mapping)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    action_mean, action_std = core.action_normalization_arrays(action_normalization)
    group_order = [str(row["logical_group"]) for row in source_train_rows]
    bootstrap = core.logical_group_bootstrap_weights(
        group_order, members=5, seed=args.split_seed
    )
    bootstrap_by_group = {
        group: bootstrap[:, index].tolist()
        for index, group in enumerate(group_order)
    }
    checkpoint_paths: list[Path] = []
    summaries = []
    variant_dir = output / variant
    variant_dir.mkdir()
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = core.MultibodyCanonicalEventWorldModel(
            core.ModelConfig(body_count=max(mapping.values()) + 1)
        ).to(device)
        model.action.set_normalization(
            torch.as_tensor(action_mean, device=device),
            torch.as_tensor(action_std, device=device),
        )
        hook = reserve_zero_target_clock_row(model, reserved_target_id)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=core.collate_rows,
        )
        iterator = iter(loader)
        best_key = None
        best_metrics = None
        best_step = 0
        checkpoint = variant_dir / f"member_{member:02d}_seed_{seed}_best.pt"
        last_loss = math.inf
        for step in range(1, args.steps + 1):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = core._move_batch(raw, device)
            weights = torch.tensor(
                [bootstrap_by_group[group][member] for group in raw["logical_group"]],
                device=device,
            )
            prediction = model(batch)
            loss, _ = core.compute_multitask_loss(
                prediction, batch, sample_weight=weights
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {variant} member {member} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if reserved_target_id is not None and not torch.equal(
                model.clock.body_beta.weight[reserved_target_id],
                torch.zeros_like(model.clock.body_beta.weight[reserved_target_id]),
            ):
                raise RuntimeError("reserved target clock row changed during source fit")
            last_loss = float(loss.detach())
            if step % args.eval_every and step != args.steps:
                continue
            metrics = core.evaluate_validation_model(model, validation_loader, device)
            metrics["split"] = "source_validation_only"
            score, components = core.validation_selection_score(
                metrics, source_validation_baseline
            )
            metrics["selection_score"] = score
            metrics["selection_components"] = components
            key = core.validation_selection_key(metrics, score, step)
            if best_key is None or key < best_key:
                best_key = key
                best_step = step
                best_metrics = metrics
                torch.save(
                    {
                        "format": FORMAT,
                        "variant": variant,
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(model.config),
                        "body_to_id": mapping,
                        "reserved_target_body_id": reserved_target_id,
                        "held_out_body": held_out_body,
                        "action_normalization": action_normalization,
                        "source_train_baseline": source_train_baseline,
                        "source_validation_baseline": source_validation_baseline,
                        "protocol": protocol,
                        "member": member,
                        "seed": seed,
                        "step": step,
                        "validation": metrics,
                    },
                    checkpoint,
                )
            model.train()
        if hook is not None:
            hook.remove()
        if best_metrics is None:
            raise RuntimeError(f"{variant} member {member} selected no checkpoint")
        checkpoint_paths.append(checkpoint)
        summaries.append(
            {
                "member": member,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": core.sha256_file(checkpoint),
                "best_step": best_step,
                "last_train_loss": last_loss,
                "best_source_validation": best_metrics,
            }
        )
    core.atomic_json(
        variant_dir / "source_selection_summary.json",
        {
            "format": FORMAT,
            "variant": variant,
            "held_out_body": held_out_body,
            "checkpoint_selection_split": "source_validation_only",
            "target_development_opened": 0,
            "test_group_hdf5_opened": 0,
            "members": summaries,
        },
    )
    return checkpoint_paths, mapping


def _restore_models(
    checkpoints: Sequence[Path],
    device: torch.device,
    *,
    expected_variant: str,
    expected_held_out_body: str,
    expected_mapping: Mapping[str, int],
) -> list[core.MultibodyCanonicalEventWorldModel]:
    if len(checkpoints) != 5:
        raise ValueError("target evaluation requires exactly five selected checkpoints")
    models = []
    members = set()
    for path in checkpoints:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format") != FORMAT:
            raise ValueError(f"checkpoint {path} has the wrong format")
        if payload.get("variant") != expected_variant:
            raise ValueError(f"checkpoint {path} has the wrong variant")
        if payload.get("held_out_body") != expected_held_out_body:
            raise ValueError(f"checkpoint {path} has the wrong held-out body")
        if payload.get("body_to_id") != dict(expected_mapping):
            raise ValueError(f"checkpoint {path} has the wrong body mapping")
        member = int(payload.get("member", -1))
        if member in members or not 0 <= member < 5:
            raise ValueError("checkpoint ensemble member ids are invalid")
        members.add(member)
        model = core.MultibodyCanonicalEventWorldModel(
            core.ModelConfig(**payload["config"])
        ).to(device)
        model.load_state_dict(payload["model"], strict=True)
        reserved = payload.get("reserved_target_body_id")
        if reserved is not None and not torch.equal(
            model.clock.body_beta.weight[int(reserved)],
            torch.zeros_like(model.clock.body_beta.weight[int(reserved)]),
        ):
            raise ValueError("checkpoint changed the reserved target clock row")
        models.append(model.eval())
    if members != set(range(5)):
        raise ValueError("checkpoint ensemble does not contain members 0..4")
    return models


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output is None:
        raise ValueError("train mode requires --output")
    if args.split_plan is None or args.split_plan_sha256 is None:
        raise ValueError("train mode requires a frozen split plan path and SHA-256")
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("steps and eval-every must be positive")
    output = core.reject_forbidden_path(args.output, "LOBO training output")
    if output.exists():
        raise FileExistsError("training output must be a new immutable path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    binding, audit, split, plan = run_preflight(args)
    frozen = verify_frozen_split_plan(
        args.split_plan, args.split_plan_sha256, plan
    )
    output.mkdir(parents=True)
    protocol = {
        **audit,
        "frozen_split_plan": frozen,
        "checkpoint_selection_split": "source_validation_only",
        "target_development_evaluation_timing": "after_all_checkpoint_selection",
        "target_development_used_for_checkpoint_selection": False,
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
    }
    core.atomic_json(output / "protocol_receipt_before_payload_open.json", protocol)
    event_spec = json.loads(binding.event_spec.read_text(encoding="utf-8"))
    source_train_rows, source_validation_rows = materialize_source_rows(
        split, event_spec, args.held_out_body
    )
    action_normalization = fit_source_only_action_normalization(source_train_rows)
    source_train_baseline = core.fit_train_baselines(source_train_rows)
    source_validation_baseline = core.evaluate_train_only_baselines(
        source_train_baseline, source_validation_rows
    )
    protocol.update(
        {
            "source_train_transitions": len(source_train_rows),
            "source_validation_transitions": len(source_validation_rows),
            "target_transition_count_before_checkpoint_selection": "unknown_not_loaded",
            "action_normalization": action_normalization,
            "source_train_only_baseline": source_train_baseline,
            "source_validation_baseline_metrics": source_validation_baseline,
        }
    )
    device = torch.device(args.device)
    checkpoints: dict[str, list[Path]] = {}
    mappings: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        checkpoints[variant], mappings[variant] = _variant_train(
            variant=variant,
            args=args,
            output=output,
            source_train_rows=source_train_rows,
            source_validation_rows=source_validation_rows,
            held_out_body=_canonical_held_out_body(args.held_out_body),
            action_normalization=action_normalization,
            source_train_baseline=source_train_baseline,
            source_validation_baseline=source_validation_baseline,
            protocol=protocol,
            device=device,
        )
    # This is the only target-payload boundary, and it occurs after all ten
    # checkpoints (five per variant) have been selected on source validation.
    target_rows = materialize_target_development_rows(
        split,
        event_spec,
        args.held_out_body,
        checkpoints_selected=all(len(value) == 5 for value in checkpoints.values()),
    )
    target_metrics = {}
    for variant in VARIANTS:
        models = _restore_models(
            checkpoints[variant],
            device,
            expected_variant=variant,
            expected_held_out_body=_canonical_held_out_body(args.held_out_body),
            expected_mapping=mappings[variant],
        )
        target_metrics[variant] = grouped_target_metrics(
            models,
            target_rows,
            mappings[variant],
            source_train_baseline,
            device,
            batch_size=args.batch_size,
        )
        target_metrics[variant]["evaluated_checkpoint_sha256"] = [
            core.sha256_file(path) for path in checkpoints[variant]
        ]
    target_metrics["source_body_clock_delta_vs_body_agnostic"] = _metric_delta(
        target_metrics["source_body_clock"]["global"],
        target_metrics["body_agnostic"]["global"],
    )
    summary = {
        "format": FORMAT,
        "status": "training_and_frozen_target_development_evaluation_complete",
        "held_out_body": _canonical_held_out_body(args.held_out_body),
        "estimand": "zero_target_label_leave_one_body_out_transfer",
        "source_bodies": sorted(
            {str(row["body"]) for row in source_train_rows}
        ),
        "target_development_transitions": len(target_rows),
        "target_development_opened_after_all_checkpoint_selection": True,
        "target_unused_train_payload_opened": 0,
        "sealed_test_evaluated": False,
        "test_group_hdf5_opened": 0,
        "target_metrics": target_metrics,
        "protocol": protocol,
    }
    core.atomic_json(output / "lobo_training_summary.json", summary)
    return summary


def run_synthetic_smoke() -> dict[str, Any]:
    descriptors = []
    for body in ("aloha-agilex", "ARX-X5", "piper", "ur5-wsg"):
        for seed in range(20):
            descriptors.append(
                core.GroupDescriptor(
                    source="synthetic",
                    body=body,
                    policy="synthetic",
                    task="move_can_pot",
                    seed=seed,
                    path=Path(f"{body}_{seed}.hdf5"),
                )
            )
    split = strict_leave_one_body_out_split(
        descriptors, held_out_body="piper", split_seed=20260828
    )
    rows = []
    batch = core.synthetic_batch(15)
    for index in range(15):
        row = {key: value[index].numpy() for key, value in batch.items()}
        row.update(
            {
                "logical_group": f"g{index}",
                "body": ("aloha-agilex", "ARX-X5", "ur5-wsg")[index % 3],
                "policy": "synthetic",
                "task": "move_can_pot",
            }
        )
        rows.append(row)
    mapping, target_id = body_mapping(rows, "piper", "source_body_clock")
    model = core.MultibodyCanonicalEventWorldModel(
        core.ModelConfig(body_count=max(mapping.values()) + 1, dropout=0.0)
    )
    hook = reserve_zero_target_clock_row(model, target_id)
    loss, _ = core.compute_multitask_loss(model(core.synthetic_batch(15)), core.synthetic_batch(15))
    loss.backward()
    assert model.clock.body_beta.weight.grad is not None
    assert torch.equal(
        model.clock.body_beta.weight.grad[target_id],
        torch.zeros_like(model.clock.body_beta.weight.grad[target_id]),
    )
    if hook is not None:
        hook.remove()
    return {
        "status": "synthetic_smoke_passed",
        "held_out_body": "piper",
        "source_train_groups": len(split.source_train),
        "source_validation_groups": len(split.source_validation),
        "target_development_groups": len(split.target_development),
        "target_unused_train_payload_opened": 0,
        "sealed_test_group_hdf5_opened": 0,
        "reserved_target_clock_gradient_is_zero": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "freeze-split", "train", "synthetic-smoke"),
        required=True,
    )
    parser.add_argument("--held-out-body", choices=ALLOWED_HELD_OUT_BODIES)
    parser.add_argument("--stage1-root", type=Path)
    parser.add_argument("--stage1-source-manifest", type=Path)
    parser.add_argument("--stage1-source-manifest-sha256")
    parser.add_argument("--stage1-target-manifest", type=Path)
    parser.add_argument("--stage1-target-manifest-sha256")
    parser.add_argument("--event-spec", type=Path)
    parser.add_argument("--event-spec-sha256")
    parser.add_argument("--openvla-schema5-manifest", type=Path)
    parser.add_argument("--openvla-schema5-manifest-sha256")
    parser.add_argument("--split-plan-output", type=Path)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--split-plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument(
        "--ensemble-seeds",
        nargs=5,
        type=int,
        default=[20260828, 20260829, 20260830, 20260831, 20260832],
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic-smoke":
        print("SYNTHETIC_SMOKE=" + json.dumps(run_synthetic_smoke(), sort_keys=True))
    elif args.mode == "preflight":
        _, audit, _, plan = run_preflight(args)
        print("PREFLIGHT=" + json.dumps({"audit": audit, "plan": plan}, sort_keys=True))
    elif args.mode == "freeze-split":
        print("FROZEN_SPLIT=" + json.dumps(freeze_split(args), sort_keys=True))
    else:
        print("TRAINING=" + json.dumps(train(args), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWED_HELD_OUT_BODIES",
    "LoboSplit",
    "VARIANTS",
    "body_mapping",
    "build_frozen_split_plan",
    "evaluate_lobo_ensemble",
    "evaluate_source_only_baseline",
    "fit_source_only_action_normalization",
    "grouped_target_metrics",
    "materialize_source_rows",
    "materialize_target_development_rows",
    "reserve_zero_target_clock_row",
    "run_synthetic_smoke",
    "strict_leave_one_body_out_split",
    "verify_frozen_split_plan",
]
