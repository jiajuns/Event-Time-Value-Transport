#!/usr/bin/env python3
"""Train one strict five-body RoboTwin2 LOBO shared event/effect head.

This entry point consumes *prepared public* canonical transition groups.  It
does not open RoboTwin zip files and it does not train a VLA.  A separately
audited, deterministic and label-free parser must first map every robot to the
same 27-D state, 14-D action-effect and five-event vocabulary.

The held-out body is manifest-visible for fold assignment only.  Its group
payloads are not opened, its labels never fit normalization/parameters or
select checkpoints, and the model has a single shared body row.  Task-success
evaluation is deliberately a later live-simulator stage using the frozen actor
and the same ordered candidate set for baseline and ETSF.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_multibody_canonical_event_world_model as core
import robotwin2_cross_body_canonical_adapter_v1 as canonical_adapter
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event
import verify_robotwin2_move_can_pot_public_materialization_v1 as public_materialization


FORMAT = "etsf_robotwin2_five_body_lobo_shared_event_head_v1"
BINDING_FORMAT = "etsf_robotwin2_five_body_lobo_training_binding_v1"
MANIFEST_FORMAT = "etsf_robotwin2_canonical_transition_manifest_v1"
ACTOR_FORMAT = "etsf_robotwin2_frozen_native_actor_authority_v1"
MATERIALIZATION_FORMAT = public_materialization.FORMAT
DATASET_REPO = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
TASK = "move_can_pot"
PREREGISTRATION_SHA256 = (
    "75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee"
)
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
SOURCE_EVENT_SAMPLING_HZ = 15.0
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
CANDIDATE_COUNT = 4
CANDIDATE_RANK_FEATURE_DIM = core.SEMANTIC_DIM + 64 + 2
ABLATION_VARIANTS = (
    "success_only",
    "no_time_duration",
    "no_object_effect",
    "full",
)


def ablation_contract(variant: str) -> dict[str, Any]:
    if variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    return {
        "variant": variant,
        "candidate_score": (
            "proper_success_logit"
            if variant == "success_only"
            else "pairwise_candidate_rank_logit"
        ),
        "multitask_heads_enabled": (
            ["success"]
            if variant == "success_only"
            else [
                "post_event",
                "next_event",
                *([] if variant == "no_time_duration" else ["duration"]),
                "success",
                "recovery",
            ]
        ),
        "time_duration_rank_features_enabled": variant
        not in {"success_only", "no_time_duration"},
        "object_effect_loss_and_rank_target_enabled": variant
        not in {"success_only", "no_object_effect"},
        "same_seed_disjoint_split": True,
        "heldout_labels_used_for_training_or_selection": False,
    }


def ablation_selection_components(
    components: Mapping[str, float], variant: str
) -> dict[str, float]:
    allowed = {
        "success_only": {"success_brier_ratio"},
        "no_time_duration": {
            "post_event_macro_error_ratio",
            "next_event_macro_error_ratio",
            "success_brier_ratio",
            "object_rmse_ratio",
        },
        "no_object_effect": {
            "post_event_macro_error_ratio",
            "next_event_macro_error_ratio",
            "observed_duration_mae_ratio",
            "success_brier_ratio",
        },
        "full": set(components),
    }
    if variant not in allowed:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    selected = {
        name: float(value) for name, value in components.items() if name in allowed[variant]
    }
    if not selected:
        raise FiveBodyContractError(
            f"{variant} has no enabled validation diagnostic for checkpoint selection"
        )
    return selected


def checkpoint_candidate_rank_contract(variant: str) -> dict[str, Any]:
    """Describe the score path saved in one selected checkpoint."""

    if variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    if variant == "success_only":
        feature_blocks = ["proper_success_logit_from_transitioned_semantic"]
    elif variant == "no_time_duration":
        feature_blocks = [
            "transitioned_semantic_end_to_end",
            "clock_hidden_forced_zero",
            "current_event_duration_log_mean_forced_zero",
            "current_event_duration_log_scale_forced_zero",
        ]
    else:
        feature_blocks = [
            "transitioned_semantic_end_to_end",
            "clock_hidden_detached",
            "current_event_duration_log_mean_detached",
            "current_event_duration_log_scale_detached",
        ]
    return {
        "feature_blocks": feature_blocks,
        "feature_dim": CANDIDATE_RANK_FEATURE_DIM,
        "dt_has_numeric_score_path": variant not in {"success_only", "no_time_duration"},
        "pairwise_rank_loss_enabled": variant != "success_only",
        "rank_loss_updates_clock_or_duration_heads": False,
        "rank_loss_updates_semantic_action_transition": variant != "success_only",
    }


def summary_candidate_rank_contract(variant: str) -> dict[str, Any]:
    """Describe rank supervision without claiming disabled ablation paths."""

    checkpoint = checkpoint_candidate_rank_contract(variant)
    return {
        "candidate_score": ablation_contract(variant)["candidate_score"],
        "time_and_duration_effect_used": variant
        not in {"success_only", "no_time_duration"},
        "dt_has_numeric_score_path": checkpoint["dt_has_numeric_score_path"],
        "pairwise_rank_loss_enabled": checkpoint["pairwise_rank_loss_enabled"],
        "rank_loss_updates_clock_or_duration_heads": False,
        "rank_loss_updates_semantic_action_transition": checkpoint[
            "rank_loss_updates_semantic_action_transition"
        ],
    }
CANONICAL_STATE_SCHEMA = canonical_adapter.STATE_SCHEMA
CANONICAL_ACTION_SCHEMA = canonical_adapter.ACTION_SCHEMA
REQUIRED_ARRAYS = {
    "state",
    "actions",
    "action_mask",
    "current_event_id",
    "post_event_id",
    "post_event_mask",
    "next_event_id",
    "next_event_mask",
    "duration",
    "duration_observed",
    "duration_mask",
    "success",
    "success_mask",
    "recovery",
    "recovery_mask",
    "object_delta",
    "object_delta_mask",
    "candidate_index",
    "dt",
}


class FiveBodyContractError(RuntimeError):
    """A five-body training authority or payload failed closed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    """Hash a checkpoint directory as ordered relative path/size/file hashes."""

    root = path.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FiveBodyContractError("checkpoint tree must be a real directory")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise FiveBodyContractError("checkpoint tree contains a symbolic link")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise FiveBodyContractError("checkpoint tree contains a special file")
        rows.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise FiveBodyContractError("checkpoint tree is empty")
    return canonical_sha256(rows), len(rows), sum(row["size_bytes"] for row in rows)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FiveBodyContractError(f"{label} is not a file: {resolved}")
    if not _is_sha(expected_sha256) or sha256_file(resolved) != expected_sha256.lower():
        raise FiveBodyContractError(f"{label} SHA-256 mismatch")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveBodyContractError(f"{label} must be a JSON object")
    return value


def _resolve_contained(parent: Path, relative: str, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise FiveBodyContractError(f"{label} path must be relative")
    resolved_parent = parent.resolve()
    lexical = resolved_parent
    for component in raw.parts:
        lexical = lexical / component
        if lexical.is_symlink():
            raise FiveBodyContractError(f"{label} path contains a symbolic link")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(resolved_parent)
    except ValueError as error:
        raise FiveBodyContractError(f"{label} escapes its manifest directory") from error
    return resolved


def _lexical_contained_payload_path(parent: Path, relative: str, label: str) -> Path:
    """Resolve a declared payload path without stat/open/read of that payload."""

    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise FiveBodyContractError(f"{label} path must be a safe relative path")
    absolute_parent = Path(os.path.abspath(os.fspath(parent)))
    absolute = Path(os.path.abspath(os.fspath(absolute_parent / raw)))
    try:
        absolute.relative_to(absolute_parent)
    except ValueError as error:
        raise FiveBodyContractError(f"{label} escapes its manifest directory") from error
    return absolute


def _verify_signed(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise FiveBodyContractError(f"{label} logical SHA-256 mismatch")


def validate_materialization_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        public_materialization.validate_receipt(value)
    except public_materialization.PublicMaterializationError as error:
        raise FiveBodyContractError("public payload materialization is not verified") from error
    if (
        value.get("format") != MATERIALIZATION_FORMAT
        or value.get("status") != public_materialization.STATUS
        or value.get("hf_repo_id") != DATASET_REPO
        or value.get("hf_repo_revision") != DATASET_REVISION
        or value.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or value.get("official_file_count") != 11
        or value.get("all_exact_archive_payload_sha256_verified") is not True
    ):
        raise FiveBodyContractError("public receipt is not the frozen five-body slice")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 11:
        raise FiveBodyContractError("materialization must bind all 11 official slice files")
    paths = [item.get("path") for item in files if isinstance(item, Mapping)]
    if len(paths) != 11 or len(set(paths)) != 11:
        raise FiveBodyContractError("materialization file paths are incomplete or duplicated")
    for item in files:
        if (
            not isinstance(item, Mapping)
            or not _is_sha(item.get("observed_payload_sha256"))
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
            or item.get("size_match") is not True
            or item.get("payload_sha256_match") is not True
        ):
            raise FiveBodyContractError("materialization contains an unverified payload")
    return {"file_count": 11, "file_set_sha256": canonical_sha256(files)}


def validate_actor_authority(
    value: Mapping[str, Any], *, authority_dir: Path
) -> dict[str, Any]:
    _verify_signed(value, "actor authority")
    if value.get("format") != ACTOR_FORMAT or value.get("task") != TASK:
        raise FiveBodyContractError("actor authority format/task mismatch")
    actors = value.get("actors")
    if not isinstance(actors, Mapping) or set(actors) != set(BODIES):
        raise FiveBodyContractError("actor authority must bind exactly five bodies")
    for body in BODIES:
        actor = actors[body]
        if (
            not isinstance(actor, Mapping)
            or actor.get("frozen") is not True
            or actor.get("optimizer_updates_allowed") is not False
            or actor.get("candidate_count") != CANDIDATE_COUNT
            or actor.get("candidate_zero_is_actor_baseline") is not True
            or actor.get("same_ordered_candidate_set_for_baseline_and_etsf") is not True
            or not _is_sha(actor.get("checkpoint_sha256"))
            or not _is_sha(actor.get("sampling_contract_sha256"))
            or actor.get("checkpoint_kind") not in {"file", "directory_tree"}
        ):
            raise FiveBodyContractError(f"actor authority is incomplete for {body}")
        checkpoint = _resolve_contained(
            authority_dir, str(actor.get("checkpoint_path", "")), f"{body} actor checkpoint"
        )
        if actor["checkpoint_kind"] == "file":
            observed_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
        else:
            observed_sha = sha256_tree(checkpoint)[0] if checkpoint.is_dir() else None
        if observed_sha != actor["checkpoint_sha256"]:
            raise FiveBodyContractError(f"frozen actor checkpoint missing/tampered for {body}")
    return {
        "actor_frozen": True,
        "bodies": list(BODIES),
        "candidate_count": CANDIDATE_COUNT,
        "same_ordered_candidate_set": True,
    }


def validate_body_manifest(
    value: Mapping[str, Any], *, expected_body: str, manifest_dir: Path
) -> dict[str, Any]:
    _verify_signed(value, f"{expected_body} canonical manifest")
    if (
        value.get("format") != MANIFEST_FORMAT
        or value.get("dataset_repo") != DATASET_REPO
        or value.get("dataset_revision") != DATASET_REVISION
        or value.get("task") != TASK
        or value.get("body") != expected_body
    ):
        raise FiveBodyContractError(f"canonical manifest identity mismatch for {expected_body}")
    adapter = value.get("schema_adapter")
    if (
        not isinstance(adapter, Mapping)
        or adapter.get("kind") != "analytic_label_free_canonical_v1"
        or adapter.get("trainable") is not False
        or adapter.get("labels_or_outcomes_used_to_fit") is not False
        or adapter.get("heldout_supervision_allowed") is not False
        or adapter.get("state_dim") != core.STATE_DIM
        or adapter.get("action_dim") != core.ACTION_DIM
        or adapter.get("state_schema") != CANONICAL_STATE_SCHEMA
        or adapter.get("action_schema") != CANONICAL_ACTION_SCHEMA
        or adapter.get("elapsed_time_unit") != "seconds"
        or adapter.get("duration_unit") != "seconds"
        or adapter.get("event_names") != list(core.CANONICAL_EVENTS)
        or not _is_sha(adapter.get("implementation_sha256"))
    ):
        raise FiveBodyContractError(f"{expected_body} adapter is not analytic/label-free")
    physical_time = value.get("physical_time_contract")
    try:
        analytic_event.validate_event_contract(value.get("analytic_event_contract"))
    except analytic_event.AnalyticEventSpecError as error:
        raise FiveBodyContractError(
            f"{expected_body} analytic event contract changed"
        ) from error
    if (
        value.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or not _is_sha(value.get("event_derivation_implementation_sha256"))
        or value.get("state27_relative_goal_contract")
        != (
            "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
            "event_labels_and_online_state27_channels_0_2"
        )
        or not isinstance(physical_time, Mapping)
        or physical_time.get("source")
        != "counted_successful_sapien_scene_step_calls"
        or physical_time.get("simulator_timestep_source") != "scene.get_timestep"
        or physical_time.get("policy_action_call_count_used_as_time") is not False
        or physical_time.get("wall_clock_used_as_time") is not False
        or physical_time.get("dt_semantics")
        != "planned_first_candidate_chunk_seconds"
        or physical_time.get("planned_action_steps") != 5
        or physical_time.get("actor_control_hz") != SOURCE_EVENT_SAMPLING_HZ
        or physical_time.get("planned_dt_seconds") != 5.0 / SOURCE_EVENT_SAMPLING_HZ
        or physical_time.get("duration_semantics")
        != "simulator_elapsed_seconds_to_event_boundary"
        or physical_time.get("zero_elapsed_duration_masked") is not True
        or physical_time.get("stationary_window_seconds")
        != analytic_event.THRESHOLDS["stationary_window_seconds"]
        or physical_time.get("stationary_speed_threshold_m_per_s")
        != analytic_event.THRESHOLDS["stationary_speed_m_per_s"]
    ):
        raise FiveBodyContractError(
            f"{expected_body} lacks the physical simulator time/event contract"
        )
    candidate_action = value.get("candidate_action_contract")
    if candidate_action != {
        "critic_observation_time": "before_candidate_execution",
        "planned_action_horizon": 5,
        "action_mask_source": "planned_first_chunk_not_executed_count",
        "executed_action_count_used_for_action_mask": False,
        "executed_action_count_used_for_sim_time_accounting_only": True,
        "zero_step_infeasible_candidate_keeps_failure_and_action_binding": True,
    }:
        raise FiveBodyContractError(
            f"{expected_body} censors planned candidates after execution"
        )
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) < 4:
        raise FiveBodyContractError(f"{expected_body} needs at least four canonical groups")
    identities: set[str] = set()
    conditions: set[str] = set()
    normalized = []
    for item in groups:
        if not isinstance(item, Mapping):
            raise FiveBodyContractError("canonical group entry must be an object")
        group_id = item.get("group_id")
        condition = item.get("condition")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in identities
            or condition not in CONDITIONS
            or isinstance(item.get("requested_seed"), bool)
            or not isinstance(item.get("requested_seed"), int)
            or not _is_sha(item.get("sha256"))
        ):
            raise FiveBodyContractError(f"invalid canonical group for {expected_body}")
        # Manifest validation is deliberately payload-blind.  In particular,
        # a later-held-out body must not even have its group file stat'ed or
        # hashed.  Source groups are resolved, stat'ed and hashed only at the
        # source payload boundary in ``_npz_rows``.
        path = _lexical_contained_payload_path(
            manifest_dir, str(item.get("path", "")), "group"
        )
        identities.add(group_id)
        conditions.add(condition)
        normalized.append({**dict(item), "resolved_path": str(path)})
    if conditions != set(CONDITIONS):
        raise FiveBodyContractError(f"{expected_body} lacks clean/randomized groups")
    return {
        "body": expected_body,
        "groups": normalized,
        "group_identity_sha256": canonical_sha256(sorted(identities)),
        "adapter_sha256": adapter["implementation_sha256"],
        "event_derivation_implementation_sha256": value[
            "event_derivation_implementation_sha256"
        ],
    }


def load_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    binding = _read_bound_json(path, expected_sha256, "training binding")
    _verify_signed(binding, "training binding")
    if (
        binding.get("format") != BINDING_FORMAT
        or binding.get("dataset_repo") != DATASET_REPO
        or binding.get("dataset_revision") != DATASET_REVISION
        or binding.get("task") != TASK
        or binding.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or binding.get("heldout_labels_may_train_fit_calibrate_or_select") is not False
        or binding.get("canonical_shared_body_rows") != 1
    ):
        raise FiveBodyContractError("training binding violates strict LOBO")
    authority = binding.get("execution_authority")
    if authority != {
        "explicit_user_training_request_recorded": True,
        "public_data_only": True,
        "protected_internal_data_allowed": False,
        "remote_cuda_only": True,
    }:
        raise FiveBodyContractError("training binding lacks explicit public remote authority")
    root = path.expanduser().resolve().parent
    materialization_path = _resolve_contained(
        root, str(binding.get("materialization_receipt", {}).get("path", "")), "materialization"
    )
    materialization = _read_bound_json(
        materialization_path,
        str(binding.get("materialization_receipt", {}).get("sha256", "")),
        "materialization receipt",
    )
    materialization_audit = validate_materialization_receipt(materialization)
    actor_path = _resolve_contained(
        root, str(binding.get("actor_authority", {}).get("path", "")), "actor authority"
    )
    actors = _read_bound_json(
        actor_path,
        str(binding.get("actor_authority", {}).get("sha256", "")),
        "actor authority",
    )
    actor_audit = validate_actor_authority(actors, authority_dir=actor_path.parent)
    body_bindings = binding.get("body_manifests")
    if not isinstance(body_bindings, Mapping) or set(body_bindings) != set(BODIES):
        raise FiveBodyContractError("binding must contain exactly five body manifests")
    manifests = {}
    for body in BODIES:
        item = body_bindings[body]
        if not isinstance(item, Mapping):
            raise FiveBodyContractError(f"body manifest binding missing for {body}")
        manifest_path = _resolve_contained(root, str(item.get("path", "")), "body manifest")
        manifest = _read_bound_json(
            manifest_path, str(item.get("sha256", "")), f"{body} body manifest"
        )
        manifests[body] = validate_body_manifest(
            manifest, expected_body=body, manifest_dir=manifest_path.parent
        )
    event_implementations = {
        item["event_derivation_implementation_sha256"] for item in manifests.values()
    }
    if len(event_implementations) != 1:
        raise FiveBodyContractError(
            "five bodies do not share one analytic event implementation"
        )
    return {
        "binding": binding,
        "binding_file_sha256": expected_sha256.lower(),
        "materialization": materialization_audit,
        "actor": actor_audit,
        "manifests": manifests,
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": event_implementations.pop(),
    }


def source_group_split(
    audit: Mapping[str, Any], *, held_out_body: str, split_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if held_out_body not in BODIES:
        raise FiveBodyContractError(f"unknown held-out body {held_out_body!r}")
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for body in BODIES:
        groups = list(audit["manifests"][body]["groups"])
        if body == held_out_body:
            heldout.extend(groups)
            continue
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            by_condition[str(group["condition"])].append(group)
        for condition in CONDITIONS:
            by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for group in by_condition[condition]:
                by_seed[int(group["requested_seed"])].append(group)
            ordered_seeds = sorted(
                by_seed,
                key=lambda seed: hashlib.sha256(
                    f"{split_seed}|{body}|{condition}|{seed}".encode()
                ).hexdigest(),
            )
            if len(ordered_seeds) < 2:
                raise FiveBodyContractError(
                    f"{body}/{condition} cannot form seed-disjoint source validation"
                )
            validation_seed_count = max(1, int(round(0.2 * len(ordered_seeds))))
            validation_seeds = set(ordered_seeds[:validation_seed_count])
            for seed in ordered_seeds:
                target = validation if seed in validation_seeds else training
                target.extend(
                    {**row, "body": body}
                    for row in sorted(by_seed[seed], key=lambda item: item["group_id"])
                )
    if not training or not validation or not heldout:
        raise FiveBodyContractError("LOBO split contains an empty lane")
    return training, validation, heldout


def build_preflight_receipt(
    audit: Mapping[str, Any], *, held_out_body: str, split_seed: int
) -> dict[str, Any]:
    training, validation, heldout = source_group_split(
        audit, held_out_body=held_out_body, split_seed=split_seed
    )
    def identity(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        return sorted(f"{row.get('body', held_out_body)}|{row['group_id']}" for row in rows)
    receipt = {
        "format": FORMAT,
        "status": "preflight_passed_payloads_still_unopened",
        "dataset_revision": DATASET_REVISION,
        "event_spec_sha256": audit["event_spec_sha256"],
        "event_derivation_implementation_sha256": audit[
            "event_derivation_implementation_sha256"
        ],
        "binding_file_sha256": audit["binding_file_sha256"],
        "held_out_body": held_out_body,
        "source_bodies": [body for body in BODIES if body != held_out_body],
        "split_seed": split_seed,
        "source_train_groups": len(training),
        "source_validation_groups": len(validation),
        "heldout_groups_deferred": len(heldout),
        "source_train_identity_sha256": canonical_sha256(identity(training)),
        "source_validation_identity_sha256": canonical_sha256(identity(validation)),
        "heldout_identity_sha256": canonical_sha256(
            sorted(f"{held_out_body}|{row['group_id']}" for row in heldout)
        ),
        "assignment_uses_labels": False,
        "split_unit": "body_condition_requested_seed_all_queries",
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "model_body_rows": 1,
        "heldout_specific_trainable_parameters": 0,
        "actor_frozen": True,
        "same_ordered_candidate_set_required": True,
        "candidate_zero_is_actor_baseline": True,
        "task_success_evaluation_authorized": False,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return receipt


def _npz_rows(group: Mapping[str, Any], *, body: str) -> list[dict[str, Any]]:
    path = Path(str(group["resolved_path"]))
    if not path.is_file() or sha256_file(path) != group["sha256"]:
        raise FiveBodyContractError(f"source canonical group missing/tampered: {path}")
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != REQUIRED_ARRAYS:
            raise FiveBodyContractError(
                f"{path} arrays mismatch: missing={sorted(REQUIRED_ARRAYS-set(values.files))}, "
                f"extra={sorted(set(values.files)-REQUIRED_ARRAYS)}"
            )
        arrays = {name: np.asarray(values[name]) for name in values.files}
    count = len(arrays["state"])
    if count != CANDIDATE_COUNT:
        raise FiveBodyContractError(
            f"{path} must contain one complete four-candidate decision"
        )
    shapes = {
        "state": (count, core.STATE_DIM),
        "actions": (count, arrays["actions"].shape[1], core.ACTION_DIM),
        "action_mask": (count, arrays["actions"].shape[1]),
        "object_delta": (count, core.OBJECT_DELTA_DIM),
    }
    for name, expected in shapes.items():
        if arrays[name].shape != expected:
            raise FiveBodyContractError(f"{path} array {name} shape mismatch")
    scalar = REQUIRED_ARRAYS - set(shapes)
    if any(arrays[name].shape != (count,) for name in scalar):
        raise FiveBodyContractError(f"{path} scalar supervision array shape mismatch")
    if count == 0 or any(not np.isfinite(value).all() for value in arrays.values()):
        raise FiveBodyContractError(f"{path} contains empty/non-finite canonical rows")
    if np.any((arrays["current_event_id"] < 0) | (arrays["current_event_id"] >= 5)):
        raise FiveBodyContractError(f"{path} current event ids are invalid")
    expected_event_onehot = np.zeros((count, 5), dtype=np.float32)
    expected_event_onehot[np.arange(count), arrays["current_event_id"].astype(int)] = 1.0
    if not np.array_equal(
        np.asarray(arrays["state"][:, 18:23], dtype=np.float32),
        expected_event_onehot,
    ):
        raise FiveBodyContractError(
            f"{path} state event onehot disagrees with current_event_id"
        )
    if np.any(arrays["dt"] <= 0):
        raise FiveBodyContractError(f"{path} contains non-positive planned dt")
    if not np.allclose(
        arrays["dt"], 5.0 / SOURCE_EVENT_SAMPLING_HZ, atol=1e-6, rtol=0.0
    ):
        raise FiveBodyContractError(f"{path} planned dt is not fixed 5/15 seconds")
    if np.any(arrays["duration"] < 0):
        raise FiveBodyContractError(f"{path} contains invalid simulator duration")
    if not np.array_equal(
        np.asarray(arrays["duration_mask"] > 0.5), arrays["duration"] > 0.0
    ):
        raise FiveBodyContractError(
            f"{path} duration mask is not tied to positive simulator exposure"
        )
    horizon = arrays["actions"].shape[1]
    planned_mask = np.arange(horizon) < 5
    if horizon < 5 or not np.array_equal(
        np.asarray(arrays["action_mask"], dtype=bool),
        np.repeat(planned_mask[None], CANDIDATE_COUNT, axis=0),
    ):
        raise FiveBodyContractError(
            f"{path} action mask does not expose the full planned first chunk"
        )
    if not np.array_equal(arrays["candidate_index"], np.arange(CANDIDATE_COUNT)):
        raise FiveBodyContractError(f"{path} candidate order is not [0,1,2,3]")
    if not np.array_equal(
        arrays["state"], np.repeat(arrays["state"][:1], CANDIDATE_COUNT, axis=0)
    ) or not np.all(arrays["current_event_id"] == arrays["current_event_id"][0]):
        raise FiveBodyContractError(
            f"{path} candidates do not share one root state/event"
        )
    rows = []
    for index in range(count):
        row = {name: arrays[name][index] for name in REQUIRED_ARRAYS}
        row.update(
            {
                "action_available": np.float32(1.0),
                "action_schema_id": np.int64(0),
                "logical_group": f"{body}|{group['condition']}|{group['group_id']}",
                "body": body,
                "policy": "frozen_native_actor",
                "task": TASK,
            }
        )
        rows.append(row)
    return rows


class CompleteDecisionBatchSampler:
    """Shuffle decisions while keeping every four-candidate set intact."""

    def __init__(
        self, rows: Sequence[Mapping[str, Any]], *, batch_size: int, seed: int
    ) -> None:
        if batch_size < CANDIDATE_COUNT:
            raise FiveBodyContractError("batch size must fit one complete decision")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["logical_group"])].append(index)
        for group, indices in grouped.items():
            ordered = sorted(indices, key=lambda index: int(rows[index]["candidate_index"]))
            if len(ordered) != CANDIDATE_COUNT or [
                int(rows[index]["candidate_index"]) for index in ordered
            ] != list(range(CANDIDATE_COUNT)):
                raise FiveBodyContractError(f"incomplete candidate decision {group}")
            grouped[group] = ordered
        self.decisions = [grouped[name] for name in sorted(grouped)]
        self.decisions_per_batch = max(1, batch_size // CANDIDATE_COUNT)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        decisions = list(self.decisions)
        generator.shuffle(decisions)
        for start in range(0, len(decisions), self.decisions_per_batch):
            yield [
                index
                for decision in decisions[start : start + self.decisions_per_batch]
                for index in decision
            ]

    def __len__(self) -> int:
        return (len(self.decisions) + self.decisions_per_batch - 1) // self.decisions_per_batch


class EffectAlignedSharedEventHead(core.MultibodyCanonicalEventWorldModel):
    """Shared event model with a scalar head trained for best-of-four choice."""

    def __init__(self, ablation_variant: str = "full") -> None:
        if ablation_variant not in ABLATION_VARIANTS:
            raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
        super().__init__(core.ModelConfig(body_count=1, action_schema_count=1))
        self.ablation_variant = ablation_variant
        self.register_buffer("state_mean", torch.zeros(core.STATE_DIM))
        self.register_buffer("state_std", torch.ones(core.STATE_DIM))
        self.candidate_rank = torch.nn.Sequential(
            torch.nn.LayerNorm(CANDIDATE_RANK_FEATURE_DIM),
            torch.nn.Linear(CANDIDATE_RANK_FEATURE_DIM, core.SEMANTIC_DIM // 2),
            torch.nn.GELU(),
            torch.nn.Linear(core.SEMANTIC_DIM // 2, 1),
        )

    @torch.no_grad()
    def set_state_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != (core.STATE_DIM,) or std.shape != (core.STATE_DIM,):
            raise FiveBodyContractError("state normalization must be 27-D")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise FiveBodyContractError("state normalization contains non-finite values")
        if bool((std < 1e-4).any()):
            raise FiveBodyContractError("state normalization std is below the floor")
        self.state_mean.copy_(mean.to(self.state_mean))
        self.state_std.copy_(std.to(self.state_std))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized_batch = dict(batch)
        normalized_batch["state"] = (
            batch["state"] - self.state_mean
        ) / self.state_std
        output = super().forward(normalized_batch)
        # Ranking sees semantic action effects, the dt-dependent isolated
        # clock, and the current-event duration distribution.  The semantic
        # block remains end-to-end so rank supervision can improve the shared
        # state/action transition.  Clock/duration features are detached so
        # rank loss cannot directly move their proper likelihood parameters.
        clock_features = output["clock_hidden"].detach()
        duration_features = torch.stack(
            (
                output["duration_selected_log_mean"].detach(),
                output["duration_selected_log_scale"].detach(),
            ),
            dim=-1,
        )
        if self.ablation_variant == "no_time_duration":
            clock_features = torch.zeros_like(clock_features)
            duration_features = torch.zeros_like(duration_features)
        rank_features = torch.cat(
            (
                output["transitioned"],
                clock_features,
                duration_features,
            ),
            dim=-1,
        )
        output["candidate_rank_features"] = rank_features
        output["candidate_rank_logit"] = (
            output["success_logit"]
            if self.ablation_variant == "success_only"
            else self.candidate_rank(rank_features).squeeze(-1)
        )
        return output


def _effect_aligned_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize robust object effects and within-root ordering.

    Event, outcome, recovery and duration heads keep their unweighted proper
    losses in ``compute_multitask_loss`` so their probabilities remain
    interpretable.  This auxiliary objective only fixes the two pieces that
    were misaligned with best-of-four inference.
    """

    labels = batch["success"].to(output["success_logit"])

    # A Student-t(3) likelihood retains a calibrated scale head but prevents
    # a few large object errors from dominating the ranking objective.
    object_scale = torch.exp(output["object_delta_log_scale"]).clamp_min(1e-4)
    standardized = (
        batch["object_delta"].to(output["object_delta_mean"])
        - output["object_delta_mean"]
    ) / object_scale
    object_rows = (
        output["object_delta_log_scale"]
        + 2.0 * torch.log1p(standardized.square() / 3.0)
    ).mean(dim=-1)
    object_effect = core._weighted_mean(
        object_rows,
        sample_weight
        * batch["action_available"].to(sample_weight)
        * batch["object_delta_mask"].to(sample_weight),
    )

    ranking_terms: list[torch.Tensor] = []
    ranking_weights: list[torch.Tensor] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(batch["logical_group"]):
        by_group[str(group)].append(index)
    event_value = batch["post_event_id"].to(output["candidate_rank_logit"]).float()
    relative_start = batch["state"][:, 0:3].to(output["candidate_rank_logit"])
    relative_end = relative_start + batch["object_delta"][:, 3:6].to(relative_start)
    goal_progress = relative_start.norm(dim=-1) - relative_end.norm(dim=-1)
    if ablation_variant == "no_object_effect":
        goal_progress = torch.zeros_like(goal_progress)
    # Lexicographic target: task success first, then event stage, then geometric
    # progress.  The large fixed gaps prevent a partial-progress failure from
    # outranking any successful candidate.
    target = 100.0 * labels + 10.0 * event_value + goal_progress
    score = output["candidate_rank_logit"]
    for indices in by_group.values():
        if len(indices) != CANDIDATE_COUNT:
            raise FiveBodyContractError("training batch split a candidate decision")
        for left in range(CANDIDATE_COUNT):
            for right in range(left + 1, CANDIDATE_COUNT):
                first, second = indices[left], indices[right]
                difference = target[first] - target[second]
                if bool(torch.abs(difference) <= 1e-6):
                    continue
                sign = torch.sign(difference).detach()
                ranking_terms.append(
                    torch.nn.functional.softplus(-sign * (score[first] - score[second]))
                )
                ranking_weights.append(torch.minimum(sample_weight[first], sample_weight[second]))
    if ranking_terms:
        ranking = core._weighted_mean(
            torch.stack(ranking_terms), torch.stack(ranking_weights)
        )
    else:
        ranking = score.sum() * 0.0
    if ablation_variant == "success_only":
        object_effect = object_effect * 0.0
        ranking = ranking * 0.0
    elif ablation_variant == "no_object_effect":
        object_effect = object_effect * 0.0
    total = 0.5 * object_effect + ranking
    return total, {
        "robust_object_effect": object_effect,
        "pairwise_ranking": ranking,
    }


@torch.no_grad()
def evaluate_candidate_ranking(
    model: EffectAlignedSharedEventHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    groups: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for raw in loader:
        batch = core._move_batch(raw, device)
        output = model(batch)
        for index, group in enumerate(raw["logical_group"]):
            relative_start = batch["state"][index, 0:3]
            relative_end = relative_start + batch["object_delta"][index, 3:6]
            target = (
                100.0 * float(batch["success"][index])
                + 10.0 * float(batch["post_event_id"][index])
                + float(relative_start.norm() - relative_end.norm())
            )
            groups[str(group)].append(
                (
                    int(batch["candidate_index"][index]),
                    float(output["candidate_rank_logit"][index]),
                    float(batch["success"][index]),
                    target,
                )
            )
    decisions: list[dict[str, Any]] = []
    pair_correct = pair_total = 0
    for group, rows in groups.items():
        rows = sorted(rows)
        if len(rows) != CANDIDATE_COUNT or [row[0] for row in rows] != list(range(4)):
            raise FiveBodyContractError(f"validation decision incomplete: {group}")
        identity = group.split("|", 2)
        if len(identity) != 3 or identity[0] not in BODIES or identity[1] not in CONDITIONS:
            raise FiveBodyContractError(f"validation decision identity changed: {group}")
        decisions.append(
            {
                "body": identity[0],
                "condition": identity[1],
                "baseline_success": rows[0][2],
                "selected_success": max(rows, key=lambda row: row[1])[2],
                "oracle_success": max(row[2] for row in rows),
            }
        )
        for left in range(4):
            for right in range(left + 1, 4):
                target_difference = rows[left][3] - rows[right][3]
                if abs(target_difference) <= 1e-6:
                    continue
                pair_total += 1
                score_difference = rows[left][1] - rows[right][1]
                pair_correct += int(score_difference * target_difference > 0)
    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        baseline = float(np.mean([float(row["baseline_success"]) for row in rows]))
        selected = float(np.mean([float(row["selected_success"]) for row in rows]))
        return {
            "decision_groups": len(rows),
            "baseline_success_rate": baseline,
            "selected_success_rate": selected,
            "delta_success_rate": selected - baseline,
            "oracle_success_rate": float(
                np.mean([float(row["oracle_success"]) for row in rows])
            ),
        }

    global_metrics = summarize(decisions)
    units: dict[str, dict[str, float | int]] = {}
    for body in BODIES:
        for condition in CONDITIONS:
            selected_rows = [
                row
                for row in decisions
                if row["body"] == body and row["condition"] == condition
            ]
            if selected_rows:
                units[f"{body}|{condition}"] = summarize(selected_rows)
    macro_delta = float(np.mean([row["delta_success_rate"] for row in units.values()]))
    macro_selected = float(
        np.mean([row["selected_success_rate"] for row in units.values()])
    )
    macro_oracle = float(
        np.mean([row["oracle_success_rate"] for row in units.values()])
    )
    return {
        **global_metrics,
        "body_condition_units": units,
        "macro_delta_success_rate": macro_delta,
        "macro_selected_success_rate": macro_selected,
        "macro_oracle_success_rate": macro_oracle,
        "pairwise_accuracy": float(pair_correct / pair_total) if pair_total else None,
        "pairwise_comparisons": pair_total,
    }


def materialize_source_rows(
    groups: Sequence[Mapping[str, Any]], *, held_out_body: str
) -> list[dict[str, Any]]:
    if any(group.get("body") == held_out_body for group in groups):
        raise FiveBodyContractError("held-out group reached source payload loader")
    rows = []
    for group in groups:
        rows.extend(_npz_rows(group, body=str(group["body"])))
    if any(row["body"] == held_out_body for row in rows):
        raise FiveBodyContractError("held-out row reached source fitting")
    return rows


def _train_fold(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("training output must be a new path")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise FiveBodyContractError("real fold training is remote CUDA-only")
    if "4090" not in torch.cuda.get_device_name(0):
        raise FiveBodyContractError("real fold training requires the authorized RTX 4090")
    train_groups, validation_groups, _heldout_groups = source_group_split(
        audit, held_out_body=args.held_out_body, split_seed=args.split_seed
    )
    preflight = build_preflight_receipt(
        audit, held_out_body=args.held_out_body, split_seed=args.split_seed
    )
    output.mkdir(parents=True)
    core.atomic_json(output / "preflight_receipt.json", preflight)
    train_rows = materialize_source_rows(train_groups, held_out_body=args.held_out_body)
    validation_rows = materialize_source_rows(
        validation_groups, held_out_body=args.held_out_body
    )
    successes = np.asarray(
        [float(row["success"]) for row in train_rows if bool(row["success_mask"])]
    )
    if not len(successes) or set(np.unique(successes).tolist()) != {0.0, 1.0}:
        raise FiveBodyContractError(
            "source training requires real positive and negative outcome supervision"
        )
    outcome_by_group: dict[str, set[float]] = defaultdict(set)
    for row in train_rows:
        if bool(row["success_mask"]):
            outcome_by_group[str(row["logical_group"])].add(float(row["success"]))
    mixed_outcome_decisions = sum(len(values) > 1 for values in outcome_by_group.values())
    if mixed_outcome_decisions == 0:
        raise FiveBodyContractError(
            "source training has no within-root success/failure candidate comparison"
        )
    source_normalization = core.fit_train_action_normalization(
        train_rows, required_schema_ids=(0,)
    )
    canonical_stats = dict(source_normalization["schemas"]["aloha"])
    canonical_stats["schema_id"] = 0
    normalization = {
        "format": "etsf_five_body_canonical_train_only_normalization_v2",
        "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
        "canonical_action_schema_id": 0,
        "storage_slot_origin": "legacy_schema0_relabelled_no_aloha_semantic_claim",
        "schema": canonical_stats,
        "heldout_rows_used": 0,
    }
    normalization["sha256"] = core.canonical_json_sha256(normalization)
    action_mean = np.asarray(canonical_stats["mean"], dtype=np.float32)[None]
    action_std = np.asarray(canonical_stats["std"], dtype=np.float32)[None]
    source_states = np.stack(
        [np.asarray(row["state"], dtype=np.float32) for row in train_rows]
    )
    state_mean = np.zeros(core.STATE_DIM, dtype=np.float32)
    state_std = np.ones(core.STATE_DIM, dtype=np.float32)
    # Channels 0:18 are continuous geometry/gripper/quaternion.  Event and
    # predicate channels 18:27 stay exact binary inputs.
    state_mean[:18] = source_states[:, :18].mean(axis=0)
    state_std[:18] = np.maximum(source_states[:, :18].std(axis=0), 1e-4)
    state_normalization = {
        "format": "etsf_five_body_canonical_state_train_only_normalization_v1",
        "canonical_state_schema": CANONICAL_STATE_SCHEMA,
        "continuous_channels": list(range(18)),
        "binary_channels_unchanged": list(range(18, core.STATE_DIM)),
        "mean": state_mean.tolist(),
        "std": state_std.tolist(),
        "heldout_rows_used": 0,
    }
    state_normalization["sha256"] = core.canonical_json_sha256(state_normalization)
    baseline = core.fit_train_baselines(train_rows)
    validation_baseline = core.evaluate_train_only_baselines(baseline, validation_rows)
    body_to_id = {body: 0 for body in preflight["source_bodies"]}
    train_dataset = core.TransitionDataset(train_rows, body_to_id)
    validation_loader = DataLoader(
        core.TransitionDataset(validation_rows, body_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    device = torch.device(args.device)
    group_order = [str(row["logical_group"]) for row in train_rows]
    bootstrap = core.logical_group_bootstrap_weights(
        group_order, members=5, seed=args.split_seed
    )
    group_weight = {
        group: bootstrap[:, index].tolist() for index, group in enumerate(group_order)
    }
    source_negative_to_positive_ratio = float(
        (successes <= 0.5).sum() / (successes > 0.5).sum()
    )
    base_loss_weights = dict(core.DEFAULT_LOSS_WEIGHTS)
    base_loss_weights["object"] = 0.0
    if args.ablation_variant == "success_only":
        base_loss_weights = {name: 0.0 for name in base_loss_weights}
        base_loss_weights["success"] = 1.0
    elif args.ablation_variant == "no_time_duration":
        base_loss_weights["duration"] = 0.0
    members = []
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = EffectAlignedSharedEventHead(args.ablation_variant).to(device)
        model.action.set_normalization(
            torch.as_tensor(action_mean, device=device),
            torch.as_tensor(action_std, device=device),
        )
        model.set_state_normalization(
            torch.as_tensor(state_mean, device=device),
            torch.as_tensor(state_std, device=device),
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=CompleteDecisionBatchSampler(
                train_rows, batch_size=args.batch_size, seed=seed
            ),
            collate_fn=core.collate_rows,
        )
        iterator = iter(loader)
        best_key = None
        best_metrics = None
        best_step = 0
        checkpoint = output / f"member_{member:02d}_seed_{seed}_best.pt"
        for step in range(1, args.steps + 1):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = core._move_batch(raw, device)
            weights = torch.tensor(
                [group_weight[group][member] for group in raw["logical_group"]],
                device=device,
            )
            prediction = model(batch)
            multitask_loss, pieces = core.compute_multitask_loss(
                prediction,
                batch,
                sample_weight=weights,
                loss_weights=base_loss_weights,
            )
            decision_loss, decision_pieces = _effect_aligned_loss(
                prediction,
                batch,
                weights,
                ablation_variant=args.ablation_variant,
            )
            loss = multitask_loss + decision_loss
            if not torch.isfinite(loss):
                raise FiveBodyContractError("non-finite shared-head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if step % args.eval_every and step != args.steps:
                continue
            metrics = core.evaluate_validation_model(model, validation_loader, device)
            metrics["candidate_ranking"] = evaluate_candidate_ranking(
                model, validation_loader, device
            )
            diagnostic_score, components = core.validation_selection_score(
                metrics, validation_baseline
            )
            selection_components = ablation_selection_components(
                components, args.ablation_variant
            )
            diagnostic_score = float(np.mean(list(selection_components.values())))
            metrics["diagnostic_multitask_score"] = diagnostic_score
            metrics["diagnostic_multitask_components"] = components
            metrics["checkpoint_selection_diagnostic_components"] = selection_components
            metrics["train_objective_last"] = {
                "total": float(loss.detach()),
                **{name: float(value.detach()) for name, value in pieces.items() if name != "total"},
                **{name: float(value.detach()) for name, value in decision_pieces.items()},
            }
            ranking = metrics["candidate_ranking"]
            pairwise = ranking["pairwise_accuracy"]
            key = (
                -float(ranking["macro_delta_success_rate"]),
                -float(ranking["macro_selected_success_rate"]),
                -float(pairwise) if pairwise is not None else 0.0,
                float(diagnostic_score),
                int(step),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_metrics = metrics
                best_step = step
                torch.save(
                    {
                        "format": FORMAT,
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(model.config),
                        "member": member,
                        "seed": seed,
                        "step": step,
                        "held_out_body": args.held_out_body,
                        "source_bodies": preflight["source_bodies"],
                        "body_adapter": "single_shared_row_zero_heldout_parameters",
                        "model_family": "effect_aligned_time_aware_shared_event_head_v3",
                        "ablation": ablation_contract(args.ablation_variant),
                        "candidate_rank_contract": checkpoint_candidate_rank_contract(
                            args.ablation_variant
                        ),
                        "canonical_state_schema": CANONICAL_STATE_SCHEMA,
                        "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
                        "event_spec_sha256": EVENT_SPEC_SHA256,
                        "event_derivation_implementation_sha256": preflight[
                            "event_derivation_implementation_sha256"
                        ],
                        "action_stem_count": 1,
                        "body_to_id_source_only": body_to_id,
                        "heldout_rows_used_for_training_normalization_or_selection": 0,
                        "actor_frozen": True,
                        "action_normalization": normalization,
                        "state_normalization": state_normalization,
                        "preflight_logical_sha256": preflight["logical_sha256"],
                        "validation": metrics,
                    },
                    checkpoint,
                )
            model.train()
        if best_metrics is None:
            raise FiveBodyContractError("ensemble member selected no checkpoint")
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": best_step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_validation": best_metrics,
            }
        )
    summary = {
        "format": FORMAT,
        "status": "source_only_checkpoint_selection_complete",
        "held_out_body": args.held_out_body,
        "source_bodies": preflight["source_bodies"],
        "canonical_state_schema": CANONICAL_STATE_SCHEMA,
        "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": preflight[
            "event_derivation_implementation_sha256"
        ],
        "candidate_rank_contract": summary_candidate_rank_contract(
            args.ablation_variant
        ),
        "ablation": ablation_contract(args.ablation_variant),
        "training_budget": {
            "steps_per_member": args.steps,
            "eval_every_steps": args.eval_every,
            "batch_size_rows": args.batch_size,
            "learning_rate": args.learning_rate,
            "ensemble_members": len(args.ensemble_seeds),
        },
        "mixed_outcome_source_decisions": mixed_outcome_decisions,
        "source_negative_to_positive_ratio": source_negative_to_positive_ratio,
        "success_probability_training_loss": "unweighted_proper_binary_cross_entropy",
        "checkpoint_selection_primary": "validation_best_of_4_delta_success_rate",
        "members": members,
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "heldout_specific_trainable_parameters": 0,
        "actor_frozen": True,
        "same_ordered_candidate_set_required_for_live_evaluation": True,
        "task_success_evaluation_authorized": False,
        "next_required_stage": "label_blind_live_candidate_scoring_then_paired_success_evaluator",
        "preflight": preflight,
    }
    core.atomic_json(output / "training_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train-fold"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--held-out-body", choices=BODIES, required=True)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--ablation-variant", choices=ABLATION_VARIANTS, default="full"
    )
    parser.add_argument(
        "--ensemble-seeds", nargs=5, type=int,
        default=[20260901, 20260902, 20260903, 20260904, 20260905],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = load_binding(args.binding, args.binding_sha256)
    if args.mode == "preflight":
        receipt = build_preflight_receipt(
            audit, held_out_body=args.held_out_body, split_seed=args.split_seed
        )
        print("PREFLIGHT=" + json.dumps(receipt, sort_keys=True))
        return
    if args.output is None:
        raise FiveBodyContractError("train-fold requires --output")
    if args.steps <= 0 or args.eval_every <= 0:
        raise FiveBodyContractError("steps/eval-every must be positive")
    print("TRAINING=" + json.dumps(_train_fold(args, audit), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ABLATION_VARIANTS",
    "ACTOR_FORMAT", "BINDING_FORMAT", "BODIES", "CANONICAL_ACTION_SCHEMA",
    "CANONICAL_STATE_SCHEMA", "CompleteDecisionBatchSampler",
    "CONDITIONS", "FORMAT",
    "EffectAlignedSharedEventHead", "FiveBodyContractError", "MANIFEST_FORMAT",
    "EVENT_SPEC_SHA256", "MATERIALIZATION_FORMAT",
    "ablation_contract", "ablation_selection_components",
    "build_preflight_receipt", "canonical_sha256",
    "checkpoint_candidate_rank_contract", "load_binding",
    "evaluate_candidate_ranking", "materialize_source_rows", "sha256_file",
    "sha256_tree", "source_group_split", "summary_candidate_rank_contract",
    "validate_actor_authority", "validate_body_manifest",
    "validate_materialization_receipt",
]
