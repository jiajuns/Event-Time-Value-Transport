#!/usr/bin/env python3
"""Freeze the label-free Source dense260 reset-identity partition.

This module is deliberately a post-reset, pre-policy freezer.  It accepts one
already completed reset-only candidate manifest plus aggregate-only identity
disjointness attestations.  It never imports a simulator, policy, torch, numpy,
or an HDF reader.  It cannot execute a reset, action, rollout, collection, or
training stage.

The candidate reset manifest must cover the exact preregistered 400-seed
namespace.  The freezer takes the first 260 rows whose resolved seeds are
unique, fails closed if their reset identities are not also unique, then
assigns those rows to 100 train, 80 calibration, and 80 independent-validation
groups by a frozen SHA-256 order.  External
attestations prove that the whole candidate pool is disjoint, on requested,
resolved, and reset-identity axes, from every registered prior/protected role.
Only commitments and zero intersection counts are accepted; private identities
must not appear in an attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_smolvla_schema5_source_dense260_preregistration_v1"
STATUS = (
    "complete_label_free_reset_identity_partition_collection_not_authorized"
)
RESET_MANIFEST_FORMAT = (
    "etsf_smolvla_schema5_source_dense260_reset_candidate_manifest_v1"
)
RESET_MANIFEST_STATUS = (
    "complete_reset_identity_only_no_policy_step_action_trajectory_or_label"
)
ATTESTATION_FORMAT = (
    "etsf_source_dense260_aggregate_identity_disjoint_attestation_v1"
)
ATTESTATION_STATUS = (
    "verified_disjoint_without_disclosing_reference_identities"
)

NAMESPACE = "schema5_aloha_source_dense260_20260829_v1"
CANDIDATE_START = 2_026_083_500
CANDIDATE_COUNT = 400
CANDIDATE_STEP = 1
SELECTED_GROUPS = 260
SPLIT_COUNTS = {
    "train": 100,
    "calibration": 80,
    "validation": 80,
}
TASK = "move_can_pot"
BODY = "aloha-agilex"
POLICY = "smolvla"
ACTOR_ID = "smolvla_aloha_agilex_source_dense260"
TARGET_ROLE = "source_dense260_reset_candidate_pool"
REQUIRED_REFERENCE_ROLES = frozenset(
    {
        "official150",
        "source63",
        "prior_development",
        "piper_development300",
        "formal_target_validation",
        "evaluation400",
    }
)
AXES = ("requested_seed", "resolved_seed", "reset_identity")
EVENTS = ("e0", "e12", "e3", "e4", "eK")
SHA_CHARS = frozenset("0123456789abcdef")
MAX_SEED = 2**31 - 1
FORBIDDEN_PATH_MARKERS = frozenset(
    {
        "fresh",
        "confirmation",
        "protected",
        "evaluation",
        "formal",
        "target",
        "trajectory",
        "label",
        "outcome",
        "hdf",
    }
)

RESET_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "status",
        "namespace",
        "task",
        "body",
        "policy",
        "candidate_range",
        "reset_identity_contract",
        "capability",
        "rows",
        "candidate_identity_counts",
        "candidate_identity_sets_sha256",
        "manifest_sha256",
    }
)
RESET_ROW_FIELDS = frozenset(
    {
        "ordinal",
        "requested_seed",
        "status",
        "resolved_seed",
        "reset_identity_sha256",
    }
)
ATTESTATION_FIELDS = frozenset(
    {
        "format",
        "status",
        "reference_role",
        "target_role",
        "reference_identity_sets_sha256",
        "target_identity_sets_sha256",
        "intersection_counts",
        "sensitive_identities_included",
        "only_aggregate_commitments_disclosed",
        "attestation_sha256",
    }
)
GROUP_FIELDS = frozenset(
    {
        "global_ordinal",
        "split",
        "split_ordinal",
        "selection_ordinal",
        "candidate_ordinal",
        "requested_seed",
        "resolved_seed",
        "reset_identity_sha256",
        "group_id",
        "split_order_sha256",
    }
)
PREREGISTRATION_FIELDS = frozenset(
    {
        "format",
        "status",
        "namespace",
        "task",
        "body",
        "policy",
        "actor_id",
        "purpose",
        "candidate_contract",
        "reset_candidate_source",
        "external_identity_attestations",
        "selection_audit",
        "groups",
        "collection_contract",
        "capability",
        "stopping_and_extension",
        "preregistration_sha256",
    }
)

RESET_CAPABILITY = {
    "reset_identity_only": True,
    "environment_reset_per_candidate_maximum": 1,
    "environment_step_calls": 0,
    "policy_import_or_forward_calls": 0,
    "action_generation_or_execution_calls": 0,
    "trajectory_or_hdf_files_opened": 0,
    "reward_success_event_or_outcome_read": False,
    "labels_read": False,
    "images_or_policy_features_persisted": False,
    "internal_seed_retry_allowed": False,
}

RESET_IDENTITY_CONTRACT = {
    "format": "etsf_cross_body_semantic_reset_identity_v1",
    "hash_algorithm": "sha256",
    "canonicalization": "json_sort_keys_compact_ascii_no_nan",
    "payload_fields": [
        "task",
        "instruction_semantics_sha256",
        "initial_scene_state_sha256",
    ],
    "capture_boundary": "after_reset_before_policy_action_reward_event_or_label",
    "initial_scene_state_contract": (
        "sorted_semantic_object_names_with_canonical_float32_world_poses_v1"
    ),
    "embodiment_policy_joint_and_drive_state_excluded": True,
    "requested_and_resolved_seed_excluded_from_reset_identity": True,
    "component_values_disclosed_in_public_manifest": False,
}


class SourceDense260PreregistrationError(RuntimeError):
    """A reset identity, attestation, path, split, or signature failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _sign(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise SourceDense260PreregistrationError(
            f"refusing to replace signature field {field}"
        )
    result[field] = canonical_sha256(result)
    return result


def _verify_signature(
    value: Mapping[str, Any], field: str, role: str
) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise SourceDense260PreregistrationError(f"{role} signature mismatch")
    return str(recorded)


def _strict_json(raw: bytes, role: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("nonstandard constant")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise SourceDense260PreregistrationError(
            f"{role} is not strict JSON"
        ) from None
    if not isinstance(value, dict):
        raise SourceDense260PreregistrationError(
            f"{role} must contain one JSON object"
        )
    return value


def _reject_sensitive_path(path: Path, role: str) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise SourceDense260PreregistrationError(
            f"{role} path must be absolute"
        )
    absolute = raw.absolute()
    for component in absolute.parts:
        lowered = component.casefold()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise SourceDense260PreregistrationError(
                f"{role} path uses a forbidden protected namespace"
            )
    return absolute


def _reject_symlink_components(path: Path, role: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            if current.is_symlink():
                raise SourceDense260PreregistrationError(
                    f"{role} path contains a symbolic link"
                )
        except OSError:
            raise SourceDense260PreregistrationError(
                f"{role} path is unavailable"
            ) from None


def _read_regular_file(path: Path, role: str) -> tuple[bytes, str]:
    absolute = _reject_sensitive_path(path, role)
    _reject_symlink_components(absolute, role)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(absolute), flags)
    except OSError:
        raise SourceDense260PreregistrationError(
            f"{role} is unavailable or unsafe"
        ) from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 2:
            raise SourceDense260PreregistrationError(
                f"{role} is not a non-empty regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if (
            remaining != 0
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
        ):
            raise SourceDense260PreregistrationError(
                f"{role} changed while being authenticated"
            )
        raw = b"".join(chunks)
        return raw, hashlib.sha256(raw).hexdigest()
    except SourceDense260PreregistrationError:
        raise
    except OSError:
        raise SourceDense260PreregistrationError(
            f"{role} could not be read safely"
        ) from None
    finally:
        os.close(descriptor)


def _read_json_file(path: Path, role: str) -> tuple[dict[str, Any], str]:
    raw, file_digest = _read_regular_file(path, role)
    return _strict_json(raw, role), file_digest


def _set_commitment(values: Sequence[int | str]) -> str:
    return canonical_sha256(sorted(set(values)))


def _candidate_axis_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int | str]]:
    stable = [row for row in rows if row["status"] == "stable_reset_identity_observed"]
    return {
        "requested_seed": [int(row["requested_seed"]) for row in rows],
        "resolved_seed": [int(row["resolved_seed"]) for row in stable],
        "reset_identity": [str(row["reset_identity_sha256"]) for row in stable],
    }


def _axis_commitments(
    axes: Mapping[str, Sequence[int | str]]
) -> dict[str, str]:
    return {axis: _set_commitment(axes[axis]) for axis in AXES}


def validate_reset_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest_sha = _verify_signature(
        value, "manifest_sha256", "reset candidate manifest"
    )
    if set(value) != RESET_MANIFEST_FIELDS:
        raise SourceDense260PreregistrationError(
            "reset candidate manifest fields changed"
        )
    expected_range = {
        "start": CANDIDATE_START,
        "count": CANDIDATE_COUNT,
        "step": CANDIDATE_STEP,
    }
    if (
        value.get("format") != RESET_MANIFEST_FORMAT
        or value.get("status") != RESET_MANIFEST_STATUS
        or value.get("namespace") != NAMESPACE
        or value.get("task") != TASK
        or value.get("body") != BODY
        or value.get("policy") != POLICY
        or value.get("candidate_range") != expected_range
        or value.get("reset_identity_contract") != RESET_IDENTITY_CONTRACT
        or value.get("capability") != RESET_CAPABILITY
    ):
        raise SourceDense260PreregistrationError(
            "reset candidate scope or pre-policy capability changed"
        )
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != CANDIDATE_COUNT:
        raise SourceDense260PreregistrationError(
            "reset candidate manifest must contain all 400 candidates"
        )
    expected_requested = [
        CANDIDATE_START + CANDIDATE_STEP * ordinal
        for ordinal in range(CANDIDATE_COUNT)
    ]
    if expected_requested[-1] > MAX_SEED:
        raise SourceDense260PreregistrationError(
            "frozen candidate namespace exceeds the simulator seed range"
        )
    stable_count = 0
    for ordinal, (row, requested) in enumerate(zip(rows, expected_requested, strict=True)):
        if not isinstance(row, Mapping) or set(row) != RESET_ROW_FIELDS:
            raise SourceDense260PreregistrationError(
                "reset candidate row schema changed"
            )
        status = row.get("status")
        if row.get("ordinal") != ordinal or row.get("requested_seed") != requested:
            raise SourceDense260PreregistrationError(
                "reset candidate order or requested seed changed"
            )
        if status == "stable_reset_identity_observed":
            if (
                not _strict_int(row.get("resolved_seed"))
                or int(row["resolved_seed"]) > MAX_SEED
                or not _is_sha(row.get("reset_identity_sha256"))
            ):
                raise SourceDense260PreregistrationError(
                    "stable reset candidate lacks a valid resolved identity"
                )
            stable_count += 1
        elif status == "reset_identity_unavailable_failed_closed":
            if (
                row.get("resolved_seed") is not None
                or row.get("reset_identity_sha256") is not None
            ):
                raise SourceDense260PreregistrationError(
                    "unavailable reset candidate disclosed a fabricated identity"
                )
        else:
            raise SourceDense260PreregistrationError(
                "reset candidate status is not label-free and recognized"
            )
    axes = _candidate_axis_values(rows)
    counts = {axis: len(set(axes[axis])) for axis in AXES}
    commitments = _axis_commitments(axes)
    if (
        value.get("candidate_identity_counts") != counts
        or value.get("candidate_identity_sets_sha256") != commitments
    ):
        raise SourceDense260PreregistrationError(
            "reset candidate identity commitments changed"
        )
    return {
        "manifest_sha256": manifest_sha,
        "rows": [dict(row) for row in rows],
        "stable_rows": stable_count,
        "identity_counts": counts,
        "identity_sets_sha256": commitments,
    }


def validate_identity_attestation(
    value: Mapping[str, Any],
    *,
    target_identity_sets_sha256: Mapping[str, str],
    expected_reference_role: str | None = None,
) -> dict[str, Any]:
    attestation_sha = _verify_signature(
        value, "attestation_sha256", "identity disjointness attestation"
    )
    if set(value) != ATTESTATION_FIELDS:
        raise SourceDense260PreregistrationError(
            "identity disjointness attestation fields changed"
        )
    role = value.get("reference_role")
    reference = value.get("reference_identity_sets_sha256")
    target = value.get("target_identity_sets_sha256")
    intersections = value.get("intersection_counts")
    if (
        value.get("format") != ATTESTATION_FORMAT
        or value.get("status") != ATTESTATION_STATUS
        or role not in REQUIRED_REFERENCE_ROLES
        or (expected_reference_role is not None and role != expected_reference_role)
        or value.get("target_role") != TARGET_ROLE
        or not isinstance(reference, Mapping)
        or set(reference) != set(AXES)
        or any(not _is_sha(reference.get(axis)) for axis in AXES)
        or target != dict(target_identity_sets_sha256)
        or intersections != {axis: 0 for axis in AXES}
        or value.get("sensitive_identities_included") is not False
        or value.get("only_aggregate_commitments_disclosed") is not True
    ):
        raise SourceDense260PreregistrationError(
            "identity disjointness attestation is incomplete or nonzero"
        )
    return {
        "reference_role": str(role),
        "reference_identity_sets_sha256": dict(reference),
        "attestation_sha256": attestation_sha,
    }


def _select_first_resolved_unique(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    seen_resolved: set[int] = set()
    skipped = {
        "reset_identity_unavailable": 0,
        "duplicate_resolved_seed": 0,
        "after_first_260_unique": 0,
    }
    for row in rows:
        if len(selected) == SELECTED_GROUPS:
            skipped["after_first_260_unique"] += 1
            continue
        if row["status"] != "stable_reset_identity_observed":
            skipped["reset_identity_unavailable"] += 1
            continue
        resolved = int(row["resolved_seed"])
        reset_identity = str(row["reset_identity_sha256"])
        if resolved in seen_resolved:
            skipped["duplicate_resolved_seed"] += 1
            continue
        seen_resolved.add(resolved)
        selected.append(
            {
                "selection_ordinal": len(selected),
                "candidate_ordinal": int(row["ordinal"]),
                "requested_seed": int(row["requested_seed"]),
                "resolved_seed": resolved,
                "reset_identity_sha256": reset_identity,
            }
        )
    if len(selected) != SELECTED_GROUPS:
        raise SourceDense260PreregistrationError(
            "candidate reset manifest has fewer than 260 resolved-unique identities"
        )
    if len({row["reset_identity_sha256"] for row in selected}) != SELECTED_GROUPS:
        raise SourceDense260PreregistrationError(
            "first 260 resolved-unique candidates contain duplicate reset identities"
        )
    return selected, skipped


def _split_order_sha(row: Mapping[str, Any]) -> str:
    identity = {
        "namespace": NAMESPACE,
        "requested_seed": int(row["requested_seed"]),
        "resolved_seed": int(row["resolved_seed"]),
        "reset_identity_sha256": str(row["reset_identity_sha256"]),
    }
    return canonical_sha256(identity)


def _partition(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in selected),
        key=lambda row: (_split_order_sha(row), int(row["candidate_ordinal"])),
    )
    result: list[dict[str, Any]] = []
    offset = 0
    global_ordinal = 0
    for split, count in SPLIT_COUNTS.items():
        for split_ordinal, row in enumerate(ordered[offset : offset + count]):
            identity = {
                "task": TASK,
                "body": BODY,
                "policy": POLICY,
                "requested_seed": int(row["requested_seed"]),
                "resolved_seed": int(row["resolved_seed"]),
                "reset_identity_sha256": str(row["reset_identity_sha256"]),
            }
            result.append(
                {
                    "global_ordinal": global_ordinal,
                    "split": split,
                    "split_ordinal": split_ordinal,
                    **row,
                    "group_id": canonical_sha256(identity),
                    "split_order_sha256": _split_order_sha(row),
                }
            )
            global_ordinal += 1
        offset += count
    if offset != len(ordered) or len(result) != SELECTED_GROUPS:
        raise SourceDense260PreregistrationError(
            "frozen split counts do not cover the selected identities"
        )
    return result


def build_preregistration(
    *,
    reset_manifest: Mapping[str, Any],
    reset_manifest_file_sha256: str,
    attestations: Sequence[tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    if not _is_sha(reset_manifest_file_sha256):
        raise SourceDense260PreregistrationError(
            "reset candidate manifest file SHA256 is invalid"
        )
    reset = validate_reset_manifest(reset_manifest)
    target_commitments = reset["identity_sets_sha256"]
    attestation_records: dict[str, dict[str, Any]] = {}
    for raw, file_digest in attestations:
        if not _is_sha(file_digest):
            raise SourceDense260PreregistrationError(
                "identity attestation file SHA256 is invalid"
            )
        decoded = validate_identity_attestation(
            raw, target_identity_sets_sha256=target_commitments
        )
        role = decoded["reference_role"]
        if role in attestation_records:
            raise SourceDense260PreregistrationError(
                "identity attestation reference role is duplicated"
            )
        attestation_records[role] = {
            **decoded,
            "file_sha256": file_digest,
        }
    if set(attestation_records) != set(REQUIRED_REFERENCE_ROLES):
        raise SourceDense260PreregistrationError(
            "identity attestations do not cover every required reference role"
        )
    selected, selection_audit = _select_first_resolved_unique(reset["rows"])
    groups = _partition(selected)
    selected_axes = {
        "requested_seed": [row["requested_seed"] for row in selected],
        "resolved_seed": [row["resolved_seed"] for row in selected],
        "reset_identity": [row["reset_identity_sha256"] for row in selected],
    }
    selected_commitments = _axis_commitments(selected_axes)
    selected_counts = {axis: len(set(selected_axes[axis])) for axis in AXES}
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "namespace": NAMESPACE,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "actor_id": ACTOR_ID,
        "purpose": (
            "source_development_only_never_fresh_confirmation_formal_or_evaluation"
        ),
        "candidate_contract": {
            "range": {
                "start": CANDIDATE_START,
                "count": CANDIDATE_COUNT,
                "step": CANDIDATE_STEP,
            },
            "selection_rule": (
                "first_260_candidate_order_rows_with_unique_resolved_seed_then_require_unique_reset_identity"
            ),
            "split_assignment": (
                "sha256_order_over_selected_requested_resolved_reset_identity_v1"
            ),
            "selected_groups": SELECTED_GROUPS,
            "split_counts": dict(SPLIT_COUNTS),
            "labels_or_policy_outputs_used_for_selection": False,
            "adaptive_replacement_after_label_access": False,
        },
        "reset_candidate_source": {
            "file_sha256": reset_manifest_file_sha256,
            "manifest_sha256": reset["manifest_sha256"],
            "stable_rows": reset["stable_rows"],
            "candidate_identity_counts": reset["identity_counts"],
            "candidate_identity_sets_sha256": target_commitments,
        },
        "external_identity_attestations": [
            attestation_records[role] for role in sorted(attestation_records)
        ],
        "selection_audit": {
            "selected_groups": SELECTED_GROUPS,
            "selected_identity_counts": selected_counts,
            "selected_identity_sets_sha256": selected_commitments,
            "skipped_before_or_after_selection": selection_audit,
            "requested_resolved_reset_identity_unique_within_selected": True,
            "requested_resolved_reset_identity_disjoint_across_splits": True,
            "external_intersection_count_on_every_axis_and_role": 0,
        },
        "groups": groups,
        "collection_contract": {
            "schema_version": 5,
            "event_vocab": list(EVENTS),
            "candidate_count": 8,
            "action_exec_steps": 5,
            "max_steps": 200,
            "action_chunk": 50,
            "action_dim": 14,
            "split_unit": (
                "task_body_policy_requested_resolved_reset_identity_logical_group"
            ),
            "root_candidate_state_bit_exact_required": True,
            "unique_deterministic_baseline_required": True,
            "fixed_instruction_and_simulator_root_required": True,
            "per_exec5_event_state_supervision_required": True,
            "event_transition_and_right_censored_duration_supervision_required": True,
            "success_failure_and_recovery_supervision_required": True,
            "per_step_object_pose_and_proprio_supervision_required": True,
            "object_state_delta_supervision_required": True,
            "candidate_dispersion_and_calibration_split_for_uncertainty_required": True,
        },
        "capability": {
            "input_scope": (
                "reset_identity_manifest_and_aggregate_disjoint_attestations_only"
            ),
            "simulator_imported": False,
            "environment_reset_by_this_freezer": False,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "action_generation_or_execution_calls": 0,
            "trajectory_or_hdf_files_opened": 0,
            "labels_or_outcomes_read": False,
            "fresh_confirmation_formal_or_evaluation_identity_disclosed": False,
            "collection_authorized": False,
            "training_authorized": False,
            "performance_or_transfer_claim_authorized": False,
        },
        "stopping_and_extension": {
            "freeze_requires_all_400_reset_candidate_rows": True,
            "selection_stops_at_first_260_unique_before_any_label": True,
            "early_stop_or_extension_based_on_success_event_recovery_or_metric": False,
            "insufficient_post_collection_support_action": (
                "fail_closed_or_create_new_versioned_preregistration"
            ),
            "existing_manifest_membership_mutation_allowed": False,
        },
    }
    result = _sign(base, "preregistration_sha256")
    validate_preregistration(result)
    return result


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    signature = _verify_signature(
        value, "preregistration_sha256", "Source dense260 preregistration"
    )
    if (
        set(value) != PREREGISTRATION_FIELDS
        or value.get("format") != FORMAT
        or value.get("status") != STATUS
        or value.get("namespace") != NAMESPACE
        or value.get("task") != TASK
        or value.get("body") != BODY
        or value.get("policy") != POLICY
        or value.get("actor_id") != ACTOR_ID
        or value.get("purpose")
        != "source_development_only_never_fresh_confirmation_formal_or_evaluation"
    ):
        raise SourceDense260PreregistrationError(
            "Source dense260 preregistration scope changed"
        )
    candidate = value.get("candidate_contract")
    collection = value.get("collection_contract")
    capability = value.get("capability")
    source = value.get("reset_candidate_source")
    attestations = value.get("external_identity_attestations")
    groups = value.get("groups")
    audit = value.get("selection_audit")
    if not all(
        isinstance(item, Mapping)
        for item in (candidate, collection, capability, source, audit)
    ) or not isinstance(attestations, list) or not isinstance(groups, list):
        raise SourceDense260PreregistrationError(
            "Source dense260 preregistration sections are incomplete"
        )
    if (
        set(candidate)
        != {
            "range",
            "selection_rule",
            "split_assignment",
            "selected_groups",
            "split_counts",
            "labels_or_policy_outputs_used_for_selection",
            "adaptive_replacement_after_label_access",
        }
        or candidate.get("range")
        != {
            "start": CANDIDATE_START,
            "count": CANDIDATE_COUNT,
            "step": CANDIDATE_STEP,
        }
        or candidate.get("selected_groups") != SELECTED_GROUPS
        or candidate.get("split_counts") != SPLIT_COUNTS
        or candidate.get("selection_rule")
        != "first_260_candidate_order_rows_with_unique_resolved_seed_then_require_unique_reset_identity"
        or candidate.get("split_assignment")
        != "sha256_order_over_selected_requested_resolved_reset_identity_v1"
        or candidate.get("labels_or_policy_outputs_used_for_selection") is not False
        or candidate.get("adaptive_replacement_after_label_access") is not False
    ):
        raise SourceDense260PreregistrationError(
            "candidate selection or split contract changed"
        )
    if collection != {
        "schema_version": 5,
        "event_vocab": list(EVENTS),
        "candidate_count": 8,
        "action_exec_steps": 5,
        "max_steps": 200,
        "action_chunk": 50,
        "action_dim": 14,
        "split_unit": (
            "task_body_policy_requested_resolved_reset_identity_logical_group"
        ),
        "root_candidate_state_bit_exact_required": True,
        "unique_deterministic_baseline_required": True,
        "fixed_instruction_and_simulator_root_required": True,
        "per_exec5_event_state_supervision_required": True,
        "event_transition_and_right_censored_duration_supervision_required": True,
        "success_failure_and_recovery_supervision_required": True,
        "per_step_object_pose_and_proprio_supervision_required": True,
        "object_state_delta_supervision_required": True,
        "candidate_dispersion_and_calibration_split_for_uncertainty_required": True,
    }:
        raise SourceDense260PreregistrationError(
            "immutable dense collection contract changed"
        )
    if capability != {
        "input_scope": (
            "reset_identity_manifest_and_aggregate_disjoint_attestations_only"
        ),
        "simulator_imported": False,
        "environment_reset_by_this_freezer": False,
        "environment_step_calls": 0,
        "policy_import_or_forward_calls": 0,
        "action_generation_or_execution_calls": 0,
        "trajectory_or_hdf_files_opened": 0,
        "labels_or_outcomes_read": False,
        "fresh_confirmation_formal_or_evaluation_identity_disclosed": False,
        "collection_authorized": False,
        "training_authorized": False,
        "performance_or_transfer_claim_authorized": False,
    }:
        raise SourceDense260PreregistrationError(
            "pre-policy fail-closed capability changed"
        )
    target_commitments = source.get("candidate_identity_sets_sha256")
    if (
        set(source)
        != {
            "file_sha256",
            "manifest_sha256",
            "stable_rows",
            "candidate_identity_counts",
            "candidate_identity_sets_sha256",
        }
        or not _is_sha(source.get("file_sha256"))
        or not _is_sha(source.get("manifest_sha256"))
        or not isinstance(target_commitments, Mapping)
        or set(target_commitments) != set(AXES)
        or any(not _is_sha(target_commitments.get(axis)) for axis in AXES)
        or not _strict_int(source.get("stable_rows"))
        or not isinstance(source.get("candidate_identity_counts"), Mapping)
        or set(source["candidate_identity_counts"]) != set(AXES)
        or any(
            not _strict_int(source["candidate_identity_counts"].get(axis))
            for axis in AXES
        )
        or source["candidate_identity_counts"].get("requested_seed")
        != CANDIDATE_COUNT
        or not SELECTED_GROUPS <= int(source["stable_rows"]) <= CANDIDATE_COUNT
        or any(
            not SELECTED_GROUPS
            <= int(source["candidate_identity_counts"].get(axis))
            <= int(source["stable_rows"])
            for axis in ("resolved_seed", "reset_identity")
        )
    ):
        raise SourceDense260PreregistrationError(
            "reset candidate source binding is invalid"
        )
    roles: set[str] = set()
    for item in attestations:
        if not isinstance(item, Mapping):
            raise SourceDense260PreregistrationError(
                "external identity attestation record is invalid"
            )
        expected = {
            "reference_role",
            "reference_identity_sets_sha256",
            "attestation_sha256",
            "file_sha256",
        }
        role = item.get("reference_role")
        reference = item.get("reference_identity_sets_sha256")
        if (
            set(item) != expected
            or role not in REQUIRED_REFERENCE_ROLES
            or role in roles
            or not isinstance(reference, Mapping)
            or set(reference) != set(AXES)
            or any(not _is_sha(reference.get(axis)) for axis in AXES)
            or not _is_sha(item.get("attestation_sha256"))
            or not _is_sha(item.get("file_sha256"))
        ):
            raise SourceDense260PreregistrationError(
                "external identity attestation inventory is invalid"
            )
        attestation_payload = {
            "format": ATTESTATION_FORMAT,
            "status": ATTESTATION_STATUS,
            "reference_role": role,
            "target_role": TARGET_ROLE,
            "reference_identity_sets_sha256": dict(reference),
            "target_identity_sets_sha256": dict(target_commitments),
            "intersection_counts": {axis: 0 for axis in AXES},
            "sensitive_identities_included": False,
            "only_aggregate_commitments_disclosed": True,
        }
        if item.get("attestation_sha256") != canonical_sha256(
            attestation_payload
        ):
            raise SourceDense260PreregistrationError(
                "external identity attestation signature cannot be reconstructed"
            )
        roles.add(str(role))
    if roles != set(REQUIRED_REFERENCE_ROLES):
        raise SourceDense260PreregistrationError(
            "external identity attestation roles are incomplete"
        )
    if len(groups) != SELECTED_GROUPS:
        raise SourceDense260PreregistrationError(
            "dense260 group inventory has the wrong size"
        )
    per_split = {split: 0 for split in SPLIT_COUNTS}
    axes_by_split = {
        split: {axis: set() for axis in AXES} for split in SPLIT_COUNTS
    }
    selection_ordinals: set[int] = set()
    candidate_ordinals: set[int] = set()
    split_order: list[tuple[str, int]] = []
    for global_ordinal, row in enumerate(groups):
        if not isinstance(row, Mapping) or set(row) != GROUP_FIELDS:
            raise SourceDense260PreregistrationError(
                "dense260 group row schema changed"
            )
        split = row.get("split")
        if (
            row.get("global_ordinal") != global_ordinal
            or split not in SPLIT_COUNTS
            or row.get("split_ordinal") != per_split[str(split)]
            or not _strict_int(row.get("selection_ordinal"))
            or int(row["selection_ordinal"]) >= SELECTED_GROUPS
            or not _strict_int(row.get("candidate_ordinal"))
            or int(row["candidate_ordinal"]) >= CANDIDATE_COUNT
            or row.get("requested_seed")
            != CANDIDATE_START + CANDIDATE_STEP * int(row["candidate_ordinal"])
            or not _strict_int(row.get("resolved_seed"))
            or not _is_sha(row.get("reset_identity_sha256"))
        ):
            raise SourceDense260PreregistrationError(
                "dense260 group identity or ordinal is invalid"
            )
        identity = {
            "task": TASK,
            "body": BODY,
            "policy": POLICY,
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
            "reset_identity_sha256": row["reset_identity_sha256"],
        }
        expected_split_sha = _split_order_sha(row)
        if (
            row.get("group_id") != canonical_sha256(identity)
            or row.get("split_order_sha256") != expected_split_sha
        ):
            raise SourceDense260PreregistrationError(
                "dense260 group content address changed"
            )
        per_split[str(split)] += 1
        selection_ordinals.add(int(row["selection_ordinal"]))
        candidate_ordinals.add(int(row["candidate_ordinal"]))
        axes_by_split[str(split)]["requested_seed"].add(row["requested_seed"])
        axes_by_split[str(split)]["resolved_seed"].add(row["resolved_seed"])
        axes_by_split[str(split)]["reset_identity"].add(
            row["reset_identity_sha256"]
        )
        split_order.append((expected_split_sha, int(row["candidate_ordinal"])))
    if (
        per_split != SPLIT_COUNTS
        or selection_ordinals != set(range(SELECTED_GROUPS))
        or len(candidate_ordinals) != SELECTED_GROUPS
    ):
        raise SourceDense260PreregistrationError(
            "dense260 group membership is duplicated or incomplete"
        )
    flattened_axes: dict[str, list[int | str]] = {axis: [] for axis in AXES}
    for axis in AXES:
        sets = [axes_by_split[split][axis] for split in SPLIT_COUNTS]
        if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise SourceDense260PreregistrationError(
                "requested/resolved/reset identity leaks across splits"
            )
        flattened_axes[axis] = [
            row[
                "reset_identity_sha256"
                if axis == "reset_identity"
                else axis
            ]
            for row in groups
        ]
        if len(set(flattened_axes[axis])) != SELECTED_GROUPS:
            raise SourceDense260PreregistrationError(
                "selected requested/resolved/reset identities are not unique"
            )
    expected_selected_commitments = _axis_commitments(flattened_axes)
    if (
        set(audit)
        != {
            "selected_groups",
            "selected_identity_counts",
            "selected_identity_sets_sha256",
            "skipped_before_or_after_selection",
            "requested_resolved_reset_identity_unique_within_selected",
            "requested_resolved_reset_identity_disjoint_across_splits",
            "external_intersection_count_on_every_axis_and_role",
        }
        or audit.get("selected_groups") != SELECTED_GROUPS
        or audit.get("selected_identity_counts")
        != {axis: SELECTED_GROUPS for axis in AXES}
        or audit.get("selected_identity_sets_sha256")
        != expected_selected_commitments
        or audit.get("requested_resolved_reset_identity_unique_within_selected")
        is not True
        or audit.get("requested_resolved_reset_identity_disjoint_across_splits")
        is not True
        or audit.get("external_intersection_count_on_every_axis_and_role") != 0
        or not isinstance(audit.get("skipped_before_or_after_selection"), Mapping)
        or set(audit["skipped_before_or_after_selection"])
        != {
            "reset_identity_unavailable",
            "duplicate_resolved_seed",
            "after_first_260_unique",
        }
        or any(
            not _strict_int(count)
            for count in audit["skipped_before_or_after_selection"].values()
        )
        or sum(audit["skipped_before_or_after_selection"].values())
        != CANDIDATE_COUNT - SELECTED_GROUPS
    ):
        raise SourceDense260PreregistrationError(
            "dense260 selected identity audit is inconsistent"
        )
    ordered_by_hash = sorted(split_order)
    expected_splits = (
        ["train"] * SPLIT_COUNTS["train"]
        + ["calibration"] * SPLIT_COUNTS["calibration"]
        + ["validation"] * SPLIT_COUNTS["validation"]
    )
    observed_by_hash = {
        (row["split_order_sha256"], int(row["candidate_ordinal"])): row["split"]
        for row in groups
    }
    if (
        split_order != ordered_by_hash
        or [observed_by_hash[key] for key in ordered_by_hash] != expected_splits
    ):
        raise SourceDense260PreregistrationError(
            "dense260 SHA-ordered split assignment changed"
        )
    if value.get("stopping_and_extension") != {
        "freeze_requires_all_400_reset_candidate_rows": True,
        "selection_stops_at_first_260_unique_before_any_label": True,
        "early_stop_or_extension_based_on_success_event_recovery_or_metric": False,
        "insufficient_post_collection_support_action": (
            "fail_closed_or_create_new_versioned_preregistration"
        ),
        "existing_manifest_membership_mutation_allowed": False,
    }:
        raise SourceDense260PreregistrationError(
            "stopping or immutable extension contract changed"
        )
    return {
        "status": "verified_source_dense260_label_free_preregistration",
        "preregistration_sha256": signature,
        "groups": SELECTED_GROUPS,
        "split_counts": dict(SPLIT_COUNTS),
        "candidate_count": 8,
        "action_exec_steps": 5,
        "max_steps": 200,
        "external_reference_roles": sorted(roles),
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "collection_authorized": False,
    }


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    absolute = _reject_sensitive_path(path, "output")
    _reject_symlink_components(absolute.parent, "output parent")
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError:
        raise SourceDense260PreregistrationError(
            "output parent is unavailable"
        ) from None
    if not parent.is_dir() or absolute.exists() or absolute.is_symlink():
        raise FileExistsError(absolute)
    validate_preregistration(value)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            os.fspath(absolute), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
        )
    except FileExistsError:
        raise
    except OSError:
        raise SourceDense260PreregistrationError(
            "output could not be created exclusively"
        ) from None
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        # Keep a partial create-once artifact as failure evidence.
        raise


def freeze_from_paths(
    *,
    reset_manifest_path: Path,
    attestation_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    if len(attestation_paths) != len(REQUIRED_REFERENCE_ROLES):
        raise SourceDense260PreregistrationError(
            "exactly six aggregate identity attestations are required"
        )
    reset_manifest, reset_file_sha = _read_json_file(
        reset_manifest_path, "reset candidate manifest"
    )
    attestations: list[tuple[Mapping[str, Any], str]] = []
    for index, path in enumerate(attestation_paths):
        value, digest = _read_json_file(
            path, f"identity attestation {index}"
        )
        attestations.append((value, digest))
    result = build_preregistration(
        reset_manifest=reset_manifest,
        reset_manifest_file_sha256=reset_file_sha,
        attestations=attestations,
    )
    write_json_new(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-manifest", type=Path, required=True)
    parser.add_argument(
        "--identity-attestation", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_from_paths(
        reset_manifest_path=args.reset_manifest,
        attestation_paths=args.identity_attestation,
        output_path=args.output,
    )
    print(
        "SOURCE_DENSE260_PREREGISTERED="
        + json.dumps(
            {
                "groups": SELECTED_GROUPS,
                "split_counts": SPLIT_COUNTS,
                "preregistration_sha256": result["preregistration_sha256"],
                "labels_or_outcomes_read": False,
                "collection_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ATTESTATION_FORMAT",
    "ATTESTATION_STATUS",
    "AXES",
    "BODY",
    "CANDIDATE_COUNT",
    "CANDIDATE_START",
    "CANDIDATE_STEP",
    "FORMAT",
    "NAMESPACE",
    "POLICY",
    "REQUIRED_REFERENCE_ROLES",
    "RESET_CAPABILITY",
    "RESET_IDENTITY_CONTRACT",
    "RESET_MANIFEST_FORMAT",
    "RESET_MANIFEST_STATUS",
    "SELECTED_GROUPS",
    "SPLIT_COUNTS",
    "STATUS",
    "SourceDense260PreregistrationError",
    "TARGET_ROLE",
    "TASK",
    "build_preregistration",
    "canonical_sha256",
    "freeze_from_paths",
    "validate_identity_attestation",
    "validate_preregistration",
    "validate_reset_manifest",
    "write_json_new",
]
