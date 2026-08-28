#!/usr/bin/env python3
"""Pure offline contracts for per-head body-mode selection.

This module deliberately has no filesystem, simulator, HDF, checkpoint,
trajectory, label-loading, training, signing, or deployment capability.  It
validates three content-addressed JSON-shaped documents:

* an unsigned, no-promotion selection plan;
* aggregate OOF calibration evidence supplied in memory; and
* a deterministic per-head selection receipt.

Canonical SHA-256 values below are content addresses only.  They are not
signatures and can never authorize collection, training, deployment, or a
performance claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


PLAN_FORMAT = "etsf_per_head_body_mode_selection_plan_v1"
PLAN_STATUS = "frozen_offline_unsigned_schema_no_promotion_authority"
EVIDENCE_FORMAT = "etsf_per_head_body_mode_calibration_evidence_v1"
EVIDENCE_STATUS = "complete_offline_unsigned_aggregate_no_promotion_authority"
RECEIPT_FORMAT = "etsf_per_head_body_mode_selection_receipt_v1"
RECEIPT_STATUS = "deterministic_offline_selection_no_promotion_authority"

RESET_IDENTITY_FORMAT = "etsf_cross_body_semantic_reset_identity_v2"
EXECUTION_IDENTITY_FORMAT = "etsf_body_actor_execution_group_identity_v1"
SAMPLE_IDENTITY_FORMAT = "etsf_body_actor_execution_sample_identity_v1"

HEADS = (
    "post_event",
    "next_event",
    "duration",
    "success",
    "recovery",
    "object_effect",
)
EVENT_VOCAB = ("e0", "e12", "e3", "e4", "eK")
MEMBER_COUNT = 5
FOLD_COUNT = 5

PURE_CLOCK_SCOPE = "pure_clock_ablation"
FULL_BODY_ADAPTER_SCOPE = "full_body_adapter_ablation"
VARIANT_SCOPES = (PURE_CLOCK_SCOPE, FULL_BODY_ADAPTER_SCOPE)

PURE_REFERENCE_VARIANT = "body_agnostic"
PURE_CANDIDATE_VARIANT = "source_body_clock"
FULL_REFERENCE_VARIANT = "body_agnostic_adapter"
FULL_CANDIDATE_VARIANT = "body_conditioned_adapter"

MINIMUM_SUPPORT_PER_CATEGORY = {
    "post_event": 10,
    "next_event": 10,
    "duration": 10,
    "success": 50,
    "recovery": 10,
    "object_effect": 50,
}
SUPPORT_CATEGORIES = {
    "post_event": EVENT_VOCAB,
    # e0 is an initial/current state, not an observed future reached event.
    "next_event": EVENT_VOCAB[1:],
    "duration": ("observed", "censored"),
    "success": ("positive", "negative"),
    "recovery": ("positive", "negative"),
    "object_effect": ("nonzero", "near_zero"),
}

BOOTSTRAP_SEED = 20260828
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_CONFIDENCE = 0.95
MINIMUM_PAIRED_GAIN_LCB = 0.0
MAXIMUM_HARMFUL_RATE_UCB = 0.10
SHA_CHARS = frozenset("0123456789abcdef")


class BodyModeSelectionError(RuntimeError):
    """An offline schema, identity, OOF, support, or gate invariant failed."""


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


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def _require_sha(value: Any, role: str) -> str:
    if not is_sha256(value):
        raise BodyModeSelectionError(f"{role} must be exact lowercase SHA-256")
    return str(value)


def _require_string(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise BodyModeSelectionError(f"{role} must be a non-empty canonical string")
    return value


def _require_int(
    value: Any,
    role: str,
    *,
    minimum: int | None = None,
    expected: int | None = None,
) -> int:
    if type(value) is not int:
        raise BodyModeSelectionError(f"{role} must be an exact integer, not bool")
    if minimum is not None and value < minimum:
        raise BodyModeSelectionError(f"{role} is below its minimum")
    if expected is not None and value != expected:
        raise BodyModeSelectionError(f"{role} differs from the frozen value")
    return value


def _require_float(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BodyModeSelectionError(f"{role} must be finite numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise BodyModeSelectionError(f"{role} must be finite")
    return result


def _require_bool(value: Any, expected: bool, role: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise BodyModeSelectionError(f"{role} must be exact {expected!r}")
    return value


def _signed_document(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(base)
    if field in value:
        raise BodyModeSelectionError(f"{field} already exists before hashing")
    return {**value, field: canonical_sha256(value)}


def _verify_document(
    value: Mapping[str, Any],
    *,
    field: str,
    expected_fields: set[str],
    role: str,
) -> str:
    if not isinstance(value, Mapping) or set(value) != expected_fields | {field}:
        raise BodyModeSelectionError(f"{role} fields changed")
    logical = _require_sha(value.get(field), f"{role} logical SHA")
    unsigned = {key: child for key, child in value.items() if key != field}
    if logical != canonical_sha256(unsigned):
        raise BodyModeSelectionError(f"{role} canonical SHA mismatch")
    return logical


def _context_base(value: Mapping[str, Any], task: str) -> dict[str, Any]:
    fields = {
        "task",
        "body_id",
        "body_contract_sha256",
        "policy_id",
        "actor_id",
        "actor_contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) not in (fields, fields | {"execution_context_sha256"}):
        raise BodyModeSelectionError("execution context fields changed")
    base = {key: value[key] for key in fields}
    if base["task"] != task:
        raise BodyModeSelectionError("execution context task changed")
    for name in ("task", "body_id", "policy_id", "actor_id"):
        _require_string(base[name], f"execution context {name}")
    for name in (
        "body_contract_sha256",
        "actor_contract_sha256",
    ):
        _require_sha(base[name], f"execution context {name}")
    return base


def _normalize_contexts(
    values: Sequence[Mapping[str, Any]], task: str
) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise BodyModeSelectionError("at least one execution context is required")
    result = []
    seen: set[str] = set()
    for value in values:
        base = _context_base(value, task)
        logical = canonical_sha256(base)
        if "execution_context_sha256" in value and value["execution_context_sha256"] != logical:
            raise BodyModeSelectionError("execution context SHA changed")
        if logical in seen:
            raise BodyModeSelectionError("duplicate execution context")
        seen.add(logical)
        result.append({**base, "execution_context_sha256": logical})
    return sorted(result, key=lambda row: row["execution_context_sha256"])


def _variant_names(scope: str) -> tuple[str, str]:
    if scope == PURE_CLOCK_SCOPE:
        return PURE_REFERENCE_VARIANT, PURE_CANDIDATE_VARIANT
    if scope == FULL_BODY_ADAPTER_SCOPE:
        return FULL_REFERENCE_VARIANT, FULL_CANDIDATE_VARIANT
    raise BodyModeSelectionError("variant scope is not supported")


def _variant_allowed_heads(scope: str) -> tuple[str, ...]:
    return ("duration",) if scope == PURE_CLOCK_SCOPE else HEADS


def _validate_variant_rows(
    scope: str, values: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    reference_name, candidate_name = _variant_names(scope)
    expected_names = {reference_name, candidate_name}
    fields = {
        "variant_id",
        "body_mode",
        "adapter_scope",
        "member_seeds",
        "member_checkpoint_file_sha256",
        "shared_core_checkpoint_sha256",
        "non_clock_state_sha256",
        "clock_state_sha256",
        "clock_contract_sha256",
    }
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 2
    ):
        raise BodyModeSelectionError("exactly two body-mode variants are required")
    decoded: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping) or set(value) != fields:
            raise BodyModeSelectionError("variant contract fields changed")
        name = _require_string(value["variant_id"], "variant id")
        if name not in expected_names or name in decoded:
            raise BodyModeSelectionError("variant id or coverage changed")
        expected_mode = "body_agnostic" if name == reference_name else "body_conditioned"
        if value["body_mode"] != expected_mode:
            raise BodyModeSelectionError("variant body mode changed")
        expected_adapter_scope = (
            "clock_only_non_clock_frozen"
            if scope == PURE_CLOCK_SCOPE
            else "full_body_adapter"
        )
        if value["adapter_scope"] != expected_adapter_scope:
            raise BodyModeSelectionError("variant adapter scope changed")
        seeds = value["member_seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != MEMBER_COUNT
            or len(set(seeds)) != MEMBER_COUNT
            or any(type(seed) is not int for seed in seeds)
        ):
            raise BodyModeSelectionError("variant member seeds changed")
        normalized = dict(value)
        normalized["member_seeds"] = list(seeds)
        for field in (
            "member_checkpoint_file_sha256",
            "shared_core_checkpoint_sha256",
            "non_clock_state_sha256",
            "clock_state_sha256",
        ):
            items = value[field]
            if (
                not isinstance(items, list)
                or len(items) != MEMBER_COUNT
                or any(not is_sha256(item) for item in items)
            ):
                raise BodyModeSelectionError(f"variant {field} changed")
            normalized[field] = list(items)
        _require_sha(value["clock_contract_sha256"], "variant clock contract SHA")
        decoded[name] = normalized
    reference = decoded[reference_name]
    candidate = decoded[candidate_name]
    if reference["member_seeds"] != candidate["member_seeds"]:
        raise BodyModeSelectionError("variants must pair the same five member seeds")
    if reference["shared_core_checkpoint_sha256"] != candidate["shared_core_checkpoint_sha256"]:
        raise BodyModeSelectionError("variants must bind the same shared core members")
    if (
        scope == PURE_CLOCK_SCOPE
        and reference["non_clock_state_sha256"] != candidate["non_clock_state_sha256"]
    ):
        raise BodyModeSelectionError(
            "pure clock ablation must keep every non-clock member state identical"
        )
    return [decoded[reference_name], decoded[candidate_name]]


def build_selection_plan(
    *,
    protocol_namespace: str,
    task: str,
    instruction_semantics_sha256: str,
    reset_identity_contract_sha256: str,
    event_contract_sha256: str,
    object_state_contract_sha256: str,
    dataset_format: str,
    dataset_file_sha256: str,
    dataset_logical_sha256: str,
    partition_sha256: str,
    lane: str,
    lane_group_count: int,
    lane_execution_group_set_sha256: str,
    execution_contexts: Sequence[Mapping[str, Any]],
    variant_scope: str,
    variants: Sequence[Mapping[str, Any]],
    bootstrap_draws_sha256: str,
) -> dict[str, Any]:
    """Build a versioned, unsigned plan that can never authorize promotion."""

    protocol_namespace = _require_string(protocol_namespace, "protocol namespace")
    task = _require_string(task, "task")
    instruction_semantics_sha256 = _require_sha(
        instruction_semantics_sha256, "instruction semantics SHA"
    )
    reset_identity_contract_sha256 = _require_sha(
        reset_identity_contract_sha256, "reset identity contract SHA"
    )
    contexts = _normalize_contexts(execution_contexts, task)
    decoded_variants = _validate_variant_rows(variant_scope, variants)
    reference, candidate = _variant_names(variant_scope)
    base = {
        "format": PLAN_FORMAT,
        "status": PLAN_STATUS,
        "protocol_namespace": protocol_namespace,
        "task": task,
        "instruction_semantics_sha256": instruction_semantics_sha256,
        "event_vocab": list(EVENT_VOCAB),
        "event_contract_sha256": _require_sha(
            event_contract_sha256, "event contract SHA"
        ),
        "object_state_contract_sha256": _require_sha(
            object_state_contract_sha256, "object-state contract SHA"
        ),
        "dataset_binding": {
            "format": _require_string(dataset_format, "dataset format"),
            "file_sha256": _require_sha(dataset_file_sha256, "dataset file SHA"),
            "logical_sha256": _require_sha(
                dataset_logical_sha256, "dataset logical SHA"
            ),
            "partition_sha256": _require_sha(partition_sha256, "partition SHA"),
            "lane": _require_string(lane, "dataset lane"),
            "lane_group_count": _require_int(
                lane_group_count, "lane group count", minimum=FOLD_COUNT
            ),
            "lane_execution_group_set_sha256": _require_sha(
                lane_execution_group_set_sha256,
                "lane execution-group set SHA",
            ),
        },
        "identity_contract": {
            "semantic_reset_identity_format": RESET_IDENTITY_FORMAT,
            "reset_identity_contract_sha256": reset_identity_contract_sha256,
            "logical_group_identity_semantics": "dataset_local_immutable_group_id",
            "execution_group_identity_format": EXECUTION_IDENTITY_FORMAT,
            "sample_identity_format": SAMPLE_IDENTITY_FORMAT,
            "body_actor_excluded_from_semantic_reset_identity": True,
            "body_actor_bound_orthogonally_in_execution_group_identity": True,
        },
        "execution_contexts": contexts,
        "variant_scope": variant_scope,
        "variants": decoded_variants,
        "variant_selection_contract": {
            "reference_variant_id": reference,
            "candidate_variant_id": candidate,
            "eligible_heads": list(_variant_allowed_heads(variant_scope)),
            "pure_clock_non_duration_outputs_must_be_bit_exact": (
                variant_scope == PURE_CLOCK_SCOPE
            ),
            "full_body_adapter_must_not_be_described_as_clock_only": True,
        },
        "statistical_contract": {
            "fold_count": FOLD_COUNT,
            "fold_assignment": "explicit_execution_group_owner_five_fold_v1",
            "outer_oof_requirement": "every_execution_group_heldout_exactly_once",
            "metric_weighting": "equal_execution_group_then_equal_rows_within_group",
            "paired_variant_estimand": "identical_sample_group_order_and_applicability_mask",
            "support_minimum_per_category": dict(MINIMUM_SUPPORT_PER_CATEGORY),
            "support_required_globally_and_for_every_body_actor_context": True,
            "bootstrap_unit": (
                "semantic_reset_cluster_if_reused_else_execution_group_v1"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
            "bootstrap_draws_sha256": _require_sha(
                bootstrap_draws_sha256, "bootstrap draws SHA"
            ),
            "shared_draws_across_heads_and_variants": True,
            "paired_gain_lcb_must_be_strictly_greater_than": (
                MINIMUM_PAIRED_GAIN_LCB
            ),
            "harmful_rate_ucb_must_be_at_most": MAXIMUM_HARMFUL_RATE_UCB,
            "uncertainty_gate_required": True,
            "baseline_performance_gate_required": True,
            "multiple_head_selection_control": (
                "shared_draws_simultaneous_or_bonferroni_one_sided_alpha_0.05_over_6"
            ),
        },
        "fallback_contract": {
            "candidate_requires_reference_support_performance_uncertainty": True,
            "candidate_requires_strict_paired_gain_lcb": True,
            "candidate_requires_harmful_rate_ucb": True,
            "failed_candidate_falls_back_to_body_agnostic_if_reference_passes": True,
            "failed_reference_or_support_disables_head": True,
            "disabled_head_falls_back_to_actor_baseline": True,
        },
        "rank_route_contract": {
            "source_contract_rank_score_is_a_six_head_route": False,
            "prior_rank_contract_sha_reusable_for_body_mode_selection": False,
            "independent_whole_provider_oof_required": True,
            "independent_whole_provider_oof_passed": False,
            "rank_route_authorized": False,
            "unauthorized_rank_route_fallback": "actor_baseline",
        },
        "capability": {
            "filesystem_or_hdf_access": False,
            "training_or_collection": False,
            "signature_or_issuer_verification": False,
            "canonical_sha_is_signature": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "performance_claim_authorized": False,
        },
    }
    result = _signed_document(base, "plan_sha256")
    validate_selection_plan(result)
    return result


def validate_selection_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "format",
        "status",
        "protocol_namespace",
        "task",
        "instruction_semantics_sha256",
        "event_vocab",
        "event_contract_sha256",
        "object_state_contract_sha256",
        "dataset_binding",
        "identity_contract",
        "execution_contexts",
        "variant_scope",
        "variants",
        "variant_selection_contract",
        "statistical_contract",
        "fallback_contract",
        "rank_route_contract",
        "capability",
    }
    logical = _verify_document(
        value,
        field="plan_sha256",
        expected_fields=fields,
        role="selection plan",
    )
    task = _require_string(value.get("task"), "plan task")
    if (
        value.get("format") != PLAN_FORMAT
        or value.get("status") != PLAN_STATUS
        or value.get("event_vocab") != list(EVENT_VOCAB)
    ):
        raise BodyModeSelectionError("selection plan format or event vocabulary changed")
    _require_string(value.get("protocol_namespace"), "plan protocol namespace")
    _require_sha(value.get("instruction_semantics_sha256"), "plan instruction SHA")
    _require_sha(value.get("event_contract_sha256"), "plan event contract SHA")
    _require_sha(
        value.get("object_state_contract_sha256"), "plan object-state contract SHA"
    )
    identity = value.get("identity_contract")
    if identity != {
        "semantic_reset_identity_format": RESET_IDENTITY_FORMAT,
        "reset_identity_contract_sha256": identity.get("reset_identity_contract_sha256")
        if isinstance(identity, Mapping)
        else None,
        "logical_group_identity_semantics": "dataset_local_immutable_group_id",
        "execution_group_identity_format": EXECUTION_IDENTITY_FORMAT,
        "sample_identity_format": SAMPLE_IDENTITY_FORMAT,
        "body_actor_excluded_from_semantic_reset_identity": True,
        "body_actor_bound_orthogonally_in_execution_group_identity": True,
    }:
        raise BodyModeSelectionError("three-layer identity contract changed")
    _require_sha(identity["reset_identity_contract_sha256"], "reset identity SHA")
    dataset = value.get("dataset_binding")
    dataset_fields = {
        "format",
        "file_sha256",
        "logical_sha256",
        "partition_sha256",
        "lane",
        "lane_group_count",
        "lane_execution_group_set_sha256",
    }
    if not isinstance(dataset, Mapping) or set(dataset) != dataset_fields:
        raise BodyModeSelectionError("plan dataset binding fields changed")
    _require_string(dataset["format"], "dataset format")
    _require_string(dataset["lane"], "dataset lane")
    _require_int(dataset["lane_group_count"], "lane group count", minimum=FOLD_COUNT)
    for name in (
        "file_sha256",
        "logical_sha256",
        "partition_sha256",
        "lane_execution_group_set_sha256",
    ):
        _require_sha(dataset[name], f"dataset {name}")
    contexts = _normalize_contexts(value.get("execution_contexts"), task)
    if contexts != value["execution_contexts"]:
        raise BodyModeSelectionError("execution context ordering changed")
    scope = value.get("variant_scope")
    if scope not in VARIANT_SCOPES:
        raise BodyModeSelectionError("variant scope changed")
    variants = _validate_variant_rows(str(scope), value.get("variants"))
    if variants != value["variants"]:
        raise BodyModeSelectionError("variant ordering changed")
    reference, candidate = _variant_names(str(scope))
    if value.get("variant_selection_contract") != {
        "reference_variant_id": reference,
        "candidate_variant_id": candidate,
        "eligible_heads": list(_variant_allowed_heads(str(scope))),
        "pure_clock_non_duration_outputs_must_be_bit_exact": (
            scope == PURE_CLOCK_SCOPE
        ),
        "full_body_adapter_must_not_be_described_as_clock_only": True,
    }:
        raise BodyModeSelectionError("variant-selection contract changed")
    expected_statistics = {
        "fold_count": FOLD_COUNT,
        "fold_assignment": "explicit_execution_group_owner_five_fold_v1",
        "outer_oof_requirement": "every_execution_group_heldout_exactly_once",
        "metric_weighting": "equal_execution_group_then_equal_rows_within_group",
        "paired_variant_estimand": "identical_sample_group_order_and_applicability_mask",
        "support_minimum_per_category": dict(MINIMUM_SUPPORT_PER_CATEGORY),
        "support_required_globally_and_for_every_body_actor_context": True,
        "bootstrap_unit": "semantic_reset_cluster_if_reused_else_execution_group_v1",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_draws_sha256": value.get("statistical_contract", {}).get(
            "bootstrap_draws_sha256"
        )
        if isinstance(value.get("statistical_contract"), Mapping)
        else None,
        "shared_draws_across_heads_and_variants": True,
        "paired_gain_lcb_must_be_strictly_greater_than": MINIMUM_PAIRED_GAIN_LCB,
        "harmful_rate_ucb_must_be_at_most": MAXIMUM_HARMFUL_RATE_UCB,
        "uncertainty_gate_required": True,
        "baseline_performance_gate_required": True,
        "multiple_head_selection_control": (
            "shared_draws_simultaneous_or_bonferroni_one_sided_alpha_0.05_over_6"
        ),
    }
    if value.get("statistical_contract") != expected_statistics:
        raise BodyModeSelectionError("statistical contract changed")
    _require_sha(expected_statistics["bootstrap_draws_sha256"], "bootstrap draws SHA")
    if value.get("fallback_contract") != {
        "candidate_requires_reference_support_performance_uncertainty": True,
        "candidate_requires_strict_paired_gain_lcb": True,
        "candidate_requires_harmful_rate_ucb": True,
        "failed_candidate_falls_back_to_body_agnostic_if_reference_passes": True,
        "failed_reference_or_support_disables_head": True,
        "disabled_head_falls_back_to_actor_baseline": True,
    }:
        raise BodyModeSelectionError("fallback contract changed")
    if value.get("rank_route_contract") != {
        "source_contract_rank_score_is_a_six_head_route": False,
        "prior_rank_contract_sha_reusable_for_body_mode_selection": False,
        "independent_whole_provider_oof_required": True,
        "independent_whole_provider_oof_passed": False,
        "rank_route_authorized": False,
        "unauthorized_rank_route_fallback": "actor_baseline",
    }:
        raise BodyModeSelectionError("rank route must remain isolated and unauthorized")
    if value.get("capability") != {
        "filesystem_or_hdf_access": False,
        "training_or_collection": False,
        "signature_or_issuer_verification": False,
        "canonical_sha_is_signature": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "performance_claim_authorized": False,
    }:
        raise BodyModeSelectionError("plan capability must remain no-promotion")
    return {
        "plan_sha256": logical,
        "task": task,
        "contexts": contexts,
        "variant_scope": str(scope),
        "variants": variants,
        "reference_variant_id": reference,
        "candidate_variant_id": candidate,
        "eligible_heads": _variant_allowed_heads(str(scope)),
        "dataset_binding": dict(dataset),
        "statistical_contract": dict(expected_statistics),
    }


def execution_group_id(identity: Mapping[str, Any]) -> str:
    fields = {
        "namespace",
        "task",
        "instruction_semantics_sha256",
        "body_id",
        "body_contract_sha256",
        "policy_id",
        "actor_id",
        "actor_contract_sha256",
        "requested_seed",
        "resolved_seed",
        "semantic_reset_cluster_id",
    }
    if not isinstance(identity, Mapping) or set(identity) != fields:
        raise BodyModeSelectionError("execution identity input fields changed")
    base = {"format": EXECUTION_IDENTITY_FORMAT, **dict(identity)}
    for name in ("namespace", "task", "body_id", "policy_id", "actor_id"):
        _require_string(base[name], f"execution identity {name}")
    for name in (
        "instruction_semantics_sha256",
        "body_contract_sha256",
        "actor_contract_sha256",
        "semantic_reset_cluster_id",
    ):
        _require_sha(base[name], f"execution identity {name}")
    _require_int(base["requested_seed"], "requested seed", minimum=0)
    _require_int(base["resolved_seed"], "resolved seed", minimum=0)
    return canonical_sha256(base)


def execution_sample_id(execution_id: str, group_row_ordinal: int) -> str:
    _require_sha(execution_id, "execution group id")
    _require_int(group_row_ordinal, "group row ordinal", minimum=0)
    return canonical_sha256(
        {
            "format": SAMPLE_IDENTITY_FORMAT,
            "execution_group_id": execution_id,
            "group_row_ordinal": group_row_ordinal,
        }
    )


def _row_context_sha(row: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "task": row["task"],
            "body_id": row["body_id"],
            "body_contract_sha256": row["body_contract_sha256"],
            "policy_id": row["policy_id"],
            "actor_id": row["actor_id"],
            "actor_contract_sha256": row["actor_contract_sha256"],
        }
    )


def _validate_identity_rows(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], audit: Mapping[str, Any]
) -> list[dict[str, Any]]:
    fields = {
        "sample_id",
        "logical_group_id",
        "semantic_reset_cluster_id",
        "execution_group_id",
        "group_row_ordinal",
        "namespace",
        "task",
        "instruction_semantics_sha256",
        "requested_seed",
        "resolved_seed",
        "body_id",
        "body_contract_sha256",
        "policy_id",
        "actor_id",
        "actor_contract_sha256",
        "support_category_by_head",
    }
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise BodyModeSelectionError("evidence requires identity rows")
    context_ids = {row["execution_context_sha256"] for row in audit["contexts"]}
    decoded: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    logical_to_execution: dict[str, str] = {}
    execution_to_logical: dict[str, str] = {}
    ordinal_by_execution: dict[str, list[int]] = defaultdict(list)
    semantic_by_execution: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise BodyModeSelectionError("identity row fields changed")
        row = dict(raw)
        for name in ("sample_id", "semantic_reset_cluster_id", "execution_group_id"):
            _require_sha(row[name], f"identity row {name}")
        logical = _require_string(row["logical_group_id"], "logical group id")
        ordinal = _require_int(row["group_row_ordinal"], "group row ordinal", minimum=0)
        if row["namespace"] != plan["protocol_namespace"]:
            raise BodyModeSelectionError("identity row namespace changed")
        if row["task"] != audit["task"]:
            raise BodyModeSelectionError("identity row task changed")
        if row["instruction_semantics_sha256"] != plan["instruction_semantics_sha256"]:
            raise BodyModeSelectionError("identity row instruction semantics changed")
        if _row_context_sha(row) not in context_ids:
            raise BodyModeSelectionError("identity row body/actor context is not frozen")
        identity = {
            "namespace": row["namespace"],
            "task": row["task"],
            "instruction_semantics_sha256": row["instruction_semantics_sha256"],
            "body_id": row["body_id"],
            "body_contract_sha256": row["body_contract_sha256"],
            "policy_id": row["policy_id"],
            "actor_id": row["actor_id"],
            "actor_contract_sha256": row["actor_contract_sha256"],
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
            "semantic_reset_cluster_id": row["semantic_reset_cluster_id"],
        }
        expected_execution = execution_group_id(identity)
        if row["execution_group_id"] != expected_execution:
            raise BodyModeSelectionError("execution group identity does not recompute")
        expected_sample = execution_sample_id(expected_execution, ordinal)
        if row["sample_id"] != expected_sample or expected_sample in sample_ids:
            raise BodyModeSelectionError("sample identity changed or duplicated")
        sample_ids.add(expected_sample)
        prior_execution = logical_to_execution.setdefault(logical, expected_execution)
        prior_logical = execution_to_logical.setdefault(expected_execution, logical)
        if prior_execution != expected_execution or prior_logical != logical:
            raise BodyModeSelectionError("logical/execution group mapping is not bijective")
        semantic = str(row["semantic_reset_cluster_id"])
        if semantic_by_execution.setdefault(expected_execution, semantic) != semantic:
            raise BodyModeSelectionError("one execution group spans reset clusters")
        ordinal_by_execution[expected_execution].append(ordinal)
        categories = row["support_category_by_head"]
        if not isinstance(categories, Mapping) or set(categories) != set(HEADS):
            raise BodyModeSelectionError("support-category head inventory changed")
        normalized_categories: dict[str, str | None] = {}
        for head in HEADS:
            category = categories[head]
            if category is not None and category not in SUPPORT_CATEGORIES[head]:
                raise BodyModeSelectionError(f"{head} support category changed")
            normalized_categories[head] = category
        row["support_category_by_head"] = normalized_categories
        decoded.append(row)
    for execution, ordinals in ordinal_by_execution.items():
        if ordinals != list(range(len(ordinals))):
            raise BodyModeSelectionError(
                f"group row order is not contiguous for execution {execution}"
            )
    if decoded != sorted(
        decoded, key=lambda row: (row["execution_group_id"], row["group_row_ordinal"])
    ):
        raise BodyModeSelectionError("identity rows are not in canonical group/row order")
    groups = sorted(ordinal_by_execution)
    dataset = audit["dataset_binding"]
    if len(groups) != dataset["lane_group_count"]:
        raise BodyModeSelectionError("evidence execution-group count changed")
    if canonical_sha256(groups) != dataset["lane_execution_group_set_sha256"]:
        raise BodyModeSelectionError("evidence execution-group set changed")
    return decoded


def _validate_oof_folds(
    folds: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    fields = {
        "fold_index",
        "training_execution_group_ids",
        "heldout_execution_group_ids",
        "training_execution_group_ids_sha256",
        "heldout_execution_group_ids_sha256",
    }
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or len(folds) != FOLD_COUNT:
        raise BodyModeSelectionError("OOF requires exactly five folds")
    all_groups = sorted({str(row["execution_group_id"]) for row in rows})
    all_set = set(all_groups)
    owner: dict[str, int] = {}
    decoded = []
    for expected_index, raw in enumerate(folds):
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise BodyModeSelectionError("OOF fold fields changed")
        if _require_int(raw["fold_index"], "fold index", expected=expected_index) != expected_index:
            raise BodyModeSelectionError("fold index changed")
        train = raw["training_execution_group_ids"]
        heldout = raw["heldout_execution_group_ids"]
        if (
            not isinstance(train, list)
            or not isinstance(heldout, list)
            or train != sorted(train)
            or heldout != sorted(heldout)
            or not heldout
            or len(set(train)) != len(train)
            or len(set(heldout)) != len(heldout)
            or set(train) & set(heldout)
            or set(train) | set(heldout) != all_set
            or set(train) != all_set - set(heldout)
        ):
            raise BodyModeSelectionError("OOF train/heldout membership changed")
        if raw["training_execution_group_ids_sha256"] != canonical_sha256(train):
            raise BodyModeSelectionError("OOF training group SHA changed")
        if raw["heldout_execution_group_ids_sha256"] != canonical_sha256(heldout):
            raise BodyModeSelectionError("OOF heldout group SHA changed")
        for group in heldout:
            if group in owner:
                raise BodyModeSelectionError("OOF group held out more than once")
            owner[group] = expected_index
        decoded.append(dict(raw))
    if set(owner) != all_set:
        raise BodyModeSelectionError("OOF does not hold every group out exactly once")
    semantics: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        semantics[str(row["semantic_reset_cluster_id"])].add(
            str(row["execution_group_id"])
        )
    for groups in semantics.values():
        if len({owner[group] for group in groups}) != 1:
            raise BodyModeSelectionError("one semantic reset cluster crosses OOF folds")
    return decoded, canonical_sha256(decoded)


def _support_summary(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    context_ids = [str(row["execution_context_sha256"]) for row in contexts]
    result: dict[str, Any] = {"heads": {}}
    for head in HEADS:
        categories = SUPPORT_CATEGORIES[head]
        minimum = MINIMUM_SUPPORT_PER_CATEGORY[head]
        global_groups = {category: set() for category in categories}
        context_groups = {
            context: {category: set() for category in categories}
            for context in context_ids
        }
        for row in rows:
            category = row["support_category_by_head"][head]
            if category is None:
                continue
            group = str(row["execution_group_id"])
            context = _row_context_sha(row)
            global_groups[category].add(group)
            context_groups[context][category].add(group)
        global_counts = {
            category: len(global_groups[category]) for category in categories
        }
        by_context = {
            context: {
                category: len(context_groups[context][category])
                for category in categories
            }
            for context in context_ids
        }
        passed = all(count >= minimum for count in global_counts.values()) and all(
            count >= minimum
            for counts in by_context.values()
            for count in counts.values()
        )
        result["heads"][head] = {
            "categories": list(categories),
            "minimum_required_per_category": minimum,
            "independent_execution_group_counts": global_counts,
            "by_body_actor_context": by_context,
            "support_gate_passed": passed,
        }
    result["support_sha256"] = canonical_sha256(result)
    return result


def _bootstrap_cluster_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    semantic_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        semantic_groups[str(row["semantic_reset_cluster_id"])].add(
            str(row["execution_group_id"])
        )
    keys = []
    for semantic, executions in semantic_groups.items():
        if len(executions) > 1:
            keys.append(f"semantic_reset:{semantic}")
        else:
            keys.append(f"execution_group:{next(iter(executions))}")
    return sorted(keys)


def _expected_masks(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    return {
        head: [row["support_category_by_head"][head] is not None for row in rows]
        for head in HEADS
    }


def _validate_variant_views(
    values: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fields = {
        "variant_id",
        "member_checkpoint_file_sha256",
        "sample_id",
        "logical_group_id",
        "execution_group_id",
        "sample_order_sha256",
        "applicable_masks",
        "applicable_mask_set_sha256",
        "head_output_sha256",
    }
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise BodyModeSelectionError("evidence requires both variant views")
    expected_samples = [str(row["sample_id"]) for row in rows]
    expected_logical = [str(row["logical_group_id"]) for row in rows]
    expected_execution = [str(row["execution_group_id"]) for row in rows]
    expected_masks = _expected_masks(rows)
    expected_order_sha = canonical_sha256(
        {
            "sample_id": expected_samples,
            "logical_group_id": expected_logical,
            "execution_group_id": expected_execution,
        }
    )
    plan_variants = {row["variant_id"]: row for row in audit["variants"]}
    decoded: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise BodyModeSelectionError("variant evidence view fields changed")
        name = raw.get("variant_id")
        if name not in plan_variants or name in decoded:
            raise BodyModeSelectionError("variant evidence coverage changed")
        if raw["member_checkpoint_file_sha256"] != plan_variants[name][
            "member_checkpoint_file_sha256"
        ]:
            raise BodyModeSelectionError("variant evidence checkpoint binding changed")
        if (
            raw["sample_id"] != expected_samples
            or raw["logical_group_id"] != expected_logical
            or raw["execution_group_id"] != expected_execution
            or raw["sample_order_sha256"] != expected_order_sha
        ):
            raise BodyModeSelectionError(
                "variants do not share identical sample/group/order"
            )
        masks = raw["applicable_masks"]
        if not isinstance(masks, Mapping) or set(masks) != set(HEADS):
            raise BodyModeSelectionError("variant applicability head inventory changed")
        normalized_masks: dict[str, list[bool]] = {}
        for head in HEADS:
            mask = masks[head]
            if (
                not isinstance(mask, list)
                or len(mask) != len(rows)
                or any(type(item) is not bool for item in mask)
                or mask != expected_masks[head]
            ):
                raise BodyModeSelectionError(
                    "variants do not share identical applicable masks"
                )
            normalized_masks[head] = list(mask)
        if raw["applicable_mask_set_sha256"] != canonical_sha256(normalized_masks):
            raise BodyModeSelectionError("variant applicability mask SHA changed")
        outputs = raw["head_output_sha256"]
        if (
            not isinstance(outputs, Mapping)
            or set(outputs) != set(HEADS)
            or any(not is_sha256(outputs[head]) for head in HEADS)
        ):
            raise BodyModeSelectionError("variant head output commitments changed")
        decoded[str(name)] = {
            **dict(raw),
            "applicable_masks": normalized_masks,
            "head_output_sha256": dict(outputs),
        }
    reference = audit["reference_variant_id"]
    candidate = audit["candidate_variant_id"]
    if set(decoded) != {reference, candidate}:
        raise BodyModeSelectionError("variant evidence coverage is incomplete")
    if audit["variant_scope"] == PURE_CLOCK_SCOPE:
        for head in HEADS:
            if head != "duration" and decoded[reference]["head_output_sha256"][head] != decoded[candidate]["head_output_sha256"][head]:
                raise BodyModeSelectionError(
                    "pure clock ablation changed a non-duration head output"
                )
    return [decoded[reference], decoded[candidate]]


def _validate_head_evidence(
    value: Mapping[str, Any], audit: Mapping[str, Any], draws_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(HEADS):
        raise BodyModeSelectionError("head evidence inventory changed")
    variant_names = {
        audit["reference_variant_id"], audit["candidate_variant_id"]
    }
    result: dict[str, Any] = {}
    for head in HEADS:
        row = value[head]
        if not isinstance(row, Mapping) or set(row) != {
            "variant_gates", "paired_comparison"
        }:
            raise BodyModeSelectionError(f"{head} evidence fields changed")
        gates = row["variant_gates"]
        if not isinstance(gates, Mapping) or set(gates) != variant_names:
            raise BodyModeSelectionError(f"{head} variant-gate coverage changed")
        normalized_gates = {}
        for variant in sorted(variant_names):
            gate = gates[variant]
            if not isinstance(gate, Mapping) or set(gate) != {
                "baseline_performance_gate_passed",
                "uncertainty_gate_passed",
                "per_fold_support_passed",
                "oof_metric_evidence_sha256",
                "uncertainty_evidence_sha256",
                "per_fold_support_evidence_sha256",
                "deployment_calibration_parameter_sha256",
            }:
                raise BodyModeSelectionError(f"{head}/{variant} gate fields changed")
            if type(gate["baseline_performance_gate_passed"]) is not bool or type(
                gate["uncertainty_gate_passed"]
            ) is not bool or type(gate["per_fold_support_passed"]) is not bool:
                raise BodyModeSelectionError(f"{head}/{variant} gate booleans changed")
            _require_sha(
                gate["oof_metric_evidence_sha256"], f"{head}/{variant} OOF evidence"
            )
            _require_sha(
                gate["uncertainty_evidence_sha256"],
                f"{head}/{variant} uncertainty evidence",
            )
            _require_sha(
                gate["per_fold_support_evidence_sha256"],
                f"{head}/{variant} per-fold support evidence",
            )
            _require_sha(
                gate["deployment_calibration_parameter_sha256"],
                f"{head}/{variant} deployment calibration parameter",
            )
            normalized_gates[variant] = dict(gate)
        comparison = row["paired_comparison"]
        eligible = head in audit["eligible_heads"]
        if eligible:
            expected_fields = {
                "status",
                "reference_variant_id",
                "candidate_variant_id",
                "paired_gain_lcb95",
                "harmful_rate_ucb95",
                "shared_bootstrap_draws_sha256",
                "paired_unit",
                "identical_sample_group_order_and_mask",
                "selection_labels_used_only_in_oof_training_folds",
            }
            if not isinstance(comparison, Mapping) or set(comparison) != expected_fields:
                raise BodyModeSelectionError(f"{head} paired comparison fields changed")
            gain = _require_float(comparison["paired_gain_lcb95"], f"{head} gain LCB")
            harmful = _require_float(
                comparison["harmful_rate_ucb95"], f"{head} harmful-rate UCB"
            )
            if (
                comparison["status"] != "evaluated_paired_oof"
                or comparison["reference_variant_id"]
                != audit["reference_variant_id"]
                or comparison["candidate_variant_id"]
                != audit["candidate_variant_id"]
                or comparison["shared_bootstrap_draws_sha256"] != draws_sha256
                or comparison["paired_unit"]
                != "semantic_reset_cluster_if_reused_else_execution_group"
                or comparison["identical_sample_group_order_and_mask"] is not True
                or comparison["selection_labels_used_only_in_oof_training_folds"]
                is not True
                or not 0.0 <= harmful <= 1.0
            ):
                raise BodyModeSelectionError(f"{head} paired comparison changed")
            normalized_comparison = dict(comparison)
        else:
            expected = {
                "status": "not_applicable_pure_clock_non_duration",
                "reason": "clock_dependency_graph_excludes_this_head",
                "reference_variant_id": audit["reference_variant_id"],
                "candidate_variant_id": audit["candidate_variant_id"],
            }
            if comparison != expected:
                raise BodyModeSelectionError(
                    f"{head} must not perform a pure-clock variant comparison"
                )
            normalized_comparison = dict(comparison)
        result[head] = {
            "variant_gates": normalized_gates,
            "paired_comparison": normalized_comparison,
        }
    return result


def build_calibration_evidence(
    *,
    plan: Mapping[str, Any],
    identity_rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    variant_views: Sequence[Mapping[str, Any]],
    head_evidence: Mapping[str, Any],
    evidence_kind: str = "synthetic_contract_test",
) -> dict[str, Any]:
    """Build aggregate in-memory evidence; this never verifies metric truth."""

    audit = validate_selection_plan(plan)
    if evidence_kind not in {
        "synthetic_contract_test",
        "external_aggregate_unsigned",
    }:
        raise BodyModeSelectionError("evidence kind changed")
    rows = _validate_identity_rows(identity_rows, plan, audit)
    decoded_folds, folds_sha = _validate_oof_folds(folds, rows)
    decoded_views = _validate_variant_views(variant_views, rows, audit)
    support = _support_summary(rows, audit["contexts"])
    group_order = _bootstrap_cluster_order(rows)
    bootstrap = {
        "unit": audit["statistical_contract"]["bootstrap_unit"],
        "group_order": group_order,
        "group_order_sha256": canonical_sha256(group_order),
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "draws_sha256": audit["statistical_contract"]["bootstrap_draws_sha256"],
        "shared_across_heads_and_variants": True,
        "equal_cluster_weight": True,
        "rows_within_execution_group_not_iid": True,
    }
    decoded_head_evidence = _validate_head_evidence(
        head_evidence, audit, bootstrap["draws_sha256"]
    )
    base = {
        "format": EVIDENCE_FORMAT,
        "status": EVIDENCE_STATUS,
        "plan_sha256": audit["plan_sha256"],
        "evidence_kind": evidence_kind,
        "identity_rows": rows,
        "identity_rows_sha256": canonical_sha256(rows),
        "oof_folds": decoded_folds,
        "oof_folds_sha256": folds_sha,
        "variant_views": decoded_views,
        "variant_views_sha256": canonical_sha256(decoded_views),
        "head_support": support,
        "head_evidence": decoded_head_evidence,
        "bootstrap": bootstrap,
        "capability": {
            "input_is_aggregate_or_synthetic_only": True,
            "metric_truth_verified_by_this_schema": False,
            "filesystem_hdf_or_label_files_opened": 0,
            "signatures_verified": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "performance_claim_authorized": False,
        },
    }
    result = _signed_document(base, "evidence_sha256")
    validate_calibration_evidence(result, plan)
    return result


def validate_calibration_evidence(
    value: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    audit = validate_selection_plan(plan)
    fields = {
        "format",
        "status",
        "plan_sha256",
        "evidence_kind",
        "identity_rows",
        "identity_rows_sha256",
        "oof_folds",
        "oof_folds_sha256",
        "variant_views",
        "variant_views_sha256",
        "head_support",
        "head_evidence",
        "bootstrap",
        "capability",
    }
    logical = _verify_document(
        value,
        field="evidence_sha256",
        expected_fields=fields,
        role="calibration evidence",
    )
    if (
        value.get("format") != EVIDENCE_FORMAT
        or value.get("status") != EVIDENCE_STATUS
        or value.get("plan_sha256") != audit["plan_sha256"]
        or value.get("evidence_kind")
        not in {"synthetic_contract_test", "external_aggregate_unsigned"}
    ):
        raise BodyModeSelectionError("calibration evidence scope changed")
    rows = _validate_identity_rows(value.get("identity_rows"), plan, audit)
    if value.get("identity_rows_sha256") != canonical_sha256(rows):
        raise BodyModeSelectionError("identity row SHA changed")
    folds, fold_sha = _validate_oof_folds(value.get("oof_folds"), rows)
    if value.get("oof_folds_sha256") != fold_sha:
        raise BodyModeSelectionError("OOF fold-set SHA changed")
    views = _validate_variant_views(value.get("variant_views"), rows, audit)
    if value.get("variant_views_sha256") != canonical_sha256(views):
        raise BodyModeSelectionError("variant-view SHA changed")
    support = _support_summary(rows, audit["contexts"])
    if value.get("head_support") != support:
        raise BodyModeSelectionError("reported support differs from identity rows")
    expected_group_order = _bootstrap_cluster_order(rows)
    expected_bootstrap = {
        "unit": audit["statistical_contract"]["bootstrap_unit"],
        "group_order": expected_group_order,
        "group_order_sha256": canonical_sha256(expected_group_order),
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "draws_sha256": audit["statistical_contract"]["bootstrap_draws_sha256"],
        "shared_across_heads_and_variants": True,
        "equal_cluster_weight": True,
        "rows_within_execution_group_not_iid": True,
    }
    if value.get("bootstrap") != expected_bootstrap:
        raise BodyModeSelectionError("bootstrap evidence contract changed")
    head_evidence = _validate_head_evidence(
        value.get("head_evidence"), audit, expected_bootstrap["draws_sha256"]
    )
    if value.get("head_evidence") != head_evidence:
        raise BodyModeSelectionError("head evidence normalization changed")
    if value.get("capability") != {
        "input_is_aggregate_or_synthetic_only": True,
        "metric_truth_verified_by_this_schema": False,
        "filesystem_hdf_or_label_files_opened": 0,
        "signatures_verified": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "performance_claim_authorized": False,
    }:
        raise BodyModeSelectionError("evidence capability must remain no-promotion")
    return {
        "evidence_sha256": logical,
        "support": support,
        "head_evidence": head_evidence,
        "variant_scope": audit["variant_scope"],
        "reference_variant_id": audit["reference_variant_id"],
        "candidate_variant_id": audit["candidate_variant_id"],
        "eligible_heads": audit["eligible_heads"],
    }


def _variant_gate_passed(gate: Mapping[str, Any], support: bool) -> bool:
    return bool(
        support
        and gate["baseline_performance_gate_passed"] is True
        and gate["uncertainty_gate_passed"] is True
        and gate["per_fold_support_passed"] is True
    )


def _head_decision(
    head: str, audit: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    support = bool(evidence["support"]["heads"][head]["support_gate_passed"])
    row = evidence["head_evidence"][head]
    reference = audit["reference_variant_id"]
    candidate = audit["candidate_variant_id"]
    reference_passed = _variant_gate_passed(row["variant_gates"][reference], support)
    candidate_passed = _variant_gate_passed(row["variant_gates"][candidate], support)
    base: dict[str, Any] = {
        "head": head,
        "support_gate_passed": support,
        "reference_variant_gate_passed": reference_passed,
        "candidate_variant_gate_passed": candidate_passed,
    }
    if not support:
        return {
            **base,
            "selection_status": "head_disabled",
            "selected_variant_id": None,
            "reason": "insufficient_independent_group_support",
            "paired_gain_lcb95": None,
            "harmful_rate_ucb95": None,
            "selected_deployment_calibration_parameter_sha256": None,
            "actor_baseline_fallback_required": True,
        }
    if not reference_passed:
        return {
            **base,
            "selection_status": "head_disabled",
            "selected_variant_id": None,
            "reason": "body_agnostic_reference_performance_or_uncertainty_gate_failed",
            "paired_gain_lcb95": None,
            "harmful_rate_ucb95": None,
            "selected_deployment_calibration_parameter_sha256": None,
            "actor_baseline_fallback_required": True,
        }
    if head not in audit["eligible_heads"]:
        return {
            **base,
            "selection_status": "fixed_reference_pure_clock_invariant",
            "selected_variant_id": reference,
            "reason": "clock_dependency_graph_excludes_this_head",
            "paired_gain_lcb95": None,
            "harmful_rate_ucb95": None,
            "selected_deployment_calibration_parameter_sha256": row[
                "variant_gates"
            ][reference]["deployment_calibration_parameter_sha256"],
            "actor_baseline_fallback_required": False,
        }
    comparison = row["paired_comparison"]
    gain = float(comparison["paired_gain_lcb95"])
    harmful = float(comparison["harmful_rate_ucb95"])
    if (
        candidate_passed
        and gain > MINIMUM_PAIRED_GAIN_LCB
        and harmful <= MAXIMUM_HARMFUL_RATE_UCB
    ):
        return {
            **base,
            "selection_status": "selected_body_conditioned_candidate",
            "selected_variant_id": candidate,
            "reason": "support_performance_uncertainty_gain_and_harm_gates_passed",
            "paired_gain_lcb95": gain,
            "harmful_rate_ucb95": harmful,
            "selected_deployment_calibration_parameter_sha256": row[
                "variant_gates"
            ][candidate]["deployment_calibration_parameter_sha256"],
            "actor_baseline_fallback_required": False,
        }
    if not candidate_passed:
        reason = "candidate_performance_or_uncertainty_gate_failed"
    elif gain <= MINIMUM_PAIRED_GAIN_LCB:
        reason = "paired_gain_lcb_not_strictly_positive"
    else:
        reason = "harmful_rate_ucb_exceeds_limit"
    return {
        **base,
        "selection_status": "fallback_body_agnostic_reference",
        "selected_variant_id": reference,
        "reason": reason,
        "paired_gain_lcb95": gain,
        "harmful_rate_ucb95": harmful,
        "selected_deployment_calibration_parameter_sha256": row["variant_gates"][
            reference
        ]["deployment_calibration_parameter_sha256"],
        "actor_baseline_fallback_required": False,
    }


def build_selection_receipt(
    plan: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Deterministically select each head without creating any authority."""

    plan_audit = validate_selection_plan(plan)
    evidence_audit = validate_calibration_evidence(evidence, plan)
    decisions = {
        head: _head_decision(head, plan_audit, evidence_audit) for head in HEADS
    }
    all_enabled = all(
        row["selection_status"] != "head_disabled" for row in decisions.values()
    )
    base = {
        "format": RECEIPT_FORMAT,
        "status": RECEIPT_STATUS,
        "plan_sha256": plan_audit["plan_sha256"],
        "evidence_sha256": evidence_audit["evidence_sha256"],
        "variant_scope": plan_audit["variant_scope"],
        "head_decisions": decisions,
        "all_six_heads_enabled": all_enabled,
        "system_fallback": (
            "actor_baseline_required_rank_route_not_authorized"
            if all_enabled
            else "actor_baseline_required_rank_route_not_authorized_and_heads_disabled"
        ),
        "rank_route_contract": {
            "source_contract_rank_score_is_a_six_head_route": False,
            "prior_rank_contract_sha_reusable": False,
            "independent_whole_provider_oof_passed": False,
            "rank_route_authorized": False,
            "rank_action_selection_must_fallback_to_actor_baseline": True,
        },
        "capability": {
            "deterministic_schema_decision_only": True,
            "metric_truth_recomputed_by_this_schema": False,
            "selected_variant_ids_are_provisional_non_executable": True,
            "runtime_route_exported": False,
            "signature_or_issuer_verification": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "performance_claim_authorized": False,
        },
    }
    return _signed_document(base, "receipt_sha256")


def validate_selection_receipt(
    value: Mapping[str, Any], plan: Mapping[str, Any], evidence: Mapping[str, Any]
) -> str:
    fields = {
        "format",
        "status",
        "plan_sha256",
        "evidence_sha256",
        "variant_scope",
        "head_decisions",
        "all_six_heads_enabled",
        "system_fallback",
        "rank_route_contract",
        "capability",
    }
    logical = _verify_document(
        value,
        field="receipt_sha256",
        expected_fields=fields,
        role="selection receipt",
    )
    expected = build_selection_receipt(plan, evidence)
    if dict(value) != expected:
        raise BodyModeSelectionError(
            "selection receipt differs from deterministic recomputation"
        )
    if (
        value.get("format") != RECEIPT_FORMAT
        or value.get("status") != RECEIPT_STATUS
        or value.get("capability", {}).get("promotion_authorized") is not False
    ):
        raise BodyModeSelectionError("selection receipt scope changed")
    return logical


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "BodyModeSelectionError",
    "EVENT_VOCAB",
    "EVIDENCE_FORMAT",
    "FOLD_COUNT",
    "FULL_BODY_ADAPTER_SCOPE",
    "FULL_CANDIDATE_VARIANT",
    "FULL_REFERENCE_VARIANT",
    "HEADS",
    "MEMBER_COUNT",
    "PLAN_FORMAT",
    "PURE_CANDIDATE_VARIANT",
    "PURE_CLOCK_SCOPE",
    "PURE_REFERENCE_VARIANT",
    "RECEIPT_FORMAT",
    "build_calibration_evidence",
    "build_selection_plan",
    "build_selection_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "execution_group_id",
    "execution_sample_id",
    "is_sha256",
    "validate_calibration_evidence",
    "validate_selection_plan",
    "validate_selection_receipt",
]
