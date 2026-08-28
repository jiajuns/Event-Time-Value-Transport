#!/usr/bin/env python3
"""Development-only routing of detached predictions from two providers.

This module is deliberately an in-memory boundary.  It does not load models,
calibrators, receipts, or datasets, and it grants no deployment or promotion
authority.  Trusted callers must provide the expected content addresses for
the provider set, the already-verified route receipt, and the provider-specific
calibration set.  Routing is only for two providers evaluated on the exact same
ordered development batch.  It is not an inference router for unseen samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Torch is optional for NumPy-only callers.
    import torch
except ImportError:  # pragma: no cover - exercised only in NumPy-only installs.
    torch = None  # type: ignore[assignment]


PROVIDER_SET_FORMAT = "etsf_smolvla_piper_dual_provider_set_v1"
PROVIDER_SET_STATUS = "frozen_development_only_no_promotion_authority"
ROUTE_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_dual_provider_raw_recomputed_route_receipt_v1"
)
ROUTE_RECEIPT_STATUS = (
    "verified_raw_recomputed_development_route_no_promotion_authority"
)
PROVISIONAL_SELECTION_RECEIPT_FORMAT = (
    "etsf_per_head_body_mode_selection_receipt_v1"
)
ROUTED_PREDICTION_FORMAT = "etsf_smolvla_piper_dual_provider_routed_prediction_v1"
ROUTED_PREDICTION_STATUS = "detached_development_only_no_promotion_authority"

HEAD_FIELDS: dict[str, tuple[str, ...]] = {
    "post_event": ("post_event_logits",),
    "next_event": ("next_event_logits",),
    "duration": ("duration_log_mean", "duration_log_scale"),
    "success": ("success_logit",),
    "recovery": ("recovery_logit",),
    "object_effect": ("object_mean", "object_log_scale"),
}
HEADS = tuple(HEAD_FIELDS)
PREDICTION_FIELDS = tuple(
    field for head in HEADS for field in HEAD_FIELDS[head]
)
RANK_FIELD = "source_contract_rank_score"
REFERENCE_PROVIDER_ID = "body_agnostic_adapter"
CANDIDATE_PROVIDER_ID = "body_conditioned_adapter"
CANONICAL_PROVIDER_IDS = frozenset(
    {REFERENCE_PROVIDER_ID, CANDIDATE_PROVIDER_ID}
)
FULL_BODY_ADAPTER_SCOPE = "full_body_adapter_ablation"
MINIMUM_PAIRED_GAIN_LCB = 0.0
MAXIMUM_HARMFUL_RATE_UCB = 0.10
MEMBER_COUNT = 5
EVENT_CLASS_COUNT = 5
SHA_CHARS = frozenset("0123456789abcdef")

_LEGACY_RANK_SHA_KEYS = frozenset(
    {
        "rank_contract_sha256",
        "root_group_ranker_sha256",
        "single_provider_rank_sha256",
        "single_provider_rank_contract_sha256",
        "source_contract_rank_score_sha256",
        "source_contract_rank_sha256",
    }
)


class DualProviderPredictionRouterError(RuntimeError):
    """A provider, receipt, calibration, tensor, or routing invariant failed."""


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def _require_sha(value: Any, role: str) -> str:
    if not _is_sha256(value):
        raise DualProviderPredictionRouterError(
            f"{role} must be an exact lowercase SHA-256"
        )
    return str(value)


def _require_string(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise DualProviderPredictionRouterError(
            f"{role} must be a non-empty canonical string"
        )
    return value


def _require_finite_float(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DualProviderPredictionRouterError(f"{role} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DualProviderPredictionRouterError(f"{role} must be finite numeric")
    return result


def _reject_legacy_rank_sha(value: Mapping[str, Any], role: str) -> None:
    rejected = sorted(set(value) & _LEGACY_RANK_SHA_KEYS)
    if rejected:
        raise DualProviderPredictionRouterError(
            f"{role} contains forbidden legacy single-provider rank SHA: "
            + ", ".join(rejected)
        )


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in base:
        raise DualProviderPredictionRouterError(f"{field} exists before hashing")
    logical = dict(base)
    return {**logical, field: canonical_sha256(logical)}


def _verify_hash(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    expected_fields: set[str],
    role: str,
) -> str:
    if not isinstance(value, Mapping):
        raise DualProviderPredictionRouterError(f"{role} must be a mapping")
    _reject_legacy_rank_sha(value, role)
    if set(value) != expected_fields | {digest_field}:
        raise DualProviderPredictionRouterError(f"{role} fields changed")
    digest = _require_sha(value[digest_field], f"{role} SHA")
    unsigned = {key: child for key, child in value.items() if key != digest_field}
    if canonical_sha256(unsigned) != digest:
        raise DualProviderPredictionRouterError(f"{role} canonical SHA mismatch")
    return digest


def _normalize_head_calibrations(value: Any, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(HEADS):
        raise DualProviderPredictionRouterError(
            f"{role} must bind exactly the six prediction heads"
        )
    return {
        head: _require_sha(value[head], f"{role} {head} calibration parameter")
        for head in HEADS
    }


def _normalize_provider_descriptor(value: Any) -> dict[str, Any]:
    fields = {
        "provider_id",
        "provider_artifact_sha256",
        "calibration_bundle_sha256",
        "head_calibration_parameter_sha256",
    }
    if not isinstance(value, Mapping):
        raise DualProviderPredictionRouterError("provider descriptor must be a mapping")
    _reject_legacy_rank_sha(value, "provider descriptor")
    if set(value) != fields:
        raise DualProviderPredictionRouterError("provider descriptor fields changed")
    return {
        "provider_id": _require_string(value["provider_id"], "provider id"),
        "provider_artifact_sha256": _require_sha(
            value["provider_artifact_sha256"], "provider artifact SHA"
        ),
        "calibration_bundle_sha256": _require_sha(
            value["calibration_bundle_sha256"], "provider calibration bundle SHA"
        ),
        "head_calibration_parameter_sha256": _normalize_head_calibrations(
            value["head_calibration_parameter_sha256"], "provider"
        ),
    }


def build_provider_set(
    *,
    route_receipt_sha256: str,
    sample_identity_order_sha256: str,
    applicability_mask_set_sha256: str,
    providers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a content-addressed two-provider development contract."""

    route_sha = _require_sha(route_receipt_sha256, "route receipt SHA")
    sample_order_sha = _require_sha(
        sample_identity_order_sha256, "sample identity order SHA"
    )
    mask_set_sha = _require_sha(
        applicability_mask_set_sha256, "applicability mask-set SHA"
    )
    if (
        not isinstance(providers, Sequence)
        or isinstance(providers, (str, bytes))
        or len(providers) != 2
    ):
        raise DualProviderPredictionRouterError("exactly two providers are required")
    decoded = [_normalize_provider_descriptor(row) for row in providers]
    if {row["provider_id"] for row in decoded} != CANONICAL_PROVIDER_IDS:
        raise DualProviderPredictionRouterError(
            "provider ids must be the canonical body-agnostic/body-conditioned pair"
        )
    decoded.sort(key=lambda row: row["provider_id"])
    calibration_binding = [
        {
            "provider_id": row["provider_id"],
            "provider_artifact_sha256": row["provider_artifact_sha256"],
            "calibration_bundle_sha256": row["calibration_bundle_sha256"],
            "head_calibration_parameter_sha256": row[
                "head_calibration_parameter_sha256"
            ],
        }
        for row in decoded
    ]
    base = {
        "format": PROVIDER_SET_FORMAT,
        "status": PROVIDER_SET_STATUS,
        "route_receipt_sha256": route_sha,
        "sample_identity_order_sha256": sample_order_sha,
        "applicability_mask_set_sha256": mask_set_sha,
        "providers": decoded,
        "calibration_set_sha256": canonical_sha256(calibration_binding),
        "prediction_fields": list(PREDICTION_FIELDS),
        "rank_contract": {
            "source_contract_rank_score_is_a_six_head_route": False,
            "provider_specific_rank_scores_accepted": False,
            "legacy_single_provider_rank_sha_accepted": False,
            "fallback": "actor_baseline_or_explicit_rejection",
        },
        "capability": {
            "detached_in_memory_routing_only": True,
            "filesystem_or_model_loading": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    result = _signed(base, "provider_set_sha256")
    validate_provider_set(result)
    return result


def validate_provider_set(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "format",
        "status",
        "route_receipt_sha256",
        "sample_identity_order_sha256",
        "applicability_mask_set_sha256",
        "providers",
        "calibration_set_sha256",
        "prediction_fields",
        "rank_contract",
        "capability",
    }
    digest = _verify_hash(
        value,
        digest_field="provider_set_sha256",
        expected_fields=fields,
        role="provider set",
    )
    if (
        value["format"] != PROVIDER_SET_FORMAT
        or value["status"] != PROVIDER_SET_STATUS
        or value["prediction_fields"] != list(PREDICTION_FIELDS)
    ):
        raise DualProviderPredictionRouterError("provider-set scope changed")
    route_sha = _require_sha(value["route_receipt_sha256"], "route receipt SHA")
    sample_order_sha = _require_sha(
        value["sample_identity_order_sha256"], "sample identity order SHA"
    )
    mask_set_sha = _require_sha(
        value["applicability_mask_set_sha256"], "applicability mask-set SHA"
    )
    providers = value["providers"]
    if not isinstance(providers, list) or len(providers) != 2:
        raise DualProviderPredictionRouterError("provider set must contain two providers")
    decoded = [_normalize_provider_descriptor(row) for row in providers]
    if (
        {row["provider_id"] for row in decoded} != CANONICAL_PROVIDER_IDS
        or decoded != sorted(decoded, key=lambda row: row["provider_id"])
    ):
        raise DualProviderPredictionRouterError(
            "provider ids must be the canonical pair in canonical order"
        )
    calibration_binding = [
        {
            "provider_id": row["provider_id"],
            "provider_artifact_sha256": row["provider_artifact_sha256"],
            "calibration_bundle_sha256": row["calibration_bundle_sha256"],
            "head_calibration_parameter_sha256": row[
                "head_calibration_parameter_sha256"
            ],
        }
        for row in decoded
    ]
    calibration_sha = _require_sha(
        value["calibration_set_sha256"], "calibration-set SHA"
    )
    if calibration_sha != canonical_sha256(calibration_binding):
        raise DualProviderPredictionRouterError("calibration-set SHA mismatch")
    if value["rank_contract"] != {
        "source_contract_rank_score_is_a_six_head_route": False,
        "provider_specific_rank_scores_accepted": False,
        "legacy_single_provider_rank_sha_accepted": False,
        "fallback": "actor_baseline_or_explicit_rejection",
    }:
        raise DualProviderPredictionRouterError("rank route must remain rejected")
    if value["capability"] != {
        "detached_in_memory_routing_only": True,
        "filesystem_or_model_loading": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }:
        raise DualProviderPredictionRouterError(
            "provider set must remain development-only and no-promotion"
        )
    return {
        "provider_set_sha256": digest,
        "route_receipt_sha256": route_sha,
        "sample_identity_order_sha256": sample_order_sha,
        "applicability_mask_set_sha256": mask_set_sha,
        "calibration_set_sha256": calibration_sha,
        "providers": {row["provider_id"]: row for row in decoded},
    }


def _validate_route_receipt(
    value: Mapping[str, Any], expected_sha256: str
) -> dict[str, Any]:
    if (
        isinstance(value, Mapping)
        and value.get("format") == PROVISIONAL_SELECTION_RECEIPT_FORMAT
    ):
        raise DualProviderPredictionRouterError(
            "provisional non-executable selection receipt cannot authorize routing"
        )
    fields = {
        "format",
        "status",
        "plan_sha256",
        "evidence_sha256",
        "variant_scope",
        "sample_identity_order_sha256",
        "applicability_mask_set_sha256",
        "head_decisions",
        "all_six_heads_enabled",
        "system_fallback",
        "rank_route_contract",
        "capability",
    }
    digest = _verify_hash(
        value,
        digest_field="receipt_sha256",
        expected_fields=fields,
        role="route receipt",
    )
    if digest != _require_sha(expected_sha256, "expected verified route receipt SHA"):
        raise DualProviderPredictionRouterError(
            "route receipt differs from the trusted verified SHA"
        )
    if value["format"] != ROUTE_RECEIPT_FORMAT or value["status"] != ROUTE_RECEIPT_STATUS:
        raise DualProviderPredictionRouterError("route receipt scope changed")
    _require_sha(value["plan_sha256"], "route plan SHA")
    _require_sha(value["evidence_sha256"], "route evidence SHA")
    if value["variant_scope"] != FULL_BODY_ADAPTER_SCOPE:
        raise DualProviderPredictionRouterError(
            "route receipt must use the full-body adapter ablation scope"
        )
    sample_order_sha = _require_sha(
        value["sample_identity_order_sha256"], "receipt sample identity order SHA"
    )
    mask_set_sha = _require_sha(
        value["applicability_mask_set_sha256"], "receipt applicability mask-set SHA"
    )
    if (
        value["all_six_heads_enabled"] is not True
        or value["system_fallback"]
        != "actor_baseline_required_rank_route_not_authorized"
    ):
        raise DualProviderPredictionRouterError(
            "all six heads must be enabled while rank remains on actor baseline"
        )
    if value["rank_route_contract"] != {
        "source_contract_rank_score_is_a_six_head_route": False,
        "prior_rank_contract_sha_reusable": False,
        "independent_whole_provider_oof_passed": False,
        "rank_route_authorized": False,
        "rank_action_selection_must_fallback_to_actor_baseline": True,
    }:
        raise DualProviderPredictionRouterError(
            "route receipt does not reject the legacy rank route"
        )
    if value["capability"] != {
        "raw_metric_truth_recomputed": True,
        "outer_oof_predictions_content_bound": True,
        "per_fold_support_recomputed": True,
        "provider_specific_calibration_content_bound": True,
        "selected_variant_ids_are_provisional_non_executable": False,
        "runtime_route_exported_for_development_only": True,
        "production_route_exported": False,
        "signature_or_issuer_verification": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "performance_claim_authorized": False,
    }:
        raise DualProviderPredictionRouterError(
            "route receipt must remain development-only and no-promotion"
        )
    decisions = value["head_decisions"]
    if not isinstance(decisions, Mapping) or set(decisions) != set(HEADS):
        raise DualProviderPredictionRouterError(
            "route receipt must contain exactly six head decisions"
        )
    decoded: dict[str, dict[str, str]] = {}
    required_decision_fields = {
        "head",
        "support_gate_passed",
        "reference_variant_gate_passed",
        "candidate_variant_gate_passed",
        "selection_status",
        "selected_variant_id",
        "reason",
        "paired_gain_lcb95",
        "harmful_rate_ucb95",
        "selected_deployment_calibration_parameter_sha256",
        "actor_baseline_fallback_required",
    }
    for head in HEADS:
        row = decisions[head]
        if not isinstance(row, Mapping) or set(row) != required_decision_fields:
            raise DualProviderPredictionRouterError(f"{head} route fields changed")
        if (
            row["head"] != head
            or row["support_gate_passed"] is not True
            or row["reference_variant_gate_passed"] is not True
            or type(row["candidate_variant_gate_passed"]) is not bool
            or row["actor_baseline_fallback_required"] is not False
        ):
            raise DualProviderPredictionRouterError(f"{head} is not routable")
        provider_id = _require_string(
            row["selected_variant_id"], f"{head} selected provider"
        )
        calibration_sha = _require_sha(
            row["selected_deployment_calibration_parameter_sha256"],
            f"{head} selected calibration parameter SHA",
        )
        gain = _require_finite_float(row["paired_gain_lcb95"], f"{head} gain LCB")
        harmful = _require_finite_float(
            row["harmful_rate_ucb95"], f"{head} harmful-rate UCB"
        )
        if not 0.0 <= harmful <= 1.0:
            raise DualProviderPredictionRouterError(
                f"{head} harmful-rate UCB must lie in [0, 1]"
            )
        candidate_gate = row["candidate_variant_gate_passed"]
        if provider_id == CANDIDATE_PROVIDER_ID:
            if (
                row["selection_status"] != "selected_body_conditioned_candidate"
                or candidate_gate is not True
                or gain <= MINIMUM_PAIRED_GAIN_LCB
                or harmful > MAXIMUM_HARMFUL_RATE_UCB
            ):
                raise DualProviderPredictionRouterError(
                    f"{head} candidate selection status, gates, gain, or harm disagree"
                )
        elif provider_id == REFERENCE_PROVIDER_ID:
            candidate_would_pass = (
                candidate_gate is True
                and gain > MINIMUM_PAIRED_GAIN_LCB
                and harmful <= MAXIMUM_HARMFUL_RATE_UCB
            )
            if (
                row["selection_status"] != "fallback_body_agnostic_reference"
                or candidate_would_pass
            ):
                raise DualProviderPredictionRouterError(
                    f"{head} reference fallback status contradicts candidate evidence"
                )
        else:
            raise DualProviderPredictionRouterError(
                f"{head} selected a non-canonical provider"
            )
        decoded[head] = {
            "provider_id": provider_id,
            "calibration_parameter_sha256": calibration_sha,
        }
    return {
        "receipt_sha256": digest,
        "sample_identity_order_sha256": sample_order_sha,
        "applicability_mask_set_sha256": mask_set_sha,
        "head_decisions": decoded,
    }


def _tensor_kind(value: Any, role: str) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            raise DualProviderPredictionRouterError(
                f"{role} must be a non-empty floating NumPy array"
            )
        if not bool(np.isfinite(value).all()):
            raise DualProviderPredictionRouterError(f"{role} contains non-finite values")
        return "numpy"
    if torch is not None and isinstance(value, torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise DualProviderPredictionRouterError(
                f"{role} must already be detached from autograd"
            )
        if value.numel() == 0 or not torch.is_floating_point(value):
            raise DualProviderPredictionRouterError(
                f"{role} must be a non-empty real floating Torch tensor"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise DualProviderPredictionRouterError(f"{role} contains non-finite values")
        return "torch"
    raise DualProviderPredictionRouterError(
        f"{role} must be a NumPy array or detached Torch tensor"
    )


def _tensor_spec(value: Any, role: str) -> tuple[str, tuple[int, ...], Any, Any]:
    kind = _tensor_kind(value, role)
    device = value.device if kind == "torch" else None
    return kind, tuple(value.shape), value.dtype, device


def _copy_detached(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="K")
    return value.detach().clone(memory_format=torch.preserve_format)


def _normalize_sample_ids(value: Any, role: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise DualProviderPredictionRouterError(
            f"{role} must be a non-empty ordered sample-id sequence"
        )
    result = [_require_string(item, f"{role} item") for item in value]
    if len(set(result)) != len(result):
        raise DualProviderPredictionRouterError(f"{role} must contain unique ids")
    return result


def _normalize_applicable_masks(value: Any, sample_count: int, role: str) -> dict[str, list[bool]]:
    if not isinstance(value, Mapping) or set(value) != set(HEADS):
        raise DualProviderPredictionRouterError(
            f"{role} must contain exactly the six prediction heads"
        )
    result: dict[str, list[bool]] = {}
    for head in HEADS:
        mask = value[head]
        if (
            not isinstance(mask, Sequence)
            or isinstance(mask, (str, bytes))
            or len(mask) != sample_count
            or any(type(item) is not bool for item in mask)
        ):
            raise DualProviderPredictionRouterError(
                f"{role}/{head} must be an exact bool sequence of length N"
            )
        result[head] = list(mask)
    return result


def _validate_prediction_shapes(
    specs: Mapping[str, tuple[str, tuple[int, ...], Any, Any]],
    *,
    sample_count: int,
    provider_id: str,
) -> int:
    event_shape = (MEMBER_COUNT, sample_count, EVENT_CLASS_COUNT)
    binary_shape = (MEMBER_COUNT, sample_count)
    for field in ("post_event_logits", "next_event_logits"):
        if specs[field][1] != event_shape:
            raise DualProviderPredictionRouterError(
                f"{provider_id}/{field} shape must be (5, N, 5)"
            )
    for field in (
        "duration_log_mean",
        "duration_log_scale",
        "success_logit",
        "recovery_logit",
    ):
        if specs[field][1] != binary_shape:
            raise DualProviderPredictionRouterError(
                f"{provider_id}/{field} shape must be (5, N)"
            )
    object_shape = specs["object_mean"][1]
    if (
        len(object_shape) != 3
        or object_shape[0] != MEMBER_COUNT
        or object_shape[1] != sample_count
        or object_shape[2] < 1
        or specs["object_log_scale"][1] != object_shape
    ):
        raise DualProviderPredictionRouterError(
            f"{provider_id} object mean/scale shape must be (5, N, D) with D >= 1"
        )
    return int(object_shape[2])


def _validate_prediction_bundle(
    value: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    provider_set_sha256: str,
    route_receipt_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[str, tuple[int, ...], Any, Any]],
    list[str],
    dict[str, list[bool]],
    int,
]:
    fields = {
        "provider_id",
        "provider_set_sha256",
        "route_receipt_sha256",
        "sample_identity_order_sha256",
        "applicability_mask_set_sha256",
        "sample_id",
        "applicable_masks",
        "provider_artifact_sha256",
        "calibration_bundle_sha256",
        "predictions",
    }
    if not isinstance(value, Mapping):
        raise DualProviderPredictionRouterError("provider prediction bundle must be a mapping")
    _reject_legacy_rank_sha(value, "provider prediction bundle")
    if set(value) != fields:
        raise DualProviderPredictionRouterError(
            "provider prediction bundle fields changed"
        )
    provider_id = provider["provider_id"]
    if (
        value["provider_id"] != provider_id
        or value["provider_set_sha256"] != provider_set_sha256
        or value["route_receipt_sha256"] != route_receipt_sha256
        or value["sample_identity_order_sha256"]
        != provider["sample_identity_order_sha256"]
        or value["applicability_mask_set_sha256"]
        != provider["applicability_mask_set_sha256"]
        or value["provider_artifact_sha256"]
        != provider["provider_artifact_sha256"]
        or value["calibration_bundle_sha256"]
        != provider["calibration_bundle_sha256"]
    ):
        raise DualProviderPredictionRouterError(
            f"{provider_id} prediction provenance binding changed"
        )
    sample_ids = _normalize_sample_ids(value["sample_id"], f"{provider_id}/sample_id")
    if canonical_sha256(sample_ids) != value["sample_identity_order_sha256"]:
        raise DualProviderPredictionRouterError(
            f"{provider_id} sample identity order SHA does not match actual sample_id"
        )
    masks = _normalize_applicable_masks(
        value["applicable_masks"], len(sample_ids), f"{provider_id}/applicable_masks"
    )
    if canonical_sha256(masks) != value["applicability_mask_set_sha256"]:
        raise DualProviderPredictionRouterError(
            f"{provider_id} applicability mask-set SHA does not match actual masks"
        )
    predictions = value["predictions"]
    if not isinstance(predictions, Mapping):
        raise DualProviderPredictionRouterError(
            f"{provider_id} predictions must be a mapping"
        )
    if RANK_FIELD in predictions:
        raise DualProviderPredictionRouterError(
            "provider-specific source_contract_rank_score is forbidden"
        )
    if set(predictions) != set(PREDICTION_FIELDS):
        raise DualProviderPredictionRouterError(
            f"{provider_id} prediction head fields changed or are incomplete"
        )
    normalized = dict(predictions)
    specs = {
        field: _tensor_spec(normalized[field], f"{provider_id}/{field}")
        for field in PREDICTION_FIELDS
    }
    object_dim = _validate_prediction_shapes(
        specs, sample_count=len(sample_ids), provider_id=provider_id
    )
    for head, paired_fields in HEAD_FIELDS.items():
        if len(paired_fields) == 2 and specs[paired_fields[0]] != specs[paired_fields[1]]:
            raise DualProviderPredictionRouterError(
                f"{provider_id}/{head} paired prediction fields differ in "
                "backend, shape, dtype, or device"
            )
    return normalized, specs, sample_ids, masks, object_dim


def route_detached_predictions(
    *,
    provider_set: Mapping[str, Any],
    route_receipt: Mapping[str, Any],
    provider_prediction_bundles: Mapping[str, Mapping[str, Any]],
    expected_provider_set_sha256: str,
    expected_route_receipt_sha256: str,
    expected_calibration_set_sha256: str,
    actor_baseline_rank_score: Any | None = None,
    legacy_single_provider_rank_sha256: str | None = None,
) -> dict[str, Any]:
    """Route six heads for the exact same ordered development batch.

    The returned arrays/tensors are copies.  Inputs are never mutated, and a
    Torch tensor carrying an autograd graph is rejected rather than detached
    silently.  This function is not an inference or generalization route for
    new/unseen samples.
    """

    if legacy_single_provider_rank_sha256 is not None:
        _require_sha(
            legacy_single_provider_rank_sha256,
            "legacy single-provider rank SHA",
        )
        raise DualProviderPredictionRouterError(
            "legacy single-provider rank SHA is never reusable for dual-provider routing"
        )
    provider_audit = validate_provider_set(provider_set)
    expected_provider_sha = _require_sha(
        expected_provider_set_sha256, "expected provider-set SHA"
    )
    expected_receipt_sha = _require_sha(
        expected_route_receipt_sha256, "expected verified route receipt SHA"
    )
    expected_calibration_sha = _require_sha(
        expected_calibration_set_sha256, "expected verified calibration-set SHA"
    )
    if provider_audit["provider_set_sha256"] != expected_provider_sha:
        raise DualProviderPredictionRouterError(
            "provider set differs from the trusted expected SHA"
        )
    if provider_audit["route_receipt_sha256"] != expected_receipt_sha:
        raise DualProviderPredictionRouterError(
            "provider set is bound to a different route receipt"
        )
    if provider_audit["calibration_set_sha256"] != expected_calibration_sha:
        raise DualProviderPredictionRouterError(
            "provider set differs from the trusted calibration-set SHA"
        )
    receipt_audit = _validate_route_receipt(route_receipt, expected_receipt_sha)
    if (
        receipt_audit["sample_identity_order_sha256"]
        != provider_audit["sample_identity_order_sha256"]
        or receipt_audit["applicability_mask_set_sha256"]
        != provider_audit["applicability_mask_set_sha256"]
    ):
        raise DualProviderPredictionRouterError(
            "provider set and route receipt disagree on sample order or applicability masks"
        )
    providers = provider_audit["providers"]
    if (
        not isinstance(provider_prediction_bundles, Mapping)
        or set(provider_prediction_bundles) != set(providers)
    ):
        raise DualProviderPredictionRouterError(
            "prediction bundles must cover exactly the frozen two-provider set"
        )
    predictions: dict[str, dict[str, Any]] = {}
    specs: dict[str, dict[str, tuple[str, tuple[int, ...], Any, Any]]] = {}
    sample_ids_by_provider: dict[str, list[str]] = {}
    masks_by_provider: dict[str, dict[str, list[bool]]] = {}
    object_dims: dict[str, int] = {}
    for provider_id, provider in providers.items():
        bound_provider = {
            **provider,
            "sample_identity_order_sha256": provider_audit[
                "sample_identity_order_sha256"
            ],
            "applicability_mask_set_sha256": provider_audit[
                "applicability_mask_set_sha256"
            ],
        }
        (
            predictions[provider_id],
            specs[provider_id],
            sample_ids_by_provider[provider_id],
            masks_by_provider[provider_id],
            object_dims[provider_id],
        ) = _validate_prediction_bundle(
            provider_prediction_bundles[provider_id],
            provider=bound_provider,
            provider_set_sha256=expected_provider_sha,
            route_receipt_sha256=expected_receipt_sha,
        )
    first_id, second_id = sorted(providers)
    if sample_ids_by_provider[first_id] != sample_ids_by_provider[second_id]:
        raise DualProviderPredictionRouterError(
            "providers do not contain the same ordered sample identities"
        )
    if masks_by_provider[first_id] != masks_by_provider[second_id]:
        raise DualProviderPredictionRouterError(
            "providers do not contain identical per-head applicability masks"
        )
    if object_dims[first_id] != object_dims[second_id]:
        raise DualProviderPredictionRouterError(
            "providers disagree on object prediction dimension D"
        )
    for field in PREDICTION_FIELDS:
        if specs[first_id][field] != specs[second_id][field]:
            raise DualProviderPredictionRouterError(
                f"providers disagree on {field} backend, shape, dtype, or device"
            )

    routed: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for head, fields in HEAD_FIELDS.items():
        decision = receipt_audit["head_decisions"][head]
        provider_id = decision["provider_id"]
        if provider_id not in providers:
            raise DualProviderPredictionRouterError(
                f"{head} selected provider is absent from the frozen provider set"
            )
        provider = providers[provider_id]
        selected_parameter_sha = decision["calibration_parameter_sha256"]
        if (
            provider["head_calibration_parameter_sha256"][head]
            != selected_parameter_sha
        ):
            raise DualProviderPredictionRouterError(
                f"{head} route would mix provider-specific calibration"
            )
        for field in fields:
            routed[field] = _copy_detached(predictions[provider_id][field])
        provenance[head] = {
            "provider_id": provider_id,
            "provider_artifact_sha256": provider["provider_artifact_sha256"],
            "calibration_bundle_sha256": provider["calibration_bundle_sha256"],
            "calibration_parameter_sha256": selected_parameter_sha,
            "routed_fields": list(fields),
        }

    rank_route: dict[str, Any]
    if actor_baseline_rank_score is None:
        rank_route = {
            "status": "provider_rank_rejected_no_actor_baseline_score_supplied",
            "source_contract_rank_score_present": False,
            "actor_baseline_fallback_required": True,
        }
    else:
        baseline_spec = _tensor_spec(
            actor_baseline_rank_score, "actor baseline source_contract_rank_score"
        )
        success_spec = specs[
            provenance["success"]["provider_id"]
        ]["success_logit"]
        if baseline_spec != success_spec:
            raise DualProviderPredictionRouterError(
                "actor baseline rank score must match success-logit backend, shape, "
                "dtype, and device"
            )
        routed[RANK_FIELD] = _copy_detached(actor_baseline_rank_score)
        rank_route = {
            "status": "actor_baseline_rank_score_used",
            "source_contract_rank_score_present": True,
            "actor_baseline_fallback_required": True,
        }

    return {
        "format": ROUTED_PREDICTION_FORMAT,
        "status": ROUTED_PREDICTION_STATUS,
        "provider_set_sha256": expected_provider_sha,
        "route_receipt_sha256": expected_receipt_sha,
        "calibration_set_sha256": expected_calibration_sha,
        "sample_identity_order_sha256": provider_audit[
            "sample_identity_order_sha256"
        ],
        "applicability_mask_set_sha256": provider_audit[
            "applicability_mask_set_sha256"
        ],
        "sample_id": list(sample_ids_by_provider[first_id]),
        "applicable_masks": {
            head: list(mask)
            for head, mask in masks_by_provider[first_id].items()
        },
        "head_routes": provenance,
        "predictions": routed,
        "rank_route": rank_route,
        "capability": {
            "detached_in_memory_routing_only": True,
            "input_mutated": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
        },
    }


__all__ = [
    "DualProviderPredictionRouterError",
    "CANDIDATE_PROVIDER_ID",
    "FULL_BODY_ADAPTER_SCOPE",
    "HEAD_FIELDS",
    "HEADS",
    "PREDICTION_FIELDS",
    "PROVISIONAL_SELECTION_RECEIPT_FORMAT",
    "PROVIDER_SET_FORMAT",
    "REFERENCE_PROVIDER_ID",
    "RANK_FIELD",
    "ROUTED_PREDICTION_FORMAT",
    "build_provider_set",
    "canonical_bytes",
    "canonical_sha256",
    "route_detached_predictions",
    "validate_provider_set",
]
