from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smolvla_piper_dual_provider_prediction_router_v1 as router  # noqa: E402


REFERENCE = "body_agnostic_adapter"
CANDIDATE = "body_conditioned_adapter"
SAMPLE_IDS = ["development-sample-000", "development-sample-001", "development-sample-002"]
APPLICABLE_MASKS = {
    head: [True, index % 2 == 0, True]
    for index, head in enumerate(router.HEADS)
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resign(value: dict[str, Any], field: str) -> dict[str, Any]:
    unsigned = {key: child for key, child in value.items() if key != field}
    return {**unsigned, field: router.canonical_sha256(unsigned)}


def receipt(selected: dict[str, str] | None = None) -> dict[str, Any]:
    selected = selected or {
        "post_event": REFERENCE,
        "next_event": CANDIDATE,
        "duration": CANDIDATE,
        "success": REFERENCE,
        "recovery": CANDIDATE,
        "object_effect": REFERENCE,
    }
    decisions = {}
    for head in router.HEADS:
        provider = selected[head]
        decisions[head] = {
            "head": head,
            "support_gate_passed": True,
            "reference_variant_gate_passed": True,
            "candidate_variant_gate_passed": True,
            "selection_status": (
                "selected_body_conditioned_candidate"
                if provider == CANDIDATE
                else "fallback_body_agnostic_reference"
            ),
            "selected_variant_id": provider,
            "reason": "synthetic_contract_test",
            "paired_gain_lcb95": 0.02 if provider == CANDIDATE else 0.0,
            "harmful_rate_ucb95": 0.05,
            "selected_deployment_calibration_parameter_sha256": sha(
                f"{provider}-{head}-calibration-parameter"
            ),
            "actor_baseline_fallback_required": False,
        }
    base = {
        "format": router.ROUTE_RECEIPT_FORMAT,
        "status": router.ROUTE_RECEIPT_STATUS,
        "plan_sha256": sha("plan"),
        "evidence_sha256": sha("evidence"),
        "variant_scope": router.FULL_BODY_ADAPTER_SCOPE,
        "sample_identity_order_sha256": router.canonical_sha256(SAMPLE_IDS),
        "applicability_mask_set_sha256": router.canonical_sha256(
            APPLICABLE_MASKS
        ),
        "head_decisions": decisions,
        "all_six_heads_enabled": True,
        "system_fallback": "actor_baseline_required_rank_route_not_authorized",
        "rank_route_contract": {
            "source_contract_rank_score_is_a_six_head_route": False,
            "prior_rank_contract_sha_reusable": False,
            "independent_whole_provider_oof_passed": False,
            "rank_route_authorized": False,
            "rank_action_selection_must_fallback_to_actor_baseline": True,
        },
        "capability": {
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
        },
    }
    return {**base, "receipt_sha256": router.canonical_sha256(base)}


def provider_descriptors(route: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for provider in (REFERENCE, CANDIDATE):
        result.append(
            {
                "provider_id": provider,
                "provider_artifact_sha256": sha(f"{provider}-artifact"),
                "calibration_bundle_sha256": sha(f"{provider}-calibration-bundle"),
                "head_calibration_parameter_sha256": {
                    head: sha(f"{provider}-{head}-calibration-parameter")
                    for head in router.HEADS
                },
            }
        )
    return result


def numpy_predictions(offset: float) -> dict[str, np.ndarray]:
    return {
        "post_event_logits": np.full((5, 3, 5), offset + 1, np.float32),
        "next_event_logits": np.full((5, 3, 5), offset + 2, np.float32),
        "duration_log_mean": np.full((5, 3), offset + 3, np.float32),
        "duration_log_scale": np.full((5, 3), offset + 4, np.float32),
        "success_logit": np.full((5, 3), offset + 5, np.float32),
        "recovery_logit": np.full((5, 3), offset + 6, np.float32),
        "object_mean": np.full((5, 3, 7), offset + 7, np.float32),
        "object_log_scale": np.full((5, 3, 7), offset + 8, np.float32),
    }


def components() -> dict[str, Any]:
    route = receipt()
    provider_set = router.build_provider_set(
        route_receipt_sha256=route["receipt_sha256"],
        sample_identity_order_sha256=route["sample_identity_order_sha256"],
        applicability_mask_set_sha256=route[
            "applicability_mask_set_sha256"
        ],
        providers=provider_descriptors(route),
    )
    by_id = {row["provider_id"]: row for row in provider_set["providers"]}
    predictions = {
        REFERENCE: numpy_predictions(10.0),
        CANDIDATE: numpy_predictions(20.0),
    }
    bundles = {
        provider: {
            "provider_id": provider,
            "provider_set_sha256": provider_set["provider_set_sha256"],
            "route_receipt_sha256": route["receipt_sha256"],
            "sample_identity_order_sha256": route[
                "sample_identity_order_sha256"
            ],
            "applicability_mask_set_sha256": route[
                "applicability_mask_set_sha256"
            ],
            "sample_id": list(SAMPLE_IDS),
            "applicable_masks": copy.deepcopy(APPLICABLE_MASKS),
            "provider_artifact_sha256": by_id[provider][
                "provider_artifact_sha256"
            ],
            "calibration_bundle_sha256": by_id[provider][
                "calibration_bundle_sha256"
            ],
            "predictions": predictions[provider],
        }
        for provider in (REFERENCE, CANDIDATE)
    }
    return {
        "route": route,
        "provider_set": provider_set,
        "predictions": predictions,
        "bundles": bundles,
    }


def route(parts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    provider_set = parts["provider_set"]
    return router.route_detached_predictions(
        provider_set=provider_set,
        route_receipt=parts["route"],
        provider_prediction_bundles=parts["bundles"],
        expected_provider_set_sha256=provider_set["provider_set_sha256"],
        expected_route_receipt_sha256=parts["route"]["receipt_sha256"],
        expected_calibration_set_sha256=provider_set["calibration_set_sha256"],
        **kwargs,
    )


def rebind_route(parts: dict[str, Any], changed: dict[str, Any]) -> None:
    provider_set = router.build_provider_set(
        route_receipt_sha256=changed["receipt_sha256"],
        sample_identity_order_sha256=changed["sample_identity_order_sha256"],
        applicability_mask_set_sha256=changed[
            "applicability_mask_set_sha256"
        ],
        providers=provider_descriptors(changed),
    )
    parts["route"] = changed
    parts["provider_set"] = provider_set
    by_id = {row["provider_id"]: row for row in provider_set["providers"]}
    for provider, bundle in parts["bundles"].items():
        bundle["provider_set_sha256"] = provider_set["provider_set_sha256"]
        bundle["route_receipt_sha256"] = changed["receipt_sha256"]
        bundle["sample_identity_order_sha256"] = changed[
            "sample_identity_order_sha256"
        ]
        bundle["applicability_mask_set_sha256"] = changed[
            "applicability_mask_set_sha256"
        ]
        bundle["provider_artifact_sha256"] = by_id[provider][
            "provider_artifact_sha256"
        ]
        bundle["calibration_bundle_sha256"] = by_id[provider][
            "calibration_bundle_sha256"
        ]


def test_routes_all_six_heads_atomically_without_mutating_inputs() -> None:
    parts = components()
    result = route(parts)
    selected = {
        head: parts["route"]["head_decisions"][head]["selected_variant_id"]
        for head in router.HEADS
    }
    for head, fields in router.HEAD_FIELDS.items():
        for field in fields:
            expected = parts["predictions"][selected[head]][field]
            assert np.array_equal(result["predictions"][field], expected)
            assert result["predictions"][field] is not expected
        assert result["head_routes"][head]["provider_id"] == selected[head]
        assert result["head_routes"][head]["routed_fields"] == list(fields)
    assert set(result["predictions"]) == set(router.PREDICTION_FIELDS)
    assert result["sample_id"] == SAMPLE_IDS
    assert result["applicable_masks"] == APPLICABLE_MASKS
    assert result["rank_route"]["status"] == (
        "provider_rank_rejected_no_actor_baseline_score_supplied"
    )
    assert result["capability"]["promotion_authorized"] is False
    assert result["capability"]["deployment_authorized"] is False


def test_actor_baseline_is_the_only_accepted_rank_score() -> None:
    parts = components()
    baseline = np.full((5, 3), 99.0, np.float32)
    result = route(parts, actor_baseline_rank_score=baseline)
    assert np.array_equal(result["predictions"][router.RANK_FIELD], baseline)
    assert result["predictions"][router.RANK_FIELD] is not baseline
    assert result["rank_route"]["status"] == "actor_baseline_rank_score_used"

    parts = components()
    parts["bundles"][REFERENCE]["predictions"][router.RANK_FIELD] = baseline
    with pytest.raises(router.DualProviderPredictionRouterError, match="forbidden"):
        route(parts)

    parts = components()
    with pytest.raises(router.DualProviderPredictionRouterError, match="never reusable"):
        route(parts, legacy_single_provider_rank_sha256=sha("old-rank"))


@pytest.mark.parametrize(
    ("expected_name", "message"),
    [
        ("expected_provider_set_sha256", "trusted expected SHA"),
        ("expected_route_receipt_sha256", "different route receipt"),
        ("expected_calibration_set_sha256", "trusted calibration-set SHA"),
    ],
)
def test_rejects_wrong_trusted_content_addresses(
    expected_name: str, message: str
) -> None:
    parts = components()
    provider_set = parts["provider_set"]
    kwargs = {
        "provider_set": provider_set,
        "route_receipt": parts["route"],
        "provider_prediction_bundles": parts["bundles"],
        "expected_provider_set_sha256": provider_set["provider_set_sha256"],
        "expected_route_receipt_sha256": parts["route"]["receipt_sha256"],
        "expected_calibration_set_sha256": provider_set[
            "calibration_set_sha256"
        ],
    }
    kwargs[expected_name] = sha(f"wrong-{expected_name}")
    with pytest.raises(router.DualProviderPredictionRouterError, match=message):
        router.route_detached_predictions(**kwargs)


def test_rejects_provider_specific_calibration_mix() -> None:
    parts = components()
    changed = copy.deepcopy(parts["provider_set"])
    chosen = parts["route"]["head_decisions"]["duration"]["selected_variant_id"]
    for descriptor in changed["providers"]:
        if descriptor["provider_id"] == chosen:
            descriptor["head_calibration_parameter_sha256"]["duration"] = sha(
                "wrong-duration-calibration"
            )
    binding = [
        {
            key: descriptor[key]
            for key in (
                "provider_id",
                "provider_artifact_sha256",
                "calibration_bundle_sha256",
                "head_calibration_parameter_sha256",
            )
        }
        for descriptor in changed["providers"]
    ]
    changed["calibration_set_sha256"] = router.canonical_sha256(binding)
    changed = resign(changed, "provider_set_sha256")
    parts["provider_set"] = changed
    by_id = {row["provider_id"]: row for row in changed["providers"]}
    for provider, bundle in parts["bundles"].items():
        bundle["provider_set_sha256"] = changed["provider_set_sha256"]
        bundle["provider_artifact_sha256"] = by_id[provider][
            "provider_artifact_sha256"
        ]
        bundle["calibration_bundle_sha256"] = by_id[provider][
            "calibration_bundle_sha256"
        ]
    with pytest.raises(router.DualProviderPredictionRouterError, match="mix"):
        route(parts)


@pytest.mark.parametrize(
    "binding",
    ["sample_identity_order_sha256", "applicability_mask_set_sha256"],
)
def test_rejects_provider_bundle_order_or_mask_mismatch(binding: str) -> None:
    parts = components()
    parts["bundles"][CANDIDATE][binding] = sha(f"wrong-{binding}")
    with pytest.raises(
        router.DualProviderPredictionRouterError,
        match="provenance binding changed",
    ):
        route(parts)


def test_recomputes_sample_order_sha_from_actual_candidate_ids() -> None:
    parts = components()
    candidate_ids = parts["bundles"][CANDIDATE]["sample_id"]
    candidate_ids[0], candidate_ids[1] = candidate_ids[1], candidate_ids[0]
    with pytest.raises(
        router.DualProviderPredictionRouterError,
        match="does not match actual sample_id",
    ):
        route(parts)


def test_recomputes_mask_sha_from_actual_candidate_masks() -> None:
    parts = components()
    masks = parts["bundles"][CANDIDATE]["applicable_masks"]
    masks["recovery"][1] = not masks["recovery"][1]
    with pytest.raises(
        router.DualProviderPredictionRouterError,
        match="does not match actual masks",
    ):
        route(parts)


def test_rejects_selection_status_provider_contradiction() -> None:
    parts = components()
    changed = copy.deepcopy(parts["route"])
    changed["head_decisions"]["duration"]["selection_status"] = (
        "fallback_body_agnostic_reference"
    )
    changed = resign(changed, "receipt_sha256")
    rebind_route(parts, changed)
    with pytest.raises(
        router.DualProviderPredictionRouterError,
        match="candidate selection status",
    ):
        route(parts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("paired_gain_lcb95", float("nan"), "finite numeric"),
        ("harmful_rate_ucb95", 1.1, r"\[0, 1\]"),
    ],
)
def test_rejects_nonfinite_gain_or_out_of_range_harm(
    field: str, value: float, message: str
) -> None:
    parts = components()
    changed = copy.deepcopy(parts["route"])
    changed["head_decisions"]["duration"][field] = value
    # NaN cannot be canonically hashed, so a trusted receipt cannot contain it.
    if np.isfinite(value):
        changed = resign(changed, "receipt_sha256")
        rebind_route(parts, changed)
        with pytest.raises(router.DualProviderPredictionRouterError, match=message):
            route(parts)
    else:
        with pytest.raises(ValueError, match="Out of range float values"):
            resign(changed, "receipt_sha256")


def test_provider_ids_must_be_the_canonical_adapter_pair() -> None:
    route_receipt = receipt()
    wrong = provider_descriptors(route_receipt)
    wrong[0]["provider_id"] = "body_agnostic"
    with pytest.raises(router.DualProviderPredictionRouterError, match="canonical"):
        router.build_provider_set(
            route_receipt_sha256=route_receipt["receipt_sha256"],
            sample_identity_order_sha256=route_receipt[
                "sample_identity_order_sha256"
            ],
            applicability_mask_set_sha256=route_receipt[
                "applicability_mask_set_sha256"
            ],
            providers=wrong,
        )


@pytest.mark.parametrize("failure", ["shape", "dtype", "finite", "missing"])
def test_rejects_prediction_schema_mismatch(failure: str) -> None:
    parts = components()
    predictions = parts["bundles"][CANDIDATE]["predictions"]
    if failure == "shape":
        predictions["duration_log_scale"] = np.zeros((5, 2), np.float32)
        message = "shape must"
    elif failure == "dtype":
        predictions["object_log_scale"] = predictions["object_log_scale"].astype(
            np.float64
        )
        message = "paired prediction"
    elif failure == "finite":
        predictions["success_logit"][0, 0] = np.nan
        message = "non-finite"
    else:
        del predictions["recovery_logit"]
        message = "incomplete"
    with pytest.raises(router.DualProviderPredictionRouterError, match=message):
        route(parts)


def test_rejects_cross_provider_shape_or_dtype_mismatch() -> None:
    parts = components()
    parts["bundles"][CANDIDATE]["predictions"]["success_logit"] = np.zeros(
        (5, 4), np.float32
    )
    with pytest.raises(router.DualProviderPredictionRouterError, match="shape must"):
        route(parts)

    parts = components()
    parts["bundles"][CANDIDATE]["predictions"]["next_event_logits"] = np.zeros(
        (5, 3, 5), np.float64
    )
    with pytest.raises(router.DualProviderPredictionRouterError, match="providers disagree"):
        route(parts)


@pytest.mark.parametrize(
    ("field", "shape", "message"),
    [
        ("success_logit", (4, 3), r"\(5, N\)"),
        ("next_event_logits", (5, 2, 5), r"\(5, N, 5\)"),
        ("post_event_logits", (5, 3, 4), r"\(5, N, 5\)"),
        ("object_mean", (5, 3), r"\(5, N, D\)"),
    ],
)
def test_rejects_member_sample_class_or_object_dimension_mismatch(
    field: str, shape: tuple[int, ...], message: str
) -> None:
    parts = components()
    parts["bundles"][CANDIDATE]["predictions"][field] = np.zeros(
        shape, dtype=np.float32
    )
    with pytest.raises(router.DualProviderPredictionRouterError, match=message):
        route(parts)


def test_torch_inputs_must_already_be_detached_and_are_copied() -> None:
    parts = components()
    for bundle in parts["bundles"].values():
        bundle["predictions"] = {
            field: torch.from_numpy(value.copy())
            for field, value in bundle["predictions"].items()
        }
    parts["bundles"][CANDIDATE]["predictions"]["success_logit"] = torch.ones(
        (5, 3), dtype=torch.float32, requires_grad=True
    )
    with pytest.raises(router.DualProviderPredictionRouterError, match="detached"):
        route(parts)

    parts = components()
    for bundle in parts["bundles"].values():
        bundle["predictions"] = {
            field: torch.from_numpy(value.copy())
            for field, value in bundle["predictions"].items()
        }
    result = route(parts)
    assert all(
        isinstance(value, torch.Tensor)
        and value.requires_grad is False
        and value.grad_fn is None
        for value in result["predictions"].values()
    )
    assert result["predictions"]["success_logit"].data_ptr() != parts[
        "bundles"
    ][REFERENCE]["predictions"]["success_logit"].data_ptr()


def test_rejects_disabled_or_incomplete_receipt() -> None:
    parts = components()
    changed = copy.deepcopy(parts["route"])
    changed["all_six_heads_enabled"] = False
    changed["system_fallback"] = (
        "actor_baseline_required_rank_route_not_authorized_and_heads_disabled"
    )
    changed = resign(changed, "receipt_sha256")
    rebind_route(parts, changed)
    with pytest.raises(router.DualProviderPredictionRouterError, match="six heads"):
        route(parts)


def test_provider_set_rejects_legacy_rank_sha_even_if_rehashed() -> None:
    parts = components()
    changed = copy.deepcopy(parts["provider_set"])
    changed["root_group_ranker_sha256"] = sha("old-production-rank")
    changed = resign(changed, "provider_set_sha256")
    with pytest.raises(router.DualProviderPredictionRouterError, match="legacy"):
        router.validate_provider_set(changed)


def test_real_provisional_selection_builder_receipt_is_not_executable() -> None:
    helper_path = ROOT / "tests" / "test_verify_smolvla_per_head_body_mode_selection_v1.py"
    spec = importlib.util.spec_from_file_location(
        "_body_mode_contract_test_helpers", helper_path
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    parts = helper.components(helper.mode.FULL_BODY_ADAPTER_SCOPE)
    evidence = helper.build_evidence(parts)
    provisional = helper.mode.build_selection_receipt(parts["plan"], evidence)
    assert provisional["capability"][
        "selected_variant_ids_are_provisional_non_executable"
    ] is True
    assert provisional["capability"]["runtime_route_exported"] is False

    provider_set = router.build_provider_set(
        route_receipt_sha256=provisional["receipt_sha256"],
        sample_identity_order_sha256=evidence["variant_views"][0][
            "sample_order_sha256"
        ],
        applicability_mask_set_sha256=evidence["variant_views"][0][
            "applicable_mask_set_sha256"
        ],
        providers=provider_descriptors(provisional),
    )
    with pytest.raises(
        router.DualProviderPredictionRouterError,
        match="provisional non-executable",
    ):
        router.route_detached_predictions(
            provider_set=provider_set,
            route_receipt=provisional,
            provider_prediction_bundles={},
            expected_provider_set_sha256=provider_set["provider_set_sha256"],
            expected_route_receipt_sha256=provisional["receipt_sha256"],
            expected_calibration_set_sha256=provider_set[
                "calibration_set_sha256"
            ],
        )
