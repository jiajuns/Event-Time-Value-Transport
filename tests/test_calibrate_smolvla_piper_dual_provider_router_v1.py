from __future__ import annotations

import copy
import functools
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import calibrate_smolvla_piper_dual_provider_router_v1 as router  # noqa: E402
import smolvla_piper_dual_provider_prediction_router_v1 as prediction_router  # noqa: E402


def synthetic_bundle(group_count: int = 250):
    rows = 2
    n = group_count * rows
    group_number = np.repeat(np.arange(group_count), rows)
    ordinal = np.tile(np.arange(rows), group_count)
    sample = np.asarray([f"sample-{index:04d}" for index in range(n)])
    group = np.asarray([f"group-{index:04d}" for index in group_number])
    semantic_number = group_number // 2
    current = semantic_number % 5
    post = (current + 1 + ordinal) % 5
    next_event = 1 + (semantic_number % 4)
    observed = ordinal == 0
    duration = 1.0 + ((semantic_number * 3 + ordinal) % 9).astype(float)
    success = (ordinal == 0).astype(np.int64)
    recovery = success.copy()
    object_target = np.zeros((n, 3), dtype=float)
    object_target[observed] = np.asarray([3.0, -2.0, 1.0])
    labels = {
        "sample_id": sample,
        "group_id": group,
        "group_row_ordinal": ordinal.astype(np.int64),
        "semantic_reset_cluster_id": np.asarray(
            [f"semantic-{index:04d}" for index in semantic_number]
        ),
        "body_id": np.asarray(
            ["piper" if index % 2 == 0 else "franka" for index in group_number]
        ),
        "actor_contract_id": np.asarray(["actor-v1"] * n),
        "current_event": current.astype(np.int64),
        "post_event": post.astype(np.int64),
        "post_event_observed": np.ones(n, dtype=bool),
        "next_event": next_event.astype(np.int64),
        "next_event_observed": observed.copy(),
        "success": success,
        "success_observed": np.ones(n, dtype=bool),
        "regress": np.ones(n, dtype=bool),
        "recovery": recovery,
        "recovery_observed": np.ones(n, dtype=bool),
        "duration": duration,
        "duration_applicable": np.ones(n, dtype=bool),
        "duration_observed": observed.copy(),
        "object_target": object_target,
        "object_observed": np.ones(n, dtype=bool),
    }
    masks = {
        "post_event": labels["post_event_observed"].copy(),
        "next_event": labels["next_event_observed"].copy(),
        "duration": labels["duration_applicable"].copy(),
        "success": labels["success_observed"].copy(),
        "recovery": labels["recovery_observed"].copy(),
        "object_effect": labels["object_observed"].copy(),
    }

    def make_provider(name: str, candidate: bool):
        bad = np.isin(semantic_number % 11, [0, 3])
        provider_bad = (semantic_number % 11 == 0) if candidate else bad
        structured_bad = semantic_number % 11 == 0
        post_logits = np.full((5, n, 5), -4.0)
        next_logits = np.full((5, n, 5), -4.0)
        binary = np.where(success == 1, 5.0, -5.0)
        recovery_logit = np.where(recovery == 1, 5.0, -5.0)
        duration_mean = np.log1p(duration + np.where(observed, 0.0, 2.0))
        duration_scale = np.full(n, 0.0)
        object_mean = object_target.copy()
        object_scale = np.full_like(object_target, 0.0)
        for index in range(n):
            post_logits[:, index, post[index]] = 5.0
            next_logits[:, index, next_event[index]] = 5.0
        if bool(provider_bad.any()):
            for index in np.flatnonzero(provider_bad):
                wrong_post = (post[index] + 1) % 5
                wrong_next = 1 + (next_event[index] % 4)
                post_logits[:, index, :] = 0.0
                post_logits[:, index, wrong_post] = 0.0 if candidate else 1e-6
                next_logits[:, index, :] = 0.0
                next_logits[:, index, wrong_next] = 0.0 if candidate else 1e-6
            magnitude = 0.0 if candidate else 1e-6
            binary[provider_bad] = np.where(success[provider_bad] == 1, -magnitude, magnitude)
            recovery_logit[provider_bad] = np.where(recovery[provider_bad] == 1, -magnitude, magnitude)
        if bool(structured_bad.any()):
            offset = 1.5 if candidate else 2.5
            duration_mean[structured_bad] = np.log1p(np.maximum(duration[structured_bad] - offset, 0.1))
            duration_scale[structured_bad] = 0.8
            object_mean[structured_bad] += (
                np.asarray([2.0, -1.5, 1.0])
                if candidate else np.asarray([2.0, -1.5, 1.0])
            )
            object_scale[structured_bad] = 0.8
        member_jitter = np.linspace(-0.04, 0.04, 5)[:, None]
        prediction_arrays = {
            "post_event_logits": post_logits,
            "next_event_logits": next_logits,
            "success_logit": binary[None, :] + member_jitter,
            "recovery_logit": recovery_logit[None, :] + member_jitter,
            "duration_log_mean": duration_mean[None, :] + member_jitter * 0.1,
            "duration_log_scale": np.tile(duration_scale, (5, 1)),
            "object_mean": np.tile(object_mean, (5, 1, 1))
            + member_jitter[:, :, None] * 0.01,
            "object_log_scale": np.tile(object_scale, (5, 1, 1)),
        }
        manifest = router.build_provider_manifest(
            provider_id=name,
            provider_artifact_sha256=("a" if candidate else "b") * 64,
            shared_core_lineage_sha256="c" * 64,
            prediction_tensor_sha256=router.prediction_tensor_set_sha256(
                prediction_arrays
            ),
            training_execution_group_ids=[f"training-group-{index:04d}" for index in range(5)],
            training_semantic_reset_cluster_ids=[f"training-semantic-{index:04d}" for index in range(5)],
            members=[
                {
                    "member_index": index,
                    "seed": 700 + index,
                    "checkpoint_sha256": (f"{index + (6 if candidate else 1):x}" * 64)[:64],
                }
                for index in range(5)
            ],
        )
        return {
            "provider_id": name,
            "provider_artifact_sha256": ("a" if candidate else "b") * 64,
            "provider_manifest": manifest,
            "member_count": 5,
            "sample_id": sample.copy(),
            "group_id": group.copy(),
            "group_row_ordinal": ordinal.astype(np.int64),
            "applicable_masks": {key: value.copy() for key, value in masks.items()},
            **prediction_arrays,
        }

    providers = {
        router.REFERENCE_PROVIDER: make_provider(router.REFERENCE_PROVIDER, False),
        router.CANDIDATE_PROVIDER: make_provider(router.CANDIDATE_PROVIDER, True),
    }
    folds = router.build_five_fold_group_plan(
        group, labels["semantic_reset_cluster_id"]
    )
    return providers, labels, folds


def run(bundle):
    return router.calibrate_dual_provider_router(*bundle, bootstrap_samples=200)


@functools.lru_cache(maxsize=1)
def positive_result():
    providers, labels, folds = synthetic_bundle()
    return providers, labels, run((providers, labels, folds))


def test_nested_oof_receipt_is_raw_recomputed_and_content_bound():
    _, _, result = positive_result()
    receipt = result["route_receipt"]
    assert receipt["format"] == "etsf_smolvla_piper_dual_provider_raw_recomputed_route_receipt_v1"
    assert receipt["capability"]["raw_metric_truth_recomputed"] is True
    assert receipt["capability"]["outer_oof_predictions_content_bound"] is True
    assert receipt["capability"]["production_route_exported"] is False
    assert receipt["capability"]["deployment_authorized"] is False
    assert receipt["capability"]["promotion_authorized"] is False
    assert receipt["all_six_heads_enabled"] is True
    assert result["rank_route"]["required_action_route"] == "actor_baseline"
    assert result["every_sample_heldout_exactly_once"] is True
    assert len(result["folds"]) == 5
    for fold in result["folds"]:
        assert len(fold["inner_fold_receipts"]) == 5
        assert fold["outer_heldout_labels_used_for_calibration_or_provider_selection"] is False
        assert all(value == 1 for value in fold["heldout_inference_calls_per_provider"].values())
    assert all(
        result["head_routes"][head]["selected_provider"] in router.PROVIDERS
        for head in router.HEADS
    )


def test_calibrator_receipt_routes_through_real_prediction_router():
    providers, labels, result = positive_result()
    receipt = result["route_receipt"]
    provider_set = prediction_router.build_provider_set(
        route_receipt_sha256=receipt["receipt_sha256"],
        sample_identity_order_sha256=receipt["sample_identity_order_sha256"],
        applicability_mask_set_sha256=receipt["applicability_mask_set_sha256"],
        providers=result["provider_descriptors_for_prediction_router"],
    )
    bundles = {}
    prediction_fields = prediction_router.PREDICTION_FIELDS
    for provider_id, source in providers.items():
        descriptor = next(
            row for row in result["provider_descriptors_for_prediction_router"]
            if row["provider_id"] == provider_id
        )
        bundles[provider_id] = {
            "provider_id": provider_id,
            "provider_set_sha256": provider_set["provider_set_sha256"],
            "route_receipt_sha256": receipt["receipt_sha256"],
            "sample_identity_order_sha256": receipt["sample_identity_order_sha256"],
            "applicability_mask_set_sha256": receipt["applicability_mask_set_sha256"],
            "sample_id": source["sample_id"].tolist(),
            "applicable_masks": {
                head: mask.tolist() for head, mask in source["applicable_masks"].items()
            },
            "provider_artifact_sha256": descriptor["provider_artifact_sha256"],
            "calibration_bundle_sha256": descriptor["calibration_bundle_sha256"],
            "predictions": {field: source[field] for field in prediction_fields},
        }
    routed = prediction_router.route_detached_predictions(
        provider_set=provider_set,
        route_receipt=receipt,
        provider_prediction_bundles=bundles,
        expected_provider_set_sha256=provider_set["provider_set_sha256"],
        expected_route_receipt_sha256=receipt["receipt_sha256"],
        expected_calibration_set_sha256=provider_set["calibration_set_sha256"],
        actor_baseline_rank_score=np.zeros_like(
            providers[router.REFERENCE_PROVIDER]["success_logit"]
        ),
    )
    assert routed["rank_route"]["status"] == "actor_baseline_rank_score_used"
    for head in router.HEADS:
        assert routed["head_routes"][head]["provider_id"] == receipt["head_decisions"][head]["selected_variant_id"]
    assert result["head_routes"]["post_event"]["selected_provider"] == router.CANDIDATE_PROVIDER
    bindings = result["content_bindings"]
    assert {
        "provider_fitted_outer_oof_full_array_set_sha256",
        "provider_effective_outer_oof_full_array_set_sha256",
        "provider_uncalibrated_outer_oof_full_array_set_sha256",
        "train_only_baseline_outer_oof_full_array_set_sha256",
        "routed_calibrated_outer_oof_full_array_set_sha256",
        "routed_uncalibrated_outer_oof_full_array_set_sha256",
    } <= set(bindings)
    assert all(
        fold["outer_train_fitted_parameters_sha256"]
        != fold["outer_train_effective_parameters_sha256"]
        or fold["outer_train_fitted_parameters"] == fold["outer_train_effective_parameters"]
        for fold in result["folds"]
    )
    assert all(
        result["head_routes"][head]["deployment_parameter_sha256"]
        for head in router.HEADS
    )


@pytest.mark.parametrize("kind", ["sample", "group", "order", "mask"])
def test_identity_and_mask_mismatch_fail_closed(kind):
    providers, labels, folds = synthetic_bundle()
    candidate = providers[router.CANDIDATE_PROVIDER]
    if kind == "sample":
        candidate["sample_id"][[0, 1]] = candidate["sample_id"][[1, 0]]
    elif kind == "group":
        candidate["group_id"][0] = "wrong-group"
    elif kind == "order":
        candidate["group_row_ordinal"][0] = 7
    else:
        candidate["applicable_masks"]["duration"][0] = False
    with pytest.raises(router.DualProviderRouterError):
        run((providers, labels, folds))


def test_outer_fold_duplicate_heldout_group_fails_exact_once():
    providers, labels, folds = synthetic_bundle()
    duplicate = folds[0]["heldout_group_ids"][0]
    displaced = folds[1]["heldout_group_ids"][0]
    folds[1]["heldout_group_ids"] = sorted(
        [duplicate] + folds[1]["heldout_group_ids"][1:]
    )
    universe = set(labels["group_id"].tolist())
    folds[1]["training_group_ids"] = sorted(universe - set(folds[1]["heldout_group_ids"]))
    assert displaced not in folds[1]["heldout_group_ids"]
    with pytest.raises(router.DualProviderRouterError, match="exact-once"):
        run((providers, labels, folds))


def test_noncanonical_but_exact_once_fold_plan_is_rejected():
    providers, labels, folds = synthetic_bundle()
    heldout = [row["heldout_group_ids"] for row in folds]
    rotated = heldout[1:] + heldout[:1]
    universe = set(labels["group_id"].tolist())
    noncanonical = [
        {
            "fold_index": index,
            "heldout_group_ids": sorted(groups),
            "training_group_ids": sorted(universe - set(groups)),
        }
        for index, groups in enumerate(rotated)
    ]
    with pytest.raises(router.DualProviderRouterError, match="canonical"):
        run((providers, labels, noncanonical))


def test_one_semantic_cluster_cannot_cross_folds():
    providers, labels, folds = synthetic_bundle()
    left = folds[0]["heldout_group_ids"][0]
    same_cluster = labels["semantic_reset_cluster_id"][labels["group_id"] == left][0]
    pair = next(
        group for group, cluster in zip(labels["group_id"], labels["semantic_reset_cluster_id"])
        if cluster == same_cluster and group != left
    )
    other = folds[1]["heldout_group_ids"][0]
    folds[0]["heldout_group_ids"] = sorted(
        [other if value == pair else value for value in folds[0]["heldout_group_ids"]]
    )
    folds[1]["heldout_group_ids"] = sorted(
        [pair if value == other else value for value in folds[1]["heldout_group_ids"]]
    )
    universe = set(labels["group_id"].tolist())
    for fold in folds:
        fold["training_group_ids"] = sorted(universe - set(fold["heldout_group_ids"]))
    with pytest.raises(router.DualProviderRouterError, match="semantic reset cluster"):
        run((providers, labels, folds))


def test_constant_uncertainty_cannot_gain_from_source_order():
    count = 20
    evidence = router._uncertainty_evidence(
        np.ones(count), np.tile([0.0, 1.0], count // 2),
        np.asarray([f"semantic-{index:02d}" for index in range(count)]),
        np.ones(count, dtype=bool), samples=200, role="constant-tie-test",
    )
    assert evidence["passed"] is False
    assert evidence["aurc_gain_over_random"] == pytest.approx(0.0)
    assert evidence["high_minus_low_uncertainty_quartile_error"] == pytest.approx(0.0)


def test_zero_harm_in_ten_clusters_has_wilson_ucb_above_gate():
    count = 10
    evidence = router._harm_evidence(
        np.zeros(count), np.zeros(count), np.ones(count), np.ones(count),
        np.asarray([f"semantic-{index:02d}" for index in range(count)]),
        np.ones(count, dtype=bool), samples=200, role="small-zero-harm",
    )
    assert evidence["proper_loss"]["point"] == 0.0
    assert evidence["proper_loss"]["ucb95"] > 0.10
    assert evidence["both_harmful_rate_ucb_passed"] is False


def test_partial_nan_inside_same_semantic_cluster_is_rejected_not_dropped():
    values = np.asarray([0.0, np.nan, 1.0, 1.0])
    groups = np.asarray(["semantic-a", "semantic-a", "semantic-b", "semantic-b"])
    executions = np.asarray(["exec-a1", "exec-a2", "exec-b1", "exec-b2"])
    with pytest.raises(router.DualProviderRouterError, match="row deletion is forbidden"):
        router._gain_evidence(
            values, np.zeros(4), groups, np.ones(4, dtype=bool),
            execution_groups=executions, samples=200, role="partial-nan-attack",
        )


def test_wide_coverage_interval_containing_nominal_still_fails_absolute_error_gate():
    coverage = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    common = {
        "proper": np.zeros(5),
        "decision": np.zeros(5),
        "uncertainty": np.linspace(0.0, 1.0, 5),
        "calibration_secondary": np.zeros(5),
        "interval_coverage": coverage,
    }
    labels = {
        "semantic_reset_cluster_id": np.asarray([f"semantic-{i}" for i in range(5)]),
        "group_id": np.asarray([f"exec-{i}" for i in range(5)]),
    }
    quality = router._calibration_quality(
        "object_effect", common, common, labels, np.ones(5, dtype=bool),
        bootstrap_samples=200, role="wide-coverage-test",
    )
    interval = quality["nominal_90pct_interval_coverage"]
    assert interval["lcb95"] <= 0.90 <= interval["ucb95"]
    assert interval["absolute_coverage_error_ucb95"] > 0.10
    assert interval["coverage_gate_passed"] is False
    assert quality["fitted_parameter_passed"] is False


@pytest.mark.parametrize("attack", ["artifact", "seed", "lineage", "training_overlap"])
def test_provider_manifest_attacks_fail_closed(attack):
    providers, labels, folds = synthetic_bundle()
    candidate = providers[router.CANDIDATE_PROVIDER]
    manifest = candidate["provider_manifest"]
    if attack == "artifact":
        candidate["provider_artifact_sha256"] = "d" * 64
    elif attack == "seed":
        manifest["members"][1]["seed"] = manifest["members"][0]["seed"]
    elif attack == "lineage":
        manifest["shared_core_lineage_sha256"] = "d" * 64
    else:
        manifest["training_execution_group_ids"] = sorted(
            manifest["training_execution_group_ids"] + [labels["group_id"][0]]
        )
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = router.canonical_sha256(unsigned)
    with pytest.raises(router.DualProviderRouterError):
        run((providers, labels, folds))


def test_integer_prediction_tensor_is_rejected():
    providers, labels, folds = synthetic_bundle()
    providers[router.CANDIDATE_PROVIDER]["success_logit"] = np.zeros_like(
        providers[router.CANDIDATE_PROVIDER]["success_logit"], dtype=np.int64
    )
    with pytest.raises(router.DualProviderRouterError, match="floating prediction tensor"):
        run((providers, labels, folds))


def test_manifest_prediction_tensor_sha_mismatch_is_rejected():
    providers, labels, folds = synthetic_bundle()
    providers[router.CANDIDATE_PROVIDER]["success_logit"][0, 0] += 1e-3
    with pytest.raises(router.DualProviderRouterError, match="prediction tensor SHA mismatch"):
        run((providers, labels, folds))


def test_float32_prediction_manifest_roundtrip_uses_canonical_float64_hash():
    providers, labels, _ = synthetic_bundle()
    candidate = providers[router.CANDIDATE_PROVIDER]
    for key in router.PREDICTION_ARRAY_KEYS:
        candidate[key] = candidate[key].astype(np.float32)
    old = candidate["provider_manifest"]
    candidate["provider_manifest"] = router.build_provider_manifest(
        provider_id=router.CANDIDATE_PROVIDER,
        provider_artifact_sha256=candidate["provider_artifact_sha256"],
        shared_core_lineage_sha256=old["shared_core_lineage_sha256"],
        prediction_tensor_sha256=router.prediction_tensor_set_sha256(candidate),
        training_execution_group_ids=old["training_execution_group_ids"],
        training_semantic_reset_cluster_ids=old["training_semantic_reset_cluster_ids"],
        members=old["members"],
    )
    decoded_labels = router._validate_labels(labels)
    decoded = router._validate_provider(
        candidate, router.CANDIDATE_PROVIDER, decoded_labels
    )
    assert decoded["success_logit"].dtype == np.float64


def test_ndarray_content_address_changes_on_single_value_mutation():
    original = np.zeros((5, 3), dtype=np.float64)
    mutated = original.copy()
    mutated[2, 1] = 1e-9
    assert router._ndarray_sha256(original) != router._ndarray_sha256(mutated)


def test_under_supported_body_actor_context_suppresses_executable_receipt():
    providers, labels, folds = synthetic_bundle()
    rare_cluster = labels["semantic_reset_cluster_id"][0]
    labels["body_id"][labels["semantic_reset_cluster_id"] == rare_cluster] = "rare-body"
    result = run((providers, labels, folds))
    assert result["route_receipt"] is None
    assert result["route_receipt_export_status"] == "not_exported_one_or_more_heads_disabled"


def test_zero_recovery_regress_and_zero_object_observed_disable_without_crash():
    providers, labels, folds = synthetic_bundle()
    labels["regress"][:] = False
    labels["recovery"][:] = 0
    labels["recovery_observed"][:] = False
    labels["object_observed"][:] = False
    for provider in providers.values():
        provider["applicable_masks"]["recovery"][:] = False
        provider["applicable_masks"]["object_effect"][:] = False
    result = run((providers, labels, folds))
    assert result["head_routes"]["recovery"]["selected_provider"] is None
    assert result["head_routes"]["object_effect"]["selected_provider"] is None
    assert result["route_receipt"] is None


def test_identity_fallback_is_locked_before_outer_heldout_and_sha_bound(monkeypatch):
    original_fit = router._fit_parameter

    def forced_bad_success_fit(head, provider, labels, training):
        parameter = original_fit(head, provider, labels, training)
        if head == "success":
            parameter["value"] = 20.0
            parameter["fit_passed"] = True
        return parameter

    monkeypatch.setattr(router, "_fit_parameter", forced_bad_success_fit)
    providers, labels, folds = synthetic_bundle()
    result = run((providers, labels, folds))
    assert result["route_receipt"] is not None
    for fold in result["folds"]:
        for provider in router.PROVIDERS:
            fitted = fold["outer_train_fitted_parameters"][provider]["success"]
            effective = fold["outer_train_effective_parameters"][provider]["success"]
            assert fitted["value"] == 20.0
            assert effective["value"] == 1.0
            assert effective["crossfit_selected_calibration_mode"] == "identity_parameter_fallback"
            assert effective["identity_parameter_sentinel"] == "exact_identity_value_one_selected_by_inner_oof"
    route = result["head_routes"]["success"]
    selected = route["selected_provider"]
    assert selected in router.PROVIDERS
    assert route["deployment_parameter"]["value"] == 1.0
    assert route["deployment_parameter"]["identity_parameter_sentinel"] == (
        "exact_identity_value_one_selected_by_full_development_crossfit"
    )
    descriptor = next(
        row for row in result["provider_descriptors_for_prediction_router"]
        if row["provider_id"] == selected
    )
    assert descriptor["head_calibration_parameter_sha256"]["success"] == route["deployment_parameter_sha256"]
    assert result["route_receipt"]["head_decisions"]["success"][
        "selected_deployment_calibration_parameter_sha256"
    ] == route["deployment_parameter_sha256"]
    heldout = np.isin(labels["group_id"], folds[0]["heldout_group_ids"])
    mutated_labels = copy.deepcopy(labels)
    mutated_labels["post_event"][heldout] = (
        mutated_labels["post_event"][heldout] + 1
    ) % len(router.EVENT_VOCAB)
    mutated_labels["next_event"][heldout] = 1 + (
        mutated_labels["next_event"][heldout] % (len(router.EVENT_VOCAB) - 1)
    )
    mutated_labels["success"][heldout] = 1 - mutated_labels["success"][heldout]
    mutated_labels["recovery"][heldout] = 1 - mutated_labels["recovery"][heldout]
    mutated_labels["duration"][heldout] = 1e100
    mutated_labels["object_target"][heldout] = 1e100
    mutated = run((copy.deepcopy(providers), mutated_labels, copy.deepcopy(folds)))
    assert mutated["folds"][0]["outer_train_effective_parameters_sha256"] == result["folds"][0]["outer_train_effective_parameters_sha256"]
    assert mutated["folds"][0]["route_decision_sha256"] == result["folds"][0]["route_decision_sha256"]


def test_outer_heldout_label_mutation_cannot_change_that_fold_route_or_fit():
    providers, labels, folds = synthetic_bundle()
    original = positive_result()[2]
    heldout = np.isin(labels["group_id"], folds[0]["heldout_group_ids"])
    mutated_labels = copy.deepcopy(labels)
    mutated_labels["success"][heldout] = 1 - mutated_labels["success"][heldout]
    mutated = run((copy.deepcopy(providers), mutated_labels, copy.deepcopy(folds)))
    left, right = original["folds"][0], mutated["folds"][0]
    assert left["inner_oof_selected_route_by_head"] == right["inner_oof_selected_route_by_head"]
    assert left["provider_parameters"] == right["provider_parameters"]
    assert left["route_decision_sha256"] == right["route_decision_sha256"]


def test_repeating_rows_of_one_long_group_does_not_change_group_equal_route():
    providers, labels, folds = synthetic_bundle()
    original = positive_result()[2]
    target_group = "group-0001"
    indices = []
    for index, name in enumerate(labels["group_id"]):
        indices.extend([index] * (9 if name == target_group else 1))
    indices = np.asarray(indices)
    repeated_labels = {key: np.asarray(value)[indices].copy() for key, value in labels.items()}
    target = repeated_labels["group_id"] == target_group
    repeated_labels["group_row_ordinal"][target] = np.arange(int(target.sum()))
    repeated_labels["sample_id"] = np.asarray(
        [f"repeated-{index:04d}" for index in range(len(indices))]
    )
    repeated_providers = {}
    for provider_name, source in providers.items():
        row = {
            "provider_id": provider_name,
            "provider_artifact_sha256": source["provider_artifact_sha256"],
            "provider_manifest": copy.deepcopy(source["provider_manifest"]),
            "member_count": 5,
            "sample_id": repeated_labels["sample_id"].copy(),
            "group_id": repeated_labels["group_id"].copy(),
            "group_row_ordinal": repeated_labels["group_row_ordinal"].copy(),
            "applicable_masks": {
                head: np.asarray(mask)[indices].copy()
                for head, mask in source["applicable_masks"].items()
            },
        }
        for key in (
            "post_event_logits", "next_event_logits", "success_logit", "recovery_logit",
            "duration_log_mean", "duration_log_scale", "object_mean", "object_log_scale",
        ):
            row[key] = np.asarray(source[key])[:, indices].copy()
        source_manifest = source["provider_manifest"]
        row["provider_manifest"] = router.build_provider_manifest(
            provider_id=provider_name,
            provider_artifact_sha256=source["provider_artifact_sha256"],
            shared_core_lineage_sha256=source_manifest["shared_core_lineage_sha256"],
            prediction_tensor_sha256=router.prediction_tensor_set_sha256(row),
            training_execution_group_ids=source_manifest["training_execution_group_ids"],
            training_semantic_reset_cluster_ids=source_manifest["training_semantic_reset_cluster_ids"],
            members=source_manifest["members"],
        )
        repeated_providers[provider_name] = row
    repeated_folds = router.build_five_fold_group_plan(
        repeated_labels["group_id"], repeated_labels["semantic_reset_cluster_id"]
    )
    repeated = run((repeated_providers, repeated_labels, repeated_folds))
    for head in router.HEADS:
        assert repeated["head_routes"][head]["selected_provider"] == original["head_routes"][head]["selected_provider"]
        left = original["head_routes"][head]["paired_provider_proper_loss_gain"]["point"]
        right = repeated["head_routes"][head]["paired_provider_proper_loss_gain"]["point"]
        assert right == pytest.approx(left, abs=1e-10)


def test_duplicating_identical_execution_inside_semantic_cluster_keeps_parameters_and_route():
    providers, labels, _ = synthetic_bundle()
    original = positive_result()[2]
    source_group = "group-0002"
    source_indices = np.flatnonzero(labels["group_id"] == source_group)
    indices = np.concatenate([np.arange(len(labels["sample_id"])), source_indices])
    duplicated_labels = {key: np.asarray(value)[indices].copy() for key, value in labels.items()}
    for identity_key in (
        "sample_id",
        "group_id",
        "semantic_reset_cluster_id",
        "body_id",
        "actor_contract_id",
    ):
        duplicated_labels[identity_key] = duplicated_labels[identity_key].astype(object)
    appended = np.arange(len(labels["sample_id"]), len(indices))
    duplicated_labels["sample_id"][appended] = np.asarray(
        [f"duplicate-execution-{index}" for index in range(len(appended))]
    )
    duplicated_labels["group_id"][appended] = "group-0002-duplicate"
    duplicated_labels["group_row_ordinal"][appended] = np.arange(len(appended))
    duplicated_providers = {}
    for provider_name, source in providers.items():
        row = {
            "provider_id": provider_name,
            "provider_artifact_sha256": source["provider_artifact_sha256"],
            "member_count": 5,
            "sample_id": duplicated_labels["sample_id"].copy(),
            "group_id": duplicated_labels["group_id"].copy(),
            "group_row_ordinal": duplicated_labels["group_row_ordinal"].copy(),
            "applicable_masks": {
                head: np.asarray(mask)[indices].copy()
                for head, mask in source["applicable_masks"].items()
            },
        }
        for key in router.PREDICTION_ARRAY_KEYS:
            row[key] = np.asarray(source[key])[:, indices].copy()
        old = source["provider_manifest"]
        row["provider_manifest"] = router.build_provider_manifest(
            provider_id=provider_name,
            provider_artifact_sha256=source["provider_artifact_sha256"],
            shared_core_lineage_sha256=old["shared_core_lineage_sha256"],
            prediction_tensor_sha256=router.prediction_tensor_set_sha256(row),
            training_execution_group_ids=old["training_execution_group_ids"],
            training_semantic_reset_cluster_ids=old["training_semantic_reset_cluster_ids"],
            members=old["members"],
        )
        duplicated_providers[provider_name] = row
    folds = router.build_five_fold_group_plan(
        duplicated_labels["group_id"], duplicated_labels["semantic_reset_cluster_id"]
    )
    duplicated = run((duplicated_providers, duplicated_labels, folds))
    assert duplicated["full_data_provider_specific_refit_parameters"] == original[
        "full_data_provider_specific_refit_parameters"
    ]
    assert {
        head: duplicated["head_routes"][head]["selected_provider"]
        for head in router.HEADS
    } == {
        head: original["head_routes"][head]["selected_provider"]
        for head in router.HEADS
    }
