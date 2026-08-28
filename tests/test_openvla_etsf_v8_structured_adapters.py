from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_DURATION_RESIDUAL_MULTIPLIER,
    V8_OBJECT_MODE,
    V8DetachedStructuredAdapters,
    V8StructuredAdapterConfig,
    assert_optimizer_adapter_only,
    compute_v8_adapter_loss,
    frozen_tensor_mapping_sha256,
    module_state_sha256,
    train_v8_adapter_one_step,
    validate_schema5_adapter_batch,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_TRAINING_CHECKPOINT_FORMAT,
    V8_TRAINING_INPUT_FORMAT,
    cpu_one_step_smoke,
    load_authenticated_training_payload,
    train_v8_payload,
    train_v8_payload_lbfgs,
    structured_payload_sha256,
    validate_v8_training_payload,
)
from openvla_etsf_counterfactual_oof import canonical_sha256  # noqa: E402


def _fixture(
    *, count: int = 8, transition_dim: int = 6, object_dim: int = 3
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    factual = {
        "transition": torch.randn(count, transition_dim),
        "duration_selected_log_mean": torch.linspace(0.5, 2.0, count),
        "next_event_logits": torch.randn(count, 5),
        "next_reached_event_logits": torch.randn(count, 5),
        "aleatoric_uncertainty": torch.linspace(0.1, 0.8, count),
    }
    batch = {
        "terminal_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
        "structured_mask": torch.ones(count, dtype=torch.bool),
        "dense_mask": torch.ones(count, dtype=torch.bool),
        "duration": torch.tensor([4, 5, 8, 3, 9, 6, 4, 7], dtype=torch.float32),
        "duration_observed": torch.tensor([1, 1, 0, 1, 0, 1, 1, 0], dtype=torch.bool),
        "success": torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
        "trajectory_regress": torch.tensor([0, 1, 1, 0, 1, 0, 1, 0], dtype=torch.bool),
        "trajectory_recovery": torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.bool),
        "object_delta": torch.zeros(count, object_dim),
        "current_event_id": torch.tensor([0, 0, 1, 1, 4, 0, 1, 2]),
        "next_event_id": torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
        "next_reached_event_id": torch.tensor([1, 2, 3, 4, 0, 1, 2, 3]),
        "body_id": torch.zeros(count, dtype=torch.long),
        "policy_id": torch.zeros(count, dtype=torch.long),
        "group_index": torch.tensor([0, 0, 0, 0, -1, -1, -1, -1]),
        "baseline_mask": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
        "group_keys": ["task|body|1"],
        "candidate_names": [
            "deterministic",
            "sample_blend_0.250",
            "sample_blend_0.500",
            "sample_blend_0.750",
            "continuation_0",
            "continuation_1",
            "continuation_2",
            "continuation_3",
        ],
    }
    baseline = torch.linspace(1.0, 1.2, count)
    fallback = torch.tensor([0.0, 0.1, -0.1])
    return factual, batch, baseline, fallback


def _adapters(transition_dim: int = 6) -> V8DetachedStructuredAdapters:
    return V8DetachedStructuredAdapters(
        V8StructuredAdapterConfig(transition_dim=transition_dim)
    )


def _payload() -> dict:
    factual, batch, baseline, fallback = _fixture()
    uncertainty_contract = {
        "format": "etsf_v8_single_factual_uncertainty_materialization_v1",
        "stored_tensor": "aleatoric_uncertainty",
        "stored_tensor_source": "factual_forward_model_aleatoric_uncertainty",
        "epistemic_uncertainty": "unavailable_requires_frozen_ensemble",
        "total_uncertainty": "unavailable_not_fabricated_fail_closed",
        "allowed_claim": "developmental_single_member_risk_coverage_only",
        "ensemble_total_uncertainty_claim": False,
    }
    uncertainty_contract["uncertainty_materialization_contract_sha256"] = (
        canonical_sha256(uncertainty_contract)
    )
    duration_scale_contract = {
        "format": "etsf_v8_outer_training_duration_laplace_scale_v1",
        "owner_fold_id": 2,
        "fit_scope": "outer_training_observed_only",
        "estimator": "median_absolute_deviation_divided_by_log_2",
        "censored_rows_used": False,
        "model_location": "event_body_median_plus_0.375_frozen_residual",
        "baseline_location": "outer_training_event_body_median",
        "minimum_scale": 1e-4,
        "outer_training_observed_support": 4,
        "model_log_scale": -1.0,
        "baseline_log_scale": -0.5,
    }
    duration_scale_contract["contract_sha256"] = canonical_sha256(
        duration_scale_contract
    )
    payload = {
        "format": V8_TRAINING_INPUT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": 6},
        "batches": [
            {
                "logical_group_key": "task|body|1",
                "split_role": "outer_training",
                "outer_fold_id": 2,
                "group_metadata": {
                    "logical_group_key": "task|body|1",
                    "schema_version": 5,
                    "policy": "openvla",
                    "candidate_names": [
                        "deterministic",
                        "sample_blend_0.250",
                        "sample_blend_0.500",
                        "sample_blend_0.750",
                    ],
                },
                "batch": batch,
                "factual_outputs": factual,
                "factual_outputs_sha256": frozen_tensor_mapping_sha256(factual),
                "factual_outputs_require_grad": False,
                "total_uncertainty_status": (
                    "unavailable_single_forward_has_aleatoric_only_requires_ensemble_fail_closed"
                ),
                "duration_baseline_log1p": baseline,
                "object_fallback": fallback,
                "object_delta_physical": batch["object_delta"].clone(),
            }
        ],
        "provenance": {
            "base_checkpoint_sha256": "a" * 64,
            "outer_training_groups_sha256": "b" * 64,
            "label_derivation_sha256": "c" * 64,
            "duration_baseline_contract_sha256": "d" * 64,
            "duration_laplace_scale_contract_sha256": duration_scale_contract[
                "contract_sha256"
            ],
            "duration_laplace_scale_contract": duration_scale_contract,
            "object_fallback_contract_sha256": "e" * 64,
            "uncertainty_materialization_contract_sha256": uncertainty_contract[
                "uncertainty_materialization_contract_sha256"
            ],
            "outer_fold_id": 2,
            "target_outer_fold_labels_used": False,
            "factual_outputs_frozen": True,
            "object_mode": V8_OBJECT_MODE,
            "object_pose_quality_status": (
                "unavailable_schema5_collector_has_no_quality_field_fail_closed"
            ),
            "base_target_outer_fold_exclusion_status": "unproven_development_only",
            "uncertainty_materialization_contract": uncertainty_contract,
        },
    }
    payload["payload_sha256"] = structured_payload_sha256(payload)
    return payload


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v8_config_freezes_duration_and_object_contracts() -> None:
    config = V8StructuredAdapterConfig(transition_dim=6)
    assert config.duration_residual_multiplier == V8_DURATION_RESIDUAL_MULTIPLIER
    assert config.object_mode == V8_OBJECT_MODE
    with pytest.raises(ValueError, match="frozen at 0.375"):
        V8StructuredAdapterConfig(
            transition_dim=6, duration_residual_multiplier=0.5
        )
    with pytest.raises(ValueError, match="robust/zero fallback"):
        V8StructuredAdapterConfig(transition_dim=6, object_mode="learned")


def test_v8_forward_repairs_duration_and_never_learns_object() -> None:
    factual, _, baseline, fallback = _fixture()
    adapters = _adapters()
    output = adapters(
        factual,
        duration_baseline_log1p=baseline,
        object_fallback=fallback,
    )
    expected_duration = baseline + 0.375 * (
        factual["duration_selected_log_mean"] - baseline
    )
    assert torch.allclose(output["duration_repaired_log1p_mean"], expected_duration)
    assert torch.equal(
        output["object_delta_point"], fallback[None].expand(len(baseline), -1)
    )
    assert output["learned_object_output_authorized"] is False
    assert output["object_prediction_status"] == V8_OBJECT_MODE
    assert torch.allclose(
        output["failure_probability"] + output["success_probability"],
        torch.ones(len(baseline)),
    )
    assert torch.allclose(
        output["recovery_probability"],
        output["regress_probability"]
        * output["recovery_given_regress_probability"],
    )
    assert set(adapters.trainable_parameter_names()) == {
        "success_head.weight",
        "success_head.bias",
        "regress_head.weight",
        "regress_head.bias",
        "recovery_given_regress_head.weight",
        "recovery_given_regress_head.bias",
    }
    assert "duration" not in " ".join(adapters.trainable_parameter_names())
    assert "object" not in " ".join(adapters.trainable_parameter_names())


def test_v8_losses_are_unweighted_and_recovery_is_conditional() -> None:
    factual, batch, baseline, fallback = _fixture()
    adapters = _adapters()
    with torch.no_grad():
        for parameter in adapters.parameters():
            parameter.zero_()
    total, losses, diagnostics = compute_v8_adapter_loss(
        adapters,
        factual,
        batch,
        duration_baseline_log1p=baseline,
        object_fallback=fallback,
    )
    assert losses.keys() == {
        "success_unweighted_bce",
        "regress_unweighted_bce",
        "recovery_given_regress_unweighted_bce",
    }
    assert all(
        float(value.detach()) == pytest.approx(math.log(2.0))
        for value in losses.values()
    )
    assert float(total.detach()) == pytest.approx(3.0 * math.log(2.0))
    assert diagnostics["terminal"] == 4
    assert diagnostics["structured"] == 8
    assert diagnostics["conditional_recovery_support"] == 4
    assert diagnostics["conditional_recovery_positive"] == 2
    assert diagnostics["duration_is_fixed_not_an_optimization_loss"] is True
    assert diagnostics["object_is_fallback_not_an_optimization_loss"] is True


def test_schema5_continuation_success_placeholders_are_terminal_masked() -> None:
    factual, batch, baseline, fallback = _fixture()
    adapters = _adapters()
    with torch.no_grad():
        adapters.success_head.weight.zero_()
        adapters.success_head.bias.fill_(1.5)
    first, first_losses, _ = compute_v8_adapter_loss(
        adapters,
        factual,
        batch,
        duration_baseline_log1p=baseline,
        object_fallback=fallback,
    )
    changed = dict(batch)
    changed["success"] = batch["success"].clone()
    changed["success"][-2:] = 1.0
    second, second_losses, _ = compute_v8_adapter_loss(
        adapters,
        factual,
        changed,
        duration_baseline_log1p=baseline,
        object_fallback=fallback,
    )
    assert torch.equal(
        first_losses["success_unweighted_bce"],
        second_losses["success_unweighted_bce"],
    )
    assert torch.equal(first, second)


def test_schema5_recovery_without_regression_fails_closed() -> None:
    _, batch, _, _ = _fixture()
    invalid = dict(batch)
    invalid["trajectory_recovery"] = batch["trajectory_recovery"].clone()
    invalid["trajectory_recovery"][0] = True
    with pytest.raises(ValueError, match="requires trajectory_regress"):
        validate_schema5_adapter_batch(invalid)

    invalid_regress = dict(batch)
    invalid_regress["structured_mask"] = batch["structured_mask"].clone()
    invalid_regress["structured_mask"][1] = False
    with pytest.raises(ValueError, match="trajectory_regress requires structured"):
        validate_schema5_adapter_batch(invalid_regress)


def test_optimizer_scope_rejects_any_factual_or_extra_parameter() -> None:
    adapters = _adapters()
    extra = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([*adapters.parameters(), extra], lr=0.1)
    with pytest.raises(RuntimeError, match="parameters must equal"):
        assert_optimizer_adapter_only(optimizer, adapters)


def test_one_step_preserves_factual_tensor_hash_and_gradient_boundary() -> None:
    factual, batch, baseline, fallback = _fixture()
    factual["transition"].requires_grad_(True)
    factual["duration_selected_log_mean"].requires_grad_(True)
    input_hash = frozen_tensor_mapping_sha256(factual)
    adapters = _adapters()
    adapters.initialize_probability_biases(
        success_prevalence=2 / 6,
        regress_prevalence=4 / 8,
        recovery_given_regress_prevalence=2 / 4,
    )
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=1e-2)
    before = module_state_sha256(adapters)
    report = train_v8_adapter_one_step(
        adapters,
        optimizer,
        factual,
        batch,
        duration_baseline_log1p=baseline,
        object_fallback=fallback,
    )
    assert report["factual_input_sha256_before"] == report["factual_input_sha256_after"]
    assert frozen_tensor_mapping_sha256(factual) == input_hash
    assert factual["transition"].grad is None
    assert factual["duration_selected_log_mean"].grad is None
    assert report["adapter_state_sha256_before"] == before
    assert report["adapter_parameters_changed"] is True
    assert report["gradient_clip_scope"] == "independent_per_probability_head"


def test_training_payload_requires_outer_fold_provenance() -> None:
    payload = _payload()
    config, records, provenance = validate_v8_training_payload(payload)
    assert config.transition_dim == 6
    assert len(records) == 1
    assert provenance["outer_fold_id"] == 2
    invalid = _payload()
    invalid["provenance"]["target_outer_fold_labels_used"] = True
    invalid["payload_sha256"] = structured_payload_sha256(invalid)
    with pytest.raises(ValueError, match="exclude target outer-fold labels"):
        validate_v8_training_payload(invalid)
    invalid_sha = _payload()
    invalid_sha["provenance"]["base_checkpoint_sha256"] = "unknown"
    invalid_sha["payload_sha256"] = structured_payload_sha256(invalid_sha)
    with pytest.raises(ValueError, match="SHA fields invalid"):
        validate_v8_training_payload(invalid_sha)


def test_training_payload_requires_authenticated_dynamic_current_event_label() -> None:
    missing = _payload()
    missing["batches"][0]["batch"].pop("current_event_id")
    missing["payload_sha256"] = structured_payload_sha256(missing)
    with pytest.raises(ValueError, match="current_event_id is invalid"):
        validate_v8_training_payload(missing)

    outside_vocabulary = _payload()
    outside_vocabulary["batches"][0]["batch"]["current_event_id"][0] = 5
    outside_vocabulary["payload_sha256"] = structured_payload_sha256(
        outside_vocabulary
    )
    with pytest.raises(ValueError, match="current_event_id is invalid"):
        validate_v8_training_payload(outside_vocabulary)


def test_training_payload_verifies_each_materialized_factual_tensor_hash() -> None:
    tampered = _payload()
    tampered["batches"][0]["factual_outputs"]["transition"][0, 0] += 1.0
    tampered["payload_sha256"] = structured_payload_sha256(tampered)
    with pytest.raises(ValueError, match="factual_outputs_sha256 mismatch"):
        validate_v8_training_payload(tampered)
    requires_grad = _payload()
    requires_grad["batches"][0]["factual_outputs_require_grad"] = True
    requires_grad["payload_sha256"] = structured_payload_sha256(requires_grad)
    with pytest.raises(ValueError, match="explicitly gradient-free"):
        validate_v8_training_payload(requires_grad)


def test_train_payload_serializes_only_adapters_and_safe_contract() -> None:
    checkpoint = train_v8_payload(_payload(), epochs=1, device="cpu")
    assert checkpoint["format"] == V8_TRAINING_CHECKPOINT_FORMAT
    assert checkpoint["training_contract"]["shared_core_trainable"] is False
    assert checkpoint["training_contract"]["duration_trainable"] is False
    assert checkpoint["training_contract"]["object_trainable"] is False
    assert checkpoint["training_contract"]["object_mode"] == V8_OBJECT_MODE
    assert checkpoint["all_steps_factual_inputs_bit_exact"] is True
    assert checkpoint["strict_oof_base_exclusion_eligible"] is False
    assert len(checkpoint["frozen_input_sha256_by_batch"]) == 1
    assert len(checkpoint["frozen_input_aggregate_sha256"]) == 64
    assert checkpoint["fresh_confirmation_data_or_labels_read"] is False
    assert checkpoint["authorization_guard_changed"] is False
    assert set(checkpoint["state_dict"]) == {
        "duration_residual_multiplier",
        "success_head.weight",
        "success_head.bias",
        "regress_head.weight",
        "regress_head.bias",
        "recovery_given_regress_head.weight",
        "recovery_given_regress_head.bias",
    }


def test_full_batch_lbfgs_fits_each_probability_head_independently() -> None:
    payload = _payload()
    before = payload["batches"][0]["factual_outputs_sha256"]
    checkpoint = train_v8_payload_lbfgs(
        payload,
        max_iter=50,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        device="cpu",
    )
    assert checkpoint["optimizer"]["name"] == "independent_full_batch_LBFGS"
    assert checkpoint["optimizer"]["all_heads_finite_improved"] is True
    assert checkpoint["all_steps_factual_inputs_bit_exact"] is True
    assert payload["batches"][0]["factual_outputs_sha256"] == before
    reports = checkpoint["optimizer"]["head_reports"]
    assert set(reports) == {"success", "regress", "recovery_given_regress"}
    for report in reports.values():
        assert report["final_loss"] <= report["initial_loss"]
        assert math.isfinite(report["maximum_absolute_gradient"])
        assert report["closure_calls"] >= 1
    assert checkpoint["training_contract"]["shared_core_trainable"] is False
    assert checkpoint["training_contract"]["optimizer_parameter_scope"] == (
        "one_probability_head_at_a_time_exactly"
    )


def test_trainer_authenticates_complete_signed_bundle_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    rows = []
    for fold_id in range(5):
        payload = _payload()
        payload["provenance"]["outer_fold_id"] = fold_id
        payload["batches"][0]["outer_fold_id"] = fold_id
        payload["payload_sha256"] = structured_payload_sha256(payload)
        train_path = tmp_path / f"fold_{fold_id}_train.pt"
        holdout_path = tmp_path / f"fold_{fold_id}_holdout.pt"
        torch.save(payload, train_path)
        torch.save({"fold": fold_id}, holdout_path)
        rows.append(
            {
                "outer_fold_id": fold_id,
                "training_groups_sha256": payload["provenance"][
                    "outer_training_groups_sha256"
                ],
                "train_artifact": str(train_path.resolve()),
                "train_artifact_sha256": _file_sha(train_path),
                "train_payload_sha256": payload["payload_sha256"],
                "holdout_artifact": str(holdout_path.resolve()),
                "holdout_artifact_sha256": _file_sha(holdout_path),
                "holdout_payload_sha256": "f" * 64,
            }
        )
    manifest = {
        "format": "etsf_v8_oof_materialization_manifest_v1",
        "status": "complete_development_only",
        "prospective_claim_for_v8": False,
        "folds": rows,
    }
    manifest["materialization_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload, audit = load_authenticated_training_payload(
        input_path=tmp_path / "fold_2_train.pt",
        materialization_manifest_path=manifest_path,
        outer_fold_id=2,
    )
    assert payload["provenance"]["outer_fold_id"] == 2
    assert audit["status"].startswith("authenticated_complete_five_fold")
    with (tmp_path / "fold_4_holdout.pt").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(RuntimeError, match="holdout artifact SHA changed"):
        load_authenticated_training_payload(
            input_path=tmp_path / "fold_2_train.pt",
            materialization_manifest_path=manifest_path,
            outer_fold_id=2,
        )


def test_cpu_one_step_smoke_is_frozen_and_gpu_free() -> None:
    report = cpu_one_step_smoke(seed=91)
    assert report["status"] == "passed"
    assert report["device"] == "cpu"
    assert report["cuda_used"] is False
    assert report["factual_state_sha256_before"] == report["factual_state_sha256_after"]
    assert report["factual_input_sha256_before"] == report["factual_input_sha256_after"]
    assert report["adapter_parameters_changed"] is True
    assert report["learned_object_output_authorized"] is False
