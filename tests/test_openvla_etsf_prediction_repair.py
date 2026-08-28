from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openvla_etsf_prediction_repair import (  # noqa: E402
    DURATION_RESIDUAL_PROTOCOL,
    OBJECT_REPAIR_PROTOCOL,
    WEIGHTED_BINARY_CALIBRATION_PROTOCOL,
    FrozenFeatureRecoveryAdapter,
    apply_duration_residual_contract,
    crossfit_duration_residual_contract,
    apply_recorded_weighted_binary_prior_shift,
    fit_duration_residual_contract,
    fit_object_repair_contract,
    object_quality_mask,
    observed_duration_residual_laplace_nll,
    robust_object_student_t_nll,
    recovery_adapter_loss,
    recovery_adapter_training_contract,
)


def _brier(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(np.square(np.asarray(probability) - np.asarray(labels)).mean())


def test_crossfit_prior_shift_repairs_weighted_bce_probability() -> None:
    # At the weighted-BCE optimum, a true p=0.1 becomes a training probability
    # of 0.5 when pos_weight=9.  Fold-local prior correction must recover 0.1.
    folds = np.repeat(np.arange(5), 100)
    labels = np.tile(np.r_[np.ones(10), np.zeros(90)], 5)
    weighted_logit = np.zeros((3, len(labels)), dtype=np.float64)
    result = apply_recorded_weighted_binary_prior_shift(
        weighted_logit,
        folds,
        recorded_positive_weights={fold: 9.0 for fold in range(5)},
    )
    assert result["protocol"] == WEIGHTED_BINARY_CALIBRATION_PROTOCOL
    assert result["labels_used_to_reconstruct_weight"] is False
    assert np.mean(result["probability"]) == pytest.approx(0.1, abs=1e-5)
    assert _brier(labels, result["probability"]) < _brier(
        labels, np.full(len(labels), 0.5)
    )
    assert all(
        row["positive_weight"] == pytest.approx(9.0)
        and row["heldout_labels_used_for_fit"] is False
        and row["other_outer_oof_predictions_used_for_fit"] is False
        for row in result["folds"].values()
    )


def test_recorded_prior_shift_has_no_label_input_or_crossfold_fit() -> None:
    folds = np.repeat(np.arange(5), 40)
    labels = np.tile(np.r_[np.ones(8), np.zeros(32)], 5)
    logits = np.repeat(np.linspace(-2.0, 2.0, len(labels))[None], 3, axis=0)
    first = apply_recorded_weighted_binary_prior_shift(
        logits,
        folds,
        recorded_positive_weights={fold: 4.0 for fold in range(5)},
    )
    second = apply_recorded_weighted_binary_prior_shift(
        logits,
        folds,
        recorded_positive_weights={fold: 4.0 for fold in range(5)},
    )
    assert np.allclose(
        first["probability"][folds == 2],
        second["probability"][folds == 2],
    )


def test_recorded_prior_shift_requires_every_owner_fold_weight() -> None:
    folds = np.repeat(np.arange(5), 20)
    labels = np.tile(np.r_[np.ones(2), np.zeros(18)], 5)
    logits = np.zeros((3, len(labels)))
    with pytest.raises(ValueError, match="cover five owner folds"):
        apply_recorded_weighted_binary_prior_shift(
            logits,
            folds,
            recorded_positive_weights={fold: 9.0 for fold in range(4)},
        )


def test_outer_oof_repair_never_claims_bias_temperature_fit() -> None:
    folds = np.repeat(np.arange(5), 20)
    labels = np.tile(np.r_[np.ones(2), np.zeros(18)], 5)
    result = apply_recorded_weighted_binary_prior_shift(
        np.zeros((3, len(labels))),
        folds,
        recorded_positive_weights={fold: 9.0 for fold in range(5)},
    )
    assert result["other_outer_oof_predictions_used_for_fit"] is False
    assert result["bias_temperature_status"] == (
        "requires_separate_nested_inner_oof_not_available"
    )
    assert all("bias" not in row and "temperature" not in row for row in result["folds"].values())


def test_duration_contract_uses_observed_training_rows_only() -> None:
    duration = np.asarray([4.0, 1000.0, 8.0, 2000.0])
    observed = np.asarray([True, False, True, False])
    event = np.asarray([0, 0, 1, 1])
    body = np.zeros(4, dtype=np.int64)
    contract = fit_duration_residual_contract(duration, observed, event, body)
    assert contract["protocol"] == DURATION_RESIDUAL_PROTOCOL
    assert contract["observed_support"] == 2
    baseline, source = apply_duration_residual_contract(
        contract, np.asarray([0, 1, 9]), np.asarray([0, 0, 0])
    )
    assert baseline[:2] == pytest.approx(np.log1p([4.0, 8.0]))
    assert source.tolist() == ["event_body", "event_body", "body"]


def test_duration_crossfit_target_fold_labels_do_not_leak() -> None:
    folds = np.repeat(np.arange(5), 8)
    observed = np.ones(40, dtype=bool)
    event = np.tile(np.asarray([0, 0, 1, 1, 2, 2, 3, 3]), 5)
    body = np.tile(np.asarray([0, 1, 0, 1, 0, 1, 0, 1]), 5)
    duration = np.arange(1.0, 41.0)
    first = crossfit_duration_residual_contract(
        duration, observed, folds, event, body
    )
    changed = duration.copy()
    changed[folds == 3] += 10_000.0
    second = crossfit_duration_residual_contract(
        changed, observed, folds, event, body
    )
    assert np.array_equal(
        first["baseline_log1p_duration"][folds == 3],
        second["baseline_log1p_duration"][folds == 3],
    )


def test_duration_residual_loss_masks_censored_rows() -> None:
    predicted = torch.tensor([0.0, 0.0], requires_grad=True)
    log_scale = torch.zeros(2, requires_grad=True)
    duration = torch.tensor([9.0, 1_000_000.0])
    observed = torch.tensor([True, False])
    baseline = torch.log1p(torch.tensor([9.0, 1.0]))
    loss = observed_duration_residual_laplace_nll(
        predicted, log_scale, duration, observed, baseline
    )
    changed = observed_duration_residual_laplace_nll(
        predicted,
        log_scale,
        torch.tensor([9.0, 1.0]),
        observed,
        baseline,
    )
    assert loss.item() == pytest.approx(float(np.log(2.0)))
    assert loss.item() == pytest.approx(changed.item())


def test_object_contract_masks_simulation_explosion_and_is_robust() -> None:
    ordinary = np.linspace(-0.2, 0.2, 1000 * 3).reshape(1000, 3)
    training = np.r_[ordinary, [[100.0, -80.0, 40.0]]]
    contract = fit_object_repair_contract(training)
    assert contract["protocol"] == OBJECT_REPAIR_PROTOCOL
    mask = object_quality_mask(training, contract)
    assert not mask[-1]
    assert int(mask[:-1].sum()) >= 995
    assert max(contract["coordinate_robust_scale"]) < 0.2


def test_object_student_t_loss_has_bounded_outlier_influence() -> None:
    prediction = torch.zeros((2, 1), requires_grad=True)
    log_scale = torch.zeros((2, 1), requires_grad=True)
    target = torch.tensor([[1.0], [1_000_000.0]])
    loss = robust_object_student_t_nll(
        prediction,
        log_scale,
        target,
        torch.tensor([True, True]),
        torch.zeros(1),
        torch.ones(1),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    # Student-t location gradients are bounded and the million-unit row cannot
    # recreate the Gaussian NLL/gradient explosion seen in the frozen logs.
    assert float(prediction.grad.abs().max()) < 1.0


def test_recovery_adapter_detaches_shared_transition_features() -> None:
    adapter = FrozenFeatureRecoveryAdapter(4)
    transition = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    mask = torch.ones(6, dtype=torch.bool)
    loss = recovery_adapter_loss(adapter, transition, labels, mask)
    loss.backward()
    assert transition.grad is None
    assert adapter.head.weight.grad is not None
    assert torch.isfinite(adapter.head.weight.grad).all()


def test_recovery_contract_fails_closed_without_both_classes() -> None:
    failed = recovery_adapter_training_contract(
        np.zeros(57), np.ones(57, dtype=bool)
    )
    assert failed["status"] == "fail_closed_insufficient_class_support"
    labels = np.r_[np.ones(57), np.zeros(100)]
    ready = recovery_adapter_training_contract(
        labels, np.ones(len(labels), dtype=bool)
    )
    assert ready["status"] == "trainable"
    assert ready["shared_core_trainable"] is False
