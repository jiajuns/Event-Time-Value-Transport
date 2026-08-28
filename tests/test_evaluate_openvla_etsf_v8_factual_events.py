from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from evaluate_openvla_etsf_v8_factual_events import (
    HOLDOUT_FORMAT,
    SINGLE_MEMBER_TOTAL_STATUS,
    _records_to_arrays,
    evaluate_factual_event_arrays,
)
from openvla_etsf_counterfactual_oof import canonical_sha256
from train_openvla_etsf_v8_structured_adapters import structured_payload_sha256


def _fixture(*, perfect: bool = True) -> tuple[dict, dict[int, dict]]:
    classes = 3
    groups = []
    folds = []
    current = []
    immediate = []
    destination = []
    structured = []
    observed = []
    uncertainty = []
    for fold_id in range(5):
        for local_group in range(2):
            group = f"task|body|fold{fold_id}_group{local_group}"
            labels = np.asarray([0, 1, 2, (fold_id + local_group) % classes])
            groups.extend([group] * 4)
            folds.extend([fold_id] * 4)
            current.extend([0, 0, 1, 1])
            immediate.extend(labels.tolist())
            destination.extend(np.roll(labels, 1).tolist())
            structured.extend([1, 1, 1, 1])
            observed.extend([1, 1, 0, 1])
            uncertainty.extend([0.05, 0.10, 0.90, 0.20])
    immediate = np.asarray(immediate, dtype=np.int64)
    destination = np.asarray(destination, dtype=np.int64)
    immediate_prediction = immediate if perfect else (immediate + 1) % classes
    destination_prediction = destination if perfect else (destination + 1) % classes

    def logits(prediction: np.ndarray) -> np.ndarray:
        result = np.full((len(prediction), classes), -4.0)
        result[np.arange(len(prediction)), prediction] = 4.0
        return result

    oof = {
        "logical_group": np.asarray(groups),
        "owner_fold_id": np.asarray(folds, dtype=np.int64),
        "current_event_id": np.asarray(current, dtype=np.int64),
        "next_event_id": immediate,
        "next_reached_event_id": destination,
        "structured_mask": np.asarray(structured, dtype=bool),
        "duration_observed": np.asarray(observed, dtype=bool),
        "next_event_logits": logits(immediate_prediction),
        "next_reached_event_logits": logits(destination_prediction),
        "aleatoric_uncertainty": np.asarray(uncertainty, dtype=np.float64),
        # Deliberately ignored: a freeze declaration cannot create accuracy.
        "factual_state_bit_exact": True,
    }
    training = {}
    all_groups = np.asarray(groups)
    all_folds = np.asarray(folds)
    for fold_id in range(5):
        mask = all_folds != fold_id
        training[fold_id] = {
            "logical_group": all_groups[mask],
            "next_event_id": immediate[mask],
            "next_reached_event_id": destination[mask],
            "structured_mask": np.ones(mask.sum(), dtype=bool),
            "duration_observed": np.asarray(observed, dtype=bool)[mask],
        }
    return oof, training


def test_perfect_factual_predictions_are_measured_from_labels() -> None:
    oof, training = _fixture(perfect=True)
    result = evaluate_factual_event_arrays(
        oof, training, bootstrap_samples=40, bootstrap_seed=7
    )
    assert result["status"] == "complete_adaptive_development_only"
    assert result["immediate_event"]["model"]["accuracy"] == 1.0
    assert result["observed_destination_event"]["model"]["accuracy"] == 1.0
    assert result["observed_destination_event"]["support_rows"] == 30
    assert result["immediate_event"]["uncertainty"]["aurc"] == 0.0
    assert result["frozen_factual_state"]["bit_exact_is_accuracy_evidence"] is False
    assert result["uncertainty_scope"]["epistemic_uncertainty_available"] is False
    assert result["authorization"] == {
        "fresh50_confirmation_authorized": False,
        "selector_authorized": False,
        "deployment_authorized": False,
        "policy_success_claim_authorized": False,
    }
    assert len(result["result_sha256"]) == 64


def test_bit_exact_declaration_does_not_turn_wrong_logits_into_accuracy() -> None:
    oof, training = _fixture(perfect=False)
    result = evaluate_factual_event_arrays(
        oof, training, bootstrap_samples=20, bootstrap_seed=9
    )
    assert oof["factual_state_bit_exact"] is True
    assert result["immediate_event"]["model"]["accuracy"] == 0.0
    assert result["observed_destination_event"]["model"]["accuracy"] == 0.0
    assert result["frozen_factual_state"]["accuracy_status"] == (
        "evaluated_not_inferred_from_freeze_hash"
    )


def test_unstructured_and_censored_rows_are_excluded_from_correct_domains() -> None:
    oof, training = _fixture(perfect=True)
    oof["structured_mask"][0] = False
    oof["next_event_logits"][0] = np.asarray([-9.0, 9.0, -9.0])
    oof["next_reached_event_logits"][2] = np.asarray([9.0, -9.0, -9.0])
    result = evaluate_factual_event_arrays(
        oof, training, bootstrap_samples=10, bootstrap_seed=11
    )
    assert result["immediate_event"]["support_rows"] == 39
    assert result["observed_destination_event"]["support_rows"] == 29
    assert result["destination_mask"] == (
        "structured_and_duration_observed_only_no_censored_placeholder"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda oof, training: oof.pop("aleatoric_uncertainty"),
            "need signed aleatoric_uncertainty",
        ),
        (
            lambda oof, training: oof["aleatoric_uncertainty"].__setitem__(0, np.nan),
            "finite and non-negative",
        ),
        (
            lambda oof, training: oof["next_event_id"].__setitem__(0, 99),
            "outside the event vocabulary",
        ),
        (
            lambda oof, training: oof["owner_fold_id"].__setitem__(0, 1),
            "multiple OOF owner folds",
        ),
    ],
)
def test_missing_or_invalid_factual_fields_fail_closed(mutation, match: str) -> None:
    oof, training = _fixture()
    mutation(oof, training)
    with pytest.raises(ValueError, match=match):
        evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)


def test_owner_training_overlap_fails_closed() -> None:
    oof, training = _fixture()
    heldout_group = oof["logical_group"][oof["owner_fold_id"] == 0][0]
    training[0]["logical_group"][0] = heldout_group
    with pytest.raises(ValueError, match="overlaps its holdout"):
        evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)


def test_incomplete_outer_training_complement_fails_closed() -> None:
    oof, training = _fixture()
    keep = training[0]["logical_group"] != training[0]["logical_group"][0]
    for key in list(training[0]):
        training[0][key] = training[0][key][keep]
    with pytest.raises(ValueError, match="exact complement"):
        evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)


def test_total_uncertainty_requires_aleatoric_only_provenance() -> None:
    oof, training = _fixture()
    oof["total_uncertainty"] = oof.pop("aleatoric_uncertainty")
    with pytest.raises(ValueError, match="requires provenance"):
        evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)
    oof["uncertainty_provenance"] = {
        "factual_members": 1,
        "total_uncertainty_semantics": "aleatoric_only_single_factual_member_alias",
        "epistemic_uncertainty_available": False,
    }
    result = evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)
    assert result["uncertainty_scope"]["evaluated_quantity"] == (
        "single_factual_member_composite_aleatoric_score"
    )


def test_missing_destination_support_in_one_fold_fails_closed() -> None:
    oof, training = _fixture()
    oof["duration_observed"][oof["owner_fold_id"] == 3] = False
    with pytest.raises(ValueError, match="lacks observed destination support"):
        evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)


def test_frequency_baseline_is_fit_from_owner_training_not_holdout() -> None:
    oof, training = _fixture()
    first = evaluate_factual_event_arrays(oof, training, bootstrap_samples=5)
    changed = copy.deepcopy(oof)
    changed["next_event_id"][changed["owner_fold_id"] == 0] = 2
    second = evaluate_factual_event_arrays(changed, training, bootstrap_samples=5)
    first_counts = first["immediate_event"]["owner_fold_frequency_baseline"][
        "training_counts_by_owner_fold"
    ]["0"]
    second_counts = second["immediate_event"]["owner_fold_frequency_baseline"][
        "training_counts_by_owner_fold"
    ]["0"]
    assert first_counts == second_counts


def _materialized_holdout_payload(*, include_current: bool) -> dict:
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
    batch = {
        # The deliberate mismatch is regression coverage: dynamic phase and
        # canonical duration clock are distinct labels after event regression.
        "clock_event_id": torch.tensor([0, 2], dtype=torch.long),
        "next_event_id": torch.tensor([1, 0], dtype=torch.long),
        "next_reached_event_id": torch.tensor([2, 1], dtype=torch.long),
        "structured_mask": torch.tensor([True, True]),
        "duration_observed": torch.tensor([True, True]),
    }
    if include_current:
        batch["current_event_id"] = torch.tensor([1, 1], dtype=torch.long)
    payload = {
        "format": HOLDOUT_FORMAT,
        "batches": [
            {
                "split_role": "outer_holdout",
                "outer_fold_id": 0,
                "logical_group_key": "task|body|seed0",
                "total_uncertainty_status": SINGLE_MEMBER_TOTAL_STATUS,
                "batch": batch,
                "factual_outputs": {
                    "next_event_logits": torch.tensor(
                        [[0.0, 2.0, -1.0], [2.0, 0.0, -1.0]], dtype=torch.float32
                    ),
                    "next_reached_event_logits": torch.tensor(
                        [[-1.0, 0.0, 2.0], [-1.0, 2.0, 0.0]], dtype=torch.float32
                    ),
                    "aleatoric_uncertainty": torch.tensor(
                        [0.1, 0.2], dtype=torch.float32
                    ),
                },
            }
        ],
        "provenance": {
            "outer_fold_id": 0,
            "uncertainty_materialization_contract": uncertainty_contract,
            "uncertainty_materialization_contract_sha256": uncertainty_contract[
                "uncertainty_materialization_contract_sha256"
            ],
        },
    }
    payload["payload_sha256"] = structured_payload_sha256(payload)
    return payload


def test_materialized_clock_event_id_cannot_replace_missing_current_event_id() -> None:
    payload = _materialized_holdout_payload(include_current=False)
    with pytest.raises(
        ValueError,
        match="clock_event_id.*cannot replace it.*rematerialization is required",
    ):
        _records_to_arrays(
            payload, fold_id=0, expected_role="outer_holdout", include_logits=True
        )


def test_materialized_dynamic_current_event_id_is_preserved_when_clock_differs() -> None:
    payload = _materialized_holdout_payload(include_current=True)
    arrays = _records_to_arrays(
        payload, fold_id=0, expected_role="outer_holdout", include_logits=True
    )
    assert arrays["current_event_id"].tolist() == [1, 1]
    assert payload["batches"][0]["batch"]["clock_event_id"].tolist() == [0, 2]
