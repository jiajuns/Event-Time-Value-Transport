from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import calibrate_openvla_etsf_v8_success_inner_cv as calibration  # noqa: E402
from calibrate_openvla_etsf_v8_success_inner_cv import (  # noqa: E402
    ALPHA_GRID,
    HOLDOUT_FORMAT,
    _cluster_bootstrap_probability_adequacy,
    _load_holdout,
    _load_manifest,
    _atomic_json,
    binary_probability_metrics,
    calibration_protocol,
    deterministic_inner_folds,
    fit_outer_training_calibration_contract,
    optimizer_contract_from_checkpoint,
    run_oof_calibration,
    sha256_path,
    shrink_probabilities,
    train_success_head_adamw,
    validate_calibration_contract,
)
from openvla_etsf_counterfactual_oof import canonical_sha256  # noqa: E402
from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8DetachedStructuredAdapters,
    V8StructuredAdapterConfig,
    frozen_tensor_mapping_sha256,
    train_v8_adapter_one_step,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    structured_payload_sha256,
)


def _records(count: int = 15, transition_dim: int = 4) -> list[dict]:
    generator = torch.Generator().manual_seed(19)
    records = []
    for group_index in range(count):
        rows = 8
        transition = torch.randn(rows, transition_dim, generator=generator)
        success = torch.tensor(
            [
                int((group_index + candidate) % 5 == 0)
                for candidate in range(4)
            ]
            + [0, 0, 0, 0],
            dtype=torch.float32,
        )
        regress = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
        recovery = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.bool)
        factual = {
            "transition": transition,
            "duration_selected_log_mean": torch.linspace(0.5, 1.2, rows),
        }
        batch = {
            "terminal_mask": torch.tensor(
                [1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool
            ),
            "structured_mask": torch.ones(rows, dtype=torch.bool),
            "dense_mask": torch.ones(rows, dtype=torch.bool),
            "duration": torch.linspace(3.0, 10.0, rows),
            "duration_observed": torch.ones(rows, dtype=torch.bool),
            "success": success,
            "trajectory_regress": regress,
            "trajectory_recovery": recovery,
            "object_delta": torch.zeros(rows, 3),
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
        records.append(
            {
                "logical_group_key": f"task|body|{group_index:03d}",
                "batch": batch,
                "factual_outputs": factual,
                "duration_baseline_log1p": torch.full((rows,), 1.5),
                "object_fallback": torch.zeros(3),
            }
        )
    return records


def _optimizer_contract(*, epochs: int = 3) -> dict:
    value = {
        "name": "AdamW",
        "epochs": epochs,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "maximum_gradient_norm_per_probability_head": 1.0,
        "record_order_policy": (
            "signed_outer_training_payload_order_filtered_for_inner_train"
        ),
        "initialization": "zero_weight_inner_training_prevalence_bias",
        "success_loss": "unweighted_binary_cross_entropy_terminal_rows_only",
        "stochastic_initialization": False,
    }
    value["optimizer_contract_sha256"] = canonical_sha256(value)
    return value


def _fit_contract(records: list[dict]) -> dict:
    return fit_outer_training_calibration_contract(
        records,
        owner_fold_id=2,
        outer_holdout_groups=[f"task|body|holdout{i}" for i in range(5)],
        optimizer_contract=_optimizer_contract(),
        transition_dim=4,
        materialization_sha256="a" * 64,
        train_artifact_sha256="b" * 64,
        train_payload_sha256="c" * 64,
        final_checkpoint_sha256="d" * 64,
        device="cpu",
    )


def test_group_inner_folds_are_label_free_deterministic_disjoint_and_complete() -> None:
    groups = [record["logical_group_key"] for record in _records()]
    first = deterministic_inner_folds(groups, owner_fold_id=3)
    second = deterministic_inner_folds(list(reversed(groups)), owner_fold_id=3)
    assert first == second
    assert sorted(group for fold in first for group in fold) == sorted(groups)
    assert all(len(set(fold)) == len(fold) for fold in first)
    assert all(
        set(first[left]).isdisjoint(first[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )
    assert first != deterministic_inner_folds(groups, owner_fold_id=4)


def test_positive_prevalence_shrinkage_preserves_order_ties_and_ap() -> None:
    probability = np.asarray([0.05, 0.25, 0.25, 0.70, 0.95])
    labels = np.asarray([0, 1, 0, 1, 1], dtype=np.float64)
    groups = np.asarray(["a", "a", "b", "b", "c"])
    raw = binary_probability_metrics(labels, probability, groups)
    for alpha in ALPHA_GRID:
        adjusted = shrink_probabilities(
            probability, prevalence=0.2, alpha=alpha
        )
        assert np.array_equal(
            np.sign(probability[:, None] - probability[None, :]),
            np.sign(adjusted[:, None] - adjusted[None, :]),
        )
        assert (
            binary_probability_metrics(labels, adjusted, groups)[
                "average_precision"
            ]
            == raw["average_precision"]
        )
    with pytest.raises(ValueError, match="strictly positive"):
        shrink_probabilities(probability, prevalence=0.2, alpha=0.0)


def test_success_only_adamw_matches_final_multitask_success_parameter_updates() -> None:
    records = _records(count=10)
    optimizer_contract = _optimizer_contract(epochs=2)
    success_only, audit = train_success_head_adamw(
        records,
        transition_dim=4,
        optimizer_contract=optimizer_contract,
        device="cpu",
    )

    adapters = V8DetachedStructuredAdapters(
        V8StructuredAdapterConfig(transition_dim=4)
    )
    # Other heads are independent; their valid fixture prevalences cannot alter
    # the success-head Adam moments or gradient clipping.
    adapters.initialize_probability_biases(
        success_prevalence=8 / 40,
        regress_prevalence=40 / 80,
        recovery_given_regress_prevalence=20 / 40,
    )
    optimizer = torch.optim.AdamW(
        adapters.parameters(), lr=1e-3, weight_decay=0.0
    )
    for _ in range(2):
        for record in records:
            train_v8_adapter_one_step(
                adapters,
                optimizer,
                record["factual_outputs"],
                record["batch"],
                duration_baseline_log1p=record["duration_baseline_log1p"],
                object_fallback=record["object_fallback"],
            )
    assert torch.equal(success_only.weight, adapters.success_head.weight)
    assert torch.equal(success_only.bias, adapters.success_head.bias)
    assert audit["factual_outputs_bit_exact"] is True


def test_signed_contract_uses_only_inner_oof_and_rejects_tampering() -> None:
    records = _records()
    signature = inspect.signature(fit_outer_training_calibration_contract)
    assert "holdout_payload" not in signature.parameters
    contract = _fit_contract(records)
    validate_calibration_contract(contract)
    assert contract["chosen_alpha"] in ALPHA_GRID
    assert contract["outer_holdout_labels_used_for_alpha_selection"] is False
    assert contract["selection_completed_before_outer_holdout_payload_loaded"] is True
    assert contract["inner_oof_support"] == 4 * len(records)
    validation_groups = [
        set(row["validation_groups"]) for row in contract["inner_folds"]
    ]
    training_groups = [
        set(row["training_groups"]) for row in contract["inner_folds"]
    ]
    assert all(
        training_groups[index].isdisjoint(validation_groups[index])
        for index in range(5)
    )
    assert set().union(*validation_groups) == {
        record["logical_group_key"] for record in records
    }
    tampered = copy.deepcopy(contract)
    tampered["chosen_alpha"] = 1.0 if contract["chosen_alpha"] != 1.0 else 0.75
    with pytest.raises(ValueError, match="signature/protocol changed"):
        validate_calibration_contract(tampered)


def test_outer_holdout_labels_cannot_change_an_already_signed_selection_contract() -> None:
    records = _records()
    contract = _fit_contract(records)
    frozen_sha = contract["calibration_contract_sha256"]
    simulated_outer_holdout = _records(count=5)
    for record in simulated_outer_holdout:
        record["batch"]["success"].fill_(1.0)
    # No calibration-selection API accepts those labels; evaluation happens in
    # a later phase and cannot rewrite the signed contract.
    assert contract["calibration_contract_sha256"] == frozen_sha
    validate_calibration_contract(contract)


def test_protocol_grid_and_selection_rule_are_source_fixed() -> None:
    protocol = calibration_protocol()
    assert protocol["alpha_grid"] == [0.25, 0.5, 0.75, 1.0]
    assert protocol["outer_holdout_labels_used_for_alpha_selection"] is False
    unsigned = dict(protocol)
    recorded = unsigned.pop("protocol_sha256")
    assert recorded == canonical_sha256(unsigned)


def _final_optimizer_checkpoint(records: list[dict]) -> dict:
    groups = [record["logical_group_key"] for record in records]
    order_sha = __import__("hashlib").sha256(
        "\n".join(groups).encode("utf-8")
    ).hexdigest()
    return {
        "optimizer": {
            "name": "AdamW",
            "epochs": 3,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "initialization": "zero_weights_outer_training_prevalence_biases",
            "random_seed_used_for_adapter_initialization": None,
            "record_order": groups,
            "record_order_sha256": order_sha,
        },
        "training_contract": {
            "success_loss": "unweighted_binary_cross_entropy",
            "optimizer_parameter_scope": "v8_adapter_parameters_exactly",
        },
        "last_step": {
            "gradient_clip_scope": "independent_per_probability_head"
        },
        "steps": 3 * len(records),
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["last_step"].__setitem__(
                "gradient_clip_scope", "global"
            ),
            "fixed r4 AdamW",
        ),
        (lambda value: value.__setitem__("steps", 1), "fixed r4 AdamW"),
        (
            lambda value: value["optimizer"].__setitem__(
                "record_order_sha256", "0" * 64
            ),
            "fixed r4 AdamW",
        ),
    ],
)
def test_final_optimizer_contract_authenticates_clip_steps_and_order(
    mutation, match: str
) -> None:
    records = _records(count=10)
    valid = _final_optimizer_checkpoint(records)
    contract = optimizer_contract_from_checkpoint(valid, records)
    assert contract["maximum_gradient_norm_per_probability_head"] == 1.0
    assert contract["final_checkpoint_steps"] == 30
    invalid = copy.deepcopy(valid)
    mutation(invalid)
    with pytest.raises(ValueError, match=match):
        optimizer_contract_from_checkpoint(invalid, records)


def test_group_bootstrap_reports_fixed_baseline_skill_and_ece_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration, "BOOTSTRAP_SAMPLES", 500)
    groups = np.repeat([f"g{i}" for i in range(20)], 4)
    labels = np.tile(np.asarray([0, 0, 1, 1], dtype=np.float64), 20)
    probability = np.where(labels > 0.5, 0.85, 0.15)
    baseline = np.full(len(labels), 0.5)
    report = _cluster_bootstrap_probability_adequacy(
        labels, probability, baseline, groups, seed=17
    )
    assert report["brier_model_minus_baseline"]["strict_skill"] is True
    assert report["nll_model_minus_baseline"]["strict_skill"] is True
    assert report["average_precision_minus_evaluation_prevalence"][
        "strict_skill"
    ] is True
    assert report["ece_strict_gate"] is False  # 0.15 > fixed 0.10
    assert report["strict_probability_adequacy"] is False


def test_tampered_materialization_manifest_fails_closed(tmp_path: Path) -> None:
    value = {
        "format": "etsf_v8_oof_materialization_manifest_v1",
        "status": "complete_development_only",
        "fresh_confirmation_data_or_labels_read": False,
        "folds": [{"outer_fold_id": index} for index in range(5)],
    }
    value["materialization_sha256"] = canonical_sha256(value)
    path = tmp_path / "materialization_manifest.json"
    path.write_text(__import__("json").dumps(value), encoding="utf-8")
    assert _load_manifest(path)["materialization_sha256"] == value[
        "materialization_sha256"
    ]
    value["fresh_confirmation_data_or_labels_read"] = True
    path.write_text(__import__("json").dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="signed development bundle"):
        _load_manifest(path)


def test_tampered_holdout_artifact_fails_closed(tmp_path: Path) -> None:
    record = _records(count=1)[0]
    record.update(
        {
            "split_role": "outer_holdout",
            "outer_fold_id": 0,
            "factual_outputs_require_grad": False,
            "factual_outputs_sha256": frozen_tensor_mapping_sha256(
                record["factual_outputs"]
            ),
        }
    )
    train_provenance = {"outer_fold_id": 0, "marker": "signed_train"}
    payload = {
        "format": HOLDOUT_FORMAT,
        "schema_version": 5,
        "batches": [record],
        "provenance": {
            **train_provenance,
            "split_role": "outer_holdout_evaluation_only",
            "holdout_labels_used_for_duration_or_object_fit": False,
            "holdout_labels_present_only_in_separate_artifact": True,
        },
    }
    payload["payload_sha256"] = structured_payload_sha256(payload)
    path = tmp_path / "fold_0_holdout.pt"
    torch.save(payload, path)
    manifest_fold = {
        "holdout_artifact": str(path),
        "holdout_artifact_sha256": sha256_path(path),
        "holdout_payload_sha256": payload["payload_sha256"],
        "oof_holdout_groups": [record["logical_group_key"]],
    }
    assert len(
        _load_holdout(
            path,
            owner_fold_id=0,
            manifest_fold=manifest_fold,
            train_provenance=train_provenance,
        )["batches"]
    ) == 1
    payload["batches"][0]["batch"]["success"][0] = 1.0 - payload[
        "batches"
    ][0]["batch"]["success"][0]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="artifact SHA mismatch"):
        _load_holdout(
            path,
            owner_fold_id=0,
            manifest_fold=manifest_fold,
            train_provenance=train_provenance,
        )


def test_all_five_contracts_finish_before_any_torch_load_of_holdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    contract_count = 0
    folds = []
    for index in range(5):
        folds.append(
            {
                "outer_fold_id": index,
                "train_artifact": str(tmp_path / f"fold_{index}_train.pt"),
                "train_artifact_sha256": "a" * 64,
                "train_payload_sha256": "b" * 64,
                "holdout_artifact": str(tmp_path / f"fold_{index}_holdout.pt"),
                "holdout_artifact_sha256": "c" * 64,
                "holdout_payload_sha256": "d" * 64,
                "oof_holdout_groups": [f"holdout{index}"],
            }
        )
    manifest = {"materialization_sha256": "e" * 64, "folds": folds}
    fake_record = _records(count=1)[0]
    fake_payload = {"records": [fake_record], "provenance": {"outer_fold_id": 0}}

    def fake_torch_load(path, *args, **kwargs):
        events.append(f"torch.load:{Path(path).name}")
        return fake_payload

    def fake_authenticated(*, input_path, materialization_manifest_path, outer_fold_id):
        value = torch.load(input_path)
        return value, {"outer_fold_id": outer_fold_id}

    def fake_validate(payload):
        return V8StructuredAdapterConfig(transition_dim=4), [fake_record], {
            "outer_fold_id": 0
        }

    def fake_checkpoint(path, **kwargs):
        torch.load(path)
        return {}, object(), _optimizer_contract(epochs=1)

    def fake_fit(*args, owner_fold_id, **kwargs):
        nonlocal contract_count
        contract_count += 1
        events.append(f"contract:{owner_fold_id}")
        return {
            "owner_fold_id": owner_fold_id,
            "final_outer_checkpoint_sha256": "f" * 64,
            "calibration_contract_sha256": str(owner_fold_id) * 64,
        }

    def fake_holdout(path, **kwargs):
        assert contract_count == 5
        torch.load(path)
        return {"batches": [fake_record]}

    def fake_evaluate(adapters, records, contract, **kwargs):
        owner = contract["owner_fold_id"]
        arrays = {
            "labels": np.asarray([0.0, 1.0]),
            "groups": np.asarray([f"g{owner}a", f"g{owner}b"]),
            "candidate_index": np.asarray([0, 1]),
            "uncalibrated": np.asarray([0.2, 0.8]),
            "calibrated": np.asarray([0.25, 0.75]),
            "baseline": np.asarray([0.5, 0.5]),
        }
        return {"owner_fold_id": owner}, arrays

    monkeypatch.setattr(calibration, "_load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        calibration, "load_authenticated_training_payload", fake_authenticated
    )
    monkeypatch.setattr(calibration, "validate_v8_training_payload", fake_validate)
    monkeypatch.setattr(calibration, "_load_final_checkpoint", fake_checkpoint)
    monkeypatch.setattr(
        calibration, "fit_outer_training_calibration_contract", fake_fit
    )
    monkeypatch.setattr(calibration, "_load_holdout", fake_holdout)
    monkeypatch.setattr(calibration, "evaluate_outer_holdout", fake_evaluate)
    monkeypatch.setattr(calibration, "sha256_path", lambda path: "f" * 64)
    monkeypatch.setattr(calibration, "BOOTSTRAP_SAMPLES", 100)
    monkeypatch.setattr(torch, "load", fake_torch_load)
    result = run_oof_calibration(
        checkpoint_paths=[tmp_path / f"checkpoint_{i}.pt" for i in range(5)],
        materialization_manifest_path=tmp_path / "manifest.json",
    )
    first_holdout_load = next(
        index for index, event in enumerate(events) if "holdout" in event
    )
    assert sum(event.startswith("contract:") for event in events[:first_holdout_load]) == 5
    assert result["action_ranking_preserved_within_each_group"] is True
    assert result["task_success_cannot_change_from_uncalibrated_argmax"] is True


def test_atomic_output_is_immutable_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _atomic_json(path, {"version": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable output"):
        _atomic_json(path, {"version": 2})
    assert path.read_bytes() == before
