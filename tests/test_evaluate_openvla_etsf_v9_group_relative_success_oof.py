from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_openvla_etsf_v9_group_relative_success_oof as evaluation  # noqa: E402
from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    frozen_tensor_mapping_sha256,
)


def _records(prefix: str, count: int, transition_dim: int = 3) -> list[dict]:
    generator = torch.Generator().manual_seed(1717 + count)
    result = []
    names = [
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
        "continuation_0",
        "continuation_1",
        "continuation_2",
        "continuation_3",
    ]
    for index in range(count):
        winner = (index % 3) + 1
        common = torch.randn(transition_dim, generator=generator) * 2.0
        transition = common.repeat(8, 1)
        labels = torch.zeros(8)
        labels[winner] = 1.0
        for candidate in range(4):
            transition[candidate, 0] += 3.0 if candidate == winner else -1.0
            transition[candidate, 1] += candidate * 0.1
        factual = {
            "transition": transition,
            "duration_selected_log_mean": torch.ones(8),
        }
        result.append(
            {
                "logical_group_key": f"{prefix}|group_{index:03d}",
                "batch": {
                    "terminal_mask": torch.tensor(
                        [True, True, True, True, False, False, False, False]
                    ),
                    "success": labels,
                    "candidate_names": names,
                },
                "factual_outputs": factual,
                "factual_outputs_sha256": frozen_tensor_mapping_sha256(factual),
                "factual_outputs_require_grad": False,
            }
        )
    return result


def test_inner_group_split_is_label_free_deterministic_disjoint_and_complete() -> None:
    groups = [record["logical_group_key"] for record in _records("train", 20)]
    first = evaluation.deterministic_inner_folds(groups, owner_fold_id=2)
    second = evaluation.deterministic_inner_folds(
        list(reversed(groups)), owner_fold_id=2
    )
    assert first == second
    assert sorted(group for fold in first for group in fold) == sorted(groups)
    assert all(len(fold) == 4 for fold in first)
    assert not any(set(first[left]) & set(first[right]) for left in range(5) for right in range(left))


def test_nested_contract_uses_only_outer_training_and_fixed_eight_item_grid() -> None:
    records = _records("train", 20)
    holdout_groups = [f"holdout|group_{index:03d}" for index in range(5)]
    contract, adapter = evaluation.fit_outer_training_contract(
        records,
        owner_fold_id=1,
        outer_holdout_groups=holdout_groups,
        transition_dim=3,
        materialization_sha256="a" * 64,
        train_artifact_sha256="b" * 64,
        train_payload_sha256="c" * 64,
        device="cpu",
    )
    evaluation.validate_fold_contract(contract)
    assert contract["grid_report_count"] == 8
    assert contract["chosen_config_id"] in contract["grid_reports"]
    assert contract[
        "all_hyperparameters_selected_before_outer_holdout_payload_loaded"
    ] is True
    assert contract[
        "outer_holdout_labels_used_for_model_or_hyperparameter_fit"
    ] is False
    assert contract["probability_and_ranking_parameters_disjoint"] is True
    assert contract[
        "same_D250_reuse_cannot_authorize_confirmation_or_deployment"
    ] is True
    assert set(contract["outer_training_groups"]).isdisjoint(holdout_groups)
    assert adapter.config.config_id == contract["chosen_config_id"]
    for report in contract["grid_reports"].values():
        validation = [
            group
            for fold in report["inner_folds"]
            for group in fold["validation_groups"]
        ]
        assert sorted(validation) == contract["outer_training_groups"]
        for fold in report["inner_folds"]:
            assert set(fold["training_groups"]).isdisjoint(
                fold["validation_groups"]
            )


def test_contract_tampering_fails_closed_even_if_outer_signature_is_recomputed() -> None:
    contract, _ = evaluation.fit_outer_training_contract(
        _records("train", 15),
        owner_fold_id=0,
        outer_holdout_groups=[f"held|{index}" for index in range(5)],
        transition_dim=3,
        materialization_sha256="1" * 64,
        train_artifact_sha256="2" * 64,
        train_payload_sha256="3" * 64,
    )
    changed = copy.deepcopy(contract)
    changed["chosen_config_id"] = sorted(changed["grid_reports"])[-1]
    changed["fold_contract_sha256"] = evaluation.canonical_sha256(
        {key: value for key, value in changed.items() if key != "fold_contract_sha256"}
    )
    with pytest.raises(ValueError, match="contract changed"):
        evaluation.validate_fold_contract(changed)


def test_ranking_rule_and_group_bootstrap_measure_task_success_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation, "BOOTSTRAP_SAMPLES", 300)
    groups = [f"g{index:03d}" for index in range(30)]
    labels = np.zeros((30, 4), dtype=np.float64)
    labels[:, 1] = 1.0
    scores = np.zeros((30, 4), dtype=np.float64)
    scores[:, 1] = 3.0
    prediction = {
        "success_label": labels,
        "candidate_ranking_score": scores,
        "success_probability": np.full((30, 4), 0.25),
        "logical_groups": groups,
    }
    metrics = evaluation.ranking_group_metrics(prediction)
    adequacy = evaluation.ranking_group_bootstrap_adequacy(prediction, seed=7)
    assert metrics["selected_success_rate"] == 1.0
    assert metrics["deterministic_candidate_success_rate"] == 0.0
    assert metrics["within_group_pair_accuracy"] == 1.0
    assert adequacy["strict_ranking_adequacy"] is True

    tied = dict(prediction)
    tied["candidate_ranking_score"] = np.zeros((30, 4))
    tied_metrics = evaluation.ranking_group_metrics(tied)
    assert tied_metrics["selected_candidate_counts"] == {
        "0": 30,
        "1": 0,
        "2": 0,
        "3": 0,
    }
    assert tied_metrics["selected_success_rate"] == 0.0


def test_pair_accuracy_uses_equal_logical_group_estimand() -> None:
    # g0 has three discordant pairs and is perfectly ranked; g1 has four and
    # is completely reversed.  Equal-group accuracy is 0.5, whereas silently
    # weighting by pair count would yield 3/7.
    labels = np.asarray([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.float64)
    scores = np.asarray([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=np.float64)
    prediction = {
        "success_label": labels,
        "candidate_ranking_score": scores,
        "success_probability": np.full((2, 4), 0.5),
        "logical_groups": ["g0", "g1"],
    }
    metrics = evaluation.ranking_group_metrics(prediction)
    assert metrics["within_group_pair_accuracy"] == pytest.approx(0.5)
    assert metrics["within_group_pair_accuracy_pair_weighted"] == pytest.approx(3 / 7)
    assert metrics["within_group_pair_estimand"] == "equal_logical_group_mean"


def test_full_run_loads_no_holdout_until_all_five_contracts_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluation, "BOOTSTRAP_SAMPLES", 100)
    manifest_path = tmp_path / "materialization.json"
    manifest_path.write_text("{}", encoding="utf-8")
    folds = []
    training_by_fold = []
    holdout_by_fold = []
    for owner in range(5):
        train_path = tmp_path / f"fold_{owner}_train.pt"
        holdout_path = tmp_path / f"fold_{owner}_holdout.pt"
        train_path.write_bytes(b"train")
        holdout_path.write_bytes(b"holdout")
        training = _records(f"owner{owner}|train", 10)
        holdout = _records(f"owner{owner}|held", 50)
        training_by_fold.append(training)
        holdout_by_fold.append(holdout)
        folds.append(
            {
                "outer_fold_id": owner,
                "train_artifact": str(train_path),
                "train_artifact_sha256": evaluation.sha256_path(train_path),
                "train_payload_sha256": f"{owner + 1:064x}",
                "holdout_artifact": str(holdout_path),
                "holdout_artifact_sha256": evaluation.sha256_path(holdout_path),
                "holdout_payload_sha256": f"{owner + 10:064x}",
                "oof_holdout_groups": [
                    record["logical_group_key"] for record in holdout
                ],
            }
        )
    manifest = {
        "materialization_sha256": "a" * 64,
        "folds": folds,
    }
    fit_count = 0
    trace = []

    monkeypatch.setattr(evaluation, "_load_manifest", lambda path: manifest)

    def fake_load_training(*, input_path, materialization_manifest_path, outer_fold_id):
        trace.append(("train", outer_fold_id))
        return {"owner": outer_fold_id}, {"owner": outer_fold_id}

    def fake_validate(payload):
        owner = payload["owner"]
        groups = [record["logical_group_key"] for record in training_by_fold[owner]]
        return (
            SimpleNamespace(transition_dim=3),
            training_by_fold[owner],
            {"outer_training_groups": groups, "owner": owner},
        )

    def fake_fit(records, **kwargs):
        nonlocal fit_count
        owner = kwargs["owner_fold_id"]
        fit_count += 1
        contract = {
            "fold_contract_sha256": f"{owner + 20:064x}",
            "chosen_config_id": "fixed-synthetic",
            "final_outer_training_audit": {"prevalence": 0.25},
        }
        return contract, object()

    def fake_load_holdout(path, *, owner_fold_id, **kwargs):
        assert fit_count == 5
        trace.append(("holdout", owner_fold_id))
        return {"batches": holdout_by_fold[owner_fold_id]}

    def fake_evaluate(adapter, records, *, device):
        labels = np.stack(
            [record["batch"]["success"][:4].numpy() for record in records]
        )
        probability = np.clip(0.1 + 0.75 * labels, 0.01, 0.99)
        scores = labels * 3.0 + np.arange(4)[None, :] * 1e-3
        prediction = {
            "success_probability": probability,
            "candidate_ranking_score": scores,
            "success_label": labels,
            "logical_groups": [record["logical_group_key"] for record in records],
        }
        flat_probability, flat_labels, row_groups, _ = evaluation._flatten_probability_arrays(
            prediction
        )
        report = {
            "probability": evaluation.binary_probability_metrics(
                flat_labels, flat_probability, row_groups
            ),
            "ranking": evaluation._public_ranking_metrics(
                evaluation.ranking_group_metrics(prediction)
            ),
        }
        return report, prediction

    monkeypatch.setattr(evaluation, "load_authenticated_training_payload", fake_load_training)
    monkeypatch.setattr(evaluation, "validate_v8_training_payload", fake_validate)
    monkeypatch.setattr(evaluation, "fit_outer_training_contract", fake_fit)
    monkeypatch.setattr(evaluation, "validate_fold_contract", lambda contract: None)
    monkeypatch.setattr(evaluation, "_load_holdout", fake_load_holdout)
    monkeypatch.setattr(evaluation, "evaluate_adapter_records", fake_evaluate)
    monkeypatch.setattr(
        evaluation,
        "_cluster_bootstrap_probability_adequacy",
        lambda *args, **kwargs: {"strict_probability_adequacy": True},
    )
    result = evaluation.run_nested_oof(
        materialization_manifest_path=manifest_path, device="cpu"
    )
    assert trace == [("train", owner) for owner in range(5)] + [
        ("holdout", owner) for owner in range(5)
    ]
    assert result["oof_row_count"] == 1000
    assert result[
        "all_outer_contracts_selected_before_any_outer_holdout_deserialized"
    ] is True
    assert result["task_success_improvement_claim_authorized"] is False
    assert result["selector_deployment_authorized"] is False
    assert result["fresh_inputs_accepted"] is False
    assert result["fresh_labels_read"] is False
    unsigned = dict(result)
    recorded = unsigned.pop("result_sha256")
    assert recorded == evaluation.canonical_sha256(unsigned)


def test_scope_paths_and_immutable_output_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        evaluation._reject_scope_path(
            tmp_path / "Fresh50" / "artifact.pt", role="input"
        )
    with pytest.raises(RuntimeError):
        evaluation._reject_scope_path(
            tmp_path / "confirmation_data" / "artifact.pt", role="input"
        )
    output = tmp_path / "result.json"
    evaluation.write_immutable_result(output, {"status": "synthetic"})
    assert json.loads(output.read_text()) == {"status": "synthetic"}
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        evaluation.write_immutable_result(output, {"status": "changed"})
