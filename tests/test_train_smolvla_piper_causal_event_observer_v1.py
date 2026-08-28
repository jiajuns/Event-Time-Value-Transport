from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import train_smolvla_piper_causal_event_observer_v1 as trainer  # noqa: E402
from smolvla_piper_causal_event_observer_v1 import (  # noqa: E402
    ADAPTER_CHECKPOINT_FORMAT,
    CORE_CHECKPOINT_FORMAT,
    EXPECTED_EVENTS,
    EXPECTED_PREDICATES,
    MAX_HISTORY_STEPS,
    STATE_DIM,
    causal_history_contract,
    canonical_sha256,
    load_frozen_causal_observer_runtime,
    tensor_bundle_sha256,
)
from test_materialize_smolvla_piper_causal_event_observer_dataset_v1 import (  # noqa: E402
    make_minimal_materialized_dataset_fixture,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split(seed: int, split_name: str, *, groups_per_actor: int = 3) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    rows = 2 * groups_per_actor * 2
    history = np.zeros((rows, MAX_HISTORY_STEPS, STATE_DIM), dtype=np.float32)
    history_mask = np.zeros((rows, MAX_HISTORY_STEPS), dtype=np.bool_)
    proprio = np.zeros((rows, 14), dtype=np.float32)
    event_label = np.zeros(rows, dtype=np.int64)
    predicate_label = np.zeros((rows, len(EXPECTED_PREDICATES)), dtype=np.float32)
    actor_index = np.zeros(rows, dtype=np.int64)
    current_query_index = np.zeros(rows, dtype=np.int64)
    query_step = np.zeros(rows, dtype=np.int64)
    prior_execution_present = np.zeros(rows, dtype=np.bool_)
    prior_executed_control_steps = np.zeros(rows, dtype=np.int64)
    sample_id: list[str] = []
    logical_group_id: list[str] = []
    branch_id: list[str] = []
    source_file_sha256: list[str] = []
    prior_action_sha256: list[str] = []
    row = 0
    for actor in range(2):
        for group in range(groups_per_actor):
            group_id = f"{split_name}-actor{actor}-group{group}"
            for query in range(2):
                valid = query + 1
                event = (actor * groups_per_actor * 2 + group * 2 + query) % len(
                    EXPECTED_EVENTS
                )
                values = generator.normal(0.0, 0.03, (valid, STATE_DIM)).astype(
                    np.float32
                )
                values[:, event] += np.float32(2.0)
                values[:, 8 + actor] += np.float32(1.0)
                history[row, :valid] = values
                history_mask[row, :valid] = True
                proprio[row] = generator.normal(0.0, 0.03, 14).astype(np.float32)
                proprio[row, event % 5] += np.float32(1.0)
                event_label[row] = event
                predicate_label[row] = np.array(
                    [
                        event in (1, 2, 3, 4),
                        event in (2, 3, 4),
                        event in (3, 4),
                        event in (0, 4),
                        event == 4,
                    ],
                    dtype=np.float32,
                )
                actor_index[row] = actor
                current_query_index[row] = query
                query_step[row] = query
                prior_execution_present[row] = query > 0
                prior_executed_control_steps[row] = query
                action_sha = _sha(f"{split_name}-{actor}-{group}-action0") if query else ""
                prior_action_sha256.append(action_sha)
                source_sha = _sha(f"source-{actor}")
                source_file_sha256.append(source_sha)
                logical_group_id.append(group_id)
                branch_id.append(f"{group_id}-branch")
                sample_id.append(
                    _sha(
                        f"{split_name}|{actor}|{group}|{query}|{source_sha}|{action_sha}"
                    )
                )
                row += 1
    return {
        "history": history,
        "history_mask": history_mask,
        "proprio": proprio,
        "event_label": event_label,
        "predicate_label": predicate_label,
        "actor_index": actor_index,
        "current_query_index": current_query_index,
        "query_step": query_step,
        "prior_execution_present": prior_execution_present,
        "prior_executed_control_steps": prior_executed_control_steps,
        "prior_action_sha256": np.asarray(prior_action_sha256, dtype="<U64"),
        "sample_id": np.asarray(sample_id, dtype="<U64"),
        "logical_group_id": np.asarray(logical_group_id),
        "branch_id": np.asarray(branch_id),
        "source_file_sha256": np.asarray(source_file_sha256, dtype="<U64"),
    }


def _dataset(tmp_path: Path) -> trainer.LoadedDataset:
    actors = (
        {
            "actor_name": "piper",
            "policy_family": "smolvla",
            "body": "piper",
            "policy": "smolvla",
            "state_feature_source_sha256": _sha("piper-feature-source"),
            "actor_index": 0,
        },
        {
            "actor_name": "cross_body",
            "policy_family": "openvla",
            "body": "widowx",
            "policy": "openvla",
            "state_feature_source_sha256": _sha("cross-body-feature-source"),
            "actor_index": 1,
        },
    )
    splits = {
        "train": _split(11, "train"),
        "calibration": _split(13, "calibration"),
        "validation": _split(17, "validation"),
    }
    manifest_path = tmp_path / "manifest.json"
    split_records = {
        name: {"logical_sha256": _sha(f"{name}-logical")}
        for name in trainer.DATASET_SPLITS
    }
    manifest = {
        "format": trainer.DATASET_FORMAT,
        "status": "synthetic_cpu_test_only",
        "event_names": list(EXPECTED_EVENTS),
        "predicate_names": list(EXPECTED_PREDICATES),
        "state_dim": STATE_DIM,
        "history_steps": MAX_HISTORY_STEPS,
        "proprio_dim": 14,
        "image_feature_dim": 0,
        "history_contract_sha256": causal_history_contract()["contract_sha256"],
        "event_spec": {"path": "event.json", "file_sha256": _sha("event-spec")},
        "actor_registry": list(actors),
        "split_unit": "logical_reset_group",
        "split_group_disjoint": True,
        "privileged_label_source_available_to_model_inputs": False,
        "future_query_features_available_to_model_inputs": False,
        "splits": split_records,
        "manifest_sha256": _sha("synthetic-manifest"),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    actor_names, records = trainer._strict_manifest_checks(manifest, splits)
    return trainer.LoadedDataset(
        manifest_path=manifest_path,
        manifest=manifest,
        splits=splits,
        actor_names=actor_names,
        actor_records=records,
    )


def _config() -> trainer.TrainingConfig:
    return trainer.TrainingConfig(
        hidden_dim=16,
        adapter_rank=2,
        epochs=2,
        batch_size_per_actor=2,
        learning_rate=1.0e-3,
        bootstrap_samples=100,
        calibration_grid_size=21,
        minimum_calibration_accepts=2,
        seed=101,
        device="cpu",
    )


def test_dataset_contract_rejects_split_leakage_and_unbound_prior_action(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    leaked = copy.deepcopy(dataset.splits)
    leaked["validation"]["logical_group_id"][0] = leaked["train"][
        "logical_group_id"
    ][0]
    with pytest.raises(trainer.ObserverTrainingError, match="group leakage"):
        trainer._strict_manifest_checks(dataset.manifest, leaked)

    missing_action = copy.deepcopy(dataset.splits)
    row = int(np.flatnonzero(missing_action["train"]["current_query_index"] > 0)[0])
    missing_action["train"]["prior_action_sha256"][row] = ""
    with pytest.raises(trainer.ObserverTrainingError, match="prior execution"):
        trainer._strict_manifest_checks(dataset.manifest, missing_action)


def test_manifest_is_consumed_only_through_content_addressed_bridge_api(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = make_minimal_materialized_dataset_fixture(
        tmp_path / "bridge"
    )
    dataset = trainer.load_supervision_dataset(manifest_path)
    assert dataset.manifest["manifest_sha256"] == manifest["manifest_sha256"]
    assert dataset.actor_names == ("smolvla_aloha", "smolvla_piper")
    assert set(dataset.splits) == set(trainer.DATASET_SPLITS)
    assert all(
        set(arrays) == trainer.ARRAY_FIELDS
        for arrays in dataset.splits.values()
    )
    model, receipt = trainer.train_model(
        dataset,
        trainer.TrainingConfig(
            hidden_dim=16,
            adapter_rank=2,
            epochs=1,
            batch_size_per_actor=1,
            bootstrap_samples=100,
            calibration_grid_size=11,
            seed=19,
        ),
    )
    assert tuple(model.actor_names) == dataset.actor_names
    assert receipt["calibration_or_validation_used_by_optimizer"] is False


def test_balanced_sampler_and_loss_give_each_actor_equal_weight() -> None:
    actor_index = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    order = trainer._balanced_epoch_indices(
        actor_index, 2, np.random.default_rng(7)
    )
    assert np.bincount(actor_index[order], minlength=2).tolist() == [5, 5]

    output = {
        "event_logits": torch.tensor(
            [[8.0, 0, 0, 0, 0], [8.0, 0, 0, 0, 0], [0.0, 8, 0, 0, 0]]
        ),
        "predicate_logits": torch.zeros(3, 5),
    }
    loss, _, _ = trainer._balanced_loss(
        output,
        torch.tensor([0, 0, 0]),
        torch.zeros(3, 5),
        torch.tensor([0, 0, 1]),
        2,
        _config(),
    )
    per_row_event = torch.nn.functional.cross_entropy(
        output["event_logits"], torch.tensor([0, 0, 0]), reduction="none"
    )
    per_row_predicate = torch.nn.functional.binary_cross_entropy_with_logits(
        output["predicate_logits"], torch.zeros(3, 5), reduction="none"
    ).mean(dim=1)
    expected = (
        torch.stack((per_row_event[:2].mean(), per_row_event[2:].mean())).mean()
        + torch.stack((
            per_row_predicate[:2].mean(), per_row_predicate[2:].mean()
        )).mean()
    )
    assert torch.allclose(loss, expected)


def test_cpu_training_is_deterministic_and_uses_only_train_optimizer_split(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    first, first_receipt = trainer.train_model(dataset, _config())
    second, second_receipt = trainer.train_model(dataset, _config())
    assert tensor_bundle_sha256(first.state_dict()) == tensor_bundle_sha256(
        second.state_dict()
    )
    assert first_receipt == second_receipt
    assert first_receipt["actor_balanced_sampling"] is True
    assert first_receipt["actor_balanced_loss"] is True
    assert first_receipt["calibration_or_validation_used_by_optimizer"] is False
    assert len(first_receipt["epochs"]) == 2


def test_group_calibration_validation_and_synthetic_freeze_are_fail_closed(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    config = _config()
    model, training_receipt = trainer.train_model(dataset, config)
    calibration, calibration_fit = trainer.fit_group_calibration(
        model, dataset, config
    )
    model.calibration = calibration
    validation = trainer.evaluate_independent_validation(
        model, dataset, calibration, calibration_fit, config
    )
    assert calibration_fit["equal_group_weighting"] is True
    assert validation["bootstrap"]["unit"] == "logical_reset_group"
    assert validation["confidence_reject"]["wilson_unit"] == (
        "logical_reset_group_any_false_accept"
    )
    assert set(validation["per_actor_validation_groups"]) == {
        "piper", "cross_body"
    }
    assert validation["future_feature_perturbation"]["passed"] is True
    assert validation["cross_branch_isolation"]["passed"] is True
    assert validation["privileged_input_static_audit"]["passed"] is True

    output = tmp_path / "frozen"
    result = trainer.freeze_bundle(
        model=model,
        dataset=dataset,
        calibration=calibration,
        calibration_fit=calibration_fit,
        validation=validation,
        training_receipt=training_receipt,
        config=config,
        output_directory=output,
        synthetic_evidence=True,
    )
    assert result["promotion_enabled"] is False
    assert result["v4_rerank_authority_issued"] is False
    assert not (output / "authority_manifest.json").exists()
    assert not (output / "promotion_evidence.json").exists()
    assert (output / "monitor_freeze_manifest.json").is_file()

    core_path = output / "observer_core_state.pt"
    core = torch.load(core_path, map_location="cpu", weights_only=True)
    assert core["format"] == CORE_CHECKPOINT_FORMAT
    assert tensor_bundle_sha256(core["core_state_dict"]) == core[
        "core_tensor_set_sha256"
    ]
    assert all(not name.startswith("actor_adapters.") for name in core["core_state_dict"])
    adapter_manifest = json.loads(
        (output / "actor_adapter_manifest.json").read_text(encoding="utf-8")
    )
    assert [item["actor_name"] for item in adapter_manifest["ordered_adapters"]] == [
        "piper", "cross_body"
    ]
    for item in adapter_manifest["ordered_adapters"]:
        checkpoint = torch.load(
            output / item["checkpoint_file"], map_location="cpu", weights_only=True
        )
        assert checkpoint["format"] == ADAPTER_CHECKPOINT_FORMAT
        assert tensor_bundle_sha256(checkpoint["adapter_state_dict"]) == checkpoint[
            "adapter_state_sha256"
        ]
    monitor = json.loads(
        (output / "monitor_freeze_manifest.json").read_text(encoding="utf-8")
    )
    assert monitor["synthetic_or_test_evidence"] is True
    assert monitor["real_task_success_or_cross_embodiment_improvement_claimed"] is False


def test_static_audit_detects_no_privileged_online_model_argument() -> None:
    result = trainer._static_privileged_input_audit()
    assert result["passed"] is True
    assert result["forbidden_identifiers_found"] == []
    assert "future_hidden" not in result["forward_parameter_names"]
    assert "object_poses" not in result["forward_parameter_names"]


def test_test_only_forced_gate_fixture_matches_frozen_v4_runtime_loader(
    tmp_path: Path,
) -> None:
    """Exercise artifact plumbing only; these numbers are not model evidence."""

    dataset = _dataset(tmp_path)
    config = _config()
    model, training_receipt = trainer.train_model(dataset, config)
    calibration, calibration_fit = trainer.fit_group_calibration(
        model, dataset, config
    )
    # Contract-path fixture only: make the calibration artifact internally
    # consistent with the forced passing metrics below.
    calibration = dict(calibration)
    calibration["reject_all"] = False
    calibration_base = dict(calibration)
    calibration_base.pop("calibration_sha256")
    calibration["calibration_sha256"] = canonical_sha256(calibration_base)
    calibration_fit = copy.deepcopy(calibration_fit)
    calibration_fit["calibration_sha256"] = calibration["calibration_sha256"]
    calibration_fit["reject_all"] = False
    calibration_fit["promotion_calibration_support_gate_passed"] = True
    calibration_fit["low_confidence_reject_fit"]["status"] = (
        "highest_coverage_threshold_meeting_false_accept_gate"
    )
    calibration_fit_base = dict(calibration_fit)
    calibration_fit_base.pop("calibration_fit_receipt_sha256")
    calibration_fit["calibration_fit_receipt_sha256"] = canonical_sha256(
        calibration_fit_base
    )
    model.calibration = calibration
    validation = trainer.evaluate_independent_validation(
        model, dataset, calibration, calibration_fit, config
    )
    validation["status"] = "independent_validation_passed_all_gates"
    validation["independent_validation_groups"] = 60
    validation["per_actor_validation_groups"] = {
        "piper": 30,
        "cross_body": 30,
    }
    validation["event_macro_accuracy"]["group_bootstrap_lcb95"] = 0.80
    validation["predicate_macro_f1"]["group_bootstrap_lcb95"] = 0.75
    validation["maximum_event_ece"] = 0.05
    validation["maximum_predicate_ece"] = 0.05
    validation["confidence_reject"]["false_accept_wilson_ucb95"] = 0.02
    validation["gates"] = {name: True for name in validation["gates"]}
    validation["all_promotion_gates_passed"] = True
    unsigned = dict(validation)
    unsigned.pop("validation_receipt_sha256")
    validation["validation_receipt_sha256"] = canonical_sha256(unsigned)

    output = tmp_path / "forced_contract_fixture"
    result = trainer.freeze_bundle(
        model=model,
        dataset=dataset,
        calibration=calibration,
        calibration_fit=calibration_fit,
        validation=validation,
        training_receipt=training_receipt,
        config=config,
        output_directory=output,
        # This is deliberately false only to reach the production artifact
        # parser in a temporary unit-test directory. No result is persisted.
        synthetic_evidence=False,
    )
    assert result["v4_rerank_authority_issued"] is True
    runtime = load_frozen_causal_observer_runtime(output)
    assert runtime.authority["rerank_enabled"] is True
    assert runtime.model.training is False
    assert all(not parameter.requires_grad for parameter in runtime.model.parameters())
