from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_openvla_etsf_v8_oof_bridge as bridge  # noqa: E402
import evaluate_openvla_etsf_v8_structured_heads_arrays as evaluator  # noqa: E402
from openvla_etsf_v8_adaptive_development_protocol import (  # noqa: E402
    make_adaptive_development_contract,
    validate_adaptive_development_contract,
)
from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_LOSS_CONTRACT,
    V8_OBJECT_MODE,
    V8DetachedStructuredAdapters,
    V8StructuredAdapterConfig,
    frozen_tensor_mapping_sha256,
    module_state_sha256,
)
from openvla_etsf_v8_structured_heads_protocol import canonical_sha256  # noqa: E402
from materialize_openvla_etsf_v8_oof_inputs import (  # noqa: E402
    _synthetic_group,
    build_outer_fold_payloads,
)
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_TRAINING_CHECKPOINT_FORMAT,
    structured_payload_sha256,
    train_v8_payload,
)


@pytest.fixture(autouse=True)
def _small_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator, "BOOTSTRAP_SAMPLES", 100)


def _signed_contract(value: dict, key: str) -> dict:
    value[key] = canonical_sha256(value)
    return value


def _duration_baseline_contract() -> dict:
    return _signed_contract(
        {
            "protocol": "observed_only_laplace_log1p_residual_about_training_event_body_median_v1",
            "fit_labels": "observed_training_transitions_only",
            "observed_support": 80,
            "exact_event_body": {
                "1:0": {"median_log1p_duration": float(np.log1p(5.0)), "support": 80}
            },
            "event_fallback": {
                "1": {"median_log1p_duration": float(np.log1p(5.0)), "support": 80}
            },
            "body_fallback": {
                "0": {"median_log1p_duration": float(np.log1p(5.0)), "support": 80}
            },
            "global": {"median_log1p_duration": float(np.log1p(5.0)), "support": 80},
        },
        "duration_baseline_contract_sha256",
    )


def _object_fallback_contract() -> dict:
    return _signed_contract(
        {
            "format": "etsf_v8_outer_training_object_fallback_contract_v1",
            "repair_protocol": "training_fold_robust_median_scale_q995_quality_mask_student_t_df3_v1",
            "fit_rows": "outer_training_dense_rows_only",
            "object_mode": V8_OBJECT_MODE,
            "learned_object_output_authorized": False,
            "physical_coordinate_median": [0.0, 0.0],
            "schema5_normalized_fallback": [0.0, 0.0],
            "robust_repair_contract": {"synthetic": True},
        },
        "object_fallback_contract_sha256",
    )


def _duration_scale(owner: int) -> dict:
    return _signed_contract(
        {
            "format": bridge.DURATION_SCALE_FORMAT,
            "owner_fold_id": owner,
            "fit_scope": "outer_training_observed_only",
            "estimator": "median_absolute_deviation_divided_by_log_2",
            "censored_rows_used": False,
            "model_location": "event_body_median_plus_0.375_frozen_residual",
            "baseline_location": "outer_training_event_body_median",
            "minimum_scale": bridge.MINIMUM_DURATION_SCALE,
            "outer_training_observed_support": 80,
            "model_log_scale": float(np.log(0.4)),
            "baseline_log_scale": float(np.log(0.5)),
        },
        "contract_sha256",
    )


def _fold_artifacts(
    root: Path, owner: int, groups: list[str]
) -> tuple[Path, Path, dict]:
    heldout = [groups[owner]]
    training = sorted(set(groups) - set(heldout))
    duration_contract = _duration_baseline_contract()
    duration_scale = _duration_scale(owner)
    object_contract = _object_fallback_contract()
    provenance = {
        "outer_fold_id": owner,
        "outer_training_groups": training,
        "outer_training_groups_sha256": canonical_sha256(
            {"logical_groups": training}
        ),
        "oof_holdout_groups": heldout,
        "oof_holdout_groups_sha256": canonical_sha256(
            {"logical_groups": heldout}
        ),
        "target_outer_fold_labels_used": False,
        "factual_outputs_frozen": True,
        "base_checkpoint": "/synthetic/base.pt",
        "base_checkpoint_sha256": "c" * 64,
        "base_target_outer_fold_exclusion_status": "proven",
        "base_exclusion_audit": {
            "status": "proven",
            "reason": "synthetic",
            "base_identity_contract_sha256": "d" * 64,
            "legacy_old100_holdout_groups": [],
        },
        "event_spec": "/synthetic/event.json",
        "event_spec_sha256": "f" * 64,
        "label_derivation_contract": {"synthetic": True},
        "label_derivation_sha256": "b" * 64,
        "duration_baseline_contract": duration_contract,
        "duration_baseline_contract_sha256": duration_contract[
            "duration_baseline_contract_sha256"
        ],
        "duration_laplace_scale_contract": duration_scale,
        "duration_laplace_scale_contract_sha256": duration_scale[
            "contract_sha256"
        ],
        "object_fallback_contract": object_contract,
        "object_fallback_contract_sha256": object_contract[
            "object_fallback_contract_sha256"
        ],
        "object_mode": V8_OBJECT_MODE,
        "fresh_confirmation_data_or_labels_read": False,
    }
    adapters = V8DetachedStructuredAdapters(V8StructuredAdapterConfig(transition_dim=3))
    adapters.initialize_probability_biases(
        success_prevalence=0.5,
        regress_prevalence=0.5,
        recovery_given_regress_prevalence=0.5,
    )
    checkpoint = {
        "format": V8_TRAINING_CHECKPOINT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": 3},
        "state_dict": adapters.state_dict(),
        "training_contract": {
            "loss": V8_LOSS_CONTRACT,
            "success_loss": "unweighted_binary_cross_entropy",
            "regress_loss": "unweighted_binary_cross_entropy",
            "recovery_loss": "unweighted_binary_cross_entropy_on_true_regress_rows_only",
        },
        "provenance": provenance,
        "support": {
            "success_support": 80,
            "success_positive": 40,
            "regress_support": 100,
            "regress_positive": 50,
            "recovery_given_regress_support": 50,
            "recovery_given_regress_positive": 25,
        },
        "duration_laplace_scale_contract": duration_scale,
        "all_steps_factual_inputs_bit_exact": True,
        "adapter_state_sha256": module_state_sha256(adapters),
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
    }
    checkpoint_path = root / f"fold_{owner}_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    count = 5
    transition = torch.tensor(
        [[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]],
        dtype=torch.float32,
    )
    factual = {
        "transition": transition,
        "duration_selected_log_mean": torch.full(
            (count,), float(np.log1p(4.0)), dtype=torch.float32
        ),
    }
    batch = {
        "terminal_mask": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
        "structured_mask": torch.ones(count, dtype=torch.bool),
        "dense_mask": torch.ones(count, dtype=torch.bool),
        "duration": torch.tensor([4, 5, 6, 7, 5], dtype=torch.float32),
        "duration_observed": torch.ones(count, dtype=torch.bool),
        "success": torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32),
        "trajectory_regress": torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool),
        "trajectory_recovery": torch.tensor([0, 1, 0, 0, 0], dtype=torch.bool),
        "object_delta": torch.zeros(count, 2),
        "clock_event_id": torch.ones(count, dtype=torch.long),
        "body_id": torch.zeros(count, dtype=torch.long),
        "group_index": torch.tensor([0, 0, 0, 0, -1], dtype=torch.long),
        "baseline_mask": torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool),
        "group_keys": heldout,
        "candidate_names": [
            "deterministic",
            "blend_0.25",
            "blend_0.50",
            "blend_0.75",
            "continuation_0",
        ],
    }
    record = {
        "logical_group_key": heldout[0],
        "split_role": "outer_holdout",
        "outer_fold_id": owner,
        "batch": batch,
        "factual_outputs": factual,
        "factual_outputs_sha256": frozen_tensor_mapping_sha256(factual),
        "duration_baseline_log1p": torch.full(
            (count,), float(np.log1p(5.0)), dtype=torch.float32
        ),
        "duration_baseline_source": ["event_body"] * count,
        "object_fallback": torch.zeros(2),
        "object_delta_physical": torch.tensor(
            [[0.0, 0.0], [0.01, -0.01], [0.0, 0.0], [0.02, 0.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        # Deliberately omit object_pose_quality_valid.  The bridge must mark it
        # unavailable and force the object domain to retain the fallback.
        "factual_outputs_require_grad": False,
    }
    audit = {
        "factual_state_sha256_before": "e" * 64,
        "factual_state_sha256_after": "e" * 64,
        "factual_state_bit_exact": True,
        "records": 1,
        "rows": count,
    }
    holdout = {
        "format": bridge.HOLDOUT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": 3},
        "batches": [record],
        "provenance": {
            **provenance,
            "split_role": "outer_holdout_evaluation_only",
            "holdout_labels_used_for_duration_or_object_fit": False,
            "holdout_labels_present_only_in_separate_artifact": True,
        },
        "materialization_audit": audit,
    }
    holdout["payload_sha256"] = structured_payload_sha256(holdout)
    holdout_path = root / f"fold_{owner}_holdout.pt"
    torch.save(holdout, holdout_path)
    train_record = copy.deepcopy(record)
    train_record["logical_group_key"] = training[0]
    train_record["split_role"] = "outer_training"
    train_record["batch"]["group_keys"] = [training[0]]
    train = {
        "format": bridge.V8_TRAINING_INPUT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": 3},
        "batches": [train_record],
        "provenance": provenance,
        "materialization_audit": audit,
    }
    train["payload_sha256"] = structured_payload_sha256(train)
    train_path = root / f"fold_{owner}_train.pt"
    torch.save(train, train_path)
    fold_row = {
        "outer_fold_id": owner,
        "training_groups": training,
        "training_groups_sha256": provenance["outer_training_groups_sha256"],
        "oof_holdout_groups": heldout,
        "oof_holdout_groups_sha256": provenance["oof_holdout_groups_sha256"],
        "train_artifact": str(train_path),
        "train_artifact_sha256": bridge.sha256_path(train_path),
        "train_payload_sha256": train["payload_sha256"],
        "holdout_artifact": str(holdout_path),
        "holdout_artifact_sha256": bridge.sha256_path(holdout_path),
        "holdout_payload_sha256": holdout["payload_sha256"],
        "base_exclusion_audit": provenance["base_exclusion_audit"],
        "target_outer_fold_labels_used_for_training": False,
    }
    return checkpoint_path, holdout_path, fold_row


def _artifacts(tmp_path: Path) -> tuple[list[Path], list[Path], Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    groups = [f"move_can_pot|piper|g{fold}" for fold in range(5)]
    checkpoints: list[Path] = []
    holdouts: list[Path] = []
    folds: list[dict] = []
    for owner in range(5):
        checkpoint, holdout, fold = _fold_artifacts(tmp_path, owner, groups)
        checkpoints.append(checkpoint)
        holdouts.append(holdout)
        folds.append(fold)
    materialization = {
        "format": bridge.MATERIALIZATION_FORMAT,
        "status": "complete_development_only",
        "source_oof_preregistration_sha256": "1" * 64,
        "base_checkpoint_sha256": "c" * 64,
        "event_spec_sha256": "f" * 64,
        "label_derivation_sha256": "b" * 64,
        "source_collection_audit": {"synthetic": True},
        "development_groups": groups,
        "development_groups_sha256": canonical_sha256({"logical_groups": groups}),
        "folds": folds,
        "fresh_confirmation_data_or_labels_read": False,
        "remote_write_performed": False,
        "authorization_guard_changed": False,
    }
    materialization["materialization_sha256"] = canonical_sha256(materialization)
    path = tmp_path / "materialization_manifest.json"
    path.write_text(json.dumps(materialization), encoding="utf-8")
    fold_by_owner = {int(row["outer_fold_id"]): row for row in folds}
    for owner, checkpoint_path in enumerate(checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["input_artifact_authentication"] = (
            bridge._expected_input_artifact_authentication(
                materialization=materialization,
                materialization_path=path,
                owner=owner,
                materialization_folds=fold_by_owner,
            )
        )
        torch.save(checkpoint, checkpoint_path)
    return checkpoints, holdouts, path, "d" * 64


def _adaptive_contract(materialization_path: Path, base_identity_sha: str) -> dict:
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    return make_adaptive_development_contract(
        implementation_sha256=bridge.sha256_path(Path(bridge.__file__)),
        label_derivation_sha256=materialization["label_derivation_sha256"],
        base_checkpoint_sha256=materialization["base_checkpoint_sha256"],
        base_identity_contract_sha256=base_identity_sha,
    )


def _refresh_checkpoint_authentication(
    checkpoints: list[Path], materialization_path: Path
) -> None:
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    fold_by_owner = {
        int(row["outer_fold_id"]): row for row in materialization["folds"]
    }
    for owner, checkpoint_path in enumerate(checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["input_artifact_authentication"] = (
            bridge._expected_input_artifact_authentication(
                materialization=materialization,
                materialization_path=materialization_path,
                owner=owner,
                materialization_folds=fold_by_owner,
            )
        )
        torch.save(checkpoint, checkpoint_path)


def _resign_materialized_payload(
    *,
    checkpoints: list[Path],
    materialization_path: Path,
    owner: int,
    role: str,
    payload: dict,
) -> None:
    manifest = json.loads(materialization_path.read_text(encoding="utf-8"))
    row = manifest["folds"][owner]
    artifact = Path(row[f"{role}_artifact"])
    payload["payload_sha256"] = structured_payload_sha256(payload)
    torch.save(payload, artifact)
    row[f"{role}_artifact_sha256"] = bridge.sha256_path(artifact)
    row[f"{role}_payload_sha256"] = payload["payload_sha256"]
    manifest.pop("materialization_sha256")
    manifest["materialization_sha256"] = canonical_sha256(manifest)
    materialization_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_checkpoint_authentication(checkpoints, materialization_path)


def test_synthetic_five_fold_bridge_runs_adaptive_evaluation_and_fallback(
    tmp_path: Path,
) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path)
    r3_style_manifest = json.loads(materialization.read_text(encoding="utf-8"))
    assert all(
        "training_materialization_audit" not in fold
        and "holdout_materialization_audit" not in fold
        for fold in r3_style_manifest["folds"]
    )
    value = bridge.evaluate_oof_artifacts(
        checkpoint_paths=checkpoints,
        holdout_paths=holdouts,
        materialization_manifest_path=materialization,
        base_identity_contract_sha256=base_sha,
    )
    assert value["adaptive_contract"]["source_sha256"][
        "base_identity_contract"
    ] == base_sha
    assert "base_training_groups" not in value["adaptive_contract"]["source_sha256"]
    arrays = value["arrays"]
    assert len(arrays["logical_group"]) == 25
    for fold in range(5):
        selected = arrays["fold_id"] == fold
        assert arrays["candidate_index"][selected].tolist() == [0, 1, 2, 3, 4]
        assert arrays["success_mask"][selected].tolist() == [True, True, True, True, False]
    assert not arrays["object_pose_quality_valid"].any()
    assert np.array_equal(arrays["object_model_delta"], arrays["object_robust_median_delta"])
    result = value["evaluation_result"]
    assert result["evidence_design"] == "adaptive_current_d250_after_collection_started"
    assert result["prospective_claim_allowed"] is False
    assert result["domains"]["next_event"]["evaluated"] is False
    assert result["domains"]["object"]["activated_output"] == (
        "outer_training_robust_median_fallback"
    )
    assert value["bridge_provenance"]["object_output"] == (
        "outer_training_robust_physical_fallback_never_learned"
    )
    assert value["bridge_provenance"]["all_primary_base_exclusion_proven"] is True
    assert all(
        status == "unavailable_all_rows_fail_closed"
        for status in value["bridge_provenance"]["object_quality_status_by_fold"].values()
    )
    for head in ("success", "regress", "recovery_given_regress"):
        for fold in range(5):
            provenance = value["probability_weight_provenance"][head][str(fold)]
            assert provenance["outer_training_prevalence"] == 0.5
            assert provenance["owner_holdout_labels_used"] is False


def test_payload_local_train_and_holdout_factual_audits_fail_closed(
    tmp_path: Path,
) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path / "missing")
    manifest = json.loads(materialization.read_text(encoding="utf-8"))
    train_path = Path(manifest["folds"][0]["train_artifact"])
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    del train["materialization_audit"]
    _resign_materialized_payload(
        checkpoints=checkpoints,
        materialization_path=materialization,
        owner=0,
        role="train",
        payload=train,
    )
    with pytest.raises(RuntimeError, match="train lacks bit-exact factual-state audit"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )

    checkpoints, holdouts, materialization, base_sha = _artifacts(
        tmp_path / "disagree"
    )
    manifest = json.loads(materialization.read_text(encoding="utf-8"))
    train_path = Path(manifest["folds"][0]["train_artifact"])
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    train["materialization_audit"]["factual_state_sha256_before"] = "9" * 64
    train["materialization_audit"]["factual_state_sha256_after"] = "9" * 64
    _resign_materialized_payload(
        checkpoints=checkpoints,
        materialization_path=materialization,
        owner=0,
        role="train",
        payload=train,
    )
    with pytest.raises(RuntimeError, match="train/holdout factual state hashes disagree"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )


def test_bridge_writes_complete_arrays_contracts_and_result(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    checkpoints, holdouts, materialization, base_sha = _artifacts(root)
    value = bridge.evaluate_oof_artifacts(
        checkpoint_paths=checkpoints,
        holdout_paths=holdouts,
        materialization_manifest_path=materialization,
        base_identity_contract_sha256=base_sha,
    )
    summary = bridge.write_evaluation_output(tmp_path / "output", value)
    assert summary["status"] == "complete_adaptive_development_only"
    assert summary["prospective_claim_allowed"] is False
    with np.load(summary["arrays"], allow_pickle=False) as arrays:
        assert set(arrays.files) == set(value["arrays"])
    contracts = json.loads(Path(summary["contracts"]).read_text(encoding="utf-8"))
    assert contracts["bridge_provenance"]["fresh50_labels_read"] is False
    assert bridge.sha256_path(Path(summary["arrays"])) == contracts["arrays_sha256"]


def test_bridge_consumes_real_materializer_and_trainer_payloads(tmp_path: Path) -> None:
    config = EventWorldModelConfig(
        state_input_dim=12,
        action_dim=4,
        proprio_dim=3,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=6,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=True,
        dropout=0.0,
    )
    model = ActionConditionedEventWorldModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    context = {
        "model": model,
        "config": config,
        "checkpoint_path": "/synthetic/factual.pt",
        "checkpoint_sha256": "c" * 64,
        "event_spec_path": "/synthetic/event.json",
        "event_spec_sha256": "f" * 64,
        "object_mean": np.zeros(3, dtype=np.float32),
        "object_std": np.ones(3, dtype=np.float32),
        "label_derivation_contract": {
            "format": "synthetic",
            "label_derivation_sha256": "b" * 64,
        },
    }
    groups = [
        _synthetic_group(
            f"move_can_pot|piper|real{fold}",
            config=config,
            duration_offset=float(fold),
        )
        for fold in range(5)
    ]
    checkpoints: list[Path] = []
    holdouts: list[Path] = []
    fold_rows: list[dict] = []
    all_keys = sorted(group.logical_key for group in groups)
    for owner in range(5):
        holdout_groups = [groups[owner]]
        training_groups = [group for index, group in enumerate(groups) if index != owner]
        fold = {
            "outer_fold_id": owner,
            "training_groups": sorted(group.logical_key for group in training_groups),
            "oof_holdout_groups": sorted(group.logical_key for group in holdout_groups),
        }
        built = build_outer_fold_payloads(
            fold=fold,
            training_groups=training_groups,
            holdout_groups=holdout_groups,
            context=context,
            device="cpu",
        )
        proven_audit = {
            "status": "proven",
            "reason": "synthetic_checkpoint_bound_base_identity",
            "base_identity_contract_sha256": "d" * 64,
            "legacy_old100_holdout_groups": [],
        }
        for role in ("training_payload", "holdout_payload"):
            built[role]["provenance"][
                "base_target_outer_fold_exclusion_status"
            ] = "proven"
            built[role]["provenance"]["base_exclusion_audit"] = proven_audit
            built[role]["payload_sha256"] = structured_payload_sha256(built[role])
        built["fold_manifest"]["base_exclusion_audit"] = proven_audit
        checkpoint = train_v8_payload(
            built["training_payload"], epochs=1, device="cpu"
        )
        checkpoint_path = tmp_path / f"real_fold_{owner}_checkpoint.pt"
        train_path = tmp_path / f"real_fold_{owner}_train.pt"
        holdout_path = tmp_path / f"real_fold_{owner}_holdout.pt"
        torch.save(checkpoint, checkpoint_path)
        torch.save(built["training_payload"], train_path)
        torch.save(built["holdout_payload"], holdout_path)
        checkpoints.append(checkpoint_path)
        holdouts.append(holdout_path)
        fold_rows.append(
            {
                **built["fold_manifest"],
                "train_artifact": str(train_path),
                "train_artifact_sha256": bridge.sha256_path(train_path),
                "train_payload_sha256": built["training_payload"]["payload_sha256"],
                "holdout_artifact": str(holdout_path),
                "holdout_artifact_sha256": bridge.sha256_path(holdout_path),
                "holdout_payload_sha256": built["holdout_payload"]["payload_sha256"],
            }
        )
    materialization_value = {
        "format": bridge.MATERIALIZATION_FORMAT,
        "status": "complete_development_only",
        "source_oof_preregistration_sha256": "1" * 64,
        "base_checkpoint_sha256": "c" * 64,
        "event_spec_sha256": "f" * 64,
        "label_derivation_sha256": "b" * 64,
        "source_collection_audit": {"synthetic": True},
        "development_groups": all_keys,
        "development_groups_sha256": canonical_sha256(
            {"logical_groups": all_keys}
        ),
        "folds": fold_rows,
        "fresh_confirmation_data_or_labels_read": False,
        "remote_write_performed": False,
        "authorization_guard_changed": False,
    }
    materialization_value["materialization_sha256"] = canonical_sha256(
        materialization_value
    )
    materialization_path = tmp_path / "real_materialization_manifest.json"
    materialization_path.write_text(
        json.dumps(materialization_value), encoding="utf-8"
    )
    fold_by_owner = {
        int(row["outer_fold_id"]): row for row in materialization_value["folds"]
    }
    for owner, checkpoint_path in enumerate(checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["input_artifact_authentication"] = (
            bridge._expected_input_artifact_authentication(
                materialization=materialization_value,
                materialization_path=materialization_path,
                owner=owner,
                materialization_folds=fold_by_owner,
            )
        )
        torch.save(checkpoint, checkpoint_path)
    value = bridge.evaluate_oof_artifacts(
        checkpoint_paths=checkpoints,
        holdout_paths=holdouts,
        materialization_manifest_path=materialization_path,
        base_identity_contract_sha256="d" * 64,
    )
    assert len(value["arrays"]["logical_group"]) == 25
    assert value["evaluation_result"]["prospective_claim_allowed"] is False
    assert value["evaluation_result"]["domains"]["object"]["passed"] is False
    assert value["bridge_provenance"]["all_primary_base_exclusion_proven"] is True


def test_file_hash_state_owner_and_role_tampering_fail_closed(tmp_path: Path) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path)
    contract = _adaptive_contract(materialization, base_sha)
    bundle = bridge.make_bridge_bundle(
        checkpoint_paths=checkpoints,
        holdout_paths=holdouts,
        materialization_manifest_path=materialization,
        adaptive_contract=contract,
    )
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["success_head.bias"] += 1.0
    torch.save(checkpoint, checkpoints[0])
    with pytest.raises(RuntimeError, match="file SHA mismatch"):
        bridge.build_evaluation_inputs(bundle=bundle, adaptive_contract=contract)

    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path / "second")
    contract = _adaptive_contract(materialization, base_sha)
    holdout = torch.load(holdouts[0], map_location="cpu", weights_only=True)
    holdout["provenance"]["split_role"] = "outer_training"
    holdout["payload_sha256"] = structured_payload_sha256(holdout)
    torch.save(holdout, holdouts[0])
    manifest = json.loads(materialization.read_text(encoding="utf-8"))
    manifest["folds"][0]["holdout_artifact_sha256"] = bridge.sha256_path(holdouts[0])
    manifest["folds"][0]["holdout_payload_sha256"] = holdout["payload_sha256"]
    manifest.pop("materialization_sha256")
    manifest["materialization_sha256"] = canonical_sha256(manifest)
    materialization.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = bridge.make_bridge_bundle(
        checkpoint_paths=checkpoints,
        holdout_paths=holdouts,
        materialization_manifest_path=materialization,
        adaptive_contract=contract,
    )
    with pytest.raises(
        RuntimeError, match="input artifact authentication|provenance differs|role"
    ):
        bridge.build_evaluation_inputs(bundle=bundle, adaptive_contract=contract)


def test_missing_outer_train_scale_or_physical_object_fields_fail_closed(
    tmp_path: Path,
) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path)
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    del checkpoint["duration_laplace_scale_contract"]
    torch.save(checkpoint, checkpoints[0])
    with pytest.raises(RuntimeError, match="prediction agent must publish"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )

    second = tmp_path / "second"
    second.mkdir()
    checkpoints, holdouts, materialization, base_sha = _artifacts(second)
    holdout = torch.load(holdouts[0], map_location="cpu", weights_only=True)
    del holdout["batches"][0]["object_delta_physical"]
    holdout["payload_sha256"] = structured_payload_sha256(holdout)
    torch.save(holdout, holdouts[0])
    # Keep the signed materialization row aligned with the deliberately changed
    # holdout so the failure reaches the semantic physical-unit check.
    manifest = json.loads(materialization.read_text(encoding="utf-8"))
    manifest["folds"][0]["holdout_artifact_sha256"] = bridge.sha256_path(holdouts[0])
    manifest["folds"][0]["holdout_payload_sha256"] = holdout["payload_sha256"]
    manifest.pop("materialization_sha256")
    manifest["materialization_sha256"] = canonical_sha256(manifest)
    materialization.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_checkpoint_authentication(checkpoints, materialization)
    with pytest.raises(RuntimeError, match="object_delta_physical"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )


def test_base_identity_contract_must_match_all_proven_materialization_folds(
    tmp_path: Path,
) -> None:
    checkpoints, holdouts, materialization, _ = _artifacts(tmp_path)
    with pytest.raises(RuntimeError, match="base identity contract differs"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256="9" * 64,
        )


def test_resigned_top_level_duration_scale_cannot_diverge_from_provenance(
    tmp_path: Path,
) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path)
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    scale = dict(checkpoint["duration_laplace_scale_contract"])
    scale.pop("contract_sha256")
    scale["model_log_scale"] = float(scale["model_log_scale"]) + 0.25
    scale["contract_sha256"] = canonical_sha256(scale)
    checkpoint["duration_laplace_scale_contract"] = scale
    torch.save(checkpoint, checkpoints[0])
    with pytest.raises(RuntimeError, match="duration scale differs"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )


def test_checkpoint_training_artifact_receipt_and_train_file_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path)
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    checkpoint["input_artifact_authentication"]["train_payload_sha256"] = "8" * 64
    torch.save(checkpoint, checkpoints[0])
    with pytest.raises(RuntimeError, match="input artifact authentication mismatch"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )

    checkpoints, holdouts, materialization, base_sha = _artifacts(
        tmp_path / "train_file"
    )
    manifest = json.loads(materialization.read_text(encoding="utf-8"))
    train_path = Path(manifest["folds"][0]["train_artifact"])
    with train_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(
        RuntimeError,
        match="authenticated train artifact changed|materialized train artifact file SHA mismatch",
    ):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )


def test_candidate_mapping_and_adaptive_temporal_claim_are_immutable(tmp_path: Path) -> None:
    checkpoints, holdouts, materialization, base_sha = _artifacts(tmp_path)
    holdout = torch.load(holdouts[0], map_location="cpu", weights_only=True)
    holdout["batches"][0]["batch"]["baseline_mask"] = torch.tensor(
        [0, 1, 0, 0, 0], dtype=torch.bool
    )
    holdout["payload_sha256"] = structured_payload_sha256(holdout)
    torch.save(holdout, holdouts[0])
    manifest = json.loads(materialization.read_text(encoding="utf-8"))
    manifest["folds"][0]["holdout_artifact_sha256"] = bridge.sha256_path(holdouts[0])
    manifest["folds"][0]["holdout_payload_sha256"] = holdout["payload_sha256"]
    manifest.pop("materialization_sha256")
    manifest["materialization_sha256"] = canonical_sha256(manifest)
    materialization.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_checkpoint_authentication(checkpoints, materialization)
    with pytest.raises(RuntimeError, match="candidate0"):
        bridge.evaluate_oof_artifacts(
            checkpoint_paths=checkpoints,
            holdout_paths=holdouts,
            materialization_manifest_path=materialization,
            base_identity_contract_sha256=base_sha,
        )

    contract = _adaptive_contract(materialization, base_sha)
    changed = copy.deepcopy(contract)
    changed["temporal_provenance"]["prospective_claim_allowed"] = True
    changed["contract_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "contract_sha256"}
    )
    with pytest.raises(RuntimeError, match="frozen contract|temporal"):
        validate_adaptive_development_contract(changed)
