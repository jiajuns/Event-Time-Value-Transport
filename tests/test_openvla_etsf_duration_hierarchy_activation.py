from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_openvla_etsf_duration_hierarchy_oof as r5_evaluator  # noqa: E402
import freeze_openvla_etsf_duration_hierarchy_activation as freezer  # noqa: E402
from openvla_etsf_duration_hierarchy import canonical_sha256  # noqa: E402
from openvla_etsf_duration_hierarchy_adapter import (  # noqa: E402
    fit_final_duration_hierarchy,
    load_duration_activation,
    predict_duration_candidates,
    validate_duration_activation,
)
from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    frozen_tensor_mapping_sha256,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    structured_payload_sha256,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2), encoding="utf-8"
    )


def _record(group: str, group_index: int, fold: int, role: str) -> dict:
    count = 2
    duration = torch.tensor(
        [10.0 + group_index % 7, 25.0 + group_index % 11],
        dtype=torch.float32,
    )
    current = torch.tensor(
        [group_index % 3, (group_index + 1) % 3], dtype=torch.int64
    )
    body = torch.zeros(count, dtype=torch.int64)
    batch = {
        "duration": duration,
        "duration_observed": torch.ones(count, dtype=torch.bool),
        "dense_mask": torch.ones(count, dtype=torch.bool),
        "current_event_id": current,
        "clock_event_id": current.clone(),
        "body_id": body,
        "policy_id": torch.zeros(count, dtype=torch.int64),
    }
    factual = {
        "transition": torch.zeros((count, 2), dtype=torch.float32),
        "duration_selected_log_mean": torch.log1p(duration),
    }
    return {
        "logical_group_key": group,
        "split_role": role,
        "outer_fold_id": fold,
        "group_metadata": {
            "logical_group_key": group,
            "body": "piper_piper_0.6",
            "body_id": 0,
            "policy": "openvla",
            "policy_id": 0,
        },
        "batch": batch,
        "factual_outputs": factual,
        "factual_outputs_sha256": frozen_tensor_mapping_sha256(factual),
        "factual_outputs_require_grad": False,
    }


def _payload(
    groups: list[str],
    *,
    group_to_index: dict[str, int],
    training: list[str],
    holdout: list[str],
    fold: int,
    role: str,
    state_sha: str,
    checkpoint_sha: str,
    event_spec_sha: str,
) -> dict:
    record_role = "outer_training" if role == "train" else "outer_holdout"
    provenance = {
        "outer_fold_id": fold,
        "outer_training_groups": training,
        "outer_training_groups_sha256": r5_evaluator.logical_group_list_sha256(
            training
        ),
        "oof_holdout_groups": holdout,
        "oof_holdout_groups_sha256": r5_evaluator.logical_group_list_sha256(
            holdout
        ),
        "target_outer_fold_labels_used": False,
        "factual_outputs_frozen": True,
        "base_checkpoint_sha256": checkpoint_sha,
        "event_spec_sha256": event_spec_sha,
    }
    if role == "holdout":
        provenance.update(
            {
                "split_role": "outer_holdout_evaluation_only",
                "holdout_labels_used_for_duration_or_object_fit": False,
                "holdout_labels_present_only_in_separate_artifact": True,
            }
        )
    value = {
        "format": (
            r5_evaluator.TRAINING_FORMAT
            if role == "train"
            else r5_evaluator.HOLDOUT_FORMAT
        ),
        "schema_version": 5,
        "config": {"transition_dim": 2},
        "batches": [
            _record(group, group_to_index[group], fold, record_role)
            for group in groups
        ],
        "provenance": provenance,
        "materialization_audit": {
            "factual_state_sha256_before": state_sha,
            "factual_state_sha256_after": state_sha,
            "factual_state_bit_exact": True,
        },
    }
    value["payload_sha256"] = structured_payload_sha256(value)
    return value


@pytest.fixture(scope="module")
def signed_evidence(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("duration_activation_evidence")
    materialized = root / "r3_materialized"
    materialized.mkdir()
    groups = [
        f"move_can_pot|piper_piper_0.6|{index:03d}" for index in range(250)
    ]
    groups = sorted(groups)
    group_to_index = {group: index for index, group in enumerate(groups)}
    checkpoint_sha = "a" * 64
    event_spec_sha = "b" * 64
    state_sha = "c" * 64
    folds = []
    for fold in range(5):
        holdout = groups[fold * 50 : (fold + 1) * 50]
        training = sorted(set(groups) - set(holdout))
        train = _payload(
            training,
            group_to_index=group_to_index,
            training=training,
            holdout=holdout,
            fold=fold,
            role="train",
            state_sha=state_sha,
            checkpoint_sha=checkpoint_sha,
            event_spec_sha=event_spec_sha,
        )
        test = _payload(
            holdout,
            group_to_index=group_to_index,
            training=training,
            holdout=holdout,
            fold=fold,
            role="holdout",
            state_sha=state_sha,
            checkpoint_sha=checkpoint_sha,
            event_spec_sha=event_spec_sha,
        )
        train_path = materialized / f"fold_{fold}_train.pt"
        holdout_path = materialized / f"fold_{fold}_holdout.pt"
        torch.save(train, train_path)
        torch.save(test, holdout_path)
        folds.append(
            {
                "outer_fold_id": fold,
                "training_groups": training,
                "training_groups_sha256": r5_evaluator.logical_group_list_sha256(
                    training
                ),
                "oof_holdout_groups": holdout,
                "oof_holdout_groups_sha256": r5_evaluator.logical_group_list_sha256(
                    holdout
                ),
                "train_artifact": str(train_path.resolve()),
                "train_artifact_sha256": r5_evaluator.sha256_path(train_path),
                "train_payload_sha256": train["payload_sha256"],
                "holdout_artifact": str(holdout_path.resolve()),
                "holdout_artifact_sha256": r5_evaluator.sha256_path(holdout_path),
                "holdout_payload_sha256": test["payload_sha256"],
            }
        )
    manifest = {
        "format": r5_evaluator.MATERIALIZATION_FORMAT,
        "status": "complete_development_only",
        "base_checkpoint_sha256": checkpoint_sha,
        "event_spec_sha256": event_spec_sha,
        "development_groups": groups,
        "development_groups_sha256": r5_evaluator.logical_group_list_sha256(groups),
        "folds": folds,
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
        "timing_scope": "adaptive_development_only_designed_after_v7_collection_started",
        "prospective_claim_for_v8": False,
    }
    manifest["materialization_sha256"] = canonical_sha256(manifest)
    manifest_path = materialized / "materialization_manifest.json"
    _write_json(manifest_path, manifest)

    result_value = r5_evaluator.evaluate_duration_hierarchy_oof(
        manifest_path, bootstrap_samples=100, bootstrap_seed=19
    )
    assert result_value["summary"]["passed"] is True
    output = root / "r5_duration_result"
    r5_evaluator.write_duration_hierarchy_evaluation(output, result_value)
    return {
        "manifest": manifest_path,
        "result": output / "duration_hierarchy_evaluation.json",
        "rows": output / "duration_hierarchy_rows.npz",
    }


def _freeze(evidence: dict[str, Path]) -> dict:
    return freezer.freeze_duration_activation(
        materialization_manifest=evidence["manifest"],
        r5_result_json=evidence["result"],
        r5_rows_npz=evidence["rows"],
    )


def test_freeze_authenticates_d250_and_limits_permission(
    signed_evidence: dict[str, Path], tmp_path: Path
) -> None:
    activation = _freeze(signed_evidence)
    assert activation["development_coverage"][
        "five_holdouts_cover_each_group_exactly_once"
    ] is True
    assert activation["development_coverage"]["logical_groups"] == 250
    assert activation["permissions"] == {
        "duration_prediction_adapter": True,
        "actor_control": False,
        "policy_modification": False,
        "reward_or_value": False,
        "candidate_ranking": False,
        "selector": False,
    }
    assert activation["fresh50_confirmation_authorized"] is False
    assert activation["interface_actor_policy_agnostic"] is True
    assert "actor_or_policy_specific" not in activation
    assert activation["transfer_claim_authorized"] is False
    assert activation["empirical_evidence_scope"] == {
        "policy": "openvla",
        "policy_id": 0,
        "body": "piper_piper_0.6",
        "body_id": 0,
        "one_cell_only": True,
        "cross_body_validated": False,
        "cross_policy_validated": False,
    }
    output = tmp_path / "duration_activation.json"
    freezer.write_activation(output, activation)
    assert output.stat().st_mode & 0o222 == 0
    assert load_duration_activation(output)["activation_sha256"] == activation[
        "activation_sha256"
    ]


def test_adapter_formula_and_unknown_body_fallback(
    signed_evidence: dict[str, Path]
) -> None:
    activation = _freeze(signed_evidence)
    frozen = np.asarray([4.0, 5.0, 6.0])
    result = predict_duration_candidates(
        activation,
        body_registry_contract=activation["empirical_registry_contract"],
        current_event_id=np.asarray([0, 0, 999]),
        body_id=np.asarray([0, 999, 999]),
        frozen_duration_log_mean=frozen,
    )
    expected = result["baseline_log_location"] + 0.375 * (
        frozen - result["baseline_log_location"]
    )
    np.testing.assert_allclose(result["duration_log_location"], expected)
    assert result["source_kind"][1] == "event"
    assert result["source_kind"][2] == "global"
    assert np.all(result["source_support"] >= 20)
    assert result["out_of_empirical_body_scope"].tolist() == [False, True, True]
    assert not result["transfer_claim_authorized"].any()
    assert not result["cross_body_validated"].any()
    assert not result["cross_policy_validated"].any()
    assert "reward" not in inspect.signature(predict_duration_candidates).parameters
    assert "candidate_score" not in result
    with pytest.raises(TypeError, match="body_registry_contract"):
        predict_duration_candidates(
            activation,
            current_event_id=np.asarray([0]),
            body_id=np.asarray([0]),
            frozen_duration_log_mean=np.asarray([1.0]),
        )
    tampered_registry = copy.deepcopy(activation["empirical_registry_contract"])
    tampered_registry["observed_bodies"][0]["body_id"] = 7
    with pytest.raises(RuntimeError, match="registry signature mismatch"):
        predict_duration_candidates(
            activation,
            body_registry_contract=tampered_registry,
            current_event_id=np.asarray([0]),
            body_id=np.asarray([0]),
            frozen_duration_log_mean=np.asarray([1.0]),
        )


def test_activation_and_r5_tampering_fail_closed(
    signed_evidence: dict[str, Path], tmp_path: Path
) -> None:
    activation = _freeze(signed_evidence)
    tampered = copy.deepcopy(activation)
    tampered["permissions"]["selector"] = True
    with pytest.raises(RuntimeError, match="activation signature mismatch"):
        validate_duration_activation(tampered)
    tampered["activation_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "activation_sha256"}
    )
    with pytest.raises(RuntimeError, match="permission/protocol changed"):
        validate_duration_activation(tampered)

    result = json.loads(signed_evidence["result"].read_text(encoding="utf-8"))
    result["passed"] = False
    tampered_result = tmp_path / "tampered_r5.json"
    _write_json(tampered_result, result)
    with pytest.raises(RuntimeError, match="R5 duration result signature mismatch"):
        freezer.freeze_duration_activation(
            materialization_manifest=signed_evidence["manifest"],
            r5_result_json=tampered_result,
            r5_rows_npz=signed_evidence["rows"],
        )

    rows = tmp_path / "tampered_rows.npz"
    rows.write_bytes(signed_evidence["rows"].read_bytes() + b"tamper")
    result = json.loads(signed_evidence["result"].read_text(encoding="utf-8"))
    result["row_arrays"]["path"] = str(rows.resolve())
    result["result_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    rebound_result = tmp_path / "rebound_r5.json"
    _write_json(rebound_result, result)
    with pytest.raises(RuntimeError, match="R5 passed evidence"):
        freezer.freeze_duration_activation(
            materialization_manifest=signed_evidence["manifest"],
            r5_result_json=rebound_result,
            r5_rows_npz=rows,
        )


def test_duplicate_holdout_ownership_fails_before_payload_read(
    signed_evidence: dict[str, Path], tmp_path: Path, monkeypatch
) -> None:
    manifest = json.loads(signed_evidence["manifest"].read_text(encoding="utf-8"))
    duplicate = manifest["folds"][0]["oof_holdout_groups"][0]
    manifest["folds"][1]["oof_holdout_groups"][0] = duplicate
    manifest["folds"][1]["oof_holdout_groups"] = sorted(
        manifest["folds"][1]["oof_holdout_groups"]
    )
    manifest["folds"][1]["oof_holdout_groups_sha256"] = (
        r5_evaluator.logical_group_list_sha256(
            manifest["folds"][1]["oof_holdout_groups"]
        )
    )
    manifest["materialization_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "materialization_sha256"}
    )
    path = tmp_path / "duplicate_manifest.json"
    _write_json(path, manifest)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("payload read before ownership validation")

    monkeypatch.setattr(r5_evaluator, "_authenticate_artifact", forbidden)
    with pytest.raises(RuntimeError, match="ownership changed|multiple holdout owners"):
        freezer.freeze_duration_activation(
            materialization_manifest=path,
            r5_result_json=signed_evidence["result"],
            r5_rows_npz=signed_evidence["rows"],
        )
    assert called is False


def test_low_support_final_fit_and_fresh_paths_fail_closed() -> None:
    groups = [f"task|body|{index:03d}" for index in range(250)]
    with pytest.raises(RuntimeError, match="fixed support of 20"):
        fit_final_duration_hierarchy(
            duration=np.ones(250),
            duration_observed=np.asarray([True] * 19 + [False] * 231),
            dense_mask=np.ones(250, dtype=bool),
            current_event_id=np.zeros(250, dtype=np.int64),
            body_id=np.zeros(250, dtype=np.int64),
            logical_group=np.asarray(groups),
            development_groups=groups,
            materialization_groups_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="duration_observed must be binary"):
        fit_final_duration_hierarchy(
            duration=np.ones(250),
            duration_observed=np.asarray([0.5] + [1.0] * 249),
            dense_mask=np.ones(250, dtype=bool),
            current_event_id=np.zeros(250, dtype=np.int64),
            body_id=np.zeros(250, dtype=np.int64),
            logical_group=np.asarray(groups),
            development_groups=groups,
            materialization_groups_sha256="d" * 64,
        )
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        freezer.freeze_duration_activation(
            materialization_manifest=Path("/srv/Fresh50/materialization.json"),
            r5_result_json=Path("/safe/result.json"),
            r5_rows_npz=Path("/safe/rows.npz"),
        )
