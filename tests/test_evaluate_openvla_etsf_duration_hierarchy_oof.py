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

import evaluate_openvla_etsf_duration_hierarchy_oof as evaluator  # noqa: E402
from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    frozen_tensor_mapping_sha256,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    structured_payload_sha256,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _record(
    group: str,
    fold: int,
    role: str,
    *,
    divergence: bool,
    count: int,
) -> dict:
    duration = torch.arange(1, count + 1, dtype=torch.float32) * 10.0
    current = torch.zeros(count, dtype=torch.int64)
    clock = current.clone()
    if divergence:
        clock[0] = 1
    batch = {
        "duration": duration,
        "duration_observed": torch.ones(count, dtype=torch.bool),
        "dense_mask": torch.ones(count, dtype=torch.bool),
        "current_event_id": current,
        "clock_event_id": clock,
        "body_id": torch.zeros(count, dtype=torch.int64),
    }
    factual = {
        "transition": torch.zeros((count, 2)),
        "duration_selected_log_mean": torch.log1p(duration),
    }
    return {
        "logical_group_key": group,
        "split_role": role,
        "outer_fold_id": fold,
        "batch": batch,
        "factual_outputs": factual,
        "factual_outputs_sha256": frozen_tensor_mapping_sha256(factual),
        "factual_outputs_require_grad": False,
    }


def _payload(
    groups: list[str],
    *,
    all_training_groups: list[str],
    holdout_groups: list[str],
    fold: int,
    role: str,
    state_sha: str,
    rows_per_record: int,
) -> dict:
    record_role = "outer_training" if role == "train" else "outer_holdout"
    records = [
        _record(
            group,
            fold,
            record_role,
            divergence=(role == "holdout" and index == 0),
            count=rows_per_record,
        )
        for index, group in enumerate(groups)
    ]
    provenance = {
        "outer_fold_id": fold,
        "outer_training_groups": all_training_groups,
        "outer_training_groups_sha256": evaluator.logical_group_list_sha256(
            all_training_groups
        ),
        "oof_holdout_groups": holdout_groups,
        "oof_holdout_groups_sha256": evaluator.logical_group_list_sha256(
            holdout_groups
        ),
        "target_outer_fold_labels_used": False,
        "factual_outputs_frozen": True,
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
        "format": evaluator.TRAINING_FORMAT if role == "train" else evaluator.HOLDOUT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": 2},
        "batches": records,
        "provenance": provenance,
        "materialization_audit": {
            "factual_state_sha256_before": state_sha,
            "factual_state_sha256_after": state_sha,
            "factual_state_bit_exact": True,
        },
    }
    value["payload_sha256"] = structured_payload_sha256(value)
    return value


def _fixture(tmp_path: Path, *, rows_per_record: int = 4) -> Path:
    assert rows_per_record > 0
    root = tmp_path / "r3_materialized"
    root.mkdir(parents=True)
    groups = [f"move_can_pot|piper|{index:03d}" for index in range(10)]
    state_sha = "a" * 64
    fold_rows = []
    for fold in range(5):
        holdout = groups[fold * 2 : fold * 2 + 2]
        training = sorted(set(groups) - set(holdout))
        train_payload = _payload(
            training,
            all_training_groups=training,
            holdout_groups=holdout,
            fold=fold,
            role="train",
            state_sha=state_sha,
            rows_per_record=rows_per_record,
        )
        holdout_payload = _payload(
            holdout,
            all_training_groups=training,
            holdout_groups=holdout,
            fold=fold,
            role="holdout",
            state_sha=state_sha,
            rows_per_record=rows_per_record,
        )
        train_path = root / f"fold_{fold}_train.pt"
        holdout_path = root / f"fold_{fold}_holdout.pt"
        torch.save(train_payload, train_path)
        torch.save(holdout_payload, holdout_path)
        fold_rows.append(
            {
                "outer_fold_id": fold,
                "training_groups": training,
                "training_groups_sha256": evaluator.logical_group_list_sha256(training),
                "oof_holdout_groups": holdout,
                "oof_holdout_groups_sha256": evaluator.logical_group_list_sha256(holdout),
                "train_artifact": str(train_path.resolve()),
                "train_artifact_sha256": evaluator.sha256_path(train_path),
                "train_payload_sha256": train_payload["payload_sha256"],
                "holdout_artifact": str(holdout_path.resolve()),
                "holdout_artifact_sha256": evaluator.sha256_path(holdout_path),
                "holdout_payload_sha256": holdout_payload["payload_sha256"],
            }
        )
    manifest = {
        "format": evaluator.MATERIALIZATION_FORMAT,
        "status": "complete_development_only",
        "timing_scope": "adaptive_development_only_designed_after_v7_collection_started",
        "prospective_claim_for_v8": False,
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
        "development_groups": groups,
        "development_groups_sha256": evaluator.logical_group_list_sha256(groups),
        "folds": fold_rows,
    }
    manifest["materialization_sha256"] = evaluator.canonical_sha256(manifest)
    path = root / "materialization_manifest.json"
    _write_json(path, manifest)
    return path


def _resign_payload_and_manifest(
    manifest_path: Path, *, fold: int, role: str, payload: dict
) -> None:
    payload["payload_sha256"] = structured_payload_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    artifact = manifest_path.parent / f"fold_{fold}_{role}.pt"
    torch.save(payload, artifact)
    manifest = evaluator._load_json(manifest_path)
    unsigned = {
        key: value for key, value in manifest.items() if key != "materialization_sha256"
    }
    unsigned["folds"][fold][f"{role}_payload_sha256"] = payload["payload_sha256"]
    unsigned["folds"][fold][f"{role}_artifact_sha256"] = evaluator.sha256_path(
        artifact
    )
    unsigned["materialization_sha256"] = evaluator.canonical_sha256(unsigned)
    _write_json(manifest_path, unsigned)


def test_evaluation_authenticates_train_then_contract_then_holdout(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    value = evaluator.evaluate_duration_hierarchy_oof(
        manifest, bootstrap_samples=200, bootstrap_seed=4
    )
    summary = value["summary"]
    arrays = value["arrays"]
    assert summary["minimum_applied_source_support"] >= 20
    assert set(summary["implementation_files"]) == set(
        evaluator.IMPLEMENTATION_FILENAMES
    )
    assert all(
        len(value) == 64 for value in summary["implementation_files"].values()
    )
    assert summary["current_clock_divergence_rows"] == 5
    assert summary["current_event_source"].endswith("never_clock_proxy")
    divergent = arrays["current_clock_divergence"].astype(bool)
    assert divergent.sum() == 5
    assert set(arrays["source_key"][divergent].tolist()) == {"0:0"}
    assert len(arrays["owner_fold_id"]) == len(arrays["logical_group"]) == len(
        arrays["row_index"]
    )
    trace = summary["read_trace"]
    for fold in range(5):
        signed = next(
            item["sequence"]
            for item in trace
            if item["fold"] == fold and item["event"] == "duration_v2_contract_signed"
        )
        opened = next(
            item["sequence"]
            for item in trace
            if item["fold"] == fold and item["event"] == "holdout_payload_deserialized"
        )
        assert signed < opened


def test_write_outputs_signed_json_and_aligned_npz(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    value = evaluator.evaluate_duration_hierarchy_oof(
        manifest, bootstrap_samples=100, bootstrap_seed=2
    )
    output = tmp_path / "r5_duration_output"
    result = evaluator.write_duration_hierarchy_evaluation(output, value)
    result_path = output / "duration_hierarchy_evaluation.json"
    arrays_path = output / "duration_hierarchy_rows.npz"
    recorded = evaluator._load_json(result_path)
    unsigned = {key: item for key, item in recorded.items() if key != "result_sha256"}
    assert recorded["result_sha256"] == evaluator.canonical_sha256(unsigned)
    assert recorded["row_arrays"]["file_sha256"] == evaluator.sha256_path(arrays_path)
    arrays = np.load(arrays_path, allow_pickle=False)
    assert arrays["source_kind"].dtype.kind == "U"
    assert len(arrays["source_support"]) == result["row_arrays"]["rows"]


def test_tampered_artifact_and_factual_hash_fail_closed(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    artifact = manifest.parent / "fold_4_holdout.pt"
    with artifact.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="artifact SHA mismatch"):
        evaluator.evaluate_duration_hierarchy_oof(manifest, bootstrap_samples=100)

    manifest = _fixture(tmp_path / "second")
    train_path = manifest.parent / "fold_0_train.pt"
    payload = torch.load(train_path, map_location="cpu", weights_only=True)
    payload["batches"][0]["factual_outputs_sha256"] = "0" * 64
    _resign_payload_and_manifest(manifest, fold=0, role="train", payload=payload)
    with pytest.raises(RuntimeError, match="factual tensor hash changed"):
        evaluator.evaluate_duration_hierarchy_oof(manifest, bootstrap_samples=100)


def test_holdout_cannot_be_opened_before_signed_contract(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _fixture(tmp_path)
    manifest = evaluator._load_json(manifest_path)
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("holdout was deserialized")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(RuntimeError, match="before duration contract"):
        evaluator._authenticate_artifact(
            manifest_root=manifest_path.parent,
            fold_row=manifest["folds"][0],
            fold_id=0,
            role="holdout",
            trace=[],
            signed_duration_contract=None,
        )
    assert called is False


def test_outer_training_leakage_and_low_support_fail_closed(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    train_path = manifest.parent / "fold_0_train.pt"
    payload = torch.load(train_path, map_location="cpu", weights_only=True)
    payload["batches"][0]["split_role"] = "outer_holdout"
    _resign_payload_and_manifest(manifest, fold=0, role="train", payload=payload)
    with pytest.raises(RuntimeError, match="record role changed"):
        evaluator.evaluate_duration_hierarchy_oof(manifest, bootstrap_samples=100)

    # Each fold has eight training groups * two rows = 16 observed rows, so
    # even the global source cannot reach the fixed support of 20.  Ownership
    # remains otherwise valid and the failure therefore occurs inside the
    # authenticated evaluator's hierarchy fit.
    manifest = _fixture(tmp_path / "low", rows_per_record=2)
    with pytest.raises(RuntimeError, match="fixed support of 20"):
        evaluator.evaluate_duration_hierarchy_oof(manifest, bootstrap_samples=100)


def test_clock_only_materialization_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    train_path = manifest.parent / "fold_0_train.pt"
    payload = torch.load(train_path, map_location="cpu", weights_only=True)
    del payload["batches"][0]["batch"]["current_event_id"]
    payload["batches"][0]["factual_outputs_sha256"] = frozen_tensor_mapping_sha256(
        payload["batches"][0]["factual_outputs"]
    )
    _resign_payload_and_manifest(manifest, fold=0, role="train", payload=payload)
    with pytest.raises(RuntimeError, match="clock_event_id cannot substitute"):
        evaluator.evaluate_duration_hierarchy_oof(manifest, bootstrap_samples=100)


def test_fresh_named_paths_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        evaluator.evaluate_duration_hierarchy_oof(
            Path("/srv/Fresh50/materialization_manifest.json"),
            bootstrap_samples=100,
        )
