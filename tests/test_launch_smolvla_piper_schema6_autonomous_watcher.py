from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_schema6_autonomous_watcher as watcher  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _event_spec(path: Path) -> Path:
    _write_json(
        path,
        {
            "calibration": {"move_can_pot": {"moving": "can", "anchor": None}},
            "chains": {
                "move_can_pot": {
                    "merge_e1_e2": True,
                    "chain": ["e0", "e12", "e3", "e4", "eK"],
                }
            },
        },
    )
    return path


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _synthetic_lobo_audit(lobo_root: Path) -> dict:
    deployment = {
        "path": str(
            watcher.EXPECTED_SOURCE_ROOT
            / watcher.EXPECTED_SOURCE_ENSEMBLE_RELATIVE
        ),
        "sha256": "e" * 64,
        "policy": "smolvla",
        "checkpoint_family": "smolvla_native_event_world_model",
        "policy_feature_action_bridge_contract_sha256": "3" * 64,
        "source_native_checkpoint": True,
    }
    native_binding = {
        "source_training_root": str(watcher.EXPECTED_SOURCE_ROOT),
        "source_launch_plan": {
            "path": str(watcher.EXPECTED_SOURCE_ROOT / "launch_plan.json"),
            "file_sha256": watcher.EXPECTED_SOURCE_PLAN_SHA256,
            "logical_sha256": watcher.EXPECTED_SOURCE_STATIC_PLAN_SHA256,
        },
        "source_launcher_sha256": watcher.EXPECTED_SOURCE_LAUNCHER_SHA256,
        "source_implementation_bundle_sha256": (
            watcher.EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
        ),
        "source_binding_receipt_path": str(lobo_root / "source_binding_receipt.json"),
        "source_binding_receipt_file_sha256": "1" * 64,
        "source_binding_sha256": "2" * 64,
        "deployment_rerank_checkpoint": deployment,
        "policy_feature_action_bridge_sha256": "3" * 64,
        "lobo_checkpoints_rerank_authorized": False,
    }
    source_binding_receipt = {
        "path": str(lobo_root / "source_binding_receipt.json"),
        "file_sha256": "1" * 64,
        "binding_sha256": "2" * 64,
        "source_final_receipt_file_sha256": "7" * 64,
        "source_final_receipt_logical_sha256": "8" * 64,
        "policy_feature_action_bridge_contract_sha256": "3" * 64,
        "deployment_rerank_checkpoint": deployment,
        "lobo_checkpoints_rerank_authorized": False,
    }
    stage_audits = {
        stage: {
            "held_out_body": body,
            "returncode": 0,
            "stage_receipt_sha256": f"{index + 1:x}" * 64,
            "run_exit_sha256": f"{index + 3:x}" * 64,
            "log_sha256": f"{index + 5:x}" * 64,
            "summary_sha256": f"{index + 7:x}" * 64,
            "artifact_inventory_sha256": f"{index + 9:x}" * 64,
        }
        for index, (stage, body) in enumerate(watcher.LOBO_STAGES)
    }
    value = {
        "status": "verified_piper_then_ur5_lobo_terminal_exit_zero",
        "lobo_root": str(lobo_root),
        "lobo_launcher_sha256": watcher.EXPECTED_LOBO_LAUNCHER_SHA256,
        "final_receipt_sha256": "4" * 64,
        "final_receipt_logical_sha256": "5" * 64,
        "static_plan_sha256": watcher.EXPECTED_LOBO_STATIC_PLAN_SHA256,
        "source_binding_receipt": source_binding_receipt,
        "deployment_rerank_checkpoint": deployment,
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
        "watcher_run_exit_sha256": "6" * 64,
        "execution_order": [stage for stage, _body in watcher.LOBO_STAGES],
        "transitive_source63_audit_sha256": "d" * 64,
        "native_source_binding_audit": native_binding,
        "native_source_binding_audit_sha256": watcher.canonical_sha256(
            native_binding
        ),
        "stage_audits": stage_audits,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
    }
    value["summary_sha256"] = watcher.canonical_sha256(value)
    return value


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source63_training"
    training = source / "counterfactual_training"
    (source / "stage_receipts").mkdir(parents=True)
    (source / "logs").mkdir()
    (training / "members").mkdir(parents=True)
    initialized = source / "smolvla_schema5_native_initialized.pt"
    initialized.write_bytes(b"initialized-native-core")

    members = []
    member_logs = []
    for seed in watcher.SOURCE_MEMBER_SEEDS:
        directory = training / "members" / f"seed_{seed}"
        directory.mkdir()
        checkpoint = directory / "event_world_model_counterfactual_best.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        train_log = directory / "train_log.jsonl"
        train_log.write_text(f'{{"seed":{seed},"step":3000}}\n', encoding="utf-8")
        members.append(
            {
                "path": str(checkpoint.resolve()),
                "sha256": watcher.file_sha256(checkpoint),
                "seed": seed,
            }
        )
        member_logs.append(watcher.file_sha256(train_log))
    ensemble = training / "counterfactual_ensemble.pt"
    ensemble.write_bytes(b"ensemble")
    manifest = {
        "format": "etsf_counterfactual_ensemble_v1",
        "ensemble_checkpoint": {
            "path": str(ensemble.resolve()),
            "sha256": watcher.file_sha256(ensemble),
        },
        "members": members,
        "test_policy": "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened",
    }
    manifest_path = training / "ensemble_manifest.json"
    _write_json(manifest_path, manifest)
    training_audit = {
        "status": watcher.SOURCE_TRAINING_STATUS,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": watcher.file_sha256(manifest_path),
        "ensemble_checkpoint": str(ensemble.resolve()),
        "ensemble_checkpoint_sha256": watcher.file_sha256(ensemble),
        "member_count": 5,
        "member_seeds": list(watcher.SOURCE_MEMBER_SEEDS),
        "member_proof_sha256": [f"{index + 1:064x}" for index in range(5)],
        "member_training_log_sha256": member_logs,
        "member_training_steps_verified": [watcher.SOURCE_TRAINING_STEPS] * 5,
        "target_data_read": False,
        "target_labels_read": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
        "test_hdf_identity_attrs_opened": 5,
    }

    stage_receipts = {}
    for stage in watcher.SOURCE_STAGE_NAMES:
        log = source / "logs" / f"{stage}.log"
        log.write_text(f"{stage} complete\n", encoding="utf-8")
        argv = ["/venv/bin/python", f"/code/{stage}.py"]
        receipt = {
            "format": watcher.SOURCE_FORMAT,
            "stage": stage,
            "status": "complete",
            "returncode": 0,
            "argv": argv,
            "argv_sha256": watcher.canonical_sha256(argv),
            "log": str(log.resolve()),
            "log_sha256": watcher.file_sha256(log),
        }
        if stage == watcher.SOURCE_STAGE_NAMES[1]:
            receipt["artifact_audit"] = training_audit
        stage_receipts[stage] = receipt
        _write_json(source / "stage_receipts" / f"{stage}.json", receipt)

    plan = {
        "format": watcher.SOURCE_FORMAT,
        "output_root": str(source.resolve()),
        "fresh_inputs_accepted": False,
        "hdf5_opened_during_static_preflight": False,
    }
    plan["static_plan_sha256"] = watcher.canonical_sha256(plan)
    _write_json(source / "launch_plan.json", plan)
    execution = {
        "format": watcher.SOURCE_FORMAT,
        "execution_order": list(watcher.SOURCE_STAGE_NAMES),
        "test_hdf_label_datasets_opened": 0,
    }
    execution["execution_plan_sha256"] = watcher.canonical_sha256(execution)
    _write_json(source / "execution_plan.json", execution)
    state = {
        "format": "etsf_smolvla_schema5_source63_native_training_state_v1",
        "status": watcher.SOURCE_TERMINAL_STATUS,
        "stage_results": {
            name: {"returncode": receipt["returncode"]}
            for name, receipt in stage_receipts.items()
        },
        "test_hdf_label_datasets_opened": 0,
    }
    _write_json(source / "launch_state.json", state)
    inventory = {
        "format": watcher.SOURCE_FORMAT,
        "status": "complete_pre_freeze_inventory",
        "file_count": 0,
        "files": [],
    }
    inventory["inventory_sha256"] = watcher.canonical_sha256(inventory)
    _write_json(source / "artifact_inventory.json", inventory)
    final = {
        "format": watcher.SOURCE_FORMAT,
        "status": watcher.SOURCE_TERMINAL_STATUS,
        "static_plan_sha256": plan["static_plan_sha256"],
        "execution_plan_sha256": execution["execution_plan_sha256"],
        "snapshot_sha256": "a" * 64,
        "artifact_inventory_sha256": inventory["inventory_sha256"],
        "initialized_checkpoint_sha256": watcher.file_sha256(initialized),
        "training_audit": training_audit,
        "target_data_read": False,
        "target_labels_read": False,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
        "test_hdf_identity_attrs_opened": 5,
        "artifacts_frozen_read_only": True,
    }
    _write_json(source / "final_receipt.json", final)
    trap = source / "source_snapshot" / "heldout_test_trap.hdf5"
    trap.parent.mkdir()
    trap.write_bytes(b"must-never-open")
    _freeze(source)
    lobo = tmp_path / "lobo_aggregate"
    monkeypatch.setattr(watcher, "DESIGNATED_LOBO_ROOT", lobo)
    return source


def _static_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source_training"
    lobo = tmp_path / "lobo_aggregate"
    code = tmp_path / "code_r6g"
    (code / "scripts").mkdir(parents=True)
    source.mkdir()
    r6f = tmp_path / "r6f.json"
    event = _event_spec(tmp_path / "events.json")
    _write_json(
        r6f,
        {"status": "preregistered_R6f_feasibility_simulation_only_not_executed"},
    )
    r6f.chmod(0o444)
    event.chmod(0o444)
    code.chmod(0o555)
    monkeypatch.setattr(watcher, "DESIGNATED_LOBO_ROOT", lobo)
    monkeypatch.setattr(watcher, "DESIGNATED_CODE_ROOT", code)
    monkeypatch.setattr(watcher, "DESIGNATED_PYTHON", Path(sys.executable))
    monkeypatch.setattr(
        watcher,
        "implementation_closure",
        lambda _root: {"scripts/a.py": {"path": "/bound/a.py", "sha256": "a" * 64, "size": 1}},
    )
    args = argparse.Namespace(
        command="preflight",
        source_training_root=source,
        lobo_root=lobo,
        code_root=code,
        r6f_preregistration=r6f,
        event_spec=event,
        output=tmp_path / "watcher_output",
        python_bin=Path(sys.executable),
        gpu_index=0,
        gpu_lock=tmp_path / "gpu.lock",
        poll_seconds=1.0,
        source_timeout_seconds=0.0,
        lobo_timeout_seconds=0.0,
        gpu_timeout_seconds=0.0,
        materializer_timeout_seconds=10.0,
        freezer_timeout_seconds=10.0,
        collection_timeout_seconds=0.0,
        max_episode_steps=4,
        omp_threads=2,
        detach_receipt=tmp_path / "detach.json",
        detach_log=tmp_path / "daemon.log",
    )
    return args


def test_sensitive_paths_are_rejected_before_resolution(tmp_path: Path) -> None:
    bad = tmp_path / "sensitive_fresh_lane" / "x"
    with pytest.raises(watcher.WatcherContractError, match="forbidden"):
        watcher.reject_path_text(bad, "bad")
    with pytest.raises(watcher.WatcherContractError, match="embeds"):
        watcher._audit_embedded_paths({"path": "/srv/confirmation_lane/input.json"})


def _signed_r6f_with_legacy_seed_path(path: Path) -> tuple[Path, str]:
    legacy = "/sealed/prior_development_confirmation/seed_manifest.json"
    inherited = {
        "development_seed": {
            "path": legacy,
            "sha256": "1" * 64,
            "seed_registry": "explicit_v7_prospective_development",
            "requested_seed": watcher.FIXED_REQUESTED_SEED,
            "expected_resolved_seed": watcher.FIXED_REQUESTED_SEED,
            "fresh_confirmation_eligible": False,
            "label_free": True,
        }
    }
    base = {
        "status": "preregistered_R6f_feasibility_simulation_only_not_executed",
        "inherited_R6e_contract": inherited,
        "inherited_R6e_contract_sha256": watcher.canonical_sha256(inherited),
    }
    value = {**base, "preregistration_sha256": watcher.canonical_sha256(base)}
    _write_json(path, value)
    return path, legacy


def test_signed_legacy_seed_path_is_committed_but_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r6f, legacy = _signed_r6f_with_legacy_seed_path(tmp_path / "r6f.json")
    original_open = Path.open
    opened: list[str] = []

    def guarded_open(self: Path, *args: object, **kwargs: object):
        text = str(self)
        opened.append(text)
        if "confirmation" in text.casefold() or "fresh" in text.casefold():
            raise AssertionError("protected lineage path was opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    value, audit = watcher._load_signed_r6f_lineage_metadata(r6f)
    assert value["inherited_R6e_contract"]["development_seed"]["path"] == legacy
    assert audit["legacy_sensitive_path_present"] is True
    assert audit["legacy_sensitive_path_opened"] is False
    assert legacy not in json.dumps(audit, sort_keys=True)
    assert opened == [str(r6f)]


def test_legacy_seed_path_exception_is_exact_and_signature_bound(tmp_path: Path) -> None:
    r6f, _legacy = _signed_r6f_with_legacy_seed_path(tmp_path / "r6f.json")
    value = json.loads(r6f.read_text(encoding="utf-8"))
    value["another_path"] = "/unsafe/fresh_extra/file.json"
    _write_json(r6f, value)
    with pytest.raises(watcher.WatcherContractError, match="unapproved"):
        watcher._load_signed_r6f_lineage_metadata(r6f)

    r6f, _legacy = _signed_r6f_with_legacy_seed_path(tmp_path / "r6f2.json")
    value = json.loads(r6f.read_text(encoding="utf-8"))
    value["inherited_R6e_contract"]["development_seed"]["requested_seed"] += 1
    _write_json(r6f, value)
    with pytest.raises(watcher.WatcherContractError, match="SHA mismatch"):
        watcher._load_signed_r6f_lineage_metadata(r6f)


def test_real_stage1_event_shape_accepts_empty_anchor_as_no_anchor(
    tmp_path: Path,
) -> None:
    event = tmp_path / "canonical_event_spec.json"
    payload = {
        "calibration": {
            "move_can_pot": {
                "moving": "can",
                "anchor": "",
                "offset": [0.0, 0.0, 0.0],
                "centers": [[0.12, -0.03, 0.84], [0.14, -0.01, 0.84]],
                "tau_v": 0.004,
                "delta_z": 0.015,
                "tau_d": 0.025,
                "k": 3,
            }
        },
        "chains": {
            "move_can_pot": {
                "merge_e1_e2": True,
                "chain": ["e0", "e12", "e3", "e4", "eK"],
            }
        },
    }
    _write_json(event, payload)
    watcher._validate_event_spec(event)
    assert payload["calibration"]["move_can_pot"]["anchor"] == ""
    assert not payload["calibration"]["move_can_pot"]["anchor"]


@pytest.mark.parametrize("bad_anchor", ["table", "can", "none", 0, False])
def test_event_spec_still_rejects_noncanonical_anchor_values(
    tmp_path: Path, bad_anchor: object
) -> None:
    event = _event_spec(tmp_path / "canonical_event_spec.json")
    payload = json.loads(event.read_text(encoding="utf-8"))
    payload["calibration"]["move_can_pot"]["anchor"] = bad_anchor
    _write_json(event, payload)
    with pytest.raises(watcher.WatcherContractError, match="canonical move_can_pot"):
        watcher._validate_event_spec(event)


def test_source_terminal_summary_proves_zero_and_never_opens_test_hdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    original_open = Path.open
    touched = []

    def guarded_open(self, *args, **kwargs):
        if self.suffix.lower() in watcher.HDF_SUFFIXES:
            touched.append(self)
            raise AssertionError("HDF must not be opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    audit = watcher.validate_source_training_summary(source)
    assert audit["stage_returncodes"] == {
        name: 0 for name in watcher.SOURCE_STAGE_NAMES
    }
    assert audit["member_seeds"] == list(watcher.SOURCE_MEMBER_SEEDS)
    assert audit["source_test_hdf5_opened_by_this_watcher"] == 0
    assert touched == []


def test_source_checkpoint_or_recorded_exit_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    source.chmod(0o755)
    member = next((source / "counterfactual_training" / "members").glob("*/event*.pt"))
    member.write_bytes(b"tampered")
    _freeze(source)
    with pytest.raises(watcher.WatcherContractError, match="checkpoint SHA"):
        watcher.validate_source_training_summary(source)


def test_waiting_for_source_only_publishes_hdf_blind_heartbeat(
    tmp_path: Path,
) -> None:
    source = tmp_path / "running_source"
    source.mkdir()
    (source / "unrelated_test.hdf5").write_bytes(b"trap")
    state_path = tmp_path / "state.json"
    state = {}
    with pytest.raises(TimeoutError):
        watcher.wait_for_source_training(
            source,
            state=state,
            state_path=state_path,
            poll_seconds=0.001,
            timeout_seconds=0.001,
            sleep=lambda _: None,
        )
    saved = json.loads(state_path.read_text())
    assert saved["source_summary_read"] is False
    assert saved["source_hdf5_opened"] == 0
    assert saved["source_test_hdf5_opened"] == 0


def test_source_final_is_not_consumed_until_root_freeze_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path, monkeypatch)
    source.chmod(0o755)
    (source / "final_receipt.json").chmod(0o644)
    sleeps = []

    def publish_freeze(_seconds: float) -> None:
        sleeps.append(True)
        (source / "final_receipt.json").chmod(0o444)
        source.chmod(0o555)

    state = {}
    audit = watcher.wait_for_source_training(
        source,
        state=state,
        state_path=tmp_path / "source_wait_state.json",
        poll_seconds=0.01,
        timeout_seconds=1,
        sleep=publish_freeze,
    )
    assert sleeps == [True]
    assert audit["status"] == "verified_source63_terminal_exit_zero_and_summary"


def _lobo_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "lobo_aggregate"
    piper_output = tmp_path / "lobo_piper_output"
    ur5_output = tmp_path / "lobo_ur5_output"
    outputs = {"piper": piper_output, "ur5-wsg": ur5_output}
    root.mkdir()
    source_root = tmp_path / "source_training"
    source_root.mkdir()
    source_launcher_sha = "7" * 64
    source_bundle_sha = "6" * 64
    source_plan_base = {
        "output_root": str(source_root),
        "implementation_bundle_sha256": source_bundle_sha,
        "implementation_files": {
            "scripts/launch_smolvla_schema5_source63_native_training.py": {
                "path": str(source_root / "source_launcher.py"),
                "sha256": source_launcher_sha,
                "size": 1,
            }
        },
    }
    source_plan_document = {
        **source_plan_base,
        "static_plan_sha256": watcher.canonical_sha256(source_plan_base),
    }
    source_plan = source_root / "launch_plan.json"
    _write_json(source_plan, source_plan_document)
    monkeypatch.setattr(watcher, "EXPECTED_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        watcher, "EXPECTED_SOURCE_PLAN_SHA256", watcher.file_sha256(source_plan)
    )
    monkeypatch.setattr(
        watcher,
        "EXPECTED_SOURCE_STATIC_PLAN_SHA256",
        source_plan_document["static_plan_sha256"],
    )
    monkeypatch.setattr(
        watcher, "EXPECTED_SOURCE_LAUNCHER_SHA256", source_launcher_sha
    )
    monkeypatch.setattr(
        watcher,
        "EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256",
        source_bundle_sha,
    )
    source_checkpoint = source_root / "counterfactual_ensemble.pt"
    source_checkpoint.write_bytes(b"native-source-ensemble")
    source_final = source_root / "final_receipt.json"
    _write_json(source_final, {"status": watcher.SOURCE_TERMINAL_STATUS})
    source_bridge_sha = "8" * 64
    source_audit = {
        "status": watcher.SOURCE_TERMINAL_STATUS,
        "final_receipt_path": str(source_final),
        "final_receipt_sha256": watcher.file_sha256(source_final),
        "final_receipt_logical_sha256": "1" * 64,
        "static_plan_sha256": source_plan_document["static_plan_sha256"],
        "execution_plan_sha256": "3" * 64,
        "ensemble_checkpoint": str(source_checkpoint),
        "ensemble_checkpoint_sha256": watcher.file_sha256(source_checkpoint),
        "policy_feature_action_bridge_sha256": source_bridge_sha,
        "member_count": 5,
        "member_training_steps_verified": [watcher.SOURCE_TRAINING_STEPS] * 5,
        "output_tree_read_only": True,
        "test_hdf_label_datasets_opened": 0,
    }
    stage_results = {}
    stage_lifecycles = []
    deployment_checkpoint = {
        "path": str(source_checkpoint),
        "sha256": watcher.file_sha256(source_checkpoint),
        "policy": "smolvla",
        "checkpoint_family": "smolvla_native_event_world_model",
        "policy_feature_action_bridge_contract_sha256": source_bridge_sha,
        "source_native_checkpoint": True,
    }
    source_binding_base = {
        "format": watcher.LOBO_SOURCE_BINDING_FORMAT,
        "status": watcher.LOBO_SOURCE_BINDING_STATUS,
        "source_training_root": str(source_root),
        "source_launch_plan": {
            "path": str(source_plan),
            "file_sha256": watcher.file_sha256(source_plan),
            "logical_sha256": source_plan_document["static_plan_sha256"],
        },
        "source_final_receipt": {
            "path": str(source_final),
            "file_sha256": watcher.file_sha256(source_final),
            "logical_sha256": source_audit["final_receipt_logical_sha256"],
            "status": watcher.SOURCE_TERMINAL_STATUS,
        },
        "deployment_rerank_checkpoint": deployment_checkpoint,
        "policy_feature_action_bridge_contract_sha256": source_bridge_sha,
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
    }
    source_binding_document = {
        **source_binding_base,
        "binding_sha256": watcher.canonical_sha256(source_binding_base),
    }
    source_binding_path = root / "source_binding_receipt.json"
    _write_json(source_binding_path, source_binding_document)
    source_binding_audit = {
        "path": str(source_binding_path),
        "file_sha256": watcher.file_sha256(source_binding_path),
        "binding_sha256": source_binding_document["binding_sha256"],
        "source_final_receipt_file_sha256": watcher.file_sha256(source_final),
        "source_final_receipt_logical_sha256": source_audit[
            "final_receipt_logical_sha256"
        ],
        "policy_feature_action_bridge_contract_sha256": source_bridge_sha,
        "deployment_rerank_checkpoint": deployment_checkpoint,
        "lobo_checkpoints_rerank_authorized": False,
    }
    for index, (stage, body) in enumerate(watcher.LOBO_STAGES, start=1):
        external = outputs[body]
        external.mkdir()
        summary = {
            "format": watcher.LOBO_OUTPUT_FORMAT,
            "status": watcher.LOBO_OUTPUT_TERMINAL_STATUS,
            "held_out_body": body,
            "estimand": "zero_target_label_leave_one_body_out_transfer",
            "target_development_opened_after_all_checkpoint_selection": True,
            "target_unused_train_payload_opened": 0,
            "sealed_test_evaluated": False,
            "test_group_hdf5_opened": 0,
        }
        summary_path = external / "lobo_training_summary.json"
        _write_json(summary_path, summary)
        stage_root = root / "stages" / stage
        stage_root.mkdir(parents=True)
        log = stage_root / "run.log"
        log.write_text(f"{stage} complete\n", encoding="utf-8")
        exit_path = stage_root / "run.exit"
        exit_path.write_bytes(b"0\n")
        argv = ["/venv/bin/python", "/code/train_lobo.py", "--held-out-body", body]
        process_pid = 43000 + index
        source_contract_base = {
            "format": watcher.LOBO_STAGE_SOURCE_BINDING_FORMAT,
            "stage": stage,
            "held_out_body": body,
            "argv_sha256": watcher.canonical_sha256(argv),
            "source_binding_receipt": {
                "path": str(source_binding_path),
                "file_sha256": watcher.file_sha256(source_binding_path),
                "binding_sha256": source_binding_document["binding_sha256"],
            },
            "deployment_rerank_checkpoint": deployment_checkpoint,
            "policy_feature_action_bridge_contract_sha256": source_bridge_sha,
            "lobo_checkpoints_rerank_authorized": False,
        }
        source_contract = {
            **source_contract_base,
            "contract_sha256": watcher.canonical_sha256(source_contract_base),
        }
        result = {
            "format": watcher.LOBO_WATCHER_FORMAT,
            "stage": stage,
            "held_out_body": body,
            "status": "complete",
            "returncode": 0,
            "pid": process_pid,
            "process_reaped": True,
            "process_group_id": process_pid,
            "process_group_isolated": True,
            "process_group_reaped": True,
            "argv": argv,
            "argv_sha256": watcher.canonical_sha256(argv),
            "output": str(external),
            "log": str(log),
            "run_exit": str(exit_path),
            "log_sha256": watcher.file_sha256(log),
            "run_exit_sha256": watcher.file_sha256(exit_path),
            "lobo_checkpoints_rerank_authorized": False,
            "deployment_rerank_checkpoint": deployment_checkpoint,
            "source_binding_contract": source_contract,
            "artifact_audit": {
                "status": watcher.LOBO_OUTPUT_TERMINAL_STATUS,
                "held_out_body": body,
                "summary_path": str(summary_path),
                "summary_sha256": watcher.file_sha256(summary_path),
                "artifact_inventory_sha256": ("a" if body == "piper" else "b") * 64,
                "target_unused_train_payload_opened": 0,
                "test_group_hdf5_opened": 0,
                "test_labels_read_by_watcher": False,
                "lobo_checkpoints_rerank_authorized": False,
                "deployment_rerank_checkpoint": deployment_checkpoint,
                "source_binding_contract": source_contract,
            },
        }
        stage_results[stage] = result
        stage_lifecycles.append(
            {
                "stage": stage,
                "popen_attempted": True,
                "popen_reached": True,
                "process_pid": process_pid,
                "process_reaped": True,
                "process_group_id": process_pid,
                "process_group_isolated": True,
                "process_group_reaped": True,
                "returncode": 0,
            }
        )
        _write_json(stage_root / "stage_receipt.json", result)
    plan = {
        "format": watcher.LOBO_WATCHER_FORMAT,
        "output_root": str(root.resolve()),
        "execution_order": [stage for stage, _body in watcher.LOBO_STAGES],
        "launcher": {"sha256": watcher.EXPECTED_LOBO_LAUNCHER_SHA256},
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
    }
    plan["static_plan_sha256"] = watcher.canonical_sha256(plan)
    monkeypatch.setattr(
        watcher, "EXPECTED_LOBO_STATIC_PLAN_SHA256", plan["static_plan_sha256"]
    )
    _write_json(root / "launch_plan.json", plan)
    state = {
        "format": "etsf_multibody_lobo_autonomous_state_v1",
        "status": watcher.LOBO_WATCHER_FROZEN_STATE_STATUS,
        "execution_order": plan["execution_order"],
        "stage_results": stage_results,
        "stages_started": plan["execution_order"],
        "stage_lifecycles": stage_lifecycles,
        "current_stage": None,
        "stage_pid": None,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
    }
    _write_json(root / "launch_state.json", state)
    final_base = {
        "format": watcher.LOBO_WATCHER_FORMAT,
        "status": watcher.LOBO_WATCHER_TERMINAL_STATUS,
        "static_plan_sha256": plan["static_plan_sha256"],
        "source63_audit": source_audit,
        "execution_order": plan["execution_order"],
        "stage_results": stage_results,
        "stage_lifecycles": stage_lifecycles,
        "source_binding_receipt": source_binding_audit,
        "deployment_rerank_checkpoint": deployment_checkpoint,
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
        "preregistered_outputs": {
            "piper": str(piper_output),
            "ur5": str(ur5_output),
        },
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
        "artifacts_frozen_read_only": True,
    }
    final = {**final_base, "receipt_sha256": watcher.canonical_sha256(final_base)}
    _write_json(root / "final_receipt.json", final)
    (root / "run.exit").write_bytes(b"0\n")
    _freeze(piper_output)
    _freeze(ur5_output)
    _freeze(source_root)
    _freeze(root)
    monkeypatch.setattr(watcher, "DESIGNATED_LOBO_ROOT", root)
    monkeypatch.setattr(
        watcher,
        "DESIGNATED_LOBO_OUTPUTS",
        {"piper": piper_output, "ur5-wsg": ur5_output},
    )
    return root


def test_lobo_gate_binds_both_zero_exits_and_external_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lobo_fixture(tmp_path, monkeypatch)
    audit = watcher.validate_lobo_terminal_summary(root)
    assert audit["execution_order"] == [stage for stage, _body in watcher.LOBO_STAGES]
    assert {
        value["held_out_body"] for value in audit["stage_audits"].values()
    } == {"piper", "ur5-wsg"}
    assert all(value["returncode"] == 0 for value in audit["stage_audits"].values())
    assert audit["test_hdf5_opened_by_watcher"] == 0


def test_deployment_constants_bind_exact_r12_authority() -> None:
    assert watcher.EXPECTED_LOBO_LAUNCHER_SHA256 == (
        "3af8933fa5ccd09e7b06dc1912926510e5a9fb0508b2aee3c9d323adafb71206"
    )
    assert watcher.EXPECTED_LOBO_STATIC_PLAN_SHA256 == (
        "467091737465220a1733aca1acb91b9f941cd788aa9368fcbb4b5c6ea7859986"
    )
    assert watcher.DESIGNATED_LOBO_ROOT.name == (
        "etsf_multibody_lobo_autonomous_r12_20260828"
    )
    assert watcher.EXPECTED_SOURCE_ROOT.name == (
        "etsf_smolvla_schema5_native_source_training_r12_20260828"
    )
    assert watcher.EXPECTED_SOURCE_STATIC_PLAN_SHA256 == (
        "10ed8ceb1eb2d5374225df247fe078b220414d4994f5d970af8a0c552fa4aac4"
    )
    assert watcher.EXPECTED_SOURCE_LAUNCHER_SHA256 == (
        "1713fe07a0416ea692cde171061bd739016f4832dc76b0eff7c43904b1c68d57"
    )


def test_lobo_gate_rejects_pre_r8e_terminal_state_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lobo_fixture(tmp_path, monkeypatch)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    state_path = root / "launch_state.json"
    state = json.loads(state_path.read_text())
    state["status"] = watcher.LOBO_WATCHER_TERMINAL_STATUS
    _write_json(state_path, state)
    _freeze(root)
    with pytest.raises(watcher.WatcherContractError, match="state contract"):
        watcher.validate_lobo_terminal_summary(root)


def test_lobo_gate_rejects_rogue_state_stage_or_running_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lobo_fixture(tmp_path, monkeypatch)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    state_path = root / "launch_state.json"
    state = json.loads(state_path.read_text())
    state["stage_results"]["rogue_unplanned_stage"] = {
        "status": "complete"
    }
    state["stages_started"].append("rogue_unplanned_stage")
    state["current_stage"] = "rogue_unplanned_stage"
    state["stage_pid"] = 99999
    _write_json(state_path, state)
    _freeze(root)
    with pytest.raises(watcher.WatcherContractError, match="state contract"):
        watcher.validate_lobo_terminal_summary(root)


def test_lobo_lifecycle_proof_rejects_unreaped_and_bool_returncode() -> None:
    result = {
        "stage": "train_lobo_piper",
        "status": "complete",
        "pid": 45123,
        "returncode": 0,
        "process_reaped": True,
        "process_group_id": 45123,
        "process_group_isolated": True,
        "process_group_reaped": True,
    }
    lifecycle = {
        "stage": "train_lobo_piper",
        "popen_attempted": True,
        "popen_reached": True,
        "process_pid": 45123,
        "process_reaped": True,
        "process_group_id": 45123,
        "process_group_isolated": True,
        "process_group_reaped": False,
        "returncode": 0,
    }
    with pytest.raises(watcher.WatcherContractError, match="lifecycle proof"):
        watcher._validate_lobo_stage_lifecycle_proof(
            stage="train_lobo_piper", result=result, lifecycle=lifecycle
        )
    lifecycle["process_group_reaped"] = True
    result["returncode"] = False
    lifecycle["returncode"] = False
    with pytest.raises(watcher.WatcherContractError, match="lifecycle proof"):
        watcher._validate_lobo_stage_lifecycle_proof(
            stage="train_lobo_piper", result=result, lifecycle=lifecycle
        )


def test_lobo_gate_rejects_mutated_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lobo_fixture(tmp_path, monkeypatch)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    (root / "stages" / "train_lobo_ur5" / "run.exit").write_bytes(b"1\n")
    _freeze(root)
    with pytest.raises(watcher.WatcherContractError, match="exact exit zero"):
        watcher.validate_lobo_terminal_summary(root)


def test_lobo_gate_rejects_native_source_checkpoint_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lobo_fixture(tmp_path, monkeypatch)
    final = json.loads((root / "final_receipt.json").read_text())
    checkpoint = Path(final["source63_audit"]["ensemble_checkpoint"])
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"mutated-native-source-ensemble")
    checkpoint.chmod(0o444)
    with pytest.raises(watcher.WatcherContractError, match="artifacts changed"):
        watcher.validate_lobo_terminal_summary(root)


def test_static_plan_binds_exact_h1_order_and_no_hdf_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _static_fixture(tmp_path, monkeypatch)
    plan = watcher.static_preflight(args)
    assert plan["execution_order"] == [
        "materialize_reset_only_registry",
        "freeze_one_seed_h1_authority",
        "collect_one_seed_h1_schema6",
    ]
    assert plan["root_action_horizon"] == 1
    assert plan["continuation_action_horizon"] == 1
    assert plan["seed_count"] == 1
    assert plan["lobo_summary_read_during_static_preflight"] is False
    assert "source_training_root" not in plan
    assert all(
        Path(item).suffix.lower() not in watcher.HDF_SUFFIXES
        for command in plan["commands"]
        for item in command["argv"]
    )


def test_gpu_audit_requires_exact_4090_and_zero_compute_processes() -> None:
    def idle_runner(argv):
        if any("query-gpu" in item for item in argv):
            return subprocess.CompletedProcess(argv, 0, "0, NVIDIA GeForce RTX 4090, GPU-abc\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    assert watcher.audit_idle_4090(0, runner=idle_runner)["status"] == "idle_designated_rtx4090"

    def busy_runner(argv):
        if any("query-gpu" in item for item in argv):
            return subprocess.CompletedProcess(argv, 0, "0, NVIDIA GeForce RTX 4090, GPU-abc\n", "")
        return subprocess.CompletedProcess(argv, 0, "1234, python, 2000, GPU-abc\n", "")

    assert watcher.audit_idle_4090(0, runner=busy_runner)["compute_process_count"] == 1

    def wrong_runner(argv):
        return subprocess.CompletedProcess(argv, 0, "0, NVIDIA A100, GPU-x\n", "")

    with pytest.raises(watcher.WatcherContractError, match="4090"):
        watcher.audit_idle_4090(0, runner=wrong_runner)


def test_stage_environment_scrubs_python_and_conda_inheritance() -> None:
    environment, audit = watcher.isolated_stage_environment(
        {
            "PATH": "/usr/bin",
            "PYTHONPATH": "/wrong/torch24/site-packages",
            "PYTHONHOME": "/wrong/python",
            "VIRTUAL_ENV": "/wrong/venv",
            "CONDA_PREFIX": "/wrong/conda",
        },
        gpu_index=0,
        omp_threads=8,
    )
    assert all(name not in environment for name in watcher.SCRUBBED_PYTHON_ENVIRONMENT)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert audit["pythonpath_inherited"] is False
    assert audit["scrubbed_names_present_in_parent"] == [
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ]
    assert audit["audit_sha256"] == watcher.canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )


def test_materializer_audit_binds_runtime_ids_without_trajectory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    reset = output / "reset_contract"
    reset.mkdir(parents=True)
    registry = {
        "format": "etsf_schema6_object_registry_v1",
        "objects": [
            {
                "name": "can",
                "stable_sim_actor_id": "task_attr=can;sapien_actor_name=can_actor",
                "asset_model_id": "105_sauce-can/base3",
                "role": "manipulated",
                "is_static": False,
            },
            {
                "name": "pot",
                "stable_sim_actor_id": "task_attr=pot;sapien_actor_name=pot_actor",
                "asset_model_id": "060_kitchenpot/base8",
                "role": "receptacle",
                "is_static": False,
            },
        ],
    }
    spec = {
        "format": "etsf_schema6_pose_quality_spec_v1",
        "schema_version": 6,
        "object_registry_sha256": watcher.canonical_sha256(registry),
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "frozen_before_collection": True,
        },
    }
    registry_path = reset / "object_registry.json"
    spec_path = reset / "pose_quality_spec.json"
    _write_json(registry_path, registry)
    _write_json(spec_path, spec)
    result = {
        "status": "materialized_reset_only_schema6_contract",
        "requested_seed": watcher.FIXED_REQUESTED_SEED,
        "resolved_seed": watcher.FIXED_REQUESTED_SEED,
        "environment_steps": 0,
        "policy_imported_or_forwarded": False,
        "trajectory_or_labels_read": False,
        "fresh_inputs_used": False,
        "object_registry": {
            "path": str(registry_path.resolve()),
            "file_sha256": watcher.file_sha256(registry_path),
            "logical_sha256": watcher.canonical_sha256(registry),
        },
        "pose_quality_spec": {
            "path": str(spec_path.resolve()),
            "file_sha256": watcher.file_sha256(spec_path),
            "logical_sha256": watcher.canonical_sha256(spec),
        },
    }
    log = output / "materialize.log"
    log.write_text(json.dumps(result) + "\n", encoding="utf-8")
    reset.chmod(0o555)
    audit = watcher.validate_materializer_output({"output_root": str(output)}, log)
    assert audit["environment_steps"] == 0
    assert audit["trajectory_or_labels_read"] is False
    assert audit["hdf5_opened"] == 0


def _authority_fixture(output: Path, event: Path) -> tuple[Path, dict]:
    reset = output / "reset_contract"
    reset.mkdir(parents=True, exist_ok=True)
    registry = reset / "object_registry.json"
    spec = reset / "pose_quality_spec.json"
    registry.write_text("{}\n", encoding="utf-8")
    spec.write_text("{}\n", encoding="utf-8")
    base = {
        "format": watcher.AUTHORITY_FORMAT,
        "status": watcher.AUTHORITY_STATUS,
        "scope": {
            "requested_seed": watcher.FIXED_REQUESTED_SEED,
            "expected_resolved_seed": watcher.FIXED_REQUESTED_SEED,
            "seed_count": 1,
            "candidate_indices": [0, 1, 2, 3],
            "root_action_horizon": 1,
            "continuation_action_horizon": 1,
            "max_episode_steps": 4,
        },
        "output_contract": {"directory": str(output / "schema6_collection")},
        "input_artifacts": {
            "object_registry": {"path": str(registry)},
            "pose_quality_spec": {"path": str(spec)},
            "event_spec": {"path": str(event)},
        },
        "capability_contract": {
            "fresh_inputs_allowed": False,
            "fresh_trajectory_or_label_opened": False,
            "performance_evaluation_authorized": False,
            "transfer_claim_authorized": False,
        },
    }
    authority = {**base, "authority_sha256": watcher.canonical_sha256(base)}
    path = output / "collection_authority.json"
    _write_json(path, authority)
    path.chmod(0o444)
    return path, authority


def test_authority_audit_requires_content_addressed_one_seed_h1(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    event = tmp_path / "events.json"
    event.write_text("{}\n", encoding="utf-8")
    authority_path, authority = _authority_fixture(output, event)
    result = {
        "status": watcher.AUTHORITY_STATUS,
        "path": str(authority_path.resolve()),
        "file_sha256": watcher.file_sha256(authority_path),
        "authority_sha256": authority["authority_sha256"],
    }
    log = output / "freeze.log"
    log.write_text(json.dumps(result) + "\n", encoding="utf-8")
    plan = {"output_root": str(output), "event_spec": str(event), "max_episode_steps": 4}
    audit = watcher.validate_authority_output(plan, log)
    assert audit["seed_count"] == 1
    assert audit["root_action_horizon"] == 1
    assert audit["authority_logical_sha256"] == authority["authority_sha256"]


@pytest.mark.parametrize(
    ("returncode", "status", "groups"),
    [
        (0, watcher.COLLECTION_SUCCESS_STATUS, 1),
        (20, watcher.COLLECTION_EMPTY_STATUS, 0),
    ],
)
def test_collection_audit_accepts_only_authorized_terminal_exits(
    tmp_path: Path, returncode: int, status: str, groups: int
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    event = tmp_path / "events.json"
    event.write_text("{}\n", encoding="utf-8")
    authority_path, authority = _authority_fixture(output, event)
    collection = output / "schema6_collection"
    collection.mkdir()
    group_record = None
    if returncode == 0:
        group = collection / f"group_seed_{watcher.FIXED_REQUESTED_SEED}.hdf5"
        group.write_bytes(b"development-only-group")
        group_record = {
            "path": str(group),
            "file_sha256": hashlib.sha256(group.read_bytes()).hexdigest(),
            "audit": {"schema_version": 6},
        }
    manifest = {
        "format": watcher.COLLECTION_MANIFEST_FORMAT,
        "status": status,
        "requested_seeds": [watcher.FIXED_REQUESTED_SEED],
        "completed_groups": groups,
        "group": group_record,
        "fresh_inputs_used": False,
        "task_success_claimed": False,
        "performance_evaluation_authorized": False,
        "transfer_claim_authorized": False,
    }
    manifest_path = collection / "manifest.json"
    _write_json(manifest_path, manifest)
    receipt_base = {
        "format": watcher.COLLECTION_RECEIPT_FORMAT,
        "status": status,
        "exit_code": returncode,
        "authority": {
            "path": str(authority_path.resolve()),
            "file_sha256": watcher.file_sha256(authority_path),
            "logical_sha256": authority["authority_sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": watcher.file_sha256(manifest_path),
        },
        "group": group_record,
        "failure": None,
        "fresh_inputs_used": False,
        "real_robot_execution": False,
        "task_success_claimed": False,
        "performance_evaluation_authorized": False,
        "transfer_claim_authorized": False,
    }
    receipt = {
        **receipt_base,
        "receipt_logical_sha256": watcher.canonical_sha256(receipt_base),
    }
    receipt_path = collection / "collection_receipt.json"
    _write_json(receipt_path, receipt)
    stdout = {
        "exit_code": returncode,
        "status": status,
        "receipt_file_sha256": watcher.file_sha256(receipt_path),
        "receipt_logical_sha256": receipt["receipt_logical_sha256"],
        "manifest_file_sha256": watcher.file_sha256(manifest_path),
    }
    log = output / "collect.log"
    log.write_text(json.dumps(stdout) + "\n", encoding="utf-8")
    audit = watcher.validate_collection_output({"output_root": str(output)}, log, returncode)
    assert audit["returncode"] == returncode
    assert audit["completed_groups"] == groups
    assert audit["test_hdf5_opened"] == 0


def test_subprocess_nonzero_exit_has_atomic_receipt_and_log_sha(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    state_path = output / "state.json"
    argv = [sys.executable, "-c", "print('failure'); raise SystemExit(3)"]
    stage = {
        "stage": "failure_smoke",
        "argv": argv,
        "argv_sha256": watcher.canonical_sha256(argv),
        "accepted_returncodes": [0],
    }
    lifecycle = {}
    with pytest.raises(watcher.WatcherContractError, match="exit 3"):
        watcher.run_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state={},
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=5,
            environment_audit_sha256="a" * 64,
            validator=lambda _log, _code: {},
            lifecycle=lifecycle,
        )
    receipt_path = output / "stage_receipts" / "failure_smoke.json"
    receipt = json.loads(receipt_path.read_text())
    log = output / "logs" / "failure_smoke.log"
    run_exit = output / "stage_exits" / "failure_smoke.exit"
    assert receipt["status"] == "failed_closed"
    assert receipt["returncode"] == 3
    assert receipt["log_sha256"] == watcher.file_sha256(log)
    assert run_exit.read_bytes() == b"3\n"
    assert receipt["run_exit_sha256"] == watcher.file_sha256(run_exit)
    assert receipt["process_reaped"] is True
    assert receipt["process_group_id"] == receipt["pid"]
    assert receipt["process_group_isolated"] is True
    assert receipt["process_group_reaped"] is True
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_reaped"] is True
    assert lifecycle["returncode"] == 3
    assert receipt["receipt_payload_sha256"] == watcher.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    )


def test_run_stage_timeout_reaps_parent_and_child_process_group(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    state_path = output / "state.json"
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    argv = [sys.executable, "-c", script]
    stage = {
        "stage": "timeout_tree_smoke",
        "argv": argv,
        "argv_sha256": watcher.canonical_sha256(argv),
        "accepted_returncodes": [0],
    }
    lifecycle = {}
    with pytest.raises(TimeoutError, match="timed out"):
        watcher.run_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state={},
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=0.15,
            environment_audit_sha256="b" * 64,
            validator=lambda _log, _code: {},
            lifecycle=lifecycle,
        )
    assert lifecycle["popen_attempted"] is True
    assert lifecycle["popen_reached"] is True
    assert lifecycle["process_pid"] == lifecycle["process_group_id"]
    assert lifecycle["process_group_isolated"] is True
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_reaped"] is True
    assert lifecycle["returncode"] == -signal.SIGTERM
    assert watcher._process_group_exists(lifecycle["process_group_id"]) is False


@pytest.mark.parametrize("getpgid_mode", ["raises", "mismatch"])
def test_run_stage_short_circuits_when_process_group_proof_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    getpgid_mode: str,
) -> None:
    output = tmp_path / getpgid_mode
    output.mkdir()
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    stage = {
        "stage": "process_group_proof_smoke",
        "argv": argv,
        "argv_sha256": watcher.canonical_sha256(argv),
        "accepted_returncodes": [0],
    }
    if getpgid_mode == "raises":
        def broken_getpgid(_pid: int) -> int:
            raise ProcessLookupError("injected getpgid failure")

        monkeypatch.setattr(watcher.os, "getpgid", broken_getpgid)
        expected_error = ProcessLookupError
    else:
        monkeypatch.setattr(watcher.os, "getpgid", lambda pid: pid + 1)
        expected_error = watcher.WatcherContractError
    lifecycle = {}
    started = time.monotonic()
    with pytest.raises(expected_error):
        watcher.run_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state={},
            state_path=output / "state.json",
            poll_seconds=0.01,
            timeout_seconds=0.0,
            environment_audit_sha256="c" * 64,
            validator=lambda _log, _code: {},
            lifecycle=lifecycle,
        )
    assert time.monotonic() - started < 2.0
    assert lifecycle["popen_attempted"] is True
    assert lifecycle["popen_reached"] is True
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_reaped"] is False
    assert watcher._unreaped_schema6_stage([lifecycle]) == stage["stage"]


@pytest.mark.parametrize("failing_write", ["running_receipt", "running_state"])
def test_run_stage_write_failure_after_popen_reaps_parent_and_child_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_write: str,
) -> None:
    output = tmp_path / failing_write
    output.mkdir()
    state_path = output / "state.json"
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    argv = [sys.executable, "-c", script]
    stage = {
        "stage": "atomic_write_failure_smoke",
        "argv": argv,
        "argv_sha256": watcher.canonical_sha256(argv),
        "accepted_returncodes": [0],
    }
    original_atomic_json = watcher.atomic_json
    injected = {"done": False}

    def injected_atomic_json(path: Path, value) -> None:
        is_target = (
            failing_write == "running_receipt"
            and path.name == "atomic_write_failure_smoke.json"
            and value.get("status") == "running"
        ) or (
            failing_write == "running_state"
            and path == state_path
            and value.get("status") == "running_atomic_write_failure_smoke"
        )
        if is_target and not injected["done"]:
            injected["done"] = True
            raise OSError(f"injected {failing_write} failure")
        original_atomic_json(path, value)

    monkeypatch.setattr(watcher, "atomic_json", injected_atomic_json)
    lifecycle = {}
    with pytest.raises(OSError, match=failing_write):
        watcher.run_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state={},
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=0.0,
            environment_audit_sha256="d" * 64,
            validator=lambda _log, _code: {},
            lifecycle=lifecycle,
        )
    assert injected["done"] is True
    assert lifecycle["popen_attempted"] is True
    assert lifecycle["popen_reached"] is True
    assert lifecycle["process_pid"] == lifecycle["process_group_id"]
    assert lifecycle["process_group_isolated"] is True
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_reaped"] is True
    assert watcher._process_group_exists(lifecycle["process_group_id"]) is False
    receipt = json.loads(
        (output / "stage_receipts" / "atomic_write_failure_smoke.json").read_text()
    )
    assert receipt["status"] == "failed_closed"
    assert receipt["process_reaped"] is True
    assert receipt["process_group_reaped"] is True


def test_unproven_popen_attempt_retains_owned_gpu_lock() -> None:
    lifecycle = {
        "stage": "materialize_reset_only_registry",
        "popen_attempted": True,
        "popen_reached": False,
        "process_pid": None,
        "process_reaped": False,
        "process_group_id": None,
        "process_group_isolated": False,
        "process_group_reaped": False,
        "returncode": None,
    }
    assert watcher._unreaped_schema6_stage([lifecycle]) == lifecycle["stage"]
    assert (
        watcher._owned_gpu_lock_release_allowed(
            gpu_lock_acquired=True, stage_lifecycles=[lifecycle]
        )
        is False
    )


def test_terminal_publication_failure_removes_hidden_files_and_restores_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    artifact = root / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    def fail_freeze(*_args, **_kwargs):
        raise OSError("injected freeze failure")

    monkeypatch.setattr(watcher, "freeze_tree", fail_freeze)
    with pytest.raises(OSError, match="injected freeze failure"):
        watcher.publish_frozen_terminal_receipt(
            root,
            terminal_name="failure_receipt.json",
            receipt={"artifacts_frozen_read_only": True},
            exit_code=1,
        )
    assert not (root / "failure_receipt.json").exists()
    assert not (root / "run.exit").exists()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_lock_rejects_duplicate_or_stale_owner(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    watcher.acquire_lock(lock, {"pid": 1, "token": "a"})
    with pytest.raises(watcher.WatcherContractError, match="lock exists"):
        watcher.acquire_lock(lock, {"pid": 2, "token": "b"})


def test_detach_uses_new_session_and_does_not_read_source_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _static_fixture(tmp_path, monkeypatch)
    args.command = "detach"
    captured = {}

    class Process:
        pid = 424242

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(watcher.subprocess, "Popen", fake_popen)
    receipt = watcher.detach(args)
    assert captured["start_new_session"] is True
    assert captured["close_fds"] is True
    assert receipt["survives_client_disconnect"] is True
    assert receipt["lobo_summary_read_by_detach"] is False
    assert receipt["lobo_test_hdf5_opened_by_detach"] == 0
    assert not args.output.exists()


def test_execute_orders_lobo_idle_materialize_authority_idle_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    lock = tmp_path / "pipeline.lock"
    commands = [
        {
            "stage": name,
            "argv": [sys.executable, "-c", "pass"],
            "argv_sha256": watcher.canonical_sha256([sys.executable, "-c", "pass"]),
            "accepted_returncodes": accepted,
        }
        for name, accepted in (
            ("materialize_reset_only_registry", [0]),
            ("freeze_one_seed_h1_authority", [0]),
            ("collect_one_seed_h1_schema6", [0, 20]),
        )
    ]
    plan = {
        "format": watcher.FORMAT,
        "output_root": str(output),
        "gpu_lock": str(lock),
        "lobo_root": str(tmp_path / "lobo"),
        "commands": commands,
        "execution_order": [row["stage"] for row in commands],
        "static_plan_sha256": "a" * 64,
    }
    lobo_audit = _synthetic_lobo_audit(Path(plan["lobo_root"]))
    order = []
    monkeypatch.setattr(watcher, "static_preflight", lambda _args: plan)
    monkeypatch.setattr(watcher, "verify_static_bindings", lambda _plan: None)
    monkeypatch.setattr(
        watcher, "DESIGNATED_LOBO_ROOT", Path(plan["lobo_root"])
    )
    monkeypatch.setattr(
        watcher, "validate_lobo_terminal_summary", lambda _root: lobo_audit
    )

    def wait_gpu(*_args, **kwargs):
        order.append(f"idle:{kwargs['phase']}")
        value = {"status": "idle_designated_rtx4090", "compute_process_count": 0}
        value["audit_sha256"] = watcher.canonical_sha256(value)
        return value

    def wait_lobo(*_args, **_kwargs):
        order.append("lobo")
        return lobo_audit

    def stage_runner(stage, **kwargs):
        order.append(stage["stage"])
        process_pid = 6000 + len(order)
        lifecycle = kwargs["lifecycle"]
        lifecycle.update(
            {
                "stage": stage["stage"],
                "popen_attempted": True,
                "popen_reached": True,
                "process_pid": process_pid,
                "process_reaped": True,
                "process_group_id": process_pid,
                "process_group_isolated": True,
                "process_group_reaped": True,
                "returncode": 0,
            }
        )
        result = {
            "format": watcher.FORMAT,
            "stage": stage["stage"],
            "status": "complete_verified",
            "accepted_returncodes": stage["accepted_returncodes"],
            "pid": process_pid,
            "returncode": 0,
            "process_reaped": True,
            "process_group_id": process_pid,
            "process_group_isolated": True,
            "process_group_reaped": True,
            "artifact_audit": {"status": "ok", "test_hdf5_opened": 0},
        }
        result["receipt_payload_sha256"] = watcher.canonical_sha256(result)
        return result

    monkeypatch.setattr(watcher, "wait_for_lobo_completion", wait_lobo)
    monkeypatch.setattr(watcher, "wait_for_idle_4090", wait_gpu)
    monkeypatch.setattr(watcher, "run_stage", stage_runner)
    args = argparse.Namespace(
        gpu_index=0,
        poll_seconds=1.0,
        lobo_timeout_seconds=0.0,
        gpu_timeout_seconds=0.0,
        materializer_timeout_seconds=10.0,
        freezer_timeout_seconds=10.0,
        collection_timeout_seconds=0.0,
        omp_threads=2,
    )
    result = watcher.execute(args)
    assert order == [
        "lobo",
        "idle:reset_only_materialization",
        "materialize_reset_only_registry",
        "freeze_one_seed_h1_authority",
        "idle:one_seed_h1_collection",
        "collect_one_seed_h1_schema6",
    ]
    assert result["status"] == watcher.TERMINAL_STATUS
    assert result["lobo_test_hdf5_opened"] == 0
    assert result["execution_order"] == list(watcher.SCHEMA6_EXECUTION_ORDER)
    assert len(result["stage_lifecycles"]) == 3
    assert (output / "run.exit").read_bytes() == b"0\n"
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert stat.S_IMODE((output / "final_receipt.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((output / "run.exit").stat().st_mode) == 0o444
    state = json.loads((output / "launch_state.json").read_text())
    assert state["status"] == watcher.TERMINAL_PENDING_STATUS
    assert watcher.validate_schema6_success_terminal_receipt(result) == result
    assert not lock.exists()


def test_success_terminal_rejects_bool_numeric_lifecycle_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def capture_publish(_root, *, terminal_name, receipt, exit_code):
        assert terminal_name == "final_receipt.json"
        assert exit_code == 0
        captured.update(receipt)

    monkeypatch.setattr(watcher, "publish_frozen_terminal_receipt", capture_publish)
    output = tmp_path / "output"
    lock = tmp_path / "pipeline.lock"
    commands = [
        {
            "stage": name,
            "argv": [sys.executable, "-c", "pass"],
            "argv_sha256": watcher.canonical_sha256(
                [sys.executable, "-c", "pass"]
            ),
            "accepted_returncodes": list(
                watcher.SCHEMA6_STAGE_ACCEPTED_RETURNCODES[name]
            ),
        }
        for name in watcher.SCHEMA6_EXECUTION_ORDER
    ]
    plan = {
        "format": watcher.FORMAT,
        "output_root": str(output),
        "gpu_lock": str(lock),
        "lobo_root": str(tmp_path / "lobo"),
        "commands": commands,
        "execution_order": list(watcher.SCHEMA6_EXECUTION_ORDER),
        "static_plan_sha256": "a" * 64,
    }
    lobo_audit = _synthetic_lobo_audit(Path(plan["lobo_root"]))
    monkeypatch.setattr(watcher, "static_preflight", lambda _args: plan)
    monkeypatch.setattr(watcher, "verify_static_bindings", lambda _plan: None)
    monkeypatch.setattr(
        watcher, "DESIGNATED_LOBO_ROOT", Path(plan["lobo_root"])
    )
    monkeypatch.setattr(
        watcher, "validate_lobo_terminal_summary", lambda _root: lobo_audit
    )
    monkeypatch.setattr(
        watcher,
        "wait_for_idle_4090",
        lambda *_args, **_kwargs: {
            "status": "idle_designated_rtx4090",
            "compute_process_count": 0,
            "audit_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        watcher,
        "wait_for_lobo_completion",
        lambda *_args, **_kwargs: lobo_audit,
    )

    def stage_runner(stage, **kwargs):
        pid = 7000 + len(kwargs["state"]["stages_started"])
        kwargs["lifecycle"].update(
            {
                "stage": stage["stage"],
                "popen_attempted": True,
                "popen_reached": True,
                "process_pid": pid,
                "process_reaped": True,
                "process_group_id": pid,
                "process_group_isolated": True,
                "process_group_reaped": True,
                "returncode": 0,
            }
        )
        result = {
            "format": watcher.FORMAT,
            "stage": stage["stage"],
            "status": "complete_verified",
            "accepted_returncodes": stage["accepted_returncodes"],
            "pid": pid,
            "returncode": 0,
            "process_reaped": True,
            "process_group_id": pid,
            "process_group_isolated": True,
            "process_group_reaped": True,
            "artifact_audit": {"status": "ok"},
        }
        result["receipt_payload_sha256"] = watcher.canonical_sha256(result)
        return result

    monkeypatch.setattr(watcher, "run_stage", stage_runner)
    args = argparse.Namespace(
        gpu_index=0,
        poll_seconds=1.0,
        lobo_timeout_seconds=0.0,
        gpu_timeout_seconds=0.0,
        materializer_timeout_seconds=10.0,
        freezer_timeout_seconds=10.0,
        collection_timeout_seconds=0.0,
        omp_threads=2,
    )
    watcher.execute(args)
    tampered = json.loads(json.dumps(captured))
    tampered["stage_lifecycles"][0]["returncode"] = False
    unsigned = dict(tampered)
    unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = watcher.canonical_sha256(unsigned)
    with pytest.raises(watcher.WatcherContractError, match="lifecycle proof"):
        watcher.validate_schema6_success_terminal_receipt(tampered)

    tampered = json.loads(json.dumps(captured))
    tampered["stage_returncodes"][watcher.SCHEMA6_EXECUTION_ORDER[0]] = False
    unsigned = dict(tampered)
    unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = watcher.canonical_sha256(unsigned)
    with pytest.raises(watcher.WatcherContractError, match="stage summary"):
        watcher.validate_schema6_success_terminal_receipt(tampered)

    tampered = json.loads(json.dumps(captured))
    tampered["lobo_gate"]["lobo_launcher_sha256"] = "f" * 64
    gate_unsigned = dict(tampered["lobo_gate"])
    gate_unsigned.pop("summary_sha256")
    gate_sha = watcher.canonical_sha256(gate_unsigned)
    tampered["lobo_gate"]["summary_sha256"] = gate_sha
    tampered["lobo_audit_sha256"] = gate_sha
    unsigned = dict(tampered)
    unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = watcher.canonical_sha256(unsigned)
    with pytest.raises(watcher.WatcherContractError, match="semantics"):
        watcher.validate_schema6_success_terminal_receipt(tampered)

    for mutation in (
        "drop_source_binding_receipt",
        "invalidate_final_receipt_sha256",
        "drop_stage_audits",
    ):
        tampered = json.loads(json.dumps(captured))
        if mutation == "drop_source_binding_receipt":
            tampered["lobo_gate"].pop("source_binding_receipt")
        elif mutation == "invalidate_final_receipt_sha256":
            tampered["lobo_gate"]["final_receipt_sha256"] = "not-a-sha"
        else:
            tampered["lobo_gate"].pop("stage_audits")
        gate_unsigned = dict(tampered["lobo_gate"])
        gate_unsigned.pop("summary_sha256")
        gate_sha = watcher.canonical_sha256(gate_unsigned)
        tampered["lobo_gate"]["summary_sha256"] = gate_sha
        tampered["lobo_audit_sha256"] = gate_sha
        unsigned = dict(tampered)
        unsigned.pop("receipt_sha256")
        tampered["receipt_sha256"] = watcher.canonical_sha256(unsigned)
        with pytest.raises(watcher.WatcherContractError, match="semantics"):
            watcher.validate_schema6_success_terminal_receipt(tampered)

    tampered = json.loads(json.dumps(captured))
    for native in (
        tampered["native_source_binding_audit"],
        tampered["lobo_gate"]["native_source_binding_audit"],
    ):
        native["source_training_root"] = "/home/user/rebound_source"
    native_sha = watcher.canonical_sha256(
        tampered["native_source_binding_audit"]
    )
    tampered["native_source_binding_audit_sha256"] = native_sha
    tampered["lobo_gate"]["native_source_binding_audit_sha256"] = native_sha
    gate_unsigned = dict(tampered["lobo_gate"])
    gate_unsigned.pop("summary_sha256")
    gate_sha = watcher.canonical_sha256(gate_unsigned)
    tampered["lobo_gate"]["summary_sha256"] = gate_sha
    tampered["lobo_audit_sha256"] = gate_sha
    unsigned = dict(tampered)
    unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = watcher.canonical_sha256(unsigned)
    with pytest.raises(watcher.WatcherContractError, match="semantics"):
        watcher.validate_schema6_success_terminal_receipt(tampered)

    tampered = json.loads(json.dumps(captured))
    rebound = dict(tampered["deployment_rerank_checkpoint"])
    rebound["path"] = "/home/user/rebound_source/ensemble.pt"
    tampered["deployment_rerank_checkpoint"] = rebound
    tampered["native_source_binding_audit"][
        "deployment_rerank_checkpoint"
    ] = rebound
    tampered["lobo_gate"]["deployment_rerank_checkpoint"] = rebound
    tampered["lobo_gate"]["native_source_binding_audit"][
        "deployment_rerank_checkpoint"
    ] = rebound
    native_sha = watcher.canonical_sha256(
        tampered["native_source_binding_audit"]
    )
    tampered["native_source_binding_audit_sha256"] = native_sha
    tampered["lobo_gate"]["native_source_binding_audit_sha256"] = native_sha
    gate_unsigned = dict(tampered["lobo_gate"])
    gate_unsigned.pop("summary_sha256")
    gate_sha = watcher.canonical_sha256(gate_unsigned)
    tampered["lobo_gate"]["summary_sha256"] = gate_sha
    tampered["lobo_audit_sha256"] = gate_sha
    unsigned = dict(tampered)
    unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = watcher.canonical_sha256(unsigned)
    with pytest.raises(watcher.WatcherContractError, match="semantics"):
        watcher.validate_schema6_success_terminal_receipt(tampered)
