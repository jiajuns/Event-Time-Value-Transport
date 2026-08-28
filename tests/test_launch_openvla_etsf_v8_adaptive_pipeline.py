from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "launch_openvla_etsf_v8_adaptive_pipeline.py"
SPEC = importlib.util.spec_from_file_location("v8_adaptive_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _signed(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = launcher.canonical_sha256(result)
    return result


def _materialization(path: Path) -> dict:
    value = _signed(
        {
            "format": "etsf_v8_oof_materialization_manifest_v1",
            "status": "complete_development_only",
            "fresh_confirmation_data_or_labels_read": False,
        },
        "materialization_sha256",
    )
    _write_json(path, value)
    return value


def test_full_output_path_fresh_guard() -> None:
    assert launcher._reject_fresh_path(
        Path("/srv/etsf/v8_adaptive"), role="output"
    ) == Path("/srv/etsf/v8_adaptive")
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        launcher._reject_fresh_path(
            Path("/srv/Fresh50/innocent/v8_adaptive"), role="output"
        )
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        launcher._reject_fresh_path(
            Path("/srv/confirmation/archive/v8_adaptive"), role="output"
        )


def test_gpu_query_is_bound_to_explicit_physical_index(monkeypatch) -> None:
    seen: list[str] = []

    def fake_run(argv, **kwargs):
        del kwargs
        seen.extend(argv)
        return SimpleNamespace(stdout="41\n41\n57\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert launcher._gpu_compute_pids(3) == [41, 57]
    assert seen[seen.index("--id") + 1] == "3"


def test_stage_uses_controlled_env_and_rechecks_every_implementation(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("pass\n", encoding="utf-8")
    second.write_text("pass\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    state_path = tmp_path / "state.json"
    state = {
        "implementation_files": {
            "first": {"path": str(first), "sha256": launcher.sha256_path(first)},
            "second": {
                "path": str(second),
                "sha256": launcher.sha256_path(second),
            },
        }
    }
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    child_env = {
        "CUDA_VISIBLE_DEVICES": "4",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/controlled/scripts",
    }
    launcher._run_stage(
        stage="ok",
        argv=[sys.executable, str(first)],
        state=state,
        state_path=state_path,
        logs_dir=logs,
        code_root=tmp_path,
        child_env=child_env,
    )
    assert captured["cwd"] == tmp_path
    assert captured["env"] == child_env
    second.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="implementation changed.*second"):
        launcher._run_stage(
            stage="tampered",
            argv=[sys.executable, str(first)],
            state=state,
            state_path=state_path,
            logs_dir=logs,
            code_root=tmp_path,
            child_env=child_env,
        )


@pytest.mark.parametrize(
    ("state_format", "execution_kind"),
    [
        ("etsf_openvla_v7_prospective_server_launch_v1", "original"),
        ("etsf_openvla_v7_resolved_seed_recovery_v1", "recovery"),
    ],
)
def test_v7_completion_binds_state_result_preregistration_and_collection(
    tmp_path: Path, state_format: str, execution_kind: str
) -> None:
    data = tmp_path / "development250"
    manifest_path = data / "manifest.json"
    manifest = {
        "status": "complete",
        "completed": 250,
        "schema_version": 5,
        "candidate_count": 4,
        "seed_registry": "explicit_v7_prospective_development",
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
    }
    _write_json(manifest_path, manifest)
    preregistration_path = tmp_path / "v7_preregistration.json"
    preregistration = _signed(
        {
            "format": "etsf_openvla_v7_prospective_development_confirmation_v1",
            "status": "preregistered_before_labels",
        },
        "preregistration_sha256",
    )
    _write_json(preregistration_path, preregistration)
    result_path = tmp_path / "v7_result.json"
    result = _signed(
        {
            "format": "etsf_openvla_v7_prospective_development_result_v1",
            "status": "complete_development_only",
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "collection_manifest": str(manifest_path.resolve()),
            "collection_manifest_sha256": launcher.sha256_path(manifest_path),
            "metrics": {"development_gate_pass": True},
            "authorization": {
                "fresh50_confirmation_authorized": True,
                "automatic_fresh_launch": False,
            },
            "fresh_confirmation_labels_read": False,
        },
        "result_sha256",
    )
    _write_json(result_path, result)
    state_path = tmp_path / "v7_state.json"
    state = {
        "format": state_format,
        "terminal_status": launcher.V7_TERMINAL_STATUS,
        "status": launcher.V7_TERMINAL_STATUS,
        "current_stage": None,
        "last_completed_stage": "evaluate",
        "automatic_fresh_launch": False,
        "fresh_confirmation_labels_read": False,
        "fresh50_confirmation_authorized": True,
        "stage_results": {
            "preregister": {
                "status": "complete",
                "artifact": str(preregistration_path.resolve()),
                "sha256": launcher.sha256_path(preregistration_path),
                "preregistration_sha256": preregistration["preregistration_sha256"],
            },
            "collect": {
                "status": "complete",
                "artifact": str(manifest_path.resolve()),
                "sha256": launcher.sha256_path(manifest_path),
                "groups": 250,
            },
            "evaluate": {
                "status": "complete",
                "artifact": str(result_path.resolve()),
                "sha256": launcher.sha256_path(result_path),
                "development_gate_pass": True,
                "fresh50_confirmation_authorized": True,
            },
        },
    }
    _write_json(state_path, state)
    audit = launcher._validate_v7_completion(
        state_path=state_path, result_path=result_path, data_root=data
    )
    assert audit["v7_execution_kind"] == execution_kind
    assert audit["fresh50_read_or_launched"] is False

    manifest["completed"] = 249
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="collection SHA mismatch"):
        launcher._validate_v7_completion(
            state_path=state_path, result_path=result_path, data_root=data
        )


def test_factual_output_requires_signed_no_fresh_materialization_binding(
    tmp_path: Path,
) -> None:
    materialization_path = tmp_path / "materialization_manifest.json"
    materialization = _materialization(materialization_path)
    result_path = tmp_path / "factual.json"
    result = _signed(
        {
            "format": "etsf_v8_factual_event_oof_diagnostics_v1",
            "status": "complete_adaptive_development_only",
            "evidence_scope": "D250_adaptive_development_only_not_prospective",
            "authorization": {
                "fresh50_confirmation_authorized": False,
                "selector_authorized": False,
                "deployment_authorized": False,
                "policy_success_claim_authorized": False,
            },
            "fresh_confirmation_data_or_labels_read": False,
            "source_materialization": {
                "path": str(materialization_path.resolve()),
                "file_sha256": launcher.sha256_path(materialization_path),
                "materialization_sha256": materialization["materialization_sha256"],
            },
        },
        "result_sha256",
    )
    _write_json(result_path, result)
    audit = launcher._validate_factual_output(
        result_path=result_path, materialization_manifest=materialization_path
    )
    assert audit["fresh50_read_or_authorized"] is False

    result["authorization"]["selector_authorized"] = True
    result = _signed(
        {key: value for key, value in result.items() if key != "result_sha256"},
        "result_sha256",
    )
    _write_json(result_path, result)
    with pytest.raises(RuntimeError, match="adaptive/no-Fresh"):
        launcher._validate_factual_output(
            result_path=result_path, materialization_manifest=materialization_path
        )


def test_bridge_output_requires_signed_status_and_no_fresh_contract(
    tmp_path: Path,
) -> None:
    materialization_path = tmp_path / "materialization_manifest.json"
    materialization = _materialization(materialization_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    arrays_path = output / "structured_heads_arrays.npz"
    arrays_path.write_bytes(b"synthetic arrays")
    adaptive = _signed({"format": "adaptive"}, "contract_sha256")
    bundle = _signed({"format": "bundle"}, "bridge_bundle_sha256")
    contracts = _signed(
        {
            "format": "etsf_v8_authenticated_oof_evaluation_output_v1",
            "adaptive_contract": adaptive,
            "bridge_bundle": bundle,
            "input_contract": {
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
            },
            "bridge_provenance": {
                "materialization_sha256": materialization["materialization_sha256"],
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
                "prospective_claim_allowed": False,
            },
            "arrays": str(arrays_path.resolve()),
            "arrays_sha256": launcher.sha256_path(arrays_path),
        },
        "contracts_sha256",
    )
    contracts_path = output / "structured_heads_contracts.json"
    _write_json(contracts_path, contracts)
    result = _signed(
        {
            "format": "etsf_v8_structured_heads_array_evaluation_v1",
            "status": "fail_closed_one_or_more_domains",
            "development_only": True,
            "prospective_claim_allowed": False,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "fresh50_confirmation_authorized": False,
            "action_selector_authorized": False,
            "adaptive_development_contract_sha256": adaptive["contract_sha256"],
            "all_domain_pass": False,
        },
        "result_sha256",
    )
    result_path = output / "structured_heads_evaluation.json"
    _write_json(result_path, result)
    audit = launcher._validate_bridge_output(
        output_dir=output, materialization_manifest=materialization_path
    )
    assert audit["status"] == "fail_closed_one_or_more_domains"
    assert audit["fresh50_read_or_authorized"] is False

    result["fresh50_labels_read"] = True
    result = _signed(
        {key: value for key, value in result.items() if key != "result_sha256"},
        "result_sha256",
    )
    _write_json(result_path, result)
    with pytest.raises(RuntimeError, match="adaptive/no-Fresh"):
        launcher._validate_bridge_output(
            output_dir=output, materialization_manifest=materialization_path
        )


def test_terminal_publication_retracts_complete_summary_if_state_fails(
    tmp_path: Path, monkeypatch
) -> None:
    summary_path = tmp_path / "pipeline_summary.json"
    state_path = tmp_path / "pipeline_state.json"
    original_atomic_json = launcher.atomic_json

    def fail_terminal_state(path: Path, value: dict) -> None:
        if path == state_path and value.get("status") == launcher.TERMINAL_STATUS:
            raise OSError("synthetic terminal state failure")
        original_atomic_json(path, value)

    monkeypatch.setattr(launcher, "atomic_json", fail_terminal_state)
    state = {"status": "running_adaptive_development"}
    summary = {"status": launcher.TERMINAL_STATUS, "summary_sha256": "x"}
    with pytest.raises(OSError, match="terminal state failure"):
        launcher._publish_terminal_state(
            summary_path=summary_path,
            state_path=state_path,
            summary=summary,
            state=state,
        )
    assert not summary_path.exists()
    assert state["status"] == "running_adaptive_development"
