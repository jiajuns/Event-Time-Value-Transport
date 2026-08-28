from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "launch_openvla_etsf_v9_group_relative_success_r6.py"
)
SPEC = importlib.util.spec_from_file_location("r6_group_relative_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(launcher)

R5_TEST_PATH = ROOT / "tests" / "test_launch_openvla_etsf_r5_repair_evaluations.py"
R5_SPEC = importlib.util.spec_from_file_location("r5_launcher_test_helpers", R5_TEST_PATH)
assert R5_SPEC is not None and R5_SPEC.loader is not None
r5_helpers = importlib.util.module_from_spec(R5_SPEC)
R5_SPEC.loader.exec_module(r5_helpers)


def _bundle(tmp_path: Path) -> dict[str, Path]:
    paths = r5_helpers._make_input_bundle(tmp_path)
    paths["code_root"] = ROOT
    paths["python_bin"] = Path(sys.executable)
    paths["output_root"] = tmp_path / "r6_v9_output"
    return paths


def _plan(paths: dict[str, Path]) -> dict:
    return launcher.build_plan(
        code_root=paths["code_root"],
        materialization_manifest=paths["manifest"],
        r4_summary=paths["r4_summary"],
        output_root=paths["output_root"],
        python_bin=paths["python_bin"],
        gpu_index=0,
    )


def _signed(value: dict, key: str) -> dict:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = launcher.canonical_sha256(result)
    return result


def _write_result(path: Path, plan: dict) -> dict:
    contracts = []
    rows = []
    trace = []
    for owner, fold in enumerate(plan["materialization"]["folds"]):
        contract = _signed(
            {
                "format": launcher.FOLD_CONTRACT_FORMAT,
                "owner_fold_id": owner,
                "materialization_sha256": plan["materialization"][
                    "materialization_sha256"
                ],
                "train_artifact_sha256": fold["train_artifact_sha256"],
                "train_payload_sha256": fold["train_payload_sha256"],
                "outer_training_groups": fold["training_groups"],
                "outer_training_groups_sha256": fold["training_groups_sha256"],
                "outer_holdout_groups": fold["oof_holdout_groups"],
                "outer_holdout_groups_sha256": fold[
                    "oof_holdout_groups_sha256"
                ],
                "all_hyperparameters_selected_before_outer_holdout_payload_loaded": True,
                "outer_holdout_labels_used_for_model_or_hyperparameter_fit": False,
                "fresh_inputs_or_labels_used": False,
            },
            "fold_contract_sha256",
        )
        contracts.append(contract)
        trace.append({"role": "train", "owner_fold_id": owner})
        for group in fold["oof_holdout_groups"]:
            for candidate, name in enumerate(launcher.CANDIDATE_NAMES):
                rows.append(
                    {
                        "owner_fold_id": owner,
                        "logical_group": group,
                        "candidate_index": candidate,
                        "candidate_name": name,
                        "success_label": int(candidate == owner % 4),
                        "success_probability": 0.2 + 0.1 * candidate,
                        "candidate_ranking_score": float(candidate - 1),
                        "fold_contract_sha256": contract[
                            "fold_contract_sha256"
                        ],
                    }
                )
    trace.extend(
        {"role": "holdout", "owner_fold_id": owner} for owner in range(5)
    )
    result = _signed(
        {
            "format": launcher.RESULT_FORMAT,
            "status": "fail_closed_adaptive_development_only",
            "implementation_files": plan["result_implementation_files"],
            "materialization_manifest": plan["materialization"]["path"],
            "materialization_file_sha256": plan["materialization"]["file_sha256"],
            "materialization_sha256": plan["materialization"][
                "materialization_sha256"
            ],
            "fold_contracts": contracts,
            "outer_holdout_evaluation": {
                "pooled_oof": {"strict_development_adequacy": False}
            },
            "oof_rows": rows,
            "oof_rows_sha256": launcher.canonical_sha256(rows),
            "oof_row_count": len(rows),
            "read_trace": trace,
            "read_trace_sha256": launcher.canonical_sha256(trace),
            "all_outer_contracts_selected_before_any_outer_holdout_deserialized": True,
            "probability_output_used_for_action_selection": False,
            "task_success_improvement_claim_authorized": False,
            "selector_deployment_authorized": False,
            "fresh_confirmation_authorized": False,
            "fresh_inputs_accepted": False,
            "fresh_labels_read": False,
        },
        "result_sha256",
    )
    path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def test_plan_binds_current_evaluator_adapter_d250_and_r4_lineage(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path)
    plan = _plan(paths)
    assert plan["materialization_scope"] == "R3_materialized_OOF_D250_only"
    assert len(plan["materialization"]["artifacts"]) == 10
    assert len(plan["r4_adamw_lineage"]["checkpoints"]) == 5
    assert plan["r4_checkpoints_are_lineage_only"] is True
    assert plan["r4_checkpoints_are_evaluator_cli_inputs"] is False
    assert plan["result_implementation_files"][
        "evaluate_openvla_etsf_v9_group_relative_success_oof.py"
    ] == "bcd489569a5ddcd9ca69fd8878c2bdd932bf16757fa0b052dc6c4a83e2d2b116"
    assert plan["result_implementation_files"][
        "openvla_etsf_v9_group_relative_success_adapter.py"
        ] == "35f38653b8c94214eaae9210bede87ca6613faa213d78e319c36ed0d2cb600b3"
    argv = plan["commands"][0]["argv"]
    assert argv[-1] == "cuda"
    assert "--materialization-manifest" in argv
    assert not any(
        checkpoint["path"] in argv
        for checkpoint in plan["r4_adamw_lineage"]["checkpoints"]
    )
    assert plan["fresh_paths_accepted"] is False
    assert plan["fresh_inputs_accepted"] is False
    assert plan["fresh_labels_read"] is False


def test_preregister_refuses_all_fresh_or_confirmation_paths(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        launcher.build_plan(
            code_root=paths["code_root"],
            materialization_manifest=paths["manifest"],
            r4_summary=paths["r4_summary"],
            output_root=tmp_path / "Fresh50" / "r6_output",
            python_bin=paths["python_bin"],
            gpu_index=0,
        )
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        launcher.preregister(
            code_root=paths["code_root"],
            materialization_manifest=paths["manifest"],
            r4_summary=paths["r4_summary"],
            output_root=paths["output_root"],
            python_bin=paths["python_bin"],
            gpu_index=0,
            plan_output=tmp_path / "confirmation" / "plan.json",
        )


def test_materialization_or_r4_fresh_attestation_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["fresh_confirmation_data_or_labels_read"] = True
    manifest = _signed(manifest, "materialization_sha256")
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="materialization contract is invalid"):
        _plan(paths)

    paths = _bundle(tmp_path / "second")
    summary = json.loads(paths["r4_summary"].read_text(encoding="utf-8"))
    summary["fresh50_labels_read"] = True
    summary = _signed(summary, "summary_sha256")
    paths["r4_summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no-Fresh AdamW run"):
        _plan(paths)


def test_output_authentication_requires_250_groups_four_candidates_and_read_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_deep_validate_fold_contract", lambda value: None)
    paths = _bundle(tmp_path)
    plan = _plan(paths)
    output = tmp_path / "result.json"
    value = _write_result(output, plan)
    audit = launcher._validate_result(output, plan=plan)
    assert audit["rows"] == 1000
    assert audit["logical_groups"] == 250

    value["read_trace"][0]["role"] = "holdout"
    value["read_trace_sha256"] = launcher.canonical_sha256(value["read_trace"])
    value = _signed(value, "result_sha256")
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="read order"):
        launcher._validate_result(output, plan=plan)


def test_output_rejects_duplicate_candidate_even_with_resigned_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_deep_validate_fold_contract", lambda value: None)
    paths = _bundle(tmp_path)
    plan = _plan(paths)
    output = tmp_path / "result.json"
    value = _write_result(output, plan)
    value["oof_rows"][1]["candidate_index"] = 0
    value["oof_rows"][1]["candidate_name"] = launcher.CANDIDATE_NAMES[0]
    value["oof_rows_sha256"] = launcher.canonical_sha256(value["oof_rows"])
    value = _signed(value, "result_sha256")
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicates"):
        launcher._validate_result(output, plan=plan)


def test_detach_writes_signed_recoverable_remote_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _bundle(tmp_path)
    plan_path = tmp_path / "r6_plan.json"
    plan = launcher.preregister(
        code_root=paths["code_root"],
        materialization_manifest=paths["manifest"],
        r4_summary=paths["r4_summary"],
        output_root=paths["output_root"],
        python_bin=paths["python_bin"],
        gpu_index=0,
        plan_output=plan_path,
    )

    captured = {}

    class Process:
        pid = 43210

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    receipt_path = tmp_path / "r6_receipt.json"
    receipt = launcher.detach(
        plan_path,
        poll_seconds=5.0,
        nohup_log=tmp_path / "r6_nohup.log",
        receipt_path=receipt_path,
    )
    assert receipt["pid"] == 43210
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert captured["kwargs"]["start_new_session"] is True
    assert receipt["remote_recovery_state"].endswith("launch_state.json")
    unsigned = dict(receipt)
    signature = unsigned.pop("receipt_sha256")
    assert signature == launcher.canonical_sha256(unsigned)
