from __future__ import annotations

import json
import stat
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for path in (SCRIPTS, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import materialize_smolvla_piper_schema6_development300_identity_authority as identity  # noqa: E402
import execute_smolvla_piper_schema6_development300_group as worker  # noqa: E402
import run_smolvla_piper_schema6_development300_collection as runner  # noqa: E402
from test_materialize_smolvla_piper_schema6_development300_identity_authority import (  # noqa: E402
    attestation,
    complete_receipt,
    make_reset_authority,
    write_json,
)


def make_collection_inputs(tmp_path: Path) -> dict[str, Any]:
    preregistration, runtime, reset_authority, paths = make_reset_authority(tmp_path)
    reset_path = write_json(tmp_path / "reset-authority.json", reset_authority)
    receipt = complete_receipt(
        preregistration,
        runtime,
        reset_authority,
        authority_file_sha256=identity.file_sha256(reset_path),
    )
    selected = attestation(
        identity.SELECTED_TARGET_ROLE, receipt["selected_identity_set_sha256"]
    )
    receipt_path = write_json(tmp_path / "identity-receipt.json", receipt)
    selected_path = write_json(tmp_path / "selected-attestation.json", selected)
    output_root = tmp_path / "schema6-development300-output"
    freeze_root = tmp_path / "development300-identity-freeze"
    materialized = identity.materialize_collection(
        preregistration_path=paths["preregistration"],
        expected_preregistration_file_sha256=identity.file_sha256(
            paths["preregistration"]
        ),
        reset_authority_path=reset_path,
        expected_reset_authority_file_sha256=identity.file_sha256(reset_path),
        identity_receipt_path=receipt_path,
        expected_identity_receipt_file_sha256=identity.file_sha256(receipt_path),
        selected_attestation_path=selected_path,
        expected_selected_attestation_file_sha256=identity.file_sha256(selected_path),
        future_collection_root=output_root,
        output_directory=freeze_root,
    )
    event_spec = write_json(
        tmp_path / "synthetic-event-spec.json",
        {"format": "synthetic-schema6-event-spec", "labels_opened": False},
    )
    identity_path = Path(materialized["collection_identity_authority_path"])
    collection_path = Path(materialized["collection_preregistration_path"])
    return {
        "runtime": runtime,
        "runtime_path": paths["runtime"],
        "identity_path": identity_path,
        "identity": json.loads(identity_path.read_text(encoding="utf-8")),
        "collection_path": collection_path,
        "collection": json.loads(collection_path.read_text(encoding="utf-8")),
        "event_spec": event_spec,
        "output_root": output_root,
        "gpu_lock": tmp_path / "locks" / "gpu0.lock",
    }


def build_authority(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = make_collection_inputs(tmp_path)
    inputs["gpu_lock"].parent.mkdir()
    collector = SCRIPTS / "collect_smolvla_piper_schema6_dense_event_branches.py"
    adapter = SCRIPTS / "smolvla_piper_schema6_runtime_adapter_v2.py"
    worker = SCRIPTS / "execute_smolvla_piper_schema6_development300_group.py"
    authority = runner.build_runner_authority(
        identity_authority_path=inputs["identity_path"],
        expected_identity_authority_file_sha256=identity.file_sha256(
            inputs["identity_path"]
        ),
        expected_identity_authority_sha256=inputs["identity"][
            "identity_authority_sha256"
        ],
        collection_preregistration_path=inputs["collection_path"],
        expected_collection_preregistration_file_sha256=identity.file_sha256(
            inputs["collection_path"]
        ),
        expected_collection_preregistration_sha256=inputs["collection"][
            "collection_preregistration_sha256"
        ],
        runtime_contract_path=inputs["runtime_path"],
        expected_runtime_contract_file_sha256=identity.file_sha256(
            inputs["runtime_path"]
        ),
        expected_runtime_contract_sha256=inputs["runtime"][
            "runtime_contract_sha256"
        ],
        collector_path=collector,
        expected_collector_file_sha256=identity.file_sha256(collector),
        runtime_adapter_path=adapter,
        expected_runtime_adapter_file_sha256=identity.file_sha256(adapter),
        sealed_worker_path=worker,
        expected_sealed_worker_file_sha256=identity.file_sha256(worker),
        event_spec_path=inputs["event_spec"],
        expected_event_spec_file_sha256=identity.file_sha256(inputs["event_spec"]),
        gpu_lock_path=inputs["gpu_lock"],
        verify_runtime_files=True,
    )
    return inputs, authority


def persist_authority_and_plan(
    tmp_path: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path]:
    authority_path = write_json(tmp_path / "runner-authority.json", authority)
    plan = runner.build_static_plan(
        authority_path=authority_path,
        expected_authority_file_sha256=identity.file_sha256(authority_path),
    )
    root = Path(inputs["output_root"])
    root.mkdir(mode=0o755)
    control = root / "_runner"
    control.mkdir(mode=0o700)
    (control / "stages").mkdir(mode=0o700)
    (control / "staging").mkdir(mode=0o700)
    runner.atomic_json(control / "static_plan.json", plan)
    runner.replace_json(control / "state.json", {"status": "synthetic_claim"})
    return authority_path, plan, root


def write_synthetic_payload(
    *, command: list[str], log_path: Path, authority_path: Path
) -> int:
    del authority_path
    ordinal = int(command[command.index("--global-ordinal") + 1])
    payload = Path(command[command.index("--staging-payload") + 1])
    authority = json.loads(
        Path(command[command.index("--authority") + 1]).read_text(encoding="utf-8")
    )
    prereg_path = Path(authority["collection_preregistration"]["path"])
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    collection_command = prereg["commands"][ordinal]
    identity_authority = json.loads(
        Path(authority["collection_identity_authority"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    identity_row = identity_authority["selected_rows"][ordinal]
    static_plan = json.loads(
        Path(command[command.index("--static-plan") + 1]).read_text(encoding="utf-8")
    )
    payload.mkdir(mode=0o700)
    runner.atomic_json(payload / "object_registry.json", {"synthetic": True})
    runner.atomic_json(payload / "pose_quality_spec.json", {"synthetic": True})
    reset = runner.signed(
        {
            "format": runner.WORKER_RESET_RECEIPT_FORMAT,
            "status": "identity_verified_before_first_policy_query",
            "runner_authority_sha256": authority["runner_authority_sha256"],
            "runner_plan_sha256": static_plan["runner_plan_sha256"],
            "command_sha256": collection_command["command_sha256"],
            "global_ordinal": ordinal,
            "split": collection_command["split"],
            "requested_seed": collection_command["requested_seed"],
            "resolved_seed": collection_command["expected_resolved_seed"],
            "pair_id": collection_command["pair_id"],
            "initial_scene_state_sha256": identity_row[
                "initial_scene_state_sha256"
            ],
            "initial_measured_joint_state_sha256": identity_row[
                "initial_measured_joint_state_sha256"
            ],
            "initial_commanded_drive_target_sha256": identity_row[
                "initial_commanded_drive_target_sha256"
            ],
            "object_registry_sha256": "1" * 64,
            "pose_spec_sha256": "2" * 64,
            "identity_validation_count_before_policy_query": 1,
            "policy_queries_before_reset_receipt": 0,
            "outcome_or_label_read_before_reset_receipt": False,
            "evaluation400": False,
        },
        "reset_receipt_sha256",
    )
    runner.atomic_json(payload / "per_seed_reset_receipt.json", reset)
    accounting = runner.signed(
        {
            "format": runner.WORKER_ACCOUNTING_FORMAT,
            "status": "complete_four_original_candidate_records",
            "command_sha256": collection_command["command_sha256"],
            "candidate_original_indices": [0, 1, 2, 3],
            "records": [
                {
                    "original_candidate_index": index,
                    "native_action_sha256": f"{index + 3:x}" * 64,
                    "feasible": True,
                    "executed": True,
                    "right_censored": False,
                    "execution_status": "executed_legal_branch",
                }
                for index in range(4)
            ],
            "success_event_outcome_or_label_included": False,
        },
        "candidate_accounting_sha256",
    )
    runner.atomic_json(payload / "candidate_accounting.json", accounting)
    group_path = payload / "schema6_group.hdf5"
    group_path.write_bytes(f"opaque synthetic group {ordinal}".encode("ascii"))
    formal = collection_command["split"] == "formal_target_validation"
    receipt = runner.signed(
        {
            "format": runner.WORKER_GROUP_RECEIPT_FORMAT,
            "status": "complete_exact_four_candidate_accounting",
            "runner_authority_sha256": authority["runner_authority_sha256"],
            "command_sha256": collection_command["command_sha256"],
            "global_ordinal": ordinal,
            "split": collection_command["split"],
            "requested_seed": collection_command["requested_seed"],
            "resolved_seed": collection_command["expected_resolved_seed"],
            "pair_id": collection_command["pair_id"],
            "candidate_original_indices": [0, 1, 2, 3],
            "candidate_accounting_records": 4,
            "candidate_accounting_sha256": accounting[
                "candidate_accounting_sha256"
            ],
            "per_seed_reset_receipt_sha256": reset["reset_receipt_sha256"],
            "object_registry_sha256": "1" * 64,
            "pose_spec_sha256": "2" * 64,
            "group_file_sha256": runner.opaque_file_sha256(group_path),
            "formal_payload_sealed": formal,
            "outcome_or_label_fields_disclosed_to_runner": False,
            "evaluation400": False,
        },
        runner.GROUP_SIGNATURE,
    )
    runner.atomic_json(payload / "completed_group_receipt.json", receipt)
    if formal:
        for path in payload.iterdir():
            path.chmod(0o400)
        payload.chmod(0o500)
    else:
        for path in payload.iterdir():
            path.chmod(0o444)
        payload.chmod(0o555)
    log_path.write_text("synthetic sealed worker complete\n", encoding="utf-8")
    return 0


def test_authority_consumes_and_revalidates_exact_300_group_contract(
    tmp_path: Path,
) -> None:
    inputs, authority = build_authority(tmp_path)
    decoded = runner.validate_runner_authority(
        authority, verify_runtime_files=True
    )
    assert decoded["output_root"] == inputs["output_root"]
    assert len(decoded["commands"]) == 300
    assert authority["exact_execution"]["split_counts"] == {
        "adaptation_train": 80,
        "adaptation_internal_validation": 30,
        "formal_target_validation": 190,
    }
    assert authority["exact_execution"]["planned_candidate_accounting_records"] == 1200
    assert authority["exact_execution"]["retry_failed_command_allowed"] is False
    assert authority["permissions"]["evaluation400_execution_authorized"] is False
    assert authority["label_boundary"] == {
        "sealed_worker_may_generate_group_payload": True,
        "runner_or_watcher_opens_group_hdf5": False,
        "runner_or_watcher_reads_success_event_outcome_or_label": False,
        "formal_target_validation_groups_sealed_immediately": 190,
        "formal_target_validation_label_open_authorized": False,
        "formal_target_validation_checkpoint_selection_authorized": False,
    }


def test_dry_run_plan_binds_all_command_and_implementation_shas(tmp_path: Path) -> None:
    inputs, authority = build_authority(tmp_path)
    authority_path = write_json(tmp_path / "runner-authority.json", authority)
    plan = runner.build_static_plan(
        authority_path=authority_path,
        expected_authority_file_sha256=identity.file_sha256(authority_path),
    )
    assert plan["command_count"] == 300
    assert plan["candidate_accounting_records"] == 1200
    assert len(plan["command_sha256"]) == len(set(plan["command_sha256"])) == 300
    assert plan["retry_or_resume_allowed"] is False
    assert plan["evaluation400_commands"] == 0
    assert not inputs["output_root"].exists()


def test_detach_claim_is_server_side_and_non_reentrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, authority = build_authority(tmp_path)
    authority_path = write_json(tmp_path / "runner-authority.json", authority)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: types.SimpleNamespace(pid=424242),
    )
    receipt = runner.claim_output_and_detach(
        authority_path=authority_path,
        expected_authority_file_sha256=identity.file_sha256(authority_path),
        idle_interval_seconds=1.0,
    )
    assert receipt["status"] == (
        "detached_new_session_ppid1_required_before_gpu_or_worker"
    )
    assert receipt["pid"] == 424242
    assert receipt["resume_entrypoint_exposed"] is False
    assert receipt["command"][2] == "serve-existing"
    assert Path(inputs["output_root"], "_runner", "detach_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        runner.claim_output_and_detach(
            authority_path=authority_path,
            expected_authority_file_sha256=identity.file_sha256(authority_path),
            idle_interval_seconds=1.0,
        )


def test_sealed_worker_candidate_adapter_strips_label_bearing_record() -> None:
    record = {
        "status": "collected_development_group",
        "root_query": {
            "feasibility_mask": [True, False, True, True],
            "native_action_sha256": [f"{index + 3:x}" * 64 for index in range(4)],
        },
        "branches": [
            {
                "original_candidate_index": index,
                "success": index == 2,
                "trajectory_success": [False, index == 2],
                "event_names": ["e0", "eK"],
            }
            for index in (0, 2, 3)
        ],
    }
    accounting = worker._candidate_accounting(record, command_sha256="a" * 64)
    assert accounting["candidate_original_indices"] == [0, 1, 2, 3]
    assert accounting["success_event_outcome_or_label_included"] is False
    assert all(
        set(row)
        == {
            "original_candidate_index",
            "native_action_sha256",
            "feasible",
            "executed",
            "right_censored",
            "execution_status",
        }
        for row in accounting["records"]
    )
    assert not any(
        key in row
        for row in accounting["records"]
        for key in ("success", "trajectory_success", "event_names", "outcome", "label")
    )


def test_tampered_command_new_output_or_wrong_implementation_sha_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, authority = build_authority(tmp_path)
    tampered = dict(inputs["collection"])
    tampered["commands"] = [dict(row) for row in tampered["commands"]]
    tampered["commands"][0]["requested_seed"] += 1
    tampered.pop("collection_preregistration_sha256")
    tampered = runner.signed(tampered, "collection_preregistration_sha256")
    with pytest.raises(runner.Development300RunnerError):
        runner.validate_collection_preregistration(
            tampered,
            identity_authority=inputs["identity"],
            identity_authority_path=inputs["identity_path"],
            identity_authority_file_sha256=identity.file_sha256(
                inputs["identity_path"]
            ),
        )

    inputs["output_root"].mkdir()
    authority_path = write_json(tmp_path / "runner-authority.json", authority)
    with pytest.raises(FileExistsError):
        runner.build_static_plan(
            authority_path=authority_path,
            expected_authority_file_sha256=identity.file_sha256(authority_path),
        )

    changed = dict(authority)
    changed["implementations"] = dict(authority["implementations"])
    changed["implementations"]["dense_collector"] = dict(
        authority["implementations"]["dense_collector"]
    )
    changed["implementations"]["dense_collector"]["file_sha256"] = "f" * 64
    changed.pop("runner_authority_sha256")
    changed = runner.signed(changed, "runner_authority_sha256")
    with pytest.raises(runner.Development300RunnerError, match="file SHA"):
        runner.validate_runner_authority(changed, verify_runtime_files=True)


def test_synthetic_exact_sequence_publishes_300_once_and_seals_formal190(
    tmp_path: Path,
) -> None:
    inputs, authority = build_authority(tmp_path)
    authority_path, plan, root = persist_authority_and_plan(
        tmp_path, inputs, authority
    )
    terminal = runner.execute_exact_sequence(
        root=root,
        plan=plan,
        authority_path=authority_path,
        authority_file_sha256=identity.file_sha256(authority_path),
        launch_worker=write_synthetic_payload,
    )
    assert terminal["status"] == runner.TERMINAL_SUCCESS
    assert terminal["completed_groups"] == 300
    assert terminal["candidate_accounting_records"] == 1200
    assert terminal["formal_payloads_sealed"] == 190
    assert terminal["formal_label_opened_by_runner_or_watcher"] is False
    assert terminal["evaluation400_commands_executed"] == 0
    formal_commands = inputs["collection"]["commands"][110:]
    assert len(formal_commands) == 190
    for command in formal_commands:
        seed_root = Path(command["outputs"]["seed_root"])
        assert stat.S_IMODE(seed_root.stat().st_mode) == 0o500
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o400
            for path in seed_root.iterdir()
        )
    assert not (root / "_runner" / "staging" / "stage_300").exists()


def test_synthetic_worker_failure_writes_atomic_terminal_receipt_and_never_retries(
    tmp_path: Path,
) -> None:
    inputs, authority = build_authority(tmp_path)
    authority_path, plan, root = persist_authority_and_plan(
        tmp_path, inputs, authority
    )
    calls: list[int] = []

    def fail_at_five(*, command: list[str], log_path: Path, authority_path: Path) -> int:
        ordinal = int(command[command.index("--global-ordinal") + 1])
        calls.append(ordinal)
        if ordinal == 5:
            log_path.write_text("synthetic failure\n", encoding="utf-8")
            return 17
        return write_synthetic_payload(
            command=command, log_path=log_path, authority_path=authority_path
        )

    with pytest.raises(runner.Development300RunnerError, match="retry and resume"):
        runner.execute_exact_sequence(
            root=root,
            plan=plan,
            authority_path=authority_path,
            authority_file_sha256=identity.file_sha256(authority_path),
            launch_worker=fail_at_five,
        )
    assert calls == [0, 1, 2, 3, 4, 5]
    final = json.loads(
        (root / "_runner" / "final_receipt.json").read_text(encoding="utf-8")
    )
    assert final["status"] == runner.TERMINAL_FAILURE
    assert final["completed_groups"] == 5
    assert final["failed_global_ordinal"] == 5
    assert final["retry_or_resume_authorized"] is False
    assert final["formal_label_opened_by_runner_or_watcher"] is False
    assert not (root / "_runner" / "stages" / "stage_006").exists()
