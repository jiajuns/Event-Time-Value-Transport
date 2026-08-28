from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for path in (SCRIPTS, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import materialize_smolvla_piper_schema6_training_manifest_v3 as v3  # noqa: E402
import run_smolvla_piper_schema6_development300_collection as runner  # noqa: E402
from etsf_schema6_pose_quality import registry_sha256, spec_sha256  # noqa: E402
from test_materialize_smolvla_piper_schema6_training_manifest_v2 import (  # noqa: E402
    _pose_spec,
    _registry,
)
from test_run_smolvla_piper_schema6_development300_collection import (  # noqa: E402
    build_authority,
    persist_authority_and_plan,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            path = base / name
            if not path.is_symlink():
                path.chmod(0o600)
        for name in names:
            path = base / name
            if not path.is_symlink():
                path.chmod(0o700)
        base.chmod(0o700)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture()
def completed_collection() -> Iterator[dict[str, Any]]:
    workspace = Path(tempfile.mkdtemp(prefix="schema6v3-", dir="/tmp"))
    try:
        inputs, authority = build_authority(workspace)
        authority_path, plan, root = persist_authority_and_plan(
            workspace, inputs, authority
        )
        identity_authority = inputs["identity"]
        stage_shas: list[str] = []
        empty_sha = hashlib.sha256(b"").hexdigest()
        plan_path = root / "_runner" / "static_plan.json"
        plan_file_sha = v3.metadata_file_sha256(plan_path, "runner static plan")
        detach = runner.signed(
            {
                "format": runner.DETACH_RECEIPT_FORMAT,
                "status": "detached_new_session_ppid1_required_before_gpu_or_worker",
                "pid": 424242,
                "runner_plan_sha256": plan["runner_plan_sha256"],
                "output_root": str(root),
                "command": [
                    authority["implementations"]["runtime_python"]["path"],
                    authority["implementations"]["runner"]["path"],
                    "serve-existing",
                    "--output-root",
                    str(root),
                    "--idle-interval-seconds",
                    "1.0",
                ],
                "resume_entrypoint_exposed": False,
            },
            "detach_receipt_sha256",
        )
        _write_json(root / "_runner" / "detach_receipt.json", detach)
        (root / "_runner" / "runner.log").touch()
        for ordinal, command in enumerate(inputs["collection"]["commands"]):
            seed_root = Path(command["outputs"]["seed_root"])
            seed_root.mkdir(parents=True)
            registry = _registry(ordinal)
            pose = _pose_spec(registry)
            registry_logical = registry_sha256(registry)
            pose_logical = spec_sha256(
                pose, expected_registry_sha256=registry_logical
            )
            _write_json(seed_root / "object_registry.json", registry)
            _write_json(seed_root / "pose_quality_spec.json", pose)
            identity_row = identity_authority["selected_rows"][ordinal]
            reset = runner.signed(
                {
                    "format": runner.WORKER_RESET_RECEIPT_FORMAT,
                    "status": "identity_verified_before_first_policy_query",
                    "runner_authority_sha256": authority["runner_authority_sha256"],
                    "runner_plan_sha256": plan["runner_plan_sha256"],
                    "command_sha256": command["command_sha256"],
                    "global_ordinal": ordinal,
                    "split": command["split"],
                    "requested_seed": command["requested_seed"],
                    "resolved_seed": command["expected_resolved_seed"],
                    "pair_id": command["pair_id"],
                    "initial_scene_state_sha256": identity_row[
                        "initial_scene_state_sha256"
                    ],
                    "initial_measured_joint_state_sha256": identity_row[
                        "initial_measured_joint_state_sha256"
                    ],
                    "initial_commanded_drive_target_sha256": identity_row[
                        "initial_commanded_drive_target_sha256"
                    ],
                    "object_registry_sha256": registry_logical,
                    "pose_spec_sha256": pose_logical,
                    "identity_validation_count_before_policy_query": 1,
                    "policy_queries_before_reset_receipt": 0,
                    "outcome_or_label_read_before_reset_receipt": False,
                    "evaluation400": False,
                },
                "reset_receipt_sha256",
            )
            _write_json(seed_root / "per_seed_reset_receipt.json", reset)
            accounting = runner.signed(
                {
                    "format": runner.WORKER_ACCOUNTING_FORMAT,
                    "status": "complete_four_original_candidate_records",
                    "command_sha256": command["command_sha256"],
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
            _write_json(seed_root / "candidate_accounting.json", accounting)
            group_path = seed_root / "schema6_group.hdf5"
            group_path.touch()
            group_path.chmod(0o444)
            formal = command["split"] == "formal_target_validation"
            receipt = runner.signed(
                {
                    "format": runner.WORKER_GROUP_RECEIPT_FORMAT,
                    "status": "complete_exact_four_candidate_accounting",
                    "runner_authority_sha256": authority["runner_authority_sha256"],
                    "command_sha256": command["command_sha256"],
                    "global_ordinal": ordinal,
                    "split": command["split"],
                    "requested_seed": command["requested_seed"],
                    "resolved_seed": command["expected_resolved_seed"],
                    "pair_id": command["pair_id"],
                    "candidate_original_indices": [0, 1, 2, 3],
                    "candidate_accounting_records": 4,
                    "candidate_accounting_sha256": accounting[
                        "candidate_accounting_sha256"
                    ],
                    "per_seed_reset_receipt_sha256": reset[
                        "reset_receipt_sha256"
                    ],
                    "object_registry_sha256": registry_logical,
                    "pose_spec_sha256": pose_logical,
                    "group_file_sha256": empty_sha,
                    "formal_payload_sealed": formal,
                    "outcome_or_label_fields_disclosed_to_runner": False,
                    "evaluation400": False,
                },
                runner.GROUP_SIGNATURE,
            )
            _write_json(seed_root / "completed_group_receipt.json", receipt)
            if formal:
                for path in seed_root.iterdir():
                    path.chmod(0o400)
                seed_root.chmod(0o500)
            else:
                seed_root.chmod(0o555)

            stage_root = root / "_runner" / "stages" / f"stage_{ordinal:03d}"
            stage_root.mkdir()
            worker_command = [
                authority["implementations"]["runtime_python"]["path"],
                authority["implementations"]["sealed_group_worker"]["path"],
                "collect-one",
                "--authority",
                str(authority_path),
                "--authority-file-sha256",
                v3.metadata_file_sha256(authority_path, "runner authority"),
                "--static-plan",
                str(plan_path),
                "--static-plan-file-sha256",
                plan_file_sha,
                "--global-ordinal",
                str(ordinal),
                "--staging-payload",
                str(root / "_runner" / "staging" / f"stage_{ordinal:03d}" / "payload"),
            ]
            launch = runner.signed(
                {
                    "format": runner.STAGE_RECEIPT_FORMAT,
                    "status": "launching_exact_once",
                    "global_ordinal": ordinal,
                    "command_sha256": command["command_sha256"],
                    "worker_command": worker_command,
                    "retry_allowed": False,
                },
                "stage_receipt_sha256",
            )
            _write_json(stage_root / "launch_receipt.json", launch)
            (stage_root / "worker.log").touch()
            stage = runner.signed(
                {
                    "format": runner.STAGE_RECEIPT_FORMAT,
                    "status": "published_exact_once",
                    "global_ordinal": ordinal,
                    "command_sha256": command["command_sha256"],
                    "sealed_group_receipt_sha256": receipt[
                        runner.GROUP_SIGNATURE
                    ],
                    "group_file_sha256": empty_sha,
                    "formal_payload_sealed": formal,
                    "retry_performed": False,
                },
                "stage_receipt_sha256",
            )
            _write_json(stage_root / "terminal_receipt.json", stage)
            stage_shas.append(stage["stage_receipt_sha256"])

        terminal = runner.signed(
            {
                "format": runner.TERMINAL_RECEIPT_FORMAT,
                "status": runner.TERMINAL_SUCCESS,
                "runner_authority_sha256": authority["runner_authority_sha256"],
                "runner_plan_sha256": plan["runner_plan_sha256"],
                "completed_groups": 300,
                "candidate_accounting_records": 1200,
                "split_counts": dict(runner.SPLIT_COUNTS),
                "formal_payloads_sealed": 190,
                "gap_free_exact_command_order": True,
                "retry_replacement_additional_seed_or_resume_performed": False,
                "formal_label_opened_by_runner_or_watcher": False,
                "evaluation400_commands_executed": 0,
                "stage_receipt_order_sha256": runner.canonical_sha256(stage_shas),
            },
            "terminal_receipt_sha256",
        )
        _write_json(root / "_runner" / "final_receipt.json", terminal)
        runner.replace_json(
            root / "_runner" / "state.json",
            {
                "status": runner.TERMINAL_SUCCESS,
                "receipt_sha256": terminal["terminal_receipt_sha256"],
            },
        )
        runner._freeze_tree(root)
        yield {
            "workspace": workspace,
            "inputs": inputs,
            "authority": authority,
            "authority_path": authority_path,
            "plan": plan,
            "root": root,
            "terminal": terminal,
            "target_path": workspace / "development300-prereg.json",
            "trainer_path": SCRIPTS
            / "train_smolvla_piper_schema6_embodiment_adapter.py",
        }
    finally:
        _make_writable(workspace)
        shutil.rmtree(workspace, ignore_errors=True)


def _kwargs(case: Mapping[str, Any], output: Path) -> dict[str, Any]:
    terminal_path = case["root"] / "_runner" / "final_receipt.json"
    identity_path = case["inputs"]["identity_path"]
    return {
        "collection_root": case["root"],
        "terminal_receipt_path": terminal_path,
        "expected_terminal_receipt_file_sha256": v3.metadata_file_sha256(
            terminal_path, "terminal"
        ),
        "expected_terminal_receipt_sha256": case["terminal"][
            "terminal_receipt_sha256"
        ],
        "runner_authority_path": case["authority_path"],
        "expected_runner_authority_file_sha256": v3.metadata_file_sha256(
            case["authority_path"], "authority"
        ),
        "expected_runner_authority_sha256": case["authority"][
            "runner_authority_sha256"
        ],
        "target_preregistration_path": case["target_path"],
        "expected_target_preregistration_file_sha256": v3.metadata_file_sha256(
            case["target_path"], "target preregistration"
        ),
        "expected_target_preregistration_sha256": json.loads(
            case["target_path"].read_text(encoding="utf-8")
        )["preregistration_sha256"],
        "identity_authority_path": identity_path,
        "expected_identity_authority_file_sha256": v3.metadata_file_sha256(
            identity_path, "identity authority"
        ),
        "expected_identity_authority_sha256": case["inputs"]["identity"][
            "identity_authority_sha256"
        ],
        "bound_trainer_path": case["trainer_path"],
        "expected_bound_trainer_file_sha256": v3.metadata_file_sha256(
            case["trainer_path"], "trainer"
        ),
        "output_directory": output,
        "verify_runtime_files": True,
    }


def test_empty_hdf_end_to_end_is_trainer_compatible_and_never_opens_hdf(
    completed_collection: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened_hdf: list[Path] = []
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        if path.suffix.casefold() in v3.HDF_SUFFIXES:
            opened_hdf.append(path)
            raise AssertionError("HDF byte access attempted")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    output = completed_collection["workspace"] / "materialized-v3"
    receipt = v3.materialize(**_kwargs(completed_collection, output))
    assert receipt["status"] == v3.COMPLETE_STATUS
    assert receipt["hdf5_content_files_opened"] == 0
    assert receipt["formal_target_validation_hdf5_or_labels_opened"] == 0
    assert opened_hdf == []
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert {path.name for path in output.iterdir()} == set(v3.OUTPUT_NAMES.values())

    manifest_path = output / v3.OUTPUT_NAMES["manifest"]
    expected_path = output / v3.OUTPUT_NAMES["expected"]
    partition = json.loads(
        (output / v3.OUTPUT_NAMES["partition"]).read_text(encoding="utf-8")
    )
    split = json.loads(
        (output / v3.OUTPUT_NAMES["split"]).read_text(encoding="utf-8")
    )
    assert len(partition["adaptation"]) == 110
    assert len(partition["validation"]) == 190
    assert list(map(len, (split["train"], split["validation"], split["test"]))) == [
        80,
        30,
        190,
    ]

    import train_smolvla_piper_schema6_embodiment_adapter as trainer

    manifest, descriptors = trainer.scan_manifest(manifest_path)
    decoded_split, audit = trainer.validate_external_split_authority(
        expected_receipt_path=expected_path,
        expected_receipt_file_sha256=trainer.file_sha256(expected_path),
        manifest_path=manifest_path,
        manifest=manifest,
        descriptors=descriptors,
    )
    assert audit["split_profile"] == "development300_v3"
    assert audit["required_trainer_group_counts"] == {
        "train": 80,
        "validation": 30,
        "test": 190,
    }
    assert len(decoded_split["test"]) == 190


def test_outputs_are_create_once(completed_collection: Mapping[str, Any]) -> None:
    output = completed_collection["workspace"] / "one-shot-v3"
    kwargs = _kwargs(completed_collection, output)
    v3.materialize(**kwargs)
    with pytest.raises(FileExistsError):
        v3.materialize(**kwargs)


def test_bool_cannot_impersonate_terminal_or_group_integer(
    completed_collection: Mapping[str, Any]
) -> None:
    first_command = completed_collection["inputs"]["collection"]["commands"][0]
    receipt_path = Path(first_command["outputs"]["completed_group_receipt"])
    group_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    group_receipt["global_ordinal"] = False
    group_receipt.pop(runner.GROUP_SIGNATURE)
    group_receipt = runner.signed(group_receipt, runner.GROUP_SIGNATURE)
    with pytest.raises(v3.TrainingManifestV3Error, match="group receipt"):
        v3._validate_group_receipt(
            group_receipt,
            command=first_command,
            authority_sha256=completed_collection["authority"][
                "runner_authority_sha256"
            ],
        )

    terminal = dict(completed_collection["terminal"])
    terminal["completed_groups"] = True
    terminal.pop("terminal_receipt_sha256")
    terminal = runner.signed(terminal, "terminal_receipt_sha256")
    with pytest.raises(v3.TrainingManifestV3Error, match="terminal"):
        v3._validate_terminal(
            terminal,
            authority_sha256=completed_collection["authority"][
                "runner_authority_sha256"
            ],
            plan_sha256=completed_collection["plan"]["runner_plan_sha256"],
            expected_logical_sha256=terminal["terminal_receipt_sha256"],
        )


def test_symlink_writable_or_forbidden_input_fails_closed(
    completed_collection: Mapping[str, Any]
) -> None:
    workspace = completed_collection["workspace"]
    outside = workspace / "opaque.bin"
    outside.touch()
    linked = workspace / "linked.hdf5"
    linked.symlink_to(outside)
    with pytest.raises(v3.TrainingManifestV3Error, match="symlink"):
        v3._regular_file(linked, "linked HDF", frozen=True, allow_hdf=True)
    with pytest.raises(v3.TrainingManifestV3Error, match="forbidden"):
        v3.safe_path(workspace / "fresh" / "anything.json", "fresh input", must_exist=False)
    with pytest.raises(v3.TrainingManifestV3Error, match="forbidden"):
        v3.safe_path(workspace / "test" / "anything.json", "test input", must_exist=False)

    root = completed_collection["root"]
    root.chmod(0o755)
    try:
        with pytest.raises(v3.TrainingManifestV3Error, match="read-only"):
            v3.materialize(
                **_kwargs(completed_collection, workspace / "writable-rejected-v3")
            )
    finally:
        root.chmod(0o555)


def test_duplicate_group_path_breaks_identity_closure(
    completed_collection: Mapping[str, Any]
) -> None:
    commands = [dict(row) for row in completed_collection["inputs"]["collection"]["commands"]]
    commands[1] = {**commands[1], "outputs": dict(commands[0]["outputs"])}
    target = json.loads(completed_collection["target_path"].read_text(encoding="utf-8"))
    with pytest.raises(v3.TrainingManifestV3Error, match="uniqueness"):
        v3._validate_identity_closure(
            target=target,
            identity=completed_collection["inputs"]["identity"],
            commands=commands,
        )
