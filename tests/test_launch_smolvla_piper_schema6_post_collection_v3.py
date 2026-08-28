from __future__ import annotations

import json
import fcntl
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_schema6_post_collection_v3 as post  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _write_json(path: Path, value: Mapping[str, Any], mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(mode)


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(base)
    value[field] = post.canonical_sha256(value)
    return value


def _source_rank_contract(source_sha256: str) -> dict[str, Any]:
    value = {
        "format": post.trainer.SOURCE_RANK_SCORE_FORMAT,
        "status": "frozen_exact_source63_training_score_scientific_rank_only",
        "source_checkpoint_file_sha256": source_sha256,
        "source_action_rank_residual": True,
        "source_action_rank_success_only": False,
        "source_freeze_factual_core": False,
        "source_rank_numeric_contract": post.trainer.SOURCE_RANK_NUMERIC_CONTRACT,
        "base_score": "candidate_rank_score",
        "event_names": ["e0", "e12", "e3", "e4", "eK"],
        "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "event_values_authority": "source_trainer_linspace_0_1_in_checkpoint_event_order",
        "duration_scale": 200.0 + int(source_sha256[0], 16),
        "duration_scale_authority": "source_member_checkpoint.duration_scale",
        "duration_scale_scope": "per_source_member_not_ensemble_mean",
        "duration_unit": "decision_steps",
        "success_temperature": post.trainer.SOURCE_RANK_SUCCESS_TEMPERATURE,
        "event_weight": post.trainer.SOURCE_RANK_EVENT_WEIGHT,
        "duration_weight": post.trainer.SOURCE_RANK_DURATION_WEIGHT,
        "residual_combination": "candidate_rank_score_plus_action_rank_residual",
        "score_variant": "source_member_training_objective_defaults",
        "source_ensemble_validation_selected_scoring_consumed": False,
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "cross_embodiment_duration_scale_calibrated": False,
        "deployment_success_probability_selector_authorized": False,
    }
    value["contract_sha256"] = post.canonical_sha256(value)
    return value


def _materializer_tree(root: Path) -> dict[str, dict[str, Any]]:
    partition = _signed(
        {
            "format": post.materializer.TARGET_PARTITION_FORMAT,
            "split_profile": post.SPLIT_PROFILE,
            "adaptation": [f"a{i}" for i in range(110)],
            "validation": [f"f{i}" for i in range(190)],
            "evaluation": [],
        },
        "partition_sha256",
    )
    split = _signed(
        {
            "format": post.materializer.EXTERNAL_SPLIT_FORMAT,
            "split_profile": post.SPLIT_PROFILE,
            "train": [f"a{i}" for i in range(80)],
            "validation": [f"a{i}" for i in range(80, 110)],
            "test": [f"f{i}" for i in range(190)],
        },
        "split_sha256",
    )
    manifest = _signed(
        {"format": post.materializer.TRAINER_MANIFEST_FORMAT}, "manifest_sha256"
    )
    paths = {
        "partition": root / post.MATERIALIZER_OUTPUTS["partition"],
        "split": root / post.MATERIALIZER_OUTPUTS["split"],
        "manifest": root / post.MATERIALIZER_OUTPUTS["manifest"],
        "expected": root / post.MATERIALIZER_OUTPUTS["expected"],
        "receipt": root / post.MATERIALIZER_OUTPUTS["receipt"],
    }
    for role, value in (("partition", partition), ("split", split), ("manifest", manifest)):
        _write_json(paths[role], value)
    records = {
        role: {
            "path": str(paths[role]),
            "file_sha256": post.file_sha256(paths[role]),
            "logical_sha256": value[signature],
        }
        for role, value, signature in (
            ("partition", partition, "partition_sha256"),
            ("split", split, "split_sha256"),
            ("manifest", manifest, "manifest_sha256"),
        )
    }
    expected = _signed(
        {
            "format": post.materializer.EXPECTED_FORMAT,
            "split_profile": post.SPLIT_PROFILE,
            "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
        },
        "expected_receipt_sha256",
    )
    _write_json(paths["expected"], expected)
    records["expected"] = {
        "path": str(paths["expected"]),
        "file_sha256": post.file_sha256(paths["expected"]),
        "logical_sha256": expected["expected_receipt_sha256"],
    }
    receipt = _signed(
        {
            "format": post.materializer.FORMAT,
            "status": post.materializer.COMPLETE_STATUS,
            "training_inputs_complete": True,
            "training_authorized": True,
            "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
            "group_count": 300,
            "formal_target_validation_hdf5_or_labels_opened": 0,
            "evaluation400_identity_or_execution_authorized": False,
            "hdf5_content_files_opened": 0,
            "labels_or_outcomes_read": False,
            "trainer_compatible_manifest": records["manifest"],
            "target_partition": records["partition"],
            "external_split": records["split"],
            "expected_manifest_split_receipt": records["expected"],
        },
        "receipt_sha256",
    )
    _write_json(paths["receipt"], receipt)
    root.chmod(0o555)
    return {"receipt": receipt, **records}


def test_materializer_v3_exact_four_outputs_and_80_30_190(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "manifest_v3"
    root.mkdir()
    _materializer_tree(root)
    split = json.loads((root / post.MATERIALIZER_OUTPUTS["split"]).read_text())
    monkeypatch.setattr(post.trainer, "scan_manifest", lambda _path: ({}, []))
    monkeypatch.setattr(
        post.trainer,
        "validate_external_split_authority",
        lambda **_kwargs: (
            split,
            {
                "split_profile": post.SPLIT_PROFILE,
                "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
            },
        ),
    )
    audit = post.validate_materializer_v3_outputs(root)
    assert audit["split_profile"] == "development300_v3"
    assert audit["required_trainer_group_counts"] == {
        "train": 80,
        "validation": 30,
        "test": 190,
    }
    assert len(audit["four_downstream_outputs"]) == 4
    assert audit["formal190_hdf5_or_labels_opened"] == 0


def test_materializer_v3_rejects_tampered_formal_count(tmp_path: Path) -> None:
    root = tmp_path / "manifest_v3"
    root.mkdir()
    _materializer_tree(root)
    root.chmod(0o755)
    split_path = root / post.MATERIALIZER_OUTPUTS["split"]
    split_path.chmod(0o644)
    split = json.loads(split_path.read_text())
    split["test"].pop()
    split["split_sha256"] = post.canonical_sha256(
        {key: value for key, value in split.items() if key != "split_sha256"}
    )
    _write_json(split_path, split)
    root.chmod(0o555)
    with pytest.raises(post.PostCollectionV3Error):
        post.validate_materializer_v3_outputs(root)


def _member_fixture(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    members: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    contract = {
        "duration_target_transform": "log1p_decision_steps",
        "next_event_observation_mask": "duration_observed",
        "success_target": "eventual_final_branch_success_repeated_per_transition",
        "recovery_target": "conditional_recovery_given_operational_regress",
        "recovery_observation_mask": "recovery_observed_and_regress",
        "recovery_shared_transition_stop_gradient": True,
        "recovery_enters_primary_before_calibration": False,
        "recovery_head_trained": True,
        "object_prediction_space": "physical_delta_xyz_m",
        "object_source_normalization_sha256": "a" * 64,
        "object_observed_policy": "row_enabled_only_if_all_selected_xyz_are_valid",
    }
    for index, seed in enumerate(post.SOURCE_MEMBER_SEEDS):
        source_path = root / "source" / f"member_{index}.pt"
        member_root = root / "members" / f"member_{index}"
        adapter_path = member_root / "adapter.pt"
        summary_path = member_root / "training_summary.json"
        predictions_path = member_root / "internal_predictions.npz"
        labels_path = member_root / "internal_labels.npz"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(f"source-{index}".encode())
        adapter_path.write_bytes(f"adapter-{index}".encode())
        predictions_path.write_bytes(f"predictions-{index}".encode())
        labels_path.write_bytes(f"labels-{index}".encode())
        summary = _signed({"status": "complete"}, "summary_sha256")
        _write_json(summary_path, summary)
        source_path.chmod(0o444)
        adapter_path.chmod(0o444)
        predictions_path.chmod(0o444)
        labels_path.chmod(0o444)
        source = {
            "member_index": index,
            "member_seed": seed,
            "path": str(source_path),
            "file_sha256": post.file_sha256(source_path),
            "checkpoint_role": "native_r7h_individual_source_member",
        }
        source_rank_contract = _source_rank_contract(source["file_sha256"])
        receipt = _signed(
            {
                "format": post.MEMBER_FORMAT,
                "status": "complete_frozen_development300_internal_validation_adapter",
                "member_index": index,
                "member_seed": seed,
                "split_profile": post.SPLIT_PROFILE,
                "split_profile_version": 3,
                "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
                "source_checkpoint_path": str(source_path),
                "source_checkpoint_sha256": source["file_sha256"],
                "source_checkpoint_role": source["checkpoint_role"],
                "summary_path": str(summary_path),
                "summary_file_sha256": post.file_sha256(summary_path),
                "summary_sha256": summary["summary_sha256"],
                "checkpoint_path": str(adapter_path),
                "checkpoint_file_sha256": post.file_sha256(adapter_path),
                "validation_predictions_path": str(predictions_path),
                "validation_predictions_file_sha256": post.file_sha256(predictions_path),
                "validation_predictions_logical_sha256": "4" * 64,
                "validation_labels_path": str(labels_path),
                "validation_labels_file_sha256": post.file_sha256(labels_path),
                "validation_labels_logical_sha256": "5" * 64,
                "validation_identity_set_sha256": "6" * 64,
                "validation_lane": "adaptation_derived_internal_validation_only",
                "internal_validation_group_count": 30,
                "training_manifest_sha256": "b" * 64,
                "split_sha256": "c" * 64,
                "source_ensemble_contract_sha256": "d" * 64,
                "prediction_contract": contract,
                "source_rank_score_contract": source_rank_contract,
                "source_rank_score_contract_sha256": source_rank_contract[
                    "contract_sha256"
                ],
                "sealed_formal_target_validation_group_count": 190,
                "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen": 0,
                "formal_target_validation_labels_opened_before_five_adapters_frozen": 0,
                "formal_target_validation_release_condition": "external_authority_after_all_five_adapter_checkpoints_are_frozen",
                "lobo_or_aggregate_checkpoint_used": False,
                "stage_result_sha256": "7" * 64,
            },
            "receipt_sha256",
        )
        _write_json(adapter_path.parent / "final_receipt.json", receipt)
        members.append(receipt)
        sources.append(source)
    source = {
        "members": sources,
        "ensemble_manifest": {"file_sha256": "d" * 64},
    }
    manifest_path = root / "manifest.json"
    expected_path = root / "expected.json"
    event_path = root / "event.json"
    for path in (manifest_path, expected_path, event_path):
        _write_json(path, {})
    materialized = {
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": post.file_sha256(manifest_path),
            "logical_sha256": "b" * 64,
        },
        "expected": {
            "path": str(expected_path),
            "file_sha256": post.file_sha256(expected_path),
            "logical_sha256": "e" * 64,
        },
    }
    plan = {"canonical_event_spec": {"path": str(event_path), "file_sha256": post.file_sha256(event_path)}}
    return members, source, materialized, plan


def test_evaluator_authority_is_created_only_for_five_native_members(tmp_path: Path) -> None:
    root = tmp_path / "post"
    (root / "formal190").mkdir(parents=True)
    members, source, materialized, plan = _member_fixture(root)
    path = post.build_evaluator_authority(
        root=root, plan=plan, materialized=materialized, members=members, source=source
    )
    value = json.loads(path.read_text())
    assert value["target_validation_group_count"] == 190
    assert value["adapter_training_complete_before_authority"] is True
    assert value["evaluation400_open_authorized"] is False
    assert value["source_rank_numeric_contract"] == (
        post.trainer.SOURCE_RANK_NUMERIC_CONTRACT
    )
    assert [row["source_checkpoint"]["file_sha256"] for row in value["members"]] == [
        row["file_sha256"] for row in source["members"]
    ]
    with pytest.raises(post.PostCollectionV3Error, match="exactly five"):
        post.build_evaluator_authority(
            root=root, plan=plan, materialized=materialized, members=members[:4], source=source
        )


def test_duplicate_source_checkpoint_cannot_authorize_formal190(tmp_path: Path) -> None:
    root = tmp_path / "post"
    (root / "formal190").mkdir(parents=True)
    members, source, _materialized, _plan = _member_fixture(root)
    members[1]["source_checkpoint_sha256"] = members[0]["source_checkpoint_sha256"]
    members[1]["receipt_sha256"] = post.canonical_sha256(
        {key: value for key, value in members[1].items() if key != "receipt_sha256"}
    )
    receipt_path = root / "members" / "member_1" / "final_receipt.json"
    receipt_path.chmod(0o644)
    _write_json(receipt_path, members[1])
    with pytest.raises(post.PostCollectionV3Error):
        post.validate_five_members(root, members, source)


def test_recovery_head_must_be_trained_before_formal190(tmp_path: Path) -> None:
    root = tmp_path / "post"
    (root / "formal190").mkdir(parents=True)
    members, source, _materialized, _plan = _member_fixture(root)
    members[2]["prediction_contract"]["recovery_head_trained"] = False
    members[2]["receipt_sha256"] = post.canonical_sha256(
        {key: value for key, value in members[2].items() if key != "receipt_sha256"}
    )
    receipt_path = root / "members" / "member_2" / "final_receipt.json"
    receipt_path.chmod(0o644)
    _write_json(receipt_path, members[2])
    with pytest.raises(post.PostCollectionV3Error, match="freeze proof"):
        post.validate_five_members(root, members, source)


def test_formal190_global_claim_is_one_shot_across_outputs(tmp_path: Path) -> None:
    collection = tmp_path / "development300"
    collection.mkdir()
    claim_root = tmp_path / ".etsf_schema6_formal190_global_claims_v1"
    claim_root.mkdir(mode=0o700)
    plan = {
        "plan_sha256": "a" * 64,
        "formal190_claim_root": str(claim_root),
        "development300_terminal": {
            "path": str(collection / "_runner" / "final_receipt.json"),
            "file_sha256": "b" * 64,
            "logical_sha256": "c" * 64,
        },
    }
    first = post.acquire_formal190_claim(plan, output_root=tmp_path / "post_a")
    assert first["consumed"] is True
    assert stat.S_IMODE(Path(first["path"]).stat().st_mode) == 0o444
    with pytest.raises(post.PostCollectionV3Error, match="already consumed"):
        post.acquire_formal190_claim(plan, output_root=tmp_path / "post_b")


def test_member_receipt_v3_is_exact_and_rejects_resigned_extra_field(tmp_path: Path) -> None:
    root = tmp_path / "post"
    (root / "formal190").mkdir(parents=True)
    members, _source, _materialized, _plan = _member_fixture(root)
    audit = post.validate_member_receipt_v3(members[0], member_index=0)
    assert audit["split_profile"] == "development300_v3"
    assert set(members[0]) == post.MEMBER_FIELDS
    tampered = dict(members[0])
    tampered["legacy_target_validation50"] = 0
    tampered["receipt_sha256"] = post.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(post.PostCollectionV3Error, match="fields changed"):
        post.validate_member_receipt_v3(tampered, member_index=0)
    bool_index = dict(members[1])
    bool_index["member_index"] = True
    bool_index["receipt_sha256"] = post.canonical_sha256(
        {key: value for key, value in bool_index.items() if key != "receipt_sha256"}
    )
    with pytest.raises(post.PostCollectionV3Error, match="semantics changed"):
        post.validate_member_receipt_v3(bool_index, member_index=1)


def test_bound_cpu_stage_proves_pid_pgid_and_reaping(tmp_path: Path) -> None:
    result = post.run_bound_stage(
        name="cpu_smoke",
        command=[sys.executable, "-c", "pass"],
        stage_root=tmp_path / "stage",
        gpu_index=None,
        poll_interval=0.01,
        pre_popen_guard=lambda: None,
    )
    lifecycle = result["lifecycle"]
    assert lifecycle["process_pid"] == lifecycle["process_pgid"]
    assert lifecycle["direct_process_reaped"] is True
    assert lifecycle["process_group_reaped"] is True
    assert stat.S_IMODE((tmp_path / "stage").stat().st_mode) == 0o555


def test_unproven_pgid_reaps_direct_child_but_remains_unproven(tmp_path: Path) -> None:
    with pytest.raises(post.UnprovenProcessGroup):
        post.run_bound_stage(
            name="bad_pgid",
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            stage_root=tmp_path / "stage",
            gpu_index=0,
            poll_interval=0.01,
            pre_popen_guard=lambda: None,
            getpgid=lambda pid: pid + 100000,
        )
    lifecycle = json.loads((tmp_path / "stage" / "lifecycle.json").read_text())
    assert lifecycle["direct_process_reaped"] is True
    assert lifecycle["process_group_reaped"] is False
    assert lifecycle["binding_status"] == "attempted_unproven"


def test_unproven_stage_lock_is_still_exclusive(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    owner = lock_path.open("a+")
    contender = lock_path.open("a+")
    try:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
        post.retain_unproven_gpu_lock(owner)
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        post._HELD_UNPROVEN_LOCKS.remove(owner)
        fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
        owner.close()
        contender.close()


def _success_terminal() -> dict[str, Any]:
    stages = [
        "materialize_development300_v3",
        *[f"train_adapter_member_{index}" for index in range(5)],
        "evaluate_frozen_five_member_ensemble_on_formal190",
        "calibrate_six_head_formal190_ensemble",
    ]
    results: dict[str, Any] = {}
    for index, name in enumerate(stages, start=100):
        lifecycle = _signed(
            {
                "popen_attempted": True,
                "popen_reached": True,
                "process_pid": index,
                "process_pgid": index,
                "process_group_isolated": True,
                "returncode": 0,
                "direct_process_reaped": True,
                "process_group_reaped": True,
                "binding_status": "bound_reaped",
            },
            "lifecycle_sha256",
        )
        result = {
            "stage": name,
            "returncode": 0,
            "command_sha256": "1" * 64,
            "lifecycle": lifecycle,
            "log_file_sha256": "2" * 64,
            "run_exit_file_sha256": "3" * 64,
        }
        result["result_sha256"] = post.canonical_sha256(result)
        results[name] = result
    return _signed(
        {
            "format": post.FORMAT,
            "status": post.TERMINAL_STATUS,
            "plan_sha256": "1" * 64,
            "execution_order": stages,
            "stage_results": results,
            "adapter_member_count": 5,
            "adapter_member_seeds": list(post.SOURCE_MEMBER_SEEDS),
            "adapter_source_policy": "one_to_one_native_r7h_individual_members_only",
            "detach_proof_sha256": "4" * 64,
            "r7h_member_checkpoint_sha256": [str(index) * 64 for index in range(1, 6)],
            "r8e_r9b_lineage_sha256": "6" * 64,
            "development300_materializer_receipt_sha256": "7" * 64,
            "formal190_opened_after_five_frozen_adapters": 190,
            "formal190_labels_opened_before_five_adapters_frozen": 0,
            "calibration_receipt_sha256": "8" * 64,
            "identity_bridge_v2_handoff": {
                "path": "/metadata/handoff.json",
                "file_sha256": "9" * 64,
                "logical_sha256": "a" * 64,
            },
            "evaluation400_hdf5_trajectory_or_labels_opened": 0,
            "evaluation400_conditions_executed": 0,
            "old_paired400_authority_waited_or_generated": False,
            "second_reserve400_created": False,
            "gpu_lock_release_sha256": "b" * 64,
            "artifacts_frozen_read_only": True,
            "terminal_publication": "mode000_then_tree_freeze_verify_then_run_exit0444_then_final_receipt0444_last",
        },
        "receipt_sha256",
    )


def test_success_terminal_is_hidden_frozen_and_final_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "artifact.txt").write_text("ok", encoding="utf-8")
    monkeypatch.setattr(post, "_validate_success_receipt", lambda _root, _receipt: None)
    post.publish_terminal(root, receipt=_success_terminal(), success=True)
    assert stat.S_IMODE((root / "artifact.txt").stat().st_mode) == 0o444
    assert stat.S_IMODE((root / "run.exit").stat().st_mode) == 0o444
    assert stat.S_IMODE((root / "final_receipt.json").stat().st_mode) == 0o444
    assert (root / "run.exit").read_text() == "0\n"
    assert not (root / "failure_receipt.json").exists()


def test_terminal_freeze_failure_removes_hidden_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    root.mkdir()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected freeze failure")

    monkeypatch.setattr(post, "freeze_tree", fail)
    monkeypatch.setattr(post, "_validate_success_receipt", lambda _root, _receipt: None)
    with pytest.raises(OSError, match="injected"):
        post.publish_terminal(root, receipt=_success_terminal(), success=True)
    assert not (root / "final_receipt.json").exists()
    assert not (root / "run.exit").exists()


def test_materializer_command_has_no_old_paired400_authority() -> None:
    record = {"path": "/safe/meta.json", "file_sha256": "1" * 64, "logical_sha256": "2" * 64}
    plan = {
        "python": {"path": "/usr/bin/python3"},
        "implementations": {
            "materializer": {"path": "/safe/materializer.py"},
            "trainer": {"path": "/safe/trainer.py", "file_sha256": "3" * 64},
        },
        "development300_collection_root": "/safe/dev300",
        "development300_terminal": record,
        "development300_runner_authority": record,
        "development300_target_preregistration": record,
        "development300_identity_authority": record,
    }
    command = post.materializer_v3_command(plan, Path("/safe/out"))
    joined = " ".join(command).casefold()
    assert "paired400" not in joined
    assert "reserve400" not in joined
    assert "--expected-bound-trainer-file-sha256" in command


def test_runtime_bindings_rehash_actual_import_and_reject_writable_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles = {
        "launcher",
        "materializer",
        "trainer",
        "evaluator",
        "calibrator",
        "identity_bridge_v2",
        "r9b_watcher",
    }
    records: dict[str, dict[str, str]] = {}
    for role in roles:
        path = tmp_path / f"{role}.py"
        path.write_text(f"# {role}\n", encoding="utf-8")
        path.chmod(0o444)
        records[role] = {"path": str(path), "file_sha256": post.file_sha256(path)}
    python = tmp_path / "python"
    event = tmp_path / "events.json"
    python.write_bytes(b"python")
    event.write_text("{}\n", encoding="utf-8")
    python.chmod(0o444)
    event.chmod(0o444)
    monkeypatch.setattr(post.r9b_watcher, "__file__", records["r9b_watcher"]["path"])
    monkeypatch.setattr(post, "R9B_WATCHER_SHA256", records["r9b_watcher"]["file_sha256"])
    monkeypatch.setattr(post, "MATERIALIZER_SHA256", records["materializer"]["file_sha256"])
    plan = {
        "python": {"path": str(python), "file_sha256": post.file_sha256(python)},
        "canonical_event_spec": {"path": str(event), "file_sha256": post.file_sha256(event)},
        "canonical_teacher": None,
        "implementations": records,
        "python_import_closure": {
            Path(record["path"]).stem: dict(record) for record in records.values()
        },
    }
    post.verify_runtime_bindings(plan)
    trainer_path = Path(records["trainer"]["path"])
    trainer_path.chmod(0o644)
    with pytest.raises(post.PostCollectionV3Error, match="frozen read-only"):
        post.verify_runtime_bindings(plan)
    trainer_path.chmod(0o444)
    pre_popen = post.make_pre_popen_guard(
        plan, command=[sys.executable, "-c", "pass"], physical_gpu=None
    )
    trainer_path.chmod(0o644)
    trainer_path.write_text("# changed after plan load\n", encoding="utf-8")
    trainer_path.chmod(0o444)
    with pytest.raises(post.PostCollectionV3Error, match="SHA changed"):
        pre_popen()


def test_guard_to_popen_rename_executes_verified_fd_and_closes_all_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles = {
        "launcher", "materializer", "trainer", "evaluator", "calibrator",
        "identity_bridge_v2", "r9b_watcher",
    }
    records: dict[str, dict[str, str]] = {}
    for role in roles:
        path = tmp_path / f"{role}.py"
        path.write_text(f"# reviewed {role}\n", encoding="utf-8")
        path.chmod(0o444)
        records[role] = {"path": str(path), "file_sha256": post.file_sha256(path)}
    python = tmp_path / "python"
    event = tmp_path / "events.json"
    python.write_bytes(b"reviewed-python")
    event.write_text("{}\n", encoding="utf-8")
    python.chmod(0o444)
    event.chmod(0o444)
    monkeypatch.setattr(post.r9b_watcher, "__file__", records["r9b_watcher"]["path"])
    monkeypatch.setattr(post, "R9B_WATCHER_SHA256", records["r9b_watcher"]["file_sha256"])
    monkeypatch.setattr(post, "MATERIALIZER_SHA256", records["materializer"]["file_sha256"])
    plan = {
        "python": {"path": str(python), "file_sha256": post.file_sha256(python)},
        "canonical_event_spec": {"path": str(event), "file_sha256": post.file_sha256(event)},
        "canonical_teacher": None,
        "implementations": records,
        "python_import_closure": {
            Path(record["path"]).stem: dict(record) for record in records.values()
        },
    }
    trainer = Path(records["trainer"]["path"])
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 515151
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def poll(self) -> int:
            return 0

    def fake_popen(argv: list[str], **kwargs: Any) -> FakeProcess:
        replacement = tmp_path / "replacement.py"
        replacement.write_text("# unreviewed replacement\n", encoding="utf-8")
        replacement.chmod(0o444)
        os.replace(replacement, trainer)
        observed["argv"] = list(argv)
        observed["pass_fds"] = tuple(kwargs["pass_fds"])
        observed["executed_script_bytes"] = Path(argv[4]).read_bytes()
        return FakeProcess()

    command = [str(python), str(trainer), "--synthetic"]
    post.run_bound_stage(
        name="fd_bound_toctou_regression",
        command=command,
        stage_root=tmp_path / "stage",
        gpu_index=None,
        poll_interval=0.0,
        pre_popen_guard=post.make_pre_popen_guard(
            plan, command=command, physical_gpu=None
        ),
        popen=fake_popen,
        getpgid=lambda pid: pid,
    )
    assert observed["argv"][0].startswith("/proc/self/fd/")
    assert observed["argv"][1:4] == ["-I", "-c", post.FD_IMPORT_BOOTSTRAP]
    assert observed["argv"][4].startswith("/proc/self/fd/")
    assert observed["executed_script_bytes"] == b"# reviewed trainer\n"
    assert trainer.read_bytes() == b"# unreviewed replacement\n"
    assert observed["pass_fds"]
    for descriptor in observed["pass_fds"]:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_fd_bound_real_python_and_script_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    shutil.copyfile(sys.executable, python)
    python.chmod(0o555)
    helper = tmp_path / "bound_helper.py"
    helper.write_text("VALUE = 'fd-bound-real'\n", encoding="utf-8")
    helper.chmod(0o444)
    stage_script = tmp_path / "bound_stage.py"
    stage_script.write_text(
        "from bound_helper import VALUE\nprint(VALUE)\n", encoding="utf-8"
    )
    stage_script.chmod(0o444)
    event = tmp_path / "events.json"
    event.write_text("{}\n", encoding="utf-8")
    event.chmod(0o444)
    implementation_record = {
        "path": str(stage_script),
        "file_sha256": post.file_sha256(stage_script),
    }
    roles = {
        "launcher", "materializer", "trainer", "evaluator", "calibrator",
        "identity_bridge_v2", "r9b_watcher",
    }
    implementation_map = {role: dict(implementation_record) for role in roles}
    plan = {
        "python": {"path": str(python), "file_sha256": post.file_sha256(python)},
        "canonical_event_spec": {
            "path": str(event), "file_sha256": post.file_sha256(event)
        },
        "canonical_teacher": None,
        "implementations": implementation_map,
        "python_import_closure": post.build_python_import_closure(implementation_map),
    }
    monkeypatch.setattr(post.r9b_watcher, "__file__", str(stage_script))
    monkeypatch.setattr(post, "R9B_WATCHER_SHA256", implementation_record["file_sha256"])
    monkeypatch.setattr(post, "MATERIALIZER_SHA256", implementation_record["file_sha256"])
    command = [str(python), str(stage_script)]
    post.run_bound_stage(
        name="fd_bound_real_smoke",
        command=command,
        stage_root=tmp_path / "stage",
        gpu_index=None,
        poll_interval=0.0,
        pre_popen_guard=post.make_pre_popen_guard(
            plan, command=command, physical_gpu=None
        ),
    )
    assert (tmp_path / "stage" / "run.log").read_text().strip() == "fd-bound-real"


def test_import_closure_includes_literal_dynamic_local_import_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "dynamic_helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    helper.chmod(0o444)
    stage = tmp_path / "dynamic_stage.py"
    stage.write_text(
        "import importlib\nimportlib.import_module('dynamic_helper')\n",
        encoding="utf-8",
    )
    stage.chmod(0o444)
    implementations = {
        "stage": {"path": str(stage), "file_sha256": post.file_sha256(stage)}
    }
    closure = post.build_python_import_closure(implementations)
    assert set(closure) == {"dynamic_helper", "dynamic_stage"}

    stage.chmod(0o644)
    stage.write_text(
        "import importlib\nname = 'dynamic_helper'\nimportlib.import_module(name)\n",
        encoding="utf-8",
    )
    stage.chmod(0o444)
    implementations["stage"]["file_sha256"] = post.file_sha256(stage)
    with pytest.raises(post.PostCollectionV3Error, match="non-literal dynamic import"):
        post.build_python_import_closure(implementations)


def test_fd_wrapper_keeps_materializer_trainer_data_paths_canonical() -> None:
    python = "/sealed/python"
    materializer = "/sealed/materializer.py"
    trainer = "/sealed/trainer.py"
    event = "/sealed/events.json"
    bindings = {python: 10, materializer: 11, trainer: 12, event: 13}
    closure = {
        "materializer": {"path": materializer, "file_sha256": "a" * 64},
        "trainer": {"path": trainer, "file_sha256": "b" * 64},
    }
    command = [
        python, materializer, "--bound-trainer", trainer,
        "--canonical-event-spec", event,
    ]
    wrapped, pass_fds = post.fd_bound_command(command, bindings, closure)
    assert wrapped[0] == "/proc/self/fd/10"
    assert wrapped[4] == "/proc/self/fd/11"
    assert wrapped[-4:] == ["--bound-trainer", trainer, "--canonical-event-spec", event]
    assert set(pass_fds) == {10, 11, 12}
    assert 13 not in pass_fds


def test_isolated_environment_drops_python_and_dynamic_loader_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/attacker")
    monkeypatch.setenv("PYTHONHOME", "/attacker")
    monkeypatch.setenv("LD_PRELOAD", "/attacker.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/attacker")
    environment = post.isolated_subprocess_environment(
        physical_gpu_uuid="GPU-reviewed"
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-reviewed"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        assert key not in environment


def test_inherited_cuda_remapping_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(post.PostCollectionV3Error, match="remapping is forbidden"):
        post.reject_inherited_cuda_remapping()


def test_gpu_stage_exports_exact_uuid_as_single_cuda_zero_namespace(
    tmp_path: Path,
) -> None:
    physical = {
        "gpu_index": 0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_uuid": "GPU-test-uuid",
        "checks": 2,
        "audit_sha256": "a" * 64,
    }
    result = post.run_bound_stage(
        name="gpu_env_smoke",
        command=[
            sys.executable,
            "-c",
            "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))",
        ],
        stage_root=tmp_path / "stage",
        gpu_index=0,
        physical_gpu=physical,
        poll_interval=0.01,
        pre_popen_guard=lambda: None,
    )
    assert result["physical_gpu"] == physical
    assert (tmp_path / "stage" / "run.log").read_text().strip() == "GPU-test-uuid"


def test_wait_for_upstreams_stops_on_authenticated_development300_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    r9b = tmp_path / "r9b"
    development = tmp_path / "development300"
    source.mkdir()
    r9b.mkdir()
    failure = _signed(
        {
            "format": post.DEVELOPMENT300_TERMINAL_FORMAT,
            "status": post.DEVELOPMENT300_FAILURE,
            "retry_or_resume_authorized": False,
            "formal_label_opened_by_runner_or_watcher": False,
            "evaluation400_commands_executed": 0,
        },
        "terminal_receipt_sha256",
    )
    _write_json(development / "_runner" / "failure_receipt.json", failure)
    plan = {
        "source_root": str(source),
        "r9b_root": str(r9b),
        "development300_collection_root": str(development),
        "development300_terminal": {
            "path": str(development / "_runner" / "final_receipt.json")
        },
    }
    with pytest.raises(post.PostCollectionV3Error, match="authenticated terminal failure"):
        post.wait_for_upstreams(
            plan, interval=0.0, heartbeat=lambda _status: None, sleep=lambda _delay: None
        )
    failure["evaluation400_commands_executed"] = False
    failure["terminal_receipt_sha256"] = post.canonical_sha256(
        {key: value for key, value in failure.items() if key != "terminal_receipt_sha256"}
    )
    path = development / "_runner" / "failure_receipt.json"
    path.chmod(0o644)
    _write_json(path, failure)
    with pytest.raises(post.PostCollectionV3Error, match="contract is invalid"):
        post.wait_for_upstreams(
            plan, interval=0.0, heartbeat=lambda _status: None, sleep=lambda _delay: None
        )


def test_formal_receipt_binds_exact_authority_and_rejects_bool_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_authority = tmp_path / "evaluator_authority.json"
    input_value = _signed({"status": "ok"}, "authority_sha256")
    _write_json(input_authority, input_value)
    calibration_authority = tmp_path / "calibration_authority.json"
    rank_contract_shas = [f"{index + 1:064x}" for index in range(5)]
    calibration_value = _signed(
        {
            "status": "ok",
            "members": [
                {"source_rank_score_contract_sha256": value}
                for value in rank_contract_shas
            ],
        },
        "input_authority_sha256",
    )
    _write_json(calibration_authority, calibration_value)
    monkeypatch.setattr(
        post.calibrator,
        "validate_input_authority",
        lambda _value: {
            "logical_sha256": calibration_value["input_authority_sha256"],
            "source_rank_numeric_contract": post.trainer.SOURCE_RANK_NUMERIC_CONTRACT,
        },
    )
    receipt_path = tmp_path / "formal_receipt.json"
    base = {
        "format": post.evaluator.RECEIPT_FORMAT,
        "status": post.evaluator.RECEIPT_STATUS,
        "input_authority_path": str(input_authority),
        "input_authority_file_sha256": post.file_sha256(input_authority),
        "input_authority_sha256": input_value["authority_sha256"],
        "target_validation_groups": 190,
        "target_validation_samples": 380,
        "target_validation_hdf5_files_opened": 190,
        "target_validation_opened_after_five_adapters_frozen": True,
        "calibration_input_authority_path": str(calibration_authority),
        "calibration_input_authority_file_sha256": post.file_sha256(calibration_authority),
        "calibration_input_authority_sha256": calibration_value["input_authority_sha256"],
        "evaluation400_membership_present": False,
        "evaluation400_hdf5_or_label_files_opened": 0,
        "fresh_or_confirmation_files_opened": 0,
        "performance_or_transfer_claim_authorized": False,
        "source_rank_numeric_contract": post.trainer.SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_score_contract_sha256s": rank_contract_shas,
    }
    _write_json(receipt_path, _signed(base, "receipt_sha256"))
    post.validate_formal190_receipt(
        receipt_path,
        expected_input_authority_path=input_authority,
        expected_input_authority_file_sha256=post.file_sha256(input_authority),
        expected_input_authority_sha256=input_value["authority_sha256"],
    )
    base["evaluation400_hdf5_or_label_files_opened"] = False
    receipt_path.chmod(0o644)
    _write_json(receipt_path, _signed(base, "receipt_sha256"))
    with pytest.raises(post.PostCollectionV3Error, match="evaluator receipt changed"):
        post.validate_formal190_receipt(
            receipt_path,
            expected_input_authority_path=input_authority,
            expected_input_authority_file_sha256=post.file_sha256(input_authority),
            expected_input_authority_sha256=input_value["authority_sha256"],
        )


def test_calibration_requires_exact_190_and_all_six_primary_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = tmp_path / "calibration.json"
    support = tmp_path / "support.json"
    ensemble = tmp_path / "ensemble.json"
    root_ranker = tmp_path / "root_ranker.json"
    receipt_path = tmp_path / "final_receipt.json"
    enabled = {
        "post_event": True,
        "next_event": True,
        "duration": True,
        "success": True,
        "recovery": True,
        "object_effect": True,
    }
    root_ranker_value = {"enabled_for_primary": True}
    calibration_value = {
        "validation_groups": 190,
        "test_hdf5_files_opened": 0,
        "all_six_heads_support_performance_uncertainty_gate_passed": True,
        "root_group_ranker_enabled_for_primary": True,
        "root_group_ranker": root_ranker_value,
    }
    support_value = {
        "heads": {
            **{name: {} for name in enabled if name != "recovery"},
            "recovery": {
                "all_member_recovery_heads_trained": True,
                "support_threshold_met": True,
                "performance_gate_passed": True,
                "uncertainty_gate_passed": True,
                "enabled_for_primary": True,
            },
        }
    }
    ensemble_value = {
        "head_enabled_for_primary": enabled,
        "all_six_heads_support_performance_uncertainty_gate_passed": True,
        "root_group_ranker_enabled_for_primary": True,
    }
    for path, value in (
        (calibration, calibration_value),
        (support, support_value),
        (ensemble, ensemble_value),
        (root_ranker, root_ranker_value),
    ):
        _write_json(path, value)
    evaluator_receipt = {
        "calibration_input_authority_path": str(tmp_path / "authority.json"),
        "calibration_input_authority_file_sha256": "a" * 64,
        "calibration_input_authority_sha256": "b" * 64,
        "source_rank_numeric_contract": post.trainer.SOURCE_RANK_NUMERIC_CONTRACT,
    }
    member_authority = {
        "source_rank_numeric_contract": post.trainer.SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": index,
                "source_checkpoint_file_sha256": f"{index + 1:064x}",
                "source_rank_score_contract_sha256": f"{index + 11:064x}",
                "success_temperature": 1.0,
            }
            for index in range(5)
        ],
    }
    base = {
        "format": post.calibrator.RECEIPT_FORMAT,
        "status": post.calibrator.RECEIPT_STATUS,
        "input_authority_path": evaluator_receipt["calibration_input_authority_path"],
        "input_authority_file_sha256": evaluator_receipt["calibration_input_authority_file_sha256"],
        "input_authority_sha256": evaluator_receipt["calibration_input_authority_sha256"],
        "member_count": 5,
        "validation_only": True,
        "abstain_threshold_enabled": True,
        "root_group_ranker_enabled_for_primary": True,
        "test_artifacts_read": False,
        "test_hdf5_files_opened": 0,
        "fresh_paths_accepted": False,
        "confirmation_artifacts_read": False,
        "paired_development_outcomes_read": False,
        "performance_or_transfer_claim_authorized": False,
        "source_rank_numeric_contract": post.trainer.SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": member_authority,
        "source_rank_member_authority_sha256": post.canonical_sha256(
            member_authority
        ),
        "artifacts_frozen_read_only": True,
        "calibration_path": str(calibration),
        "calibration_file_sha256": post.file_sha256(calibration),
        "head_support_path": str(support),
        "head_support_file_sha256": post.file_sha256(support),
        "ensemble_manifest_path": str(ensemble),
        "ensemble_manifest_file_sha256": post.file_sha256(ensemble),
        "root_group_ranker_path": str(root_ranker),
        "root_group_ranker_file_sha256": post.file_sha256(root_ranker),
    }
    _write_json(receipt_path, _signed(base, "receipt_sha256"))
    monkeypatch.setattr(post.identity_bridge, "validate_calibration", lambda _value: {})
    monkeypatch.setattr(post.identity_bridge, "validate_head_support", lambda _value: {})
    monkeypatch.setattr(
        post.identity_bridge,
        "validate_ensemble_manifest",
        lambda _value, **_kwargs: {},
    )
    monkeypatch.setattr(
        post.identity_bridge,
        "validate_calibration_receipt",
        lambda *_args, **_kwargs: None,
    )
    post.validate_calibration_result(receipt_path, evaluator_receipt=evaluator_receipt)
    calibration_value["validation_groups"] = False
    calibration.chmod(0o644)
    _write_json(calibration, calibration_value)
    base["calibration_file_sha256"] = post.file_sha256(calibration)
    receipt_path.chmod(0o644)
    _write_json(receipt_path, _signed(base, "receipt_sha256"))
    with pytest.raises(post.PostCollectionV3Error, match="exactly the bound formal190"):
        post.validate_calibration_result(receipt_path, evaluator_receipt=evaluator_receipt)


def test_detach_pgid_mismatch_never_signals_unknown_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "post"
    (root / "_watcher").mkdir(parents=True)
    plan_path = root / "_watcher" / "static_plan.json"
    _write_json(plan_path, {"synthetic": True})
    plan = {
        "plan_sha256": "a" * 64,
        "python": {"path": sys.executable},
        "implementations": {"launcher": {"path": str(ROOT / "launcher.py")}},
        "python_import_closure": {
            "launcher": {"path": str(ROOT / "launcher.py")}
        },
    }

    class FakeProcess:
        pid = 424242

    monkeypatch.setattr(
        post, "load_bound_plan", lambda _path, **_kwargs: (root, plan)
    )
    monkeypatch.setattr(post, "reject_inherited_cuda_remapping", lambda: None)
    monkeypatch.setattr(post, "verify_runtime_bindings", lambda _plan: None)
    opened = [os.open("/dev/null", os.O_RDONLY), os.open("/dev/null", os.O_RDONLY)]
    monkeypatch.setattr(
        post,
        "open_verified_runtime_binding_fds",
        lambda _plan: {
            str(plan["python"]["path"]): opened[0],
            str(plan["implementations"]["launcher"]["path"]): opened[1],
        },
    )
    monkeypatch.setattr(post.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(post.os, "getpgid", lambda _pid: 31337)
    monkeypatch.setattr(post, "_stop_direct_process", lambda _process: True)
    monkeypatch.setattr(
        post.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unknown PGID signaled")),
    )
    with pytest.raises(post.PostCollectionV3Error, match="not a new process group"):
        post.detach(plan_path)
    failure = json.loads((root / "_watcher" / "detach_failure.json").read_text())
    assert failure["direct_process_reaped"] is True
    assert failure["process_group_reaped"] is False
    assert failure["unknown_process_group_signaled"] is False
    assert failure["gpu_lock_acquired"] is False
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_cpu_unproven_execute_does_not_claim_or_retain_gpu_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "post"
    for name in ("_watcher", "materialization"):
        (root / name).mkdir(parents=True, exist_ok=True)
    record = {"path": str(tmp_path / "x"), "file_sha256": "b" * 64, "logical_sha256": "c" * 64}
    plan = {
        "plan_sha256": "a" * 64,
        "python": {"path": sys.executable},
        "implementations": {
            "materializer": {"path": str(tmp_path / "materializer.py")},
            "trainer": {"path": str(tmp_path / "trainer.py"), "file_sha256": "d" * 64},
        },
        "development300_collection_root": str(tmp_path / "development300"),
        "development300_terminal": record,
        "development300_runner_authority": record,
        "development300_target_preregistration": record,
        "development300_identity_authority": record,
    }
    states: list[dict[str, Any]] = []
    monkeypatch.setattr(
        post, "load_bound_plan", lambda _path, **_kwargs: (root, plan)
    )
    monkeypatch.setattr(post, "wait_for_upstreams", lambda *_args, **_kwargs: ({}, {}, {}))
    monkeypatch.setattr(
        post,
        "run_bound_stage",
        lambda **_kwargs: (_ for _ in ()).throw(post.UnprovenProcessGroup("cpu")),
    )
    monkeypatch.setattr(
        post,
        "update_state",
        lambda _root, _plan, status, **extra: states.append({"status": status, **extra}),
    )
    with pytest.raises(post.UnprovenProcessGroup):
        post.execute(
            root / "_watcher" / "static_plan.json",
            poll_interval=0.0,
            idle_interval=0.0,
            require_ppid1=False,
            hold_unproven=False,
        )
    assert states[-1]["gpu_lock_retained"] is False
    assert states[-1]["artifacts_frozen_read_only"] is False
    assert not (root / "failure_receipt.json").exists()


def test_terminal_artifact_closure_reopens_and_rejects_log_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "post"
    claim_root = tmp_path / ".etsf_schema6_formal190_global_claims_v1"
    claim_root.mkdir(mode=0o700)
    plan_base = {
        "formal190_claim_root": str(claim_root),
        "output_root": str(root),
    }
    plan = _signed(plan_base, "plan_sha256")

    def signed_file(path: Path, signature: str) -> None:
        _write_json(path, _signed({"role": path.name}, signature))

    signed_file(root / "_watcher" / "detached_worker_proof.json", "detach_proof_sha256")
    signed_file(root / "_watcher" / "gpu_idle_before_training.json", "audit_sha256")
    signed_file(root / "_watcher" / "gpu_idle_after_formal190.json", "audit_sha256")
    signed_file(root / "_watcher" / "gpu_lock_release.json", "release_sha256")
    _write_json(root / "_watcher" / "static_plan.json", plan)
    signed_file(
        root
        / "materialization"
        / "materializer_stage"
        / "result"
        / post.MATERIALIZER_OUTPUTS["receipt"],
        "receipt_sha256",
    )
    signed_file(root / "formal190" / "evaluator_input_authority.json", "authority_sha256")
    signed_file(
        root / "formal190" / "evaluator_stage" / "result" / "final_receipt.json",
        "receipt_sha256",
    )
    signed_file(
        root / "calibration" / "calibrator_stage" / "result" / "final_receipt.json",
        "receipt_sha256",
    )
    handoff_path = root / "handoff" / "evaluation400_identity_bridge_v2_handoff.json"
    signed_file(handoff_path, "handoff_sha256")
    for index in range(5):
        signed_file(
            root / "members" / f"member_{index}" / "final_receipt.json",
            "receipt_sha256",
        )
    claim_identity = "f" * 64
    claim_path = claim_root / f"formal190-{claim_identity}.claim.json"
    signed_file(claim_path, "claim_sha256")
    claim_value = json.loads(claim_path.read_text())
    claim_value["formal190_identity_sha256"] = claim_identity
    claim_value["claim_sha256"] = post.canonical_sha256(
        {key: value for key, value in claim_value.items() if key != "claim_sha256"}
    )
    claim_path.chmod(0o644)
    _write_json(claim_path, claim_value)
    stages = [
        "materialize_development300_v3",
        *[f"train_adapter_member_{index}" for index in range(5)],
        "evaluate_frozen_five_member_ensemble_on_formal190",
        "calibrate_six_head_formal190_ensemble",
    ]
    stage_results: dict[str, dict[str, Any]] = {}
    for stage in stages:
        stage_root = post._stage_root(root, stage)
        signed_file(stage_root / "launch.json", "launch_sha256")
        signed_file(stage_root / "lifecycle.json", "lifecycle_sha256")
        log = stage_root / "run.log"
        exit_path = stage_root / "run.exit"
        log.write_text("ok\n", encoding="utf-8")
        exit_path.write_text("0\n", encoding="ascii")
        log.chmod(0o444)
        exit_path.chmod(0o444)
        stage_results[stage] = {}
    claim_record = {
        "path": str(claim_path),
        "file_sha256": post.file_sha256(claim_path),
        "logical_sha256": claim_value["claim_sha256"],
        "formal190_identity_sha256": claim_identity,
        "consumed": True,
    }
    closure = post.build_artifact_closure(
        root=root,
        plan=plan,
        stage_results=stage_results,
        handoff_path=handoff_path,
        formal190_claim=claim_record,
    )
    closure_sha = post.canonical_sha256(closure)
    post.validate_artifact_closure(root, closure, closure_sha)
    changed_log = post._stage_root(root, stages[0]) / "run.log"
    changed_log.chmod(0o644)
    changed_log.write_text("mutated\n", encoding="utf-8")
    changed_log.chmod(0o444)
    with pytest.raises(post.PostCollectionV3Error, match="file SHA changed"):
        post.validate_artifact_closure(root, closure, closure_sha)
