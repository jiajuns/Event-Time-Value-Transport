from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from launch_multibody_lobo_autonomous import (  # noqa: E402
    ENSEMBLE_SEEDS,
    FORMAT,
    LOBO_FORMAT,
    LOBO_SPLIT_FORMAT,
    LOBO_TERMINAL_STATUS,
    TRAINING_STEPS,
    UPSTREAM_FORMAT,
    UPSTREAM_TERMINAL_STATUS,
    UPSTREAM_TRAINING_STATUS,
    _owned_gpu_lock_release_allowed,
    _unreaped_lobo_stage,
    bind_lobo_stage_command,
    build_final_receipt,
    build_lobo_command,
    canonical_sha256,
    detach,
    file_sha256,
    materialize_source_binding_receipt,
    prepare_execution,
    publish_frozen_terminal_receipt,
    release_owned_lock,
    resolve_new_path,
    run_subprocess_stage,
    static_preflight,
    validate_lobo_output,
    validate_lobo_split,
    validate_source_launch_plan,
    validate_source_binding_receipt,
    validate_source_terminal_receipt,
    wait_for_idle_4090,
    wait_for_source_completion,
)


INPUT_SHA = {
    "stage1_source_manifest": "1" * 64,
    "stage1_target_manifest": "2" * 64,
    "event_spec": "3" * 64,
    "openvla_schema5_manifest": "4" * 64,
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source_plan(root: Path) -> tuple[Path, dict]:
    plan = {
        "format": UPSTREAM_FORMAT,
        "status": "static_preflight_complete_waiting_no_manifest_or_hdf5_read",
        "output_root": str(root.resolve()),
        "gpu_index": 0,
        "device": "cuda",
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "manifest_read_during_static_preflight": False,
        "hdf5_opened_during_static_preflight": False,
        "nonresumable_output": True,
    }
    plan["static_plan_sha256"] = canonical_sha256(plan)
    path = root / "launch_plan.json"
    _write_json(path, plan)
    return path, plan


def _split(path: Path, body: str, input_sha: dict | None = None) -> dict:
    input_sha = INPUT_SHA if input_sha is None else input_sha
    lanes = {}
    for index, name in enumerate(
        (
            "source_train",
            "source_validation",
            "target_development",
            "target_unused_train",
            "sealed_test",
        )
    ):
        identities = [f"body{index}|policy|task|{index}"]
        lanes[name] = {
            "groups": 1,
            "identity_sha256": canonical_sha256(identities),
            "identities": identities,
            "bodies": ["body"],
            "policies": ["policy"],
        }
    value = {
        "format": LOBO_SPLIT_FORMAT,
        "held_out_body": body,
        "split_seed": 20260828,
        "split_inputs": dict(input_sha),
        "event_spec_sha256": input_sha["event_spec"],
        "split_unit": "body_policy_task_seed_logical_group",
        "labels_used_for_assignment": False,
        "checkpoint_selection_lane": "source_validation",
        "final_evaluation_lane": "target_development",
        "target_development_used_for_checkpoint_selection": False,
        "target_unused_train_payload_opened": 0,
        "sealed_test_group_hdf5_opened": 0,
        "lanes": lanes,
    }
    value["sha256"] = canonical_sha256(value)
    _write_json(path, value)
    return value


def _upstream_terminal(root: Path, plan: dict) -> None:
    ensemble_checkpoint = root / "counterfactual_training" / "counterfactual_ensemble.pt"
    ensemble_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ensemble_checkpoint.write_bytes(b"synthetic-source-native-ensemble")
    training = {
        "status": UPSTREAM_TRAINING_STATUS,
        "ensemble_checkpoint": str(ensemble_checkpoint.resolve()),
        "ensemble_checkpoint_sha256": file_sha256(ensemble_checkpoint),
        "policy_feature_action_bridge_sha256": "5" * 64,
        "member_count": 5,
        "member_seeds": list(ENSEMBLE_SEEDS),
        "member_training_steps_verified": [TRAINING_STEPS] * 5,
        "target_data_read": False,
        "target_labels_read": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
    }
    receipt = {
        "format": UPSTREAM_FORMAT,
        "status": UPSTREAM_TERMINAL_STATUS,
        "static_plan_sha256": plan["static_plan_sha256"],
        "execution_plan_sha256": "6" * 64,
        "snapshot_sha256": "7" * 64,
        "artifact_inventory_sha256": "8" * 64,
        "initialized_checkpoint_sha256": "9" * 64,
        "training_audit": training,
        "target_data_read": False,
        "target_labels_read": False,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
        "artifacts_frozen_read_only": True,
    }
    state = {
        "status": UPSTREAM_TERMINAL_STATUS,
        "static_plan_sha256": plan["static_plan_sha256"],
        "target_data_read": False,
        "target_labels_read": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
    }
    _write_json(root / "final_receipt.json", receipt)
    _write_json(root / "launch_state.json", state)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _restore_tree(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def _materialized_source_binding(
    tmp_path: Path,
) -> tuple[Path, dict, dict, dict]:
    source_root = tmp_path / "source_training"
    source_root.mkdir()
    plan_path, source_plan = _source_plan(source_root)
    source = validate_source_launch_plan(
        plan_path,
        expected_file_sha256=file_sha256(plan_path),
        expected_root=source_root.resolve(),
        gpu_index=0,
    )
    watcher_plan = {"source63": source}
    _upstream_terminal(source_root, source_plan)
    terminal = validate_source_terminal_receipt(watcher_plan)
    watcher_root = tmp_path / "watcher_binding"
    watcher_root.mkdir()
    binding = materialize_source_binding_receipt(
        watcher_root, plan=watcher_plan, source_audit=terminal
    )
    return source_root, watcher_plan, terminal, binding


def test_frozen_splits_are_body_bound_disjoint_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "piper_split.json"
    _split(path, "piper")
    audit = validate_lobo_split(
        path,
        expected_file_sha256=file_sha256(path),
        held_out_body="piper",
        input_sha256=INPUT_SHA,
    )
    assert audit["payload_hdf5_opened"] == 0
    changed = json.loads(path.read_text())
    changed["lanes"]["sealed_test"] = changed["lanes"]["source_train"]
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
    _write_json(path, changed)
    with pytest.raises(RuntimeError, match="integrity"):
        validate_lobo_split(
            path,
            expected_file_sha256=file_sha256(path),
            held_out_body="piper",
            input_sha256=INPUT_SHA,
        )


def test_source_gate_requires_bound_plan_terminal_proof_and_read_only_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source_training"
    root.mkdir()
    plan_path, plan = _source_plan(root)
    audit = validate_source_launch_plan(
        plan_path,
        expected_file_sha256=file_sha256(plan_path),
        expected_root=root.resolve(),
        gpu_index=0,
    )
    watcher_plan = {"source63": audit}
    _upstream_terminal(root, plan)
    try:
        terminal = validate_source_terminal_receipt(watcher_plan)
        assert terminal["member_training_steps_verified"] == [3000] * 5
        assert terminal["test_hdf_label_datasets_opened"] == 0
        assert len(terminal["final_receipt_logical_sha256"]) == 64
        assert terminal["policy_feature_action_bridge_sha256"] == "5" * 64
        assert Path(terminal["ensemble_checkpoint"]).is_file()
        # Any writable terminal artifact makes the completion gate fail.
        (root / "final_receipt.json").chmod(0o644)
        with pytest.raises(RuntimeError, match="not frozen"):
            validate_source_terminal_receipt(watcher_plan)
    finally:
        _restore_tree(root)


def test_source_gate_requires_bridge_path_and_checkpoint_hash(tmp_path: Path) -> None:
    root = tmp_path / "source_training"
    root.mkdir()
    plan_path, plan = _source_plan(root)
    source = validate_source_launch_plan(
        plan_path,
        expected_file_sha256=file_sha256(plan_path),
        expected_root=root.resolve(),
        gpu_index=0,
    )
    watcher_plan = {"source63": source}
    _upstream_terminal(root, plan)
    try:
        _restore_tree(root)
        receipt_path = root / "final_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        del receipt["training_audit"]["policy_feature_action_bridge_sha256"]
        _write_json(receipt_path, receipt)
        for path in root.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)
        with pytest.raises(RuntimeError, match="incomplete"):
            validate_source_terminal_receipt(watcher_plan)
    finally:
        _restore_tree(root)


def test_source_binding_is_create_once_and_detects_checkpoint_tamper(
    tmp_path: Path,
) -> None:
    source_root, watcher_plan, terminal, binding = _materialized_source_binding(
        tmp_path
    )
    try:
        assert binding["lobo_checkpoints_rerank_authorized"] is False
        assert binding["deployment_rerank_checkpoint"]["path"] == terminal[
            "ensemble_checkpoint"
        ]
        assert binding["policy_feature_action_bridge_contract_sha256"] == "5" * 64
        assert validate_source_binding_receipt(
            Path(binding["path"]), plan=watcher_plan
        ) == binding
        with pytest.raises(FileExistsError):
            materialize_source_binding_receipt(
                Path(binding["path"]).parent,
                plan=watcher_plan,
                source_audit=terminal,
            )
        _restore_tree(source_root)
        Path(terminal["ensemble_checkpoint"]).write_bytes(b"tampered")
        for path in source_root.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        source_root.chmod(0o555)
        with pytest.raises(RuntimeError, match="checkpoint changed"):
            validate_source_binding_receipt(Path(binding["path"]))
    finally:
        _restore_tree(source_root)


def test_waiting_reads_no_hdf_and_upstream_failure_stops(tmp_path: Path) -> None:
    root = tmp_path / "source_training"
    root.mkdir()
    plan_path, plan = _source_plan(root)
    source = validate_source_launch_plan(
        plan_path,
        expected_file_sha256=file_sha256(plan_path),
        expected_root=root.resolve(),
        gpu_index=0,
    )
    watcher_plan = {"source63": source}
    state: dict = {}
    with pytest.raises(TimeoutError):
        wait_for_source_completion(
            watcher_plan,
            state=state,
            state_path=tmp_path / "state.json",
            poll_seconds=0.01,
            timeout_seconds=0,
            max_polls=1,
            sleep=lambda _: None,
        )
    assert state["source63_final_receipt_read"] is False
    assert state["watcher_hdf5_opened"] == 0
    assert state["test_hdf5_opened_by_watcher"] == 0
    _write_json(root / "failure_receipt.json", {"status": "failed_closed_source63_native_counterfactual_training_fresh_forbidden"})
    with pytest.raises(RuntimeError, match="will not start"):
        wait_for_source_completion(
            watcher_plan,
            state=state,
            state_path=tmp_path / "state.json",
            poll_seconds=0.01,
            timeout_seconds=0,
            sleep=lambda _: None,
        )


def test_commands_are_exact_sequential_3000_step_cuda_contract(tmp_path: Path) -> None:
    bindings = {
        "stage1_root": "/data/stage1",
        "stage1_source_manifest": "/data/source.json",
        "stage1_source_manifest_sha256": "1" * 64,
        "stage1_target_manifest": "/data/target.csv",
        "stage1_target_manifest_sha256": "2" * 64,
        "event_spec": "/data/event.json",
        "event_spec_sha256": "3" * 64,
        "openvla_schema5_manifest": "/data/schema5.json",
        "openvla_schema5_manifest_sha256": "4" * 64,
    }
    piper = build_lobo_command(
        python_bin=Path("/venv/python"),
        trainer=Path("/code/trainer.py"),
        held_out_body="piper",
        output=tmp_path / "piper",
        split=tmp_path / "piper.json",
        split_sha256="a" * 64,
        bindings=bindings,
    )
    ur5 = build_lobo_command(
        python_bin=Path("/venv/python"),
        trainer=Path("/code/trainer.py"),
        held_out_body="ur5-wsg",
        output=tmp_path / "ur5",
        split=tmp_path / "ur5.json",
        split_sha256="b" * 64,
        bindings=bindings,
    )
    assert [piper["stage"], ur5["stage"]] == ["train_lobo_piper", "train_lobo_ur5"]
    for command in (piper, ur5):
        argv = command["argv"]
        assert argv[argv.index("--steps") + 1] == "3000"
        assert argv[argv.index("--device") + 1] == "cuda"
        assert argv[argv.index("--ensemble-seeds") + 1 : argv.index("--steps")] == [
            str(seed) for seed in ENSEMBLE_SEEDS
        ]
        assert command["source_binding_required_after_upstream_terminal"] is True
        assert command["lobo_checkpoints_rerank_authorized"] is False
        assert command["test_group_hdf5_opened_by_watcher"] == 0


def test_gpu_idle_requires_multiple_consecutive_empty_compute_queries(tmp_path: Path) -> None:
    sequence = iter([[41], [], [42], [], []])
    state: dict = {}
    audit = wait_for_idle_4090(
        gpu_index=0,
        state=state,
        state_path=tmp_path / "state.json",
        poll_seconds=0.01,
        timeout_seconds=1,
        confirmations=2,
        name_fn=lambda _: "NVIDIA GeForce RTX 4090",
        pids_fn=lambda _: next(sequence),
        sleep=lambda _: None,
    )
    assert audit["checks"] == 5
    assert audit["consecutive_idle"] == 2
    assert audit["observations"][-1]["compute_pids"] == []


def test_each_subtask_materializes_log_exit_and_failure_receipt(tmp_path: Path) -> None:
    watcher = tmp_path / "watcher"
    watcher.mkdir()
    state: dict = {}
    state_path = watcher / "state.json"
    success = {
        "stage": "train_lobo_piper",
        "held_out_body": "piper",
        "argv": [sys.executable, "-c", "print('piper-ok')"],
        "argv_sha256": canonical_sha256(
            [sys.executable, "-c", "print('piper-ok')"]
        ),
        "output": str(tmp_path / "piper_output"),
    }
    result = run_subprocess_stage(
        success,
        watcher_root=watcher,
        environment=os.environ,
        state=state,
        state_path=state_path,
        gpu_index=0,
        poll_seconds=0.001,
        timeout_seconds=5,
        pids_fn=lambda _: [],
    )
    stage = watcher / "stages" / "train_lobo_piper"
    assert result["returncode"] == 0
    assert (stage / "run.exit").read_text() == "0\n"
    assert "piper-ok" in (stage / "run.log").read_text()

    failed = {
        "stage": "train_lobo_ur5",
        "held_out_body": "ur5-wsg",
        "argv": [sys.executable, "-c", "raise SystemExit(7)"],
        "argv_sha256": canonical_sha256(
            [sys.executable, "-c", "raise SystemExit(7)"]
        ),
        "output": str(tmp_path / "ur5_output"),
    }
    with pytest.raises(RuntimeError, match="exit 7"):
        run_subprocess_stage(
            failed,
            watcher_root=watcher,
            environment=os.environ,
            state=state,
            state_path=state_path,
            gpu_index=0,
            poll_seconds=0.001,
            timeout_seconds=5,
            pids_fn=lambda _: [],
        )
    failed_stage = watcher / "stages" / "train_lobo_ur5"
    assert (failed_stage / "run.exit").read_text() == "7\n"
    assert json.loads((failed_stage / "stage_receipt.json").read_text())[
        "status"
    ] == "failed_closed"


def test_stage_failure_reaps_spawned_child_process_group(tmp_path: Path) -> None:
    watcher = tmp_path / "watcher"
    watcher.mkdir()
    child_pid_path = tmp_path / "child.pid"
    child_program = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    argv = [sys.executable, "-c", child_program, str(child_pid_path)]
    command = {
        "stage": "train_lobo_piper",
        "held_out_body": "piper",
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "output": str(tmp_path / "piper_output"),
    }
    lifecycle: dict = {}

    def gpu_pids(_: int) -> list[int]:
        return [999999] if child_pid_path.exists() else []

    with pytest.raises(RuntimeError, match="foreign GPU compute process"):
        run_subprocess_stage(
            command,
            watcher_root=watcher,
            environment=os.environ,
            state={},
            state_path=watcher / "state.json",
            gpu_index=0,
            poll_seconds=0.005,
            timeout_seconds=5,
            pids_fn=gpu_pids,
            lifecycle=lifecycle,
        )
    child_pid = int(child_pid_path.read_text())
    assert lifecycle["popen_attempted"] is True
    assert lifecycle["popen_reached"] is True
    assert lifecycle["process_group_id"] == lifecycle["process_pid"]
    assert lifecycle["process_group_isolated"] is True
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_reaped"] is True
    assert not Path(f"/proc/{child_pid}").exists()
    receipt = json.loads(
        (watcher / "stages" / "train_lobo_piper" / "stage_receipt.json").read_text()
    )
    assert receipt["process_group_reaped"] is True


def test_popen_attempt_without_lifecycle_proof_retains_owned_gpu_lock(
    tmp_path: Path,
) -> None:
    watcher = tmp_path / "watcher"
    watcher.mkdir()
    argv = [sys.executable, "-c", "raise SystemExit(0)"]
    command = {
        "stage": "train_lobo_piper",
        "held_out_body": "piper",
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "output": str(tmp_path / "piper_output"),
    }
    lifecycle: dict = {}

    def failed_popen(*_args, **_kwargs):
        raise OSError("injected failure after Popen attempt boundary")

    with pytest.raises(OSError, match="injected failure"):
        run_subprocess_stage(
            command,
            watcher_root=watcher,
            environment=os.environ,
            state={},
            state_path=watcher / "state.json",
            gpu_index=0,
            poll_seconds=0.001,
            timeout_seconds=1,
            pids_fn=lambda _: [],
            popen_factory=failed_popen,
            lifecycle=lifecycle,
        )
    assert lifecycle["popen_attempted"] is True
    assert lifecycle["popen_reached"] is False
    assert _unreaped_lobo_stage([lifecycle]) == "train_lobo_piper"
    lock = tmp_path / "shared_gpu.lock"
    token = "owned-token"
    _write_json(lock, {"pid": os.getpid(), "token": token})
    if _owned_gpu_lock_release_allowed(
        gpu_lock_acquired=True, stage_lifecycles=[lifecycle]
    ):
        release_owned_lock(lock, token)
    assert lock.exists()


def test_success_terminal_is_published_only_after_tree_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "watcher"
    root.mkdir(mode=0o700)
    artifact = root / "launch_state.json"
    artifact.write_text("{}\n")
    receipt = {"artifacts_frozen_read_only": True}
    monkeypatch.setattr(
        "launch_multibody_lobo_autonomous.validate_lobo_success_terminal_receipt",
        lambda value: dict(value),
    )
    with pytest.raises(ValueError, match="name and exit code"):
        publish_frozen_terminal_receipt(
            root,
            terminal_name="final_receipt.json",
            receipt=receipt,
            exit_code=False,
        )
    with pytest.raises(ValueError, match="name and exit code"):
        publish_frozen_terminal_receipt(
            root,
            terminal_name="final_receipt.json",
            receipt=receipt,
            exit_code=1,
        )
    publish_frozen_terminal_receipt(
        root,
        terminal_name="final_receipt.json",
        receipt=receipt,
        exit_code=0,
    )
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    assert stat.S_IMODE((root / "run.exit").stat().st_mode) == 0o444
    assert stat.S_IMODE((root / "final_receipt.json").stat().st_mode) == 0o444
    assert (root / "run.exit").read_text() == "0\n"


def test_freeze_failure_removes_hidden_success_terminal_and_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "watcher"
    root.mkdir(mode=0o700)
    (root / "launch_state.json").write_text("{}\n")
    receipt = {"artifacts_frozen_read_only": True}
    monkeypatch.setattr(
        "launch_multibody_lobo_autonomous.validate_lobo_success_terminal_receipt",
        lambda value: dict(value),
    )

    def failed_freeze(*_args, **_kwargs) -> None:
        raise OSError("injected freeze failure")

    monkeypatch.setattr(
        "launch_multibody_lobo_autonomous.freeze_tree", failed_freeze
    )
    with pytest.raises(OSError, match="injected freeze failure"):
        publish_frozen_terminal_receipt(
            root,
            terminal_name="final_receipt.json",
            receipt=receipt,
            exit_code=0,
        )
    assert not (root / "final_receipt.json").exists()
    assert not (root / "run.exit").exists()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def _fake_lobo_output(root: Path, body: str, input_sha: dict) -> dict:
    metrics = {}
    for variant in ("source_body_clock", "body_agnostic"):
        variant_root = root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        members = []
        hashes = []
        for member, seed in enumerate(ENSEMBLE_SEEDS):
            checkpoint = variant_root / f"member_{member}.pt"
            checkpoint.write_bytes(f"{variant}-{member}".encode())
            digest = file_sha256(checkpoint)
            hashes.append(digest)
            members.append(
                {
                    "member": member,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                }
            )
        _write_json(
            variant_root / "source_selection_summary.json",
            {
                "format": LOBO_FORMAT,
                "variant": variant,
                "held_out_body": body,
                "checkpoint_selection_split": "source_validation_only",
                "target_development_opened": 0,
                "test_group_hdf5_opened": 0,
                "members": members,
            },
        )
        metrics[variant] = {"evaluated_checkpoint_sha256": hashes}
    summary = {
        "format": LOBO_FORMAT,
        "status": LOBO_TERMINAL_STATUS,
        "held_out_body": body,
        "estimand": "zero_target_label_leave_one_body_out_transfer",
        "target_development_opened_after_all_checkpoint_selection": True,
        "target_unused_train_payload_opened": 0,
        "sealed_test_evaluated": False,
        "test_group_hdf5_opened": 0,
        "target_metrics": metrics,
        "protocol": {
            "input_sha256": input_sha,
            "checkpoint_selection_split": "source_validation_only",
            "target_development_used_for_checkpoint_selection": False,
            "target_unused_train_payload_opened": 0,
            "test_group_hdf5_opened": 0,
            "frozen_split_plan": {"file_sha256": "a" * 64},
        },
    }
    _write_json(root / "lobo_training_summary.json", summary)
    return summary


def test_lobo_artifact_audit_binds_all_ten_selected_checkpoints_and_test_zero(
    tmp_path: Path,
) -> None:
    source_root, _, _, source_binding = _materialized_source_binding(tmp_path)
    root = tmp_path / "piper_output"
    root.mkdir()
    _fake_lobo_output(root, "piper", INPUT_SHA)
    base_command = {
        "stage": "train_lobo_piper",
        "output": str(root),
        "held_out_body": "piper",
        "split_file_sha256": "a" * 64,
        "argv": ["python", "trainer.py"],
        "argv_sha256": canonical_sha256(["python", "trainer.py"]),
        "source_binding_required_after_upstream_terminal": True,
        "lobo_checkpoints_rerank_authorized": False,
    }
    command = bind_lobo_stage_command(base_command, source_binding)
    try:
        with pytest.raises(RuntimeError, match="lacks"):
            validate_lobo_output(base_command, input_sha256=INPUT_SHA)
        audit = validate_lobo_output(command, input_sha256=INPUT_SHA)
        assert sum(len(value) for value in audit["checkpoint_sha256"].values()) == 10
        assert audit["test_group_hdf5_opened"] == 0
        assert audit["lobo_checkpoints_rerank_authorized"] is False
        assert audit["deployment_rerank_checkpoint"] == source_binding[
            "deployment_rerank_checkpoint"
        ]
        tampered = dict(command)
        tampered_contract = dict(command["source_binding_contract"])
        tampered_contract["policy_feature_action_bridge_contract_sha256"] = "f" * 64
        tampered["source_binding_contract"] = tampered_contract
        with pytest.raises(RuntimeError, match="changed"):
            validate_lobo_output(tampered, input_sha256=INPUT_SHA)
        summary = json.loads((root / "lobo_training_summary.json").read_text())
        summary["sealed_test_evaluated"] = True
        _write_json(root / "lobo_training_summary.json", summary)
        with pytest.raises(RuntimeError, match="frozen protocol"):
            validate_lobo_output(command, input_sha256=INPUT_SHA)
    finally:
        _restore_tree(source_root)


def test_final_receipt_requires_both_stage_bindings_and_preserves_native_deployment(
    tmp_path: Path,
) -> None:
    source_root, source_plan, terminal, source_binding = _materialized_source_binding(
        tmp_path
    )
    commands = []
    for stage, body in (
        ("train_lobo_piper", "piper"),
        ("train_lobo_ur5", "ur5-wsg"),
    ):
        argv = ["python", "trainer.py", body]
        commands.append(
            {
                "stage": stage,
                "held_out_body": body,
                "argv": argv,
                "argv_sha256": canonical_sha256(argv),
                "output": str(tmp_path / f"{body}_output"),
                "source_binding_required_after_upstream_terminal": True,
                "lobo_checkpoints_rerank_authorized": False,
            }
        )
    plan = {
        **source_plan,
        "commands": commands,
        "execution_order": [command["stage"] for command in commands],
        "preregistered_outputs": {
            "piper": str(tmp_path / "piper_output"),
            "ur5": str(tmp_path / "ur5_output"),
        },
        "static_plan_sha256": "a" * 64,
    }
    stage_results = {}
    stage_lifecycles = []
    for index, command in enumerate(commands, start=1):
        contract = bind_lobo_stage_command(command, source_binding)[
            "source_binding_contract"
        ]
        process_pid = 42000 + index
        artifact_audit = {
            "status": LOBO_TERMINAL_STATUS,
            "held_out_body": command["held_out_body"],
            "source_binding_contract": contract,
            "lobo_checkpoints_rerank_authorized": False,
            "deployment_rerank_checkpoint": source_binding[
                "deployment_rerank_checkpoint"
            ],
            "target_unused_train_payload_opened": 0,
            "test_group_hdf5_opened": 0,
            "test_labels_read_by_watcher": False,
        }
        stage_results[command["stage"]] = {
            "stage": command["stage"],
            "held_out_body": command["held_out_body"],
            "status": "complete",
            "returncode": 0,
            "source_binding_contract": contract,
            "lobo_checkpoints_rerank_authorized": False,
            "deployment_rerank_checkpoint": source_binding[
                "deployment_rerank_checkpoint"
            ],
            "pid": process_pid,
            "process_reaped": True,
            "process_group_id": process_pid,
            "process_group_isolated": True,
            "process_group_reaped": True,
            "artifact_audit": artifact_audit,
        }
        stage_lifecycles.append(
            {
                "stage": command["stage"],
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
    state = {
        "stage_results": stage_results,
        "stage_lifecycles": stage_lifecycles,
    }
    try:
        receipt = build_final_receipt(
            plan,
            state=state,
            source_audit=terminal,
            source_binding=source_binding,
        )
        assert receipt["lobo_checkpoints_rerank_authorized"] is False
        assert receipt["deployment_rerank_authority"] == "native_source_ensemble_only"
        assert receipt["deployment_rerank_checkpoint"] == source_binding[
            "deployment_rerank_checkpoint"
        ]
        assert receipt["source_final_receipt_file_sha256"] == terminal[
            "final_receipt_sha256"
        ]
        assert receipt["source_final_receipt_logical_sha256"] == terminal[
            "final_receipt_logical_sha256"
        ]
        assert receipt["receipt_sha256"] == canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        stage_lifecycles[0]["process_group_reaped"] = False
        with pytest.raises(RuntimeError, match="lifecycle proof"):
            build_final_receipt(
                plan,
                state=state,
                source_audit=terminal,
                source_binding=source_binding,
            )
        stage_lifecycles[0]["process_group_reaped"] = True
        stage_results["train_lobo_piper"]["status"] = "failed_closed"
        stage_results["train_lobo_piper"]["returncode"] = 7
        with pytest.raises(RuntimeError, match="lifecycle proof"):
            build_final_receipt(
                plan,
                state=state,
                source_audit=terminal,
                source_binding=source_binding,
            )
        stage_results["train_lobo_piper"]["status"] = "complete"
        stage_results["train_lobo_piper"]["returncode"] = 0
        stage_results["train_lobo_piper"]["returncode"] = False
        stage_lifecycles[0]["returncode"] = False
        with pytest.raises(RuntimeError, match="lifecycle proof"):
            build_final_receipt(
                plan,
                state=state,
                source_audit=terminal,
                source_binding=source_binding,
            )
        stage_results["train_lobo_piper"]["returncode"] = 0
        stage_lifecycles[0]["returncode"] = 0
        piper_pid = stage_results["train_lobo_piper"]["pid"]
        stage_results["train_lobo_piper"]["process_group_id"] = float(piper_pid)
        stage_lifecycles[0]["process_group_id"] = float(piper_pid)
        with pytest.raises(RuntimeError, match="lifecycle proof"):
            build_final_receipt(
                plan,
                state=state,
                source_audit=terminal,
                source_binding=source_binding,
            )
        stage_results["train_lobo_piper"]["process_group_id"] = piper_pid
        stage_lifecycles[0]["process_group_id"] = piper_pid
        changed_terminal = dict(terminal)
        changed_terminal["policy_feature_action_bridge_sha256"] = "f" * 64
        with pytest.raises(RuntimeError, match="terminal audit changed"):
            build_final_receipt(
                plan,
                state=state,
                source_audit=changed_terminal,
                source_binding=source_binding,
            )
        del stage_results["train_lobo_ur5"]["artifact_audit"]
        with pytest.raises(RuntimeError, match="preserve"):
            build_final_receipt(
                plan,
                state=state,
                source_audit=terminal,
                source_binding=source_binding,
            )
    finally:
        _restore_tree(source_root)


def test_preregistration_is_create_once_and_claims_outputs_before_launch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "watcher"
    piper = tmp_path / "piper"
    ur5 = tmp_path / "ur5"
    plan = {
        "format": FORMAT,
        "output_root": str(output),
        "preregistered_outputs": {"piper": str(piper), "ur5": str(ur5)},
    }
    plan["static_plan_sha256"] = canonical_sha256(plan)
    prepared, token = prepare_execution(plan)
    assert prepared == output
    assert (output / "launch_plan.json").is_file()
    assert (output / "launch.lock").is_file()
    assert len(token) == 64
    with pytest.raises(FileExistsError):
        prepare_execution(plan)


def test_sensitive_and_existing_output_names_are_rejected(tmp_path: Path) -> None:
    sensitive = tmp_path / "fresh_lane"
    sensitive.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        resolve_new_path(sensitive / "output", role="test output")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        resolve_new_path(existing, role="test output")


def _full_args(tmp_path: Path) -> argparse.Namespace:
    source_root = tmp_path / "source_training"
    source_root.mkdir()
    source_plan, _ = _source_plan(source_root)
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    source_manifest = stage1 / "source_manifest.json"
    target_manifest = stage1 / "target_manifest.csv"
    event_spec = tmp_path / "event_spec.json"
    schema5_manifest = tmp_path / "schema5_manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    target_manifest.write_text("header\n", encoding="utf-8")
    event_spec.write_text("{}\n", encoding="utf-8")
    schema5_manifest.write_text("{}\n", encoding="utf-8")
    hashes = {
        "stage1_source_manifest": file_sha256(source_manifest),
        "stage1_target_manifest": file_sha256(target_manifest),
        "event_spec": file_sha256(event_spec),
        "openvla_schema5_manifest": file_sha256(schema5_manifest),
    }
    piper_split = tmp_path / "piper_split.json"
    ur5_split = tmp_path / "ur5_split.json"
    _split(piper_split, "piper", hashes)
    _split(ur5_split, "ur5-wsg", hashes)
    return argparse.Namespace(
        code_root=ROOT,
        trainer=None,
        source_training_root=source_root,
        source_launch_plan=source_plan,
        source_launch_plan_sha256=file_sha256(source_plan),
        stage1_root=stage1,
        stage1_source_manifest=source_manifest,
        stage1_source_manifest_sha256=hashes["stage1_source_manifest"],
        stage1_target_manifest=target_manifest,
        stage1_target_manifest_sha256=hashes["stage1_target_manifest"],
        event_spec=event_spec,
        event_spec_sha256=hashes["event_spec"],
        openvla_schema5_manifest=schema5_manifest,
        openvla_schema5_manifest_sha256=hashes["openvla_schema5_manifest"],
        piper_split=piper_split,
        piper_split_sha256=file_sha256(piper_split),
        ur5_split=ur5_split,
        ur5_split_sha256=file_sha256(ur5_split),
        output=tmp_path / "watcher_output",
        piper_output=tmp_path / "piper_output",
        ur5_output=tmp_path / "ur5_output",
        python_bin=Path(sys.executable),
        gpu_index=0,
        gpu_lock=tmp_path / "shared_gpu.lock",
        poll_seconds=0.01,
        source_timeout_seconds=0.0,
        gpu_timeout_seconds=1.0,
        stage_timeout_seconds=0.0,
        idle_confirmations=2,
        omp_threads=2,
        detach_receipt=tmp_path / "detach_receipt.json",
        detach_log=tmp_path / "daemon.log",
    )


def test_static_preflight_hash_binds_inputs_and_never_opens_hdf5(tmp_path: Path) -> None:
    args = _full_args(tmp_path)
    plan = static_preflight(args)
    assert plan["execution_order"] == ["train_lobo_piper", "train_lobo_ur5"]
    assert plan["watcher_hdf5_imported"] is False
    assert plan["watcher_hdf5_opened"] == 0
    assert plan["test_hdf5_opened_by_watcher"] == 0
    assert plan["output_paths_absent_at_preregistration"] is True
    assert plan["source_binding_materialized_create_once_after_authenticated_terminal"] is True
    assert plan["lobo_checkpoints_rerank_authorized"] is False
    assert plan["deployment_rerank_authority"] == "native_source_ensemble_only"
    assert plan["splits"]["piper"]["held_out_body"] == "piper"
    assert plan["splits"]["ur5"]["held_out_body"] == "ur5-wsg"


def test_detach_preregisters_before_new_session_process_and_emits_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _full_args(tmp_path)
    captured: dict = {}

    class FakeProcess:
        pid = 424242

    def fake_popen(argv, **kwargs):
        # The create-once output, immutable plan and lock exist before spawn.
        assert args.output.is_dir()
        assert (args.output / "launch_plan.json").is_file()
        assert (args.output / "launch.lock").is_file()
        assert not args.piper_output.exists()
        assert not args.ur5_output.exists()
        captured.update({"argv": argv, **kwargs})
        return FakeProcess()

    monkeypatch.setattr("launch_multibody_lobo_autonomous.subprocess.Popen", fake_popen)
    receipt = detach(args)
    assert receipt["pid"] == 424242
    assert receipt["outputs_preregistered_before_process_start"] is True
    assert receipt["new_os_session"] is True
    assert captured["start_new_session"] is True
    assert json.loads(args.detach_receipt.read_text())["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]
    with pytest.raises(FileExistsError):
        detach(args)
