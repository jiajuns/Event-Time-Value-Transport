from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from launch_smolvla_schema5_source63_native_training import (  # noqa: E402
    EXPECTED_GROUPS,
    FAILURE_STATUS,
    FORMAT,
    TERMINAL_STATUS,
    TRAINING_SEEDS,
    TRAINING_STEPS,
    _owned_gpu_lock_release_allowed,
    _process_group_exists,
    _proc_start_ticks,
    acquire_lock,
    audit_training_start_guard,
    audit_group_files,
    build_stage_commands,
    canonical_environment_contract,
    canonical_python_environment,
    canonical_sha256,
    detach,
    execute,
    external_suite_parent_alive,
    external_suite_parent_guard,
    file_sha256,
    freeze_source_snapshot,
    read_collector_exit_once,
    read_external_process_identity,
    publish_frozen_terminal_receipt,
    release_owned_lock,
    sign_canonical_receipt,
    read_frozen_split,
    resolve_new_path,
    run_subprocess_stage,
    assert_canonical_python_environment,
    assert_runtime_matches,
    static_preflight,
    tree_freeze_contract,
    validate_collector_metadata,
    validate_published_source63_terminal_receipt,
    validate_source63_terminal_receipt,
    wait_for_collector_exit,
    wait_for_idle_4090,
)
from run_etsf_bound_python_stage import run_bound_target  # noqa: E402


SOURCE_SPLIT = ROOT / "configs" / "smolvla_schema5_native_source63_split_r7.json"
MODELING_SHA = "0" * 64
BRIDGE_SHA = "1" * 64


def _fake_runtime(python_bin: Path = Path(sys.executable)) -> dict:
    module = Path(__file__).resolve()
    value = {
        "format": "etsf_isolated_python_torch_runtime_v1",
        "isolated": True,
        "python_executable": str(python_bin.resolve()),
        "python_version": "test-python",
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "torch_version": "test-torch",
        "torch_cuda_version": "test-cuda",
        "torch_module": {"path": str(module), "sha256": file_sha256(module)},
        "torch_c_module": {"path": str(module), "sha256": file_sha256(module)},
    }
    value["runtime_contract_sha256"] = canonical_sha256(value)
    return value


def _patch_runtime_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.probe_python_runtime",
        lambda python_bin, _environment: _fake_runtime(Path(python_bin)),
    )
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.gpu_audit",
        lambda index: {
            "gpu_index": index,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-test-4090",
            "compute_pids": [],
        },
    )


def _split() -> dict:
    return read_frozen_split(SOURCE_SPLIT)


def _event_spec(tmp_path: Path) -> Path:
    path = tmp_path / "event_spec.json"
    path.write_text('{"calibration":{"move_can_pot":{}}}\n', encoding="utf-8")
    return path


def _collector(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Path, dict]:
    root = tmp_path / "collector"
    groups = root / "groups"
    groups.mkdir(parents=True)
    split = _split()
    event_spec = _event_spec(tmp_path)
    rows = []
    for index, seed in enumerate(split["all_requested_seeds"]):
        name = f"group_{index:03d}_seed_{seed}.hdf5"
        (groups / name).write_bytes(f"opaque-hdf5-{index}".encode())
        rows.append(
            {
                "index": index,
                "path": name,
                "status": "collected",
                "seed": seed,
                "resolved_seed": seed + 1000,
                # The watcher must not read or branch on these outcome fields.
                "success": [bool(index % 2), False, True, False],
                "steps": [1, 2, 3, 4],
                "query_transitions": 4,
            }
        )
    manifest = {
        "schema_version": 5,
        "status": "complete" if complete else "collecting",
        "task": "move_can_pot",
        "body": "aloha-agilex",
        "policy": "smolvla",
        "requested_seeds": split["all_requested_seeds"],
        "resolved_seeds": [row["resolved_seed"] for row in rows],
        "event_spec_sha256": file_sha256(event_spec),
        "shared_state_modeling_sha256": MODELING_SHA,
        "shared_state_bridge_sha256": BRIDGE_SHA,
        "completed": EXPECTED_GROUPS if complete else 0,
        "candidate_count": 4,
        "hidden_dim": 960,
        "action_dim": 14,
        "action_chunk": 50,
        "model_path": "/models/smolvla",
        "checkpoint": "/models/smolvla",
        "vlm_metadata_path": "/models/smolvla-vlm",
        "modeling_source": "/code/modeling_smolvla.py",
        "bridge_source": "/code/vlm_bridge.py",
        "event_spec": "/protocol/event_spec.json",
        "groups": rows,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root, event_spec, manifest


def _metadata(root: Path, event_spec: Path) -> dict:
    return validate_collector_metadata(
        root / "manifest.json",
        split=_split(),
        event_spec_sha256=file_sha256(event_spec),
    )


def _validator_calls():
    calls: list[int] = []

    def validate(path: Path, seed: int, *_args, **_kwargs):
        calls.append(seed)
        return {"seed": seed, "resolved_seed": seed + 1000}

    return calls, validate


def test_waiting_for_exit_zero_never_reads_manifest_or_hdf5(tmp_path: Path) -> None:
    root, _, _ = _collector(tmp_path, complete=False)
    # Make both downstream artifacts invalid traps.  Waiting must still touch
    # only the absent run.exit and its own state file.
    (root / "manifest.json").write_text("not-json", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state: dict = {}
    with pytest.raises(TimeoutError):
        wait_for_collector_exit(
            root / "run.exit",
            state=state,
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=0,
            max_polls=1,
            sleep=lambda _: None,
        )
    assert state["manifest_read"] is False
    assert state["hdf5_opened"] is False
    assert json.loads(state_path.read_text())["hdf5_opened"] is False

    (root / "run.exit").write_text("0\n", encoding="ascii")
    audit = wait_for_collector_exit(
        root / "run.exit",
        state=state,
        state_path=state_path,
        poll_seconds=0.01,
        timeout_seconds=1,
        sleep=lambda _: None,
    )
    assert audit["exit_code"] == 0
    assert audit["manifest_read_before_exit_zero"] is False
    assert audit["hdf5_opened_before_exit_zero"] is False


def test_nonzero_or_malformed_collector_exit_fails_closed(tmp_path: Path) -> None:
    exit_path = tmp_path / "run.exit"
    exit_path.write_text("7\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="unsuccessfully"):
        read_collector_exit_once(exit_path)
    exit_path.write_text("success\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="exactly one integer"):
        read_collector_exit_once(exit_path)


def _external_parent_identity(
    *,
    pid: int = 1830377,
    start_ticks: int = 987654,
    script: str = "/home/user/openvla-repro/run_all_full.sh",
) -> dict:
    cmdline = b"bash\0" + script.encode() + b"\0"
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "cmdline_sha256": __import__("hashlib").sha256(cmdline).hexdigest(),
        "cmdline_arguments": ["bash", script],
    }


def _external_parent_guard(identity: dict | None = None) -> dict:
    identity = identity or _external_parent_identity()
    guard = {
        "format": "etsf_external_suite_parent_identity_guard_v1",
        "enabled": True,
        "pid": identity["pid"],
        "start_ticks": identity["start_ticks"],
        "boot_id": identity["boot_id"],
        "cmdline_sha256": identity["cmdline_sha256"],
        "script_path": identity["cmdline_arguments"][1],
        "script_sha256": "a" * 64,
        "identity_policy": "linux_boot_id_proc_starttime_raw_cmdline_and_script_sha256_v2",
        "pid_reuse_policy": "fail_closed",
        "idle_confirmations_required_after_exit": 2,
    }
    guard["guard_contract_sha256"] = canonical_sha256(guard)
    return guard


def test_external_parent_proc_identity_uses_start_ticks_and_raw_cmdline(
    tmp_path: Path,
) -> None:
    pid = 1830377
    process = tmp_path / str(pid)
    process.mkdir()
    boot = tmp_path / "sys" / "kernel" / "random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text(
        "11111111-2222-3333-4444-555555555555\n", encoding="ascii"
    )
    tail = ["S", *(["0"] * 18), "987654", "0"]
    (process / "stat").write_text(
        f"{pid} (bash suite parent) {' '.join(tail)}\n", encoding="ascii"
    )
    cmdline = b"bash\0/home/user/openvla-repro/run_all_full.sh\0"
    (process / "cmdline").write_bytes(cmdline)
    identity = read_external_process_identity(pid, proc_root=tmp_path)
    assert identity == _external_parent_identity()


def test_proc_identity_ignores_mutable_stat_fields_but_rejects_starttime_change() -> None:
    pid = 1830377
    fields_a = ["S", *(["0"] * 18), "987654", "1", "2"]
    fields_b = ["R", *(["9"] * 18), "987654", "8", "7"]
    fields_reused = ["R", *(["9"] * 18), "987655", "8", "7"]
    prefix = f"{pid} (suite parent) "
    assert _proc_start_ticks(pid, (prefix + " ".join(fields_a)).encode()) == 987654
    assert _proc_start_ticks(pid, (prefix + " ".join(fields_b)).encode()) == 987654
    assert _proc_start_ticks(pid, (prefix + " ".join(fields_reused)).encode()) == 987655


def test_partial_proc_or_missing_boot_id_never_counts_as_parent_absence(
    tmp_path: Path,
) -> None:
    pid = 1830377
    process = tmp_path / str(pid)
    process.mkdir()
    tail = ["S", *(["0"] * 18), "987654", "0"]
    (process / "stat").write_text(
        f"{pid} (suite parent) {' '.join(tail)}\n", encoding="ascii"
    )
    boot = tmp_path / "sys" / "kernel" / "random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text(
        "11111111-2222-3333-4444-555555555555\n", encoding="ascii"
    )
    with pytest.raises(RuntimeError, match="partial while PID still exists"):
        read_external_process_identity(pid, proc_root=tmp_path)

    (process / "stat").unlink()
    assert read_external_process_identity(pid, proc_root=tmp_path) is None
    (boot / "boot_id").unlink()
    with pytest.raises(RuntimeError, match="boot ID is unavailable"):
        read_external_process_identity(pid, proc_root=tmp_path)


def test_external_parent_guard_requires_complete_identity_and_rejects_pid_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "run_all_full.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    identity = _external_parent_identity(script=str(script))
    args = argparse.Namespace(
        external_suite_parent_pid=identity["pid"],
        external_suite_parent_start_ticks=identity["start_ticks"],
        external_suite_parent_boot_id=identity["boot_id"],
        external_suite_parent_cmdline_sha256=identity["cmdline_sha256"],
        external_suite_parent_script=script,
        external_suite_parent_script_sha256=file_sha256(script),
    )
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.read_external_process_identity",
        lambda _pid: identity,
    )
    guard = external_suite_parent_guard(args)
    assert external_suite_parent_alive(guard, identity_reader=lambda _pid: identity)

    reused = {**identity, "start_ticks": identity["start_ticks"] + 1}
    with pytest.raises(RuntimeError, match="reused"):
        external_suite_parent_alive(guard, identity_reader=lambda _pid: reused)

    args.external_suite_parent_cmdline_sha256 = None
    with pytest.raises(ValueError, match="requires PID"):
        external_suite_parent_guard(args)

    args.external_suite_parent_cmdline_sha256 = identity["cmdline_sha256"]
    args.external_suite_parent_script_sha256 = "f" * 64
    with pytest.raises(RuntimeError, match="script SHA256 changed"):
        external_suite_parent_guard(args)

    args.external_suite_parent_script_sha256 = file_sha256(script)
    monkeypatch.chdir(tmp_path)
    args.external_suite_parent_script = Path(script.name)
    with pytest.raises(ValueError, match="must be absolute"):
        external_suite_parent_guard(args)


def test_external_parent_script_must_be_one_exact_cmdline_token() -> None:
    identity = _external_parent_identity()
    guard = _external_parent_guard(identity)
    duplicate_raw = (
        b"bash\0/home/user/openvla-repro/run_all_full.sh\0"
        b"/home/user/openvla-repro/run_all_full.sh\0"
    )
    duplicate = {
        **identity,
        "cmdline_sha256": __import__("hashlib").sha256(duplicate_raw).hexdigest(),
        "cmdline_arguments": [
            "bash",
            "/home/user/openvla-repro/run_all_full.sh",
            "/home/user/openvla-repro/run_all_full.sh",
        ],
    }
    guard["cmdline_sha256"] = duplicate["cmdline_sha256"]
    guard.pop("guard_contract_sha256")
    guard["guard_contract_sha256"] = canonical_sha256(guard)
    with pytest.raises(RuntimeError, match="frozen identity changed"):
        external_suite_parent_alive(guard, identity_reader=lambda _pid: duplicate)


def test_absent_external_parent_cannot_sign_initial_enabled_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "run_all_full.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    identity = _external_parent_identity(script=str(script))
    args = argparse.Namespace(
        external_suite_parent_pid=identity["pid"],
        external_suite_parent_start_ticks=identity["start_ticks"],
        external_suite_parent_boot_id=identity["boot_id"],
        external_suite_parent_cmdline_sha256=identity["cmdline_sha256"],
        external_suite_parent_script=script,
        external_suite_parent_script_sha256=file_sha256(script),
    )
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.read_external_process_identity",
        lambda _pid: None,
    )
    with pytest.raises(RuntimeError, match="absent during initial preflight"):
        external_suite_parent_guard(args)


def test_parent_exit_then_two_consecutive_idle_audits_are_required(
    tmp_path: Path,
) -> None:
    identity = _external_parent_identity()
    observations = iter([identity, identity, None, None, None, None])
    sleeps: list[float] = []
    state: dict = {}
    audit = wait_for_idle_4090(
        gpu_index=0,
        state=state,
        state_path=tmp_path / "state.json",
        poll_seconds=30.0,
        timeout_seconds=0.0,
        external_parent_guard=_external_parent_guard(identity),
        expected_gpu_identity={
            "gpu_index": 0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-test-4090",
        },
        identity_reader=lambda _pid: next(observations),
        gpu_audit_reader=lambda index: {
            "gpu_index": index,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-test-4090",
            "compute_pids": [],
        },
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
        wall_time=lambda: 1.0,
    )
    assert audit["checks"] == 3
    assert audit["external_suite_parent_alive"] is False
    assert audit["idle_confirmations"] == 2
    assert audit["gpu_idle_release_audit_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in audit.items()
            if key != "gpu_idle_release_audit_sha256"
        }
    )
    assert sleeps == [30.0, 30.0]
    assert state["status"] == "waiting_for_two_consecutive_exclusive_idle_rtx4090_audits"


def test_busy_sample_resets_idle_streak_until_final_two_empty_samples(
    tmp_path: Path,
) -> None:
    gpu_samples = iter([[77], [], [88], [], []])
    audit = wait_for_idle_4090(
        gpu_index=0,
        state={},
        state_path=tmp_path / "state.json",
        poll_seconds=1.0,
        timeout_seconds=0.0,
        expected_gpu_identity={
            "gpu_index": 0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-test-4090",
        },
        gpu_audit_reader=lambda index: {
            "gpu_index": index,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-test-4090",
            "compute_pids": next(gpu_samples),
        },
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
        wall_time=lambda: 1.0,
    )
    assert audit["checks"] == 5
    assert audit["idle_confirmations"] == 2
    assert len(audit["valid_idle_observations"]) == 2


def test_parent_reappearance_after_first_absence_fails_closed(tmp_path: Path) -> None:
    identity = _external_parent_identity()
    observations = iter([identity, None, identity])
    with pytest.raises(RuntimeError, match="reappeared"):
        wait_for_idle_4090(
            gpu_index=0,
            state={},
            state_path=tmp_path / "state.json",
            poll_seconds=1.0,
            timeout_seconds=0.0,
            external_parent_guard=_external_parent_guard(identity),
            expected_gpu_identity={
                "gpu_index": 0,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_uuid": "GPU-test-4090",
            },
            identity_reader=lambda _pid: next(observations),
            gpu_audit_reader=lambda index: {
                "gpu_index": index,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_uuid": "GPU-test-4090",
                "compute_pids": [],
            },
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
            wall_time=lambda: 1.0,
        )


def test_gpu_uuid_change_fails_closed_before_idle_count(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="identity changed"):
        wait_for_idle_4090(
            gpu_index=0,
            state={},
            state_path=tmp_path / "state.json",
            poll_seconds=1.0,
            timeout_seconds=0.0,
            expected_gpu_identity={
                "gpu_index": 0,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_uuid": "GPU-frozen",
            },
            gpu_audit_reader=lambda index: {
                "gpu_index": index,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_uuid": "GPU-replaced",
                "compute_pids": [],
            },
            sleep=lambda _seconds: None,
        )


def test_detached_static_plan_mismatch_fails_before_output_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.static_preflight",
        lambda _args: {"static_plan_sha256": "1" * 64},
    )
    args = argparse.Namespace(
        expected_static_plan_sha256="2" * 64,
        output=output,
    )
    with pytest.raises(RuntimeError, match="differs from preflight"):
        execute(args)
    assert not output.exists()


def test_gpu_lock_release_audit_requires_exact_owner_and_token(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    token = "1" * 64
    acquire_lock(lock, {"format": "test", "pid": os.getpid(), "token": token})
    with pytest.raises(RuntimeError, match="ownership changed"):
        release_owned_lock(lock, "2" * 64, strict=True)
    assert lock.exists()
    audit = release_owned_lock(lock, token, strict=True)
    assert audit["released"] is True
    assert audit["status"] == "released_exact_owned_gpu_lock"
    assert not lock.exists()
    assert audit["release_audit_sha256"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "release_audit_sha256"}
    )


def test_training_start_and_terminal_receipts_bind_idle_and_release(tmp_path: Path) -> None:
    gpu_identity = {
        "gpu_index": 0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_uuid": "GPU-test-4090",
    }
    gpu_reader = lambda index: {
        **gpu_identity,
        "gpu_index": index,
        "compute_pids": [],
    }
    idle = wait_for_idle_4090(
        gpu_index=0,
        state={},
        state_path=tmp_path / "state.json",
        poll_seconds=1.0,
        timeout_seconds=0.0,
        expected_gpu_identity=gpu_identity,
        gpu_audit_reader=gpu_reader,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
        wall_time=lambda: 1.0,
    )
    disabled_guard = {
        "format": "etsf_external_suite_parent_identity_guard_v1",
        "enabled": False,
        "idle_confirmations_required_after_exit": 2,
    }
    start = audit_training_start_guard(
        {
            "gpu_index": 0,
            "gpu_identity": gpu_identity,
            "external_suite_parent_guard": disabled_guard,
        },
        idle,
        gpu_audit_reader=gpu_reader,
        wall_time=lambda: 2.0,
    )
    release = {
        "format": "etsf_owned_gpu_lock_release_audit_v1",
        "path": "/tmp/test-source63-gpu.lock",
        "pid": 12345,
        "token_sha256": "b" * 64,
        "status": "released_exact_owned_gpu_lock",
        "released": True,
        "observed_unix": 5.0,
        "observed_owner_pid": 12345,
        "observed_token_sha256": "b" * 64,
        "released_unix": 5.0,
    }
    release["release_audit_sha256"] = canonical_sha256(release)
    terminal_root = tmp_path / "terminal_output"
    (terminal_root / "logs").mkdir(parents=True)
    (terminal_root / "logs" / "stage.log").write_text("complete\n")
    freeze_contract = tree_freeze_contract(terminal_root)
    stage = sign_canonical_receipt(
        {
            "status": "complete",
            "returncode": 0,
            "pid": 23456,
            "process_group_id": 23456,
            "process_group_isolated": True,
            "process_reaped": True,
            "process_group_reaped": True,
            "popen_unix": 3.0,
            "finished_unix": 4.0,
            "pre_popen_guard_audit": start,
            "external_suite_parent_guard_contract_sha256": None,
            "gpu_idle_release_audit_sha256": idle[
                "gpu_idle_release_audit_sha256"
            ],
            "training_start_guard_audit_sha256": start[
                "training_start_guard_audit_sha256"
            ],
            "gpu_lock_release_audit_sha256": release["release_audit_sha256"],
        },
        field="stage_receipt_sha256",
    )
    success = sign_canonical_receipt(
        {
            "format": FORMAT,
            "status": TERMINAL_STATUS,
            "output_root": str(terminal_root),
            "terminal_receipt_name": "final_receipt.json",
            "launcher_pid": 12345,
            "gpu_lock_path": "/tmp/test-source63-gpu.lock",
            "gpu_lock_token_sha256": "b" * 64,
            "gpu_identity": gpu_identity,
            "external_suite_parent_guard": disabled_guard,
            "external_suite_parent_guard_contract_sha256": None,
            "gpu_idle_audit": idle,
            "gpu_idle_release_audit_sha256": idle[
                "gpu_idle_release_audit_sha256"
            ],
            "training_start_guard_audit": start,
            "training_start_guard_audit_sha256": start[
                "training_start_guard_audit_sha256"
            ],
            "gpu_lock_release_audit": release,
            "gpu_lock_release_audit_sha256": release["release_audit_sha256"],
            "training_stage_receipt": stage,
            "training_stage_receipt_sha256": stage["stage_receipt_sha256"],
            "artifacts_frozen_read_only": True,
            "artifact_freeze_contract": freeze_contract,
            "artifact_freeze_contract_sha256": freeze_contract[
                "tree_freeze_contract_sha256"
            ],
        }
    )
    assert validate_source63_terminal_receipt(success)["status"] == TERMINAL_STATUS

    tampered = json.loads(json.dumps(success))
    tampered["gpu_idle_audit"]["checks"] += 1
    tampered.pop("receipt_sha256")
    tampered = sign_canonical_receipt(tampered)
    with pytest.raises(RuntimeError, match="idle release audit SHA256"):
        validate_source63_terminal_receipt(tampered)

    failure = sign_canonical_receipt(
        {
            "format": FORMAT,
            "status": FAILURE_STATUS,
            "output_root": str(terminal_root),
            "terminal_receipt_name": "failure_receipt.json",
            "launcher_pid": 12345,
            "gpu_lock_path": "/tmp/test-source63-gpu.lock",
            "gpu_lock_token_sha256": "b" * 64,
            "gpu_identity": gpu_identity,
            "external_suite_parent_guard": disabled_guard,
            "external_suite_parent_guard_contract_sha256": None,
            "failure_phase": "after_gpu_idle_release_before_training",
            "gpu_idle_guard_started": True,
            "gpu_idle_release_reached": True,
            "training_pre_popen_guard_started": False,
            "training_pre_popen_guard_completed": False,
            "training_popen_attempted": False,
            "training_popen_reached": False,
            "training_process_pid": None,
            "training_process_reaped": False,
            "training_process_group_id": None,
            "training_process_group_isolated": False,
            "training_process_group_reaped": False,
            "training_process_group_binding_status": "not_reached",
            "training_stage_returned": False,
            "gpu_lock_acquired": True,
            "gpu_lock_released": True,
            "gpu_lock_released_before_failure": False,
            "unreaped_stage_process": None,
            "gpu_lock_retained_for_unreaped_stage_process": False,
            "partial_gpu_idle_guard_audit": idle,
            "gpu_lock_release_audit": release,
            "artifacts_frozen_read_only": True,
            "artifact_freeze_contract": freeze_contract,
            "artifact_freeze_contract_sha256": freeze_contract[
                "tree_freeze_contract_sha256"
            ],
        }
    )
    assert validate_source63_terminal_receipt(failure)["status"] == FAILURE_STATUS

    contradictory = json.loads(json.dumps(failure))
    contradictory["training_popen_reached"] = True
    contradictory.pop("receipt_sha256")
    contradictory = sign_canonical_receipt(contradictory)
    with pytest.raises(RuntimeError, match="failure phase evidence"):
        validate_source63_terminal_receipt(contradictory)

    unreaped = json.loads(json.dumps(failure))
    unreaped.update(
        {
            "failure_phase": "training_process_started",
            "training_pre_popen_guard_started": True,
            "training_pre_popen_guard_completed": True,
            "training_popen_attempted": True,
            "training_popen_reached": True,
            "training_process_pid": 23456,
            "training_process_reaped": False,
            "training_process_group_id": 23456,
            "training_process_group_isolated": True,
            "training_process_group_reaped": False,
            "training_process_group_binding_status": "bound_isolated",
            "training_stage_returned": False,
            "gpu_lock_released": False,
            "gpu_lock_released_before_failure": False,
            "unreaped_stage_process": "train_source63_counterfactual_five_seed",
            "gpu_lock_retained_for_unreaped_stage_process": True,
            "gpu_lock_release_audit": None,
            "artifacts_frozen_read_only": False,
            "artifact_freeze_contract": None,
            "artifact_freeze_contract_sha256": None,
        }
    )
    unreaped.pop("receipt_sha256")
    unreaped = sign_canonical_receipt(unreaped)
    assert validate_source63_terminal_receipt(unreaped)["status"] == FAILURE_STATUS

    group_binding_failed = json.loads(json.dumps(unreaped))
    group_binding_failed.update(
        {
            "training_process_reaped": True,
            "training_process_group_id": None,
            "training_process_group_isolated": False,
            "training_process_group_reaped": False,
            "training_process_group_binding_status": "failed_unproven",
        }
    )
    group_binding_failed.pop("receipt_sha256")
    group_binding_failed = sign_canonical_receipt(group_binding_failed)
    assert (
        validate_source63_terminal_receipt(group_binding_failed)["status"]
        == FAILURE_STATUS
    )

    popen_attempt_unproven = json.loads(json.dumps(failure))
    popen_attempt_unproven.update(
        {
            "failure_phase": "training_popen_attempt",
            "training_pre_popen_guard_started": True,
            "training_pre_popen_guard_completed": True,
            "training_popen_attempted": True,
            "training_popen_reached": False,
            "training_process_pid": None,
            "training_process_reaped": True,
            "training_process_group_id": None,
            "training_process_group_isolated": False,
            "training_process_group_reaped": False,
            "training_process_group_binding_status": "popen_attempt_unproven",
            "training_stage_returned": False,
            "gpu_lock_released": False,
            "gpu_lock_release_audit": None,
            "unreaped_stage_process": "train_source63_counterfactual_five_seed",
            "gpu_lock_retained_for_unreaped_stage_process": True,
            "artifacts_frozen_read_only": False,
            "artifact_freeze_contract": None,
            "artifact_freeze_contract_sha256": None,
        }
    )
    popen_attempt_unproven.pop("receipt_sha256")
    popen_attempt_unproven = sign_canonical_receipt(popen_attempt_unproven)
    assert (
        validate_source63_terminal_receipt(popen_attempt_unproven)["status"]
        == FAILURE_STATUS
    )

    hollow_idle = {
        "format": "etsf_guarded_rtx4090_idle_release_audit_v1",
        "status": "complete_two_idle_samples_released_for_training",
    }
    hollow_idle["gpu_idle_release_audit_sha256"] = canonical_sha256(hollow_idle)
    hollow_idle_failure = json.loads(json.dumps(failure))
    hollow_idle_failure["partial_gpu_idle_guard_audit"] = hollow_idle
    hollow_idle_failure.pop("receipt_sha256")
    hollow_idle_failure = sign_canonical_receipt(hollow_idle_failure)
    with pytest.raises(RuntimeError, match="idle release audit semantics"):
        validate_source63_terminal_receipt(hollow_idle_failure)

    hollow_release = {
        "format": "etsf_owned_gpu_lock_release_audit_v1",
        "released": True,
    }
    hollow_release["release_audit_sha256"] = canonical_sha256(hollow_release)
    hollow_release_failure = json.loads(json.dumps(failure))
    hollow_release_failure["gpu_lock_release_audit"] = hollow_release
    hollow_release_failure.pop("receipt_sha256")
    hollow_release_failure = sign_canonical_receipt(hollow_release_failure)
    with pytest.raises(RuntimeError, match="lock release audit semantics"):
        validate_source63_terminal_receipt(hollow_release_failure)

    impossible_initializer = json.loads(json.dumps(failure))
    impossible_initializer.update(
        {
            "failure_phase": "before_gpu_idle_guard",
            "gpu_idle_guard_started": False,
            "gpu_idle_release_reached": False,
            "partial_gpu_idle_guard_audit": None,
            "gpu_lock_acquired": False,
            "gpu_lock_released": False,
            "gpu_lock_release_audit": None,
            "unreaped_stage_process": "initialize_native_event_core",
            "gpu_lock_retained_for_unreaped_stage_process": False,
            "artifacts_frozen_read_only": False,
            "artifact_freeze_contract": None,
            "artifact_freeze_contract_sha256": None,
        }
    )
    impossible_initializer.pop("receipt_sha256")
    impossible_initializer = sign_canonical_receipt(impossible_initializer)
    with pytest.raises(RuntimeError, match="failure phase evidence"):
        validate_source63_terminal_receipt(impossible_initializer)

    with pytest.raises(RuntimeError, match="not materialized"):
        validate_published_source63_terminal_receipt(
            terminal_root / "final_receipt.json"
        )
    publish_frozen_terminal_receipt(
        terminal_root,
        "final_receipt.json",
        success,
        freeze_contract,
    )
    assert (
        validate_published_source63_terminal_receipt(
            terminal_root / "final_receipt.json"
        )["status"]
        == TERMINAL_STATUS
    )


def test_completed_manifest_is_exact_and_outcome_fields_are_not_projected(
    tmp_path: Path,
) -> None:
    root, event_spec, _ = _collector(tmp_path)
    metadata = _metadata(root, event_spec)
    assert len(metadata["groups"]) == 63
    assert [row["split"] for row in metadata["groups"]].count("test") == 5
    assert all("success" not in row and "steps" not in row for row in metadata["groups"])

    manifest = json.loads((root / "manifest.json").read_text())
    manifest["requested_seeds"][-1] += 1
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete or differs"):
        _metadata(root, event_spec)


def test_hdf_self_validation_excludes_all_five_holdout_groups(tmp_path: Path) -> None:
    root, event_spec, _ = _collector(tmp_path)
    metadata = _metadata(root, event_spec)
    calls, validator = _validator_calls()
    audits = audit_group_files(root, metadata, group_validator=validator)
    split = _split()
    assert calls == [*split["train"], *split["validation"]]
    assert len(calls) == 58
    test_audits = [audit for audit in audits if audit["split"] == "test"]
    assert len(test_audits) == 5
    assert all(
        audit["self_validation"]["label_datasets_opened"] == 0
        and audit["self_validation"]["labels_used"] is False
        and "byte_hash_only" in audit["self_validation"]["status"]
        for audit in test_audits
    )


def test_group_mutation_during_self_validation_is_rejected(tmp_path: Path) -> None:
    root, event_spec, _ = _collector(tmp_path)
    metadata = _metadata(root, event_spec)

    def mutate(path: Path, seed: int, *_args, **_kwargs):
        path.write_bytes(path.read_bytes() + b"tamper")
        return {"seed": seed, "resolved_seed": seed + 1000}

    with pytest.raises(RuntimeError, match="changed during self-validation"):
        audit_group_files(root, metadata, group_validator=mutate)


def test_snapshot_is_independent_read_only_and_records_holdout_boundary(
    tmp_path: Path,
) -> None:
    root, event_spec, _ = _collector(tmp_path)
    (root / "run.exit").write_text("0\n", encoding="ascii")
    metadata = _metadata(root, event_spec)
    _, validator = _validator_calls()
    audits = audit_group_files(root, metadata, group_validator=validator)
    output = tmp_path / "output"
    output.mkdir()
    receipt = freeze_source_snapshot(
        output,
        collector_exit=root / "run.exit",
        manifest_path=root / "manifest.json",
        event_spec=event_spec,
        source_split=SOURCE_SPLIT,
        metadata=metadata,
        group_audits=audits,
    )
    source_group = root / "groups" / metadata["groups"][0]["path"]
    copied_group = output / "source_snapshot" / "groups" / source_group.name
    assert source_group.stat().st_ino != copied_group.stat().st_ino
    assert file_sha256(source_group) == file_sha256(copied_group)
    assert copied_group.stat().st_mode & 0o222 == 0
    assert receipt["train_validation_hdf_self_validated"] == 58
    assert receipt["test_hdf_byte_hashed"] == 5
    assert receipt["test_labels_used"] is False
    assert receipt["test_hdf_label_datasets_opened"] == 0
    # Restore permissions so pytest can remove its temporary directory.
    for path in sorted((output / "source_snapshot").rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    (output / "source_snapshot").chmod(0o755)


def test_stage_commands_freeze_exact_initializer_and_3000_step_bf16_contract(
    tmp_path: Path,
) -> None:
    snapshot = {
        "source_snapshot_path": str(tmp_path / "snapshot"),
        "event_spec_sha256": "2" * 64,
        "source_manifest_sha256": "3" * 64,
        "source_split_sha256": "4" * 64,
    }
    commands = build_stage_commands(
        python_bin=Path("/venv/bin/python"),
        stage_runner=Path("/code/run_bound.py"),
        launch_plan=tmp_path / "launch_plan.json",
        static_plan_sha256="5" * 64,
        runtime_contract_sha256="6" * 64,
        initializer=Path("/code/initializer.py"),
        trainer=Path("/code/trainer.py"),
        output_root=tmp_path / "result",
        snapshot_receipt=snapshot,
        modeling_sha256=MODELING_SHA,
        bridge_sha256=BRIDGE_SHA,
        num_workers=4,
    )
    initialize, train = commands
    assert initialize["argv"][:3] == [
        "/venv/bin/python",
        "-I",
        "/code/run_bound.py",
    ]
    assert initialize["isolated_python"] is True
    assert train["runtime_contract_sha256"] == "6" * 64
    assert "--source-manifest-sha256" in initialize["argv"]
    assert "--state-modeling-sha256" in initialize["argv"]
    assert train["training_seeds"] == list(TRAINING_SEEDS)
    assert train["training_steps"] == TRAINING_STEPS
    assert train["cuda_amp"] == "bf16"
    assert train["unfreeze_semantic"] is True
    assert train["object_names"] == ["can"]
    assert train["argv"][train["argv"].index("--steps") + 1] == "3000"
    assert train["argv"][train["argv"].index("--early-stopping-patience") + 1] == "0"
    assert train["argv"][train["argv"].index("--amp") + 1] == "bf16"
    assert train["argv"][train["argv"].index("--device") + 1] == "cuda"
    assert "--unfreeze-semantic" in train["argv"]
    assert "--require-policy-feature-action-bridge" in train["argv"]


def test_canonical_environment_removes_inherited_pythonpath_and_env_hints() -> None:
    inherited = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/poison/torch-2.10",
        "PYTHONHOME": "/poison/python",
        "PYTHONUSERBASE": "/poison/user",
        "PYTHONSAFEPATH": "1",
        "VIRTUAL_ENV": "/poison/venv",
        "CONDA_PREFIX": "/poison/conda",
        "LD_LIBRARY_PATH": "/cuda/lib64",
    }
    environment = canonical_python_environment(
        inherited, gpu_index=0, omp_threads=8
    )
    contract = canonical_environment_contract(gpu_index=0, omp_threads=8)
    assert_canonical_python_environment(environment, contract)
    assert environment["PATH"] == "/usr/bin"
    assert environment["LD_LIBRARY_PATH"] == "/cuda/lib64"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert all(
        name not in environment
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONSAFEPATH",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
        )
    )


def test_runtime_version_or_module_drift_fails_closed() -> None:
    expected = _fake_runtime()
    version_drift = json.loads(json.dumps(expected))
    version_drift["torch_version"] = "different-torch"
    version_drift.pop("runtime_contract_sha256")
    version_drift["runtime_contract_sha256"] = canonical_sha256(version_drift)
    with pytest.raises(RuntimeError, match="runtime drifted"):
        assert_runtime_matches(expected, version_drift, role="validator")

    module_drift = json.loads(json.dumps(expected))
    module_drift["torch_module"]["path"] = "/different/torch/__init__.py"
    module_drift.pop("runtime_contract_sha256")
    module_drift["runtime_contract_sha256"] = canonical_sha256(module_drift)
    with pytest.raises(RuntimeError, match="runtime drifted"):
        assert_runtime_matches(expected, module_drift, role="validator")


def test_bound_runner_authenticates_runtime_before_adding_local_import_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_root = tmp_path / "immutable_code"
    scripts = code_root / "scripts"
    scripts.mkdir(parents=True)
    target = scripts / "initializer.py"
    trainer = scripts / "trainer.py"
    target.write_text("from sibling import VALUE\n", encoding="utf-8")
    trainer.write_text("pass\n", encoding="utf-8")
    (scripts / "sibling.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = _fake_runtime()
    environment_contract = canonical_environment_contract(gpu_index=0, omp_threads=8)
    plan = {
        "format": "etsf_smolvla_schema5_source63_native_training_launcher_v1",
        "code_root": str(code_root),
        "initializer": str(target),
        "trainer": str(trainer),
        "runtime_contract": runtime,
        "environment_contract": environment_contract,
    }
    plan["static_plan_sha256"] = canonical_sha256(plan)
    plan_path = tmp_path / "launch_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    environment = canonical_python_environment(os.environ, gpu_index=0, omp_threads=8)
    for key in list(os.environ):
        if key.upper().startswith("PYTHON") or key in {
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_PYTHON_EXE",
            "__PYVENV_LAUNCHER__",
        }:
            monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("run_etsf_bound_python_stage._current_runtime", lambda: runtime)
    observed: dict = {}

    def fake_run_path(path: str, *, run_name: str) -> None:
        observed["path"] = path
        observed["run_name"] = run_name
        observed["sys_path_zero"] = sys.path[0]

    monkeypatch.setattr("run_etsf_bound_python_stage.runpy.run_path", fake_run_path)
    previous = list(sys.path)
    try:
        run_bound_target(
            launch_plan=plan_path,
            static_plan_sha256=plan["static_plan_sha256"],
            target=target,
            target_argv=["--example", "1"],
        )
    finally:
        sys.path[:] = previous
    assert observed == {
        "path": str(target.resolve()),
        "run_name": "__main__",
        "sys_path_zero": str(scripts.resolve()),
    }


def test_static_preflight_and_existing_output_are_hdf5_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runtime_probe(monkeypatch)
    suite_script = tmp_path / "run_all_full.sh"
    suite_script.write_text("#!/bin/bash\n", encoding="utf-8")
    identity = _external_parent_identity(script=str(suite_script))
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.read_external_process_identity",
        lambda _pid: identity,
    )
    collector = tmp_path / "collector"
    (collector / "groups").mkdir(parents=True)
    # Neither manifest nor group files need to exist during static preregistration.
    event_spec = _event_spec(tmp_path)
    output = tmp_path / "new_output"
    args = argparse.Namespace(
        code_root=ROOT,
        collector_root=collector,
        run_exit=None,
        source_split=SOURCE_SPLIT,
        event_spec=event_spec,
        output=output,
        python_bin=Path(sys.executable),
        initializer=None,
        trainer=None,
        gpu_index=0,
        gpu_lock=None,
        poll_seconds=30.0,
        collector_timeout_seconds=0.0,
        gpu_timeout_seconds=3600.0,
        initializer_timeout_seconds=600.0,
        training_timeout_seconds=0.0,
        expected_groups=63,
        num_workers=4,
        omp_threads=8,
        external_suite_parent_pid=identity["pid"],
        external_suite_parent_start_ticks=identity["start_ticks"],
        external_suite_parent_boot_id=identity["boot_id"],
        external_suite_parent_cmdline_sha256=identity["cmdline_sha256"],
        external_suite_parent_script=suite_script,
        external_suite_parent_script_sha256=file_sha256(suite_script),
    )
    plan = static_preflight(args)
    assert plan["manifest_read_during_static_preflight"] is False
    assert plan["hdf5_opened_during_static_preflight"] is False
    assert plan["external_suite_parent_guard"]["enabled"] is True
    assert plan["external_suite_parent_guard"]["pid"] == identity["pid"]
    output.mkdir()
    with pytest.raises(FileExistsError):
        static_preflight(args)


def test_sensitive_or_existing_new_paths_are_rejected(tmp_path: Path) -> None:
    forbidden_parent = tmp_path / "fresh_confirmation"
    forbidden_parent.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        resolve_new_path(forbidden_parent / "output", role="test output")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        resolve_new_path(existing, role="test output")


def test_detach_starts_new_session_without_manifest_or_hdf5_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runtime_probe(monkeypatch)
    suite_script = tmp_path / "run_all_full.sh"
    suite_script.write_text("#!/bin/bash\n", encoding="utf-8")
    identity = _external_parent_identity(script=str(suite_script))
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.read_external_process_identity",
        lambda _pid: identity,
    )
    collector = tmp_path / "collector"
    (collector / "groups").mkdir(parents=True)
    event_spec = _event_spec(tmp_path)
    output = tmp_path / "detached_output"
    captured: dict = {}

    class FakeProcess:
        pid = 424242

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.subprocess.Popen",
        fake_popen,
    )
    args = argparse.Namespace(
        command="detach",
        code_root=ROOT,
        collector_root=collector,
        run_exit=None,
        source_split=SOURCE_SPLIT,
        event_spec=event_spec,
        output=output,
        python_bin=Path(sys.executable),
        initializer=None,
        trainer=None,
        gpu_index=0,
        gpu_lock=None,
        poll_seconds=30.0,
        collector_timeout_seconds=0.0,
        gpu_timeout_seconds=3600.0,
        initializer_timeout_seconds=600.0,
        training_timeout_seconds=0.0,
        expected_groups=63,
        num_workers=4,
        omp_threads=8,
        external_suite_parent_pid=identity["pid"],
        external_suite_parent_start_ticks=identity["start_ticks"],
        external_suite_parent_boot_id=identity["boot_id"],
        external_suite_parent_cmdline_sha256=identity["cmdline_sha256"],
        external_suite_parent_script=suite_script,
        external_suite_parent_script_sha256=file_sha256(suite_script),
        detach_receipt=tmp_path / "detach_receipt.json",
        detach_log=tmp_path / "detach.log",
    )
    receipt = detach(args)
    assert captured["start_new_session"] is True
    assert captured["stdin"] is not None
    assert captured["close_fds"] is True
    assert captured["argv"][1] == "-I"
    assert captured["argv"][3] == "run"
    assert captured["argv"][
        captured["argv"].index("--external-suite-parent-pid") + 1
    ] == str(identity["pid"])
    assert "PYTHONPATH" not in captured["env"]
    assert "PYTHONHOME" not in captured["env"]
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert receipt["manifest_read_by_detach"] is False
    assert receipt["hdf5_opened_by_detach"] is False
    assert receipt["external_suite_parent_guard"]["guard_contract_sha256"]
    assert receipt["survives_client_disconnect"] is True
    assert not output.exists()


def test_failed_subprocess_stage_publishes_atomic_failure_receipt_and_log(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    argv = [sys.executable, "-c", "print('failed-stage'); raise SystemExit(3)"]
    stage = {"stage": "cpu_failure_smoke", "argv": argv, "argv_sha256": canonical_sha256(argv)}
    state_path = output / "launch_state.json"
    state: dict = {}
    with pytest.raises(RuntimeError, match="failed with exit 3"):
        run_subprocess_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state=state,
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=5,
        )
    receipt = json.loads(
        (output / "stage_receipts" / "cpu_failure_smoke.json").read_text()
    )
    log = output / "logs" / "cpu_failure_smoke.log"
    assert receipt["status"] == "failed_closed"
    assert receipt["returncode"] == 3
    assert receipt["log_sha256"] == file_sha256(log)
    assert "failed-stage" in log.read_text()
    assert log.stat().st_mode & 0o222 == 0


def test_runtime_drift_prevents_stage_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "runtime_drift_output"
    output.mkdir()
    argv = [sys.executable, "-c", "raise AssertionError('must not start')"]
    stage = {
        "stage": "runtime_drift_smoke",
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
    }
    expected = _fake_runtime()
    drifted = json.loads(json.dumps(expected))
    drifted["torch_version"] = "drifted-torch"
    drifted.pop("runtime_contract_sha256")
    drifted["runtime_contract_sha256"] = canonical_sha256(drifted)
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.probe_python_runtime",
        lambda *_args, **_kwargs: drifted,
    )

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("stage Popen must not occur after runtime drift")

    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.subprocess.Popen",
        forbidden_popen,
    )
    environment = canonical_python_environment(os.environ, gpu_index=0, omp_threads=8)
    with pytest.raises(RuntimeError, match="runtime drifted"):
        run_subprocess_stage(
            stage,
            output_root=output,
            environment=environment,
            state={},
            state_path=output / "launch_state.json",
            poll_seconds=0.01,
            timeout_seconds=5,
            expected_runtime_contract=expected,
        )
    receipt = json.loads(
        (output / "stage_receipts" / "runtime_drift_smoke.json").read_text()
    )
    assert receipt["status"] == "failed_closed"
    assert receipt["returncode"] is None
    assert receipt["error_type"] == "RuntimeError"


def test_pre_popen_guard_runs_after_runtime_probe_and_can_prevent_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pre_popen_guard"
    output.mkdir()
    expected = _fake_runtime()
    events: list[str] = []

    def probe(*_args, **_kwargs):
        events.append("runtime_probe")
        return expected

    def guard():
        events.append("pre_popen_guard")
        raise RuntimeError("guard denied training start")

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("Popen must not run after guard failure")

    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.probe_python_runtime",
        probe,
    )
    monkeypatch.setattr(
        "launch_smolvla_schema5_source63_native_training.subprocess.Popen",
        forbidden_popen,
    )
    argv = [sys.executable, "-c", "raise AssertionError('must not run')"]
    with pytest.raises(RuntimeError, match="guard denied"):
        run_subprocess_stage(
            {"stage": "guarded_stage", "argv": argv, "argv_sha256": canonical_sha256(argv)},
            output_root=output,
            environment=os.environ,
            state={},
            state_path=output / "state.json",
            poll_seconds=0.01,
            timeout_seconds=5,
            expected_runtime_contract=expected,
            pre_popen_guard=guard,
        )
    assert events == ["runtime_probe", "pre_popen_guard"]
    receipt = json.loads((output / "stage_receipts" / "guarded_stage.json").read_text())
    assert receipt["status"] == "failed_closed"
    assert receipt["popen_unix"] is None


@pytest.mark.parametrize("failure_target", ["running_receipt", "running_state"])
def test_post_popen_atomic_write_failure_reaps_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    import launch_smolvla_schema5_source63_native_training as launcher

    output = tmp_path / failure_target
    output.mkdir()
    state_path = output / "launch_state.json"
    receipt_path = output / "stage_receipts" / "atomic_failure_smoke.json"
    argv = [sys.executable, "-c", "import time; time.sleep(300)"]
    stage = {
        "stage": "atomic_failure_smoke",
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
    }
    real_atomic_json = launcher.atomic_json
    injected = {"done": False}

    def fail_once(path: Path, value: dict) -> None:
        resolved = Path(path)
        should_fail = (
            failure_target == "running_receipt"
            and resolved == receipt_path
            and value.get("status") == "running"
        ) or (
            failure_target == "running_state"
            and resolved == state_path
            and value.get("status") == "running_atomic_failure_smoke"
        )
        if should_fail and not injected["done"]:
            injected["done"] = True
            raise OSError(f"injected {failure_target} write failure")
        real_atomic_json(resolved, value)

    monkeypatch.setattr(launcher, "atomic_json", fail_once)
    lifecycle: dict = {}
    with pytest.raises(OSError, match="injected"):
        run_subprocess_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state={},
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=5,
            lifecycle=lifecycle,
        )
    assert injected["done"] is True
    assert lifecycle["popen_reached"] is True
    assert lifecycle["process_reaped"] is True
    assert isinstance(lifecycle["returncode"], int)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed_closed"
    assert receipt["error_type"] == "OSError"


def test_post_popen_failure_reaps_spawned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import launch_smolvla_schema5_source63_native_training as launcher

    output = tmp_path / "process_group_failure"
    output.mkdir()
    state_path = output / "launch_state.json"
    ready_path = output / "child_pid.txt"
    source = (
        "import pathlib,signal,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']);"
        "signal.signal(signal.SIGTERM,lambda *_: (child.wait(),sys.exit(143)));"
        f"pathlib.Path({str(ready_path)!r}).write_text(str(child.pid));"
        "time.sleep(300)"
    )
    argv = [sys.executable, "-c", source]
    stage = {
        "stage": "process_group_failure_smoke",
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
    }
    real_atomic_json = launcher.atomic_json
    injected = {"done": False}

    def fail_after_child_started(path: Path, value: dict) -> None:
        if (
            Path(path) == state_path
            and value.get("status") == "running_process_group_failure_smoke"
            and not injected["done"]
        ):
            for _ in range(200):
                if ready_path.is_file():
                    break
                import time

                time.sleep(0.01)
            assert ready_path.is_file()
            injected["done"] = True
            raise OSError("injected process-tree state write failure")
        real_atomic_json(Path(path), value)

    monkeypatch.setattr(launcher, "atomic_json", fail_after_child_started)
    lifecycle: dict = {}
    with pytest.raises(OSError, match="process-tree"):
        run_subprocess_stage(
            stage,
            output_root=output,
            environment=os.environ,
            state={},
            state_path=state_path,
            poll_seconds=0.01,
            timeout_seconds=5,
            lifecycle=lifecycle,
        )
    assert injected["done"] is True
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_isolated"] is True
    assert lifecycle["process_group_reaped"] is True
    assert lifecycle["process_group_id"] == lifecycle["process_pid"]
    assert not _process_group_exists(lifecycle["process_group_id"])


def test_process_group_binding_failure_is_popen_reached_and_lock_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import launch_smolvla_schema5_source63_native_training as launcher

    output = tmp_path / "getpgid_failure"
    output.mkdir()
    argv = [sys.executable, "-c", "import time; time.sleep(300)"]
    lifecycle: dict = {}

    def fail_getpgid(_pid: int) -> int:
        raise OSError("injected getpgid failure")

    monkeypatch.setattr(launcher.os, "getpgid", fail_getpgid)
    with pytest.raises(OSError, match="getpgid"):
        run_subprocess_stage(
            {
                "stage": "getpgid_failure_smoke",
                "argv": argv,
                "argv_sha256": canonical_sha256(argv),
            },
            output_root=output,
            environment=os.environ,
            state={},
            state_path=output / "state.json",
            poll_seconds=0.01,
            timeout_seconds=5,
            lifecycle=lifecycle,
        )
    assert lifecycle["popen_reached"] is True
    assert isinstance(lifecycle["process_pid"], int)
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_id"] is None
    assert lifecycle["process_group_isolated"] is False
    assert lifecycle["process_group_reaped"] is False
    assert not _owned_gpu_lock_release_allowed(
        gpu_lock_acquired=True,
        release_audit=None,
        initializer_lifecycle={},
        training_lifecycle=lifecycle,
    )


def test_popen_return_before_lifecycle_recording_retains_lock(
    tmp_path: Path,
) -> None:
    class FailPopenReachedUpdateOnce(dict):
        failed = False

        def update(self, *args, **kwargs):
            incoming = dict(*args, **kwargs)
            if incoming.get("popen_reached") is True and not self.failed:
                self.failed = True
                raise KeyboardInterrupt("injected lifecycle recording interruption")
            return super().update(incoming)

    output = tmp_path / "lifecycle_recording_failure"
    output.mkdir()
    argv = [sys.executable, "-c", "import time; time.sleep(300)"]
    lifecycle = FailPopenReachedUpdateOnce()
    with pytest.raises(KeyboardInterrupt, match="lifecycle recording"):
        run_subprocess_stage(
            {
                "stage": "lifecycle_recording_failure_smoke",
                "argv": argv,
                "argv_sha256": canonical_sha256(argv),
            },
            output_root=output,
            environment=os.environ,
            state={},
            state_path=output / "state.json",
            poll_seconds=0.01,
            timeout_seconds=5,
            lifecycle=lifecycle,
        )
    assert lifecycle.failed is True
    assert lifecycle["popen_attempted"] is True
    assert lifecycle["popen_reached"] is False
    assert lifecycle["process_reaped"] is True
    assert lifecycle["process_group_reaped"] is False
    assert not _owned_gpu_lock_release_allowed(
        gpu_lock_acquired=True,
        release_audit=None,
        initializer_lifecycle={},
        training_lifecycle=lifecycle,
    )


def test_terminal_receipt_becomes_readable_only_after_tree_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import launch_smolvla_schema5_source63_native_training as launcher

    failed_root = tmp_path / "failed_publish"
    (failed_root / "logs").mkdir(parents=True)
    (failed_root / "logs" / "stage.log").write_text("complete\n")
    failed_contract = tree_freeze_contract(failed_root)
    failed_receipt = {"artifact_freeze_contract": failed_contract}
    monkeypatch.setattr(
        launcher,
        "_verify_frozen_tree_before_terminal_publish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected freeze verification failure")
        ),
    )
    with pytest.raises(OSError, match="freeze verification"):
        publish_frozen_terminal_receipt(
            failed_root,
            "final_receipt.json",
            failed_receipt,
            failed_contract,
        )
    assert not (failed_root / "final_receipt.json").exists()

    monkeypatch.undo()
    chmod_failure_root = tmp_path / "chmod_failure_publish"
    (chmod_failure_root / "logs").mkdir(parents=True)
    (chmod_failure_root / "logs" / "stage.log").write_text("complete\n")
    chmod_failure_contract = tree_freeze_contract(chmod_failure_root)
    chmod_failure_receipt = {"artifact_freeze_contract": chmod_failure_contract}
    chmod_terminal = chmod_failure_root / "final_receipt.json"
    real_chmod = Path.chmod

    def chmod_then_raise(path: Path, mode: int, *args, **kwargs):
        result = real_chmod(path, mode, *args, **kwargs)
        if path == chmod_terminal and mode == 0o444:
            raise OSError("injected post-chmod publication failure")
        return result

    monkeypatch.setattr(Path, "chmod", chmod_then_raise)
    with pytest.raises(OSError, match="post-chmod"):
        publish_frozen_terminal_receipt(
            chmod_failure_root,
            "final_receipt.json",
            chmod_failure_receipt,
            chmod_failure_contract,
        )
    assert not chmod_terminal.exists()

    monkeypatch.undo()
    success_root = tmp_path / "successful_publish"
    (success_root / "logs").mkdir(parents=True)
    (success_root / "logs" / "stage.log").write_text("complete\n")
    success_contract = tree_freeze_contract(success_root)
    success_receipt = {"artifact_freeze_contract": success_contract}
    publish_frozen_terminal_receipt(
        success_root,
        "final_receipt.json",
        success_receipt,
        success_contract,
    )
    assert (success_root / "final_receipt.json").stat().st_mode & 0o777 == 0o444
    assert success_root.stat().st_mode & 0o777 == 0o555
    assert all(path.stat().st_mode & 0o222 == 0 for path in success_root.rglob("*"))


def test_unreaped_stage_retains_owned_gpu_lock() -> None:
    unreaped = {
        "popen_attempted": True,
        "popen_reached": True,
        "process_reaped": True,
        "process_group_reaped": False,
    }
    reaped = {
        "popen_attempted": True,
        "popen_reached": True,
        "process_reaped": True,
        "process_group_reaped": True,
    }
    assert not _owned_gpu_lock_release_allowed(
        gpu_lock_acquired=True,
        release_audit=None,
        initializer_lifecycle={},
        training_lifecycle=unreaped,
    )
    assert not _owned_gpu_lock_release_allowed(
        gpu_lock_acquired=True,
        release_audit=None,
        initializer_lifecycle=unreaped,
        training_lifecycle={},
    )
    assert _owned_gpu_lock_release_allowed(
        gpu_lock_acquired=True,
        release_audit=None,
        initializer_lifecycle=reaped,
        training_lifecycle={},
    )
