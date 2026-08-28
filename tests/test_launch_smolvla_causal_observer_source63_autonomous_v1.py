from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_causal_observer_source63_autonomous_v1 as launcher  # noqa: E402


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: launcher.canonical_sha256(value)}


def _disabled_guard() -> dict[str, Any]:
    return _signed(
        {"format": launcher.EXTERNAL_GUARD_FORMAT, "enabled": False},
        "guard_sha256",
    )


def _minimal_plan(output: Path) -> dict[str, Any]:
    manifest = "/srv/source63/manifest.json"
    base = {
        "format": launcher.PLAN_FORMAT,
        "status": "frozen_create_once_source63_only_plan",
        "server_hostname": socket.gethostname(),
        "python": {"path": sys.executable, "file_sha256": "a" * 64},
        "code_closure": {
            "root": "/srv/source63/code",
            "files": [],
            "file_count": 1,
            "closure_sha256": "b" * 64,
        },
        "entrypoints": {
            launcher.FREEZER: {"path": f"/srv/source63/{launcher.FREEZER}", "file_sha256": "1" * 64},
            launcher.MATERIALIZER: {"path": f"/srv/source63/{launcher.MATERIALIZER}", "file_sha256": "2" * 64},
            launcher.TRAINER: {"path": f"/srv/source63/{launcher.TRAINER}", "file_sha256": "3" * 64},
            "launch_smolvla_causal_observer_source63_autonomous_v1.py": {
                "path": "/srv/source63/launch_smolvla_causal_observer_source63_autonomous_v1.py",
                "file_sha256": "4" * 64,
            },
        },
        "source_inputs": {
            "schema5_manifest": {"path": manifest, "file_sha256": "c" * 64},
            "frozen_split": {"path": "/srv/source63/split.json", "file_sha256": "d" * 64},
            "event_spec": {"path": "/srv/source63/event_spec.json", "file_sha256": "e" * 64},
            "group_root": "/srv/source63/groups",
        },
        "source_identity": {
            "source_name": "smolvla_source63",
            "actor_name": "smolvla_aloha_agilex",
            "policy_family": "smolvla",
            "calibration_count": 10,
            "source_embodiment_only": True,
        },
        "output": str(output),
        "gpu_identity": {
            "gpu_index": 3,
            "gpu_uuid": "GPU-unit-4090",
            "gpu_name": "NVIDIA GeForce RTX 4090",
        },
        "gpu_lock": str(output.parent / "source63-gpu.lock"),
        "external_guard": _disabled_guard(),
        "training": {
            "hidden_dim": 96,
            "adapter_rank": 8,
            "epochs": 30,
            "batch_size_per_actor": 16,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "bootstrap_samples": 2000,
            "seed": 20260828,
            "device": "cuda",
        },
        "execution_order": list(launcher.STAGES),
        "claims": {
            "cross_embodiment_claimed": False,
            "target_task_success_claimed": False,
        },
        "protected_target_paths_allowed": False,
    }
    return _signed(base, "static_plan_sha256")


def test_recursive_code_closure_is_content_addressed_and_detects_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "code"
    (root / "scripts").mkdir(parents=True)
    first = root / "scripts" / "one.py"
    second = root / "README.md"
    first.write_text("print('one')\n", encoding="utf-8")
    second.write_text("source only\n", encoding="utf-8")
    closure = launcher.recursive_code_closure(root)
    assert closure["file_count"] == 2
    assert closure["closure_sha256"] == launcher.canonical_sha256(
        {"root": str(root.resolve()), "files": closure["files"]}
    )
    launcher.verify_code_closure(closure)
    first.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherContractError, match="closure changed"):
        launcher.verify_code_closure(closure)


@pytest.mark.parametrize(
    "component",
    ("piper_runs", "evaluation400", "tests", "fresh_data", "confirmation"),
)
def test_target_and_test_data_paths_are_rejected_before_read(component: str) -> None:
    with pytest.raises(launcher.LauncherContractError, match="protected"):
        launcher.reject_protected_data_path(
            Path("/srv/source63") / component / "manifest.json", "source manifest"
        )


def test_process_identity_binds_pid_start_boot_cmdline_and_exact_script(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    pid = 321
    process = proc / str(pid)
    process.mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    script = tmp_path / "external_watcher.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    fields = ["S", *(["0"] * 18), "777", *(["0"] * 4)]
    (process / "stat").write_text(
        f"{pid} (external watcher) " + " ".join(fields), encoding="ascii"
    )
    cmdline = b"/bin/sh\0" + os.fsencode(script) + b"\0--run\0"
    (process / "cmdline").write_bytes(cmdline)
    (proc / "sys/kernel/random/boot_id").write_text("boot-unit\n", encoding="ascii")
    identity = launcher.read_process_identity(pid, proc_root=proc)
    assert identity == {
        "pid": pid,
        "start_ticks": 777,
        "boot_id": "boot-unit",
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        "cmdline_tokens": ["/bin/sh", str(script), "--run"],
    }


def test_external_guard_rejects_pid_reuse_and_wait_never_signals(
    tmp_path: Path,
) -> None:
    script = tmp_path / "watcher.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    cmdline = b"/bin/sh\0" + os.fsencode(script) + b"\0"
    identity = {
        "pid": 88,
        "start_ticks": 99,
        "boot_id": "boot-a",
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        "cmdline_tokens": ["/bin/sh", str(script)],
    }
    args = argparse.Namespace(
        external_pid=88,
        external_start_ticks=99,
        external_boot_id="boot-a",
        external_cmdline_sha256=identity["cmdline_sha256"],
        external_script=script,
        external_script_sha256=launcher.file_sha256(script),
    )
    guard = launcher.freeze_external_guard(args, identity_reader=lambda _pid: identity)
    assert launcher.external_process_alive(guard, identity_reader=lambda _pid: identity)
    reused = {**identity, "start_ticks": 100}
    with pytest.raises(launcher.LauncherContractError, match="reused"):
        launcher.external_process_alive(guard, identity_reader=lambda _pid: reused)
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "os.kill(" not in source
    assert "send_signal(" not in source
    assert "terminate(" not in source


def test_wait_requires_exact_exit_then_two_idle_samples(tmp_path: Path) -> None:
    script = tmp_path / "watcher.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    identity = {
        "pid": 7,
        "start_ticks": 17,
        "boot_id": "boot",
        "cmdline_sha256": "7" * 64,
        "cmdline_tokens": [str(script)],
    }
    base = {
        "format": launcher.EXTERNAL_GUARD_FORMAT,
        "enabled": True,
        "pid": 7,
        "start_ticks": 17,
        "boot_id": "boot",
        "cmdline_sha256": "7" * 64,
        "script": str(script),
        "script_sha256": launcher.file_sha256(script),
    }
    guard = _signed(base, "guard_sha256")
    identities = iter([identity, None, None, None])
    pids = iter([[42], [], []])
    gpu = {
        "gpu_index": 3,
        "gpu_uuid": "GPU-unit-4090",
        "gpu_name": "NVIDIA GeForce RTX 4090",
    }
    # GPU PIDs are queried only after the parent is absent.  The busy sample
    # resets the streak, so two further empty samples are mandatory.
    audit = launcher.wait_for_external_exit_and_idle_gpu(
        guard=guard,
        expected_gpu=gpu,
        timeout_seconds=10,
        poll_seconds=0.001,
        identity_reader=lambda _pid: next(identities),
        gpu_identity_reader=lambda _uuid: gpu,
        gpu_pid_reader=lambda _uuid: next(pids),
        sleeper=lambda _seconds: None,
    )
    assert audit["status"] == "complete_external_gone_and_gpu_idle"
    assert audit["idle_confirmations"] == 2
    assert [item["external_process_alive"] for item in audit["observations"]] == [
        True, False, False, False
    ]
    assert audit["audit_sha256"] == launcher.canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )


def test_static_plan_binds_recursive_code_inputs_and_freezer_cli() -> None:
    with tempfile.TemporaryDirectory(prefix="source63-launcher-") as temporary_name:
        root = Path(temporary_name)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        for name in launcher.REQUIRED_CODE_FILES:
            (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
        manifest = root / "manifest.json"
        split = root / "split.json"
        event = root / "event_spec.json"
        for path in (manifest, split, event):
            _write_json(path, {"source": path.stem})
        args = argparse.Namespace(
            required_hostname=socket.gethostname(),
            output=root / "observer_output",
            code_root=code,
            python=Path(sys.executable).resolve(),
            schema5_manifest=manifest,
            schema5_manifest_sha256=launcher.file_sha256(manifest),
            frozen_split=split,
            frozen_split_sha256=launcher.file_sha256(split),
            event_spec=event,
            event_spec_sha256=launcher.file_sha256(event),
            group_root=None,
            gpu_uuid="GPU-unit-4090",
            gpu_lock=root / "gpu.lock",
            external_pid=None,
            external_start_ticks=None,
            external_boot_id=None,
            external_cmdline_sha256=None,
            external_script=None,
            external_script_sha256=None,
            calibration_count=10,
            source_name="smolvla_source63",
            actor_name="smolvla_aloha_agilex",
            policy_family="smolvla",
            hidden_dim=96,
            adapter_rank=8,
            epochs=30,
            batch_size_per_actor=16,
            learning_rate=0.0003,
            weight_decay=0.0001,
            bootstrap_samples=2000,
            seed=20260828,
        )
        gpu = {
            "gpu_index": 3,
            "gpu_uuid": "GPU-unit-4090",
            "gpu_name": "NVIDIA GeForce RTX 4090",
        }
        plan = launcher.build_static_plan(args, gpu_identity_reader=lambda _uuid: gpu)
        launcher.validate_static_plan(plan)
        assert plan["code_closure"]["file_count"] == len(
            launcher.REQUIRED_CODE_FILES
        )
        assert plan["source_inputs"]["schema5_manifest"]["file_sha256"] == (
            launcher.file_sha256(manifest)
        )
        launcher.initialize_output(plan)
        commands = launcher.build_stage_commands(plan)
        request = commands["request"]
        assert request[:3] == [str(Path(sys.executable).resolve()), "-I", "-c"]
        assert request[3] == launcher.ISOLATED_RUNPY_BOOTSTRAP
        assert request[6] == str(scripts / launcher.FREEZER)
        for flag in (
            "--schema5-manifest-sha256",
            "--frozen-split-sha256",
            "--event-spec-sha256",
            "--calibration-count",
        ):
            assert flag in request
        assert commands["training"][-2:] == ["--device", "cuda"]
        assert "piper" not in plan["source_inputs"]["schema5_manifest"]["path"].casefold()


def test_cuda_environment_exposes_only_exact_uuid_and_scrubs_inheritance() -> None:
    environment = launcher.canonical_cuda_environment(
        {
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "PYTHONPATH": "/untrusted",
            "CONDA_PREFIX": "/untrusted-conda",
        },
        "GPU-exact-4090",
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-exact-4090"
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "GPU-exact-4090"
    assert "PYTHONPATH" not in environment
    assert "CONDA_PREFIX" not in environment


def test_real_isolated_subprocess_validates_closure_then_imports_sibling(
    tmp_path: Path,
) -> None:
    code_root = tmp_path / "code"
    scripts = code_root / "scripts"
    scripts.mkdir(parents=True)
    sibling = scripts / "sibling_dependency.py"
    target = scripts / "target.py"
    sibling.write_text("VALUE = 'sibling-import-ok'\n", encoding="utf-8")
    target.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from sibling_dependency import VALUE\n"
        "Path(sys.argv[1]).write_text(VALUE, encoding='utf-8')\n",
        encoding="utf-8",
    )
    output = tmp_path / "isolated_output"
    output.mkdir()
    plan = _minimal_plan(output)
    plan_base = dict(plan)
    plan_base.pop("static_plan_sha256")
    plan_base["code_closure"] = launcher.recursive_code_closure(code_root)
    plan_base["entrypoints"] = {
        target.name: {
            "path": str(target.resolve()),
            "file_sha256": launcher.file_sha256(target),
        }
    }
    plan_base["python"] = {
        "path": str(Path(sys.executable).resolve()),
        "file_sha256": launcher.file_sha256(Path(sys.executable).resolve()),
    }
    plan = _signed(plan_base, "static_plan_sha256")
    _write_json(output / "static_plan.json", plan)
    result_path = tmp_path / "sibling_result.txt"
    command = [*launcher.isolated_runpy_prefix(plan, target.name), str(result_path)]
    completed = launcher.subprocess.run(
        command,
        env=launcher.cpu_stage_environment(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert result_path.read_text(encoding="utf-8") == "sibling-import-ok"

    # The command was already frozen.  Mutating a sibling must now fail in the
    # bootstrap before sys.path insertion/runpy execution.
    result_path.unlink()
    sibling.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    rejected = launcher.subprocess.run(
        command,
        env=launcher.cpu_stage_environment(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "recursive code closure changed" in rejected.stderr
    assert not result_path.exists()


def _mock_stage_runner(plan: dict[str, Any], fail_training: bool = False):
    output = Path(plan["output"])

    def run(command: list[str], *, environment: dict[str, str], log_path: Path) -> dict[str, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("mock CPU subprocess\n", encoding="utf-8")
        assert command[2] == "-c"
        assert command[3] == launcher.ISOLATED_RUNPY_BOOTSTRAP
        script = Path(command[6]).name
        if script == launcher.FREEZER:
            request_path = Path(command[command.index("--output") + 1])
            logical = {
                "format": launcher.REQUEST_FORMAT,
                "status": launcher.REQUEST_STATUS,
                "event_spec": plan["source_inputs"]["event_spec"],
                "actors": [
                    {
                        "actor_name": "smolvla_aloha_agilex",
                        "policy_family": "smolvla",
                        "body": "aloha-agilex",
                        "policy": "smolvla",
                        "state_feature_source_sha256": "f" * 64,
                    }
                ],
                "sources": [
                    {
                        "source_name": "smolvla_source63",
                        "schema_version": 5,
                        "manifest_path": plan["source_inputs"]["schema5_manifest"]["path"],
                        "manifest_file_sha256": plan["source_inputs"]["schema5_manifest"]["file_sha256"],
                        "manifest_logical_sha256": "1" * 64,
                        "group_root": "/srv/source63/groups",
                        "actor_name": "smolvla_aloha_agilex",
                    }
                ],
                "splits": {"train": [{}], "calibration": [{}], "validation": [{}]},
                "split_unit": "logical_reset_group",
                "split_leakage_allowed": False,
                "privileged_label_source_available_to_model_inputs": False,
                "future_query_features_available_to_model_inputs": False,
            }
            request = _signed(logical, "request_sha256")
            _write_json(request_path, request)
            audit_base = {
                "format": "etsf_smolvla_causal_observer_source63_request_freeze_audit_v1",
                "status": launcher.REQUEST_AUDIT_STATUS,
                "request": {
                    "path": str(request_path),
                    "file_sha256": launcher.file_sha256(request_path),
                    "request_sha256": request["request_sha256"],
                },
                "data_access_audit": {
                    "original_test_groups_excluded_from_all_request_splits": True,
                    "original_test_group_files_opened": 0,
                    "original_test_group_files_hashed": 0,
                },
            }
            _write_json(Path(str(request_path) + ".audit.json"), _signed(audit_base, "audit_sha256"))
        elif script == launcher.MATERIALIZER:
            directory = Path(command[command.index("--output-directory") + 1])
            directory.mkdir()
            splits: dict[str, Any] = {}
            for name in ("train", "calibration", "validation"):
                artifact = directory / f"{name}.npz"
                artifact.write_bytes(f"mock-{name}".encode())
                splits[name] = {"path": artifact.name, "file_sha256": launcher.file_sha256(artifact)}
            base = {
                "format": launcher.DATASET_FORMAT,
                "status": launcher.DATASET_STATUS,
                "actor_registry": [{"actor_name": "smolvla_aloha_agilex"}],
                "splits": splits,
            }
            _write_json(directory / "manifest.json", _signed(base, "manifest_sha256"))
        elif script == launcher.TRAINER:
            assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-unit-4090"
            if fail_training:
                raise RuntimeError("mock CUDA training failure")
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir()
            base = {
                "format": launcher.FREEZE_FORMAT,
                "status": launcher.MONITOR_STATUS,
                "real_task_success_or_cross_embodiment_improvement_claimed": False,
            }
            _write_json(directory / "monitor_freeze_manifest.json", _signed(base, "freeze_manifest_sha256"))
        else:
            raise AssertionError(f"unexpected mock stage: {script}")
        return {"returncode": 0, "log_path": str(log_path)}

    return run


def _idle_audit() -> dict[str, Any]:
    base = {
        "format": launcher.GPU_AUDIT_FORMAT,
        "status": "complete_external_gone_and_gpu_idle",
        "idle_confirmations": 2,
    }
    return _signed(base, "audit_sha256")


def test_cpu_mock_pipeline_monitor_only_is_honest_terminal_and_recoverable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "isolated_runpy_prefix",
        lambda plan, name: [
            str(plan["python"]["path"]),
            "-I",
            "-c",
            launcher.ISOLATED_RUNPY_BOOTSTRAP,
            str(Path(plan["output"]) / "static_plan.json"),
            str(plan["static_plan_sha256"]),
            str(plan["entrypoints"][name]["path"]),
        ],
    )
    first = tmp_path / "source63_run"
    plan = _minimal_plan(first)
    launcher.initialize_output(plan)
    monkeypatch.setattr(launcher, "query_gpu_identity", lambda _uuid: plan["gpu_identity"])
    monkeypatch.setattr(launcher, "query_gpu_compute_pids", lambda _uuid: [])
    summary = launcher.execute_plan(
        plan,
        command_runner=_mock_stage_runner(plan),
        idle_waiter=lambda **_kwargs: _idle_audit(),
    )
    assert summary["status"] == launcher.TERMINAL_STATUS
    assert summary["source_embodiment_only"] is True
    assert summary["cross_embodiment_claimed"] is False
    assert summary["target_task_success_claimed"] is False
    assert summary["target_paths_consumed"] is False
    assert summary["training"]["promotion_enabled"] is False
    assert summary["training"]["authority_issued"] is False
    assert not (first / "training" / "authority_manifest.json").exists()
    assert not Path(plan["gpu_lock"]).exists()

    second = tmp_path / "source63_failed_run"
    failed_plan = _minimal_plan(second)
    launcher.initialize_output(failed_plan)
    monkeypatch.setattr(launcher, "query_gpu_identity", lambda _uuid: failed_plan["gpu_identity"])
    with pytest.raises(RuntimeError, match="mock CUDA training failure"):
        launcher.execute_plan(
            failed_plan,
            command_runner=_mock_stage_runner(failed_plan, fail_training=True),
            idle_waiter=lambda **_kwargs: _idle_audit(),
        )
    state = json.loads((second / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == launcher.FAILURE_STATUS
    assert state["recoverable"] is True
    assert state["completed_stages"] == ["request", "dataset"]
    assert state["cross_embodiment_claimed"] is False
    assert not Path(failed_plan["gpu_lock"]).exists()


def test_detach_starts_new_session_and_binds_static_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "source63_detached"
    plan = _minimal_plan(output)
    captured: dict[str, Any] = {}

    class Process:
        pid = 12345

    def fake_popen(command: list[str], **kwargs: Any) -> Process:
        captured["command"] = command
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(launcher, "build_static_plan", lambda _args: plan)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    args = argparse.Namespace(
        code_root=Path("/srv/source63/code"),
        python=Path(sys.executable),
        required_hostname=socket.gethostname(),
        schema5_manifest=Path("/srv/source63/manifest.json"),
        schema5_manifest_sha256="c" * 64,
        frozen_split=Path("/srv/source63/split.json"),
        frozen_split_sha256="d" * 64,
        event_spec=Path("/srv/source63/event_spec.json"),
        event_spec_sha256="e" * 64,
        gpu_uuid="GPU-unit-4090",
        gpu_lock=tmp_path / "gpu.lock",
    )
    receipt = launcher.detach(args)
    assert captured["start_new_session"] is True
    assert captured["stdin"] is launcher.subprocess.DEVNULL
    assert "--expected-static-plan-sha256" in captured["command"]
    assert plan["static_plan_sha256"] in captured["command"]
    assert receipt["cross_embodiment_claimed"] is False
    assert receipt["target_task_success_claimed"] is False
