from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bootstrap_robotwin2_stable_roster_to_actor_v25_v1.py"
SPEC = importlib.util.spec_from_file_location("actor_v25_bootstrap", SCRIPT)
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def _roster() -> dict[str, object]:
    seeds = list(range(2026104000, 2026104020))
    attempts = []
    cells = [
        (body, condition)
        for body in bootstrap.EXPECTED_BODIES
        for condition in bootstrap.EXPECTED_CONDITIONS
    ]
    for seed in seeds:
        attempts.append(
            {
                "candidate_seed": seed,
                "cells": [
                    {
                        "cell_index": index,
                        "body": body,
                        "condition": condition,
                        "status": "setup_succeeded_and_closed",
                    }
                    for index, (body, condition) in enumerate(cells)
                ],
                "all_ten_setup_cells_stable": True,
                "actor_inference_calls": 0,
                "task_action_calls": 0,
                "label_or_outcome_reads": 0,
            }
        )
    unsigned = {
        "format": bootstrap.ROSTER_FORMAT,
        "status": "complete_first_twenty_common_stable_seeds",
        "body_order": list(bootstrap.EXPECTED_BODIES),
        "condition_order": list(bootstrap.EXPECTED_CONDITIONS),
        "preregistration_file_sha256": "a" * 64,
        "preregistration_logical_sha256": "b" * 64,
        "selected_seeds": seeds,
        "attempts": attempts,
        "candidate_attempt_count": 20,
        "stable_candidate_count_observed": 20,
        "pair_count": 200,
        "rollout_count_for_two_methods": 400,
        "actor_inference_calls": 0,
        "task_action_calls": 0,
        "label_or_outcome_reads": 0,
    }
    return {**unsigned, "logical_sha256": bootstrap.canonical_sha256(unsigned)}


def _runner_source() -> str:
    return """\
import hashlib, json, os, sys, time
from pathlib import Path
def arg(name): return sys.argv[sys.argv.index(name) + 1]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()
output=Path(arg('--output'))
output.mkdir(parents=True, exist_ok=False)
roster=Path(arg('--stable-seed-roster')).resolve()
base={
 'format':'etsf_robotwin2_actor_deployment_protocol_binding_v2_stable_roster',
 'runner_format':'etsf_robotwin2_five_body_actor_execute5_vs_execute50_v2_stable_roster',
 'runner_path':str(Path(__file__).resolve()),
 'runner_sha256':sha(__file__),
 'stable_seed_roster_binding':{
   'path':str(roster),
   'file_sha256':arg('--stable-seed-roster-sha256'),
   'selection_uses_labels_or_outcomes':False,
   'actor_inference_calls_during_selection':0,
 },
}
value={**base,'logical_sha256':canonical(base)}
(output/'immutable_deployment_binding.json').write_text(json.dumps(value,sort_keys=True)+'\\n')
time.sleep(30)
"""


def _guardian_source(*, runner_process_alive: bool = True) -> str:
    alive = "True" if runner_process_alive else "False"
    return f"""\
import hashlib, json, os, sys, time
from pathlib import Path
def arg(name): return sys.argv[sys.argv.index(name) + 1]
def canonical(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()
def publish(path, value):
    temporary=path.with_name('.'+path.name+'.temporary')
    temporary.write_text(json.dumps(value,sort_keys=True)+'\\n')
    os.replace(temporary,path)
runner_pid=int(arg('--runner-pid'))
state_root=Path(arg('--state-root'))
state_root.mkdir(parents=True,exist_ok=False)
base={{
 'format':'etsf_robotwin2_actor_execute5_vs_execute50_guardian_plan_v1',
 'guardian_format':'etsf_robotwin2_actor_execute5_vs_execute50_guardian_v1',
 'initial_runner_pid':runner_pid,
}}
plan={{**base,'logical_sha256':canonical(base)}}
publish(state_root/'immutable_guardian_plan.json',plan)
state={{
 'format':'etsf_robotwin2_actor_execute5_vs_execute50_guardian_state_v1',
 'status':'monitoring',
 'guardian_pid':os.getpid(),
 'managed_runner_pid':runner_pid,
 'runner_process_alive':{alive},
}}
publish(state_root/'guardian_state.json',state)
time.sleep(30)
"""


def _fixture(
    tmp_path: Path,
    *,
    runner_source: str | None = None,
    guardian_source: str | None = None,
    write_roster: bool = True,
    roster_probe_pid: int | None = None,
) -> tuple[list[str], dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    code_root = tmp_path / "v25"
    code_root.mkdir()
    runner = code_root / "runner.py"
    guardian = code_root / "guardian.py"
    runner.write_text(runner_source or _runner_source(), encoding="utf-8")
    guardian.write_text(guardian_source or _guardian_source(), encoding="utf-8")
    for directory in (
        tmp_path / "actor",
        tmp_path / "vlm",
        tmp_path / "robotwin",
        tmp_path / "runner-cwd",
        tmp_path / "guardian-cwd",
    ):
        directory.mkdir()
    event = tmp_path / "event.json"
    event.write_text("{}\n", encoding="utf-8")
    roster_path = tmp_path / "stable-roster.json"
    if write_roster:
        roster_path.write_text(
            json.dumps(_roster(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "experiment"
    bootstrap_root = tmp_path / "bootstrap"
    guardian_state = tmp_path / "guardian-state"
    argv = [
        "--runner-python",
        sys.executable,
        "--guardian-python",
        sys.executable,
        "--code-root",
        str(code_root),
        "--runner",
        str(runner),
        "--runner-sha256",
        bootstrap.sha256_file(runner),
        "--guardian",
        str(guardian),
        "--guardian-sha256",
        bootstrap.sha256_file(guardian),
        "--runner-cwd",
        str(tmp_path / "runner-cwd"),
        "--guardian-cwd",
        str(tmp_path / "guardian-cwd"),
        "--actor-checkpoint",
        str(tmp_path / "actor"),
        "--vlm-metadata-path",
        str(tmp_path / "vlm"),
        "--robotwin-root",
        str(tmp_path / "robotwin"),
        "--event-spec",
        str(event),
        "--stable-seed-roster",
        str(roster_path),
        "--roster-probe-pid",
        str(roster_probe_pid if roster_probe_pid is not None else os.getpid()),
        "--expected-preregistration-file-sha256",
        "a" * 64,
        "--output",
        str(output),
        "--runner-binding",
        str(output / "immutable_deployment_binding.json"),
        "--guardian-state-root",
        str(guardian_state),
        "--bootstrap-root",
        str(bootstrap_root),
        "--gpu-uuid",
        "GPU-test",
        "--cuda-visible-devices",
        "0",
        "--child-pythonpath",
        str(code_root),
        "--vulkan-driver-files",
        str(event),
        "--nvidia-smi",
        "/bin/true",
        "--poll-seconds",
        "0.01",
        "--guardian-poll-seconds",
        "0.1",
        "--guardian-handoff-timeout-seconds",
        "1.0",
    ]
    return argv, {
        "output": output,
        "bootstrap": bootstrap_root,
        "guardian_state": guardian_state,
        "runner": runner,
        "roster": roster_path,
    }


def _terminate_session(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            waited, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if waited == pid:
            return
        time.sleep(0.01)
    os.killpg(pid, signal.SIGKILL)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _assert_pid_absent(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _failure(paths: dict[str, Path]) -> dict[str, object]:
    path = paths["bootstrap"] / "bootstrap.failure.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    bootstrap.verify_logical_sha(value, "bootstrap failure")
    return value


def test_roster_validation_rejects_outcome_or_reordered_selection() -> None:
    value = _roster()
    assert bootstrap.validate_roster(
        value, expected_preregistration_file_sha256="a" * 64
    )["selected_seeds"][0] == 2026104000
    value["actor_inference_calls"] = 1
    unsigned = dict(value)
    unsigned.pop("logical_sha256")
    value["logical_sha256"] = bootstrap.canonical_sha256(unsigned)
    with pytest.raises(bootstrap.ActorV25BootstrapError, match="header changed"):
        bootstrap.validate_roster(
            value, expected_preregistration_file_sha256="a" * 64
        )


def test_existing_output_is_rejected_before_bootstrap_root_creation(
    tmp_path: Path,
) -> None:
    argv, paths = _fixture(tmp_path)
    paths["output"].mkdir()
    with pytest.raises(bootstrap.ActorV25BootstrapError, match="never resumes"):
        bootstrap.main(argv)
    assert not paths["bootstrap"].exists()


def test_full_bootstrap_detaches_runner_then_guardian_and_exits(
    tmp_path: Path,
) -> None:
    completed_probe = subprocess.Popen([sys.executable, "-c", "pass"])
    completed_probe.wait(timeout=3)
    argv, paths = _fixture(tmp_path, roster_probe_pid=completed_probe.pid)
    runner_pid = guardian_pid = None
    try:
        assert bootstrap.main(argv) == 0
        runner_pid = int((paths["bootstrap"] / "runner.pid").read_text())
        guardian_pid = int((paths["bootstrap"] / "guardian.pid").read_text())
        assert runner_pid != guardian_pid
        assert os.getsid(runner_pid) == runner_pid
        assert os.getsid(guardian_pid) == guardian_pid
        for name in (
            "bootstrap.plan.json",
            "runner.launch.json",
            "guardian.launch.json",
            "bootstrap.complete.json",
        ):
            receipt = paths["bootstrap"] / name
            value = json.loads(receipt.read_text(encoding="utf-8"))
            bootstrap.verify_logical_sha(value, name)
            assert receipt.stat().st_mode & 0o222 == 0
        assert (paths["bootstrap"] / "runner.log").is_file()
        assert (paths["bootstrap"] / "guardian.log").is_file()
        guardian_launch = json.loads(
            (paths["bootstrap"] / "guardian.launch.json").read_text()
        )
        command = guardian_launch["command"]
        assert command[command.index("--runner-pid") + 1] == str(runner_pid)
        assert command[
            command.index("--stable-seed-roster-sha256") + 1
        ] == bootstrap.sha256_file(paths["roster"])
        assert guardian_launch["actor_results_or_labels_read"] is False
        guardian_plan = json.loads(
            (
                paths["guardian_state"] / "immutable_guardian_plan.json"
            ).read_text()
        )
        bootstrap.verify_logical_sha(guardian_plan, "guardian plan")
        guardian_state = json.loads(
            (paths["guardian_state"] / "guardian_state.json").read_text()
        )
        assert guardian_state["status"] == "monitoring"
        assert guardian_state["guardian_pid"] == guardian_pid
        assert guardian_state["managed_runner_pid"] == runner_pid
        assert guardian_state["runner_process_alive"] is True
        complete = json.loads(
            (paths["bootstrap"] / "bootstrap.complete.json").read_text()
        )
        assert complete["guardian_plan_logical_sha256"] == guardian_plan[
            "logical_sha256"
        ]
        assert complete["guardian_state_status_at_handoff"] == "monitoring"
    finally:
        if guardian_pid is not None:
            _terminate_session(guardian_pid)
        if runner_pid is not None:
            _terminate_session(runner_pid)


def test_static_runner_sha_and_binding_path_are_fail_closed(tmp_path: Path) -> None:
    argv, paths = _fixture(tmp_path)
    argv[argv.index("--runner-sha256") + 1] = "f" * 64
    with pytest.raises(bootstrap.ActorV25BootstrapError, match="runner SHA"):
        bootstrap.main(argv)
    assert not paths["bootstrap"].exists()

    argv, paths = _fixture(tmp_path / "second")
    argv[argv.index("--runner-binding") + 1] = str(tmp_path / "elsewhere.json")
    with pytest.raises(bootstrap.ActorV25BootstrapError, match="must be output"):
        bootstrap.main(argv)


def test_dead_probe_fails_before_runner_when_roster_is_absent(
    tmp_path: Path,
) -> None:
    probe = subprocess.Popen([sys.executable, "-c", "pass"])
    probe.wait(timeout=3)
    argv, paths = _fixture(
        tmp_path,
        write_roster=False,
        roster_probe_pid=probe.pid,
    )

    with pytest.raises(
        bootstrap.ActorV25BootstrapError,
        match="probe exited before publishing",
    ):
        bootstrap.main(argv)

    assert not (paths["bootstrap"] / "runner.pid").exists()
    assert not paths["output"].exists()
    plan = json.loads(
        (paths["bootstrap"] / "bootstrap.plan.json").read_text(encoding="utf-8")
    )
    assert plan["roster_probe_pid"] == probe.pid
    failure = _failure(paths)
    assert failure["owned_child_cleanup_complete"] is True
    assert failure["orphaned_owned_children"] is False


def test_bad_binding_terminates_and_reaps_unguarded_runner(
    tmp_path: Path,
) -> None:
    bad_runner = _runner_source().replace(
        "etsf_robotwin2_five_body_actor_execute5_vs_execute50_v2_stable_roster",
        "invalid_runner_format",
    )
    argv, paths = _fixture(tmp_path, runner_source=bad_runner)
    runner_pid: int | None = None
    try:
        with pytest.raises(
            bootstrap.ActorV25BootstrapError,
            match="immutable deployment binding changed",
        ):
            bootstrap.main(argv)
        runner_pid = int((paths["bootstrap"] / "runner.pid").read_text())
        _assert_pid_absent(runner_pid)
        assert not (paths["bootstrap"] / "guardian.pid").exists()
        failure = _failure(paths)
        cleanup = {item["label"]: item for item in failure["cleanup_audit"]}
        assert cleanup["guardian"]["action"] == "none"
        assert cleanup["runner"]["action"] == "sigterm_and_reaped"
        assert failure["owned_child_cleanup_complete"] is True
        assert failure["orphaned_owned_children"] is False
    finally:
        if runner_pid is not None and bootstrap.process_exists(runner_pid):
            _terminate_session(runner_pid)


def test_guardian_early_exit_reaps_guardian_and_terminates_runner(
    tmp_path: Path,
) -> None:
    guardian_source = (
        "import time\n"
        "time.sleep(0.05)\n"
        "raise SystemExit(7)\n"
    )
    argv, paths = _fixture(tmp_path, guardian_source=guardian_source)
    runner_pid = guardian_pid = None
    try:
        with pytest.raises(
            bootstrap.ActorV25BootstrapError,
            match="guardian exited during authoritative handoff",
        ):
            bootstrap.main(argv)
        runner_pid = int((paths["bootstrap"] / "runner.pid").read_text())
        guardian_pid = int((paths["bootstrap"] / "guardian.pid").read_text())
        _assert_pid_absent(guardian_pid)
        _assert_pid_absent(runner_pid)
        failure = _failure(paths)
        cleanup = {item["label"]: item for item in failure["cleanup_audit"]}
        assert cleanup["guardian"]["action"] == "already_exited_and_reaped"
        assert cleanup["guardian"]["returncode"] == 7
        assert cleanup["runner"]["action"] == "sigterm_and_reaped"
        assert failure["owned_child_cleanup_complete"] is True
        assert failure["orphaned_owned_children"] is False
    finally:
        if guardian_pid is not None and bootstrap.process_exists(guardian_pid):
            _terminate_session(guardian_pid)
        if runner_pid is not None and bootstrap.process_exists(runner_pid):
            _terminate_session(runner_pid)


def test_false_guardian_monitoring_state_terminates_both_children(
    tmp_path: Path,
) -> None:
    argv, paths = _fixture(
        tmp_path,
        guardian_source=_guardian_source(runner_process_alive=False),
    )
    runner_pid = guardian_pid = None
    try:
        with pytest.raises(
            bootstrap.ActorV25BootstrapError,
            match="guardian monitoring state changed during handoff",
        ):
            bootstrap.main(argv)
        runner_pid = int((paths["bootstrap"] / "runner.pid").read_text())
        guardian_pid = int((paths["bootstrap"] / "guardian.pid").read_text())
        _assert_pid_absent(guardian_pid)
        _assert_pid_absent(runner_pid)
        failure = _failure(paths)
        cleanup = {item["label"]: item for item in failure["cleanup_audit"]}
        assert cleanup["guardian"]["action"] == "sigterm_and_reaped"
        assert cleanup["runner"]["action"] == "sigterm_and_reaped"
        assert failure["owned_child_cleanup_complete"] is True
        assert failure["orphaned_owned_children"] is False
    finally:
        if guardian_pid is not None and bootstrap.process_exists(guardian_pid):
            _terminate_session(guardian_pid)
        if runner_pid is not None and bootstrap.process_exists(runner_pid):
            _terminate_session(runner_pid)
