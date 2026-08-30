from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "guard_robotwin2_postformal_shared_head_upgrade_v1.py"
SPEC = importlib.util.spec_from_file_location("postformal_guardian", SCRIPT)
assert SPEC and SPEC.loader
guardian = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guardian)


def _guardian_argv(
    tmp_path: Path, child_code: str, *child_arguments: Path
) -> list[str]:
    return [
        "--run-exit",
        str(tmp_path / "run.exit"),
        "--watcher-state",
        str(tmp_path / "watcher-state.json"),
        "--state",
        str(tmp_path / "guardian-state.json"),
        "--lock",
        str(tmp_path / "guardian.lock"),
        "--poll-seconds",
        "0.01",
        "--restart-delay-seconds",
        "0",
        "--watcher-argv",
        sys.executable,
        "-c",
        child_code,
        *(str(value) for value in child_arguments),
    ]


def _state(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "guardian-state.json").read_text(encoding="utf-8"))


def test_preexisting_success_is_reused_without_starting_child(tmp_path: Path) -> None:
    (tmp_path / "run.exit").write_text("0\n", encoding="utf-8")
    marker = tmp_path / "child-started"
    result = guardian.main(
        _guardian_argv(
            tmp_path,
            "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
            marker,
        )
    )
    assert result == 0
    assert not marker.exists()
    state = _state(tmp_path)
    assert state["status"] == "complete"
    assert state["terminal_reason"] == "preexisting_run_exit_0"
    assert state["attempts_started"] == 0


def test_credentialed_stage_signal_restarts_then_observes_success(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "attempt-count"
    watcher_state = tmp_path / "watcher-state.json"
    run_exit = tmp_path / "run.exit"
    code = """
from pathlib import Path
import json, signal, sys
counter, state, run_exit = map(Path, sys.argv[1:4])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if count >= 3:
    run_exit.write_text("0\\n")
    raise SystemExit(0)
state.write_text(json.dumps({
    "format": "etsf_robotwin2_postformal_shared_head_upgrade_watcher_v3_actor_protocol",
    "status": "recoverable_child_signal_interruption",
    "error_type": "RecoverableChildSignalInterruption",
    "error_message": "nested interrupted by SIGTERM (15)",
    "child_stage": "nested",
    "child_returncode": -int(signal.SIGTERM),
    "child_signal_number": int(signal.SIGTERM),
    "child_signal_name": "SIGTERM",
    "run_exit_written": False,
    "attempt": count,
}))
raise SystemExit(75)
"""
    result = guardian.main(
        _guardian_argv(tmp_path, code, counter, watcher_state, run_exit)
    )
    assert result == 0
    assert counter.read_text(encoding="utf-8") == "3"
    state = _state(tmp_path)
    assert state["status"] == "complete"
    assert state["unexpected_restart_count"] == 2
    assert state["attempts_started"] == 3


def test_run_exit_one_fails_immediately_without_swallowing_reserve_error(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "attempt-count"
    watcher_state = tmp_path / "watcher-state.json"
    run_exit = tmp_path / "run.exit"
    code = """
from pathlib import Path
import json, sys
counter, state, run_exit = map(Path, sys.argv[1:4])
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
state.write_text(json.dumps({
    "status": "failed",
    "error_type": "ScriptedRootCollectionError",
    "error_message": "ordered reserve exhausted before a complete root pair",
}))
run_exit.write_text("1\\n")
raise SystemExit(1)
"""
    result = guardian.main(
        _guardian_argv(tmp_path, code, counter, watcher_state, run_exit)
    )
    assert result == 1
    assert counter.read_text(encoding="utf-8") == "1"
    state = _state(tmp_path)
    assert state["status"] == "failed"
    assert state["terminal_reason"] == "run_exit_1"
    assert state["unexpected_restart_count"] == 0
    assert state["watcher_failure_error"] == {
        "error_type": "ScriptedRootCollectionError",
        "error_message": "ordered reserve exhausted before a complete root pair",
    }


def test_unrecoverable_error_without_run_exit_fails_on_first_attempt(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "attempt-count"
    watcher_state = tmp_path / "watcher-state.json"
    code = """
from pathlib import Path
import json, sys
counter, state = map(Path, sys.argv[1:3])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
state.write_text(json.dumps({
    "status": "failed",
    "error_type": "SharedHeadUpgradeError",
    "error_message": "immutable authority mismatch",
    "attempt": count,
}))
raise SystemExit(23)
"""
    result = guardian.main(_guardian_argv(tmp_path, code, counter, watcher_state))
    assert result == 1
    assert counter.read_text(encoding="utf-8") == "1"
    state = _state(tmp_path)
    assert state["status"] == "failed"
    assert state["terminal_reason"] == "unrecoverable_child_exit_without_run_exit"
    assert state["unexpected_restart_count"] == 0
    assert state["watcher_failure_error"]["error_message"] == (
        "immutable authority mismatch"
    )


def test_stateless_crash_is_not_mislabeled_as_interruption(tmp_path: Path) -> None:
    counter = tmp_path / "attempt-count"
    code = """
from pathlib import Path
import sys
counter = Path(sys.argv[1])
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
raise SystemExit(31)
"""
    result = guardian.main(_guardian_argv(tmp_path, code, counter))
    assert result == 1
    assert counter.read_text(encoding="utf-8") == "1"
    state = _state(tmp_path)
    assert state["status"] == "failed"
    assert state["terminal_reason"] == "unrecoverable_child_exit_without_run_exit"
    assert state["unexpected_restart_count"] == 0
    assert not list(tmp_path.glob("guardian-state.json.partial-*"))


def test_signal_stopped_watcher_process_is_restarted(tmp_path: Path) -> None:
    counter = tmp_path / "attempt-count"
    run_exit = tmp_path / "run.exit"
    code = """
from pathlib import Path
import os, signal, sys
counter, run_exit = map(Path, sys.argv[1:3])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if count == 1:
    os.kill(os.getpid(), signal.SIGTERM)
run_exit.write_text("0\\n")
"""
    result = guardian.main(_guardian_argv(tmp_path, code, counter, run_exit))
    assert result == 0
    assert counter.read_text(encoding="utf-8") == "2"
    state = _state(tmp_path)
    assert state["unexpected_restart_count"] == 1
    assert state["attempt_history"][0]["interrupted_process"] == "watcher"
    assert state["attempt_history"][0]["interrupted_signal_name"] == "SIGTERM"


def test_shell_style_signal_exit_and_crash_signals_are_not_recoverable(
    tmp_path: Path,
) -> None:
    watcher_state = tmp_path / "watcher-state.json"
    code = """
from pathlib import Path
import json, sys
state = Path(sys.argv[1])
state.write_text(json.dumps({
    "format": "etsf_robotwin2_postformal_shared_head_upgrade_watcher_v3_actor_protocol",
    "status": "recoverable_child_signal_interruption",
    "error_type": "RecoverableChildSignalInterruption",
    "child_stage": "nested",
    "child_returncode": 143,
    "child_signal_number": 15,
    "child_signal_name": "SIGTERM",
    "run_exit_written": False,
}))
raise SystemExit(75)
"""
    result = guardian.main(_guardian_argv(tmp_path, code, watcher_state))
    assert result == 1
    assert _state(tmp_path)["terminal_reason"] == (
        "unrecoverable_child_exit_without_run_exit"
    )
    assert guardian.recoverable_watcher_signal(-11) is None
    assert guardian.recoverable_watcher_signal(143) is None
