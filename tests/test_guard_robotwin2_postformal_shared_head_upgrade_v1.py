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


def test_unexpected_exits_restart_then_observe_durable_success(tmp_path: Path) -> None:
    counter = tmp_path / "attempt-count"
    run_exit = tmp_path / "run.exit"
    code = """
from pathlib import Path
import sys
counter = Path(sys.argv[1])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if count >= 3:
    Path(sys.argv[2]).write_text("0\\n")
raise SystemExit(0 if count >= 3 else 17)
"""
    result = guardian.main(_guardian_argv(tmp_path, code, counter, run_exit))
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


def test_same_new_unrecoverable_error_three_times_fails_closed(tmp_path: Path) -> None:
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
    assert counter.read_text(encoding="utf-8") == "3"
    state = _state(tmp_path)
    assert state["status"] == "failed"
    assert state["terminal_reason"] == "same_unrecoverable_error_repeated"
    assert state["same_error_consecutive_count"] == 3
    assert state["unexpected_restart_count"] == 2


def test_stateless_crash_never_exceeds_restart_budget(tmp_path: Path) -> None:
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
    assert counter.read_text(encoding="utf-8") == str(
        guardian.MAX_UNEXPECTED_RESTARTS + 1
    )
    state = _state(tmp_path)
    assert state["status"] == "failed"
    assert state["terminal_reason"] == "unexpected_restart_limit_exhausted"
    assert state["unexpected_restart_count"] == guardian.MAX_UNEXPECTED_RESTARTS
    assert not list(tmp_path.glob("guardian-state.json.partial-*"))
