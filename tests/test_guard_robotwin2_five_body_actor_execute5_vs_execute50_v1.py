from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "guard_robotwin2_five_body_actor_execute5_vs_execute50_v1.py"
)
SPEC = importlib.util.spec_from_file_location("actor_protocol_guardian", SCRIPT)
assert SPEC and SPEC.loader
guardian = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guardian)


def _args(tmp_path: Path) -> SimpleNamespace:
    code = tmp_path / "code"
    actor = tmp_path / "actor"
    vlm = tmp_path / "vlm"
    robotwin = tmp_path / "robotwin"
    output = tmp_path / "output"
    for directory in (code / "scripts", actor, vlm, robotwin, output):
        directory.mkdir(parents=True, exist_ok=True)
    runner = code / "scripts" / guardian.RUNNER_FILENAME
    runner.write_text("# frozen runner\n", encoding="utf-8")
    event = tmp_path / "event.json"
    event.write_text("{}\n", encoding="utf-8")
    stable_roster = tmp_path / "stable-roster.json"
    stable_roster.write_text("{}\n", encoding="utf-8")
    materializer = code / "scripts" / "materialize_robotwin2_stable_seed_roster_v1.py"
    materializer.write_text("# frozen roster materializer\n", encoding="utf-8")
    return SimpleNamespace(
        runner_pid=123,
        python_bin=Path(sys.executable),
        code_root=code,
        actor_checkpoint=actor,
        vlm_metadata_path=vlm,
        robotwin_root=robotwin,
        event_spec=event,
        stable_seed_roster=stable_roster,
        stable_seed_roster_sha256=guardian.sha256_file(stable_roster),
        output=output,
        state_root=tmp_path / "guardian",
        gpu_uuid="GPU-test",
        nvidia_smi=Path(sys.executable),
        poll_seconds=0.1,
    )


def _write_static_binding(args: SimpleNamespace) -> tuple[str, str]:
    (args.actor_checkpoint / "model.bin").write_bytes(b"actor")
    (args.vlm_metadata_path / "config.json").write_bytes(b"vlm")
    runtime_file = args.robotwin_root / "task.py"
    runtime_file.write_bytes(b"task")
    actor_tree = guardian.sha256_tree(args.actor_checkpoint)
    vlm_tree = guardian.sha256_tree(args.vlm_metadata_path)
    runner = args.code_root / "scripts" / guardian.RUNNER_FILENAME
    base = {
        "format": guardian.BINDING_FORMAT,
        "runner_format": guardian.RUNNER_FORMAT,
        "runner_path": str(runner.resolve()),
        "runner_sha256": guardian.sha256_file(runner),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "actor_checkpoint_tree_sha256": actor_tree[0],
        "actor_checkpoint_file_count": actor_tree[1],
        "actor_checkpoint_size_bytes": actor_tree[2],
        "vlm_metadata_path": str(args.vlm_metadata_path.resolve()),
        "vlm_metadata_tree_sha256": vlm_tree[0],
        "vlm_metadata_file_count": vlm_tree[1],
        "vlm_metadata_size_bytes": vlm_tree[2],
        "robotwin_root": str(args.robotwin_root.resolve()),
        "event_spec": str(args.event_spec.resolve()),
        "event_spec_sha256": guardian.sha256_file(args.event_spec),
        "stable_seed_roster_binding": {
            "path": str(args.stable_seed_roster.resolve()),
            "file_sha256": guardian.sha256_file(args.stable_seed_roster),
            "logical_sha256": "a" * 64,
            "preregistration_file_sha256": "b" * 64,
            "preregistration_logical_sha256": "c" * 64,
            "materializer_path": str(
                (
                    args.code_root
                    / "scripts"
                    / "materialize_robotwin2_stable_seed_roster_v1.py"
                ).resolve()
            ),
            "materializer_file_sha256": guardian.sha256_file(
                args.code_root
                / "scripts"
                / "materialize_robotwin2_stable_seed_roster_v1.py"
            ),
            "selection_uses_labels_or_outcomes": False,
            "actor_inference_calls_during_selection": 0,
        },
        "runtime_binding": {
            "critical_files": [
                {
                    "path": str(runtime_file.resolve()),
                    "sha256": guardian.sha256_file(runtime_file),
                    "size_bytes": runtime_file.stat().st_size,
                }
            ]
        },
    }
    binding = {**base, "logical_sha256": guardian.canonical_sha256(base)}
    path = args.output / "immutable_deployment_binding.json"
    path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    return str(binding["logical_sha256"]), guardian.sha256_file(path)


def _artifact_directories(output: Path) -> None:
    for name in (
        "method_failures",
        "pair_failures",
        "pairs",
        "attempts",
        "initial_commitments",
        "method_starts",
        "method_results",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def test_build_runner_command_is_the_exact_frozen_command(tmp_path: Path) -> None:
    args = _args(tmp_path)
    command = guardian.build_runner_command(args)
    assert command == [
        str(Path(sys.executable).absolute()),
        str((args.code_root / "scripts" / guardian.RUNNER_FILENAME).absolute()),
        "--actor-checkpoint",
        str(args.actor_checkpoint.absolute()),
        "--vlm-metadata-path",
        str(args.vlm_metadata_path.absolute()),
        "--robotwin-root",
        str(args.robotwin_root.absolute()),
        "--event-spec",
        str(args.event_spec.absolute()),
        "--stable-seed-roster",
        str(args.stable_seed_roster.absolute()),
        "--stable-seed-roster-sha256",
        args.stable_seed_roster_sha256,
        "--output",
        str(args.output.absolute()),
    ]


@pytest.mark.parametrize(
    ("kind", "alive", "restarts", "expected"),
    [
        ("failure", True, 0, "fail_experiment_receipt"),
        ("complete", False, 1, "complete"),
        ("running", True, 0, "wait"),
        ("running", False, 0, "restart_once"),
        ("running", False, 1, "fail_restart_exhausted"),
    ],
)
def test_next_action_is_fail_closed_and_allows_only_one_restart(
    kind: str, alive: bool, restarts: int, expected: str
) -> None:
    assert (
        guardian.next_action(
            artifact_kind=kind,
            process_alive=alive,
            restart_count=restarts,
        )
        == expected
    )


def test_staged_method_failure_preempts_completion_or_restart(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _artifact_directories(output)
    failure = output / "method_failures" / ".pair.method.json.staged"
    failure.write_text('{"status":"failed_once_no_retry"}\n', encoding="utf-8")
    observation = guardian.observe_artifacts(
        output,
        binding_logical_sha256="a" * 64,
        binding_file_sha256="b" * 64,
    )
    assert observation["kind"] == "failure"
    assert observation["failure_receipts"] == [str(failure.resolve())]


def test_running_observation_counts_only_final_pairs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _artifact_directories(output)
    for index in range(3):
        (output / "pairs" / f"{index}.json").write_text("{}", encoding="utf-8")
    observation = guardian.observe_artifacts(
        output,
        binding_logical_sha256="a" * 64,
        binding_file_sha256="b" * 64,
    )
    assert observation == {
        "kind": "running",
        "completed_pairs": 3,
        "progress": None,
    }


def test_completion_requires_bound_files_and_exact_artifact_counts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    _artifact_directories(output)
    for directory, count in (
        ("pairs", guardian.PAIR_COUNT),
        ("attempts", guardian.PAIR_COUNT),
        ("initial_commitments", guardian.PAIR_COUNT),
        ("method_starts", guardian.ROLLOUT_COUNT),
        ("method_results", guardian.ROLLOUT_COUNT),
    ):
        for index in range(count):
            (output / directory / f"{index:03d}.json").write_text(
                "{}", encoding="utf-8"
            )
    outcome = output / "paired_outcomes.json"
    report = output / "paired_report.json"
    outcome.write_text('{"outcome":true}\n', encoding="utf-8")
    report.write_text('{"report":true}\n', encoding="utf-8")
    binding_logical = "a" * 64
    binding_file = "b" * 64
    base = {
        "format": guardian.COMPLETION_FORMAT,
        "status": guardian.COMPLETION_STATUS,
        "binding_logical_sha256": binding_logical,
        "binding_file_sha256": binding_file,
        "outcome_document_sha256": "c" * 64,
        "outcome_file_sha256": guardian.sha256_file(outcome),
        "report_sha256": "d" * 64,
        "report_file_sha256": guardian.sha256_file(report),
        "pair_count": guardian.PAIR_COUNT,
        "rollout_count": guardian.ROLLOUT_COUNT,
    }
    completion = {**base, "logical_sha256": guardian.canonical_sha256(base)}
    (output / "run.complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    assert guardian.validate_completion(
        output,
        binding_logical_sha256=binding_logical,
        binding_file_sha256=binding_file,
    ) == completion
    (output / "method_results" / "000.json").unlink()
    with pytest.raises(guardian.GuardianContractError, match="method_results count"):
        guardian.validate_completion(
            output,
            binding_logical_sha256=binding_logical,
            binding_file_sha256=binding_file,
        )


def test_static_validation_authenticates_models_event_runner_and_runtime(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    binding_logical, binding_file = _write_static_binding(args)
    contract = guardian.validate_binding_and_static_paths(args)
    assert contract["binding_logical_sha256"] == binding_logical
    assert contract["binding_file_sha256"] == binding_file
    assert contract["authenticated_runtime_file_count"] == 1
    (args.robotwin_root / "task.py").write_bytes(b"mutated")
    with pytest.raises(guardian.GuardianContractError, match="runtime binding"):
        guardian.validate_binding_and_static_paths(args)


def test_process_identity_binds_command_cwd_start_time_and_gpu(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "sleep.py"
    helper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    command = [sys.executable, str(helper)]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    process = subprocess.Popen(command, cwd=tmp_path, env=environment)
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                start, restart_environment = guardian.validate_process_identity(
                    process.pid,
                    expected_start_ticks=None,
                    command=command,
                    cwd=tmp_path,
                    gpu_contract={"physical_index": "0", "uuid": "GPU-test"},
                )
                break
            except ProcessLookupError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        assert start > 0
        assert restart_environment["CUDA_VISIBLE_DEVICES"] == "0"
        with pytest.raises(guardian.GuardianContractError, match="command differs"):
            guardian.validate_process_identity(
                process.pid,
                expected_start_ticks=start,
                command=[sys.executable, str(helper), "extra"],
                cwd=tmp_path,
                gpu_contract={"physical_index": "0", "uuid": "GPU-test"},
            )
        with pytest.raises(ProcessLookupError):
            guardian.validate_process_identity(
                process.pid,
                expected_start_ticks=start + 1,
                command=command,
                cwd=tmp_path,
                gpu_contract={"physical_index": "0", "uuid": "GPU-test"},
            )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_gpu_query_requires_exact_uuid_and_4090(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class Result:
        returncode = 0
        stdout = "0, GPU-right, NVIDIA GeForce RTX 4090 D\n"
        stderr = ""

    monkeypatch.setattr(guardian.subprocess, "run", lambda *args, **kwargs: Result())
    value = guardian.query_gpu_contract(Path(sys.executable), "GPU-right")
    assert value["physical_index"] == "0"
    assert value["uuid"] == "GPU-right"
    with pytest.raises(guardian.GuardianContractError, match="absent or ambiguous"):
        guardian.query_gpu_contract(Path(sys.executable), "GPU-wrong")


def test_atomic_guardian_receipts_never_write_experiment_tree(tmp_path: Path) -> None:
    state = tmp_path / "state"
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    guardian.atomic_json(state / "guardian_state.json", {"status": "one"})
    guardian.atomic_json(state / "guardian_state.json", {"status": "two"})
    assert json.loads((state / "guardian_state.json").read_text()) == {
        "status": "two"
    }
    assert list(experiment.iterdir()) == []


def test_main_rejects_state_nested_in_experiment_before_any_write(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    nested = args.output / "guardian_state"
    argv = [
        "--runner-pid",
        str(args.runner_pid),
        "--python-bin",
        str(args.python_bin),
        "--code-root",
        str(args.code_root),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--vlm-metadata-path",
        str(args.vlm_metadata_path),
        "--robotwin-root",
        str(args.robotwin_root),
        "--event-spec",
        str(args.event_spec),
        "--stable-seed-roster",
        str(args.stable_seed_roster),
        "--stable-seed-roster-sha256",
        args.stable_seed_roster_sha256,
        "--output",
        str(args.output),
        "--state-root",
        str(nested),
        "--gpu-uuid",
        args.gpu_uuid,
    ]
    assert guardian.main(argv) == 3
    assert not nested.exists()
