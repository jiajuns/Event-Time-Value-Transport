from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "actor_protocol_to_v13",
    SCRIPTS / "watch_robotwin2_actor_protocol_to_v13_crossbody_v1.py",
)
assert SPEC and SPEC.loader
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def _signed(value: dict[str, object], field: str) -> dict[str, object]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return {**unsigned, field: watcher.canonical_sha256(unsigned)}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _actor_completion(root: Path, guardian_state: Path) -> None:
    methods = (
        watcher.actor_execution.METHOD_EXECUTE5,
        watcher.actor_execution.METHOD_EXECUTE50,
    )
    rows = []
    for body_index, body in enumerate(watcher.actor_execution.BODIES):
        for condition_index, condition in enumerate(
            watcher.actor_execution.CONDITIONS
        ):
            for seed_index in range(20):
                order = list(
                    methods
                    if (body_index + condition_index + seed_index) % 2 == 0
                    else reversed(methods)
                )
                row = {
                    "benchmark": watcher.ACTOR_BENCHMARK,
                    "task": watcher.actor_execution.TASK,
                    "heldout_body": body,
                    "condition": condition,
                    "requested_seed": 2026104000 + seed_index,
                    "method_order": order,
                    "pair_sha256": f"{len(rows) + 1:064x}",
                }
                for method, success, stage, progress in (
                    (methods[0], 0, 0.25, 0.0),
                    (methods[1], 1, 1.0, 0.1),
                ):
                    row[f"{method}_binary_success"] = success
                    row[f"{method}_stage_progress"] = stage
                    row[f"{method}_terminal_goal_distance_m"] = 0.5
                    row[f"{method}_goal_progress_m"] = progress
                    row[f"{method}_live_first_token_effect14_rms_mean"] = 0.01
                    row[f"{method}_command_boundary_effect14_rms_mean"] = None
                rows.append(row)
    binding = _signed(
        {"format": watcher.ACTOR_BINDING_FORMAT, "frozen": True},
        "logical_sha256",
    )
    binding_path = root / "immutable_deployment_binding.json"
    _write(binding_path, binding)
    outcome = _signed(
        {
            "format": watcher.ACTOR_OUTCOME_FORMAT,
            "status": "complete_200_pairs_400_rollouts",
            "rows": rows,
            "rows_sha256": watcher.canonical_sha256(rows),
        },
        "document_sha256",
    )
    outcome_path = root / "paired_outcomes.json"
    _write(outcome_path, outcome)
    selection = {
        "ordered_criteria": list(watcher.ORDERED_SELECTION_CRITERIA),
        "rule": (
            "compare paired success first; only if exactly tied compare paired "
            "stage progress; only if still tied compare paired goal progress"
        ),
        "selected_criterion": "binary_success",
        "selected_delta_execute50_minus_execute5": 1.0,
        "preferred_protocol": methods[1],
    }
    report = _signed(
        {
            "format": watcher.ACTOR_REPORT_FORMAT,
            "status": watcher.ACTOR_REPORT_STATUS,
            "outcome_document_sha256": outcome["document_sha256"],
            "primary_hierarchical_selection": selection,
        },
        "report_sha256",
    )
    report_path = root / "paired_report.json"
    _write(report_path, report)
    completion = _signed(
        {
            "format": watcher.ACTOR_COMPLETION_FORMAT,
            "status": watcher.ACTOR_COMPLETION_STATUS,
            "pair_count": 200,
            "rollout_count": 400,
            "binding_logical_sha256": binding["logical_sha256"],
            "binding_file_sha256": watcher.sha256_file(binding_path),
            "outcome_document_sha256": outcome["document_sha256"],
            "outcome_file_sha256": watcher.sha256_file(outcome_path),
            "report_sha256": report["report_sha256"],
            "report_file_sha256": watcher.sha256_file(report_path),
        },
        "logical_sha256",
    )
    _write(root / "run.complete.json", completion)
    _write(guardian_state, {"status": "complete"})


def _resign_actor_chain(root: Path) -> None:
    outcome_path = root / "paired_outcomes.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["rows_sha256"] = watcher.canonical_sha256(outcome["rows"])
    outcome = _signed(outcome, "document_sha256")
    _write(outcome_path, outcome)

    report_path = root / "paired_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outcome_document_sha256"] = outcome["document_sha256"]
    report = _signed(report, "report_sha256")
    _write(report_path, report)

    completion_path = root / "run.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["outcome_document_sha256"] = outcome["document_sha256"]
    completion["outcome_file_sha256"] = watcher.sha256_file(outcome_path)
    completion["report_sha256"] = report["report_sha256"]
    completion["report_file_sha256"] = watcher.sha256_file(report_path)
    completion = _signed(completion, "logical_sha256")
    _write(completion_path, completion)


def test_completion_replays_hierarchical_protocol_selection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "actor"
    root.mkdir()
    guardian = tmp_path / "guardian.json"
    _actor_completion(root, guardian)
    evidence = watcher.validate_actor_completion(root, guardian)
    assert evidence is not None
    assert evidence["primary_hierarchical_selection"]["preferred_protocol"] == (
        watcher.actor_execution.METHOD_EXECUTE50
    )


@pytest.mark.parametrize("tamper", ["legacy_nested_rollouts", "missing_flat_metric"])
def test_completion_rejects_nonproduction_outcome_row_abi(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / "actor"
    root.mkdir()
    guardian = tmp_path / "guardian.json"
    _actor_completion(root, guardian)
    outcome_path = root / "paired_outcomes.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    row = outcome["rows"][0]
    method = watcher.actor_execution.METHOD_EXECUTE5
    if tamper == "legacy_nested_rollouts":
        row["rollouts"] = {
            method: {
                metric: row[f"{method}_{metric}"]
                for metric in watcher.ORDERED_SELECTION_CRITERIA
            }
        }
    del row[f"{method}_binary_success"]
    _write(outcome_path, outcome)
    _resign_actor_chain(root)
    with pytest.raises(
        watcher.CrossbodyContinuationError,
        match="actor outcome identity/method changed",
    ):
        watcher.validate_actor_completion(root, guardian)


def test_completion_rejects_resigned_flat_metric_selection_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "actor"
    root.mkdir()
    guardian = tmp_path / "guardian.json"
    _actor_completion(root, guardian)
    outcome_path = root / "paired_outcomes.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    method = watcher.actor_execution.METHOD_EXECUTE50
    outcome["rows"][0][f"{method}_binary_success"] = 0
    _write(outcome_path, outcome)
    _resign_actor_chain(root)
    with pytest.raises(
        watcher.CrossbodyContinuationError,
        match="does not replay from complete outcomes",
    ):
        watcher.validate_actor_completion(root, guardian)


def test_completion_rejects_invalid_method_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "actor"
    root.mkdir()
    guardian = tmp_path / "guardian.json"
    _actor_completion(root, guardian)
    outcome_path = root / "paired_outcomes.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    method = watcher.actor_execution.METHOD_EXECUTE5
    outcome["rows"][0]["method_order"] = [method, method]
    _write(outcome_path, outcome)
    _resign_actor_chain(root)
    with pytest.raises(
        watcher.CrossbodyContinuationError,
        match="actor outcome identity/method changed",
    ):
        watcher.validate_actor_completion(root, guardian)


def test_selected_protocol_is_create_once_and_tie_fails_closed(
    tmp_path: Path,
) -> None:
    path_root = tmp_path
    run_root = tmp_path / "run"
    run_root.mkdir()
    evidence = {
        "primary_hierarchical_selection": {
            "preferred_protocol": watcher.actor_execution.METHOD_EXECUTE50
        }
    }
    path, file_sha, receipt = watcher.materialize_selected_protocol(
        evidence,
        run_root=run_root,
        path_root=path_root,
    )
    assert path.is_file()
    assert receipt["selected_stride"] == 50
    assert receipt["actor_execution_protocol_binding"]["file_sha256"] == file_sha
    watcher.materialize_selected_protocol(
        evidence,
        run_root=run_root,
        path_root=path_root,
    )
    with pytest.raises(watcher.CrossbodyContinuationError, match="tied"):
        watcher.materialize_selected_protocol(
            {"primary_hierarchical_selection": {"preferred_protocol": "tie"}},
            run_root=run_root,
            path_root=path_root,
        )


def test_commands_propagate_one_protocol_through_complete_chain(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        robotwin_python=tmp_path / "robotwin-python",
        system_python=tmp_path / "python3",
        code_root=tmp_path / "code",
        path_root=tmp_path,
        run_root=tmp_path / "run",
        code_commit_marker="a" * 40,
        actor_checkpoint=tmp_path / "actor",
        materialization_receipt=tmp_path / "materialization.json",
        robotwin_root=tmp_path / "robotwin",
        vlm_metadata=tmp_path / "vlm",
        event_spec=tmp_path / "event.json",
        metrics_preregistration=tmp_path / "metrics.json",
    )
    protocol_path = args.run_root / "actor_execution_protocol.json"
    protocol_sha = "b" * 64
    primary = watcher.primary_command(args, protocol_path, protocol_sha)
    downstream = watcher.downstream_command(args, protocol_path, protocol_sha)
    final_report = watcher.final_report_command(args)
    for command in (primary, downstream):
        assert command[command.index("--actor-execution-protocol") + 1] == str(
            protocol_path
        )
        assert command[command.index("--actor-execution-protocol-sha256") + 1] == (
            protocol_sha
        )
        assert command[command.index("--path-root") + 1] == str(tmp_path)
    assert downstream[downstream.index("--upstream-kind") + 1] == (
        "primary_collection"
    )
    assert final_report[1] == str(args.code_root / watcher.FINAL_REPORT_MATERIALIZER)
    assert final_report[final_report.index("--nested-root") + 1] == str(
        args.run_root / "nested_n1_n4_n8"
    )
    assert final_report[final_report.index("--actor-authority") + 1] == str(
        args.run_root / "actor_authority.json"
    )
    assert final_report[final_report.index("--output-report") + 1] == str(
        args.run_root / "nested_n1_n4_n8" / "crossbody_final_report.json"
    )
