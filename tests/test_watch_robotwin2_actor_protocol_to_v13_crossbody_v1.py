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
    for body in watcher.actor_execution.BODIES:
        for condition in watcher.actor_execution.CONDITIONS:
            for seed in range(20):
                rows.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": 2026104000 + seed,
                        "rollouts": {
                            methods[0]: {
                                "binary_success": 0,
                                "stage_progress": 0.25,
                                "goal_progress_m": 0.0,
                            },
                            methods[1]: {
                                "binary_success": 1,
                                "stage_progress": 1.0,
                                "goal_progress_m": 0.1,
                            },
                        },
                    }
                )
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
