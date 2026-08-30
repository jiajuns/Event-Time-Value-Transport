from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "shared_head_upgrade_watcher",
    SCRIPTS / "watch_robotwin2_postformal_shared_head_upgrade_v1.py",
)
assert SPEC and SPEC.loader
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def test_stage_returncode_only_authorizes_direct_interruption_signals(
    tmp_path: Path,
) -> None:
    with pytest.raises(watcher.RecoverableChildSignalInterruption) as raised:
        watcher.raise_for_stage_returncode("nested", -int(signal.SIGTERM))
    error = raised.value
    assert error.stage == "nested"
    assert error.child_returncode == -int(signal.SIGTERM)
    assert error.signal_name == "SIGTERM"

    for returncode in (1, 143, -int(signal.SIGSEGV)):
        with pytest.raises(watcher.SharedHeadUpgradeError) as ordinary:
            watcher.raise_for_stage_returncode("nested", returncode)
        assert type(ordinary.value) is watcher.SharedHeadUpgradeError

    state = tmp_path / "watcher-state.json"
    frozen_code_fields = {
        "code_commit_marker": "a" * 40,
        "code_manifest": str(tmp_path / "code-manifest.json"),
        "code_manifest_logical_sha256": "b" * 64,
        "code_manifest_file_sha256": "c" * 64,
    }
    state.write_text(json.dumps(frozen_code_fields), encoding="utf-8")
    watcher.record_recoverable_interruption(state, error)
    document = json.loads(state.read_text(encoding="utf-8"))
    assert document["status"] == watcher.RECOVERABLE_INTERRUPTION_STATUS
    assert document["child_returncode"] == -int(signal.SIGTERM)
    assert document["run_exit_written"] is False
    assert all(document[key] == value for key, value in frozen_code_fields.items())
    run_exit = tmp_path / "run.exit"
    assert not run_exit.exists()

    failure_state = tmp_path / "failure-state.json"
    failure_state.write_text(json.dumps(frozen_code_fields), encoding="utf-8")
    watcher.record_failure(
        failure_state,
        run_exit,
        watcher.SharedHeadUpgradeError("nested exited 1"),
    )
    assert run_exit.read_text(encoding="utf-8") == "1\n"
    assert json.loads(failure_state.read_text(encoding="utf-8"))["status"] == (
        "failed"
    )
    failure_document = json.loads(failure_state.read_text(encoding="utf-8"))
    assert all(
        failure_document[key] == value
        for key, value in frozen_code_fields.items()
    )


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        robotwin_python=tmp_path / "robotwin-python",
        training_python=tmp_path / "training-python",
        system_python=tmp_path / "python3",
        code_root=tmp_path / "code",
        actor_checkpoint=tmp_path / "actor",
        actor_authority=tmp_path / "actor-authority.json",
        vlm_metadata=tmp_path / "vlm",
        robotwin_root=tmp_path / "robotwin",
        event_spec=tmp_path / "event.json",
        supplement_root=tmp_path / "supplement",
        supplement_binding=tmp_path / "supplement" / "binding.json",
        primary_binding=tmp_path / "primary-binding.json",
        primary_branches_root=tmp_path / "primary-branches",
        materialization_receipt=tmp_path / "materialization.json",
        augmented_lobo_root=tmp_path / "augmented-lobo",
        augmented_lobo_state=tmp_path / "augmented-lobo.state.json",
        augmented_lobo_run_exit=tmp_path / "augmented-lobo.run.exit",
        augmented_n4_root=tmp_path / "augmented-n4",
        augmented_n8_root=tmp_path / "augmented-n8",
        metrics_preregistration=tmp_path / "metrics.json",
        lerobot_root=tmp_path / "lerobot",
        lerobot_site=tmp_path / "lerobot-site",
        robotwin_eval_site=tmp_path / "robotwin-eval-site",
        etsf_site=tmp_path / "etsf-site",
        poll_seconds=30.0,
        expected_gpu_uuid=watcher.EXPECTED_GPU_UUID,
    )


def test_commands_bind_complete_supplement_and_nested_candidate_study(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    supplement_sha = "a" * 64
    collector = watcher.supplement_collector_command(args, "piper")
    assert collector[collector.index("--body") + 1] == "piper"
    assert "--seeds" not in collector
    action_index = collector.index("--action-exec-steps")
    assert collector[collector.index("--conditions") + 1 : action_index] == [
        "clean",
        "randomized",
    ]

    materializer = watcher.materializer_command(args)
    declarations = [
        materializer[index + 1]
        for index, value in enumerate(materializer)
        if value == "--body-manifest"
    ]
    assert len(declarations) == 5
    assert {value.split("=", 1)[0] for value in declarations} == set(watcher.BODIES)

    lobo = watcher.lobo_command(args, supplement_sha)
    assert lobo[lobo.index("--supplement-binding-sha256") + 1] == supplement_sha
    assert lobo[lobo.index("--supplement-binding") + 1] == str(
        args.supplement_binding
    )

    n4 = watcher.paired_n4_command(args, supplement_sha)
    n8 = watcher.paired_n8_command(args, supplement_sha)
    nested = watcher.nested_n4_n8_command(args, supplement_sha)
    for command in (n4, n8, nested):
        assert command[
            command.index("--required-supplement-binding-sha256") + 1
        ] == supplement_sha
        assert sum(value == "--lobo-fold" for value in command) == 5
    assert "--candidate-count" not in n4
    assert n8[n8.index("--candidate-count") + 1] == "8"
    assert n8[n8.index("--proposal-count") + 1] == "16"
    assert nested[nested.index("--output") + 1] == str(args.augmented_n8_root)
    assert "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py" in nested[1]
    assert "--candidate-count" not in nested
    assert watcher.N8_RETAINED_CANDIDATE_COUNT == 8
    assert watcher.N8_RAW_PROPOSAL_COUNT == 16


def test_runtime_environment_includes_explicit_etsf_dependency_site(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    environment = watcher.runtime_environment(args)
    python_paths = environment["PYTHONPATH"].split(":")
    assert str(args.etsf_site) in python_paths
    assert python_paths.index(str(args.etsf_site)) > python_paths.index(
        str(args.robotwin_eval_site)
    )
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_code_manifest_freezes_commit_and_rejects_wait_time_byte_drift(
    tmp_path: Path,
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    for index, relative in enumerate(watcher.CRITICAL_CODE_FILES):
        (code_root / relative).write_text(
            f"# frozen critical file {index}\n", encoding="utf-8"
        )
    commit_marker = "a" * 40
    manifest_path = tmp_path / "upgrade.code_manifest.json"
    frozen = watcher.freeze_code_manifest(
        manifest_path,
        code_root=code_root,
        commit_marker=commit_marker,
    )
    assert frozen["commit_marker"] == commit_marker
    assert len(frozen["critical_files"]) == len(watcher.CRITICAL_CODE_FILES)
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_sha256 = watcher.sha256_file(manifest_path)
    watcher.validate_frozen_code_manifest(
        manifest_path,
        code_root=code_root,
        commit_marker=commit_marker,
        expected_logical_sha256=frozen["logical_sha256"],
        expected_file_sha256=manifest_file_sha256,
    )

    manifest_path.write_bytes(manifest_bytes + b" \n")
    with pytest.raises(watcher.SharedHeadUpgradeError, match="bytes changed"):
        watcher.validate_frozen_code_manifest(
            manifest_path,
            code_root=code_root,
            commit_marker=commit_marker,
            expected_logical_sha256=frozen["logical_sha256"],
            expected_file_sha256=manifest_file_sha256,
        )
    manifest_path.write_bytes(manifest_bytes)

    drifted = code_root / watcher.CRITICAL_CODE_FILES[-1]
    drifted.write_text("# silently replaced while waiting\n", encoding="utf-8")
    with pytest.raises(
        watcher.SharedHeadUpgradeError,
        match="drifted after watcher startup",
    ):
        watcher.validate_frozen_code_manifest(
            manifest_path,
            code_root=code_root,
            commit_marker=commit_marker,
            expected_logical_sha256=frozen["logical_sha256"],
            expected_file_sha256=manifest_file_sha256,
        )


def test_code_manifest_requires_full_explicit_commit_sha(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    for relative in watcher.CRITICAL_CODE_FILES:
        (code_root / relative).write_text("# frozen\n", encoding="utf-8")
    with pytest.raises(watcher.SharedHeadUpgradeError, match="full lowercase"):
        watcher.build_code_manifest(code_root, "83c87a4")


def test_supplement_completion_is_exact_design_not_just_group_count(
    tmp_path: Path,
) -> None:
    roster = watcher.supplement_reserve_roster("franka")
    selected = {
        row["slot_key"]: row["ordered_requested_seeds"][1] for row in roster
    }
    groups = [
        {
            "condition": row["condition"],
            "horizon_slot": row["horizon_slot"],
            "requested_seed": selected[row["slot_key"]],
            "scripted_root_event": event,
        }
        for row in roster
        for event in watcher.TARGET_EVENTS
    ]
    attempts = []
    for row in roster:
        rejected_seed = row["ordered_requested_seeds"][0]
        selected_seed = selected[row["slot_key"]]
        attempts.extend(
            (
                {
                    "attempt_id": (
                        f"{row['slot_key']}|requested_seed={rejected_seed}"
                    ),
                    "status": "rejected_before_actor_outcomes",
                    "actor_candidate_outcomes_executed_before_selection": False,
                },
                {
                    "attempt_id": (
                        f"{row['slot_key']}|requested_seed={selected_seed}"
                    ),
                    "status": "complete",
                    "selected_before_actor_candidate_outcomes": True,
                    "actor_candidate_outcomes_executed_before_selection": False,
                },
            )
        )
    value = {
        "format": watcher.SUPPLEMENT_MANIFEST_FORMAT,
        "body": "franka",
        "conditions": list(watcher.CONDITIONS),
        "collection_status": "complete",
        "reserve_roster": roster,
        "pre_registered_seeds": [
            seed for row in roster for seed in row["ordered_requested_seeds"]
        ],
        "selected_seed_by_slot": selected,
        "groups": groups,
        "attempts": attempts,
    }
    value["logical_sha256"] = watcher.canonical_sha256(value)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert watcher.supplement_manifest_complete(path, "franka") is True

    value["groups"][-1]["scripted_root_event"] = "e3"
    unsigned = dict(value)
    unsigned.pop("logical_sha256")
    value["logical_sha256"] = watcher.canonical_sha256(unsigned)
    path.write_text(json.dumps(value), encoding="utf-8")
    assert watcher.supplement_manifest_complete(path, "franka") is False


def test_reserve_rosters_are_body_local_and_disjoint() -> None:
    by_body = {
        body: {
            seed
            for row in watcher.supplement_reserve_roster(body)
            for seed in row["ordered_requested_seeds"]
        }
        for body in watcher.BODIES
    }
    assert all(len(seeds) == 160 for seeds in by_body.values())
    for index, body in enumerate(watcher.BODIES):
        for other in watcher.BODIES[index + 1 :]:
            assert by_body[body].isdisjoint(by_body[other])


def test_upstream_gate_requires_complete_bound_ablation(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    value = {
        "format": watcher.UPSTREAM_FORMAT,
        "status": "complete",
        "summary": str(summary),
        "summary_file_sha256": watcher.sha256_file(summary),
    }
    watcher.validate_upstream_state(value)
    with pytest.raises(watcher.SharedHeadUpgradeError):
        watcher.validate_upstream_state({**value, "status": "waiting"})
    with pytest.raises(watcher.SharedHeadUpgradeError):
        watcher.validate_upstream_state(
            {**value, "summary_file_sha256": "0" * 64}
        )


def _write_nested_completion_chain(root: Path) -> None:
    root.mkdir()
    protocol_base = {
        "format": watcher.NESTED_PROTOCOL_FORMAT,
        "evaluation_seed_base": watcher.NESTED_SEED_BASE,
        "evaluation_seed_count": watcher.NESTED_SEED_COUNT,
        "formal_seed_block_reused": False,
        "seed_block_selected_before_any_nested_rollout_outcome": True,
        "balanced_body_condition_cells": len(watcher.BODIES)
        * len(watcher.CONDITIONS),
        "bootstrap_unit": (
            "requested_seed_cluster_with_all_selected_body_condition_rows_kept_together"
        ),
        "bootstrap_seed_derivation": {"overall": "frozen"},
        "pooled_mcnemar_role": "descriptive_only_due_repeated_requested_seeds",
        "single_body_condition_mcnemar_role": "inferential",
    }
    protocol = {
        **protocol_base,
        "logical_sha256": watcher.canonical_sha256(protocol_base),
    }
    contract_base = {
        "format": watcher.NESTED_CONTRACT_FORMAT,
        "runner_format": watcher.NESTED_RUNNER_FORMAT,
        "bodies": list(watcher.BODIES),
        "conditions": list(watcher.CONDITIONS),
        "evaluation_seed_base": watcher.NESTED_SEED_BASE,
        "evaluation_seed_count": watcher.NESTED_SEED_COUNT,
        "initial_condition_triplet_count": watcher.EXPECTED_NESTED_TRIPLETS,
        "rollout_count": watcher.EXPECTED_NESTED_ROLLOUTS,
        "methods": list(watcher.NESTED_METHODS),
        "same_requested_seed_and_complete_reset_tripled": True,
        "method_order_rotated_before_outcomes": True,
        "no_training": True,
        "method_result_persistence": {
            "existing_result_overwrite_or_retry_allowed": False,
            "automatic_noninformative_resume_limit_per_method": 1,
            "exception_or_action_failure_retry_allowed": False,
            "later_method_before_complete_prefix_allowed": False,
        },
        "nested_evaluation_protocol": protocol,
    }
    contract = {
        **contract_base,
        "logical_sha256": watcher.canonical_sha256(contract_base),
    }
    contract_path = root / "execution_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    rows = []
    for body in watcher.BODIES:
        for condition in watcher.CONDITIONS:
            for ordinal in range(watcher.NESTED_SEED_COUNT):
                row = {
                    "heldout_body": body,
                    "condition": condition,
                    "requested_seed": watcher.NESTED_SEED_BASE + ordinal,
                    "method_order": list(watcher.NESTED_METHODS),
                }
                for method in watcher.NESTED_METHODS:
                    row[f"{method}_binary_success"] = 0
                    row[f"{method}_stage_progress"] = 0.25
                rows.append(row)
    outcome_base = {
        "format": watcher.NESTED_OUTCOME_FORMAT,
        "status": "complete_1000_initial_condition_triplets_3000_rollouts",
        "pair_count": watcher.EXPECTED_NESTED_TRIPLETS,
        "rollout_count": watcher.EXPECTED_NESTED_ROLLOUTS,
        "methods": list(watcher.NESTED_METHODS),
        "rows": rows,
        "rows_sha256": watcher.canonical_sha256(rows),
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": watcher.sha256_file(contract_path),
        "nested_evaluation_protocol": protocol,
    }
    outcome = {
        **outcome_base,
        "document_sha256": watcher.canonical_sha256(outcome_base),
    }
    outcome_path = root / "nested_paired_outcomes.json"
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    report_base = {
        "format": watcher.NESTED_REPORT_FORMAT,
        "status": "complete_shared_raw16_nested_n4_n8_paired_report",
        "outcome_document_sha256": outcome["document_sha256"],
        "nested_evaluation_protocol": protocol,
        "by_heldout_body": {body: {} for body in watcher.BODIES},
        "by_heldout_body_and_condition": {
            f"{body}|{condition}": {}
            for body in watcher.BODIES
            for condition in watcher.CONDITIONS
        },
    }
    report = {
        **report_base,
        "report_sha256": watcher.canonical_sha256(report_base),
    }
    report_path = root / "nested_n4_n8_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    completion_base = {
        "format": watcher.NESTED_COMPLETION_FORMAT,
        "status": "complete_1000_triplets_3000_rollouts_frozen",
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": watcher.sha256_file(contract_path),
        "outcome_document_sha256": outcome["document_sha256"],
        "outcome_file_sha256": watcher.sha256_file(outcome_path),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": watcher.sha256_file(report_path),
        "initial_condition_triplet_count": watcher.EXPECTED_NESTED_TRIPLETS,
        "rollout_count": watcher.EXPECTED_NESTED_ROLLOUTS,
        "nested_evaluation_protocol_logical_sha256": protocol["logical_sha256"],
    }
    completion = {
        **completion_base,
        "logical_sha256": watcher.canonical_sha256(completion_base),
    }
    (root / "completion_receipt.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )


def test_nested_completion_gate_validates_exact_roster_and_sha_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nested"
    _write_nested_completion_chain(root)
    audit = watcher.validate_nested_completion(root)
    assert audit["completed_initial_condition_triplets"] == 1000
    assert audit["completed_rollouts"] == 3000
    assert audit["completed_rollouts_by_method"] == {
        method: 1000 for method in watcher.NESTED_METHODS
    }

    outcome_path = root / "nested_paired_outcomes.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["rows"][0]["actor_baseline_binary_success"] = 1
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    with pytest.raises(watcher.SharedHeadUpgradeError):
        watcher.validate_nested_completion(root)
