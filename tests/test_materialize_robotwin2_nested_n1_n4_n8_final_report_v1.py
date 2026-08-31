from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_robotwin2_five_body_lobo_n1_n4_n8_oracle_v1 as evaluator  # noqa: E402
import materialize_robotwin2_nested_n1_n4_n8_final_report_v1 as materializer  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: materializer.canonical_sha256(value)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def build_nested_root(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    actor_checkpoint = digest("actor checkpoint")
    folds: dict[str, Any] = {}
    for body in evaluator.BODIES:
        members = [
            {
                "member": index,
                "seed": 100 + index,
                "checkpoint": f"/fold/{body}/member_{index}.pt",
                "checkpoint_sha256": digest(f"critic|{body}|{index}"),
            }
            for index in range(5)
        ]
        folds[body] = {
            "heldout_body": body,
            "source_bodies": [candidate for candidate in evaluator.BODIES if candidate != body],
            "members": members,
            "training_summary_sha256": digest(f"summary|{body}"),
        }
    contract_base = {
        "format": materializer.NESTED_CONTRACT_FORMAT,
        "initial_condition_triplet_count": materializer.EXPECTED_PAIRS,
        "rollout_count": materializer.EXPECTED_PAIRS * 3,
        "methods": list(materializer.METHODS),
        "actor_checkpoint_tree_sha256": actor_checkpoint,
        "folds": folds,
    }
    contract = signed(contract_base, "logical_sha256")
    write_json(root / "execution_contract.json", contract)

    rows = []
    for body, condition, seed in materializer.expected_schedule():
        ordinal = seed - evaluator.SEED_BASE
        n1 = int(ordinal % 10 == 0)
        n4 = int(ordinal % 5 == 0)
        n8 = int(ordinal % 4 == 0)
        pool_sha = digest(f"pool|{body}|{condition}|{seed}")
        reset_sha = digest(f"reset|{body}|{condition}|{seed}")
        n4_pool_sha = digest(f"n4pool|{body}|{condition}|{seed}")
        decisions = {
            materializer.METHOD_ACTOR: {
                "selected_candidate_index": 0,
                "selection_pool_candidate_count": 1,
                "selection_pool_raw_indices": [0],
                "selection_pool_sha256": digest(f"n1pool|{body}|{condition}|{seed}"),
                "nested_pool_audit": {"n4_ordered_candidates_sha256": n4_pool_sha},
            },
            materializer.METHOD_N4: {
                "selected_candidate_index": 1,
                "selection_pool_candidate_count": 4,
                "selection_pool_raw_indices": [0, 3, 5, 7],
                "selection_pool_sha256": n4_pool_sha,
                "nested_pool_audit": {"n4_ordered_candidates_sha256": n4_pool_sha},
            },
            materializer.METHOD_N8: {
                "selected_candidate_index": 7,
                "selection_pool_candidate_count": 8,
                "selection_pool_raw_indices": [0, 3, 5, 7, 9, 11, 13, 15],
                "selection_pool_sha256": pool_sha,
                "nested_pool_audit": {"n4_ordered_candidates_sha256": n4_pool_sha},
            },
        }
        values = {
            materializer.METHOD_ACTOR: (n1, 1.0 if n1 else 0.25),
            materializer.METHOD_N4: (n4, 1.0 if n4 else 0.5),
            materializer.METHOD_N8: (n8, 1.0 if n8 else 0.75),
        }
        pair_base = {
            "format": materializer.NESTED_PAIR_FORMAT,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "same_resolved_reset_actor_n4_n8": True,
            "same_initial_raw16_and_nested_pool_audit": True,
            "n4_is_exact_ordered_prefix_of_n8": True,
            "initial_candidate_commitment_sha256": digest(
                f"commitment|{body}|{condition}|{seed}"
            ),
            "rollouts": {
                method: {
                    "binary_success": values[method][0],
                    "stage_progress": values[method][1],
                    "initial_reset_identity_sha256": reset_sha,
                    "decisions": [decisions[method]],
                }
                for method in materializer.METHODS
            },
        }
        pair = signed(pair_base, "pair_sha256")
        write_json(
            root / "pairs" / f"{materializer.pair_id(body, condition, seed)}.json",
            pair,
        )
        row = {
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "pair_sha256": pair["pair_sha256"],
        }
        for method in materializer.METHODS:
            row[f"{method}_binary_success"] = values[method][0]
            row[f"{method}_stage_progress"] = values[method][1]
        rows.append(row)
    outcomes_base = {
        "format": materializer.NESTED_OUTCOME_FORMAT,
        "status": "complete_1000_initial_condition_triplets_3000_rollouts",
        "pair_count": materializer.EXPECTED_PAIRS,
        "rollout_count": materializer.EXPECTED_PAIRS * 3,
        "methods": list(materializer.METHODS),
        "rows": rows,
        "rows_sha256": materializer.canonical_sha256(rows),
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": materializer.sha256_file(
            root / "execution_contract.json"
        ),
    }
    outcomes = signed(outcomes_base, "document_sha256")
    write_json(root / "nested_paired_outcomes.json", outcomes)
    report_base = {
        "format": materializer.NESTED_REPORT_FORMAT,
        "status": "complete_shared_raw16_nested_n4_n8_paired_report",
        "outcome_document_sha256": outcomes["document_sha256"],
    }
    nested_report = signed(report_base, "report_sha256")
    write_json(root / "nested_n4_n8_report.json", nested_report)
    completion_base = {
        "format": materializer.NESTED_COMPLETION_FORMAT,
        "status": "complete_1000_triplets_3000_rollouts_frozen",
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": materializer.sha256_file(
            root / "execution_contract.json"
        ),
        "outcome_document_sha256": outcomes["document_sha256"],
        "outcome_file_sha256": materializer.sha256_file(
            root / "nested_paired_outcomes.json"
        ),
        "report_sha256": nested_report["report_sha256"],
        "report_file_sha256": materializer.sha256_file(
            root / "nested_n4_n8_report.json"
        ),
        "initial_condition_triplet_count": materializer.EXPECTED_PAIRS,
        "rollout_count": materializer.EXPECTED_PAIRS * 3,
    }
    completion = signed(completion_base, "logical_sha256")
    write_json(root / "completion_receipt.json", completion)
    authority_base = {
        "format": materializer.ACTOR_AUTHORITY_FORMAT,
        "one_universal_actor_for_all_five_bodies": True,
        "upstream_training_state_file_sha256": digest("actor training receipt"),
        "actors": {
            body: {"checkpoint_sha256": actor_checkpoint}
            for body in evaluator.BODIES
        },
    }
    authority = signed(authority_base, "logical_sha256")
    authority_path = root / "actor_authority.json"
    write_json(authority_path, authority)
    return authority_path, outcomes, completion


def build_oracle_truth(
    root: Path,
    outcomes: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, Any]:
    groups = []
    for row, (body, condition, seed) in zip(
        outcomes["rows"], materializer.expected_schedule(), strict=True
    ):
        pair = json.loads(
            (
                root
                / "pairs"
                / f"{materializer.pair_id(body, condition, seed)}.json"
            ).read_text()
        )
        evidence = pair["rollouts"]
        actor = evidence[materializer.METHOD_ACTOR]
        n4 = evidence[materializer.METHOD_N4]["decisions"][0]
        n8 = evidence[materializer.METHOD_N8]["decisions"][0]
        results = []
        for index, raw_index in enumerate(n8["selection_pool_raw_indices"]):
            success = (
                actor["binary_success"]
                if index == 0
                else int(index in {1, 7})
            )
            stage = actor["stage_progress"] if index == 0 else (1.0 if success else 0.5)
            result_base = {
                "format": materializer.ORACLE_RESULT_FORMAT,
                "candidate_index": index,
                "raw_proposal_index": raw_index,
                "initial_candidate_commitment_sha256": pair[
                    "initial_candidate_commitment_sha256"
                ],
                "paired_reset_sha256": actor["initial_reset_identity_sha256"],
                "shared_raw8_candidate_pool_sha256": n8["selection_pool_sha256"],
                "continuation_policy": materializer.ORACLE_CONTINUATION_POLICY,
                "binary_success": success,
                "stage_progress": stage,
                "goal_progress": float(index),
                "action_execution_error": None,
            }
            results.append(signed(result_base, "result_sha256"))
        group_base = {
            "format": materializer.ORACLE_GROUP_FORMAT,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "decision_group_id": f"query0|{body}|{condition}|{seed}",
            "pair_sha256": row["pair_sha256"],
            "initial_candidate_commitment_sha256": pair[
                "initial_candidate_commitment_sha256"
            ],
            "paired_reset_sha256": actor["initial_reset_identity_sha256"],
            "shared_raw8_candidate_pool_sha256": n8["selection_pool_sha256"],
            "selected_index_n1": 0,
            "selected_index_n4": n4["selected_candidate_index"],
            "selected_index_n8": n8["selected_candidate_index"],
            "candidate_results": results,
        }
        groups.append(signed(group_base, "group_sha256"))
    truth_base = {
        "format": materializer.ORACLE_TRUTH_FORMAT,
        "status": materializer.ORACLE_TRUTH_STATUS,
        "nested_completion_logical_sha256": completion["logical_sha256"],
        "nested_outcome_document_sha256": outcomes["document_sha256"],
        "group_count": materializer.EXPECTED_PAIRS,
        "candidate_rollout_count": materializer.EXPECTED_CANDIDATE_ROLLOUTS,
        "groups": groups,
        "groups_sha256": materializer.canonical_sha256(groups),
    }
    return signed(truth_base, "logical_sha256")


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator, "BOOTSTRAP_SAMPLES", 20)


@pytest.fixture
def nested(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path / "nested"
    root.mkdir()
    authority, outcomes, completion = build_nested_root(root)
    return root, authority, outcomes, completion


def test_completed_nested_results_produce_policy_report_and_refuse_fake_oracle(
    nested: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    root, authority, _outcomes, _completion = nested
    materialized, report = materializer.build_materialization(
        nested_root=root,
        actor_authority_path=authority,
        oracle_truth_path=None,
    )
    assert len(materialized["policy_rows"]) == 1000
    assert report["policy_evaluation"]["pair_count"] == 1000
    assert report["oracle_branch_diagnostic"]["evidence_sufficient"] is False
    assert report["oracle_branch_diagnostic"]["oracle_regret"] is None
    assert report["transfer_claim"]["actor_zero_shot_to_all_five_bodies"] is False
    assert report["transfer_claim"]["all_five_critic_folds_are_label_free_on_target_body"] is True
    overall = report["policy_evaluation"]["global_equal_body_condition_macro"]
    assert overall["n4_minus_n1"]["success"]["delta_right_minus_left"] == 0.1
    assert overall["n4_minus_n1"]["success"]["delta_intervals"][
        "requested_seed_cluster_95pct_ci"
    ]["cluster_count"] == 100
    assert overall["n4_minus_n1"]["success"]["delta_intervals"][
        "body_condition_cluster_95pct_ci"
    ]["cluster_count"] == 10


def test_complete_8000_candidate_truth_enables_real_oracle_regret(
    nested: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    root, authority, outcomes, completion = nested
    truth = build_oracle_truth(root, outcomes, completion)
    truth_path = root / "oracle_truth.json"
    write_json(truth_path, truth)
    _materialized, report = materializer.build_materialization(
        nested_root=root,
        actor_authority_path=authority,
        oracle_truth_path=truth_path,
    )
    oracle = report["oracle_branch_diagnostic"]
    assert oracle["evidence_sufficient"] is True
    assert oracle["oracle_regret_reported"] is True
    assert oracle["decision_group_count"] == 1000
    assert oracle["global_equal_body_condition_macro"]["n4"][
        "success_oracle_regret"
    ]["mean"] >= 0.0


@pytest.mark.parametrize("mutation", ["pool", "candidate0", "incomplete"])
def test_oracle_truth_tamper_or_incompleteness_fails_closed(
    nested: tuple[Path, Path, dict[str, Any], dict[str, Any]], mutation: str
) -> None:
    root, authority, outcomes, completion = nested
    truth = build_oracle_truth(root, outcomes, completion)
    if mutation == "pool":
        truth["groups"][0]["shared_raw8_candidate_pool_sha256"] = "0" * 64
        truth["groups"][0].pop("group_sha256")
        truth["groups"][0] = signed(truth["groups"][0], "group_sha256")
    elif mutation == "candidate0":
        result = truth["groups"][0]["candidate_results"][0]
        result["binary_success"] = 1 - result["binary_success"]
        result["stage_progress"] = 1.0 if result["binary_success"] else 0.25
        result.pop("result_sha256")
        truth["groups"][0]["candidate_results"][0] = signed(
            result, "result_sha256"
        )
        truth["groups"][0].pop("group_sha256")
        truth["groups"][0] = signed(truth["groups"][0], "group_sha256")
    else:
        truth["groups"].pop()
        truth["group_count"] -= 1
        truth["candidate_rollout_count"] -= 8
    truth["groups_sha256"] = materializer.canonical_sha256(truth["groups"])
    truth.pop("logical_sha256")
    truth = signed(truth, "logical_sha256")
    truth_path = root / f"oracle_truth_{mutation}.json"
    write_json(truth_path, truth)
    with pytest.raises(materializer.FinalReportMaterializationError):
        materializer.build_materialization(
            nested_root=root,
            actor_authority_path=authority,
            oracle_truth_path=truth_path,
        )


def test_pair_tamper_is_detected_before_statistics(
    nested: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    root, authority, _outcomes, _completion = nested
    body, condition, seed = materializer.expected_schedule()[0]
    pair_path = root / "pairs" / f"{materializer.pair_id(body, condition, seed)}.json"
    pair = json.loads(pair_path.read_text())
    pair["rollouts"][materializer.METHOD_N4]["binary_success"] ^= 1
    write_json(pair_path, pair)
    with pytest.raises(materializer.FinalReportMaterializationError, match="pair_sha256"):
        materializer.build_materialization(
            nested_root=root,
            actor_authority_path=authority,
            oracle_truth_path=None,
        )


def test_cli_create_once_outputs_are_idempotent_but_tamper_fails(
    nested: tuple[Path, Path, dict[str, Any], dict[str, Any]], tmp_path: Path
) -> None:
    root, authority, _outcomes, _completion = nested
    output_input = tmp_path / "materialized.json"
    output_report = tmp_path / "final_report.json"
    assert materializer.main(
        [
            "--nested-root",
            str(root),
            "--actor-authority",
            str(authority),
            "--output-input",
            str(output_input),
            "--output-report",
            str(output_report),
        ]
    ) == 0
    assert json.loads(output_report.read_text())["oracle_branch_diagnostic"][
        "evidence_sufficient"
    ] is False
    command = [
        "--nested-root",
        str(root),
        "--actor-authority",
        str(authority),
        "--output-input",
        str(output_input),
        "--output-report",
        str(output_report),
    ]
    assert materializer.main(command) == 0
    output_report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.FinalReportMaterializationError,
        match="existing final report changed",
    ):
        materializer.main(command)
