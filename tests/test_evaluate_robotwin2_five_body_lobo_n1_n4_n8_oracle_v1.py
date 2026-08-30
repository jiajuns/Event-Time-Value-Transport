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


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _stage(success: int, failed_progress: float) -> float:
    return 1.0 if success else failed_progress


def input_document(*, actor_training_bodies: list[str] | None = None) -> dict[str, Any]:
    actor_training_bodies = (
        list(evaluator.BODIES)
        if actor_training_bodies is None
        else actor_training_bodies
    )
    folds: list[dict[str, Any]] = []
    fold_sha: dict[str, str] = {}
    for heldout in evaluator.BODIES:
        sources = [body for body in evaluator.BODIES if body != heldout]
        checkpoint = _digest(f"critic|{heldout}")
        fold_sha[heldout] = checkpoint
        folds.append(
            {
                "heldout_body": heldout,
                "source_supervision_bodies": sources,
                "selection_bodies": sources,
                "normalizer_fit_bodies": sources,
                "target_labeled_group_count": 0,
                "target_adapter": evaluator.TARGET_ADAPTER,
                "checkpoint_sha256": checkpoint,
                "training_receipt_sha256": _digest(f"receipt|{heldout}"),
            }
        )

    policy_rows: list[dict[str, Any]] = []
    oracle_groups: list[dict[str, Any]] = []
    for body in evaluator.BODIES:
        for condition in evaluator.CONDITIONS:
            for ordinal in range(evaluator.SEED_COUNT):
                seed = evaluator.SEED_BASE + ordinal
                n1 = int(ordinal % 10 == 0)
                n4 = int(ordinal % 5 == 0)
                n8 = int(ordinal % 4 == 0)
                pool_sha = _digest(f"pool|{body}|{condition}|{seed}")
                policy_rows.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": seed,
                        "paired_reset_sha256": _digest(
                            f"reset|{body}|{condition}|{seed}"
                        ),
                        "shared_raw8_candidate_pool_sha256": pool_sha,
                        "critic_checkpoint_sha256": fold_sha[body],
                        "actor_n1_binary_success": n1,
                        "actor_n1_stage_progress": _stage(n1, 0.25),
                        "critic_n4_binary_success": n4,
                        "critic_n4_stage_progress": _stage(n4, 0.5),
                        "critic_n8_binary_success": n8,
                        "critic_n8_stage_progress": _stage(n8, 0.75),
                    }
                )
                oracle_groups.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": seed,
                        "decision_group_id": f"query0|{body}|{condition}|{seed}",
                        "shared_raw8_candidate_pool_sha256": pool_sha,
                        "critic_checkpoint_sha256": fold_sha[body],
                        "candidate_binary_success": [0, 1, 0, 0, 0, 0, 0, 1],
                        "candidate_stage_progress": [
                            0.25,
                            1.0,
                            0.5,
                            0.75,
                            0.0,
                            0.25,
                            0.5,
                            1.0,
                        ],
                        "candidate_goal_progress": [
                            0.0,
                            1.0,
                            0.2,
                            0.3,
                            -0.1,
                            0.4,
                            0.5,
                            2.0,
                        ],
                        "selected_index_n1": 0,
                        "selected_index_n4": 0,
                        "selected_index_n8": 7,
                    }
                )
    base: dict[str, Any] = {
        "format": evaluator.FORMAT,
        "benchmark": evaluator.BENCHMARK,
        "task": evaluator.TASK,
        "actor_provenance": {
            "checkpoint_sha256": _digest("actor"),
            "training_data_receipt_sha256": _digest("actor receipt"),
            "training_bodies": actor_training_bodies,
        },
        "critic_folds": folds,
        "policy_rows": policy_rows,
        "oracle_groups": oracle_groups,
    }
    return {**base, "document_sha256": evaluator.canonical_sha256(base)}


def resign(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result.pop("document_sha256", None)
    result["document_sha256"] = evaluator.canonical_sha256(result)
    return result


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator, "BOOTSTRAP_SAMPLES", 200)


@pytest.fixture(scope="module")
def valid_document() -> dict[str, Any]:
    return input_document()


@pytest.fixture
def report(valid_document: dict[str, Any]) -> dict[str, Any]:
    return evaluator.build_report(valid_document)


def test_formal_report_has_complete_n1_n4_n8_paired_metrics(report: dict[str, Any]) -> None:
    policy = report["policy_evaluation"]
    assert policy["pair_count"] == 1000
    assert policy["rollout_count"] == 3000
    overall = policy["global_equal_body_condition_macro"]
    assert set(overall) == {"n4_minus_n1", "n8_minus_n1", "n8_minus_n4"}
    assert overall["n4_minus_n1"]["success"]["left_rate"] == 0.1
    assert overall["n4_minus_n1"]["success"]["right_rate"] == 0.2
    assert overall["n4_minus_n1"]["success"]["delta_right_minus_left"] == 0.1
    assert overall["n8_minus_n1"]["success"]["delta_right_minus_left"] == 0.15
    assert overall["n8_minus_n4"]["success"]["delta_right_minus_left"] == 0.05


def test_intervals_cluster_on_seed_and_body_condition(report: dict[str, Any]) -> None:
    intervals = report["policy_evaluation"]["global_equal_body_condition_macro"][
        "n4_minus_n1"
    ]["success"]["delta_intervals"]
    assert intervals["requested_seed_cluster_95pct_ci"]["cluster_count"] == 100
    assert intervals["body_condition_cluster_95pct_ci"]["cluster_count"] == 10
    assert intervals["requested_seed_cluster_95pct_ci"]["replicates"] == 200
    assert intervals["body_condition_cluster_95pct_ci"]["cluster_unit"].startswith(
        "heldout_body_x_condition"
    )


def test_mcnemar_is_exact_and_multi_cell_scope_is_not_overclaimed(
    report: dict[str, Any],
) -> None:
    overall = report["policy_evaluation"]["global_equal_body_condition_macro"][
        "n4_minus_n1"
    ]["success"]
    assert overall["discordance"]["left_only_b"] == 0
    assert overall["discordance"]["right_only_c"] == 100
    assert overall["exact_two_sided_mcnemar"]["multi_cell_value_is_descriptive"] is True
    cell = report["policy_evaluation"]["by_heldout_body_and_condition"][
        "piper|clean"
    ]["n4_minus_n1"]["success"]["exact_two_sided_mcnemar"]
    assert cell["inferentially_valid_for_this_scope"] is True
    assert evaluator.exact_two_sided_mcnemar(1, 3) == evaluator.Fraction(5, 8)


def test_critic_transfer_and_actor_zero_shot_are_separate_claims(
    report: dict[str, Any],
) -> None:
    claim = report["transfer_claim"]
    assert claim["claim_type"] == "heldout_critic_transfer_with_actor_body_exposure"
    assert all(claim["critic_transfer_proven_by_manifest_per_body"].values())
    assert not any(claim["actor_zero_shot_proven_by_manifest_per_body"].values())
    assert claim["critic_transfer_does_not_imply_actor_zero_shot"] is True


def test_actor_zero_shot_claim_requires_actor_manifest_exclusion() -> None:
    report = evaluator.build_report(input_document(actor_training_bodies=[]))
    claim = report["transfer_claim"]
    assert claim["actor_zero_shot_to_all_five_bodies"] is True
    assert claim["claim_type"] == "joint_actor_and_critic_zero_shot_to_every_heldout_body"


def test_oracle_reports_headroom_and_regret_without_entering_policy_delta(
    report: dict[str, Any],
) -> None:
    oracle = report["oracle_branch_diagnostic"]["global_equal_body_condition_macro"]
    assert oracle["n1"]["success_oracle_regret"]["mean"] == 0.0
    assert oracle["n4"]["oracle_success_rate"]["mean"] == 1.0
    assert oracle["n4"]["selected_success_rate"]["mean"] == 0.0
    assert oracle["n4"]["success_oracle_regret"]["mean"] == 1.0
    assert oracle["n8"]["success_oracle_regret"]["mean"] == 0.0
    assert oracle["n4"]["stage_oracle_regret"]["mean"] == 0.75
    assert oracle["n8"]["goal_oracle_regret"]["mean"] == 0.0
    assert oracle["n4"]["mixed_success_selection_accuracy"] == 0.0
    assert oracle["n8"]["mixed_success_selection_accuracy"] == 1.0
    assert report["oracle_branch_diagnostic"][
        "separate_from_closed_loop_success_estimand"
    ] is True


def test_oracle_is_one_lexicographic_candidate_not_independent_endpoint_maxima(
    valid_document: dict[str, Any],
) -> None:
    validated = evaluator.validate_document(valid_document)
    group = copy.deepcopy(validated["oracle_groups"][0])
    group["candidate_goal_progress"][2] = 100.0
    group["selected_indices"][4] = 1
    row = evaluator._oracle_row(group, 4)
    assert row["oracle_index"] == 1
    assert row["oracle_success"] == 1.0
    assert row["oracle_stage"] == 1.0
    assert row["oracle_goal"] == 1.0
    assert row["goal_regret"] == 0.0
    assert row["marginal_goal_ceiling"] == 100.0


def test_auc_or_prediction_fields_cannot_masquerade_as_transfer(
    valid_document: dict[str, Any],
) -> None:
    changed = copy.deepcopy(valid_document)
    changed["policy_rows"][0]["critic_auc"] = 0.99
    with pytest.raises(evaluator.CrossEmbodimentReportError, match="unknown fields"):
        evaluator.validate_document(resign(changed))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda value: value["critic_folds"][0]["source_supervision_bodies"].append(
                value["critic_folds"][0]["heldout_body"]
            ),
            "other four",
        ),
        (
            lambda value: value["critic_folds"][0].__setitem__(
                "target_labeled_group_count", 1
            ),
            "target-body labeled",
        ),
        (
            lambda value: value["policy_rows"][0].__setitem__(
                "critic_checkpoint_sha256", "0" * 64
            ),
            "LOBO checkpoint",
        ),
        (
            lambda value: value["policy_rows"].pop(),
            "5 bodies x 2 conditions x 100",
        ),
        (
            lambda value: value["oracle_groups"].pop(),
            "body-condition-seed",
        ),
        (
            lambda value: value["oracle_groups"][0].__setitem__(
                "selected_index_n4", 7
            ),
            "nested first-4",
        ),
    ],
)
def test_leakage_incomplete_rosters_and_non_nested_selection_fail_closed(
    valid_document: dict[str, Any], mutation: Any, message: str
) -> None:
    changed = copy.deepcopy(valid_document)
    mutation(changed)
    with pytest.raises(evaluator.CrossEmbodimentReportError, match=message):
        evaluator.validate_document(resign(changed))


def test_success_must_agree_with_terminal_event_stage(
    valid_document: dict[str, Any],
) -> None:
    changed = copy.deepcopy(valid_document)
    changed["oracle_groups"][0]["candidate_binary_success"][0] = 1
    with pytest.raises(evaluator.CrossEmbodimentReportError, match="disagrees"):
        evaluator.validate_document(resign(changed))


def test_report_is_deterministic_and_self_hashed(
    valid_document: dict[str, Any],
) -> None:
    first = evaluator.build_report(valid_document)
    second = evaluator.build_report(valid_document)
    assert first == second
    unsigned = dict(first)
    digest = unsigned.pop("report_sha256")
    assert digest == evaluator.canonical_sha256(unsigned)


def test_cli_writes_create_once_json(
    tmp_path: Path, valid_document: dict[str, Any]
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(valid_document), encoding="utf-8")
    output_path = tmp_path / "report.json"
    assert evaluator.main(["--input", str(input_path), "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["pair_count"] == 1000
    with pytest.raises(evaluator.CrossEmbodimentReportError, match="new .json"):
        evaluator.main(["--input", str(input_path), "--output", str(output_path)])
