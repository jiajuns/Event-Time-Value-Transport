from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_robotwin2_cross_embodiment_paired_success_v1 as evaluator  # noqa: E402


def _progress(success: int, index: int) -> float:
    if success:
        return 1.0
    return (0.0, 0.25, 0.5, 0.75)[index % 4]


def outcome_document() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for body_index, body in enumerate(evaluator.BODIES):
        for condition_index, condition in enumerate(evaluator.EVALUATION_CONDITIONS):
            for seed_ordinal in range(evaluator.EVALUATION_SEED_COUNT):
                baseline = int((seed_ordinal + body_index + condition_index) % 5 == 0)
                etsf = int((seed_ordinal + 2 * body_index + condition_index) % 4 == 0)
                rows.append(
                    {
                        "benchmark": evaluator.BENCHMARK,
                        "task": evaluator.TASK,
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": evaluator.EVALUATION_SEED_BASE + seed_ordinal,
                        "method_order": (
                            list(evaluator.METHODS)
                            if seed_ordinal % 2 == 0
                            else list(reversed(evaluator.METHODS))
                        ),
                        "actor_baseline_binary_success": baseline,
                        "actor_baseline_stage_progress": _progress(
                            baseline, seed_ordinal + body_index
                        ),
                        "etsf_best_of_4_binary_success": etsf,
                        "etsf_best_of_4_stage_progress": _progress(
                            etsf, seed_ordinal + body_index + condition_index + 1
                        ),
                    }
                )
    base: dict[str, Any] = {
        "format": evaluator.INPUT_FORMAT,
        "status": evaluator.INPUT_STATUS,
        "preregistration_sha256": evaluator.APPROVED_PREREGISTRATION_SHA256,
        "rows": rows,
        "rows_sha256": evaluator.canonical_sha256(rows),
    }
    return {**base, "document_sha256": evaluator.canonical_sha256(base)}


def resign(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["rows_sha256"] = evaluator.canonical_sha256(result["rows"])
    result.pop("document_sha256", None)
    result["document_sha256"] = evaluator.canonical_sha256(result)
    return result


@pytest.fixture(scope="module")
def valid_document() -> dict[str, Any]:
    return outcome_document()


@pytest.fixture(scope="module")
def valid_report(valid_document: dict[str, Any]) -> dict[str, Any]:
    return evaluator.evaluate_document(valid_document, input_file_sha256="b" * 64)


def test_complete_report_has_all_preregistered_levels_and_self_hash(
    valid_report: dict[str, Any],
) -> None:
    assert valid_report["pair_count"] == 1000
    assert valid_report["planned_rollout_count"] == 2000
    assert len(valid_report["per_body_condition"]) == 10
    assert len(valid_report["per_body_equal_condition_macro"]) == 5
    assert len(valid_report["per_condition_equal_body_macro"]) == 2
    assert valid_report["global_equal_body_condition_macro"]["pair_count"] == 1000
    assert valid_report["per_body_condition"][0]["success"][
        "delta_etsf_minus_actor"
    ]["paired_requested_seed_cluster_bootstrap_95pct_ci"]["rows_per_cluster"] == 1
    assert valid_report["per_body_equal_condition_macro"][0]["success"][
        "delta_etsf_minus_actor"
    ]["paired_requested_seed_cluster_bootstrap_95pct_ci"]["rows_per_cluster"] == 2
    assert valid_report["per_condition_equal_body_macro"][0]["success"][
        "delta_etsf_minus_actor"
    ]["paired_requested_seed_cluster_bootstrap_95pct_ci"]["rows_per_cluster"] == 5
    unsigned = dict(valid_report)
    report_sha = unsigned.pop("report_sha256")
    assert report_sha == evaluator.canonical_sha256(unsigned)


def test_success_intervals_bootstrap_and_mcnemar_are_explicit(
    valid_report: dict[str, Any],
) -> None:
    global_row = valid_report["global_equal_body_condition_macro"]
    actor = global_row["success"]["actor_baseline"]
    assert actor["success_count"] == 200
    assert actor["rate"] == 0.2
    assert set(actor) >= {"wilson_95pct_ci", "clopper_pearson_95pct_ci"}
    delta_ci = global_row["success"]["delta_etsf_minus_actor"][
        "paired_requested_seed_cluster_bootstrap_95pct_ci"
    ]
    assert delta_ci["samples"] == 20_000
    assert delta_ci["seed"] == 2_026_090_200
    assert delta_ci["draw_index_sha256"] == evaluator.APPROVED_BOOTSTRAP_DRAW_INDEX_SHA256
    assert delta_ci["cluster_unit"].startswith("requested_seed")
    assert delta_ci["cluster_count"] == 100
    assert delta_ci["rows_per_cluster"] == 10
    assert delta_ci["method"].endswith("not_exact")
    mcnemar = global_row["success"]["exact_two_sided_mcnemar"]
    assert mcnemar["method"] == "exact_two_sided_binomial_on_discordant_b_c"
    assert mcnemar["repeated_seed_dependence_accounted_for"] is False


def test_progress_is_supporting_and_never_called_exact(valid_report: dict[str, Any]) -> None:
    progress = valid_report["global_equal_body_condition_macro"]["stage_progress"]
    assert progress["supporting_endpoint_only"] is True
    assert len(progress["stage_reach_rates"]) == 5
    assert progress["paired_requested_seed_cluster_bootstrap_95pct_ci"][
        "method"
    ].endswith("not_exact")
    assert progress["delta_conservative_95pct_ci"]["method"].endswith("not_exact")


def test_capabilities_do_not_authorize_claim_promotion_or_execution(
    valid_report: dict[str, Any],
) -> None:
    capability = valid_report["capability"]
    assert capability
    assert all(value is False for value in capability.values())
    boundary = valid_report["interpretation_boundary"]
    assert boundary["preregistration_temporal_precedence_verified_by_this_file_alone"] is False
    assert boundary["training_heldout_disjointness_cryptographically_proven_by_this_file_alone"] is False
    gate = valid_report["prospective_improvement_gate"]
    assert gate["gate_authorizes_claim_promotion_or_deployment"] is False
    assert gate["passed"] == all(gate["checks"].values())


def test_clopper_pearson_known_boundary_values() -> None:
    lower, upper = evaluator.clopper_pearson(0, 2)
    assert lower == 0.0
    assert upper == pytest.approx(0.841886116992, abs=1e-10)
    lower, upper = evaluator.clopper_pearson(2, 2)
    assert lower == pytest.approx(0.158113883008, abs=1e-10)
    assert upper == 1.0


def test_wilson_and_exact_mcnemar_known_values() -> None:
    assert evaluator.wilson_score(0, 10)[0] == 0.0
    assert evaluator.exact_two_sided_mcnemar(1, 3) == evaluator.Fraction(5, 8)
    assert evaluator.exact_two_sided_mcnemar(0, 0) == evaluator.Fraction(1, 1)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["rows"].pop(), "5 bodies x 2 conditions x 100"),
        (lambda value: value["rows"].append(copy.deepcopy(value["rows"][0])), "duplicate"),
        (lambda value: value["rows"][0].__setitem__("critic_logit", 100.0), "schema changed"),
        (lambda value: value["rows"][0].pop("etsf_best_of_4_binary_success"), "schema changed"),
        (lambda value: value["rows"][0].__setitem__("actor_baseline_binary_success", True), "integer 0 or 1"),
        (lambda value: value["rows"][0].__setitem__("actor_baseline_stage_progress", 0.3), "must be one of"),
        (lambda value: value["rows"][0].__setitem__("heldout_body", "piper_training"), "identities are forbidden"),
        (lambda value: value["rows"][0].__setitem__("requested_seed", -1), "non-negative"),
        (lambda value: value["rows"][0].__setitem__("method_order", list(reversed(evaluator.METHODS))), "preregistered seeds"),
    ],
)
def test_malformed_leaky_or_incomplete_rows_fail_closed(
    valid_document: dict[str, Any], mutation: Any, message: str
) -> None:
    changed = copy.deepcopy(valid_document)
    mutation(changed)
    changed = resign(changed)
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match=message):
        evaluator.validate_input_document(changed)


def test_success_and_terminal_stage_must_agree(valid_document: dict[str, Any]) -> None:
    changed = copy.deepcopy(valid_document)
    changed["rows"][0]["actor_baseline_binary_success"] = 1
    changed["rows"][0]["actor_baseline_stage_progress"] = 0.75
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="must agree"):
        evaluator.validate_input_document(resign(changed))


def test_nonfinite_stage_is_rejected_before_metric_computation() -> None:
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="must be one of"):
        evaluator._progress(math.nan, "synthetic stage")


def test_rows_and_document_self_hashes_fail_closed(valid_document: dict[str, Any]) -> None:
    changed = copy.deepcopy(valid_document)
    changed["rows"][0]["task"] = "changed_task"
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="rows SHA"):
        evaluator.validate_input_document(changed)
    changed = resign(changed)
    changed["document_sha256"] = "0" * 64
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="document canonical"):
        evaluator.validate_input_document(changed)


def test_unapproved_preregistration_sha_fails_closed(valid_document: dict[str, Any]) -> None:
    changed = copy.deepcopy(valid_document)
    changed["preregistration_sha256"] = "f" * 64
    changed.pop("document_sha256")
    changed["document_sha256"] = evaluator.canonical_sha256(changed)
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="not the approved"):
        evaluator.validate_input_document(changed)


def test_duplicate_json_key_and_nan_token_are_rejected() -> None:
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="duplicate JSON key"):
        evaluator._strict_json(b'{"format":"a","format":"b"}', "synthetic")
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="non-finite"):
        evaluator._strict_json(b'{"value":NaN}', "synthetic")


def test_cli_requires_frozen_content_addressed_json_and_create_once(
    tmp_path: Path, valid_document: dict[str, Any]
) -> None:
    input_path = tmp_path / "paired_outcomes.json"
    input_path.write_text(json.dumps(valid_document, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_path = tmp_path / "report.json"
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="read-only"):
        evaluator.main(
            ["--input", str(input_path), "--input-file-sha256", digest, "--output", str(output_path)]
        )
    input_path.chmod(0o444)
    assert evaluator.main(
        ["--input", str(input_path), "--input-file-sha256", digest, "--output", str(output_path)]
    ) == 0
    assert stat_mode(output_path) == 0o444
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["pair_count"] == 1000
    with pytest.raises(FileExistsError):
        evaluator.main(
            ["--input", str(input_path), "--input-file-sha256", digest, "--output", str(output_path)]
        )


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_input_symlink_and_wrong_file_sha_fail_closed(
    tmp_path: Path, valid_document: dict[str, Any]
) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(valid_document), encoding="utf-8")
    target.chmod(0o444)
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="symbolic links"):
        evaluator._secure_read_frozen_json(link, digest)
    with pytest.raises(evaluator.PairedCrossEmbodimentEvaluationError, match="mismatch"):
        evaluator._secure_read_frozen_json(target, "0" * 64)
