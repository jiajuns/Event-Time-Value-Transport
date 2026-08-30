#!/usr/bin/env python3
"""Build the formal five-body LOBO N=1/N=4/N=8 evaluation report.

This evaluator is intentionally outcome-only.  It consumes complete paired
closed-loop outcomes plus a separate, sealed candidate-branch oracle audit.  It
never treats critic discrimination metrics as evidence of policy transfer.

The report distinguishes two different claims:

* ``critic_transfer``: the evaluated body's labels never affected the critic;
* ``actor_zero_shot``: the frozen actor also never trained on that body.

Those claims are not interchangeable.  The normal RoboTwin2 study proves the
first while an actor trained on all five bodies does not prove the second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FORMAT = "etsf_robotwin2_five_body_lobo_n1_n4_n8_oracle_input_v1"
REPORT_FORMAT = "etsf_robotwin2_five_body_lobo_n1_n4_n8_oracle_report_v1"
BENCHMARK = "RoboTwin2.0"
TASK = "move_can_pot"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
METHODS = ("actor_n1", "critic_n4", "critic_n8")
POOL_SIZES = (1, 4, 8)
SEED_BASE = 2_026_091_000
SEED_COUNT = 100
BOOTSTRAP_SEED = 2_026_091_900
BOOTSTRAP_SAMPLES = 20_000
STAGE_SUPPORT = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_ADAPTER = "analytic_label_free_state_action_frame_only"
SHA_CHARS = frozenset("0123456789abcdef")

TOP_FIELDS = {
    "format",
    "benchmark",
    "task",
    "actor_provenance",
    "critic_folds",
    "policy_rows",
    "oracle_groups",
    "document_sha256",
}
ACTOR_FIELDS = {
    "checkpoint_sha256",
    "training_data_receipt_sha256",
    "training_bodies",
}
FOLD_FIELDS = {
    "heldout_body",
    "source_supervision_bodies",
    "selection_bodies",
    "normalizer_fit_bodies",
    "target_labeled_group_count",
    "target_adapter",
    "checkpoint_sha256",
    "training_receipt_sha256",
}
POLICY_FIELDS = {
    "heldout_body",
    "condition",
    "requested_seed",
    "paired_reset_sha256",
    "shared_raw8_candidate_pool_sha256",
    "critic_checkpoint_sha256",
    "actor_n1_binary_success",
    "actor_n1_stage_progress",
    "critic_n4_binary_success",
    "critic_n4_stage_progress",
    "critic_n8_binary_success",
    "critic_n8_stage_progress",
}
ORACLE_FIELDS = {
    "heldout_body",
    "condition",
    "requested_seed",
    "decision_group_id",
    "shared_raw8_candidate_pool_sha256",
    "critic_checkpoint_sha256",
    "candidate_binary_success",
    "candidate_stage_progress",
    "candidate_goal_progress",
    "selected_index_n1",
    "selected_index_n4",
    "selected_index_n8",
}


class CrossEmbodimentReportError(RuntimeError):
    """The formal outcome, LOBO, or statistical contract is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_fields(value: Any, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CrossEmbodimentReportError(
            f"{role} has missing or unknown fields; outcome/prediction leakage is forbidden"
        )
    return value


def _sha(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= SHA_CHARS
    ):
        raise CrossEmbodimentReportError(f"{role} must be a lowercase SHA-256")
    return value


def _body(value: Any, role: str) -> str:
    if value not in BODIES:
        raise CrossEmbodimentReportError(f"{role} is not a frozen benchmark body")
    return str(value)


def _condition(value: Any, role: str) -> str:
    if value not in CONDITIONS:
        raise CrossEmbodimentReportError(f"{role} is not a frozen condition")
    return str(value)


def _seed(value: Any, role: str) -> int:
    if type(value) is not int or not SEED_BASE <= value < SEED_BASE + SEED_COUNT:
        raise CrossEmbodimentReportError(f"{role} is outside the frozen seed roster")
    return value


def _binary(value: Any, role: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise CrossEmbodimentReportError(f"{role} must be exact integer 0 or 1")
    return value


def _stage(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrossEmbodimentReportError(f"{role} must be a stage-progress number")
    result = float(value)
    if not math.isfinite(result) or result not in STAGE_SUPPORT:
        raise CrossEmbodimentReportError(
            f"{role} must be one of the frozen event stages {STAGE_SUPPORT}"
        )
    return result


def _finite(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrossEmbodimentReportError(f"{role} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CrossEmbodimentReportError(f"{role} must be finite numeric")
    return result


def _unique_bodies(value: Any, role: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CrossEmbodimentReportError(f"{role} must be a body list")
    result = tuple(_body(item, role) for item in value)
    if len(set(result)) != len(result):
        raise CrossEmbodimentReportError(f"{role} contains duplicate bodies")
    return result


def _validate_actor(value: Any) -> dict[str, Any]:
    actor = _exact_fields(value, ACTOR_FIELDS, "actor_provenance")
    bodies = _unique_bodies(actor["training_bodies"], "actor training bodies")
    return {
        "checkpoint_sha256": _sha(actor["checkpoint_sha256"], "actor checkpoint"),
        "training_data_receipt_sha256": _sha(
            actor["training_data_receipt_sha256"], "actor training receipt"
        ),
        "training_bodies": list(bodies),
    }


def _validate_folds(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(BODIES):
        raise CrossEmbodimentReportError("critic_folds must contain exactly five LOBO folds")
    folds: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        fold = _exact_fields(raw, FOLD_FIELDS, f"critic_folds[{index}]")
        heldout = _body(fold["heldout_body"], "fold heldout body")
        if heldout in folds:
            raise CrossEmbodimentReportError("duplicate held-out critic fold")
        expected_sources = set(BODIES) - {heldout}
        sources = set(
            _unique_bodies(fold["source_supervision_bodies"], "source supervision bodies")
        )
        selection = set(_unique_bodies(fold["selection_bodies"], "selection bodies"))
        normalizer = set(
            _unique_bodies(fold["normalizer_fit_bodies"], "normalizer-fit bodies")
        )
        if sources != expected_sources:
            raise CrossEmbodimentReportError(
                f"{heldout} fold does not use exactly the other four source bodies"
            )
        if not selection or not selection <= expected_sources:
            raise CrossEmbodimentReportError(
                f"{heldout} fold selection leaked target labels or has no source selection"
            )
        if not normalizer or not normalizer <= expected_sources:
            raise CrossEmbodimentReportError(
                f"{heldout} fold normalizer leaked target labels or is empty"
            )
        if type(fold["target_labeled_group_count"]) is not int or fold[
            "target_labeled_group_count"
        ] != 0:
            raise CrossEmbodimentReportError(
                f"{heldout} fold contains target-body labeled supervision"
            )
        if fold["target_adapter"] != TARGET_ADAPTER:
            raise CrossEmbodimentReportError(
                f"{heldout} fold target adapter is not analytic and label-free"
            )
        folds[heldout] = {
            "heldout_body": heldout,
            "source_supervision_bodies": sorted(sources),
            "selection_bodies": sorted(selection),
            "normalizer_fit_bodies": sorted(normalizer),
            "target_labeled_group_count": 0,
            "target_adapter": TARGET_ADAPTER,
            "checkpoint_sha256": _sha(
                fold["checkpoint_sha256"], f"{heldout} critic checkpoint"
            ),
            "training_receipt_sha256": _sha(
                fold["training_receipt_sha256"], f"{heldout} training receipt"
            ),
        }
    if set(folds) != set(BODIES):
        raise CrossEmbodimentReportError("critic folds do not cover all five held-out bodies")
    return folds


def _validate_policy_rows(
    value: Any, folds: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CrossEmbodimentReportError("policy_rows must be a list")
    expected = {
        (body, condition, seed)
        for body in BODIES
        for condition in CONDITIONS
        for seed in range(SEED_BASE, SEED_BASE + SEED_COUNT)
    }
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = _exact_fields(raw, POLICY_FIELDS, f"policy_rows[{index}]")
        body = _body(row["heldout_body"], "policy heldout body")
        condition = _condition(row["condition"], "policy condition")
        seed = _seed(row["requested_seed"], "policy seed")
        identity = (body, condition, seed)
        if identity in rows:
            raise CrossEmbodimentReportError("duplicate paired policy identity")
        critic_sha = _sha(row["critic_checkpoint_sha256"], "policy critic checkpoint")
        if critic_sha != folds[body]["checkpoint_sha256"]:
            raise CrossEmbodimentReportError(
                "policy row is not bound to its held-out body's LOBO checkpoint"
            )
        normalized: dict[str, Any] = {
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "paired_reset_sha256": _sha(row["paired_reset_sha256"], "paired reset"),
            "shared_raw8_candidate_pool_sha256": _sha(
                row["shared_raw8_candidate_pool_sha256"], "shared raw8 candidate pool"
            ),
            "critic_checkpoint_sha256": critic_sha,
        }
        for method in METHODS:
            success = _binary(row[f"{method}_binary_success"], f"{method} success")
            progress = _stage(row[f"{method}_stage_progress"], f"{method} stage")
            if (progress == 1.0) != bool(success):
                raise CrossEmbodimentReportError(
                    f"{method} binary success disagrees with terminal event progress"
                )
            normalized[f"{method}_binary_success"] = success
            normalized[f"{method}_stage_progress"] = progress
        rows[identity] = normalized
    if set(rows) != expected:
        raise CrossEmbodimentReportError(
            "policy_rows must equal all 5 bodies x 2 conditions x 100 paired seeds"
        )
    return [rows[key] for key in sorted(rows)]


def _selected_index(value: Any, pool_size: int, role: str) -> int:
    if type(value) is not int or not 0 <= value < pool_size:
        raise CrossEmbodimentReportError(
            f"{role} must select inside the nested first-{pool_size} pool"
        )
    return value


def _validate_oracle_groups(
    value: Any, folds: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CrossEmbodimentReportError(
            "oracle_groups must contain a separate sealed branch diagnostic"
        )
    groups: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    seed_cell_counts = {
        (body, condition, seed): 0
        for body in BODIES
        for condition in CONDITIONS
        for seed in range(SEED_BASE, SEED_BASE + SEED_COUNT)
    }
    for index, raw in enumerate(value):
        group = _exact_fields(raw, ORACLE_FIELDS, f"oracle_groups[{index}]")
        body = _body(group["heldout_body"], "oracle heldout body")
        condition = _condition(group["condition"], "oracle condition")
        seed = _seed(group["requested_seed"], "oracle seed")
        group_id = group["decision_group_id"]
        if not isinstance(group_id, str) or not group_id or len(group_id) > 256:
            raise CrossEmbodimentReportError("oracle decision_group_id is invalid")
        identity = (body, condition, seed, group_id)
        if identity in groups:
            raise CrossEmbodimentReportError("duplicate oracle decision group")
        critic_sha = _sha(group["critic_checkpoint_sha256"], "oracle critic checkpoint")
        if critic_sha != folds[body]["checkpoint_sha256"]:
            raise CrossEmbodimentReportError(
                "oracle group is not bound to its held-out body's LOBO checkpoint"
            )
        successes = group["candidate_binary_success"]
        stages = group["candidate_stage_progress"]
        goals = group["candidate_goal_progress"]
        if not all(isinstance(items, list) and len(items) == 8 for items in (successes, stages, goals)):
            raise CrossEmbodimentReportError(
                "oracle candidate outcome arrays must all have exact length 8"
            )
        success_values = [
            _binary(item, f"oracle candidate {candidate} success")
            for candidate, item in enumerate(successes)
        ]
        stage_values = [
            _stage(item, f"oracle candidate {candidate} stage")
            for candidate, item in enumerate(stages)
        ]
        goal_values = [
            _finite(item, f"oracle candidate {candidate} goal progress")
            for candidate, item in enumerate(goals)
        ]
        for candidate, (success, stage) in enumerate(zip(success_values, stage_values)):
            if (stage == 1.0) != bool(success):
                raise CrossEmbodimentReportError(
                    f"oracle candidate {candidate} success disagrees with stage progress"
                )
        selected = {
            1: _selected_index(group["selected_index_n1"], 1, "N=1 selection"),
            4: _selected_index(group["selected_index_n4"], 4, "N=4 selection"),
            8: _selected_index(group["selected_index_n8"], 8, "N=8 selection"),
        }
        normalized = {
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "decision_group_id": group_id,
            "shared_raw8_candidate_pool_sha256": _sha(
                group["shared_raw8_candidate_pool_sha256"], "oracle raw8 pool"
            ),
            "critic_checkpoint_sha256": critic_sha,
            "candidate_binary_success": success_values,
            "candidate_stage_progress": stage_values,
            "candidate_goal_progress": goal_values,
            "selected_indices": selected,
        }
        groups[identity] = normalized
        seed_cell_counts[(body, condition, seed)] += 1
    if 0 in seed_cell_counts.values() or len(set(seed_cell_counts.values())) != 1:
        raise CrossEmbodimentReportError(
            "oracle diagnostic must have nonempty equal decision-group counts per "
            "body-condition-seed; outcome-dependent missing groups are forbidden"
        )
    return [groups[key] for key in sorted(groups)]


def validate_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = _exact_fields(value, TOP_FIELDS, "input document")
    if (
        document["format"] != FORMAT
        or document["benchmark"] != BENCHMARK
        or document["task"] != TASK
    ):
        raise CrossEmbodimentReportError("input benchmark/task/format changed")
    unsigned = dict(document)
    recorded_sha = unsigned.pop("document_sha256")
    if _sha(recorded_sha, "input document SHA") != canonical_sha256(unsigned):
        raise CrossEmbodimentReportError("input document canonical SHA mismatch")
    actor = _validate_actor(document["actor_provenance"])
    folds = _validate_folds(document["critic_folds"])
    policy_rows = _validate_policy_rows(document["policy_rows"], folds)
    oracle_groups = _validate_oracle_groups(document["oracle_groups"], folds)
    return {
        "document_sha256": recorded_sha,
        "actor": actor,
        "folds": folds,
        "policy_rows": policy_rows,
        "oracle_groups": oracle_groups,
    }


def exact_two_sided_mcnemar(left_only: int, right_only: int) -> Fraction:
    if type(left_only) is not int or type(right_only) is not int or min(left_only, right_only) < 0:
        raise ValueError("discordant counts must be non-negative integers")
    discordant = left_only + right_only
    if discordant == 0:
        return Fraction(1, 1)
    tail = min(left_only, right_only)
    value = Fraction(
        2 * sum(math.comb(discordant, index) for index in range(tail + 1)),
        2**discordant,
    )
    return min(Fraction(1, 1), value)


def _number(value: float) -> float:
    if not math.isfinite(float(value)):
        raise CrossEmbodimentReportError("computed metric is non-finite")
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CrossEmbodimentReportError("cannot aggregate an empty metric")
    return sum(float(value) for value in values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    *,
    cluster_key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    label: str,
    cluster_unit: str,
) -> dict[str, Any]:
    if len(rows) != len(values) or not rows:
        raise CrossEmbodimentReportError("bootstrap rows and values must align")
    clusters: dict[tuple[Any, ...], list[float]] = {}
    for row, value in zip(rows, values):
        clusters.setdefault(cluster_key(row), []).append(float(value))
    cluster_means = [_mean(clusters[key]) for key in sorted(clusters)]
    generator = random.Random(_derived_seed(label))
    estimates = [
        _mean([generator.choice(cluster_means) for _ in cluster_means])
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return {
        "lower": _number(_quantile(estimates, 0.025)),
        "upper": _number(_quantile(estimates, 0.975)),
        "confidence_level": 0.95,
        "method": "paired_cluster_percentile_bootstrap_not_exact",
        "cluster_unit": cluster_unit,
        "cluster_count": len(cluster_means),
        "replicates": BOOTSTRAP_SAMPLES,
        "seed_derivation": "sha256(global_seed|scope|comparison|endpoint|cluster_axis)",
        "global_seed": BOOTSTRAP_SEED,
    }


def _comparison(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, scope: str
) -> dict[str, Any]:
    success_left = [float(row[f"{left}_binary_success"]) for row in rows]
    success_right = [float(row[f"{right}_binary_success"]) for row in rows]
    stage_left = [float(row[f"{left}_stage_progress"]) for row in rows]
    stage_right = [float(row[f"{right}_stage_progress"]) for row in rows]
    success_delta = [right_value - left_value for left_value, right_value in zip(success_left, success_right)]
    stage_delta = [right_value - left_value for left_value, right_value in zip(stage_left, stage_right)]
    left_only = sum(left_value == 1.0 and right_value == 0.0 for left_value, right_value in zip(success_left, success_right))
    right_only = sum(left_value == 0.0 and right_value == 1.0 for left_value, right_value in zip(success_left, success_right))
    both_success = sum(left_value == right_value == 1.0 for left_value, right_value in zip(success_left, success_right))
    both_fail = len(rows) - left_only - right_only - both_success
    cells = {(str(row["heldout_body"]), str(row["condition"])) for row in rows}

    def intervals(values: Sequence[float], endpoint: str) -> dict[str, Any]:
        return {
            "requested_seed_cluster_95pct_ci": _cluster_bootstrap(
                rows,
                values,
                cluster_key=lambda row: (int(row["requested_seed"]),),
                label=f"{scope}|{left}|{right}|{endpoint}|requested_seed",
                cluster_unit="requested_seed_preserving_all_selected_body_condition_pairs",
            ),
            "body_condition_cluster_95pct_ci": _cluster_bootstrap(
                rows,
                values,
                cluster_key=lambda row: (str(row["heldout_body"]), str(row["condition"])),
                label=f"{scope}|{left}|{right}|{endpoint}|body_condition",
                cluster_unit="heldout_body_x_condition_cell_preserving_all_paired_seeds",
            ),
        }

    p_value = exact_two_sided_mcnemar(left_only, right_only)
    return {
        "left_method": left,
        "right_method": right,
        "pair_count": len(rows),
        "success": {
            "left_rate": _number(_mean(success_left)),
            "right_rate": _number(_mean(success_right)),
            "delta_right_minus_left": _number(_mean(success_delta)),
            "delta_intervals": intervals(success_delta, "success"),
            "discordance": {
                "both_fail_n00": both_fail,
                "left_only_b": left_only,
                "right_only_c": right_only,
                "both_success_n11": both_success,
            },
            "exact_two_sided_mcnemar": {
                "p_value": _number(float(p_value)),
                "p_value_numerator": p_value.numerator,
                "p_value_denominator": p_value.denominator,
                "inferentially_valid_for_this_scope": len(cells) == 1,
                "multi_cell_value_is_descriptive": len(cells) > 1,
            },
        },
        "stage_progress": {
            "left_mean": _number(_mean(stage_left)),
            "right_mean": _number(_mean(stage_right)),
            "delta_right_minus_left": _number(_mean(stage_delta)),
            "delta_intervals": intervals(stage_delta, "stage_progress"),
            "supporting_endpoint_not_a_substitute_for_binary_success": True,
        },
    }


def _policy_scope(rows: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
    return {
        "n4_minus_n1": _comparison(rows, "actor_n1", "critic_n4", scope=scope),
        "n8_minus_n1": _comparison(rows, "actor_n1", "critic_n8", scope=scope),
        "n8_minus_n4": _comparison(rows, "critic_n4", "critic_n8", scope=scope),
    }


def _oracle_row(group: Mapping[str, Any], pool_size: int) -> dict[str, Any]:
    selected = int(group["selected_indices"][pool_size])
    successes = [float(value) for value in group["candidate_binary_success"][:pool_size]]
    stages = [float(value) for value in group["candidate_stage_progress"][:pool_size]]
    goals = [float(value) for value in group["candidate_goal_progress"][:pool_size]]
    # One realizable candidate defines the oracle.  Independent endpoint maxima
    # could come from different branches and therefore are ceilings, not a
    # candidate oracle policy.  The final ``-index`` term makes ties choose the
    # lowest candidate index exactly.
    oracle_index = max(
        range(pool_size),
        key=lambda index: (
            successes[index],
            stages[index],
            goals[index],
            -index,
        ),
    )
    oracle_success = successes[oracle_index]
    oracle_stage = stages[oracle_index]
    oracle_goal = goals[oracle_index]
    selected_success = successes[selected]
    selected_stage = stages[selected]
    selected_goal = goals[selected]
    baseline_success = successes[0]
    baseline_stage = stages[0]
    baseline_goal = goals[0]
    mixed_success = min(successes) < max(successes)
    return {
        "heldout_body": group["heldout_body"],
        "condition": group["condition"],
        "requested_seed": group["requested_seed"],
        "oracle_index": oracle_index,
        "oracle_success": oracle_success,
        "selected_success": selected_success,
        "baseline_success": baseline_success,
        "success_headroom_over_actor": oracle_success - baseline_success,
        "success_regret": oracle_success - selected_success,
        "oracle_stage": oracle_stage,
        "selected_stage": selected_stage,
        "stage_headroom_over_actor": oracle_stage - baseline_stage,
        "stage_regret": oracle_stage - selected_stage,
        "oracle_goal": oracle_goal,
        "selected_goal": selected_goal,
        "goal_headroom_over_actor": oracle_goal - baseline_goal,
        "goal_regret": oracle_goal - selected_goal,
        "marginal_success_ceiling": max(successes),
        "marginal_stage_ceiling": max(stages),
        "marginal_goal_ceiling": max(goals),
        "mixed_success": float(mixed_success),
        "mixed_success_selected_correctly": float(mixed_success and selected_success == 1.0),
    }


def _oracle_metric(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    label: str,
    include_intervals: bool,
) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    result: dict[str, Any] = {"mean": _number(_mean(values))}
    if not include_intervals:
        return result
    return {
        **result,
        "requested_seed_cluster_95pct_ci": _cluster_bootstrap(
            rows,
            values,
            cluster_key=lambda row: (int(row["requested_seed"]),),
            label=f"oracle|{label}|{field}|requested_seed",
            cluster_unit=(
                "requested_seed_preserving_all_selected_body_condition_"
                "decision_groups"
            ),
        ),
        "body_condition_cluster_95pct_ci": _cluster_bootstrap(
            rows,
            values,
            cluster_key=lambda row: (str(row["heldout_body"]), str(row["condition"])),
            label=f"oracle|{label}|{field}|body_condition",
            cluster_unit="heldout_body_x_condition_cell_preserving_all_decision_groups",
        ),
    }


def _oracle_scope(
    groups: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    include_intervals: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pool_size in POOL_SIZES:
        rows = [_oracle_row(group, pool_size) for group in groups]
        mixed_count = sum(row["mixed_success"] == 1.0 for row in rows)
        mixed_selected = sum(row["mixed_success_selected_correctly"] == 1.0 for row in rows)
        oracle_index_histogram = {
            str(index): sum(row["oracle_index"] == index for row in rows)
            for index in range(pool_size)
        }
        result[f"n{pool_size}"] = {
            "pool_size": pool_size,
            "decision_group_count": len(rows),
            "candidate_oracle_contract": (
                "single_candidate_lexicographic_binary_success_then_stage_"
                "progress_then_goal_progress_then_lowest_index"
            ),
            "oracle_candidate_index_histogram": oracle_index_histogram,
            "oracle_success_rate": _oracle_metric(rows, "oracle_success", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "selected_success_rate": _oracle_metric(rows, "selected_success", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "oracle_stage_progress": _oracle_metric(rows, "oracle_stage", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "selected_stage_progress": _oracle_metric(rows, "selected_stage", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "oracle_goal_progress": _oracle_metric(rows, "oracle_goal", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "selected_goal_progress": _oracle_metric(rows, "selected_goal", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "success_headroom_over_actor": _oracle_metric(rows, "success_headroom_over_actor", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "success_oracle_regret": _oracle_metric(rows, "success_regret", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "stage_oracle_regret": _oracle_metric(rows, "stage_regret", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "goal_oracle_regret": _oracle_metric(rows, "goal_regret", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "stage_headroom_over_actor": _oracle_metric(rows, "stage_headroom_over_actor", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "goal_headroom_over_actor": _oracle_metric(rows, "goal_headroom_over_actor", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            "marginal_endpoint_ceiling_not_a_candidate_oracle": {
                "success": _oracle_metric(rows, "marginal_success_ceiling", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
                "stage_progress": _oracle_metric(rows, "marginal_stage_ceiling", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
                "goal_progress": _oracle_metric(rows, "marginal_goal_ceiling", label=f"{scope}|n{pool_size}", include_intervals=include_intervals),
            },
            "mixed_success_group_count": mixed_count,
            "mixed_success_group_rate": _number(mixed_count / len(rows)),
            "mixed_success_selection_accuracy": (
                None if mixed_count == 0 else _number(mixed_selected / mixed_count)
            ),
        }
    return result


def build_report(value: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_document(value)
    rows = validated["policy_rows"]
    groups = validated["oracle_groups"]
    actor_training = set(validated["actor"]["training_bodies"])
    actor_zero_shot_by_body = {
        body: body not in actor_training for body in BODIES
    }
    critic_transfer_by_body = {body: True for body in BODIES}
    all_actor_zero_shot = all(actor_zero_shot_by_body.values())
    claim_type = (
        "joint_actor_and_critic_zero_shot_to_every_heldout_body"
        if all_actor_zero_shot
        else "heldout_critic_transfer_with_actor_body_exposure"
    )
    base = {
        "format": REPORT_FORMAT,
        "status": "complete_outcome_only_metrics_no_training_or_execution_authority",
        "input_document_sha256": validated["document_sha256"],
        "benchmark": BENCHMARK,
        "task": TASK,
        "closed_loop_primary_endpoint": "paired_binary_task_success",
        "closed_loop_supporting_endpoint": "terminal_event_stage_progress",
        "candidate_pool_baselines": {
            "n1": "frozen_actor_candidate_zero",
            "n4": "heldout_body_lobo_critic_rerank_nested_first_four",
            "n8": "same_heldout_body_lobo_critic_rerank_nested_first_eight",
            "oracle": "sealed_offline_true_outcome_upper_bound_never_deployable",
        },
        "transfer_claim": {
            "claim_type": claim_type,
            "critic_transfer_proven_by_manifest_per_body": critic_transfer_by_body,
            "all_five_critic_folds_are_label_free_on_target_body": True,
            "actor_zero_shot_proven_by_manifest_per_body": actor_zero_shot_by_body,
            "actor_zero_shot_to_all_five_bodies": all_actor_zero_shot,
            "actor_training_bodies": validated["actor"]["training_bodies"],
            "critic_transfer_does_not_imply_actor_zero_shot": True,
        },
        "fold_bindings": [validated["folds"][body] for body in BODIES],
        "policy_evaluation": {
            "pair_count": len(rows),
            "rollout_count": len(rows) * len(METHODS),
            "global_equal_body_condition_macro": _policy_scope(rows, scope="global"),
            "by_heldout_body": {
                body: _policy_scope(
                    [row for row in rows if row["heldout_body"] == body],
                    scope=f"body:{body}",
                )
                for body in BODIES
            },
            "by_condition_equal_body_macro": {
                condition: _policy_scope(
                    [row for row in rows if row["condition"] == condition],
                    scope=f"condition:{condition}",
                )
                for condition in CONDITIONS
            },
            "by_heldout_body_and_condition": {
                f"{body}|{condition}": _policy_scope(
                    [
                        row
                        for row in rows
                        if row["heldout_body"] == body
                        and row["condition"] == condition
                    ],
                    scope=f"cell:{body}|{condition}",
                )
                for body in BODIES
                for condition in CONDITIONS
            },
        },
        "oracle_branch_diagnostic": {
            "decision_group_count": len(groups),
            "separate_from_closed_loop_success_estimand": True,
            "unexecuted_candidate_outcomes_never_used_online": True,
            "cannot_be_called_a_deployable_policy": True,
            "global_equal_body_condition_macro": _oracle_scope(
                groups, scope="global", include_intervals=True
            ),
            "by_heldout_body": {
                body: _oracle_scope(
                    [group for group in groups if group["heldout_body"] == body],
                    scope=f"body:{body}",
                    include_intervals=False,
                )
                for body in BODIES
            },
            "by_condition_equal_body_macro": {
                condition: _oracle_scope(
                    [group for group in groups if group["condition"] == condition],
                    scope=f"condition:{condition}",
                    include_intervals=False,
                )
                for condition in CONDITIONS
            },
            "by_heldout_body_and_condition": {
                f"{body}|{condition}": _oracle_scope(
                    [
                        group
                        for group in groups
                        if group["heldout_body"] == body
                        and group["condition"] == condition
                    ],
                    scope=f"cell:{body}|{condition}",
                    include_intervals=False,
                )
                for body in BODIES
                for condition in CONDITIONS
            },
        },
        "interpretation_boundary": {
            "critic_auc_brier_mae_are_not_transfer_success_metrics": True,
            "no_auc_is_computed_or_accepted_by_this_report": True,
            "actor_training_receipt_contents_opened_by_this_evaluator": False,
            "critic_training_receipt_contents_opened_by_this_evaluator": False,
            "training_body_roster_completeness_must_be_verified_upstream": True,
            "delta_success_rate_requires_same_seed_paired_closed_loop_rollouts": True,
            "stage_progress_cannot_rescue_a_failed_binary_success_claim": True,
            "oracle_regret_requires_separate_executed_candidate_branches": True,
            "oracle_values_are_not_mixed_into_closed_loop_delta_success_rate": True,
        },
    }
    return {**base, "report_sha256": canonical_sha256(base)}


def _reject_constant(token: str) -> None:
    raise CrossEmbodimentReportError(f"non-finite JSON number is forbidden: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CrossEmbodimentReportError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CrossEmbodimentReportError("input is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CrossEmbodimentReportError("input must be a JSON object")
    return value


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if output.suffix.casefold() != ".json" or output.exists() or output.is_symlink():
        raise CrossEmbodimentReportError("output must be a new .json file")
    output.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = build_report(read_json(arguments.input))
    write_json_new(arguments.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "claim_type": report["transfer_claim"]["claim_type"],
                "pair_count": report["policy_evaluation"]["pair_count"],
                "n4_delta_sr": report["policy_evaluation"][
                    "global_equal_body_condition_macro"
                ]["n4_minus_n1"]["success"]["delta_right_minus_left"],
                "n8_delta_sr": report["policy_evaluation"][
                    "global_equal_body_condition_macro"
                ]["n8_minus_n1"]["success"]["delta_right_minus_left"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
