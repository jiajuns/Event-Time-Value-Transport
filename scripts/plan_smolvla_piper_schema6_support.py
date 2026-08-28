#!/usr/bin/env python3
"""Label-blind sample-size planner for SmolVLA->Piper schema-6 support gates.

The planner accepts aggregate counts only.  It has no dataset/HDF5 input and
does not inspect target, validation, or evaluation labels.  IID calculations
are explicitly reported as planning assumptions, while support that cannot be
identified from aggregate branch outcomes is returned as unknown rather than
silently imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


FORMAT = "etsf_smolvla_piper_schema6_label_blind_support_plan_v1"
STATUS = "preregistration_expansion_required_support_not_proved"
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation", "evaluation")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _integer(value: Any, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{role} must be an integer >= {minimum}")
    return value


def _probability(value: Any, role: str, *, open_zero: bool = False) -> float:
    result = float(value)
    lower_ok = result > 0 if open_zero else result >= 0
    if not math.isfinite(result) or not lower_ok or result > 1:
        bracket = "(0, 1]" if open_zero else "[0, 1]"
        raise ValueError(f"{role} must lie in {bracket}")
    return result


def binomial_interval_probability(
    trials: int, probability: float, lower: int, upper: int | None = None
) -> float:
    trials = _integer(trials, "binomial trials")
    lower = _integer(lower, "binomial lower bound")
    upper = trials if upper is None else _integer(upper, "binomial upper bound")
    probability = _probability(probability, "binomial probability")
    if lower > upper or lower > trials:
        return 0.0
    upper = min(upper, trials)
    if probability == 0:
        return float(lower == 0)
    if probability == 1:
        return float(lower <= trials <= upper)
    result = float(
        sum(
            math.comb(trials, count)
            * probability**count
            * (1 - probability) ** (trials - count)
            for count in range(lower, upper + 1)
        )
    )
    return min(1.0, max(0.0, result))


def _multinomial_term(counts: tuple[int, int, int], probs: tuple[float, float, float]) -> float:
    if any(count and probability == 0 for count, probability in zip(counts, probs)):
        return 0.0
    total = sum(counts)
    log_value = math.lgamma(total + 1) - sum(math.lgamma(count + 1) for count in counts)
    for count, probability in zip(counts, probs):
        if count:
            log_value += count * math.log(probability)
    return math.exp(log_value)


def group_category_probabilities(branch_success_probability: float, candidates: int) -> dict[str, float]:
    probability = _probability(branch_success_probability, "branch success probability")
    candidates = _integer(candidates, "candidates per group", minimum=2)
    all_failure = (1 - probability) ** candidates
    all_success = probability**candidates
    discordant = 1 - all_failure - all_success
    return {
        "all_failure": all_failure,
        "all_success": all_success,
        "discordant": discordant,
        "positive_group": 1 - all_failure,
        "negative_group": 1 - all_success,
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    successes = _integer(successes, "Wilson successes")
    total = _integer(total, "Wilson total", minimum=1)
    if successes > total or not math.isfinite(z) or z <= 0:
        raise ValueError("invalid Wilson interval inputs")
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def joint_group_support_probability(
    groups: int,
    categories: dict[str, float],
    *,
    min_positive: int,
    min_negative: int,
    min_discordant: int = 0,
) -> float:
    """Exact IID multinomial probability for group-level outcome support."""

    groups = _integer(groups, "groups")
    thresholds = tuple(
        _integer(value, role)
        for value, role in (
            (min_positive, "minimum positive groups"),
            (min_negative, "minimum negative groups"),
            (min_discordant, "minimum discordant groups"),
        )
    )
    all_failure = categories["all_failure"]
    all_success = categories["all_success"]
    discordant = categories["discordant"]
    total = 0.0
    for mixed in range(groups + 1):
        for success_only in range(groups - mixed + 1):
            failure_only = groups - mixed - success_only
            positive = mixed + success_only
            negative = mixed + failure_only
            if (
                positive < thresholds[0]
                or negative < thresholds[1]
                or mixed < thresholds[2]
            ):
                continue
            total += _multinomial_term(
                (mixed, success_only, failure_only),
                (discordant, all_success, all_failure),
            )
    return min(1.0, max(0.0, total))


def aggregate_group_bounds(
    *, groups: int, candidates: int, successes: int, failures: int
) -> dict[str, dict[str, int]]:
    """Exact feasible group-support bounds with no within-group assumptions."""

    groups = _integer(groups, "source groups", minimum=1)
    candidates = _integer(candidates, "candidates per group", minimum=2)
    successes = _integer(successes, "source successes")
    failures = _integer(failures, "source failures")
    if successes + failures != groups * candidates:
        raise ValueError("aggregate branches must equal groups*candidates")
    feasible: list[tuple[int, int, int]] = []
    for all_success_groups in range(groups + 1):
        for all_failure_groups in range(groups - all_success_groups + 1):
            mixed_groups = groups - all_success_groups - all_failure_groups
            mixed_successes = successes - all_success_groups * candidates
            mixed_failures = failures - all_failure_groups * candidates
            if mixed_groups == 0:
                valid = mixed_successes == 0 and mixed_failures == 0
            else:
                valid = (
                    mixed_groups <= mixed_successes <= mixed_groups * (candidates - 1)
                    and mixed_groups <= mixed_failures <= mixed_groups * (candidates - 1)
                    and mixed_successes + mixed_failures == mixed_groups * candidates
                )
            if valid:
                feasible.append(
                    (
                        mixed_groups + all_success_groups,
                        mixed_groups + all_failure_groups,
                        mixed_groups,
                    )
                )
    if not feasible:
        raise ValueError("aggregate branch counts have no feasible group arrangement")
    names = ("positive_groups", "negative_groups", "discordant_groups")
    return {
        name: {"minimum": min(row[index] for row in feasible), "maximum": max(row[index] for row in feasible)}
        for index, name in enumerate(names)
    }


def minimum_groups_for_joint_support(
    categories: dict[str, float],
    *,
    min_positive: int,
    min_negative: int,
    min_discordant: int,
    confidence: float,
    search_limit: int = 5000,
) -> dict[str, Any]:
    confidence = _probability(confidence, "planning confidence", open_zero=True)
    start = max(min_positive, min_negative, min_discordant)
    for groups in range(start, search_limit + 1):
        probability = joint_group_support_probability(
            groups,
            categories,
            min_positive=min_positive,
            min_negative=min_negative,
            min_discordant=min_discordant,
        )
        if probability >= confidence:
            return {"groups": groups, "probability": probability, "found": True}
    return {"groups": None, "probability": None, "found": False}


def balanced_recovery_probability(groups: int, minimum_per_class: int) -> float:
    """Illustrative upper-information scenario: exclusive balanced group labels."""

    groups = _integer(groups, "recovery groups")
    minimum_per_class = _integer(minimum_per_class, "recovery minimum", minimum=1)
    return binomial_interval_probability(
        groups, 0.5, minimum_per_class, groups - minimum_per_class
    )


def minimum_balanced_recovery_groups(
    minimum_per_class: int, confidence: float, search_limit: int = 5000
) -> dict[str, Any]:
    confidence = _probability(confidence, "recovery planning confidence", open_zero=True)
    for groups in range(2 * minimum_per_class, search_limit + 1):
        probability = balanced_recovery_probability(groups, minimum_per_class)
        if probability >= confidence:
            return {"groups": groups, "probability": probability, "found": True}
    return {"groups": None, "probability": None, "found": False}


def build_plan(
    *,
    source_groups: int = 63,
    candidates_per_group: int = 4,
    source_successes: int = 34,
    source_failures: int = 218,
    adaptation_bucket_groups: int = 80,
    internal_validation_groups: int = 20,
    target_validation_groups: int = 50,
    trainer_min_outcome_groups: int = 5,
    trainer_min_discordant_groups: int = 5,
    trainer_min_event_rows: int = 5,
    trainer_min_duration_rows: int = 5,
    recovery_min_groups_per_class: int = 10,
    calibrator_min_success_groups_per_side: int = 50,
    calibrator_min_event_groups_per_class: int = 10,
    calibrator_min_duration_groups_per_side: int = 10,
    confidence: float = 0.95,
    currently_available_target_groups: int = 1,
) -> dict[str, Any]:
    inputs = {
        "source_groups": _integer(source_groups, "source groups", minimum=1),
        "candidates_per_group": _integer(candidates_per_group, "candidates", minimum=2),
        "source_successes": _integer(source_successes, "source successes"),
        "source_failures": _integer(source_failures, "source failures"),
        "adaptation_bucket_groups": _integer(
            adaptation_bucket_groups, "adaptation bucket groups"
        ),
        "internal_validation_groups": _integer(internal_validation_groups, "internal validation groups"),
        "target_validation_groups": _integer(target_validation_groups, "target validation groups"),
        "trainer_min_outcome_groups": _integer(trainer_min_outcome_groups, "trainer outcome minimum"),
        "trainer_min_discordant_groups": _integer(trainer_min_discordant_groups, "trainer discordant minimum"),
        "trainer_min_event_rows": _integer(trainer_min_event_rows, "trainer event minimum"),
        "trainer_min_duration_rows": _integer(trainer_min_duration_rows, "trainer duration minimum"),
        "recovery_min_groups_per_class": _integer(recovery_min_groups_per_class, "recovery minimum", minimum=1),
        "calibrator_min_success_groups_per_side": _integer(calibrator_min_success_groups_per_side, "calibrator success minimum"),
        "calibrator_min_event_groups_per_class": _integer(calibrator_min_event_groups_per_class, "calibrator event minimum"),
        "calibrator_min_duration_groups_per_side": _integer(calibrator_min_duration_groups_per_side, "calibrator duration minimum"),
        "confidence": _probability(confidence, "confidence", open_zero=True),
        "currently_available_target_groups": _integer(currently_available_target_groups, "available target groups"),
    }
    if inputs["source_successes"] + inputs["source_failures"] != inputs["source_groups"] * inputs["candidates_per_group"]:
        raise ValueError("Source aggregate is inconsistent with groups*candidates")
    if inputs["internal_validation_groups"] >= inputs["adaptation_bucket_groups"]:
        raise ValueError("internal validation must be a strict subset of adaptation bucket")
    current_training_groups = (
        inputs["adaptation_bucket_groups"] - inputs["internal_validation_groups"]
    )
    branch_probability = inputs["source_successes"] / (
        inputs["source_successes"] + inputs["source_failures"]
    )
    categories = group_category_probabilities(
        branch_probability, inputs["candidates_per_group"]
    )
    wilson_lower, wilson_upper = wilson_interval(
        inputs["source_successes"],
        inputs["source_successes"] + inputs["source_failures"],
    )
    conservative_categories = group_category_probabilities(
        wilson_lower, inputs["candidates_per_group"]
    )
    arrangement_bounds = aggregate_group_bounds(
        groups=inputs["source_groups"],
        candidates=inputs["candidates_per_group"],
        successes=inputs["source_successes"],
        failures=inputs["source_failures"],
    )

    split_probabilities: dict[str, Any] = {}
    for name, groups in (
        ("adaptation_train", current_training_groups),
        ("adaptation_internal_validation", inputs["internal_validation_groups"]),
    ):
        outcome_probability = joint_group_support_probability(
            groups,
            categories,
            min_positive=inputs["trainer_min_outcome_groups"],
            min_negative=inputs["trainer_min_outcome_groups"],
            min_discordant=inputs["trainer_min_discordant_groups"],
        )
        # At least this many successful branches is necessary, but not
        # sufficient, for eK support.  Other event classes are unidentified.
        ek_necessary_upper = binomial_interval_probability(
            groups * inputs["candidates_per_group"],
            branch_probability,
            inputs["trainer_min_event_rows"],
        )
        split_probabilities[name] = {
            "iid_outcome_and_discordance_gate_probability": outcome_probability,
            "wilson95_lower_branch_rate_outcome_gate_probability": joint_group_support_probability(
                groups,
                conservative_categories,
                min_positive=inputs["trainer_min_outcome_groups"],
                min_negative=inputs["trainer_min_outcome_groups"],
                min_discordant=inputs["trainer_min_discordant_groups"],
            ),
            "event_all_classes_probability_identified": False,
            "event_all_classes_probability_bounds": [0.0, ek_necessary_upper],
            "event_upper_bound_reason": "successful branches are only a necessary eK support condition; nonterminal event frequencies are unknown",
            "duration_observed_and_censored_probability_identified": False,
            "duration_probability_bounds": [0.0, 1.0],
            "conditional_recovery_probability_identified": False,
            "conditional_recovery_probability_bounds": [0.0, 1.0],
            "conditional_recovery_balanced_exclusive_group_scenario": balanced_recovery_probability(
                groups, inputs["recovery_min_groups_per_class"]
            ),
            "mathematical_maximum_independent_groups_per_class": groups,
        }

    target_success_probability = joint_group_support_probability(
        inputs["target_validation_groups"],
        categories,
        min_positive=inputs["calibrator_min_success_groups_per_side"],
        min_negative=inputs["calibrator_min_success_groups_per_side"],
    )
    target_minimum = minimum_groups_for_joint_support(
        categories,
        min_positive=inputs["calibrator_min_success_groups_per_side"],
        min_negative=inputs["calibrator_min_success_groups_per_side"],
        min_discordant=0,
        confidence=inputs["confidence"],
    )
    conservative_target_minimum = minimum_groups_for_joint_support(
        conservative_categories,
        min_positive=inputs["calibrator_min_success_groups_per_side"],
        min_negative=inputs["calibrator_min_success_groups_per_side"],
        min_discordant=0,
        confidence=inputs["confidence"],
    )
    recovery_minimum = minimum_balanced_recovery_groups(
        inputs["recovery_min_groups_per_class"], inputs["confidence"]
    )
    current_v2_physical_total = (
        inputs["adaptation_bucket_groups"] + inputs["target_validation_groups"]
    )
    recommended_internal = max(
        inputs["internal_validation_groups"], int(recovery_minimum["groups"] or 0)
    )
    recommended_target = max(
        inputs["target_validation_groups"], int(target_minimum["groups"] or 0)
    )
    recommended_training = max(80, current_training_groups)
    recommended_adaptation_bucket = recommended_training + recommended_internal
    recommended_total = recommended_adaptation_bucket + recommended_target
    conservative_recommended_target = max(
        inputs["target_validation_groups"],
        int(conservative_target_minimum["groups"] or 0),
    )
    conservative_recommended_total = (
        recommended_adaptation_bucket + conservative_recommended_target
    )
    exact_fifty_contradiction = (
        inputs["target_validation_groups"]
        == inputs["calibrator_min_success_groups_per_side"]
    )
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "inputs": inputs,
        "label_access": {
            "dataset_paths_accepted": False,
            "hdf5_opened": 0,
            "target_labels_read": False,
            "validation_labels_read": False,
            "evaluation_labels_read": False,
            "aggregate_counts_only": True,
        },
        "source_aggregate": {
            "branch_success_probability": branch_probability,
            "branch_success_wilson95": [wilson_lower, wilson_upper],
            "iid_candidate_group_probabilities": categories,
            "wilson95_lower_branch_rate_group_probabilities": conservative_categories,
            "exact_group_arrangement_bounds_without_iid": arrangement_bounds,
            "cross_body_stationarity_proved": False,
        },
        "split_support": split_probabilities,
        "formal_target_validation": {
            "iid_success_head_support_probability": target_success_probability,
            "positive_groups_required": inputs["calibrator_min_success_groups_per_side"],
            "negative_groups_required": inputs["calibrator_min_success_groups_per_side"],
            "available_groups": inputs["target_validation_groups"],
            "all_groups_must_be_discordant_when_available_equals_each_side_minimum": exact_fifty_contradiction,
            "iid_probability_equals_discordant_probability_power_50": (
                categories["discordant"] ** inputs["target_validation_groups"]
                if exact_fifty_contradiction
                else None
            ),
            "wilson95_lower_branch_rate_success_head_support_probability": joint_group_support_probability(
                inputs["target_validation_groups"],
                conservative_categories,
                min_positive=inputs["calibrator_min_success_groups_per_side"],
                min_negative=inputs["calibrator_min_success_groups_per_side"],
            ),
            "event_all_classes_probability_identified": False,
            "event_minimum_independent_groups_per_class": inputs["calibrator_min_event_groups_per_class"],
            "event_all_classes_probability_bounds": [
                0.0,
                binomial_interval_probability(
                    inputs["target_validation_groups"],
                    categories["positive_group"],
                    inputs["calibrator_min_event_groups_per_class"],
                ),
            ],
            "event_upper_bound_reason": "at least ten success-positive groups is only a necessary eK condition; other event classes are unknown",
            "mathematical_maximum_event_support_groups_per_class": inputs[
                "target_validation_groups"
            ],
            "duration_probability_identified": False,
            "duration_probability_bounds": [0.0, 1.0],
            "duration_minimum_observed_and_censored_groups": inputs["calibrator_min_duration_groups_per_side"],
            "mathematical_maximum_duration_observed_groups": inputs[
                "target_validation_groups"
            ],
            "mathematical_maximum_duration_censored_groups": inputs[
                "target_validation_groups"
            ],
            "recovery_supported_by_versioned_target_calibrator_v2_contract": True,
            "recovery_probability_identified": False,
            "recovery_probability_bounds": [0.0, 1.0],
            "recovery_minimum_independent_groups_per_class": inputs[
                "recovery_min_groups_per_class"
            ],
            "recovery_activation_requires_all_five_trained_heads": True,
            "right_censored_nonrecoveries_count_as_negative": False,
        },
        "sample_size_plan": {
            "current_v2_physical_groups": current_v2_physical_total,
            "current_v2_adaptation_bucket_groups": inputs[
                "adaptation_bucket_groups"
            ],
            "current_v2_training_groups_inside_adaptation": current_training_groups,
            "current_v2_internal_validation_groups_inside_adaptation": inputs[
                "internal_validation_groups"
            ],
            "currently_available_target_groups": inputs["currently_available_target_groups"],
            "additional_groups_for_current_v2_physical_split": max(
                0,
                current_v2_physical_total
                - inputs["currently_available_target_groups"],
            ),
            "iid_minimum_target_validation_groups_for_success_support": target_minimum,
            "wilson95_lower_rate_minimum_target_validation_groups_for_success_support": conservative_target_minimum,
            "illustrative_minimum_internal_groups_for_balanced_exclusive_recovery": recovery_minimum,
            "iid_recommended_training_groups": recommended_training,
            "iid_recommended_internal_validation_groups": recommended_internal,
            "iid_recommended_adaptation_bucket_groups": recommended_adaptation_bucket,
            "iid_recommended_target_validation_groups": recommended_target,
            "iid_recommended_total_groups": recommended_total,
            "additional_groups_for_iid_recommended_plan": max(
                0, recommended_total - inputs["currently_available_target_groups"]
            ),
            "wilson95_lower_rate_recommended_target_validation_groups": conservative_recommended_target,
            "wilson95_lower_rate_recommended_total_groups": conservative_recommended_total,
            "additional_groups_for_wilson95_lower_rate_plan": max(
                0,
                conservative_recommended_total
                - inputs["currently_available_target_groups"],
            ),
            "operational_rounded_preregistration_target_groups": (
                int(math.ceil(conservative_recommended_total / 50.0)) * 50
            ),
            "event_duration_recovery_guarantee_from_success_aggregate": False,
            "required_collection_design": "preregister event-stratified, observed/censored-duration-stratified, and regress/recovery-enriched development groups; fail closed after split-specific support audit",
        },
        "decision": {
            "training_authorized": False,
            "preregister_expansion_required": True,
            "reason": "current target group count is below the physical v2 split total and formal target-validation50 success calibration is effectively unsupported under the Source63 IID planning model",
            "frozen_v2_protocol_modified": False,
        },
        "interpretation": {
            "iid_values_are_planning_scenarios_not_cross_body_evidence": True,
            "unknown_event_duration_recovery_rates_are_not_imputed": True,
            "no_finite_sample_guarantee_without_target_distribution_assumptions": True,
        },
    }
    result["plan_sha256"] = canonical_sha256(result)
    return result


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if any(
        token in component.casefold()
        for component in absolute.parts
        for token in SENSITIVE_PATH_TOKENS
    ):
        raise ValueError("output path is in a forbidden namespace")
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(absolute)
    absolute.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{absolute.name}.", suffix=".partial", dir=absolute.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, absolute)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-groups", type=int, default=63)
    parser.add_argument("--candidates-per-group", type=int, default=4)
    parser.add_argument("--source-successes", type=int, default=34)
    parser.add_argument("--source-failures", type=int, default=218)
    parser.add_argument("--adaptation-bucket-groups", type=int, default=80)
    parser.add_argument("--internal-validation-groups", type=int, default=20)
    parser.add_argument("--target-validation-groups", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--currently-available-target-groups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_plan(
        source_groups=args.source_groups,
        candidates_per_group=args.candidates_per_group,
        source_successes=args.source_successes,
        source_failures=args.source_failures,
        adaptation_bucket_groups=args.adaptation_bucket_groups,
        internal_validation_groups=args.internal_validation_groups,
        target_validation_groups=args.target_validation_groups,
        confidence=args.confidence,
        currently_available_target_groups=args.currently_available_target_groups,
    )
    if args.output is not None:
        write_json_new(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
