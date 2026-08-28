#!/usr/bin/env python3
"""Four-condition attribution and cross-embodiment evaluation400 v4.

The evaluator consumes only canonical records materialized after the sealed
audit terminal has completed.  It never opens simulator traces, HDF files, or
target envelopes.  Pooled results are diagnostic.  The evaluation400 records
authorize only the predeclared target-Piper strata; cross-embodiment promotion
also requires a separate content-addressed LOBO evidence authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import smolvla_piper_evaluation400_audit_contract_v1 as audit_v1


BUNDLE_FORMAT = "etsf_smolvla_piper_evaluation400_unsealed_audit_bundle_v4"
BUNDLE_STATUS = "complete_terminal_then_canonical_targets_unsealed"
RECORD_FORMAT = "etsf_smolvla_piper_evaluation400_canonical_condition_record_v4"
RECORD_STATUS = "complete_condition_target_recomputed_after_terminal"
RESULT_FORMAT = "etsf_smolvla_piper_evaluation400_four_condition_results_v4"
RESULT_STATUS = "complete_attribution_and_cross_embodiment_audit"
PAIR_COUNT = 400
CONDITION_COUNT = 1600
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_SAMPLES = 20_000
HOLM_ALPHA = 0.05
MINIMUM_CHANGED_PAIRS = 50
MINIMUM_CHANGED_PAIRS_PER_SLICE = 10
MINIMUM_DISCORDANT_PAIRS = 20
MINIMUM_DISCORDANT_PAIRS_PER_SLICE = 5
MINIMUM_SLICE_PAIRS = 50
MAXIMUM_HARMFUL_RATE = 0.10
LOBO_EVIDENCE_AUTHORITY_FORMAT = (
    "etsf_lobo_six_head_task_evidence_authority_v1"
)
LOBO_EVIDENCE_AUTHORITY_STATUS = (
    "complete_all_predeclared_lobo_slices_independently_evaluated"
)
CONDITIONS = (
    "baseline",
    "success_only_guarded",
    "composite_rank_ungated",
    "etsf",
)
CONDITION_POSITION = {name: index for index, name in enumerate(CONDITIONS)}
COMPARISONS = (
    ("primary_etsf_minus_baseline", "etsf", "baseline", "primary"),
    (
        "secondary_success_only_guarded_minus_baseline",
        "success_only_guarded",
        "baseline",
        "secondary",
    ),
    (
        "secondary_composite_rank_ungated_minus_baseline",
        "composite_rank_ungated",
        "baseline",
        "secondary",
    ),
    (
        "secondary_etsf_minus_success_only_guarded",
        "etsf",
        "success_only_guarded",
        "secondary",
    ),
    (
        "secondary_etsf_minus_composite_rank_ungated",
        "etsf",
        "composite_rank_ungated",
        "secondary",
    ),
)
HEADS = (
    "post_event",
    "next_event",
    "duration",
    "success",
    "recovery",
    "object_effect",
)
SHA_CHARS = frozenset("0123456789abcdef")


class Evaluation400V4Error(RuntimeError):
    """The v4 audit/result contract failed closed."""


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _exact_int(value: Any, role: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Evaluation400V4Error(f"{role} must be an exact integer")
    return value


def _finite(value: Any, role: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Evaluation400V4Error(f"{role} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise Evaluation400V4Error(f"{role} must be finite and valid")
    return result


def _exact_bool(value: Any, expected: bool, role: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise Evaluation400V4Error(f"{role} must be exact {expected!r}")
    return value


def _verify_signed(
    value: Mapping[str, Any], field: str, expected_fields: set[str], role: str,
) -> str:
    if not isinstance(value, Mapping) or set(value) != expected_fields | {field}:
        raise Evaluation400V4Error(f"{role} fields changed")
    logical = value.get(field)
    if not is_sha256(logical):
        raise Evaluation400V4Error(f"{role} logical SHA is invalid")
    unsigned = {key: child for key, child in value.items() if key != field}
    if logical != canonical_sha256(unsigned):
        raise Evaluation400V4Error(f"{role} canonical SHA mismatch")
    return str(logical)


def _numeric_array(
    value: Any, shape: tuple[int, ...], role: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.dtype == np.dtype(bool)
        or raw.shape != shape
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise Evaluation400V4Error(f"{role} numeric shape changed")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise Evaluation400V4Error(f"{role} contains non-finite values")
    return result


def _probability_vector(value: Any, role: str) -> list[float]:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) < 2:
        raise Evaluation400V4Error(f"{role} probability shape changed")
    result = _numeric_array(raw, raw.shape, role)
    if (
        bool((result < 0.0).any())
        or not math.isclose(float(result.sum()), 1.0, rel_tol=1e-10, abs_tol=1e-10)
    ):
        raise Evaluation400V4Error(f"{role} is not a probability vector")
    return result.tolist()


def _binary_probability(value: Any, role: str) -> float:
    result = _finite(value, role)
    if not 0.0 <= result <= 1.0:
        raise Evaluation400V4Error(f"{role} escaped [0,1]")
    return result


def _target(value: Any, classes: int, role: str) -> int:
    result = _exact_int(value, role)
    if result >= classes:
        raise Evaluation400V4Error(f"{role} is out of range")
    return result


def _validate_head_record(
    head: str, value: Mapping[str, Any], *, event_classes: int | None,
    object_dimension: int | None,
) -> tuple[dict[str, Any], int | None, int | None]:
    if not isinstance(value, Mapping):
        raise Evaluation400V4Error(f"{head} record must be a mapping")
    if head in {"post_event", "next_event"}:
        expected = {
            "probability", "target", "baseline_probability", "applicable",
            "observed", "censored", "required_classes",
        }
        if set(value) != expected:
            raise Evaluation400V4Error(f"{head} fields changed")
        probability = _probability_vector(value["probability"], f"{head} probability")
        baseline = _probability_vector(
            value["baseline_probability"], f"{head} baseline probability"
        )
        classes = len(probability)
        if len(baseline) != classes or (
            event_classes is not None and classes != event_classes
        ):
            raise Evaluation400V4Error("event class inventory changed")
        target = _target(value["target"], classes, f"{head} target")
        required = value["required_classes"]
        if (
            not isinstance(required, list)
            or not required
            or any(type(item) is not int for item in required)
            or len(set(required)) != len(required)
            or any(item < 0 or item >= classes for item in required)
        ):
            raise Evaluation400V4Error(f"{head} required classes changed")
        _exact_bool(value["applicable"], True, f"{head} applicability")
        observed = value["observed"]
        censored = value["censored"]
        if type(observed) is not bool or type(censored) is not bool:
            raise Evaluation400V4Error(f"{head} masks must be exact bool")
        if head == "post_event":
            if observed is not True or censored is not False:
                raise Evaluation400V4Error("post-event must be observed")
        elif censored is not (not observed):
            raise Evaluation400V4Error("next-event censoring changed")
        return ({
            "probability": probability,
            "target": target,
            "baseline_probability": baseline,
            "observed": observed,
            "required_classes": list(required),
        }, classes, object_dimension)
    if head in {"success", "recovery"}:
        expected = {
            "probability", "target", "baseline_probability", "applicable",
            "observed", "censored",
        }
        if set(value) != expected:
            raise Evaluation400V4Error(f"{head} fields changed")
        applicable = value["applicable"]
        observed = value["observed"]
        censored = value["censored"]
        if any(type(item) is not bool for item in (applicable, observed, censored)):
            raise Evaluation400V4Error(f"{head} masks must be exact bool")
        if observed and not applicable:
            raise Evaluation400V4Error(f"{head} observed outside applicability")
        if censored is not (applicable and not observed):
            raise Evaluation400V4Error(f"{head} censoring changed")
        if head == "success" and (not applicable or not observed or censored):
            raise Evaluation400V4Error("success must be applicable and observed")
        target = _target(value["target"], 2, f"{head} target")
        if not observed and target != 0:
            raise Evaluation400V4Error(f"{head} unobserved target sentinel changed")
        row = {
            "probability": _binary_probability(
                value["probability"], f"{head} probability"
            ),
            "target": target,
            "baseline_probability": _binary_probability(
                value["baseline_probability"], f"{head} baseline"
            ),
            "observed": observed,
        }
        if head == "recovery":
            row["applicable"] = applicable
        return row, event_classes, object_dimension
    if head == "duration":
        expected = {
            "member_log_mean", "member_log_scale", "target", "observed",
            "applicable", "censored", "baseline_location", "baseline_scale",
        }
        if set(value) != expected:
            raise Evaluation400V4Error("duration fields changed")
        mean = _numeric_array(
            value["member_log_mean"], (audit_v1.MEMBER_COUNT,),
            "duration member mean",
        )
        scale = _numeric_array(
            value["member_log_scale"], (audit_v1.MEMBER_COUNT,),
            "duration member scale",
        )
        applicable = value["applicable"]
        observed = value["observed"]
        censored = value["censored"]
        if (
            applicable is not True
            or type(applicable) is not bool
            or type(observed) is not bool
            or type(censored) is not bool
            or censored is not (not observed)
        ):
            raise Evaluation400V4Error("duration masks changed")
        target = _finite(value["target"], "duration target")
        location = _finite(value["baseline_location"], "duration baseline location")
        baseline_scale = _finite(
            value["baseline_scale"], "duration baseline scale", positive=True
        )
        if target < 0.0:
            raise Evaluation400V4Error("duration target is negative")
        return ({
            "member_log_mean": mean.tolist(),
            "member_log_scale": scale.tolist(),
            "target": target,
            "observed": observed,
            "baseline_location": location,
            "baseline_scale": baseline_scale,
        }, event_classes, object_dimension)
    if head == "object_effect":
        expected = {
            "member_mean", "member_log_scale", "target", "baseline_robust",
            "applicable", "observed", "censored", "missing",
        }
        if set(value) != expected:
            raise Evaluation400V4Error("object-effect fields changed")
        raw = np.asarray(value["member_mean"])
        if raw.ndim != 2 or raw.shape[0] != audit_v1.MEMBER_COUNT or raw.shape[1] < 1:
            raise Evaluation400V4Error("object member mean shape changed")
        dimension = int(raw.shape[1])
        if object_dimension is not None and dimension != object_dimension:
            raise Evaluation400V4Error("object dimension changed")
        mean = _numeric_array(raw, (audit_v1.MEMBER_COUNT, dimension), "object mean")
        log_scale = _numeric_array(
            value["member_log_scale"], (audit_v1.MEMBER_COUNT, dimension),
            "object log scale",
        )
        target = _numeric_array(value["target"], (dimension,), "object target")
        robust = _numeric_array(
            value["baseline_robust"], (dimension,), "object robust baseline"
        )
        applicable = value["applicable"]
        observed = value["observed"]
        censored = value["censored"]
        missing = value["missing"]
        if (
            type(applicable) is not bool
            or applicable is not True
            or type(observed) is not bool
            or type(censored) is not bool
            or censored is not False
            or type(missing) is not bool
            or missing is not (not observed)
        ):
            raise Evaluation400V4Error("object masks changed")
        return ({
            "member_mean": mean.tolist(),
            "member_log_scale": log_scale.tolist(),
            "target": target.tolist(),
            "observed": observed,
            "baseline_robust": robust.tolist(),
        }, event_classes, dimension)
    raise Evaluation400V4Error("unknown six-head record")


RECORD_FIELDS = {
    "format", "status", "protocol_core_v4_sha256", "pair_id", "pair_ordinal",
    "condition_id", "condition_position", "embodiment_id", "embodiment_role",
    "policy_id", "shared_snapshot_sha256", "candidate_registry_sha256",
    "root_prediction_commit_sha256", "candidate_count",
    "baseline_candidate_index", "selected_candidate_index",
    "changed_from_baseline", "terminal_success", "six_head",
    "target_recomputed_by_audit_contract_v1", "target_unsealed_after_terminal",
}


def validate_record(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    logical = _verify_signed(value, "record_sha256", RECORD_FIELDS, "condition record")
    if value.get("format") != RECORD_FORMAT or value.get("status") != RECORD_STATUS:
        raise Evaluation400V4Error("condition record format/status changed")
    for field in (
        "protocol_core_v4_sha256", "pair_id", "shared_snapshot_sha256",
        "candidate_registry_sha256", "root_prediction_commit_sha256",
    ):
        if not is_sha256(value.get(field)):
            raise Evaluation400V4Error(f"condition record {field} changed")
    ordinal = _exact_int(value.get("pair_ordinal"), "pair ordinal")
    if ordinal >= PAIR_COUNT:
        raise Evaluation400V4Error("pair ordinal escaped evaluation400")
    condition = value.get("condition_id")
    if condition not in CONDITIONS:
        raise Evaluation400V4Error("condition inventory changed")
    position = _exact_int(value.get("condition_position"), "condition position")
    if position != CONDITION_POSITION[condition]:
        raise Evaluation400V4Error("condition order changed")
    for field in ("embodiment_id", "policy_id"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise Evaluation400V4Error(f"{field} must be nonempty")
    if value.get("embodiment_role") not in {"lobo", "target_piper"}:
        raise Evaluation400V4Error("embodiment role changed")
    candidate_count = _exact_int(
        value.get("candidate_count"), "candidate count", minimum=2
    )
    baseline = _exact_int(value.get("baseline_candidate_index"), "baseline index")
    selected = _exact_int(value.get("selected_candidate_index"), "selected index")
    if baseline >= candidate_count or selected >= candidate_count:
        raise Evaluation400V4Error("candidate selection escaped registry")
    changed = value.get("changed_from_baseline")
    if type(changed) is not bool or changed is not (selected != baseline):
        raise Evaluation400V4Error("changed-from-baseline proof changed")
    if condition == "baseline" and (selected != baseline or changed):
        raise Evaluation400V4Error("baseline condition changed candidate")
    if type(value.get("terminal_success")) is not bool:
        raise Evaluation400V4Error("terminal success must be exact bool")
    _exact_bool(
        value.get("target_recomputed_by_audit_contract_v1"), True,
        "target recomputation proof",
    )
    _exact_bool(
        value.get("target_unsealed_after_terminal"), True,
        "target unseal chronology",
    )
    six_head = value.get("six_head")
    if not isinstance(six_head, Mapping) or set(six_head) != set(HEADS):
        raise Evaluation400V4Error("six-head record inventory changed")
    normalized: dict[str, Any] = {}
    classes = None
    object_dimension = None
    for head in HEADS:
        normalized[head], classes, object_dimension = _validate_head_record(
            head,
            six_head[head],
            event_classes=classes,
            object_dimension=object_dimension,
        )
    if six_head["next_event"]["observed"] is not six_head["duration"]["observed"]:
        raise Evaluation400V4Error("next-event/duration censoring diverged")
    if normalized["success"]["target"] != int(value["terminal_success"]):
        raise Evaluation400V4Error("success-head target differs from terminal result")
    return ({**{key: value[key] for key in RECORD_FIELDS if key != "six_head"},
             "six_head": normalized, "record_sha256": logical}, logical)


SLICE_FIELDS = {
    "embodiment_id", "embodiment_role", "policy_id", "expected_pair_count"
}
BUNDLE_FIELDS = {
    "format", "status", "protocol_core_v4_sha256",
    "audit_contract_v1_implementation_file_sha256", "terminal_completeness",
    "required_slice_inventory", "records", "targets_unsealed_after_terminal",
    "hdf_or_trajectory_files_opened", "evaluation400_subset_excluded",
    "lobo_evidence_authority",
}

LOBO_EVIDENCE_AUTHORITY_FIELDS = {
    "format", "status", "source_result_sha256", "source_result_file_sha256",
    "source_evaluator_implementation_file_sha256",
    "source_audit_contract_v1_implementation_file_sha256",
    "predeclared_slice_inventory_sha256", "pair_count",
    "primary_task_success_promotion_passed",
    "six_head_accuracy_promotion_passed",
    "every_predeclared_embodiment_policy_slice_passed",
    "pooled_result_used_for_promotion",
}


def _validate_lobo_evidence_authority(
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None
    logical = _verify_signed(
        value,
        "authority_sha256",
        LOBO_EVIDENCE_AUTHORITY_FIELDS,
        "LOBO evidence authority",
    )
    if (
        value.get("format") != LOBO_EVIDENCE_AUTHORITY_FORMAT
        or value.get("status") != LOBO_EVIDENCE_AUTHORITY_STATUS
    ):
        raise Evaluation400V4Error("LOBO evidence authority format/status changed")
    for field in (
        "source_result_sha256",
        "source_result_file_sha256",
        "source_evaluator_implementation_file_sha256",
        "source_audit_contract_v1_implementation_file_sha256",
        "predeclared_slice_inventory_sha256",
    ):
        if not is_sha256(value.get(field)):
            raise Evaluation400V4Error(f"LOBO authority {field} changed")
    _exact_int(value.get("pair_count"), "LOBO evidence pair count", minimum=50)
    for field in (
        "primary_task_success_promotion_passed",
        "six_head_accuracy_promotion_passed",
        "every_predeclared_embodiment_policy_slice_passed",
    ):
        _exact_bool(value.get(field), True, f"LOBO authority {field}")
    _exact_bool(
        value.get("pooled_result_used_for_promotion"),
        False,
        "LOBO authority pooled-result policy",
    )
    return {**dict(value), "authority_sha256": logical}


def validate_bundle(
    value: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    logical = _verify_signed(value, "bundle_sha256", BUNDLE_FIELDS, "v4 audit bundle")
    if value.get("format") != BUNDLE_FORMAT or value.get("status") != BUNDLE_STATUS:
        raise Evaluation400V4Error("v4 audit bundle format/status changed")
    protocol_sha = value.get("protocol_core_v4_sha256")
    if not is_sha256(protocol_sha):
        raise Evaluation400V4Error("protocol core v4 SHA changed")
    implementation_path = Path(audit_v1.__file__).resolve()
    if value.get("audit_contract_v1_implementation_file_sha256") != file_sha256(
        implementation_path
    ):
        raise Evaluation400V4Error("audit contract v1 implementation changed")
    try:
        audit_v1.validate_terminal_completeness(value.get("terminal_completeness"))
    except audit_v1.AuditContractError as error:
        raise Evaluation400V4Error(str(error)) from error
    _exact_bool(
        value.get("targets_unsealed_after_terminal"), True,
        "bundle target chronology",
    )
    _exact_int(
        value.get("hdf_or_trajectory_files_opened"),
        "HDF/trajectory files opened",
    )
    if value["hdf_or_trajectory_files_opened"] != 0:
        raise Evaluation400V4Error("raw trajectory/HDF access is forbidden")
    _exact_bool(
        value.get("evaluation400_subset_excluded"), False,
        "evaluation400 subset exclusion",
    )
    inventory = value.get("required_slice_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise Evaluation400V4Error("required slice inventory missing")
    slices: dict[tuple[str, str], dict[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, Mapping) or set(row) != SLICE_FIELDS:
            raise Evaluation400V4Error("required slice fields changed")
        embodiment = row.get("embodiment_id")
        policy = row.get("policy_id")
        role = row.get("embodiment_role")
        count = _exact_int(row.get("expected_pair_count"), "slice pair count", minimum=1)
        if (
            not isinstance(embodiment, str)
            or not embodiment
            or not isinstance(policy, str)
            or not policy
            or role not in {"lobo", "target_piper"}
            or (embodiment, policy) in slices
            or count < MINIMUM_SLICE_PAIRS
        ):
            raise Evaluation400V4Error("required slice inventory changed")
        slices[(embodiment, policy)] = dict(row)
    if sum(row["expected_pair_count"] for row in slices.values()) != PAIR_COUNT:
        raise Evaluation400V4Error("predeclared evaluation400 slice coverage changed")
    if not any(
        row["embodiment_role"] == "target_piper" for row in slices.values()
    ):
        raise Evaluation400V4Error("target-Piper preregistered strata are missing")
    lobo_authority = _validate_lobo_evidence_authority(
        value.get("lobo_evidence_authority")
    )
    raw_records = value.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != CONDITION_COUNT:
        raise Evaluation400V4Error("exact 1600 condition records required")
    records = []
    record_shas: set[str] = set()
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    global_event_class_count: int | None = None
    global_object_dimension: int | None = None
    global_required_classes: dict[str, tuple[int, ...]] = {}
    for raw in raw_records:
        record, record_sha = validate_record(raw)
        if record_sha in record_shas:
            raise Evaluation400V4Error("duplicate condition record SHA")
        record_shas.add(record_sha)
        if record["protocol_core_v4_sha256"] != protocol_sha:
            raise Evaluation400V4Error("condition escaped protocol core")
        event_class_count = len(record["six_head"]["post_event"]["probability"])
        object_dimension = len(record["six_head"]["object_effect"]["target"])
        if global_event_class_count is None:
            global_event_class_count = event_class_count
            global_object_dimension = object_dimension
            global_required_classes = {
                head: tuple(record["six_head"][head]["required_classes"])
                for head in ("post_event", "next_event")
            }
        elif (
            event_class_count != global_event_class_count
            or object_dimension != global_object_dimension
            or any(
                tuple(record["six_head"][head]["required_classes"])
                != global_required_classes[head]
                for head in ("post_event", "next_event")
            )
        ):
            raise Evaluation400V4Error("global six-head inventory changed")
        records.append(record)
        pair_rows.setdefault(record["pair_id"], []).append(record)
    if len(pair_rows) != PAIR_COUNT:
        raise Evaluation400V4Error("exact 400 pair identities required")
    ordinals: set[int] = set()
    observed_slice_pairs: dict[tuple[str, str], int] = {
        key: 0 for key in slices
    }
    for pair_id, rows in pair_rows.items():
        rows.sort(key=lambda row: row["condition_position"])
        if [row["condition_id"] for row in rows] != list(CONDITIONS):
            raise Evaluation400V4Error("pair lacks exact four-condition order")
        anchor = rows[0]
        invariant_fields = (
            "pair_ordinal", "embodiment_id", "embodiment_role", "policy_id",
            "shared_snapshot_sha256", "candidate_registry_sha256",
            "root_prediction_commit_sha256", "candidate_count",
            "baseline_candidate_index",
        )
        if any(
            row[field] != anchor[field]
            for row in rows[1:]
            for field in invariant_fields
        ):
            raise Evaluation400V4Error("pair four-condition invariant changed")
        ordinal = anchor["pair_ordinal"]
        if ordinal in ordinals:
            raise Evaluation400V4Error("pair ordinal duplicated")
        ordinals.add(ordinal)
        key = (anchor["embodiment_id"], anchor["policy_id"])
        if key not in slices or slices[key]["embodiment_role"] != anchor[
            "embodiment_role"
        ]:
            raise Evaluation400V4Error("record escaped required slice")
        observed_slice_pairs[key] += 1
    if ordinals != set(range(PAIR_COUNT)):
        raise Evaluation400V4Error("pair ordinals do not cover 0..399")
    if any(
        observed_slice_pairs[key] != row["expected_pair_count"]
        for key, row in slices.items()
    ):
        raise Evaluation400V4Error("required slice pair counts changed")
    records.sort(key=lambda row: (row["pair_ordinal"], row["condition_position"]))
    return records, logical, lobo_authority


def _binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(successes + 1))
    return min(1.0, 2.0 * tail / (2.0**trials))


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["mcnemar_exact_p"])
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * row["mcnemar_exact_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["holm_reject_at_0_05"] = bool(running <= HOLM_ALPHA)


def _task_comparison(
    records: Sequence[Mapping[str, Any]], model: str, comparator: str,
    *, name: str, family: str, bootstrap_samples: int, bootstrap_seed: int,
    slice_mode: bool,
) -> dict[str, Any]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(row["pair_id"], {})[row["condition_id"]] = row
    pair_ids = np.asarray(sorted(by_pair), dtype=str)
    model_success = np.asarray(
        [by_pair[pair][model]["terminal_success"] for pair in pair_ids],
        dtype=np.int64,
    )
    comparator_success = np.asarray(
        [by_pair[pair][comparator]["terminal_success"] for pair in pair_ids],
        dtype=np.int64,
    )
    model_candidate = np.asarray(
        [by_pair[pair][model]["selected_candidate_index"] for pair in pair_ids],
        dtype=np.int64,
    )
    comparator_candidate = np.asarray(
        [by_pair[pair][comparator]["selected_candidate_index"] for pair in pair_ids],
        dtype=np.int64,
    )
    changed = model_candidate != comparator_candidate
    gain = model_success - comparator_success
    helpful = changed & (gain > 0)
    harmful = changed & (gain < 0)
    discordant = helpful | harmful
    try:
        gain_ci = audit_v1.pair_cluster_bootstrap(
            gain, pair_ids, samples=bootstrap_samples, seed=bootstrap_seed
        )
        coverage_ci = audit_v1.pair_cluster_bootstrap(
            changed.astype(np.float64), pair_ids,
            samples=bootstrap_samples, seed=bootstrap_seed + 1,
        )
        harmful_ci = audit_v1.pair_cluster_bootstrap(
            harmful.astype(np.float64), pair_ids, observed=changed,
            samples=bootstrap_samples, seed=bootstrap_seed + 2,
        )
    except audit_v1.AuditContractError as error:
        raise Evaluation400V4Error(str(error)) from error
    helpful_count = int(helpful.sum())
    harmful_count = int(harmful.sum())
    discordant_count = int(discordant.sum())
    p_value = _binomial_two_sided(
        min(helpful_count, harmful_count), discordant_count
    )
    changed_minimum = (
        MINIMUM_CHANGED_PAIRS_PER_SLICE if slice_mode else MINIMUM_CHANGED_PAIRS
    )
    discordant_minimum = (
        MINIMUM_DISCORDANT_PAIRS_PER_SLICE
        if slice_mode else MINIMUM_DISCORDANT_PAIRS
    )
    passed = bool(
        len(pair_ids) >= (MINIMUM_SLICE_PAIRS if slice_mode else PAIR_COUNT)
        and int(changed.sum()) >= changed_minimum
        and discordant_count >= discordant_minimum
        and gain_ci["status"] == "complete"
        and gain_ci["lower"] is not None
        and float(gain_ci["lower"]) > 0.0
        and harmful_ci["status"] == "complete"
        and harmful_ci["upper"] is not None
        and float(harmful_ci["upper"]) <= MAXIMUM_HARMFUL_RATE
        and p_value <= HOLM_ALPHA
    )
    return {
        "name": name,
        "family": family,
        "model_condition": model,
        "comparator_condition": comparator,
        "pair_count": int(len(pair_ids)),
        "model_success_count": int(model_success.sum()),
        "model_success_rate": float(model_success.mean()),
        "comparator_success_count": int(comparator_success.sum()),
        "comparator_success_rate": float(comparator_success.mean()),
        "paired_delta": float(gain.mean()),
        "paired_delta_pair_cluster_bootstrap": gain_ci,
        "changed_pair_count": int(changed.sum()),
        "change_coverage": float(changed.mean()),
        "change_coverage_pair_cluster_bootstrap": coverage_ci,
        "helpful_pair_count": helpful_count,
        "harmful_pair_count": harmful_count,
        "discordant_pair_count": discordant_count,
        "harmful_rate_among_changed": (
            float(harmful_count / int(changed.sum())) if bool(changed.any()) else None
        ),
        "harmful_rate_pair_cluster_bootstrap": harmful_ci,
        "mcnemar_exact_p": p_value,
        "minimum_changed_pairs": changed_minimum,
        "minimum_discordant_pairs": discordant_minimum,
        "maximum_harmful_rate": MAXIMUM_HARMFUL_RATE,
        "promotion_gate_passed": passed if family == "primary" else False,
    }


def _six_head_input(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in records]
    result: dict[str, Any] = {"pair_id": pair_ids}
    for head in HEADS:
        rows = [row["six_head"][head] for row in records]
        result[head] = {}
        for key in rows[0]:
            if key == "required_classes":
                if any(row[key] != rows[0][key] for row in rows[1:]):
                    raise Evaluation400V4Error(
                        f"{head} required-class inventory drifted within scope"
                    )
                result[head][key] = list(rows[0][key])
            else:
                result[head][key] = [row[key] for row in rows]
    return result


def _six_head_noninferiority(metrics: Mapping[str, Any]) -> dict[str, Any]:
    heads = metrics.get("heads") if isinstance(metrics, Mapping) else None
    gates: dict[str, bool] = {}
    if not isinstance(heads, Mapping):
        return {"passed": False, "heads": gates}
    required_intervals = {
        "post_event": (
            "nll_gain_baseline_minus_model",
            "accuracy_gain_model_minus_baseline",
        ),
        "next_event": (
            "nll_gain_baseline_minus_model",
            "accuracy_gain_model_minus_baseline",
        ),
        "success": (
            "nll_gain_baseline_minus_model",
            "brier_gain_baseline_minus_model",
        ),
        "recovery": (
            "nll_gain_baseline_minus_model",
            "brier_gain_baseline_minus_model",
        ),
        "duration": (
            "nll_gain_baseline_minus_model",
            "mae_gain_baseline_minus_model",
        ),
        "object_effect": (
            "zero_l2_gain_baseline_minus_model",
            "robust_l2_gain_baseline_minus_model",
        ),
    }
    for head, fields in required_intervals.items():
        row = heads.get(head)
        gates[head] = bool(
            isinstance(row, Mapping)
            and row.get("status") == "complete"
            and all(
                isinstance(row.get(field), Mapping)
                and row[field].get("status") == "complete"
                and row[field].get("lower") is not None
                and float(row[field]["lower"]) >= 0.0
                for field in fields
            )
        )
    return {"passed": all(gates.values()), "heads": gates}


def _evaluate_scope(
    records: Sequence[Mapping[str, Any]], *, scope_name: str,
    bootstrap_samples: int, bootstrap_seed: int, slice_mode: bool,
) -> dict[str, Any]:
    comparisons = [
        _task_comparison(
            records,
            model,
            comparator,
            name=name,
            family=family,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 100,
            slice_mode=slice_mode,
        )
        for index, (name, model, comparator, family) in enumerate(COMPARISONS)
    ]
    secondary = [row for row in comparisons if row["family"] == "secondary"]
    _holm_adjust(secondary)
    primary = next(row for row in comparisons if row["family"] == "primary")
    primary["holm_adjusted_p"] = None
    primary["holm_reject_at_0_05"] = None
    primary["multiplicity_policy"] = "preregistered_primary_not_in_secondary_family"
    metrics_by_condition = {}
    gates_by_condition = {}
    for index, condition in enumerate(CONDITIONS):
        condition_records = [row for row in records if row["condition_id"] == condition]
        try:
            metrics = audit_v1.compute_six_head_metrics(
                _six_head_input(condition_records),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 1000 + index * 100,
            )
        except audit_v1.AuditContractError as error:
            raise Evaluation400V4Error(str(error)) from error
        metrics_by_condition[condition] = metrics
        gates_by_condition[condition] = _six_head_noninferiority(metrics)
    pair_ids = sorted({row["pair_id"] for row in records})
    return {
        "scope": scope_name,
        "pair_count": len(pair_ids),
        "comparison_order": [row[0] for row in COMPARISONS],
        "comparisons": comparisons,
        "secondary_holm_family_size": 4,
        "holm_alpha": HOLM_ALPHA,
        "six_head_metrics_by_condition": metrics_by_condition,
        "six_head_noninferiority_by_condition": gates_by_condition,
        "primary_task_gate_passed": primary["promotion_gate_passed"],
        "etsf_six_head_gate_passed": gates_by_condition["etsf"]["passed"],
        "promotion_gate_passed": bool(
            primary["promotion_gate_passed"]
            and gates_by_condition["etsf"]["passed"]
        ),
    }


def evaluate_bundle(
    value: Mapping[str, Any], *, bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_lobo_evidence_authority_sha256: str | None = None,
) -> dict[str, Any]:
    _exact_int(bootstrap_samples, "bootstrap samples", minimum=100)
    _exact_int(bootstrap_seed, "bootstrap seed")
    records, bundle_sha, lobo_authority = validate_bundle(value)
    if expected_lobo_evidence_authority_sha256 is not None:
        if not is_sha256(expected_lobo_evidence_authority_sha256):
            raise Evaluation400V4Error("expected LOBO evidence authority SHA changed")
        if (
            lobo_authority is None
            or lobo_authority["authority_sha256"]
            != expected_lobo_evidence_authority_sha256
        ):
            raise Evaluation400V4Error(
                "LOBO evidence authority differs from external content address"
            )
    lobo_authority_externally_bound = bool(
        lobo_authority is not None
        and expected_lobo_evidence_authority_sha256
        == lobo_authority["authority_sha256"]
    )
    pooled = _evaluate_scope(
        records,
        scope_name="pooled_diagnostic_only",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        slice_mode=False,
    )
    role_reports = {}
    for role in sorted({row["embodiment_role"] for row in records}):
        selected = [row for row in records if row["embodiment_role"] == role]
        role_pair_count = len({row["pair_id"] for row in selected})
        role_reports[role] = _evaluate_scope(
            selected,
            scope_name=f"embodiment_role:{role}",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + (10_000 if role == "lobo" else 20_000),
            slice_mode=role_pair_count != PAIR_COUNT,
        )
    embodiment_reports = {}
    for index, embodiment in enumerate(sorted({row["embodiment_id"] for row in records})):
        selected = [row for row in records if row["embodiment_id"] == embodiment]
        embodiment_pair_count = len({row["pair_id"] for row in selected})
        embodiment_reports[embodiment] = _evaluate_scope(
            selected,
            scope_name=f"embodiment:{embodiment}",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 30_000 + index * 1000,
            slice_mode=embodiment_pair_count != PAIR_COUNT,
        )
    slice_reports = {}
    slice_keys = sorted({(row["embodiment_id"], row["policy_id"]) for row in records})
    for index, (embodiment, policy) in enumerate(slice_keys):
        selected = [
            row for row in records
            if row["embodiment_id"] == embodiment and row["policy_id"] == policy
        ]
        key = f"{embodiment}::{policy}"
        slice_reports[key] = _evaluate_scope(
            selected,
            scope_name=f"embodiment_policy:{key}",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 50_000 + index * 1000,
            slice_mode=True,
        )
    target_records = [
        row for row in records if row["embodiment_role"] == "target_piper"
    ]
    target_embodiments = sorted({row["embodiment_id"] for row in target_records})
    target_slices = sorted({
        f"{row['embodiment_id']}::{row['policy_id']}" for row in target_records
    })
    target_task_failures = []
    target_head_failures = []
    target_role = role_reports.get("target_piper")
    if target_role is None or not target_role["primary_task_gate_passed"]:
        target_task_failures.append("embodiment_role:target_piper")
    if target_role is None or not target_role["etsf_six_head_gate_passed"]:
        target_head_failures.append("embodiment_role:target_piper")
    for embodiment in target_embodiments:
        if not embodiment_reports[embodiment]["primary_task_gate_passed"]:
            target_task_failures.append(f"embodiment:{embodiment}")
        if not embodiment_reports[embodiment]["etsf_six_head_gate_passed"]:
            target_head_failures.append(f"embodiment:{embodiment}")
    for key in target_slices:
        if not slice_reports[key]["primary_task_gate_passed"]:
            target_task_failures.append(f"embodiment_policy:{key}")
        if not slice_reports[key]["etsf_six_head_gate_passed"]:
            target_head_failures.append(f"embodiment_policy:{key}")
    target_task_passed = bool(target_records) and not target_task_failures
    target_heads_passed = bool(target_records) and not target_head_failures
    lobo_authority_passed = bool(
        lobo_authority is not None
        and lobo_authority_externally_bound
        and lobo_authority["primary_task_success_promotion_passed"]
        and lobo_authority["six_head_accuracy_promotion_passed"]
        and lobo_authority["every_predeclared_embodiment_policy_slice_passed"]
        and not lobo_authority["pooled_result_used_for_promotion"]
    )
    cross_embodiment_passed = bool(
        target_task_passed and target_heads_passed and lobo_authority_passed
    )
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": RESULT_STATUS,
        "input_bundle_sha256": bundle_sha,
        "protocol_core_v4_sha256": value["protocol_core_v4_sha256"],
        "audit_contract_v1_implementation_file_sha256": value[
            "audit_contract_v1_implementation_file_sha256"
        ],
        "pair_count": PAIR_COUNT,
        "condition_count": CONDITION_COUNT,
        "condition_order": list(CONDITIONS),
        "comparison_registry": [
            {
                "name": name, "model": model, "comparator": comparator,
                "family": family,
            }
            for name, model, comparator, family in COMPARISONS
        ],
        "pooled_diagnostic": pooled,
        "pooled_result_can_authorize_promotion": False,
        "by_embodiment_role": role_reports,
        "by_embodiment": embodiment_reports,
        "by_embodiment_policy": slice_reports,
        "target_task_success_promotion": {
            "status": "passed" if target_task_passed else "failed",
            "passed": target_task_passed,
            "requires_only_preregistered_target_piper_strata": True,
            "requires_lobo_records": False,
            "failed_target_strata": target_task_failures,
        },
        "target_six_head_accuracy_promotion": {
            "status": "passed" if target_heads_passed else "failed",
            "passed": target_heads_passed,
            "requires_only_preregistered_target_piper_strata": True,
            "failed_target_strata": target_head_failures,
        },
        "lobo_evidence_authority": (
            None if lobo_authority is None else {
                "authority_sha256": lobo_authority["authority_sha256"],
                "source_result_sha256": lobo_authority["source_result_sha256"],
                "source_result_file_sha256": lobo_authority[
                    "source_result_file_sha256"
                ],
                "pair_count": lobo_authority["pair_count"],
                "externally_bound_by_expected_authority_sha256": (
                    lobo_authority_externally_bound
                ),
                "all_required_gates_passed": lobo_authority_passed,
            }
        ),
        "cross_embodiment_promotion": {
            "status": "passed" if cross_embodiment_passed else "failed",
            "passed": cross_embodiment_passed,
            "requires_target_piper_task_and_six_head_gates": True,
            "requires_separate_content_addressed_lobo_evidence_authority": True,
            "lobo_evidence_authority_present": lobo_authority is not None,
            "lobo_evidence_authority_externally_bound": (
                lobo_authority_externally_bound
            ),
            "lobo_evidence_authority_passed": lobo_authority_passed,
            "pooled_cannot_mask_slice_failure": True,
            "target_task_success_promotion_passed": target_task_passed,
            "target_six_head_accuracy_promotion_passed": target_heads_passed,
        },
        "task_success_bootstrap": {
            "unit": "pair_id", "samples": bootstrap_samples,
            "base_seed": bootstrap_seed, "interval": "percentile_95_percent",
            "quantile_method": "linear",
        },
        "secondary_multiplicity": {
            "method": "Holm", "family_size": 4, "alpha": HOLM_ALPHA,
            "applied_separately_within_each_reported_scope": True,
        },
        "six_head_metric_weighting": "equal_pair_id_after_within_pair_mean",
        "applicability_observation_and_censoring_masks_enforced": True,
        "evaluation400_subset_excluded": False,
        "raw_hdf_trajectory_or_label_files_read": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _write_json_create_once(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--expected-lobo-evidence-authority-sha256")
    args = parser.parse_args()
    if not is_sha256(args.expected_input_file_sha256):
        raise Evaluation400V4Error("input file binding changed")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(args.input, flags)
    except OSError as error:
        raise Evaluation400V4Error("input file binding changed") from error
    try:
        metadata = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or hashlib.sha256(payload).hexdigest()
            != args.expected_input_file_sha256
        ):
            raise Evaluation400V4Error("input file binding changed")
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Evaluation400V4Error("input JSON changed") from error
    finally:
        os.close(descriptor)
    result = evaluate_bundle(
        value,
        bootstrap_samples=args.bootstrap_samples,
        expected_lobo_evidence_authority_sha256=(
            args.expected_lobo_evidence_authority_sha256
        ),
    )
    _write_json_create_once(args.output, result)


if __name__ == "__main__":
    main()
