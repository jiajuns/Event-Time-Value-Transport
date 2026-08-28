#!/usr/bin/env python3
"""Prospective v7 development-confirmation contracts and fixed statistics.

This module is label-free except for ``evaluate_fixed_policy``.  Seed
resolution, model/formula registration and every implementation SHA are frozen
before collection.  Fresh-confirmation inputs can never be evaluated here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from openvla_etsf_structured_event_time_utility import (
    GUARD_MARGIN,
    UTILITY_FORMULA,
    guarded_candidate_selection_numpy,
    structured_event_time_utility_numpy,
)


SEED_CANDIDATE_FORMAT = "etsf_robotwin_v7_development_candidates_v1"
SEED_MANIFEST_FORMAT = "etsf_robotwin_v7_development_seed_manifest_v1"
PREREGISTRATION_FORMAT = "etsf_openvla_v7_prospective_development_confirmation_v1"
RESULT_FORMAT = "etsf_openvla_v7_prospective_development_result_v1"
EXPECTED_GROUPS = 250
TASK = "move_can_pot"
DEPLOYMENT_CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)
EVENT_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
GAIN_MARGIN = GUARD_MARGIN
BOOTSTRAP_SEED = 20260907
BOOTSTRAP_SAMPLES = 20_000
HARMFUL_RATE_MAX = 0.10
MINIMUM_CHANGES = 10
FORMULA = UTILITY_FORMULA


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Content-address a frozen model directory, including relative filenames."""

    path = path.expanduser().resolve()
    files = sorted(row for row in path.rglob("*") if row.is_file())
    if not files:
        raise RuntimeError(f"v7 model directory has no files: {path}")
    digest = hashlib.sha256()
    for row in files:
        digest.update(str(row.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(row)))
    return digest.hexdigest()


def _signed(value: Mapping[str, Any], signature: str) -> dict[str, Any]:
    result = dict(value)
    result[signature] = canonical_sha256(result)
    return result


def _verify_signed(value: Mapping[str, Any], signature: str) -> None:
    unsigned = dict(value)
    recorded = str(unsigned.pop(signature, ""))
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"v7 {signature} mismatch")


def expand_candidates(value: Mapping[str, Any]) -> list[int]:
    if value.get("format") != SEED_CANDIDATE_FORMAT or value.get("status") != (
        "preregistered_unresolved_label_free"
    ):
        raise RuntimeError("v7 candidate manifest is not frozen")
    _verify_signed(value, "candidate_payload_sha256")
    seed_range = value.get("candidate_seed_range")
    if not isinstance(seed_range, Mapping):
        raise RuntimeError("v7 candidate seed range missing")
    start, count, step = (int(seed_range.get(k, -1)) for k in ("start", "count", "step"))
    if start < 0 or count < EXPECTED_GROUPS or step <= 0:
        raise RuntimeError("v7 candidate seed range invalid")
    seeds = [start + step * index for index in range(count)]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("v7 candidate seeds duplicated")
    return seeds


def identity_sets(value: Mapping[str, Any], *, expected: int, name: str) -> tuple[list[int], list[int]]:
    requested = [int(x) for x in value.get("requested_seeds", [])]
    resolved = [int(x) for x in value.get("resolved_seeds", [])]
    if len(requested) != expected or len(resolved) != expected:
        raise RuntimeError(f"{name} must contain {expected} requested/resolved seeds")
    if len(set(requested)) != expected or len(set(resolved)) != expected:
        raise RuntimeError(f"{name} requested/resolved identities are duplicated")
    return requested, resolved


def select_reset_unique_scenes(
    candidates: Sequence[int], *, resolver: Any, excluded: set[int], count: int = EXPECTED_GROUPS
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    selected, audit = [], []
    resolved_seen = set(excluded)
    for requested in map(int, candidates):
        resolved = int(resolver(requested))
        if requested in excluded:
            decision = "requested_identity_in_prior_exclusion"
        elif resolved in excluded:
            decision = "resolved_identity_in_prior_exclusion"
        elif resolved in resolved_seen:
            decision = "duplicate_resolved_v7_scene"
        else:
            decision = "selected"
            resolved_seen.add(resolved)
            selected.append({"seed": requested, "requested_seed": requested, "resolved_seed": resolved})
        audit.append({"requested_seed": requested, "resolved_seed": resolved, "decision": decision})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} v7 reset-unique scenes; need {count}")
    return selected, audit


def make_seed_manifest(
    *, selected: Sequence[Mapping[str, int]], audit: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Any], task: str = TASK,
) -> dict[str, Any]:
    rows = [dict(row) for row in selected]
    value = {
        "format": SEED_MANIFEST_FORMAT, "schema_version": 1,
        "status": "preregistered_resolved_label_free", "task": task,
        "purpose": "independent_prospective_development_confirmation_never_fresh",
        "seed_registry": "explicit_v7_prospective_development",
        "train": rows,
        "requested_seeds": [int(row["requested_seed"]) for row in rows],
        "resolved_seeds": [int(row["resolved_seed"]) for row in rows],
        "candidate_contract": dict(candidate), "exclusion_sources": dict(sources),
        "selection_rule": (
            "ascending_candidate_order_first_250_reset_unique_scenes_outside_"
            "official150_development150_fresh50_requested_union_resolved"
        ),
        "label_access_contract": (
            "reset_identity_only_no_policy_no_action_no_event_no_success_no_reward"
        ),
        "audit": [dict(row) for row in audit],
        "fresh_confirmation_eligible": False,
    }
    return _signed(value, "seed_manifest_payload_sha256")


def validate_seed_manifest(value: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    _verify_signed(value, "seed_manifest_payload_sha256")
    if (
        value.get("format") != SEED_MANIFEST_FORMAT
        or value.get("status") != "preregistered_resolved_label_free"
        or value.get("task") != TASK
        or value.get("seed_registry") != "explicit_v7_prospective_development"
        or value.get("fresh_confirmation_eligible") is not False
    ):
        raise RuntimeError("v7 seed manifest contract invalid")
    requested, resolved = identity_sets(value, expected=EXPECTED_GROUPS, name="v7")
    candidate_contract = value.get("candidate_contract")
    if not isinstance(candidate_contract, Mapping) or not isinstance(
        candidate_contract.get("payload"), Mapping
    ):
        raise RuntimeError("v7 signed candidate contract missing")
    candidate_payload = candidate_contract["payload"]
    candidates = expand_candidates(candidate_payload)
    positions = [candidates.index(seed) for seed in requested]
    if positions != sorted(positions):
        raise RuntimeError("v7 selected seeds changed frozen candidate order")
    if verify_files:
        candidate_path = Path(str(candidate_contract.get("source", ""))).expanduser()
        if not candidate_path.is_file() or sha256(candidate_path) != candidate_contract.get(
            "source_sha256"
        ):
            raise RuntimeError("v7 candidate source artifact changed")
        candidate_source = json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate_source != candidate_payload:
            raise RuntimeError("v7 candidate payload/source mismatch")
        runtime_registry = Path(
            str(candidate_contract.get("official_runtime_registry", ""))
        ).expanduser()
        if not runtime_registry.is_file() or sha256(runtime_registry) != candidate_contract.get(
            "official_runtime_registry_sha256"
        ):
            raise RuntimeError("v7 official runtime registry changed")
    rows = value.get("train")
    if not isinstance(rows, list) or [int(r.get("requested_seed", -1)) for r in rows] != requested or [
        int(r.get("resolved_seed", -1)) for r in rows
    ] != resolved:
        raise RuntimeError("v7 seed rows/mirrors changed")
    sources = value.get("exclusion_sources")
    if not isinstance(sources, Mapping) or set(sources) != {"official150", "development150", "fresh50"}:
        raise RuntimeError("v7 exclusion sources incomplete")
    excluded: set[int] = set()
    for name, expected in (("official150", 150), ("development150", 150), ("fresh50", 50)):
        source = sources[name]
        if not isinstance(source, Mapping):
            raise RuntimeError("v7 exclusion source malformed")
        path = Path(str(source.get("path", ""))).expanduser()
        if verify_files:
            if not path.is_file() or sha256(path) != source.get("sha256"):
                raise RuntimeError(f"v7 {name} exclusion artifact changed")
            source_value = json.loads(path.read_text(encoding="utf-8"))
            if name == "official150" and not source_value.get("requested_seeds"):
                task_row = source_value.get(TASK)
                values = task_row.get("success_seeds") if isinstance(task_row, Mapping) else None
                if not isinstance(values, list) or len(values) != expected:
                    raise RuntimeError("official150 registry lacks 150 success seeds")
                prior_requested = prior_resolved = [int(x) for x in values]
            else:
                prior_requested, prior_resolved = identity_sets(source_value, expected=expected, name=name)
            if canonical_sha256({"requested": prior_requested, "resolved": prior_resolved}) != source.get(
                "identity_sets_sha256"
            ):
                raise RuntimeError(f"v7 {name} exclusion identities changed")
        else:
            prior_requested = [int(x) for x in source.get("requested_seeds", [])]
            prior_resolved = [int(x) for x in source.get("resolved_seeds", [])]
            if len(prior_requested) != expected or len(prior_resolved) != expected:
                raise RuntimeError(f"v7 {name} embedded exclusion identities incomplete")
        excluded.update(prior_requested); excluded.update(prior_resolved)
    overlap = (set(requested) | set(resolved)) & excluded
    if overlap:
        raise RuntimeError(f"v7 seeds overlap prior requested/resolved: {sorted(overlap)}")
    audit = value.get("audit")
    if not isinstance(audit, list) or any(
        isinstance(row, Mapping) and set(row) & {"success", "reward", "event", "action", "prediction"}
        for row in audit
    ):
        raise RuntimeError("v7 reset audit contains labels/policy data")
    selected = [(int(r["requested_seed"]), int(r["resolved_seed"])) for r in audit if r.get("decision") == "selected"]
    if selected != list(zip(requested, resolved)):
        raise RuntimeError("v7 reset audit differs from selected identities")
    return {"requested_seeds": requested, "resolved_seeds": resolved,
            "seed_manifest_payload_sha256": value["seed_manifest_payload_sha256"]}


def validate_preregistered_source_files(value: Mapping[str, Any]) -> None:
    """Recheck every registered SHA immediately before label collection."""

    validate_preregistration(value)
    source = value.get("source_contract")
    if not isinstance(source, Mapping):
        raise RuntimeError("v7 preregistration source contract missing")
    for row in source.get("implementation_files", []):
        if not isinstance(row, Mapping):
            raise RuntimeError("v7 implementation provenance malformed")
        path = Path(str(row.get("path", ""))).expanduser()
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"v7 implementation changed before collection: {path}")
    for path_key, sha_key in (
        ("seed_manifest", "seed_manifest_file_sha256"),
        ("pretrained", "pretrained_sha256"),
        ("event_spec", "event_spec_sha256"),
    ):
        path = Path(str(source.get(path_key, ""))).expanduser()
        if not path.is_file() or sha256(path) != source.get(sha_key):
            raise RuntimeError(f"v7 frozen source changed before collection: {path}")
    actor = source.get("actor_model")
    if not isinstance(actor, Mapping):
        raise RuntimeError("v7 frozen actor-model contract missing")
    actor_path = Path(str(actor.get("path", ""))).expanduser()
    if not actor_path.is_dir() or directory_sha256(actor_path) != actor.get("tree_sha256"):
        raise RuntimeError("v7 frozen actor model changed before collection")


def make_preregistration(
    *, seed_manifest: Mapping[str, Any], source_contract: Mapping[str, Any],
    task_calibration_sha256: str,
) -> dict[str, Any]:
    validate_seed_manifest(seed_manifest, verify_files=False)
    value = {
        "format": PREREGISTRATION_FORMAT, "status": "preregistered_before_labels",
        "task": TASK, "expected_groups": EXPECTED_GROUPS,
        "seed_manifest_payload_sha256": seed_manifest["seed_manifest_payload_sha256"],
        "source_contract": dict(source_contract),
        "candidate_contract": {"names": list(DEPLOYMENT_CANDIDATE_NAMES), "count": 4,
                               "scope": "exact_first_four_only_no_fifth_candidate"},
        "model_contract": {"members": 1, "checkpoint": "frozen_factual_only",
                           "action_rank_head_used": False, "base_success_used": False},
        "event_value_registry": {"task": TASK, "values": list(EVENT_VALUES),
                                 "task_calibration_sha256": task_calibration_sha256,
                                 "implicit_ordinal_fallback": False},
        "score_contract": {"formula": FORMULA, "gain_margin": GAIN_MARGIN,
                           "population_std_ddof": 0, "zero_std_epsilon": 1e-8,
                           "tie_break": "numpy_argmax_lowest_candidate_index"},
        "statistics_contract": {
            "estimand": "unconditional_equal_group_success_delta_including_zeros",
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_ci": 0.95, "exact_two_sided_sign_test": True,
            "harmful_rate_denominator": "all_policy_changes_including_zero_delta",
            "harmful_rate_max": HARMFUL_RATE_MAX, "minimum_changes": MINIMUM_CHANGES,
            "single_gate": (
                "bootstrap_95_lcb_gt_0_and_exact_sign_p_lt_0.05_and_"
                "harmful_rate_le_0.10_and_changes_ge_10"
            ),
            "multiple_comparisons": 1,
        },
        "fresh_confirmation": {
            "inputs_accepted_by_v7": False,
            "labels_read_by_v7": False,
            "authorization_possible": "true_only_if_single_signed_gate_passes",
            "automatic_fresh_launch": False,
        },
    }
    return _signed(value, "preregistration_sha256")


def validate_preregistration(value: Mapping[str, Any]) -> None:
    _verify_signed(value, "preregistration_sha256")
    if value.get("format") != PREREGISTRATION_FORMAT or value.get("status") != "preregistered_before_labels":
        raise RuntimeError("v7 preregistration invalid")
    candidate, score, stats, event, fresh = (value.get(k) for k in (
        "candidate_contract", "score_contract", "statistics_contract", "event_value_registry", "fresh_confirmation"))
    if not all(isinstance(x, Mapping) for x in (candidate, score, stats, event, fresh)):
        raise RuntimeError("v7 preregistration incomplete")
    if candidate.get("names") != list(DEPLOYMENT_CANDIDATE_NAMES) or candidate.get("count") != 4:
        raise RuntimeError("v7 candidate contract changed")
    if score.get("formula") != FORMULA or score.get("gain_margin") != GAIN_MARGIN:
        raise RuntimeError("v7 score formula/margin changed")
    if event.get("values") != list(EVENT_VALUES) or event.get("implicit_ordinal_fallback") is not False:
        raise RuntimeError("v7 event value registry changed")
    if stats.get("multiple_comparisons") != 1 or stats.get("harmful_rate_max") != HARMFUL_RATE_MAX:
        raise RuntimeError("v7 statistical gate changed")
    if (
        fresh.get("inputs_accepted_by_v7") is not False
        or fresh.get("labels_read_by_v7") is not False
        or fresh.get("authorization_possible") != "true_only_if_single_signed_gate_passes"
        or fresh.get("automatic_fresh_launch") is not False
    ):
        raise RuntimeError("v7 fresh authorization semantics changed")


def fixed_world_utility(
    next_reached_event_logits: np.ndarray,
    next_event_logits: np.ndarray,
    duration_selected_log_mean: np.ndarray,
) -> np.ndarray:
    result = structured_event_time_utility_numpy(
        next_reached_event_logits, next_event_logits, duration_selected_log_mean,
        event_values=EVENT_VALUES,
    )
    utility = np.asarray(result["utility"], dtype=np.float64)
    if utility.shape != (4,):
        raise RuntimeError("v7 fixed utility requires one four-candidate group")
    return utility


def fixed_decision(utility: np.ndarray) -> dict[str, Any]:
    result = guarded_candidate_selection_numpy(utility)
    proposed = int(result["proposed_index"]); selected = int(result["selected_index"])
    return {"baseline_index": 0, "proposed_index": proposed, "selected_index": selected,
            "utility_gain": float(result["score_margin"]), "changed": bool(result["accepted"])}


def evaluate_fixed_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_GROUPS or len({str(r["logical_key"]) for r in rows}) != EXPECTED_GROUPS:
        raise RuntimeError("v7 result requires exactly 250 unique prospective groups")
    decisions, delta = [], []
    for row in rows:
        decision = fixed_decision(np.asarray(row["utility"]))
        labels = np.asarray(row["success"], dtype=np.float64)
        if labels.shape != (4,) or np.any((labels != 0) & (labels != 1)):
            raise RuntimeError("v7 success labels invalid")
        value = float(labels[decision["selected_index"]] - labels[0])
        decisions.append({"logical_key": str(row["logical_key"]), **decision, "success_delta": value})
        delta.append(value)
    delta = np.asarray(delta)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES)
    for start in range(0, BOOTSTRAP_SAMPLES, 1000):
        count = min(1000, BOOTSTRAP_SAMPLES - start)
        index = generator.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[index].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    helpful, harmful = int((delta > 0).sum()), int((delta < 0).sum())
    changed = int(sum(row["changed"] for row in decisions))
    nonzero, tail = helpful + harmful, min(helpful, harmful)
    sign_p = min(1.0, 2.0 * sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2.0**nonzero)) if nonzero else 1.0
    harmful_rate = harmful / max(changed, 1)
    passed = low > 0 and sign_p < 0.05 and harmful_rate <= HARMFUL_RATE_MAX and changed >= MINIMUM_CHANGES
    return {
        "groups": EXPECTED_GROUPS, "changed_groups": changed,
        "helpful_changes": helpful, "harmful_changes": harmful,
        "harmful_rate_over_all_changes": harmful_rate,
        "unconditional_mean_success_delta": float(delta.mean()),
        "unconditional_bootstrap_95_ci": [float(low), float(high)],
        "exact_two_sided_sign_test_p": float(sign_p), "development_gate_pass": bool(passed),
        "fresh50_confirmation_authorized": bool(passed),
        "v7_reads_or_launches_fresh": False,
        "decisions": decisions,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_sha256", "sha256", "directory_sha256", "expand_candidates", "identity_sets",
    "select_reset_unique_scenes", "make_seed_manifest", "validate_seed_manifest",
    "make_preregistration", "validate_preregistration",
    "validate_preregistered_source_files", "fixed_world_utility",
    "fixed_decision", "evaluate_fixed_policy",
]
