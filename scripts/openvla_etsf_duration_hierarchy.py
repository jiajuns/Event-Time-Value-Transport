#!/usr/bin/env python3
"""Leakage-resistant hierarchical duration baseline contract.

This module is deliberately independent from materialization, training and
prediction evaluators.  It consumes caller-supplied outer-training arrays and
fits an observed-only log1p-duration median hierarchy with a fixed lookup:

    current_event x body -> current_event -> body -> global

Only sources with at least twenty observed outer-training rows are eligible.
Every applied row carries its exact source and support provenance.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


DURATION_HIERARCHY_PROTOCOL_V2 = (
    "outer_training_observed_only_current_event_body_hierarchy_min20_v2"
)
DURATION_HIERARCHY_FORMAT = "etsf_duration_hierarchy_contract_v2"
MINIMUM_APPLIED_SOURCE_SUPPORT = 20
LOOKUP_ORDER = ("event_body", "event", "body", "global")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def logical_group_list_sha256(groups: Sequence[str]) -> str:
    normalized = sorted(set(map(str, groups)))
    if not normalized or any(not group for group in normalized):
        raise ValueError("outer-training logical groups must be non-empty strings")
    return canonical_sha256(normalized)


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a one-dimensional aligned array")
    return result


def _ids(value: Any, *, name: str, length: int) -> np.ndarray:
    raw = _vector(value, name=name, length=length)
    try:
        result = raw.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer ids") from error
    if not np.array_equal(raw, result) or np.any(result < 0):
        raise ValueError(f"{name} must contain non-negative integer ids")
    return result


def _boolean(value: Any, *, name: str, length: int) -> np.ndarray:
    raw = _vector(value, name=name, length=length)
    if raw.dtype == np.bool_:
        return raw.astype(bool)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be boolean") from error
    if not np.isfinite(numeric).all() or np.any((numeric != 0.0) & (numeric != 1.0)):
        raise ValueError(f"{name} must be boolean")
    return numeric.astype(bool)


def _source_row(
    target: np.ndarray, mask: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    support = int(mask.sum())
    if support <= 0:
        raise ValueError("duration source cannot be fitted without observed rows")
    logical_groups = sorted(set(map(str, groups[mask].tolist())))
    return {
        "median_log1p_duration": float(np.median(target[mask])),
        "support": support,
        "logical_group_support": len(logical_groups),
        "eligible": support >= MINIMUM_APPLIED_SOURCE_SUPPORT,
        "source_training_logical_groups": logical_groups,
        "source_training_logical_groups_sha256": canonical_sha256(logical_groups),
    }


def _event_body_key(event_id: int, body_id: int) -> str:
    return f"{int(event_id)}:{int(body_id)}"


def fit_duration_hierarchy(
    *,
    duration: Any,
    duration_observed: Any,
    current_event_id: Any,
    body_id: Any,
    logical_group: Any,
    split_role: Any,
    owner_fold_id: int,
) -> dict[str, Any]:
    """Fit one owner fold's hierarchy using outer-training labels only."""

    duration_array = _vector(duration, name="duration").astype(np.float64)
    length = len(duration_array)
    if length == 0 or not np.isfinite(duration_array).all() or np.any(
        duration_array < 0.0
    ):
        raise ValueError("duration must be a non-empty finite non-negative vector")
    observed = _boolean(
        duration_observed, name="duration_observed", length=length
    )
    events = _ids(
        current_event_id, name="current_event_id", length=length
    )
    bodies = _ids(body_id, name="body_id", length=length)
    groups_raw = _vector(logical_group, name="logical_group", length=length)
    groups = np.asarray([str(value) for value in groups_raw], dtype=object)
    if any(not value for value in groups):
        raise ValueError("logical_group values must be non-empty")
    roles = _vector(split_role, name="split_role", length=length)
    if any(str(value) != "outer_training" for value in roles):
        raise RuntimeError("duration hierarchy refuses non-outer-training rows")
    if not isinstance(owner_fold_id, int) or not 0 <= owner_fold_id < 5:
        raise ValueError("owner_fold_id must lie in [0,4]")
    if int(observed.sum()) < MINIMUM_APPLIED_SOURCE_SUPPORT:
        raise RuntimeError(
            "no duration hierarchy source reaches the fixed support of 20"
        )

    target = np.log1p(duration_array)
    exact: dict[str, Any] = {}
    event: dict[str, Any] = {}
    body: dict[str, Any] = {}
    observed_pairs = sorted(
        set(zip(events[observed].tolist(), bodies[observed].tolist()))
    )
    for event_value, body_value in observed_pairs:
        selected = observed & (events == event_value) & (bodies == body_value)
        exact[_event_body_key(event_value, body_value)] = _source_row(
            target, selected, groups
        )
    for event_value in sorted(set(events[observed].tolist())):
        selected = observed & (events == event_value)
        event[str(int(event_value))] = _source_row(target, selected, groups)
    for body_value in sorted(set(bodies[observed].tolist())):
        selected = observed & (bodies == body_value)
        body[str(int(body_value))] = _source_row(target, selected, groups)
    global_row = _source_row(target, observed, groups)
    if not global_row["eligible"]:
        raise RuntimeError("global duration source is not eligible")

    training_groups = sorted(set(map(str, groups.tolist())))
    contract: dict[str, Any] = {
        "format": DURATION_HIERARCHY_FORMAT,
        "protocol": DURATION_HIERARCHY_PROTOCOL_V2,
        "status": "fitted_outer_training_only",
        "owner_fold_id": owner_fold_id,
        "fit_scope": "outer_training_only",
        "fit_label_scope": "observed_duration_only",
        "current_event_field": "current_event_id",
        "clock_event_proxy_allowed": False,
        "target_transform": "log1p_duration",
        "lookup_order": list(LOOKUP_ORDER),
        "minimum_applied_source_support": MINIMUM_APPLIED_SOURCE_SUPPORT,
        "support_unit": "observed_outer_training_rows",
        "outer_training_rows": length,
        "outer_training_observed_rows": int(observed.sum()),
        "outer_training_logical_groups": training_groups,
        "outer_training_logical_groups_sha256": canonical_sha256(training_groups),
        "sources": {
            "event_body": exact,
            "event": event,
            "body": body,
            "global": global_row,
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    validate_duration_hierarchy_contract(contract)
    return contract


def validate_duration_hierarchy_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(contract)
    recorded = unsigned.pop("contract_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("duration hierarchy contract signature mismatch")
    if (
        contract.get("format") != DURATION_HIERARCHY_FORMAT
        or contract.get("protocol") != DURATION_HIERARCHY_PROTOCOL_V2
        or contract.get("status") != "fitted_outer_training_only"
        or contract.get("fit_scope") != "outer_training_only"
        or contract.get("fit_label_scope") != "observed_duration_only"
        or contract.get("current_event_field") != "current_event_id"
        or contract.get("clock_event_proxy_allowed") is not False
        or contract.get("lookup_order") != list(LOOKUP_ORDER)
        or contract.get("minimum_applied_source_support")
        != MINIMUM_APPLIED_SOURCE_SUPPORT
        or contract.get("support_unit") != "observed_outer_training_rows"
    ):
        raise RuntimeError("duration hierarchy frozen protocol changed")
    groups = contract.get("outer_training_logical_groups")
    if (
        not isinstance(groups, list)
        or sorted(set(map(str, groups))) != groups
        or canonical_sha256(groups)
        != contract.get("outer_training_logical_groups_sha256")
    ):
        raise RuntimeError("duration hierarchy training group SHA mismatch")
    sources = contract.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("duration hierarchy sources are missing")
    eligible_count = 0
    for kind in LOOKUP_ORDER:
        table = sources.get(kind)
        items = [table] if kind == "global" else (
            list(table.values()) if isinstance(table, Mapping) else []
        )
        if not items:
            raise RuntimeError(f"duration hierarchy source table is empty: {kind}")
        for item in items:
            if not isinstance(item, Mapping):
                raise RuntimeError("duration hierarchy source record is invalid")
            support = item.get("support")
            eligible = item.get("eligible")
            source_groups = item.get("source_training_logical_groups")
            if (
                not isinstance(support, int)
                or support <= 0
                or eligible is not (support >= MINIMUM_APPLIED_SOURCE_SUPPORT)
                or not isinstance(item.get("logical_group_support"), int)
                or item["logical_group_support"] <= 0
                or not isinstance(source_groups, list)
                or sorted(set(map(str, source_groups))) != source_groups
                or len(source_groups) != item["logical_group_support"]
                or canonical_sha256(source_groups)
                != item.get("source_training_logical_groups_sha256")
                or not isinstance(item.get("median_log1p_duration"), (int, float))
                or not np.isfinite(float(item["median_log1p_duration"]))
            ):
                raise RuntimeError("duration hierarchy source provenance is invalid")
            eligible_count += int(eligible)
    if eligible_count == 0 or sources["global"].get("eligible") is not True:
        raise RuntimeError("duration hierarchy has no globally valid fallback")
    return dict(contract)


def apply_duration_hierarchy(
    contract: Mapping[str, Any],
    *,
    current_event_id: Any,
    body_id: Any,
    expected_training_logical_groups_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply the fixed hierarchy and return per-row auditable provenance."""

    validate_duration_hierarchy_contract(contract)
    recorded_group_sha = str(contract["outer_training_logical_groups_sha256"])
    if (
        expected_training_logical_groups_sha256 is not None
        and expected_training_logical_groups_sha256 != recorded_group_sha
    ):
        raise RuntimeError("duration hierarchy owner-training group SHA mismatch")
    raw_events = _vector(current_event_id, name="current_event_id")
    length = len(raw_events)
    events = _ids(raw_events, name="current_event_id", length=length)
    bodies = _ids(body_id, name="body_id", length=length)
    sources = contract["sources"]
    baseline = np.empty(length, dtype=np.float64)
    source_kind: list[str] = []
    source_key: list[str] = []
    source_support = np.empty(length, dtype=np.int64)
    source_group_support = np.empty(length, dtype=np.int64)
    source_training_group_sha: list[str] = []
    for index, (event_value, body_value) in enumerate(zip(events, bodies)):
        choices = (
            (
                "event_body",
                _event_body_key(event_value, body_value),
                sources["event_body"].get(
                    _event_body_key(event_value, body_value)
                ),
            ),
            ("event", str(int(event_value)), sources["event"].get(str(int(event_value)))),
            ("body", str(int(body_value)), sources["body"].get(str(int(body_value)))),
            ("global", "global", sources["global"]),
        )
        selected = next(
            (
                (kind, key, item)
                for kind, key, item in choices
                if isinstance(item, Mapping) and item.get("eligible") is True
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("no eligible duration hierarchy source for row")
        kind, key, item = selected
        support = int(item["support"])
        if support < MINIMUM_APPLIED_SOURCE_SUPPORT:
            raise RuntimeError("duration hierarchy attempted to apply a sparse source")
        baseline[index] = float(item["median_log1p_duration"])
        source_kind.append(kind)
        source_key.append(key)
        source_support[index] = support
        source_group_support[index] = int(item["logical_group_support"])
        source_training_group_sha.append(
            str(item["source_training_logical_groups_sha256"])
        )
    return {
        "baseline_log1p_duration": baseline,
        "source_kind": np.asarray(source_kind, dtype=object),
        "source_key": np.asarray(source_key, dtype=object),
        "source_support": source_support,
        "source_logical_group_support": source_group_support,
        "source_training_logical_groups_sha256": np.asarray(
            source_training_group_sha, dtype=object
        ),
        "outer_training_logical_groups_sha256": recorded_group_sha,
        "minimum_applied_source_support": int(source_support.min()) if length else None,
        "target_fold_labels_used_for_fit": False,
    }


def serialize_duration_hierarchy(contract: Mapping[str, Any]) -> str:
    validate_duration_hierarchy_contract(contract)
    return canonical_json(contract)


def deserialize_duration_hierarchy(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise RuntimeError("duration hierarchy serialization must contain an object")
    return validate_duration_hierarchy_contract(parsed)


__all__ = [
    "DURATION_HIERARCHY_FORMAT",
    "DURATION_HIERARCHY_PROTOCOL_V2",
    "LOOKUP_ORDER",
    "MINIMUM_APPLIED_SOURCE_SUPPORT",
    "apply_duration_hierarchy",
    "canonical_sha256",
    "deserialize_duration_hierarchy",
    "fit_duration_hierarchy",
    "logical_group_list_sha256",
    "serialize_duration_hierarchy",
    "validate_duration_hierarchy_contract",
]
