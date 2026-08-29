#!/usr/bin/env python3
"""Measure action-selection headroom in generated five-body branch data.

This audit consumes only the public-simulator branch artifacts produced by
``collect_robotwin2_five_body_ee_candidate_branches_v1.py``.  It reports
candidate-oracle and dense terminal-consequence variation; it never presents
these offline quantities as recursive closed-loop success improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAT = "etsf_robotwin2_five_body_branch_effect_headroom_audit_v1"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
CANDIDATE_COUNT = 4
GOAL_SPREAD_EPSILON_METERS = 1e-4
ACTION_RMS_EPSILON = 1e-6


class HeadroomAuditError(RuntimeError):
    """A generated manifest, group, or diagnostic artifact is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise HeadroomAuditError(f"{label} is non-finite")
    return result


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return float(np.mean(rows)) if rows else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    rows = np.asarray(list(values), dtype=np.float64)
    if not len(rows):
        return {"mean": None, "median": None, "p90": None, "maximum": None}
    return {
        "mean": float(rows.mean()),
        "median": float(np.quantile(rows, 0.5)),
        "p90": float(np.quantile(rows, 0.9)),
        "maximum": float(rows.max()),
    }


def _vector(group: Mapping[str, np.ndarray], name: str, dtype: Any) -> np.ndarray:
    value = np.asarray(group[name], dtype=dtype)
    if value.shape != (CANDIDATE_COUNT,) or not np.isfinite(value).all():
        raise HeadroomAuditError(f"{name} must be one finite four-candidate vector")
    return value


def read_decision(
    root: Path, body: str, item: Mapping[str, Any]
) -> dict[str, Any]:
    group_id = str(item.get("group_id", ""))
    payload = root / body / str(item.get("path", ""))
    diagnostics = root / body / str(item.get("diagnostics_path", ""))
    for path, key in ((payload, "sha256"), (diagnostics, "diagnostics_sha256")):
        try:
            path.resolve().relative_to((root / body).resolve())
        except ValueError as error:
            raise HeadroomAuditError(f"{body}/{group_id} path escapes body root") from error
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != item.get(key)
        ):
            raise HeadroomAuditError(f"{body}/{group_id} payload is missing or changed")

    with np.load(payload, allow_pickle=False) as group:
        success = _vector(group, "success", np.float64)
        terminal_event = _vector(group, "terminal_max_event_id", np.int64)
        terminal_stage = _vector(group, "terminal_stage_progress", np.float64)
        terminal_goal = _vector(group, "terminal_goal_progress", np.float64)
        candidate_index = _vector(group, "candidate_index", np.int64)
        remaining_budget = _vector(group, "remaining_action_budget", np.float64)
    if not np.array_equal(candidate_index, np.arange(CANDIDATE_COUNT)):
        raise HeadroomAuditError(f"{body}/{group_id} candidate order changed")
    if not np.allclose(remaining_budget, remaining_budget[:1], atol=0.0, rtol=0.0):
        raise HeadroomAuditError(f"{body}/{group_id} candidates disagree on budget")
    if not np.isin(success, (0.0, 1.0)).all():
        raise HeadroomAuditError(f"{body}/{group_id} success is not binary")
    if np.any((terminal_event < 0) | (terminal_event > 4)):
        raise HeadroomAuditError(f"{body}/{group_id} terminal event is outside 0..4")

    with np.load(diagnostics, allow_pickle=False) as diagnostic:
        branch_error = np.asarray(diagnostic["branch_error"], dtype=bool)
        pairwise = np.asarray(
            diagnostic["candidate_action_pairwise_rms"], dtype=np.float64
        )
    if branch_error.shape != (CANDIDATE_COUNT,) or bool(branch_error.any()):
        raise HeadroomAuditError(f"{body}/{group_id} contains an invalid branch")
    if (
        pairwise.shape != (CANDIDATE_COUNT, CANDIDATE_COUNT)
        or not np.isfinite(pairwise).all()
        or not np.allclose(pairwise, pairwise.T, atol=1e-7, rtol=0.0)
    ):
        raise HeadroomAuditError(f"{body}/{group_id} action RMS matrix is invalid")
    off_diagonal = pairwise[np.triu_indices(CANDIDATE_COUNT, 1)]

    success_spread = bool(success.max() > success.min())
    event_spread = int(terminal_event.max() - terminal_event.min())
    goal_spread = float(terminal_goal.max() - terminal_goal.min())
    all_failure = not bool(success.max())
    dense_orderable = bool(
        all_failure
        and (event_spread > 0 or goal_spread > GOAL_SPREAD_EPSILON_METERS)
    )
    return {
        "body": body,
        "condition": str(item.get("condition")),
        "query": int(item.get("root_query_index", -1)),
        "budget": finite_float(remaining_budget[0], "remaining budget"),
        "baseline_success": float(success[0]),
        "oracle_success": float(success.max()),
        "mixed_success": success_spread,
        "all_failure": all_failure,
        "terminal_event_spread": event_spread,
        "terminal_stage_oracle_gain": float(terminal_stage.max() - terminal_stage[0]),
        "terminal_goal_spread_meters": goal_spread,
        "terminal_goal_oracle_gain_meters": float(
            terminal_goal.max() - terminal_goal[0]
        ),
        "dense_orderable": dense_orderable,
        "outcome_diverse": bool(success_spread or event_spread or goal_spread > GOAL_SPREAD_EPSILON_METERS),
        "action_pairwise_rms_min": float(off_diagonal.min()),
        "action_pairwise_rms_mean": float(off_diagonal.mean()),
        "action_pairwise_rms_max": float(off_diagonal.max()),
        "action_degenerate": bool(off_diagonal.max() <= ACTION_RMS_EPSILON),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = len(rows)
    if not decisions:
        return {"decisions": 0}
    baseline_success = mean(row["baseline_success"] for row in rows)
    oracle_success = mean(row["oracle_success"] for row in rows)
    assert baseline_success is not None and oracle_success is not None
    return {
        "decisions": decisions,
        "candidate_branches": decisions * CANDIDATE_COUNT,
        "baseline_candidate0_success_rate": baseline_success,
        "one_deviation_candidate_oracle_success_rate": oracle_success,
        "one_deviation_candidate_oracle_success_gain": (
            oracle_success - baseline_success
        ),
        "mixed_success_decision_rate": mean(float(row["mixed_success"]) for row in rows),
        "all_failure_decision_rate": mean(float(row["all_failure"]) for row in rows),
        "terminal_event_spread_decision_rate": mean(
            float(row["terminal_event_spread"] > 0) for row in rows
        ),
        "terminal_stage_oracle_gain": quantiles(
            row["terminal_stage_oracle_gain"] for row in rows
        ),
        "terminal_goal_spread_meters": quantiles(
            row["terminal_goal_spread_meters"] for row in rows
        ),
        "terminal_goal_oracle_gain_meters": quantiles(
            row["terminal_goal_oracle_gain_meters"] for row in rows
        ),
        "all_failure_dense_orderable_decision_rate": mean(
            float(row["dense_orderable"]) for row in rows
        ),
        "any_outcome_diversity_decision_rate": mean(
            float(row["outcome_diverse"]) for row in rows
        ),
        "candidate_action_degenerate_decision_rate": mean(
            float(row["action_degenerate"]) for row in rows
        ),
        "candidate_action_pairwise_rms": {
            "minimum_mean": mean(row["action_pairwise_rms_min"] for row in rows),
            "pair_mean": mean(row["action_pairwise_rms_mean"] for row in rows),
            "maximum_mean": mean(row["action_pairwise_rms_max"] for row in rows),
        },
        "remaining_budget": {
            "minimum": min(row["budget"] for row in rows),
            "maximum": max(row["budget"] for row in rows),
        },
    }


def audit(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    manifest_rows: dict[str, Any] = {}
    for body in BODIES:
        path = root / body / "manifest.json"
        if not path.exists():
            manifest_rows[body] = {"present": False, "decisions": 0}
            continue
        before_sha = sha256_file(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        unsigned = dict(manifest)
        logical_sha = unsigned.pop("logical_sha256", None)
        if logical_sha != canonical_sha256(unsigned):
            raise HeadroomAuditError(f"{body} manifest logical SHA changed")
        groups = manifest.get("groups")
        if not isinstance(groups, list):
            raise HeadroomAuditError(f"{body} manifest groups are invalid")
        body_rows = [read_decision(root, body, item) for item in groups]
        if sha256_file(path) != before_sha:
            raise HeadroomAuditError(f"{body} manifest changed during audit")
        rows.extend(body_rows)
        manifest_rows[body] = {
            "present": True,
            "decisions": len(body_rows),
            "status": manifest.get("status"),
            "manifest_sha256": before_sha,
        }

    by_body = {
        body: summarize([row for row in rows if row["body"] == body])
        for body in BODIES
    }
    by_body_condition = {
        f"{body}|{condition}": summarize(
            [
                row
                for row in rows
                if row["body"] == body and row["condition"] == condition
            ]
        )
        for body in BODIES
        for condition in CONDITIONS
    }
    query_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        query_counts[f"{row['body']}|{row['condition']}|query={row['query']}"] += 1
    return {
        "format": FORMAT,
        "status": "descriptive_offline_headroom_not_closed_loop_effect",
        "root": str(root),
        "manifest_inventory": manifest_rows,
        "overall": summarize(rows),
        "by_body": by_body,
        "by_body_condition": by_body_condition,
        "observed_condition_query_counts": dict(sorted(query_counts.items())),
        "estimand_boundary": {
            "candidate_oracle": "one_candidate_deviation_then_frozen_actor_continuation",
            "closed_loop_delta_success_rate_measured": False,
            "heldout_cross_embodiment_transfer_measured": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.branches_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = audit(root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
