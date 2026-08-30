from __future__ import annotations

import copy
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "nested_n4_n8_runner",
    SCRIPTS / "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _current() -> np.ndarray:
    value = np.zeros(16, dtype=np.float32)
    value[6] = 1.0
    value[14] = 1.0
    return value


def _raw16() -> np.ndarray:
    rows = []
    for index in range(runner.RAW_PROPOSAL_COUNT):
        chunk = np.repeat(_current()[None], 8, axis=0)
        chunk[:, 0] = 0.01 * index
        chunk[:, 1] = 0.002 * (index**2)
        chunk[:, 8] = -0.007 * index
        chunk[:, 9] = 0.001 * (index**2)
        chunk[:, 7] = (index % 3) / 2.0
        chunk[:, 15] = (index % 5) / 4.0
        rows.append(chunk)
    return np.stack(rows).astype(np.float32)


def _nested() -> tuple[dict[int, np.ndarray], dict[str, object]]:
    return runner.nested_pool_selection_audit(
        current_ee=_current(), raw_proposals=_raw16()
    )


def test_existing_independent_runs_do_not_authorize_pool_size_gain() -> None:
    audit = runner.existing_separate_n4_n8_comparability_audit()
    assert audit["same_body_condition_requested_seed_schedule"] is True
    assert audit["same_frozen_actor_candidate_zero_noise_identity"] is True
    assert audit["formal_n4_is_required_subset_of_current_n8"] is False
    assert audit["shared_cross_study_initial_raw_pool_commitment"] is False
    assert audit["direct_strong_causal_pool_size_comparison_authorized"] is False


def test_raw16_blind_fps_produces_exact_nested_prefix_and_candidate_zero() -> None:
    pools, audit = _nested()
    n4 = pools[runner.N4_CANDIDATE_COUNT]
    n8 = pools[runner.N8_CANDIDATE_COUNT]
    assert np.array_equal(n4, n8[: runner.N4_CANDIDATE_COUNT])
    assert np.array_equal(n4[0], _raw16()[0])
    assert audit["ordered_fps_raw_indices_n4"] == audit[
        "ordered_fps_raw_indices_n8"
    ][: runner.N4_CANDIDATE_COUNT]
    assert audit["ordered_fps_raw_indices_n8"][0] == 0
    assert audit[
        "selection_reads_outcomes_events_labels_or_critic_scores"
    ] is False
    runner.validate_nested_pool_audit(audit)

    tampered = copy.deepcopy(audit)
    tampered["ordered_fps_raw_indices_n4"] = list(reversed(
        tampered["ordered_fps_raw_indices_n4"]
    ))
    tampered["audit_sha256"] = runner.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "audit_sha256"}
    )
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_nested_pool_audit(tampered)


def test_schedule_uses_same_complete_seed_roster_and_rotates_three_arms() -> None:
    schedule = runner.evaluation_schedule()
    assert len(schedule) == len(runner.BODIES) * len(runner.CONDITIONS) * runner.SEED_COUNT
    for body in runner.BODIES:
        for condition in runner.CONDITIONS:
            rows = [
                row
                for row in schedule
                if row["heldout_body"] == body and row["condition"] == condition
            ]
            assert [row["requested_seed"] for row in rows] == [
                runner.SEED_BASE + index for index in range(runner.SEED_COUNT)
            ]
            assert all(set(row["method_order"]) == set(runner.METHODS) for row in rows)
            first_counts = Counter(row["method_order"][0] for row in rows)
            assert max(first_counts.values()) - min(first_counts.values()) <= 1


def _decision(method: str) -> dict[str, object]:
    pools, audit = _nested()
    if method == runner.METHOD_ACTOR:
        candidates = _raw16()[:1]
        indices = [0]
        scores = None
        selected = 0
    else:
        count = (
            runner.N4_CANDIDATE_COUNT
            if method == runner.METHOD_N4
            else runner.N8_CANDIDATE_COUNT
        )
        candidates = pools[count]
        indices = list(
            audit[
                "ordered_fps_raw_indices_n4"
                if method == runner.METHOD_N4
                else "ordered_fps_raw_indices_n8"
            ]
        )
        members = np.zeros((5, count), dtype=np.float64)
        members[:, count - 1] = 2.0
        aggregate = (
            runner.shared_head.aggregate_risk_adjusted_rank_scores(
                torch.as_tensor(members)
            )
            if method == runner.METHOD_N4
            else runner.pool_runner.aggregate_risk_adjusted_rank_scores(
                torch.as_tensor(members)
            )
        ).numpy()
        selected = int(np.argmax(aggregate))
        scores = {
            "candidate_rank_score_members": members.tolist(),
            "candidate_rank_score_epistemic_lcb_ensemble": aggregate.tolist(),
            "selected_candidate_index": selected,
        }
    return {
        "query_index": 0,
        "raw_proposal_count": runner.RAW_PROPOSAL_COUNT,
        "raw_ordered_proposals_sha256": audit["raw_ordered_proposals_sha256"],
        "raw_proposal_zero_sha256": audit["raw_proposal_zero_sha256"],
        "nested_pool_audit": audit,
        "selection_pool_candidate_count": len(candidates),
        "selection_pool_raw_indices": indices,
        "selection_pool_sha256": runner.array_sha256(candidates),
        "selected_candidate_index": selected,
        "selected_raw_proposal_index": indices[selected],
        "critic_scores": scores,
        "event_age_seconds": None if scores is None else 0.0,
    }


def _rollout(method: str, commitment_sha: str) -> dict[str, object]:
    return {
        "method": method,
        "heldout_body": runner.BODIES[0],
        "condition": runner.CONDITIONS[0],
        "requested_seed": runner.SEED_BASE,
        "initial_reset_snapshot": {"format": "reset", "seed": runner.SEED_BASE},
        "initial_canonical_query_snapshot": {"format": "canonical", "step": 1},
        "initial_candidate_commitment_sha256": commitment_sha,
        "binary_success": 1 if method == runner.METHOD_N8 else 0,
        "stage_progress": 1.0 if method == runner.METHOD_N8 else 0.5,
        "max_event_id": 4 if method == runner.METHOD_N8 else 2,
        "action_execution_error": None,
        "policy_query_count": 1,
        "decisions": [_decision(method)],
    }


def test_triplet_materialization_requires_one_reset_and_initial_raw16() -> None:
    _pools, audit = _nested()
    commitment = {
        "commitment_sha256": "c" * 64,
        "reset_snapshot": {"format": "reset", "seed": runner.SEED_BASE},
        "canonical_query_snapshot": {"format": "canonical", "step": 1},
        "raw_ordered_proposals_sha256": audit["raw_ordered_proposals_sha256"],
        "nested_pool_audit": audit,
    }
    expected = runner.evaluation_schedule()[0]
    rollouts = {
        method: _rollout(method, commitment["commitment_sha256"])
        for method in runner.METHODS
    }
    pair = runner.materialize_triplet(
        expected,
        rollouts,
        commitment=commitment,
        attempt_sha256="a" * 64,
        execution_contract_logical_sha256="b" * 64,
    )
    assert pair["same_resolved_reset_actor_n4_n8"] is True
    assert pair["same_initial_raw16_and_nested_pool_audit"] is True
    assert pair["n4_is_exact_ordered_prefix_of_n8"] is True

    broken = copy.deepcopy(rollouts)
    broken[runner.METHOD_N4]["decisions"][0][
        "raw_ordered_proposals_sha256"
    ] = "0" * 64
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.materialize_triplet(
            expected,
            broken,
            commitment=commitment,
            attempt_sha256="a" * 64,
            execution_contract_logical_sha256="b" * 64,
        )


def test_rollout_validator_replays_n4_and_n8_selection_without_new_gate() -> None:
    expected = runner.evaluation_schedule()[0]
    for method in runner.METHODS:
        rollout = _rollout(method, "c" * 64)
        runner.validate_rollout(rollout, method=method, expected=expected)
    assert runner.nested_pool_contract()[
        "additional_authorization_or_confidence_gate"
    ] is False
