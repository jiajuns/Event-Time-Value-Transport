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


def _complete_outcome_rows() -> list[dict[str, object]]:
    rows = []
    for expected in runner.evaluation_schedule():
        row = {
            **expected,
            "pair_sha256": runner.canonical_sha256(expected),
        }
        for method in runner.METHODS:
            row[f"{method}_binary_success"] = 0
            row[f"{method}_stage_progress"] = 0.25
        rows.append(row)
    return rows


def test_complete_outcome_roster_rejects_an_entire_missing_cell() -> None:
    rows = _complete_outcome_rows()
    runner.validate_complete_outcome_rows(rows)
    incomplete = [
        row
        for row in rows
        if not (
            row["heldout_body"] == runner.BODIES[-1]
            and row["condition"] == runner.CONDITIONS[-1]
        )
    ]
    with pytest.raises(
        runner.NestedCandidatePoolError, match="complete 1000-triplet schedule"
    ):
        runner.validate_complete_outcome_rows(incomplete)


def test_nested_protocol_declares_all_bootstrap_seed_offsets() -> None:
    protocol = runner.nested_evaluation_protocol()
    assert protocol["formal_seed_block_reused"] is False
    assert protocol["bootstrap_seed_base"] == runner.BOOTSTRAP_SEED
    assert set(protocol["bootstrap_seed_derivation"]) == {
        "overall_comparisons",
        "body_comparisons",
        "body_condition_comparisons",
    }


def test_overall_interval_clusters_all_ten_cells_by_requested_seed() -> None:
    rows = _complete_outcome_rows()
    summary = runner._comparison_summary(
        rows,
        runner.METHOD_N4,
        runner.METHOD_N8,
        seed=runner.BOOTSTRAP_SEED,
    )
    contract = summary["paired_success_delta_interval_contract"]
    assert contract["cluster_count"] == runner.SEED_COUNT
    assert contract["rows_per_cluster"] == len(runner.BODIES) * len(
        runner.CONDITIONS
    )
    assert summary["mcnemar_contract"]["role"] == "descriptive_only"


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
        method_result_bindings={
            method: {
                "logical_sha256": f"{index + 1:x}" * 64,
                "file_sha256": f"{index + 4:x}" * 64,
            }
            for index, method in enumerate(runner.METHODS)
        },
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
            method_result_bindings={
                method: {
                    "logical_sha256": f"{index + 1:x}" * 64,
                    "file_sha256": f"{index + 4:x}" * 64,
                }
                for index, method in enumerate(runner.METHODS)
            },
        )


def test_rollout_validator_replays_n4_and_n8_selection_without_new_gate() -> None:
    expected = runner.evaluation_schedule()[0]
    for method in runner.METHODS:
        rollout = _rollout(method, "c" * 64)
        runner.validate_rollout(rollout, method=method, expected=expected)
    assert runner.nested_pool_contract()[
        "additional_authorization_or_confidence_gate"
    ] is False


def test_method_result_is_create_once_and_replay_validated(tmp_path: Path) -> None:
    expected = runner.evaluation_schedule()[0]
    method = runner.METHOD_ACTOR
    start = runner.build_method_start(
        expected,
        method=method,
        method_ordinal=0,
        attempt_sha256="a" * 64,
        commitment_sha256="c" * 64,
        execution_contract_logical_sha256="b" * 64,
        completed_prefix_result_sha256=[],
    )
    result = runner.build_method_result(
        expected,
        method=method,
        method_ordinal=0,
        rollout=_rollout(method, "c" * 64),
        method_start_sha256=start["method_start_sha256"],
        attempt_sha256="a" * 64,
        commitment_sha256="c" * 64,
        execution_contract_logical_sha256="b" * 64,
        execution_contract_file_sha256="d" * 64,
        completed_prefix_result_sha256=[],
    )
    path = tmp_path / "result.json"
    first_file_sha = runner.promote_create_once_json(
        path, result, label="test method result"
    )
    assert first_file_sha == runner.sha256_file(path)
    loaded, staged_only = runner.read_create_once_json(
        path, label="test method result"
    )
    assert staged_only is False
    rollout = runner.validate_method_result(
        loaded,
        expected,
        method=method,
        method_ordinal=0,
        method_start_sha256=start["method_start_sha256"],
        attempt_sha256="a" * 64,
        commitment_sha256="c" * 64,
        execution_contract_logical_sha256="b" * 64,
        execution_contract_file_sha256="d" * 64,
        completed_prefix_result_sha256=[],
    )
    assert rollout["method"] == method
    changed = copy.deepcopy(result)
    changed["rollout"]["binary_success"] = 1
    with pytest.raises(runner.NestedCandidatePoolError, match="create-once value"):
        runner.promote_create_once_json(
            path, changed, label="test method result"
        )


def _persist_complete_triplet(
    tmp_path: Path, *, tamper_embedded_actor_outcome: bool = False
) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    expected = runner.evaluation_schedule()[0]
    identity = runner.pair_id(
        expected["heldout_body"], expected["condition"], expected["requested_seed"]
    )
    logical_contract_sha = "b" * 64
    file_contract_sha = "d" * 64
    attempt_base = {
        "format": "etsf_robotwin2_nested_n4_n8_attempt_v2",
        "status": "started_once_fixed_method_order_with_bounded_resume",
        "pair_id": identity,
        **expected,
        "execution_contract_logical_sha256": logical_contract_sha,
        "execution_contract_file_sha256": file_contract_sha,
        "attempt_number": 1,
    }
    attempt = {
        **attempt_base,
        "attempt_sha256": runner.canonical_sha256(attempt_base),
    }
    reset_snapshot = {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "kind": "reset",
    }
    canonical_snapshot = {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "kind": "canonical_query",
    }
    _pools, audit = _nested()
    commitment_base = {
        "format": runner.INITIAL_COMMITMENT_FORMAT,
        "heldout_body": expected["heldout_body"],
        "condition": expected["condition"],
        "requested_seed": expected["requested_seed"],
        "resolved_seed": expected["requested_seed"],
        "nested_pool_audit": audit,
        "raw_ordered_proposals_sha256": audit["raw_ordered_proposals_sha256"],
        "reset_snapshot": reset_snapshot,
        "reset_identity_sha256": runner.formal.reset_identity(reset_snapshot),
        "canonical_query_snapshot": canonical_snapshot,
        "canonical_query_identity_sha256": runner.formal.reset_identity(
            canonical_snapshot
        ),
        "candidate_generation_advanced_simulator": False,
        "frozen_before_any_method_execution": True,
    }
    commitment = {
        **commitment_base,
        "commitment_sha256": runner.canonical_sha256(commitment_base),
    }
    paths = {
        "pair_path": tmp_path / "pairs" / f"{identity}.json",
        "attempt_path": tmp_path / "attempts" / f"{identity}.json",
        "commitment_path": tmp_path / "initial_commitments" / f"{identity}.json",
        "method_starts_dir": tmp_path / "method_starts",
        "method_results_dir": tmp_path / "method_results",
        "method_failures_dir": tmp_path / "method_failures",
    }
    runner.promote_create_once_json(
        paths["attempt_path"], attempt, label="test attempt"
    )
    runner.promote_create_once_json(
        paths["commitment_path"], commitment, label="test commitment"
    )

    rollouts = {}
    bindings = {}
    prefix_result_shas = []
    for method_ordinal, method in enumerate(expected["method_order"]):
        stem = f"{identity}.{method_ordinal:02d}.{method}"
        start = runner.build_method_start(
            expected,
            method=method,
            method_ordinal=method_ordinal,
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            execution_contract_logical_sha256=logical_contract_sha,
            completed_prefix_result_sha256=prefix_result_shas,
        )
        start_path = paths["method_starts_dir"] / f"{stem}.json"
        runner.promote_create_once_json(start_path, start, label="test method start")
        rollout = _rollout(method, commitment["commitment_sha256"])
        rollout["initial_reset_snapshot"] = reset_snapshot
        rollout["initial_canonical_query_snapshot"] = canonical_snapshot
        result = runner.build_method_result(
            expected,
            method=method,
            method_ordinal=method_ordinal,
            rollout=rollout,
            method_start_sha256=start["method_start_sha256"],
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            execution_contract_logical_sha256=logical_contract_sha,
            execution_contract_file_sha256=file_contract_sha,
            completed_prefix_result_sha256=prefix_result_shas,
        )
        result_path = paths["method_results_dir"] / f"{stem}.json"
        result_file_sha = runner.promote_create_once_json(
            result_path, result, label="test method result"
        )
        rollouts[method] = rollout
        bindings[method] = {
            "logical_sha256": result["method_result_sha256"],
            "file_sha256": result_file_sha,
        }
        prefix_result_shas.append(result["method_result_sha256"])

    pair = runner.materialize_triplet(
        expected,
        rollouts,
        commitment=commitment,
        attempt_sha256=attempt["attempt_sha256"],
        execution_contract_logical_sha256=logical_contract_sha,
        method_result_bindings=bindings,
    )
    if tamper_embedded_actor_outcome:
        pair["rollouts"][runner.METHOD_ACTOR]["binary_success"] = 1
        pair["rollouts"][runner.METHOD_ACTOR]["stage_progress"] = 1.0
        pair["pair_sha256"] = runner.canonical_sha256(
            {key: value for key, value in pair.items() if key != "pair_sha256"}
        )
    runner.promote_create_once_json(paths["pair_path"], pair, label="test pair")
    context = {
        "identity": identity,
        "expected": expected,
        "attempt": attempt,
        "execution_contract_logical_sha256": logical_contract_sha,
        "execution_contract_file_sha256": file_contract_sha,
    }
    return context, pair, paths


def test_existing_pair_embedded_outcome_must_match_method_result(
    tmp_path: Path,
) -> None:
    context, _pair, paths = _persist_complete_triplet(
        tmp_path, tamper_embedded_actor_outcome=True
    )
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="differs from its method results",
    ):
        runner.recover_complete_existing_triplet(**paths, **context)


def test_existing_pair_with_missing_method_result_fails_without_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _pair, paths = _persist_complete_triplet(tmp_path)
    expected = context["expected"]
    first_method = expected["method_order"][0]
    identity = context["identity"]
    missing = paths["method_results_dir"] / f"{identity}.00.{first_method}.json"
    missing.unlink()
    rollout_called = False

    def forbidden_rollout(**_kwargs: object) -> dict[str, object]:
        nonlocal rollout_called
        rollout_called = True
        raise AssertionError("existing pair recovery must not execute a rollout")

    monkeypatch.setattr(runner, "execute_rollout", forbidden_rollout)
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="lacks a method result",
    ):
        runner.recover_complete_existing_triplet(**paths, **context)
    assert rollout_called is False
