from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "postformal_pool_runner",
    SCRIPTS / "run_robotwin2_five_body_postformal_candidate_pool_v1.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _current() -> np.ndarray:
    return np.asarray(
        [
            0.45,
            0.10,
            0.80,
            1.0,
            0.0,
            0.0,
            0.0,
            0.2,
            -0.45,
            0.10,
            0.80,
            1.0,
            0.0,
            0.0,
            0.0,
            0.2,
        ],
        dtype=np.float32,
    )


def _proposals(count: int, horizon: int = 6) -> np.ndarray:
    current = _current()
    proposals = np.repeat(current[None, None], count * horizon, axis=0).reshape(
        count, horizon, 16
    )
    for index in range(count):
        proposals[index, :, 0] += (index + 1) * np.linspace(
            0.0005, 0.003, horizon, dtype=np.float32
        )
        proposals[index, :, 8] -= (index + 1) * np.linspace(
            0.00025, 0.0015, horizon, dtype=np.float32
        )
        proposals[index, :, 7] = np.clip(0.2 + 0.01 * index, 0.0, 1.0)
    return proposals


def test_default_budget_is_raw_n_and_does_not_claim_subset_selection() -> None:
    assert runner.proposal_count(8) == 8
    assert runner.proposal_count(16) == 16
    contract = runner.candidate_pool_contract(8)
    assert contract["raw_proposal_count"] == 8
    assert contract["subset_selection_applied"] is False
    assert contract["selection"] == "identity_original_actor_order"

    raw = _proposals(8)
    retained, audit = runner.pool_selection_audit(
        current_ee=_current(), raw_proposals=raw, candidate_count=8
    )
    assert np.array_equal(retained, raw)
    assert audit["selected_raw_proposal_indices"] == list(range(8))
    assert audit["subset_selection_applied"] is False
    assert audit["selection_algorithm"].startswith("identity_keep_original")
    assert audit["selection_reads_outcomes_events_or_critic_scores"] is False


def test_explicit_oversampling_is_bounded_and_uses_blind_anchored_fps() -> None:
    assert runner.proposal_count(8, 16) == 16
    for invalid in (7, 9, 18):
        with pytest.raises(runner.PostformalCandidatePoolError):
            runner.proposal_count(8, invalid)

    raw = _proposals(16)
    retained, audit = runner.pool_selection_audit(
        current_ee=_current(),
        raw_proposals=raw,
        candidate_count=8,
        raw_proposal_count=16,
    )
    selected = audit["selected_raw_proposal_indices"]
    assert retained.shape == (8, 6, 16)
    assert selected[0] == 0
    assert len(selected) == len(set(selected)) == 8
    assert np.array_equal(retained[0], raw[0])
    assert audit["subset_selection_applied"] is True
    assert audit["selection_algorithm"].startswith("greedy_maximize_minimum_rms")
    assert audit["selection_reads_outcomes_events_or_critic_scores"] is False


def test_farthest_point_ties_break_by_lowest_raw_index() -> None:
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.5],
        ],
        dtype=np.float64,
    )
    # Raw 1 and 2 are equally far from anchor zero; raw 1 must win the tie.
    assert runner.greedy_farthest_point_indices(embeddings, retain_count=2) == [0, 1]


def test_dynamic_candidate_axis_keeps_frozen_five_member_lcb_formula() -> None:
    values = torch.arange(5 * 8, dtype=torch.float64).reshape(5, 8) / 10.0
    observed = runner.aggregate_risk_adjusted_rank_scores(values)
    expected = values.mean(dim=0) - float(
        runner.shared_head.EPISTEMIC_RANK_RISK_WEIGHT
    ) * values.std(dim=0, correction=0)
    assert torch.equal(observed, expected)
    # The formal training/runtime helper remains frozen to four candidates;
    # the extension is isolated in this postformal runner.
    with pytest.raises(runner.shared_head.FiveBodyContractError):
        runner.shared_head.aggregate_risk_adjusted_rank_scores(values)


def test_score_candidates_accepts_eight_without_any_selection_gate() -> None:
    class FixedModel:
        def __init__(self, member: int):
            self.member = member

        def __call__(self, batch):
            count = int(batch["actions"].shape[0])
            zeros = torch.zeros(count)
            events = torch.zeros(count, 5)
            rank = torch.arange(count, dtype=torch.float32) / 10.0
            rank[5] = 2.0 - 0.01 * self.member
            return {
                "candidate_rank_logit": rank,
                "success_logit": zeros,
                "post_event_logits": events,
                "next_event_logits": events,
                "duration_selected_log_mean": zeros,
                "duration_selected_log_scale": zeros,
                "terminal_event_logits": events,
                "terminal_goal_progress_mean": zeros,
                "terminal_goal_progress_log_scale": zeros,
                "regression_probability": zeros,
                "joint_recovery_probability": zeros,
            }

    batch = {"actions": torch.zeros(8, 5, 14)}
    result = runner.score_candidates(
        [FixedModel(index) for index in range(5)], batch, candidate_count=8
    )
    assert result["selected_candidate_index"] == 5
    assert len(result["candidate_rank_score_epistemic_lcb_ensemble"]) == 8
    assert result["aggregation"]["candidate_axis_extension_only"] is True
    assert "gate" not in result


def test_schedule_is_full_and_baseline_is_paired_with_named_n8_method() -> None:
    schedule = runner.evaluation_schedule(8)
    assert len(schedule) == 1000
    assert schedule[0]["method_order"] == [
        "actor_baseline",
        "etsf_actor_flow_best_of_8",
    ]
    assert schedule[1]["method_order"] == [
        "etsf_actor_flow_best_of_8",
        "actor_baseline",
    ]
    assert schedule[-1]["requested_seed"] == runner.SEED_BASE + 99
