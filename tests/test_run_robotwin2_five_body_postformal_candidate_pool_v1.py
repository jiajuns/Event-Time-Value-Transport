from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
    noise_contract = contract["flow_noise_contract"]
    assert noise_contract["distribution"].startswith(
        "deterministic_independent_standard_normal"
    )
    assert noise_contract["independent_flow_noise_draws"] == 4
    assert noise_contract["raw_proposal_flow_noise_indices"] == [0, 1, 2, 3] * 2
    assert noise_contract["same_flow_noise_language_condition_pairs"] == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    assert noise_contract["candidate_zero_legacy_noise_unchanged"] is True
    assert noise_contract["formal_n4_antithetic_contract_changed"] is False
    assert noise_contract["selection_reads_outcomes_events_or_critic_scores"] is False
    assert contract["actor_call_budget_increased_beyond_raw_proposal_count"] is False
    language = contract["instruction_coverage_contract"]
    assert language["actor_training_seen_condition"] == runner.TRAINING_SEEN_INSTRUCTION
    assert language["actor_training_seen_task_index"] == 1068
    assert language["frozen_actor_training_tasks_parquet_sha256"] == (
        runner.TRAINING_TASKS_PARQUET_SHA256
    )
    assert hashlib.sha256(
        language["actor_training_seen_condition"].encode("utf-8")
    ).hexdigest() == language["actor_training_seen_condition_utf8_sha256"]
    assert "brown" not in language["actor_training_seen_condition"].lower()
    assert "lid" not in language["actor_training_seen_condition"].lower()
    assert "arm" not in language["actor_training_seen_condition"].lower()
    assert language["semantic_task_changed"] is False
    assert language["reads_outcomes_events_or_critic_scores"] is False

    raw = _proposals(8)
    retained, audit = runner.pool_selection_audit(
        current_ee=_current(), raw_proposals=raw, candidate_count=8
    )
    assert np.array_equal(retained, raw)
    assert audit["selected_raw_proposal_indices"] == list(range(8))
    assert audit["subset_selection_applied"] is False
    assert audit["selection_algorithm"].startswith("identity_keep_original")
    assert audit["selection_reads_outcomes_events_or_critic_scores"] is False


def test_independent_flow_draws_preserve_candidate_zero_and_remove_antithetic_pairs() -> None:
    config = SimpleNamespace(chunk_size=6, max_action_dim=16)
    device = torch.device("cpu")
    kwargs = {"scene_seed": 6_200_031, "query_index": 11, "device": device}

    observed = [
        runner.postformal_make_noise(config, flow_noise_index=index, **kwargs)
        for index in range(4)
    ]
    legacy_zero = runner.collector.make_noise(
        config, kwargs["scene_seed"], kwargs["query_index"], 0, device
    )
    assert torch.equal(observed[0], legacy_zero)
    for index, value in enumerate(observed):
        replay = runner.postformal_make_noise(
            config, flow_noise_index=index, **kwargs
        )
        assert torch.equal(value, replay)
    assert all(
        not torch.equal(observed[left], observed[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )
    # The formal collector deterministically spends every odd slot on the
    # negative of the preceding draw.  The postformal pool must not do that.
    assert all(
        not torch.equal(observed[index + 1], -observed[index])
        for index in range(0, 4, 2)
    )


def test_candidate_generation_uses_actor_path_then_ee_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrozenFakePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                image_features=("observation.images.cam_high",),
                chunk_size=6,
                max_action_dim=16,
            )
            self.noises: list[torch.Tensor] = []
            self.processed_instructions: list[str] = []
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def predict_action_chunk(
            self, processed: dict[str, object], *, noise: torch.Tensor
        ) -> torch.Tensor:
            assert processed["preprocessed"] == "same_observation"
            self.processed_instructions.append(str(processed["task"]))
            self.noises.append(noise.detach().clone())
            # A fake frozen actor is enough to exercise the exact external-noise,
            # postprocessor and EE-normalization plumbing without GPU inference.
            result = noise[..., :16].clone()
            if processed["task"] == runner.TRAINING_SEEN_INSTRUCTION:
                result[..., 0] += 0.01
            return result

    raw_calls: list[tuple[object, list[str], str]] = []

    def fake_raw_policy_input(
        task: object, image_features: list[str], instruction: str
    ) -> dict[str, str]:
        raw_calls.append((task, image_features, instruction))
        return {"raw": "same_observation"}

    monkeypatch.setattr(runner.collector, "raw_policy_input", fake_raw_policy_input)
    policy = FrozenFakePolicy()
    task = object()
    candidates = runner.generate_postformal_flow_candidates(
        policy=policy,
        preprocessor=lambda raw: {
            "preprocessed": raw["raw"],
            "task": raw["task"],
        },
        postprocessor=lambda value: value,
        task=task,
        instruction="move the object",
        scene_seed=6_200_043,
        query_index=3,
        candidate_count=8,
        device=torch.device("cpu"),
    )

    assert raw_calls == [
        (task, ["observation.images.cam_high"], "move the object")
    ]
    assert policy.reset_count == 8
    assert len(policy.noises) == 8
    assert policy.processed_instructions == ["move the object"] * 4 + [
        runner.TRAINING_SEEN_INSTRUCTION
    ] * 4
    assert candidates.shape == (8, 6, 16)
    assert np.isfinite(candidates).all()
    assert np.allclose(np.linalg.norm(candidates[:, :, 3:7], axis=-1), 1.0)
    assert np.allclose(np.linalg.norm(candidates[:, :, 11:15], axis=-1), 1.0)
    assert np.all((0.0 <= candidates[:, :, 7]) & (candidates[:, :, 7] <= 1.0))
    assert np.all((0.0 <= candidates[:, :, 15]) & (candidates[:, :, 15] <= 1.0))
    expected_zero = runner.collector.make_noise(
        policy.config, 6_200_043, 3, 0, torch.device("cpu")
    )
    assert torch.equal(policy.noises[0], expected_zero)
    assert not torch.equal(policy.noises[1], -policy.noises[0])
    assert all(torch.equal(policy.noises[index], policy.noises[index + 4]) for index in range(4))
    assert np.any(candidates[1:] != candidates[0])


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


def test_runtime_ensemble_contract_separates_training_and_runtime_axes() -> None:
    contract = runner.runtime_rank_ensemble_contract(8)
    assert "candidate_count" not in contract
    assert contract["training_candidate_count"] == 4
    assert contract["runtime_candidate_count"] == 8
    assert contract["member_count"] == 5
    assert contract["epistemic_risk_weight"] == 0.25
    assert contract["candidate_axis_extension_only"] is True


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
