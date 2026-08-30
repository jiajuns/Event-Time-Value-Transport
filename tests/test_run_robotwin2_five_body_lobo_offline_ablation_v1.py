from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_robotwin2_five_body_lobo_offline_ablation_v1 as ablation  # noqa: E402


def test_inventory_is_full_and_uses_late_episode_queries() -> None:
    manifests = {}
    for body_index, body in enumerate(ablation.trainer.BODIES):
        groups = []
        for condition_index, condition in enumerate(ablation.trainer.CONDITIONS):
            for query in ablation.QUERY_INDICES:
                for ordinal in range(ablation.SEEDS_PER_CONDITION_QUERY):
                    groups.append(
                        {
                            "condition": condition,
                            "root_query_index": query,
                            "requested_seed": (
                                2_026_081_000
                                + body_index * 10_000
                                + condition_index * 1_000
                                + query * 50
                                + ordinal
                            ),
                        }
                    )
        manifests[body] = {"groups": groups}
    audit = {"manifests": manifests}
    receipt = ablation.validate_complete_inventory(audit)
    assert ablation.QUERY_INDICES == tuple(range(40))
    assert receipt["root_query_indices"] == list(range(40))
    assert receipt["decisions"] == 2_000
    assert receipt["branches"] == 8_000

    audit["manifests"][ablation.trainer.BODIES[0]]["groups"][0][
        "root_query_index"
    ] = 40
    with pytest.raises(ablation.AblationError):
        ablation.validate_complete_inventory(audit)


def test_risk_adjusted_frozen_rank_ensemble_uses_bounded_member_consensus() -> None:
    class FixedRank(torch.nn.Module):
        def __init__(self, scores: list[float]) -> None:
            super().__init__()
            self.register_buffer("scores", torch.tensor(scores))

        def forward(self, _batch: dict[str, object]) -> dict[str, torch.Tensor]:
            return {"candidate_rank_logit": self.scores}

    # Every deployed member emits the same bounded physical utility.  One
    # member prefers candidate 1, while four members agree on candidate 2.
    models = [FixedRank([0.0, 1.0, 0.0, 0.0])] + [
        FixedRank([0.0, 0.0, 1.0, 0.0]) for _ in range(4)
    ]
    ensemble = ablation._FrozenRankEnsemble(models)
    output = ensemble(
        {
            "logical_group": ["g"] * 4,
            "candidate_index": torch.tensor([0, 1, 2, 3]),
        }
    )["candidate_rank_logit"]
    assert int(output.argmax()) == 2
    assert ablation.trainer.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT[
        "epistemic_risk_weight"
    ] == 0.25
    assert (
        ablation.trainer.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT[
            "within_member_candidate_standardization"
        ]
        is False
    )


class _PerfectPredictionModel(torch.nn.Module):
    def __init__(self, member_offset: float) -> None:
        super().__init__()
        self.member_offset = float(member_offset)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        success = batch["success"].float()
        recovery = batch["recovery"].float()
        post = batch["post_event_id"].long()
        following = batch["next_event_id"].long()
        terminal_event = batch["terminal_max_event_id"].long()
        terminal_goal = batch["terminal_goal_progress"].float()
        current_event = batch["current_event_id"].long()
        object_delta = batch["object_delta"].float()
        duration = batch["duration"].float()
        count = len(success)
        post_logits = torch.full((count, 5), -7.0, dtype=torch.float32)
        next_logits = torch.full((count, 5), -7.0, dtype=torch.float32)
        terminal_logits = torch.full((count, 5), -7.0, dtype=torch.float32)
        post_logits[torch.arange(count), post] = 7.0 + self.member_offset
        next_logits[torch.arange(count), following] = 7.0 + self.member_offset
        terminal_logits[torch.arange(count), terminal_event] = (
            7.0 + self.member_offset
        )
        # Candidate score is monotone with success; exact ties use the same
        # lowest-index convention as ensemble success argmax.
        rank = 5.0 * success
        return {
            "candidate_rank_logit": rank + self.member_offset,
            "success_logit": (success * 2.0 - 1.0) * (7.0 + self.member_offset),
            "post_event_logits": post_logits,
            "next_event_logits": next_logits,
            "duration_selected_log_mean": torch.log1p(duration)
            + self.member_offset * 0.001,
            "duration_selected_log_scale": torch.full_like(duration, -3.0),
            "duration_component_log_mean": (
                torch.log1p(duration)[:, None].expand(-1, 5)
                + self.member_offset * 0.001
            ),
            "duration_component_log_scale": torch.full(
                (count, 5), -3.0, dtype=duration.dtype
            ),
            "recovery_logit": (recovery * 2.0 - 1.0) * (7.0 + self.member_offset),
            "object_delta_mean": object_delta + self.member_offset * 0.001,
            "object_delta_log_scale": torch.full_like(object_delta, -3.0),
            "terminal_event_logits": terminal_logits,
            "terminal_goal_progress_mean": (
                terminal_goal + self.member_offset * 0.001
            ),
            "terminal_goal_progress_log_scale": torch.full_like(
                terminal_goal, -3.0
            ),
            "regression_probability": (post < current_event).float(),
            "joint_recovery_probability": (
                (post < current_event).float() * recovery
            ),
        }


def _prediction_batch() -> dict[str, object]:
    count = 8
    success = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    recovery = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    duration = torch.arange(1, count + 1, dtype=torch.float32) / 10.0
    object_delta = torch.arange(count * 6, dtype=torch.float32).reshape(count, 6) / 100.0
    return {
        "logical_group": [
            "piper|clean|clean|seed=2026081001|query=0"
        ]
        * 4
        + ["piper|randomized|randomized|seed=2026081002|query=10"] * 4,
        "candidate_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        "post_event_id": torch.tensor([0, 1, 2, 3, 1, 2, 3, 4]),
        "current_event_id": torch.tensor([1, 1, 2, 3, 2, 2, 3, 4]),
        "post_event_mask": torch.ones(count),
        "next_event_id": torch.tensor([1, 2, 3, 4, 0, 1, 2, 3]),
        "next_event_mask": torch.ones(count),
        "duration": duration,
        "duration_observed": torch.tensor([1, 1, 0, 1, 1, 0, 1, 1]),
        "duration_mask": torch.ones(count),
        "success": success,
        "success_mask": torch.ones(count),
        "recovery": recovery,
        "recovery_mask": torch.ones(count),
        "action_available": torch.ones(count),
        "object_delta": object_delta,
        "object_delta_mask": torch.ones(count),
        "terminal_max_event_id": torch.tensor([1, 2, 2, 4, 2, 3, 4, 4]),
        "terminal_event_mask": torch.ones(count),
        "terminal_goal_progress": torch.linspace(-0.04, 0.12, count),
        "terminal_goal_progress_mask": torch.ones(count),
    }


def test_deployed_ensemble_prediction_metrics_cover_all_heads_and_uncertainty() -> None:
    models = [_PerfectPredictionModel(offset) for offset in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    result = ablation.evaluate_deployed_ensemble_predictions(
        models, [_prediction_batch()], torch.device("cpu")
    )
    metrics = result["metrics"]
    assert metrics["success_brier"] < 1e-4
    assert metrics["success_nll"] < 0.01
    assert metrics["success_auroc"] == pytest.approx(1.0)
    assert metrics["post_event_macro_f1"] == pytest.approx(1.0)
    assert metrics["next_event_macro_f1"] == pytest.approx(1.0)
    assert metrics["duration_mixture_mae_seconds"] < 0.01
    assert metrics["duration_mixture_nll_log1p"] is not None
    assert metrics["duration_mixture_censored_nll_log1p"] is not None
    assert metrics["object_mixture_rmse"] < 0.01
    assert metrics["recovery_brier"] < 1e-4
    assert metrics["recovery_average_precision"] == pytest.approx(1.0)
    assert result["recovery_precision_recall"]["status"] == "available"
    assert result["recovery_precision_recall"]["points"][-1]["recall"] == 1.0
    assert metrics["rank_success_argmax_disagreement_rate"] == pytest.approx(0.0)
    assert result["support"]["complete_four_candidate_decisions"] == 2
    assert result["support"]["requested_seed_clusters"] == 2
    assert set(result["uncertainty_risk_coverage"]) == {
        "rank_selected_failure",
        "rank_oracle_regret",
        "success",
        "post_event",
        "next_event",
        "terminal_event",
        "duration",
        "duration_epistemic",
        "object",
        "terminal_goal_progress",
        "recovery",
        "regression",
        "joint_recovery",
    }
    assert result["statistical_units"]["dependence_cluster_unit"].endswith(
        "all_query_decisions_and_candidates"
    )


def test_recovery_pr_is_explicitly_unavailable_without_positive_labels() -> None:
    batch = _prediction_batch()
    batch["recovery"] = torch.zeros(8)
    models = [_PerfectPredictionModel(offset) for offset in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    result = ablation.evaluate_deployed_ensemble_predictions(
        models, [batch], torch.device("cpu")
    )
    assert result["metrics"]["recovery_average_precision"] is None
    assert result["support"]["recovery"]["precision_recall_available"] is False
    assert result["recovery_precision_recall"] == {
        "status": "unavailable_no_positive_labels",
        "positive": 0,
        "negative": 8,
        "average_precision": None,
        "points": [],
    }


def test_duration_evaluator_uses_exact_next_event_competing_risks_survival() -> None:
    probability = torch.tensor([0.50, 0.20, 0.15, 0.10, 0.05])
    component_mean = torch.tensor([0.0, 0.4, 0.8, 1.2, 1.6])
    component_scale = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    class FixedCompetingRiskModel(_PerfectPredictionModel):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            output = super().forward(batch)
            count = len(batch["duration"])
            output["next_event_logits"] = probability.log()[None].expand(count, -1)
            output["duration_component_log_mean"] = component_mean[None].expand(
                count, -1
            )
            output["duration_component_log_scale"] = component_scale.log()[
                None
            ].expand(count, -1)
            return output

    batch = _prediction_batch()
    batch["duration"] = torch.ones(8)
    batch["duration_observed"] = torch.zeros(8)
    models = [FixedCompetingRiskModel(0.0) for _ in range(5)]
    result = ablation.evaluate_deployed_ensemble_predictions(
        models, [batch], torch.device("cpu")
    )
    z = (math.log(2.0) - component_mean) / component_scale
    expected = -torch.logsumexp(
        probability.log() + torch.special.log_ndtr(-z), dim=-1
    )
    assert result["metrics"][
        "duration_mixture_censored_nll_log1p"
    ] == pytest.approx(float(expected), abs=1e-6)


def test_duration_observed_ensemble_conditions_by_member_event_probability_and_reports_total_uncertainty(
) -> None:
    probabilities = torch.tensor(
        [
            [0.05, 0.05, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.50, 0.15, 0.15],
            [0.15, 0.15, 0.30, 0.20, 0.20],
            [0.20, 0.20, 0.15, 0.25, 0.20],
            [0.24, 0.24, 0.04, 0.24, 0.24],
        ]
    )
    means = torch.tensor(
        [
            [0.0, 0.2, 0.60, 1.0, 1.2],
            [0.1, 0.3, 0.75, 1.1, 1.3],
            [0.2, 0.4, 0.90, 1.2, 1.4],
            [0.3, 0.5, 1.05, 1.3, 1.5],
            [0.4, 0.6, 1.20, 1.4, 1.6],
        ]
    )
    scales = torch.tensor(
        [
            [0.25, 0.30, 0.35, 0.40, 0.45],
            [0.30, 0.35, 0.40, 0.45, 0.50],
            [0.35, 0.40, 0.45, 0.50, 0.55],
            [0.40, 0.45, 0.50, 0.55, 0.60],
            [0.45, 0.50, 0.55, 0.60, 0.65],
        ]
    )

    class FixedObservedCompetingRiskModel(_PerfectPredictionModel):
        def __init__(self, member: int) -> None:
            super().__init__(0.0)
            self.member = member

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            output = super().forward(batch)
            count = len(batch["duration"])
            output["next_event_logits"] = probabilities[self.member].log()[
                None
            ].expand(count, -1)
            output["duration_component_log_mean"] = means[self.member][
                None
            ].expand(count, -1)
            output["duration_component_log_scale"] = scales[self.member].log()[
                None
            ].expand(count, -1)
            return output

    batch = _prediction_batch()
    batch["duration"] = torch.ones(8)
    batch["duration_observed"] = torch.ones(8)
    batch["next_event_id"] = torch.full((8,), 2, dtype=torch.long)
    result = ablation.evaluate_deployed_ensemble_predictions(
        [FixedObservedCompetingRiskModel(member) for member in range(5)],
        [batch],
        torch.device("cpu"),
    )

    target = math.log(2.0)
    event_probability = probabilities[:, 2].double()
    event_mean = means[:, 2].double()
    event_scale = scales[:, 2].double()
    log_pdf = (
        -0.5 * torch.square((target - event_mean) / event_scale)
        - torch.log(event_scale)
        - 0.5 * math.log(2.0 * math.pi)
    )
    expected_observed_nll = -(
        torch.logsumexp(torch.log(event_probability) + log_pdf, dim=0)
        - torch.logsumexp(torch.log(event_probability), dim=0)
    )
    assert result["metrics"][
        "duration_mixture_observed_nll_log1p"
    ] == pytest.approx(float(expected_observed_nll), abs=1e-6)

    probability_np = probabilities.double().numpy()
    mean_np = means.double().numpy()
    variance_np = np.square(scales.double().numpy())
    component_raw_mean = np.expm1(mean_np + 0.5 * variance_np)
    component_raw_variance = (
        np.expm1(variance_np) * np.exp(2.0 * mean_np + variance_np)
    )
    member_mean = np.sum(probability_np * component_raw_mean, axis=-1)
    member_aleatoric = np.sum(
        probability_np
        * (
            component_raw_variance
            + np.square(component_raw_mean - member_mean[:, None])
        ),
        axis=-1,
    )
    expected_epistemic_std = math.sqrt(float(np.var(member_mean, ddof=0)))
    expected_mean_aleatoric = float(np.mean(member_aleatoric))
    expected_total_std = math.sqrt(
        expected_mean_aleatoric + expected_epistemic_std**2
    )
    metrics = result["metrics"]
    assert metrics["duration_epistemic_std_mean_seconds"] == pytest.approx(
        expected_epistemic_std, abs=1e-6
    )
    assert metrics[
        "duration_mean_member_aleatoric_variance_mean_seconds2"
    ] == pytest.approx(expected_mean_aleatoric, abs=1e-6)
    assert metrics["duration_total_std_mean_seconds"] == pytest.approx(
        expected_total_std, abs=1e-6
    )
    assert metrics["duration_error_aurc"] == metrics["duration_total_error_aurc"]
    assert result["uncertainty_risk_coverage"]["duration"][
        "uncertainty_kind"
    ].startswith("sqrt_mean_member_raw_time_aleatoric")
    assert result["uncertainty_risk_coverage"]["duration_epistemic"][
        "uncertainty_kind"
    ].startswith("five_member_raw_time_distribution_mean")
    assert result["duration_uncertainty_decomposition"][
        "primary_duration_aurc_uses"
    ] == "total_standard_deviation"


def test_risk_coverage_retains_low_uncertainty_first() -> None:
    curve = ablation._risk_coverage(
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([0.0, 0.1, 0.8, 0.9]),
        error_kind="test_error",
        uncertainty_kind="test_uncertainty",
    )
    assert curve["support"] == 4
    assert curve["risk_at_coverage"][1]["coverage"] == 0.5
    assert curve["risk_at_coverage"][1]["risk"] == pytest.approx(0.0)
    assert curve["full_coverage_risk"] == pytest.approx(0.5)
    assert curve["error_uncertainty_spearman"] > 0.8


def test_posthoc_macro_is_equal_fold_and_keeps_risk_coverage() -> None:
    models = [_PerfectPredictionModel(offset) for offset in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    prediction = ablation.evaluate_deployed_ensemble_predictions(
        models, [_prediction_batch()], torch.device("cpu")
    )
    fold_metrics = dict(prediction["metrics"])
    fold_metrics.update(
        {
            "one_deviation_best_of_4_success_gain": 0.1,
            "one_deviation_branch_selected_success_rate": 0.6,
            "one_deviation_branch_oracle_success_rate": 0.8,
            "one_deviation_branch_pairwise_accuracy": 0.75,
        }
    )
    assert set(fold_metrics) == set(ablation.POSTHOC_ENSEMBLE_METRICS)
    evaluations = {
        variant: {
            body: {
                "metrics": dict(fold_metrics),
                "uncertainty_risk_coverage": prediction[
                    "uncertainty_risk_coverage"
                ],
            }
            for body in ablation.trainer.BODIES
        }
        for variant in ablation.VARIANTS
    }
    summary = ablation.aggregate_posthoc_heldout(evaluations)
    assert summary["full"]["equal_fold_macro"][
        "one_deviation_best_of_4_success_gain"
    ] == pytest.approx(0.1)
    assert summary["full"]["uncertainty_risk_coverage_equal_fold_macro"][
        "success"
    ]["folds_with_support"] == 5
    assert summary["full"]["metric_folds_with_support"][
        "recovery_average_precision"
    ] == 5
    assert summary["comparison_to_success_only"]["full"][
        "success_brier"
    ] == pytest.approx(0.0)
