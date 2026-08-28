from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    frozen_tensor_mapping_sha256,
    module_state_sha256,
)
from openvla_etsf_v9_group_relative_success_adapter import (  # noqa: E402
    GroupRelativeAdapterConfig,
    GroupRelativeSuccessRankingAdapter,
    adapter_losses,
    load_serialized_adapter,
    predict_group_relative_adapter,
    preregistered_config_grid,
    serialize_adapter_state,
    train_group_relative_adapter,
)


def synthetic_records(count: int = 20, transition_dim: int = 3) -> list[dict]:
    generator = torch.Generator().manual_seed(20260827)
    records = []
    names = [
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
        "continuation_0",
        "continuation_1",
        "continuation_2",
        "continuation_3",
    ]
    for group_index in range(count):
        winner = group_index % 4
        common = torch.randn(transition_dim, generator=generator) * 3.0
        transition = common.repeat(8, 1)
        labels = torch.zeros(8)
        labels[winner] = 1.0
        for candidate in range(4):
            transition[candidate, 0] += 2.5 if candidate == winner else -1.0
            transition[candidate, 1] += 0.1 * candidate
        transition[4:] += torch.randn(4, transition_dim, generator=generator)
        factual = {
            "transition": transition.detach().clone(),
            "duration_selected_log_mean": torch.ones(8),
        }
        records.append(
            {
                "logical_group_key": f"task|body|group_{group_index:03d}",
                "batch": {
                    "terminal_mask": torch.tensor(
                        [True, True, True, True, False, False, False, False]
                    ),
                    "success": labels,
                    "candidate_names": names,
                },
                "factual_outputs": factual,
                "factual_outputs_sha256": frozen_tensor_mapping_sha256(factual),
                "factual_outputs_require_grad": False,
            }
        )
    return records


def _config(
    mode: str = "deterministic_delta",
    objective: str = "pairwise_logistic",
) -> GroupRelativeAdapterConfig:
    return GroupRelativeAdapterConfig(
        transition_dim=3,
        relative_mode=mode,
        ranking_objective=objective,
        l2_regularization=1e-3,
    )


@pytest.mark.parametrize("mode", ["deterministic_delta", "group_centered"])
def test_group_relative_features_remove_group_shared_transition_offset(
    mode: str,
) -> None:
    adapter = GroupRelativeSuccessRankingAdapter(_config(mode=mode))
    transition = torch.randn(5, 4, 3)
    shared_offset = torch.randn(5, 1, 3) * 100.0
    first = adapter.raw_relative_features(transition)
    second = adapter.raw_relative_features(transition + shared_offset)
    assert torch.allclose(first, second, atol=2e-5)
    if mode == "deterministic_delta":
        assert torch.equal(first[:, 0, :3], torch.zeros_like(first[:, 0, :3]))
    else:
        assert torch.allclose(first[:, :, :3].mean(dim=1), torch.zeros(5, 3), atol=1e-6)


def test_probability_and_ranking_losses_have_disjoint_gradient_targets() -> None:
    records = synthetic_records(8)
    transition = torch.stack(
        [record["factual_outputs"]["transition"][:4] for record in records]
    )
    labels = torch.stack([record["batch"]["success"][:4] for record in records])
    adapter = GroupRelativeSuccessRankingAdapter(_config())
    adapter.fit_feature_scaler(transition)
    losses = adapter_losses(adapter, transition, labels)
    probability_to_ranking = torch.autograd.grad(
        losses["probability_objective"],
        adapter.ranking_parameters(),
        allow_unused=True,
        retain_graph=True,
    )
    ranking_to_probability = torch.autograd.grad(
        losses["ranking_objective"],
        adapter.probability_parameters(),
        allow_unused=True,
        retain_graph=True,
    )
    assert probability_to_ranking == (None, None)
    assert ranking_to_probability == (None, None)
    assert not (
        {id(value) for value in adapter.probability_parameters()}
        & {id(value) for value in adapter.ranking_parameters()}
    )


@pytest.mark.parametrize(
    "objective", ["pairwise_logistic", "listwise_success_cross_entropy"]
)
def test_training_is_bit_exact_for_factual_inputs_and_emits_independent_outputs(
    objective: str,
) -> None:
    records = synthetic_records(20)
    tensors_before = [
        record["factual_outputs"]["transition"].clone() for record in records
    ]
    adapter, audit = train_group_relative_adapter(
        records, config=_config(objective=objective), device="cpu"
    )
    prediction = predict_group_relative_adapter(adapter, records, device="cpu")
    assert prediction["success_probability"].shape == (20, 4)
    assert prediction["candidate_ranking_score"].shape == (20, 4)
    assert torch.isfinite(prediction["success_probability"]).all()
    assert torch.isfinite(prediction["candidate_ranking_score"]).all()
    assert audit["factual_outputs_bit_exact"] is True
    assert audit["probability_and_ranking_parameters_disjoint"] is True
    assert audit["shared_trainable_representation"] is False
    assert audit["unweighted_success_bce"] is True
    assert all(
        torch.equal(before, record["factual_outputs"]["transition"])
        for before, record in zip(tensors_before, records)
    )
    selected = prediction["candidate_ranking_score"].argmax(dim=1)
    labels = prediction["success_label"]
    selected_success = labels[torch.arange(len(labels)), selected].float().mean()
    assert float(selected_success) > 0.80


def test_grid_is_small_fixed_and_deterministic() -> None:
    first = preregistered_config_grid(3)
    second = preregistered_config_grid(3)
    assert first == second
    assert len(first) == 8
    assert len({config.config_id for config in first}) == 8
    assert {config.ranking_loss_weight for config in first} == {1.0}


def test_serialized_adapter_state_round_trips_with_identical_hash_and_outputs() -> None:
    records = synthetic_records(12)
    adapter, _ = train_group_relative_adapter(records, config=_config())
    restored = load_serialized_adapter(
        adapter.config, serialize_adapter_state(adapter)
    )
    assert module_state_sha256(restored) == module_state_sha256(adapter)
    first = predict_group_relative_adapter(adapter, records)
    second = predict_group_relative_adapter(restored, records)
    assert torch.equal(first["success_probability"], second["success_probability"])
    assert torch.equal(
        first["candidate_ranking_score"], second["candidate_ranking_score"]
    )


def test_record_authentication_and_terminal_layout_fail_closed() -> None:
    records = synthetic_records(8)
    changed = copy.deepcopy(records)
    changed[0]["factual_outputs"]["transition"][0, 0] += 1.0
    with pytest.raises(ValueError, match="authentication"):
        train_group_relative_adapter(changed, config=_config())
    changed = copy.deepcopy(records)
    changed[0]["batch"]["terminal_mask"][4] = True
    changed[0]["factual_outputs_sha256"] = frozen_tensor_mapping_sha256(
        changed[0]["factual_outputs"]
    )
    with pytest.raises(ValueError, match="terminal layout"):
        train_group_relative_adapter(changed, config=_config())


def test_probability_head_is_not_the_task_action_rule() -> None:
    adapter = GroupRelativeSuccessRankingAdapter(_config())
    transition = torch.randn(2, 4, 3)
    adapter.fit_feature_scaler(transition)
    with torch.no_grad():
        adapter.probability_head.weight.zero_()
        adapter.probability_head.bias.fill_(5.0)
        adapter.ranking_head.weight.zero_()
        adapter.ranking_head.bias.zero_()
    output = adapter(transition)
    assert torch.equal(
        output["candidate_ranking_score"].argmax(dim=1), torch.tensor([0, 0])
    )
    assert torch.all(torch.sigmoid(output["success_logit"]) > 0.99)


def test_prediction_probability_stays_open_interval_after_float32_saturation() -> None:
    records = synthetic_records(8)
    adapter = GroupRelativeSuccessRankingAdapter(_config())
    transition = torch.stack(
        [record["factual_outputs"]["transition"][:4] for record in records]
    )
    adapter.fit_feature_scaler(transition)
    with torch.no_grad():
        adapter.probability_head.weight.zero_()
        adapter.probability_head.bias.fill_(100.0)
    # Direct float32 sigmoid saturates, but the metric-facing probability
    # contract must remain strictly inside (0,1).
    assert torch.sigmoid(adapter(transition)["success_logit"]).eq(1.0).all()
    probability = predict_group_relative_adapter(adapter, records)[
        "success_probability"
    ]
    assert probability.dtype == torch.float64
    assert torch.all((probability > 0.0) & (probability < 1.0))
