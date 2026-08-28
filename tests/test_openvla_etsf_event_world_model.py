from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


def make_model() -> ActionConditionedEventWorldModel:
    torch.manual_seed(7)
    return ActionConditionedEventWorldModel(
        EventWorldModelConfig(
            state_input_dim=32,
            action_dim=6,
            proprio_dim=5,
            semantic_dim=12,
            action_hidden_dim=10,
            transition_hidden_dim=18,
            clock_hidden_dim=8,
            object_delta_dim=9,
            num_bodies=3,
            num_policies=4,
            metadata_dim=6,
            dropout=0.0,
        )
    ).eval()


def make_structured_model(
    *, allow_event_regress: bool = True
) -> ActionConditionedEventWorldModel:
    torch.manual_seed(11)
    return ActionConditionedEventWorldModel(
        EventWorldModelConfig(
            state_input_dim=32,
            action_dim=6,
            proprio_dim=5,
            semantic_dim=12,
            action_hidden_dim=10,
            transition_hidden_dim=18,
            clock_hidden_dim=8,
            object_delta_dim=9,
            num_bodies=3,
            num_policies=4,
            metadata_dim=6,
            structured_events=True,
            allow_event_regress=allow_event_regress,
            dropout=0.0,
        )
    ).eval()


def test_forward_shapes_and_distribution_scales_are_finite() -> None:
    model = make_model()
    hidden = torch.randn(4, 32)
    actions = torch.randn(4, 7, 6)
    output = model(
        hidden,
        actions,
        proprio=torch.randn(4, 5),
        body_id=torch.tensor([0, 1, 2, 0]),
        policy_id=torch.tensor([0, 1, 2, 3]),
        current_event_id=torch.tensor([0, 1, 2, 4]),
        beta=torch.tensor([0.0, 0.2, -0.1, 0.3]),
    )
    assert output["next_event_logits"].shape == (4, 5)
    assert output["reach_logits"].shape == (4, 5)
    assert output["reach_logit"].shape == (4,)
    assert output["duration_log_mean"].shape == (4, 5)
    assert output["duration_log_scale"].shape == (4, 5)
    assert output["success_logit"].shape == (4,)
    assert output["outcome_logits"].shape == (4, 3)
    assert output["object_delta_mean"].shape == (4, 9)
    assert output["object_delta_log_scale"].shape == (4, 9)
    assert output["predicted_next_semantic"].shape == (4, 12)
    assert output["future_latent_log_scale"].shape == (4, 12)
    assert output["aleatoric_uncertainty"].shape == (4,)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert bool((output["aleatoric_uncertainty"] >= 0).all())


def test_action_padding_is_ignored_but_temporal_order_is_not() -> None:
    model = make_model()
    hidden = torch.randn(2, 32)
    actions = torch.randn(2, 6, 6)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    changed_padding = actions.clone()
    changed_padding[:, 4:] = 10_000.0
    first = model(hidden, actions, action_mask=mask)["action_effect"]
    padded = model(hidden, changed_padding, action_mask=mask)["action_effect"]
    assert torch.equal(first, padded)

    reversed_valid = actions.clone()
    reversed_valid[:, :4] = actions[:, :4].flip(1)
    reversed_effect = model(hidden, reversed_valid, action_mask=mask)["action_effect"]
    assert not torch.allclose(first, reversed_effect)


def test_hidden_history_matches_shadow_gru_contract() -> None:
    model = make_model()
    history = torch.randn(2, 5, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    changed_padding = history.clone()
    changed_padding[0, 3:] = 10_000.0
    encoded = model.encode_state(history, mask)
    padded = model.encode_state(changed_padding, mask)
    assert torch.equal(encoded, padded)
    assert not torch.allclose(encoded, model.encode_state(history[:, -1]))

    output = model(history, torch.randn(2, 4, 6), history_mask=mask)
    assert torch.equal(output["semantic"], encoded)


def test_beta_changes_only_clock_outputs() -> None:
    model = make_model()
    hidden = torch.randn(3, 32)
    actions = torch.randn(3, 5, 6)
    common = dict(
        body_id=torch.tensor([0, 1, 2]),
        policy_id=torch.tensor([0, 1, 2]),
        current_event_id=torch.tensor([0, 1, 2]),
        dt=torch.full((3,), 5.0),
    )
    zero = model(hidden, actions, beta=torch.zeros(3), **common)
    shifted = model(hidden, actions, beta=torch.full((3,), 1.5), **common)
    beta_outputs = {
        "clock_log_tau",
        "duration_log_mean",
        "duration_log_scale",
        "duration_selected_log_mean",
        "duration_selected_log_scale",
        "aleatoric_scale_uncertainty",
        "aleatoric_uncertainty",
    }
    for key in zero:
        if key not in beta_outputs:
            assert torch.equal(zero[key], shifted[key]), key
    assert not torch.allclose(zero["clock_log_tau"], shifted["clock_log_tau"])
    assert not torch.allclose(zero["duration_log_mean"], shifted["duration_log_mean"])


def test_candidate_vectorization_and_scoring() -> None:
    model = make_model()
    hidden = torch.randn(2, 32)
    candidates = torch.randn(2, 3, 5, 6)
    predictions = model.predict_candidates(
        hidden,
        candidates,
        body_id=torch.tensor([0, 1]),
        policy_id=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        current_event_id=torch.tensor([0, 2]),
        beta=torch.tensor([0.0, 0.4]),
    )
    assert predictions["success_logit"].shape == (2, 3)
    assert predictions["next_event_logits"].shape == (2, 3, 5)
    scores = model.score_candidates(
        predictions,
        gamma=0.97,
        epistemic_uncertainty=torch.zeros(2, 3),
        candidate_distance=torch.ones(2, 3),
        distance_weight=0.2,
    )
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()


def test_action_rank_residual_is_baseline_anchored_and_candidate_sensitive() -> None:
    base = make_model()
    config = EventWorldModelConfig.from_dict(
        {**base.config_dict(), "action_rank_residual": True}
    )
    model = ActionConditionedEventWorldModel(config).eval()
    # Copy the factual model exactly; only the new relative head is missing.
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("action_rank_head.") for key in incompatible.missing_keys)
    with torch.no_grad():
        # Make the test branch visibly action-sensitive instead of relying on
        # its deliberately zero-initialized deployment state.
        model.action_rank_head[-1].weight.fill_(0.25)

    hidden = torch.randn(2, 32)
    actions = torch.randn(2, 4, 5, 6)
    actions[:, 0] = 0.0
    prediction = model.predict_candidates(
        hidden,
        actions,
        body_id=torch.tensor([0, 1]),
        policy_id=torch.tensor([0, 1]),
        current_event_id=torch.tensor([0, 2]),
    )
    assert torch.equal(
        prediction["action_rank_residual"][:, 0],
        torch.zeros(2),
    )
    assert torch.equal(
        prediction["success_logit"][:, 0],
        prediction["base_success_logit"][:, 0],
    )
    assert bool((prediction["action_rank_residual"][:, 1:].abs() > 0).any())
    assert prediction["aleatoric_uncertainty"].shape == (2, 4)
    assert torch.isfinite(prediction["aleatoric_uncertainty"]).all()


def test_success_only_action_ranker_is_exactly_2d_and_zero_at_baseline() -> None:
    base = make_model()
    config = EventWorldModelConfig.from_dict(
        {
            **base.config_dict(),
            "action_rank_residual": True,
            "action_rank_success_only": True,
        }
    )
    model = ActionConditionedEventWorldModel(config).eval()
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    assert incompatible.missing_keys == ["action_rank_head.0.weight"]
    assert sum(parameter.numel() for parameter in model.action_rank_head.parameters()) == (
        2 * config.semantic_dim
    )
    with torch.no_grad():
        model.action_rank_head[0].weight.normal_()
    hidden = torch.randn(2, 32)
    actions = torch.randn(2, 4, 5, 6)
    prediction = model.predict_candidates(
        hidden,
        actions,
        body_id=torch.tensor([0, 1]),
        policy_id=torch.tensor([0, 1]),
        current_event_id=torch.tensor([0, 2]),
    )
    assert torch.equal(prediction["action_rank_residual"][:, 0], torch.zeros(2))
    assert torch.equal(
        prediction["success_logit"][:, 0],
        prediction["base_success_logit"][:, 0],
    )


def test_success_only_flag_requires_relative_action_head() -> None:
    config = EventWorldModelConfig.from_dict(
        {**make_model().config_dict(), "action_rank_success_only": True}
    )
    with pytest.raises(ValueError, match="requires action_rank_residual"):
        ActionConditionedEventWorldModel(config)


def test_scoring_rewards_success_and_penalizes_uncertainty() -> None:
    model = make_model()
    predictions = {
        "next_event_logits": torch.zeros(1, 2, 5),
        "duration_log_mean": torch.zeros(1, 2, 5),
        "duration_log_scale": torch.full((1, 2, 5), -5.0),
        "success_logit": torch.tensor([[-3.0, 3.0]]),
        "aleatoric_uncertainty": torch.tensor([[0.1, 0.1]]),
    }
    scores = model.score_candidates(predictions, event_value_weight=0.0)
    assert scores[0, 1] > scores[0, 0]
    predictions["success_logit"] = torch.zeros(1, 2)
    predictions["aleatoric_uncertainty"] = torch.tensor([[0.1, 0.9]])
    scores = model.score_candidates(
        predictions, event_value_weight=0.0, uncertainty_weight=1.0
    )
    assert scores[0, 0] > scores[0, 1]


def test_config_and_checkpoint_round_trip() -> None:
    model = make_model()
    config = EventWorldModelConfig.from_dict(model.config_dict())
    restored = ActionConditionedEventWorldModel(config)
    restored.load_state_dict(model.checkpoint_payload(epoch=2)["model"])
    assert restored.config == model.config
    assert math.isclose(restored.config.dropout, 0.0)


def test_structured_event_distribution_is_normalized_and_consistent() -> None:
    model = make_structured_model()
    current = torch.tensor([0, 1, 3, 4])
    output = model(
        torch.randn(4, 32),
        torch.randn(4, 7, 6),
        proprio=torch.randn(4, 5),
        current_event_id=current,
        current_predicates=torch.zeros(4, 5),
    )
    probability = output["next_event_logits"].softmax(-1)
    assert torch.allclose(probability.sum(-1), torch.ones(4), atol=1e-6)
    assert output["relative_transition_logits"].shape == (4, 4)
    assert output["post_predicate_logits"].shape == (4, 5)

    absolute = output["next_event_logits"].argmax(-1)
    relative_from_absolute = model.relative_transition_targets(current, absolute)
    relative = output["relative_transition_logits"].argmax(-1)
    assert torch.equal(relative_from_absolute, relative)
    # At the first event there is no lower event; at the final event there is
    # no advance or skip destination.
    relative_probability = output["relative_transition_logits"].softmax(-1)
    assert relative_probability[0, 3] == 0
    assert relative_probability[3, 1] == 0
    assert relative_probability[3, 2] == 0


def test_structured_forward_requires_explicit_event_and_predicates() -> None:
    model = make_structured_model()
    hidden = torch.randn(2, 32)
    actions = torch.randn(2, 4, 6)
    with pytest.raises(ValueError, match="explicit current_event_id"):
        model(hidden, actions, current_predicates=torch.zeros(2, 5))
    with pytest.raises(ValueError, match="explicit current_predicates"):
        model(hidden, actions, current_event_id=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="must lie in \\[0,1\\]"):
        model(
            hidden,
            actions,
            current_event_id=torch.tensor([0, 1]),
            current_predicates=torch.full((2, 5), 1.5),
        )


def test_structured_monotonic_mode_assigns_zero_mass_to_regression() -> None:
    model = make_structured_model(allow_event_regress=False)
    current = torch.tensor([1, 2, 3, 4])
    output = model(
        torch.randn(4, 32),
        torch.randn(4, 7, 6),
        current_event_id=current,
        current_predicates=torch.zeros(4, 5),
    )
    probability = output["next_event_logits"].softmax(-1)
    for row, event_id in enumerate(current.tolist()):
        assert float(probability[row, :event_id].sum().detach()) == 0.0
    assert torch.equal(
        output["relative_transition_logits"].softmax(-1)[:, 3],
        torch.zeros(4),
    )


def test_old_unstructured_state_dict_contract_has_no_new_parameters() -> None:
    old = make_model()
    assert not any(
        "relative_transition_head" in key or "post_predicate_head" in key
        for key in old.state_dict()
    )
    restored = ActionConditionedEventWorldModel(
        EventWorldModelConfig.from_dict(old.config_dict())
    )
    restored.load_state_dict(old.state_dict(), strict=True)
