from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from etsf_transfer_adapters import (  # noqa: E402
    FrozenCoreTransferModel,
    LearnedTransferActionAdapter,
    LearnedTransferStateAdapter,
    TransferAdapterSpec,
)
from openvla_etsf_event_critic_plugin import EmbodimentSpec  # noqa: E402
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


def _core(*, policies: int = 2, bodies: int = 1) -> ActionConditionedEventWorldModel:
    torch.manual_seed(7)
    return ActionConditionedEventWorldModel(
        EventWorldModelConfig(
            state_input_dim=8,
            action_dim=4,
            semantic_dim=8,
            action_hidden_dim=8,
            transition_hidden_dim=12,
            clock_hidden_dim=6,
            metadata_dim=4,
            proprio_dim=0,
            object_delta_dim=3,
            event_names=("e0", "e1", "eK"),
            outcome_names=("failure", "success", "recovery"),
            num_policies=policies,
            num_bodies=bodies,
            dropout=0.0,
        )
    )


def test_policy_adapter_updates_only_reserved_row_and_external_modules() -> None:
    core = _core()
    wrapper = FrozenCoreTransferModel(
        core,
        TransferAdapterSpec(
            axis="policy",
            target_state_dim=6,
            core_state_dim=8,
            native_action_dim=4,
            core_action_dim=4,
            target_body_id=0,
            target_policy_id=1,
            target_embedding_row=1,
            state_bottleneck_dim=5,
            learn_state_adapter=True,
            learn_action_adapter=True,
            learn_clock=False,
        ),
    )
    before = {name: value.detach().clone() for name, value in core.state_dict().items()}
    parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1)
    output = wrapper(
        torch.randn(3, 2, 6),
        torch.randn(3, 5, 4),
        history_mask=torch.ones(3, 2, dtype=torch.bool),
        action_mask=torch.ones(3, 5, dtype=torch.bool),
        current_event_id=torch.tensor([0, 1, 2]),
        clock_event_id=torch.tensor([0, 1, 2]),
        dt=torch.ones(3),
    )
    loss = output["success_logit"].sum() + output["next_event_logits"].sum()
    optimizer.zero_grad()
    loss.backward()
    embedding = core.action_encoder.policy_embedding.weight
    assert torch.count_nonzero(embedding.grad[0]) == 0
    assert torch.count_nonzero(embedding.grad[1]) > 0
    optimizer.step()
    wrapper.enforce_frozen_core()
    after = core.state_dict()
    assert torch.equal(after["action_encoder.policy_embedding.weight"][0], before["action_encoder.policy_embedding.weight"][0])
    assert not torch.equal(after["action_encoder.policy_embedding.weight"][1], before["action_encoder.policy_embedding.weight"][1])
    for name in before:
        if name != "action_encoder.policy_embedding.weight":
            assert torch.equal(after[name], before[name]), name
    assert len(wrapper.assert_shared_core_immutable()) == 64
    payload = wrapper.adapter_payload(source_core_sha256="a" * 64)
    assert payload["authorization"] == "monitor_only_until_independent_transfer_confirmation"
    assert payload["effective_trainable_parameters"] == wrapper.effective_trainable_parameter_count()


def test_deployment_adapters_preserve_native_execution_actions() -> None:
    core = _core()
    wrapper = FrozenCoreTransferModel(
        core,
        TransferAdapterSpec(
            axis="policy",
            target_state_dim=6,
            core_state_dim=8,
            native_action_dim=4,
            core_action_dim=4,
            target_body_id=0,
            target_policy_id=1,
            target_embedding_row=1,
            learn_state_adapter=True,
            learn_action_adapter=True,
        ),
    )
    state_adapter = LearnedTransferStateAdapter(
        wrapper.state_adapter,
        policy_name="smolvla",
        calibration_id="state-contract-id",
        authorized=False,
    )
    state = state_adapter.adapt(
        torch.randn(2, 3, 6), current_event_id=torch.tensor([0, 1])
    )
    assert state.hidden.shape == (2, 3, 8)
    assert state.adapter_calibrated is False
    action_adapter = LearnedTransferActionAdapter(
        wrapper.action_adapter,  # type: ignore[arg-type]
        policy_name="smolvla",
        policy_id=1,
        calibration_id="action-contract-id",
        authorized=False,
    )
    native = torch.randn(2, 4, 5, 4)
    candidates = action_adapter.adapt(
        native,
        EmbodimentSpec(name="piper", body_id=0),
        fallback_index=torch.tensor([0, 2]),
    )
    assert candidates.actions.shape == native.shape
    assert torch.equal(candidates.execution_actions, native)
    assert candidates.adapter_calibrated is False
    assert torch.equal(candidates.resolved_fallback_index(), torch.tensor([0, 2]))


def test_embodiment_transfer_requires_and_trains_clock() -> None:
    core = _core(policies=1, bodies=2)
    wrapper = FrozenCoreTransferModel(
        core,
        TransferAdapterSpec(
            axis="embodiment",
            target_state_dim=8,
            core_state_dim=8,
            native_action_dim=4,
            core_action_dim=4,
            target_body_id=1,
            target_policy_id=0,
            target_embedding_row=1,
            learn_state_adapter=False,
            learn_action_adapter=True,
            learn_clock=True,
        ),
    )
    assert wrapper.clock_beta is not None
    assert wrapper.clock_log_step_scale is not None
    output = wrapper(
        torch.randn(2, 8),
        torch.randn(2, 5, 4),
        current_event_id=torch.tensor([0, 1]),
        dt=torch.ones(2),
    )
    output["duration_selected_log_mean"].sum().backward()
    assert wrapper.clock_beta.grad is not None
    assert wrapper.clock_log_step_scale.grad is not None


def test_transfer_spec_rejects_conflated_clock_and_dimension_shortcuts() -> None:
    with pytest.raises(ValueError, match="fixed-body clock"):
        TransferAdapterSpec(
            axis="policy",
            target_state_dim=8,
            core_state_dim=8,
            native_action_dim=4,
            core_action_dim=4,
            target_body_id=0,
            target_policy_id=1,
            target_embedding_row=1,
            learn_clock=True,
        )
    with pytest.raises(ValueError, match="identity state"):
        TransferAdapterSpec(
            axis="policy",
            target_state_dim=6,
            core_state_dim=8,
            native_action_dim=4,
            core_action_dim=4,
            target_body_id=0,
            target_policy_id=1,
            target_embedding_row=1,
            learn_state_adapter=False,
        )
