#!/usr/bin/env python3
"""v14 body-independent Liquid-CfC shared event/value head."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

import etsf_liquid_cfc_v1 as liquid
import train_multibody_canonical_event_world_model as core
import train_robotwin2_five_body_lobo_shared_event_head_v1 as v13


FORMAT = "etsf_robotwin2_liquid_shared_event_head_v14_v1"
MODEL_FAMILY = "liquid_cfc_terminal_consequence_utility_shared_event_head_v14"
SOURCE_BODY = "aloha-agilex"
TARGET_BODIES = ("arx-x5", "franka", "piper", "ur5")


class LiquidEffectAlignedSharedEventHead(v13.EffectAlignedSharedEventHead):
    """v13 proper consequence heads with continuous-time state dynamics.

    The model contains no embodiment embedding, joint-index stem, or persistent
    episode state.  Every call receives a finite canonical history and can be
    inserted in front of any policy that proposes canonical EE action chunks.
    """

    model_family = MODEL_FAMILY

    def __init__(self, ablation_variant: str = "full") -> None:
        super().__init__(ablation_variant)
        self.semantic = liquid.LiquidCanonicalSemanticEncoder(
            self.config.semantic_dim
        )
        self.action = liquid.LiquidCanonicalActionEncoder(
            self.config.action_schema_count,
            self.config.semantic_dim,
        )
        self.action.normalization_clip = v13.CROSS_BODY_STANDARDIZED_INPUT_CLIP
        self.clock = liquid.LiquidConsequenceClock(
            self.config.semantic_dim, self.config.clock_dim
        )
        self.event_age_encoder = liquid.LiquidEventAgeEncoder(
            self.config.clock_dim
        )
        # These v13 modules are replaced, not retained as dead parameters.
        self.terminal_context_encoder = nn.Identity()
        self.terminal_residual = nn.Identity()
        self.liquid_terminal = liquid.LiquidTerminalDynamics(
            self.config.semantic_dim
        )

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        history = batch.get("state_history")
        history_mask = batch.get("state_history_mask")
        history_dt = batch.get("state_history_dt")
        if not isinstance(history, torch.Tensor) or history.ndim != 3:
            raise v13.FiveBodyContractError(
                "v14 requires canonical state_history [B,K,27]"
            )
        if (
            not isinstance(history_mask, torch.Tensor)
            or not isinstance(history_dt, torch.Tensor)
            or history_mask.shape != history.shape[:2]
            or history_dt.shape != history.shape[:2]
        ):
            raise v13.FiveBodyContractError(
                "v14 state history mask/dt contract is incomplete"
            )
        prepared: dict[str, Any] = dict(batch)
        prepared["state_history"] = (
            (history - self.state_mean[None, None])
            / self.state_std[None, None]
        ).clamp(
            min=-v13.CROSS_BODY_STANDARDIZED_INPUT_CLIP,
            max=v13.CROSS_BODY_STANDARDIZED_INPUT_CLIP,
        )
        return super().forward(prepared)

    def _terminal_hidden(
        self,
        transitioned: torch.Tensor,
        terminal_context: torch.Tensor,
    ) -> torch.Tensor:
        return self.liquid_terminal(transitioned, terminal_context)


def parameter_inventory(model: nn.Module) -> dict[str, Any]:
    named = list(model.named_parameters())
    trainable = [(name, value) for name, value in named if value.requires_grad]
    body_specific = [
        name
        for name, _value in trainable
        if any(token in name.lower() for token in ("body_beta", "body_embedding"))
    ]
    return {
        "total": sum(value.numel() for _name, value in named),
        "trainable": sum(value.numel() for _name, value in trainable),
        "frozen": sum(value.numel() for _name, value in named if not value.requires_grad),
        "body_specific_trainable_names": body_specific,
        "ncps": liquid.ncps_runtime_contract(),
    }


def checkpoint_contract(history_length: int) -> dict[str, Any]:
    if history_length <= 0:
        raise v13.FiveBodyContractError("liquid history length must be positive")
    return {
        "format": FORMAT,
        "model_family": MODEL_FAMILY,
        "source_body": SOURCE_BODY,
        "sealed_target_bodies": list(TARGET_BODIES),
        "state_history_length": int(history_length),
        "state_history_timing": "past_observed_simulator_seconds_only",
        "planned_action_timing": "known_actor_control_intervals_only",
        "future_realized_duration_input": False,
        "persistent_hidden_state_between_calls": False,
        "embodiment_conditioning": False,
        "policy_conditioning": False,
        "ncps": liquid.ncps_runtime_contract(),
    }


__all__ = [
    "FORMAT",
    "MODEL_FAMILY",
    "SOURCE_BODY",
    "TARGET_BODIES",
    "LiquidEffectAlignedSharedEventHead",
    "checkpoint_contract",
    "parameter_inventory",
]
