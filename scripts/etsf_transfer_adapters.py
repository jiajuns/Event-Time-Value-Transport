#!/usr/bin/env python3
"""Small target-domain adapters around a frozen ETSF event world model.

The shared core remains an ordinary ``ActionConditionedEventWorldModel`` so its
post-adaptation checkpoint can be audited by ``verify_etsf_transfer_protocol``.
Only one pre-reserved policy/body embedding row receives gradients.  A
controller restores every other core tensor after each optimizer step, making
accidental optimizer weight decay or an overly broad parameter group fail safe.

This module defines model boundaries; it does not authorize a target adapter.
Deployment adapters must still carry the calibration id frozen in the final
plugin contract and pass the independent transfer confirmation gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from openvla_etsf_event_critic_plugin import (
    EmbodimentSpec,
    PolicyAdapter,
    StateAdapter,
)
from openvla_etsf_event_world_model import ActionConditionedEventWorldModel


FORMAT = "etsf_frozen_core_transfer_adapter_v1"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _immutable_digest(
    state: Mapping[str, torch.Tensor], *, embedding_name: str, target_row: int
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(
            json.dumps(
                [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
            ).encode()
        )
        if name != embedding_name:
            digest.update(_tensor_bytes(tensor))
            continue
        if target_row:
            digest.update(_tensor_bytes(tensor[:target_row]))
        if target_row + 1 < tensor.shape[0]:
            digest.update(_tensor_bytes(tensor[target_row + 1 :]))
    return digest.hexdigest()


@dataclass(frozen=True)
class TransferAdapterSpec:
    axis: str
    target_state_dim: int
    core_state_dim: int
    native_action_dim: int
    core_action_dim: int
    target_body_id: int
    target_policy_id: int
    target_embedding_row: int
    state_bottleneck_dim: int = 128
    learn_state_adapter: bool = True
    learn_action_adapter: bool = True
    learn_clock: bool = False
    base_beta: float = 0.0

    def __post_init__(self) -> None:
        if self.axis not in ("policy", "embodiment"):
            raise ValueError("axis must be policy or embodiment")
        dimensions = (
            self.target_state_dim,
            self.core_state_dim,
            self.native_action_dim,
            self.core_action_dim,
            self.state_bottleneck_dim,
        )
        if min(dimensions) < 1:
            raise ValueError("adapter dimensions must be positive")
        if min(self.target_body_id, self.target_policy_id, self.target_embedding_row) < 0:
            raise ValueError("target ids/row must be non-negative")
        if not self.learn_state_adapter and self.target_state_dim != self.core_state_dim:
            raise ValueError("identity state transfer requires equal dimensions")
        if not self.learn_action_adapter and self.native_action_dim != self.core_action_dim:
            raise ValueError("identity action transfer requires equal dimensions")
        if self.axis == "policy" and self.learn_clock:
            raise ValueError("policy transfer must keep the fixed-body clock frozen")
        if self.axis == "embodiment" and not self.learn_clock:
            raise ValueError("embodiment transfer requires clock adaptation")


class LowRankStateProjector(nn.Module):
    """Map target-policy hidden states into the frozen core input space."""

    def __init__(self, input_dim: int, output_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim not in (2, 3) or hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"target hidden must be [B,D]/[B,T,D] with D={self.input_dim}"
            )
        return self.network(hidden)


class AffineActionEffectProjector(nn.Module):
    """Calibrated native-action to frozen-core action-effect coordinates."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        weight = torch.zeros(output_dim, input_dim)
        diagonal = min(input_dim, output_dim)
        weight[:diagonal, :diagonal] = torch.eye(diagonal)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.log_distance_scale = nn.Parameter(torch.zeros(output_dim))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim < 2 or actions.shape[-1] != self.input_dim:
            raise ValueError(f"native actions must end in {self.input_dim} features")
        return F.linear(actions, self.weight, self.bias)

    def distance_scale(self) -> torch.Tensor:
        return F.softplus(self.log_distance_scale) + 1e-4


class FrozenCoreTransferModel(nn.Module):
    """Frozen event core plus target state/action/clock adapters."""

    def __init__(
        self,
        core: ActionConditionedEventWorldModel,
        spec: TransferAdapterSpec,
    ) -> None:
        super().__init__()
        if core.config.state_input_dim != spec.core_state_dim:
            raise ValueError("core state dimension differs from transfer spec")
        if core.config.action_dim != spec.core_action_dim:
            raise ValueError("core action dimension differs from transfer spec")
        if spec.target_body_id >= core.config.num_bodies:
            raise ValueError("target body id is outside the pre-reserved vocabulary")
        if spec.target_policy_id >= core.config.num_policies:
            raise ValueError("target policy id is outside the pre-reserved vocabulary")
        self.core = core
        self.spec = spec
        self.state_adapter: nn.Module = (
            LowRankStateProjector(
                spec.target_state_dim,
                spec.core_state_dim,
                spec.state_bottleneck_dim,
            )
            if spec.learn_state_adapter
            else nn.Identity()
        )
        self.action_adapter: AffineActionEffectProjector | nn.Identity = (
            AffineActionEffectProjector(
                spec.native_action_dim, spec.core_action_dim
            )
            if spec.learn_action_adapter
            else nn.Identity()
        )
        self.clock_beta = (
            nn.Parameter(torch.tensor(float(spec.base_beta)))
            if spec.learn_clock
            else None
        )
        self.clock_log_step_scale = (
            nn.Parameter(torch.zeros(())) if spec.learn_clock else None
        )
        self.embedding_name = (
            "action_encoder.policy_embedding.weight"
            if spec.axis == "policy"
            else "action_encoder.body_embedding.weight"
        )
        embedding_module = (
            core.action_encoder.policy_embedding
            if spec.axis == "policy"
            else core.action_encoder.body_embedding
        )
        if spec.target_embedding_row >= embedding_module.weight.shape[0]:
            raise ValueError("target embedding row is outside the reserved matrix")
        if (
            spec.axis == "policy"
            and spec.target_embedding_row != spec.target_policy_id
        ) or (
            spec.axis == "embodiment"
            and spec.target_embedding_row != spec.target_body_id
        ):
            raise ValueError("target embedding row must equal the registered target id")
        for parameter in core.parameters():
            parameter.requires_grad_(False)
        embedding_module.weight.requires_grad_(True)
        mask = torch.zeros_like(embedding_module.weight)
        mask[spec.target_embedding_row] = 1
        self.register_buffer("_embedding_gradient_mask", mask, persistent=False)
        embedding_module.weight.register_hook(
            lambda gradient: gradient * self._embedding_gradient_mask.to(gradient)
        )
        # CPU snapshots make restoration independent of optimizer implementation.
        self._frozen_core_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in core.state_dict().items()
        }
        self._immutable_before = _immutable_digest(
            self._frozen_core_state,
            embedding_name=self.embedding_name,
            target_row=spec.target_embedding_row,
        )

    def adapt_state(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.state_adapter(hidden)

    def adapt_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_adapter(actions)

    def forward(
        self,
        hidden_t: torch.Tensor,
        action_chunks: torch.Tensor,
        *,
        history_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_feature_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        current_event_id: torch.Tensor | None = None,
        clock_event_id: torch.Tensor | None = None,
        current_predicates: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.adapt_state(hidden_t)
        actions = self.adapt_actions(action_chunks)
        batch = hidden.shape[0]
        device = hidden.device
        body_id = torch.full(
            (batch,), self.spec.target_body_id, device=device, dtype=torch.long
        )
        policy_id = torch.full(
            (batch,), self.spec.target_policy_id, device=device, dtype=torch.long
        )
        beta_value = (
            self.clock_beta
            if self.clock_beta is not None
            else hidden.new_tensor(self.spec.base_beta)
        )
        beta = beta_value.to(device=device, dtype=hidden.dtype).expand(batch)
        adjusted_dt = dt
        if dt is not None and self.clock_log_step_scale is not None:
            adjusted_dt = dt * self.clock_log_step_scale.exp().to(dt)
        if action_feature_mask is not None and actions.shape[-1] != action_feature_mask.shape[-1]:
            # A learned dense projection makes every output coordinate defined.
            action_feature_mask = torch.ones_like(actions, dtype=torch.bool)
        return self.core(
            hidden,
            actions,
            history_mask=history_mask,
            action_mask=action_mask,
            action_feature_mask=action_feature_mask,
            proprio=proprio,
            body_id=body_id,
            policy_id=policy_id,
            current_event_id=current_event_id,
            clock_event_id=clock_event_id,
            current_predicates=current_predicates,
            beta=beta,
            dt=adjusted_dt,
        )

    @torch.no_grad()
    def enforce_frozen_core(self) -> None:
        """Restore every core value except the one authorized target row."""

        current = self.core.state_dict()
        target_value = current[self.embedding_name][
            self.spec.target_embedding_row
        ].detach().clone()
        for name, frozen in self._frozen_core_state.items():
            current[name].copy_(frozen.to(device=current[name].device, dtype=current[name].dtype))
        current[self.embedding_name][self.spec.target_embedding_row].copy_(target_value)

    def assert_shared_core_immutable(self) -> str:
        current = self.core.state_dict()
        digest = _immutable_digest(
            current,
            embedding_name=self.embedding_name,
            target_row=self.spec.target_embedding_row,
        )
        if digest != self._immutable_before:
            raise RuntimeError("shared core changed outside the target embedding row")
        return digest

    def effective_trainable_parameter_count(self) -> int:
        external = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("core.")
        )
        embedding = dict(self.core.named_parameters())[self.embedding_name]
        return external + int(embedding.shape[1])

    def adapter_payload(self, *, source_core_sha256: str) -> dict[str, Any]:
        immutable = self.assert_shared_core_immutable()
        return {
            "format": FORMAT,
            "spec": dataclasses.asdict(self.spec),
            "source_core_sha256": source_core_sha256,
            "immutable_shared_core_sha256": immutable,
            "state_adapter": self.state_adapter.state_dict(),
            "action_adapter": self.action_adapter.state_dict(),
            "clock_beta": None
            if self.clock_beta is None
            else self.clock_beta.detach().cpu(),
            "clock_log_step_scale": None
            if self.clock_log_step_scale is None
            else self.clock_log_step_scale.detach().cpu(),
            "target_embedding_row": self.core.state_dict()[self.embedding_name][
                self.spec.target_embedding_row
            ].detach().cpu(),
            "effective_trainable_parameters": self.effective_trainable_parameter_count(),
            "authorization": "monitor_only_until_independent_transfer_confirmation",
        }


class LearnedTransferStateAdapter(StateAdapter):
    """Deploy a trained target-state projector through the plugin boundary."""

    def __init__(
        self,
        projector: nn.Module,
        *,
        policy_name: str,
        calibration_id: str,
        authorized: bool,
    ) -> None:
        super().__init__(
            name="LearnedTransferStateAdapter",
            policy_name=policy_name,
            calibrated=authorized,
            calibration_id=calibration_id,
        )
        self.projector = projector.eval()

    @torch.no_grad()
    def to_model_history(self, native_history: torch.Tensor) -> torch.Tensor:
        return self.projector(native_history)


class LearnedTransferActionAdapter(PolicyAdapter):
    """Deploy a trained affine target-action projector while preserving execution."""

    def __init__(
        self,
        projector: AffineActionEffectProjector,
        *,
        policy_name: str,
        policy_id: int,
        calibration_id: str,
        authorized: bool,
    ) -> None:
        super().__init__(
            name=policy_name,
            policy_id=policy_id,
            calibrated=authorized,
            calibration_id=calibration_id,
        )
        self.projector = projector.eval()

    @torch.no_grad()
    def to_model_actions(
        self, native_actions: torch.Tensor, embodiment: EmbodimentSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del embodiment
        actions = self.projector(native_actions)
        return actions, torch.ones_like(actions, dtype=torch.bool)

    def action_distance_scale(
        self, canonical_actions: torch.Tensor, embodiment: EmbodimentSpec
    ) -> torch.Tensor:
        del embodiment
        return self.projector.distance_scale().to(canonical_actions)


__all__ = [
    "AffineActionEffectProjector",
    "FORMAT",
    "FrozenCoreTransferModel",
    "LearnedTransferActionAdapter",
    "LearnedTransferStateAdapter",
    "LowRankStateProjector",
    "TransferAdapterSpec",
]
