#!/usr/bin/env python3
"""Continuous-time CfC building blocks for the v14 shared event head.

The public ``ncps.torch.CfC`` sequence wrapper has had releases where a
batched ``timespans`` tensor broadcasts incorrectly inside ``CfCCell``.  This
module intentionally uses the official cell directly and supplies one
``[batch, 1]`` physical-time span per step.  Padding is masked outside the
cell, so a finite history can be passed on every critic call without carrying
robot-specific hidden state between episodes.
"""

from __future__ import annotations

import importlib.metadata
import math
from typing import Final

import torch
from ncps.torch import CfCCell
from torch import nn

import train_multibody_canonical_event_world_model as core


FORMAT: Final = "etsf_masked_ncps_cfc_cells_v1"
NCPS_DISTRIBUTION_VERSION: Final = importlib.metadata.version("ncps")
LIQUID_STATE_DIM: Final = 72
ACTOR_CONTROL_HZ: Final = 15.0


class LiquidContractError(ValueError):
    """A continuous-time input violated the deployment data contract."""


def _validate_sequence(
    values: torch.Tensor,
    mask: torch.Tensor,
    timespans: torch.Tensor,
    *,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or values.shape[-1] != input_size:
        raise LiquidContractError(
            f"CfC values must be [B,T,{input_size}], got {tuple(values.shape)}"
        )
    expected = values.shape[:2]
    if mask.shape != expected or timespans.shape != expected:
        raise LiquidContractError("CfC mask/timespan shape mismatch")
    mask = mask.bool()
    timespans = timespans.to(device=values.device, dtype=values.dtype)
    if not bool(torch.isfinite(values).all()):
        raise LiquidContractError("CfC values contain non-finite inputs")
    if not bool(torch.isfinite(timespans).all()) or bool((timespans < 0.0).any()):
        raise LiquidContractError("CfC timespans must be finite and non-negative")
    if bool((~mask.any(dim=1)).any()):
        raise LiquidContractError("every CfC row requires at least one valid step")
    # A padded slot is semantically absent and therefore has exactly zero
    # elapsed time.  This catches stale timestamps leaking through padding.
    if bool((timespans.masked_select(~mask) != 0.0).any()):
        raise LiquidContractError("padded CfC steps must use zero timespan")
    return mask, timespans


class MaskedCfCSequence(nn.Module):
    """Stateless finite-history wrapper around the official ``CfCCell``."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        mode: str = "default",
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise LiquidContractError("CfC dimensions must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cell = CfCCell(
            self.input_size,
            self.hidden_size,
            mode=mode,
            backbone_layers=0,
            backbone_units=0,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        timespans: torch.Tensor,
        initial_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask, timespans = _validate_sequence(
            values, mask, timespans, input_size=self.input_size
        )
        if initial_hidden is None:
            hidden = values.new_zeros(values.shape[0], self.hidden_size)
        else:
            if initial_hidden.shape != (values.shape[0], self.hidden_size):
                raise LiquidContractError("CfC initial hidden shape mismatch")
            if not bool(torch.isfinite(initial_hidden).all()):
                raise LiquidContractError("CfC initial hidden is non-finite")
            hidden = initial_hidden.to(values)
        for step in range(values.shape[1]):
            # Explicit [B,1] is important: it avoids the ncps sequence-wrapper
            # broadcasting failure for a genuine batched timespan tensor.
            proposal, _ = self.cell(
                values[:, step], hidden, timespans[:, step, None]
            )
            hidden = torch.where(mask[:, step, None], proposal, hidden)
        return hidden


class LiquidCanonicalSemanticEncoder(nn.Module):
    """Irregular canonical state history -> shared 96-D event geometry."""

    requires_history: Final = True

    def __init__(
        self,
        semantic_dim: int = core.SEMANTIC_DIM,
        liquid_dim: int = LIQUID_STATE_DIM,
    ) -> None:
        super().__init__()
        if semantic_dim != core.SEMANTIC_DIM:
            raise LiquidContractError("canonical semantic output must remain 96-D")
        self.input_map = nn.Sequential(
            nn.Linear(core.STATE_DIM, liquid_dim),
            nn.GELU(),
        )
        self.sequence = MaskedCfCSequence(liquid_dim, liquid_dim)
        self.output = nn.Sequential(
            nn.Linear(liquid_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        state_time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 3 or state.shape[-1] != core.STATE_DIM:
            raise LiquidContractError(
                f"liquid state history must be [B,K,{core.STATE_DIM}]"
            )
        if state_mask is None or state_time_delta is None:
            raise LiquidContractError(
                "liquid state history requires explicit mask and physical dt"
            )
        hidden = self.sequence(
            self.input_map(state), state_mask, state_time_delta
        )
        return self.output(hidden)


class LiquidCanonicalActionEncoder(nn.Module):
    """Shared continuous-time encoder for canonical candidate effects."""

    def __init__(
        self,
        schema_count: int,
        semantic_dim: int = core.SEMANTIC_DIM,
        liquid_dim: int = LIQUID_STATE_DIM,
    ) -> None:
        super().__init__()
        if schema_count <= 0:
            raise LiquidContractError("action schema count must be positive")
        self.schema_count = int(schema_count)
        self.register_buffer(
            "action_mean", torch.zeros(schema_count, core.ACTION_DIM)
        )
        self.register_buffer(
            "action_std", torch.ones(schema_count, core.ACTION_DIM)
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(core.ACTION_DIM, liquid_dim),
                    nn.GELU(),
                )
                for _ in range(schema_count)
            ]
        )
        self.sequences = nn.ModuleList(
            [MaskedCfCSequence(liquid_dim, liquid_dim) for _ in range(schema_count)]
        )
        self.outputs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(liquid_dim, semantic_dim),
                    nn.LayerNorm(semantic_dim),
                )
                for _ in range(schema_count)
            ]
        )

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (self.schema_count, core.ACTION_DIM)
        if mean.shape != expected or std.shape != expected:
            raise LiquidContractError(f"action normalization must be {expected}")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise LiquidContractError("action normalization is non-finite")
        if bool((std < 1e-4).any()):
            raise LiquidContractError("action normalization std is below its floor")
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std))

    def forward(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        action_available: torch.Tensor,
        action_schema_id: torch.Tensor,
        action_time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != core.ACTION_DIM:
            raise LiquidContractError(
                f"liquid actions must be [B,H,{core.ACTION_DIM}]"
            )
        if action_time_delta is None:
            raise LiquidContractError("liquid actions require planned_action_dt")
        if action_mask.shape != actions.shape[:2]:
            raise LiquidContractError("liquid action mask shape mismatch")
        if action_available.shape != actions.shape[:1]:
            raise LiquidContractError("liquid action availability shape mismatch")
        if action_schema_id.shape != actions.shape[:1]:
            raise LiquidContractError("liquid action schema shape mismatch")
        available = action_available.bool()
        if bool((available & (action_schema_id < 0)).any()) or bool(
            (available & (action_schema_id >= self.schema_count)).any()
        ):
            raise LiquidContractError("available liquid action schema is invalid")
        if bool(((~available) & (action_schema_id != -1)).any()):
            raise LiquidContractError("missing liquid action must use schema -1")
        if bool((available & ~action_mask.bool().any(dim=1)).any()):
            raise LiquidContractError("available liquid action has an empty prefix")
        result = actions.new_zeros(actions.shape[0], core.SEMANTIC_DIM)
        for schema in range(self.schema_count):
            selected = available & (action_schema_id == schema)
            if not bool(selected.any()):
                continue
            subset = (
                actions[selected] - self.action_mean[schema][None, None]
            ) / self.action_std[schema][None, None]
            normalization_clip = getattr(self, "normalization_clip", None)
            if normalization_clip is not None:
                clip = float(normalization_clip)
                if not math.isfinite(clip) or clip <= 0.0:
                    raise LiquidContractError("action clip must be finite/positive")
                subset = subset.clamp(-clip, clip)
            hidden = self.sequences[schema](
                self.projections[schema](subset),
                action_mask[selected],
                action_time_delta[selected],
            )
            result[selected] = self.outputs[schema](hidden)
        return result


class LiquidConsequenceClock(nn.Module):
    """Body-independent CfC clock for the planned action exposure."""

    def __init__(self, semantic_dim: int, clock_dim: int) -> None:
        super().__init__()
        self.clock_dim = int(clock_dim)
        self.input_map = nn.Sequential(
            nn.Linear(semantic_dim, clock_dim),
            nn.Tanh(),
        )
        self.sequence = MaskedCfCSequence(clock_dim, clock_dim)
        self.log_tau = nn.Linear(clock_dim, clock_dim)

    def forward(
        self,
        semantic: torch.Tensor,
        dt: torch.Tensor,
        body_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic.ndim != 2 or dt.shape != semantic.shape[:1]:
            raise LiquidContractError("liquid consequence clock input shape mismatch")
        if body_id.shape != dt.shape:
            raise LiquidContractError("liquid consequence body-id shape mismatch")
        # Body ids are accepted only for ABI compatibility.  No body embedding
        # or body-conditioned parameter exists in this module.
        detached = semantic.detach()
        values = self.input_map(detached)[:, None]
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        hidden = self.sequence(values, mask, dt[:, None])
        diagnostic_log_tau = torch.clamp(self.log_tau(hidden), -3.0, 7.0)
        return hidden, diagnostic_log_tau


class LiquidEventAgeEncoder(nn.Module):
    """Closed-form continuous event-age exposure with a pure CfC cell."""

    def __init__(self, clock_dim: int) -> None:
        super().__init__()
        self.clock_dim = int(clock_dim)
        self.cell = CfCCell(
            1,
            clock_dim,
            mode="pure",
            backbone_layers=0,
            backbone_units=0,
        )

    def forward(self, log1p_event_age: torch.Tensor) -> torch.Tensor:
        if log1p_event_age.ndim != 2 or log1p_event_age.shape[-1] != 1:
            raise LiquidContractError("liquid event age must be [B,1]")
        if not bool(torch.isfinite(log1p_event_age).all()) or bool(
            (log1p_event_age < 0.0).any()
        ):
            raise LiquidContractError("liquid event age is invalid")
        age = torch.expm1(log1p_event_age)
        hidden = log1p_event_age.new_zeros(
            log1p_event_age.shape[0], self.clock_dim
        )
        proposal, _ = self.cell(
            torch.ones_like(log1p_event_age), hidden, age
        )
        return proposal


class LiquidTerminalDynamics(nn.Module):
    """Evolve consequence state over the known remaining control horizon."""

    def __init__(
        self,
        semantic_dim: int = core.SEMANTIC_DIM,
        control_hz: float = ACTOR_CONTROL_HZ,
    ) -> None:
        super().__init__()
        if not math.isfinite(control_hz) or control_hz <= 0.0:
            raise LiquidContractError("control frequency must be finite/positive")
        self.semantic_dim = int(semantic_dim)
        self.control_hz = float(control_hz)
        self.cell = CfCCell(
            2,
            semantic_dim,
            mode="default",
            backbone_layers=0,
            backbone_units=0,
        )

    def forward(
        self,
        transitioned: torch.Tensor,
        terminal_context: torch.Tensor,
    ) -> torch.Tensor:
        if transitioned.ndim != 2 or transitioned.shape[-1] != self.semantic_dim:
            raise LiquidContractError("liquid terminal state must be [B,96]")
        if terminal_context.shape != (transitioned.shape[0], 2):
            raise LiquidContractError("liquid terminal context must be [B,2]")
        if not bool(torch.isfinite(terminal_context).all()) or bool(
            (terminal_context < 0.0).any()
        ):
            raise LiquidContractError("liquid terminal context is invalid")
        remaining_actions = torch.expm1(terminal_context[:, 1:2])
        remaining_seconds = remaining_actions / self.control_hz
        proposal, _ = self.cell(
            terminal_context, transitioned, remaining_seconds
        )
        # A bounded residual keeps the calibrated v13 consequence geometry at
        # initialization while allowing physical horizon to alter candidates.
        return transitioned + 0.25 * torch.tanh(proposal - transitioned)


def ncps_runtime_contract() -> dict[str, object]:
    return {
        "format": FORMAT,
        "distribution": "ncps",
        "distribution_version": NCPS_DISTRIBUTION_VERSION,
        "cell": "ncps.torch.CfCCell",
        "sequence_wrapper": "manual_masked_step_loop",
        "timespan_shape_per_step": "batch_by_1",
        "persistent_hidden_state_between_calls": False,
        "future_realized_duration_used_as_input": False,
    }


__all__ = [
    "ACTOR_CONTROL_HZ",
    "FORMAT",
    "LIQUID_STATE_DIM",
    "LiquidCanonicalActionEncoder",
    "LiquidCanonicalSemanticEncoder",
    "LiquidConsequenceClock",
    "LiquidContractError",
    "LiquidEventAgeEncoder",
    "LiquidTerminalDynamics",
    "MaskedCfCSequence",
    "NCPS_DISTRIBUTION_VERSION",
    "ncps_runtime_contract",
]
