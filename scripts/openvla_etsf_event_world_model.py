#!/usr/bin/env python3
"""Reusable action-conditioned event world model for ETSF candidate ranking.

The module deliberately does not import OpenVLA or RoboTwin.  A policy adapter is
responsible for producing a state feature and a padded action chunk; this module
then predicts action effects in a shared event space.  The only input allowed to
depend on the embodiment clock calibration, ``beta``, is consumed by the clock
branch below.

Tensor conventions use a flat batch for :meth:`forward` and ``[batch,
candidates, horizon, action_dim]`` for :meth:`predict_candidates`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class EventWorldModelConfig:
    """Serializable dimensions and vocabulary for the event world model."""

    state_input_dim: int = 4096
    action_dim: int = 14
    proprio_dim: int = 14
    semantic_dim: int = 96
    action_hidden_dim: int = 96
    transition_hidden_dim: int = 128
    clock_hidden_dim: int = 48
    object_delta_dim: int = 35
    num_bodies: int = 8
    num_policies: int = 8
    metadata_dim: int = 16
    event_names: tuple[str, ...] = ("e0", "e12", "e3", "e4", "eK")
    outcome_names: tuple[str, ...] = ("failure", "success", "recovery")
    predicate_names: tuple[str, ...] = (
        "moved",
        "lifted",
        "near_goal",
        "stationary",
        "success",
    )
    relative_transition_names: tuple[str, ...] = (
        "stay",
        "advance",
        "skip",
        "regress",
    )
    # False preserves the parameter/state-dict contract of factual-pretrain v1.
    # New training runs explicitly enable the structured heads.
    structured_events: bool = False
    allow_event_regress: bool = True
    # Enable only when a dataset contains an operational, trajectory-derived
    # recovery label.  Factual rollouts provide failure/success but no recovery.
    recovery_supervised: bool = False
    # Counterfactual fine-tuning can add a baseline-relative action residual to
    # candidate success scores.  Old factual checkpoints omit both this flag
    # and the optional head, preserving their strict state-dict contract.
    action_rank_residual: bool = False
    # A frozen-core ranker is trained and deployed as success-only utility.
    # Keeping this explicit prevents fixed ordinal-event or duration weights
    # from silently changing the target optimized by the relative action head.
    action_rank_success_only: bool = False
    dropout: float = 0.1

    @property
    def num_events(self) -> int:
        return len(self.event_names)

    @property
    def num_outcomes(self) -> int:
        return len(self.outcome_names)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_names)

    @property
    def num_relative_transitions(self) -> int:
        return len(self.relative_transition_names)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EventWorldModelConfig":
        values = dict(values)
        for key in (
            "event_names",
            "outcome_names",
            "predicate_names",
            "relative_transition_names",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class SemanticEncoder(nn.Module):
    """OpenVLA-hidden encoder compatible with the existing shadow checkpoint."""

    def __init__(self, input_dim: int = 4096, semantic_dim: int = 96) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.bridge = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )
        self.cell = nn.GRUCell(semantic_dim, semantic_dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or mask.shape != inputs.shape[:2]:
            raise ValueError("semantic inputs must be [B,T,D] with mask [B,T]")
        state = inputs.new_zeros(inputs.shape[0], self.semantic_dim)
        outputs = []
        for column in range(inputs.shape[1]):
            proposal = self.cell(self.bridge(inputs[:, column]), state)
            state = torch.where(mask[:, column, None], proposal, state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class TemporalActionEncoder(nn.Module):
    """Order-sensitive full-chunk encoder with body/policy FiLM adapters."""

    def __init__(self, config: EventWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))
        self.action_projection = nn.Linear(config.action_dim, config.action_hidden_dim)
        self.body_embedding = nn.Embedding(config.num_bodies, config.metadata_dim)
        self.policy_embedding = nn.Embedding(config.num_policies, config.metadata_dim)
        self.metadata_film = nn.Linear(
            2 * config.metadata_dim, 2 * config.action_hidden_dim
        )
        self.cell = nn.GRUCell(config.action_hidden_dim, config.action_hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(2 * config.action_hidden_dim, config.semantic_dim),
            nn.GELU(),
            nn.LayerNorm(config.semantic_dim),
        )

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install train-split-only action statistics without replacing buffers."""

        if mean.shape != self.action_mean.shape or std.shape != self.action_std.shape:
            raise ValueError(
                f"normalization must have shape {(self.config.action_dim,)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("action normalization contains non-finite values")
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std).clamp_min(1e-4))

    def forward(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        action_feature_mask: torch.Tensor,
        body_id: torch.Tensor,
        policy_id: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"actions must be [B,H,{self.config.action_dim}], got {actions.shape}"
            )
        if action_mask.shape != actions.shape[:2]:
            raise ValueError("action_mask must be [B,H]")
        if action_feature_mask.shape != actions.shape:
            raise ValueError("action_feature_mask must be [B,H,A]")

        normalized = (actions - self.action_mean) / self.action_std
        normalized = normalized * action_feature_mask.to(normalized.dtype)
        projected = self.action_projection(normalized)

        metadata = torch.cat(
            [self.body_embedding(body_id), self.policy_embedding(policy_id)], dim=-1
        )
        scale, shift = self.metadata_film(metadata).chunk(2, dim=-1)
        projected = projected * (1.0 + 0.1 * torch.tanh(scale[:, None]))
        projected = projected + shift[:, None]

        state = actions.new_zeros(actions.shape[0], self.config.action_hidden_dim)
        sequence = []
        for column in range(actions.shape[1]):
            proposal = self.cell(projected[:, column], state)
            state = torch.where(action_mask[:, column, None], proposal, state)
            sequence.append(state)
        sequence_tensor = torch.stack(sequence, dim=1)
        weights = action_mask.to(actions.dtype)[..., None]
        pooled = (sequence_tensor * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.output(torch.cat([state, pooled], dim=-1))


class EmbodimentClockCell(nn.Module):
    """Low-rank liquid clock; beta is intentionally local to this class."""

    def __init__(self, semantic_dim: int, clock_dim: int) -> None:
        super().__init__()
        width = semantic_dim + clock_dim
        self.candidate = nn.Linear(width, clock_dim)
        self.base_tau = nn.Linear(width, clock_dim)
        self.beta_shape = nn.Linear(width, clock_dim)

    def forward(
        self,
        semantic: torch.Tensor,
        hidden: torch.Tensor,
        timespan: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([semantic, hidden], dim=-1)
        proposal = torch.tanh(self.candidate(joined))
        base = math.log(10.0) + 1.5 * torch.tanh(self.base_tau(joined))
        shape = torch.tanh(self.beta_shape(joined))
        shape = shape - shape.mean(dim=-1, keepdim=True)
        shape = shape / torch.sqrt(shape.square().mean(dim=-1, keepdim=True) + 1e-6)
        log_tau = torch.clamp(base + 0.5 * beta[:, None] * shape, -3.0, 7.0)
        decay = torch.exp(-timespan[:, None] / torch.exp(log_tau))
        return decay * hidden + (1.0 - decay) * proposal, log_tau


def _bernoulli_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return -(
        probabilities * F.logsigmoid(logits)
        + (1.0 - probabilities) * F.logsigmoid(-logits)
    )


def _categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


class ActionConditionedEventWorldModel(nn.Module):
    """Pluggable event/time/action-effect model with distributional heads."""

    def __init__(self, config: EventWorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or EventWorldModelConfig()
        cfg = self.config
        if cfg.action_rank_success_only and not cfg.action_rank_residual:
            raise ValueError(
                "action_rank_success_only requires action_rank_residual=True"
            )
        if min(
            cfg.state_input_dim,
            cfg.action_dim,
            cfg.semantic_dim,
            cfg.action_hidden_dim,
            cfg.transition_hidden_dim,
            cfg.clock_hidden_dim,
            cfg.num_events,
            cfg.num_outcomes,
            cfg.num_bodies,
            cfg.num_policies,
        ) <= 0:
            raise ValueError("model dimensions and vocabulary sizes must be positive")
        if cfg.num_outcomes < 2:
            raise ValueError("outcome vocabulary must contain failure and success")
        if cfg.structured_events:
            if cfg.num_predicates <= 0:
                raise ValueError("structured event modeling requires predicates")
            expected_relative = ("stay", "advance", "skip", "regress")
            if cfg.relative_transition_names != expected_relative:
                raise ValueError(
                    "relative transition vocabulary must be " + repr(expected_relative)
                )

        self.semantic = SemanticEncoder(cfg.state_input_dim, cfg.semantic_dim)
        self.action_encoder = TemporalActionEncoder(cfg)
        self.event_embedding = nn.Embedding(cfg.num_events, cfg.metadata_dim)
        self.proprio_encoder = (
            nn.Sequential(
                nn.LayerNorm(cfg.proprio_dim),
                nn.Linear(cfg.proprio_dim, cfg.semantic_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.semantic_dim),
            )
            if cfg.proprio_dim > 0
            else None
        )
        self.predicate_encoder: nn.Module | None = None
        if cfg.structured_events:
            self.predicate_encoder = nn.Sequential(
                nn.Linear(cfg.num_predicates, cfg.semantic_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.semantic_dim),
            )
        transition_input_dim = 4 * cfg.semantic_dim + cfg.metadata_dim
        if cfg.structured_events:
            transition_input_dim += cfg.semantic_dim
        self.transition = nn.Sequential(
            nn.Linear(transition_input_dim, cfg.transition_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.transition_hidden_dim, cfg.semantic_dim),
            nn.LayerNorm(cfg.semantic_dim),
        )

        self.next_event_head = nn.Linear(cfg.semantic_dim, cfg.num_events)
        self.relative_transition_head: nn.Module | None = None
        self.post_predicate_head: nn.Module | None = None
        if cfg.structured_events:
            self.relative_transition_head = nn.Linear(
                cfg.semantic_dim, cfg.num_relative_transitions
            )
            self.post_predicate_head = nn.Linear(
                cfg.semantic_dim, cfg.num_predicates
            )
        self.reach_head = nn.Linear(cfg.semantic_dim, cfg.num_events)
        self.reach_any_head = nn.Linear(cfg.semantic_dim, 1)
        self.success_head = nn.Linear(cfg.semantic_dim, 1)
        self.action_rank_head: nn.Module | None = None
        if cfg.action_rank_residual:
            if cfg.action_rank_success_only:
                # A diagonal bilinear ranker is deliberately low-capacity: for
                # deployed semantic_dim=96 it has exactly 192 parameters.  The
                # second half is the element-wise state/action interaction, so
                # no hidden MLP is needed for state-conditional ranking.
                self.action_rank_head = nn.Sequential(
                    nn.Linear(2 * cfg.semantic_dim, 1, bias=False),
                )
            else:
                # Preserve the rejected v5 checkpoint architecture for strict
                # compatibility.  New frozen-core runs must opt into the
                # success-only flag and therefore use the linear branch above.
                self.action_rank_head = nn.Sequential(
                    nn.LayerNorm(2 * cfg.semantic_dim),
                    nn.Linear(2 * cfg.semantic_dim, cfg.semantic_dim),
                    nn.GELU(),
                    nn.Linear(cfg.semantic_dim, 1, bias=False),
                )
            # Fine-tuning starts from the factual model's exact score.  The
            # residual only appears after receiving within-group supervision.
            nn.init.zeros_(self.action_rank_head[-1].weight)
        self.outcome_head = nn.Linear(cfg.semantic_dim, cfg.num_outcomes)
        self.object_delta_mean_head = nn.Linear(cfg.semantic_dim, cfg.object_delta_dim)
        self.object_delta_scale_head = nn.Linear(cfg.semantic_dim, cfg.object_delta_dim)
        self.future_latent_mean_head = nn.Linear(cfg.semantic_dim, cfg.semantic_dim)
        self.future_latent_scale_head = nn.Linear(cfg.semantic_dim, cfg.semantic_dim)

        self.clock_cell = EmbodimentClockCell(cfg.semantic_dim, cfg.clock_hidden_dim)
        self.duration_mean_head = nn.Linear(cfg.clock_hidden_dim, cfg.num_events)
        self.duration_scale_head = nn.Linear(cfg.clock_hidden_dim, cfg.num_events)

    def config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    def checkpoint_payload(self, **metadata: Any) -> dict[str, Any]:
        """Return a conventional checkpoint payload for a training script."""

        return {"model": self.state_dict(), "config": self.config_dict(), **metadata}

    def load_shadow_semantic(self, state: Mapping[str, Any]) -> None:
        """Load a shadow payload, bare state, or ``semantic.``-prefixed state."""

        if "model" in state:
            candidate = state["model"]
            if not isinstance(candidate, Mapping):
                raise ValueError("checkpoint['model'] must be a state mapping")
            state = candidate

        stripped = {
            key.removeprefix("semantic."): value
            for key, value in state.items()
            if key.startswith("semantic.") or key in self.semantic.state_dict()
        }
        expected = set(self.semantic.state_dict())
        if set(stripped) != expected:
            missing = sorted(expected - set(stripped))
            extra = sorted(set(stripped) - expected)
            raise ValueError(f"incompatible semantic state: missing={missing}, extra={extra}")
        self.semantic.load_state_dict(stripped)

    def relative_action_rank_logit(
        self,
        semantic: torch.Tensor,
        action_effect: torch.Tensor,
        baseline_action_effect: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a candidate residual anchored exactly at the baseline.

        This branch consumes only the action-effect difference and its
        interaction with the shared state.  Absolute scene difficulty remains
        in ``success_head(transition)`` and cannot by itself change this score.
        """

        if not (
            semantic.shape == action_effect.shape == baseline_action_effect.shape
        ):
            raise ValueError("relative action-rank tensors must have aligned shapes")
        if semantic.shape[-1] != self.config.semantic_dim:
            raise ValueError("relative action-rank semantic dimension mismatch")
        if self.action_rank_head is None:
            return semantic.new_zeros(semantic.shape[:-1])
        action_delta = action_effect - baseline_action_effect
        features = torch.cat([action_delta, semantic * action_delta], dim=-1)
        raw = self.action_rank_head(features).squeeze(-1)
        # Subtracting the zero-delta response keeps candidate zero an exact
        # fallback anchor even after the MLP biases have trained.
        anchor = self.action_rank_head(torch.zeros_like(features)).squeeze(-1)
        return raw - anchor

    def freeze_semantic(self) -> None:
        for parameter in self.semantic.parameters():
            parameter.requires_grad_(False)

    def encode_state(
        self,
        hidden_t: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode OpenVLA hidden history and return its final valid GRU state.

        A single ``[B,D]`` hidden is accepted for compatibility with the old
        action-Q data.  ``[B,T,D]`` plus ``history_mask`` is the preferred form:
        the shadow encoder was trained with episode history and resetting its
        GRU on every query changes the representation contract.
        """

        if hidden_t.ndim == 2:
            hidden_t = hidden_t[:, None]
            if history_mask is not None and history_mask.shape != hidden_t.shape[:2]:
                raise ValueError("single-state history_mask must have shape [B,1]")
        elif hidden_t.ndim != 3:
            raise ValueError("hidden_t must be [B,D] or [B,T,D]")
        if hidden_t.shape[-1] != self.config.state_input_dim:
            raise ValueError(
                f"hidden_t last dimension must be {self.config.state_input_dim}"
            )
        if history_mask is None:
            history_mask = torch.ones(
                hidden_t.shape[:2], dtype=torch.bool, device=hidden_t.device
            )
        else:
            history_mask = history_mask.to(device=hidden_t.device, dtype=torch.bool)
        if history_mask.shape != hidden_t.shape[:2]:
            raise ValueError("history_mask must match hidden_t [B,T]")
        if bool((~history_mask.any(dim=1)).any()):
            raise ValueError("each hidden history must contain at least one valid state")
        # SemanticEncoder holds its state on masked columns, so the last output
        # is the final valid state for both right-padded and sparse histories.
        return self.semantic(hidden_t, history_mask)[:, -1]

    def _validate_ids(self, ids: torch.Tensor, upper: int, name: str) -> None:
        if ids.ndim != 1:
            raise ValueError(f"{name} must be [B]")
        if bool(((ids < 0) | (ids >= upper)).any()):
            raise ValueError(f"{name} must be in [0, {upper})")

    @staticmethod
    def relative_transition_targets(
        current_event_id: torch.Tensor, next_event_id: torch.Tensor
    ) -> torch.Tensor:
        """Map ordered absolute events to stay/advance/skip/regress targets."""

        if current_event_id.shape != next_event_id.shape:
            raise ValueError("current and next event ids must have identical shape")
        difference = next_event_id - current_event_id
        return torch.where(
            difference < 0,
            torch.full_like(difference, 3),
            torch.where(
                difference == 0,
                torch.zeros_like(difference),
                torch.where(
                    difference == 1,
                    torch.ones_like(difference),
                    torch.full_like(difference, 2),
                ),
            ),
        )

    def _structured_event_log_probs(
        self,
        destination_logits: torch.Tensor,
        relative_logits: torch.Tensor,
        current_event_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a normalized absolute event distribution.

        Every destination belongs to exactly one ordinal transition category.
        The relative head selects a category and the old absolute head only
        resolves destinations inside that category.  Empty or disallowed
        categories receive exactly zero probability, enforcing reachability
        without giving up an absolute-event API.
        """

        if destination_logits.ndim != 2 or destination_logits.shape[-1] != self.config.num_events:
            raise ValueError("destination_logits must be [B,num_events]")
        if relative_logits.shape != (
            destination_logits.shape[0],
            self.config.num_relative_transitions,
        ):
            raise ValueError("relative_logits shape does not match the configured vocabulary")
        event_ids = torch.arange(
            self.config.num_events,
            device=destination_logits.device,
            dtype=current_event_id.dtype,
        )[None]
        current = current_event_id[:, None]
        difference = event_ids - current
        destination_category = torch.where(
            difference < 0,
            torch.full_like(difference, 3),
            torch.where(
                difference == 0,
                torch.zeros_like(difference),
                torch.where(
                    difference == 1,
                    torch.ones_like(difference),
                    torch.full_like(difference, 2),
                ),
            ),
        )
        category_has_destination = torch.stack(
            [(destination_category == category).any(-1) for category in range(4)],
            dim=-1,
        )
        if not self.config.allow_event_regress:
            category_has_destination[:, 3] = False
        negative = torch.finfo(destination_logits.dtype).min
        masked_relative = relative_logits.masked_fill(~category_has_destination, negative)
        relative_log_probability = F.log_softmax(masked_relative, dim=-1)

        absolute_parts = []
        for category in range(4):
            destination_mask = destination_category == category
            if category == 3 and not self.config.allow_event_regress:
                destination_mask = torch.zeros_like(destination_mask)
            conditional = F.log_softmax(
                destination_logits.masked_fill(~destination_mask, negative), dim=-1
            )
            absolute_parts.append(
                relative_log_probability[:, category, None] + conditional
            )
        stacked = torch.stack(absolute_parts, dim=1)
        selector = F.one_hot(
            destination_category, num_classes=4
        ).permute(0, 2, 1).bool()
        absolute_log_probability = stacked.masked_fill(~selector, negative).amax(dim=1)
        return absolute_log_probability, masked_relative

    def forward(
        self,
        hidden_t: torch.Tensor,
        action_chunks: torch.Tensor,
        *,
        history_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_feature_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        body_id: torch.Tensor | None = None,
        policy_id: torch.Tensor | None = None,
        current_event_id: torch.Tensor | None = None,
        clock_event_id: torch.Tensor | None = None,
        current_predicates: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict one transition for every state/action-chunk pair in a batch."""

        if hidden_t.ndim not in (2, 3) or hidden_t.shape[-1] != self.config.state_input_dim:
            raise ValueError(
                "hidden_t must be [B,D] or [B,T,D], "
                f"with D={self.config.state_input_dim}; got {hidden_t.shape}"
            )
        if action_chunks.ndim != 3 or action_chunks.shape[0] != hidden_t.shape[0]:
            raise ValueError("action_chunks must be [B,H,A] and share hidden_t batch")
        batch, horizon, action_dim = action_chunks.shape
        if action_dim != self.config.action_dim or horizon < 1:
            raise ValueError(
                f"action_chunks must have non-empty H and A={self.config.action_dim}"
            )
        device = hidden_t.device
        if action_chunks.device != device:
            raise ValueError("hidden_t and action_chunks must be on the same device")

        action_mask = (
            torch.ones((batch, horizon), dtype=torch.bool, device=device)
            if action_mask is None
            else action_mask.to(device=device, dtype=torch.bool)
        )
        if action_feature_mask is None:
            action_feature_mask = torch.ones_like(action_chunks, dtype=torch.bool)
        elif action_feature_mask.ndim == 2 and action_feature_mask.shape == (
            batch,
            action_dim,
        ):
            action_feature_mask = action_feature_mask[:, None].expand(-1, horizon, -1)
        action_feature_mask = action_feature_mask.to(device=device, dtype=torch.bool)
        body_id = (
            torch.zeros(batch, dtype=torch.long, device=device)
            if body_id is None
            else body_id.to(device=device, dtype=torch.long)
        )
        policy_id = (
            torch.zeros(batch, dtype=torch.long, device=device)
            if policy_id is None
            else policy_id.to(device=device, dtype=torch.long)
        )
        if self.config.structured_events and current_event_id is None:
            raise ValueError(
                "structured_events=True requires an explicit current_event_id; "
                "event zero is not a valid implicit online default"
            )
        current_event_id = (
            torch.zeros(batch, dtype=torch.long, device=device)
            if current_event_id is None
            else current_event_id.to(device=device, dtype=torch.long)
        )
        self._validate_ids(body_id, self.config.num_bodies, "body_id")
        self._validate_ids(policy_id, self.config.num_policies, "policy_id")
        self._validate_ids(current_event_id, self.config.num_events, "current_event_id")
        clock_event_id = (
            current_event_id
            if clock_event_id is None
            else clock_event_id.to(device=device, dtype=torch.long)
        )
        self._validate_ids(clock_event_id, self.config.num_events, "clock_event_id")

        beta = (
            hidden_t.new_zeros(batch)
            if beta is None
            else beta.to(device=device, dtype=hidden_t.dtype).reshape(-1)
        )
        if beta.shape != (batch,):
            raise ValueError("beta must be scalar per batch item")
        if dt is None:
            dt = action_mask.sum(dim=1).to(hidden_t.dtype).clamp_min(1.0)
        else:
            dt = dt.to(device=device, dtype=hidden_t.dtype).reshape(-1)
        if dt.shape != (batch,) or bool((dt <= 0).any()):
            raise ValueError("dt must be a positive scalar per batch item")

        semantic = self.encode_state(hidden_t, history_mask)
        action_effect = self.action_encoder(
            action_chunks,
            action_mask,
            action_feature_mask,
            body_id,
            policy_id,
        )
        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError("proprio was provided but config.proprio_dim is zero")
            proprio_effect = torch.zeros_like(semantic)
        elif proprio is None:
            proprio_effect = torch.zeros_like(semantic)
        else:
            if proprio.shape != (batch, self.config.proprio_dim):
                raise ValueError(
                    f"proprio must be [B,{self.config.proprio_dim}]"
                )
            proprio_effect = self.proprio_encoder(
                proprio.to(device=device, dtype=hidden_t.dtype)
            )

        joined_parts = [
            semantic,
            action_effect,
            semantic * action_effect,
            proprio_effect,
            self.event_embedding(current_event_id),
        ]
        normalized_predicates: torch.Tensor | None = None
        if self.config.structured_events:
            if current_predicates is None:
                raise ValueError(
                    "structured_events=True requires explicit current_predicates; "
                    "silent all-zero predicates are forbidden"
                )
            if current_predicates.shape != (batch, self.config.num_predicates):
                raise ValueError(
                    "current_predicates must be "
                    f"[B,{self.config.num_predicates}]"
                )
            normalized_predicates = current_predicates.to(
                device=device, dtype=hidden_t.dtype
            )
            if not bool(torch.isfinite(normalized_predicates).all()):
                raise ValueError("current_predicates contain non-finite values")
            if bool(
                ((normalized_predicates < 0) | (normalized_predicates > 1)).any()
            ):
                raise ValueError("current_predicates must lie in [0,1]")
            assert self.predicate_encoder is not None
            joined_parts.append(self.predicate_encoder(normalized_predicates))
        elif current_predicates is not None:
            raise ValueError(
                "current_predicates require config.structured_events=True"
            )
        joined = torch.cat(joined_parts, dim=-1)
        transition = self.transition(joined)

        destination_event_logits = self.next_event_head(transition)
        relative_transition_logits: torch.Tensor | None = None
        post_predicate_logits: torch.Tensor | None = None
        if self.config.structured_events:
            assert self.relative_transition_head is not None
            assert self.post_predicate_head is not None
            relative_transition_logits = self.relative_transition_head(transition)
            next_event_logits, relative_transition_logits = (
                self._structured_event_log_probs(
                    destination_event_logits,
                    relative_transition_logits,
                    current_event_id,
                )
            )
            post_predicate_logits = self.post_predicate_head(transition)
        else:
            next_event_logits = destination_event_logits
        reach_logits = self.reach_head(transition)
        reach_logit = self.reach_any_head(transition).squeeze(-1)
        success_logit = self.success_head(transition).squeeze(-1)
        outcome_logits = self.outcome_head(transition)
        object_delta_mean = self.object_delta_mean_head(transition)
        object_delta_log_scale = torch.clamp(
            self.object_delta_scale_head(transition), -5.0, 2.0
        )
        future_latent_mean = self.future_latent_mean_head(transition)
        future_latent_log_scale = torch.clamp(
            self.future_latent_scale_head(transition), -5.0, 2.0
        )

        # This stop-gradient is the semantic/clock module boundary.  Duration
        # supervision cannot rewrite the shared event geometry, and beta never
        # participates in any computation above this point.
        clock_hidden = hidden_t.new_zeros(batch, self.config.clock_hidden_dim)
        clock_hidden, clock_log_tau = self.clock_cell(
            transition.detach(), clock_hidden, dt, beta
        )
        duration_log_mean = self.duration_mean_head(clock_hidden)
        duration_log_scale = torch.clamp(
            self.duration_scale_head(clock_hidden), -5.0, 2.0
        )
        duration_selected_log_mean = duration_log_mean.gather(
            -1, clock_event_id[:, None]
        ).squeeze(-1)
        duration_selected_log_scale = duration_log_scale.gather(
            -1, clock_event_id[:, None]
        ).squeeze(-1)

        event_entropy = _categorical_entropy(next_event_logits) / max(
            math.log(self.config.num_events), 1e-6
        )
        reach_entropy = _bernoulli_entropy_from_logits(reach_logit) / math.log(2.0)
        success_entropy = _bernoulli_entropy_from_logits(success_logit) / math.log(2.0)
        entropy_outcome_logits = (
            outcome_logits
            if self.config.recovery_supervised
            else outcome_logits[..., :2]
        )
        outcome_entropy = _categorical_entropy(entropy_outcome_logits) / max(
            math.log(entropy_outcome_logits.shape[-1]), 1e-6
        )
        scale_uncertainty = torch.stack(
            [
                torch.sigmoid(duration_log_scale).mean(-1),
                torch.sigmoid(object_delta_log_scale).mean(-1),
                torch.sigmoid(future_latent_log_scale).mean(-1),
            ],
            dim=-1,
        ).mean(-1)
        aleatoric_uncertainty = torch.stack(
            [event_entropy, reach_entropy, success_entropy, outcome_entropy, scale_uncertainty],
            dim=-1,
        ).mean(-1)

        if post_predicate_logits is not None:
            predicate_entropy = _bernoulli_entropy_from_logits(
                post_predicate_logits
            ).mean(-1) / math.log(2.0)
            aleatoric_uncertainty = torch.stack(
                [aleatoric_uncertainty, predicate_entropy], dim=-1
            ).mean(-1)

        result = {
            "semantic": semantic,
            "action_effect": action_effect,
            "transition": transition,
            "next_event_logits": next_event_logits,
            "next_reached_event_logits": destination_event_logits,
            "reach_logits": reach_logits,
            "reach_logit": reach_logit,
            "success_logit": success_logit,
            "outcome_logits": outcome_logits,
            "duration_log_mean": duration_log_mean,
            "duration_log_scale": duration_log_scale,
            "duration_selected_log_mean": duration_selected_log_mean,
            "duration_selected_log_scale": duration_selected_log_scale,
            "clock_log_tau": clock_log_tau,
            "object_delta_mean": object_delta_mean,
            "object_delta_log_scale": object_delta_log_scale,
            "object_delta_scale": object_delta_log_scale,  # training API alias
            "future_latent_mean": future_latent_mean,
            "future_latent_log_scale": future_latent_log_scale,
            "predicted_next_semantic": future_latent_mean,
            "aleatoric_event_entropy": event_entropy,
            "aleatoric_reach_entropy": reach_entropy,
            "aleatoric_success_entropy": success_entropy,
            "aleatoric_outcome_entropy": outcome_entropy,
            "aleatoric_scale_uncertainty": scale_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
        }
        if relative_transition_logits is not None and post_predicate_logits is not None:
            result.update(
                {
                    "relative_transition_logits": relative_transition_logits,
                    "post_predicate_logits": post_predicate_logits,
                    "post_predicate_probability": torch.sigmoid(
                        post_predicate_logits
                    ),
                    "predicate_delta": torch.sigmoid(post_predicate_logits)
                    - normalized_predicates,
                }
            )
        return result

    @staticmethod
    def _expand_candidate_value(
        value: torch.Tensor | None, batch: int, candidates: int, name: str
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if value.shape[0] != batch:
            raise ValueError(f"{name} first dimension must equal batch")
        candidate_specific_rank = {
            "action_mask": 3,
            "action_feature_mask": 4,
            "proprio": 3,
            "body_id": 2,
            "policy_id": 2,
            "current_event_id": 2,
            "clock_event_id": 2,
            "current_predicates": 3,
            "beta": 2,
            "dt": 2,
            "history_mask": 3,
        }.get(name)
        if candidate_specific_rank is None:
            raise ValueError(f"unknown candidate metadata: {name}")
        if value.ndim == candidate_specific_rank:
            if value.shape[1] != candidates:
                raise ValueError(f"candidate-specific {name} must have C={candidates}")
            return value.reshape(batch * candidates, *value.shape[2:])
        return value[:, None].expand(batch, candidates, *value.shape[1:]).reshape(
            batch * candidates, *value.shape[1:]
        )

    def predict_candidates(
        self,
        hidden_t: torch.Tensor,
        candidate_actions: torch.Tensor,
        **metadata: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Vectorize one state against multiple action candidates."""

        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must be [B,C,H,A]")
        batch, candidates = candidate_actions.shape[:2]
        if hidden_t.shape[0] != batch:
            raise ValueError("hidden_t and candidate_actions batch differ")
        if hidden_t.ndim == 2:
            flat_hidden = hidden_t[:, None].expand(-1, candidates, -1).reshape(
                batch * candidates, hidden_t.shape[-1]
            )
        elif hidden_t.ndim == 3:
            flat_hidden = hidden_t[:, None].expand(-1, candidates, -1, -1).reshape(
                batch * candidates, *hidden_t.shape[1:]
            )
        else:
            raise ValueError("hidden_t must be [B,D] or [B,T,D]")
        flat_metadata = {
            name: self._expand_candidate_value(value, batch, candidates, name)
            for name, value in metadata.items()
        }
        flat = self.forward(
            flat_hidden,
            candidate_actions.reshape(
                batch * candidates, *candidate_actions.shape[2:]
            ),
            **flat_metadata,
        )
        predictions = {
            key: value.reshape(batch, candidates, *value.shape[1:])
            for key, value in flat.items()
        }
        if self.action_rank_head is not None:
            action_effect = predictions["action_effect"]
            baseline_action_effect = action_effect[:, :1].expand_as(action_effect)
            residual = self.relative_action_rank_logit(
                predictions["semantic"],
                action_effect,
                baseline_action_effect,
            )
            base_success_logit = predictions["success_logit"]
            adjusted_success_logit = base_success_logit + residual
            predictions["base_success_logit"] = base_success_logit
            predictions["action_rank_residual"] = residual
            predictions["success_logit"] = adjusted_success_logit

            old_entropy = predictions["aleatoric_success_entropy"]
            new_entropy = (
                _bernoulli_entropy_from_logits(adjusted_success_logit) / math.log(2.0)
            )
            predictions["aleatoric_success_entropy"] = new_entropy
            coefficient = 0.1 if self.config.structured_events else 0.2
            predictions["aleatoric_uncertainty"] = (
                predictions["aleatoric_uncertainty"]
                + coefficient * (new_entropy - old_entropy)
            )
        return predictions

    @staticmethod
    def _lognormal_discount_moment(
        log_mean: torch.Tensor, log_scale: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if gamma == 1.0:
            return torch.ones_like(log_mean)
        # Five-point Gauss-Hermite quadrature for E[gamma**D], D log-normal.
        nodes = log_mean.new_tensor(
            [-2.0201828705, -0.9585724646, 0.0, 0.9585724646, 2.0201828705]
        )
        weights = log_mean.new_tensor(
            [0.0199532421, 0.3936193232, 0.9453087205, 0.3936193232, 0.0199532421]
        ) / math.sqrt(math.pi)
        sigma = torch.exp(log_scale).clamp(min=1e-4, max=10.0)
        log_duration = log_mean[..., None] + math.sqrt(2.0) * sigma[..., None] * nodes
        # Duration heads follow the existing Stage-3 contract: a Normal model
        # for log1p(D), not log(D).
        duration = torch.expm1(torch.clamp(log_duration, 0.0, 12.0))
        return (weights * torch.exp(math.log(gamma) * duration)).sum(dim=-1)

    def score_candidates(
        self,
        predictions: Mapping[str, torch.Tensor],
        *,
        event_values: torch.Tensor | None = None,
        gamma: float = 0.99,
        success_weight: float = 1.0,
        event_value_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        epistemic_uncertainty: torch.Tensor | None = None,
        candidate_distance: torch.Tensor | None = None,
        distance_weight: float = 0.0,
    ) -> torch.Tensor:
        """Score flat or ``[B,C]`` candidate predictions for safe reranking."""

        event_logits = predictions["next_event_logits"]
        num_events = event_logits.shape[-1]
        if num_events != self.config.num_events:
            raise ValueError("prediction event vocabulary does not match model config")
        if event_values is None:
            event_values = torch.linspace(
                0.0, 1.0, num_events, device=event_logits.device, dtype=event_logits.dtype
            )
        else:
            event_values = event_values.to(device=event_logits.device, dtype=event_logits.dtype)
        if event_values.shape != (num_events,):
            raise ValueError(f"event_values must have shape {(num_events,)}")

        event_probability = torch.softmax(event_logits, dim=-1)
        if "duration_selected_log_mean" in predictions:
            selected_discount = self._lognormal_discount_moment(
                predictions["duration_selected_log_mean"],
                predictions["duration_selected_log_scale"],
                gamma,
            )
            discount = selected_discount[..., None]
        else:
            # Compatibility for external predictors whose duration is indexed by
            # destination event rather than by the current event.
            discount = self._lognormal_discount_moment(
                predictions["duration_log_mean"], predictions["duration_log_scale"], gamma
            )
        discounted_event_value = (
            event_probability * discount * event_values
        ).sum(dim=-1)
        success_probability = torch.sigmoid(predictions["success_logit"])
        uncertainty = predictions["aleatoric_uncertainty"]
        if epistemic_uncertainty is not None:
            uncertainty = uncertainty + epistemic_uncertainty.to(uncertainty)
        if self.config.action_rank_success_only:
            # This ranker is supervised only by terminal success ordering.
            # Uncertainty and candidate distance stay available to the guard;
            # they cannot silently become an extra ranking objective.
            score = success_weight * success_probability
        else:
            score = (
                success_weight * success_probability
                + event_value_weight * discounted_event_value
                - uncertainty_weight * uncertainty
            )
        if candidate_distance is not None and not self.config.action_rank_success_only:
            score = score - distance_weight * candidate_distance.to(score)
        return score


__all__ = [
    "ActionConditionedEventWorldModel",
    "EmbodimentClockCell",
    "EventWorldModelConfig",
    "SemanticEncoder",
    "TemporalActionEncoder",
]
