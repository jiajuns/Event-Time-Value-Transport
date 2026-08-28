#!/usr/bin/env python3
"""Pluggable ensemble critic for the action-conditioned ETSF world model.

This deployment-only module deliberately keeps policy and embodiment contracts
outside the learned event core.  A :class:`PolicyAdapter` maps native candidate
actions to the checkpoint action contract, while :class:`EmbodimentSpec` carries
the registered body id and clock calibration.  Uncalibrated or unregistered
adapters may use :meth:`EventCriticPlugin.predict` for monitoring, but guarded
reranking falls back to the actor candidate by default.

No OpenVLA or RoboTwin import is required.  The plugin therefore remains usable
from OpenVLA, ACT, diffusion-policy, or RL inference loops as long as each policy
provides a calibrated adapter and the checkpoint was trained/calibrated for the
requested embodiment and policy ids.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from etsf_policy_feature_action_bridge import (
    CONTRACT_KEY as POLICY_BRIDGE_CONTRACT_KEY,
    validate_checkpoint_policy_bridge_header,
    verify_checkpoint_policy_bridge,
)
from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from openvla_etsf_state_observer import (
    SOURCE as STATE_OBSERVER_SOURCE,
    StateHiddenEventPredicateObserver,
)


@dataclass(frozen=True)
class EmbodimentSpec:
    """Deployment contract for one calibrated robot embodiment.

    ``body_id`` selects parameters already present in the checkpoint.  Merely
    choosing an unused id is not cross-embodiment transfer: the body action
    adapter and clock must first be calibrated and the resulting checkpoint
    contract must register ``name -> body_id``.
    """

    name: str
    body_id: int
    beta: float = 0.0
    action_contract: str = "native"
    action_adapter_calibrated: bool = False
    clock_calibrated: bool = False
    calibration_id: str | None = None
    clock_step_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("embodiment name must not be empty")
        if self.body_id < 0:
            raise ValueError("body_id must be non-negative")
        if not math.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not math.isfinite(self.clock_step_scale) or self.clock_step_scale <= 0:
            raise ValueError("clock_step_scale must be positive and finite")

    @property
    def calibrated_for_rerank(self) -> bool:
        return self.action_adapter_calibrated and self.clock_calibrated


@dataclass
class HistoryState:
    """Policy state at a query point, including the complete hidden history."""

    hidden: torch.Tensor
    current_event_id: torch.Tensor
    history_mask: torch.Tensor | None = None
    proprio: torch.Tensor | None = None
    current_predicates: torch.Tensor | None = None
    predicate_source: str | None = None
    predicate_calibrated: bool = False
    predicate_calibration_id: str | None = None
    policy_name: str = ""
    adapter_name: str = "unregistered_raw_state"
    adapter_calibrated: bool = False
    calibration_id: str | None = None
    policy_bridge_verification_sha256: str | None = None
    state_feature_binding_sha256: str | None = None
    observer_source: str | None = None
    observer_artifact_sha256: str | None = None
    observer_calibrated: bool = False
    observer_valid_mask: torch.Tensor | None = None
    observer_confidence: torch.Tensor | None = None

    def validate(self, config: EventWorldModelConfig) -> None:
        if self.hidden.ndim not in (2, 3):
            raise ValueError("hidden must be [B,D] or [B,T,D]")
        if self.hidden.shape[-1] != config.state_input_dim:
            raise ValueError(
                f"hidden feature dimension must be {config.state_input_dim}"
            )
        batch = self.hidden.shape[0]
        if self.current_event_id.shape != (batch,):
            raise ValueError("current_event_id must be [B]")
        if self.current_event_id.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("current_event_id must use an integer dtype")
        if bool(
            (
                (self.current_event_id < 0)
                | (self.current_event_id >= config.num_events)
            ).any()
        ):
            raise ValueError("current_event_id is outside the checkpoint vocabulary")
        if self.history_mask is not None:
            expected = self.hidden.shape[:2] if self.hidden.ndim == 3 else (batch, 1)
            if self.history_mask.shape != expected:
                raise ValueError(f"history_mask must have shape {expected}")
            if bool((~self.history_mask.to(torch.bool).any(dim=1)).any()):
                raise ValueError("every history needs at least one valid hidden state")
        if config.proprio_dim > 0 and self.proprio is not None:
            if self.proprio.shape != (batch, config.proprio_dim):
                raise ValueError(
                    f"proprio must have shape {(batch, config.proprio_dim)}"
                )
        if config.proprio_dim == 0 and self.proprio is not None:
            raise ValueError("checkpoint does not accept proprio")
        if config.structured_events:
            if self.current_predicates is None:
                raise ValueError(
                    "structured checkpoint requires explicit current_predicates; "
                    "silent all-zero predicates are forbidden"
                )
            expected = (batch, config.num_predicates)
            if self.current_predicates.shape != expected:
                raise ValueError(f"current_predicates must have shape {expected}")
            predicates = self.current_predicates
            if not bool(torch.isfinite(predicates).all()):
                raise ValueError("current_predicates contain non-finite values")
            if bool(((predicates < 0) | (predicates > 1)).any()):
                raise ValueError("current_predicates must lie in [0,1]")
            if not self.predicate_source or not self.predicate_source.strip():
                raise ValueError(
                    "structured checkpoint requires an explicit predicate_source"
                )
        elif self.current_predicates is not None:
            raise ValueError(
                "current_predicates were supplied to a non-structured checkpoint"
            )
        observer_fields = (
            self.observer_source,
            self.observer_artifact_sha256,
            self.observer_valid_mask,
            self.observer_confidence,
        )
        if any(value is not None for value in observer_fields):
            if not all(value is not None for value in observer_fields):
                raise ValueError("state observer provenance fields must be all present")
            assert self.observer_valid_mask is not None
            assert self.observer_confidence is not None
            if self.observer_valid_mask.shape != (batch,):
                raise ValueError("observer_valid_mask must be [B]")
            if self.observer_valid_mask.dtype != torch.bool:
                raise ValueError("observer_valid_mask must use bool dtype")
            if self.observer_confidence.shape != (batch,):
                raise ValueError("observer_confidence must be [B]")
            if not bool(torch.isfinite(self.observer_confidence).all()) or bool(
                ((self.observer_confidence < 0) | (self.observer_confidence > 1)).any()
            ):
                raise ValueError("observer_confidence must contain finite values in [0,1]")
            if self.predicate_source != self.observer_source:
                raise ValueError("observer/predicate sources must match")
        elif self.observer_calibrated:
            raise ValueError("observer_calibrated requires explicit observer provenance")


class StateAdapter(ABC):
    """Map a policy-native state history into a checkpoint state contract.

    This interface is deliberately separate from :class:`PolicyAdapter`: action
    compatibility does not imply state compatibility.  For example, SmolVLA's
    720-D action-expert hidden cannot be sent to a checkpoint trained on a
    4096-D OpenVLA hidden merely because both emit 14-D actions.
    """

    def __init__(
        self,
        *,
        name: str,
        policy_name: str,
        calibrated: bool = False,
        calibration_id: str | None = None,
        policy_bridge_verification_sha256: str | None = None,
        state_feature_binding_sha256: str | None = None,
    ) -> None:
        if not name or not policy_name:
            raise ValueError("state adapter and policy names must not be empty")
        self.name = name
        self.policy_name = policy_name
        self.calibrated = calibrated
        self.calibration_id = calibration_id
        self.policy_bridge_verification_sha256 = policy_bridge_verification_sha256
        self.state_feature_binding_sha256 = state_feature_binding_sha256

    @abstractmethod
    def to_model_history(self, native_history: torch.Tensor) -> torch.Tensor:
        """Return ``[B,D]`` or ``[B,T,D]`` in the checkpoint state space."""

    def adapt(
        self,
        native_history: torch.Tensor,
        *,
        current_event_id: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        current_predicates: torch.Tensor | None = None,
        predicate_source: str | None = None,
        predicate_calibrated: bool = False,
        predicate_calibration_id: str | None = None,
    ) -> HistoryState:
        model_history = self.to_model_history(native_history)
        if model_history.ndim not in (2, 3):
            raise ValueError("state adapter must return [B,D] or [B,T,D]")
        return HistoryState(
            hidden=model_history,
            current_event_id=current_event_id,
            history_mask=history_mask,
            proprio=proprio,
            current_predicates=current_predicates,
            predicate_source=predicate_source,
            predicate_calibrated=predicate_calibrated,
            predicate_calibration_id=predicate_calibration_id,
            policy_name=self.policy_name,
            adapter_name=self.name,
            adapter_calibrated=self.calibrated,
            calibration_id=self.calibration_id,
            policy_bridge_verification_sha256=(
                self.policy_bridge_verification_sha256
            ),
            state_feature_binding_sha256=self.state_feature_binding_sha256,
        )


class IdentityStateAdapter(StateAdapter):
    """Identity mapping for a policy-specific event checkpoint.

    It validates dimensions but does not create cross-policy alignment.  A
    SmolVLA identity adapter is valid only with a world-model checkpoint trained
    on the same shared SmolVLA observation-state representation.
    """

    def __init__(self, *, expected_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if expected_dim < 1:
            raise ValueError("expected_dim must be positive")
        self.expected_dim = expected_dim

    def to_model_history(self, native_history: torch.Tensor) -> torch.Tensor:
        if native_history.ndim not in (2, 3):
            raise ValueError("native state history must be [B,D] or [B,T,D]")
        if native_history.shape[-1] != self.expected_dim:
            raise ValueError(
                f"native state dimension {native_history.shape[-1]} does not match "
                f"checkpoint contract {self.expected_dim}"
            )
        return native_history


@dataclass
class CandidateBatch:
    """Canonical candidates plus native actions retained for execution.

    Candidate zero is not implicitly the fallback; ``fallback_index`` records
    the actor's actual default for every batch item.  This avoids silently
    changing the baseline when candidates are sorted or filtered upstream.
    """

    actions: torch.Tensor
    execution_actions: torch.Tensor
    policy_id: torch.Tensor
    policy_name: str
    adapter_name: str
    adapter_calibrated: bool
    calibration_id: str | None = None
    policy_bridge_verification_sha256: str | None = None
    action_mapping_binding_sha256: str | None = None
    action_mask: torch.Tensor | None = None
    action_feature_mask: torch.Tensor | None = None
    candidate_distance: torch.Tensor | None = None
    fallback_index: torch.Tensor | None = None
    candidate_valid_mask: torch.Tensor | None = None
    duration_steps: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.actions.shape[1])

    def validate(self, config: EventWorldModelConfig) -> None:
        if self.actions.ndim != 4:
            raise ValueError("actions must be [B,C,H,A]")
        batch, candidates, horizon, action_dim = self.actions.shape
        if min(batch, candidates, horizon) < 1:
            raise ValueError("candidate batch dimensions B, C, and H must be positive")
        if action_dim != config.action_dim:
            raise ValueError(f"canonical action dimension must be {config.action_dim}")
        if self.execution_actions.ndim != 4:
            raise ValueError("execution_actions must be [B,C,H,A_native]")
        if self.execution_actions.shape[:3] != (batch, candidates, horizon):
            raise ValueError("execution_actions must align with canonical B,C,H")
        if self.policy_id.shape not in ((batch,), (batch, candidates)):
            raise ValueError("policy_id must be [B] or [B,C]")
        if not self.policy_name or not self.adapter_name:
            raise ValueError("policy_name and adapter_name must not be empty")
        if self.action_mask is not None and self.action_mask.shape != (
            batch,
            candidates,
            horizon,
        ):
            raise ValueError("action_mask must be [B,C,H]")
        if self.action_feature_mask is not None and self.action_feature_mask.shape != (
            batch,
            candidates,
            horizon,
            action_dim,
        ):
            raise ValueError("action_feature_mask must be [B,C,H,A]")
        for tensor, name in (
            (self.candidate_distance, "candidate_distance"),
            (self.candidate_valid_mask, "candidate_valid_mask"),
            (self.duration_steps, "duration_steps"),
        ):
            if tensor is not None and tensor.shape != (batch, candidates):
                raise ValueError(f"{name} must be [B,C]")
        fallback = self.resolved_fallback_index()
        if bool(((fallback < 0) | (fallback >= candidates)).any()):
            raise ValueError("fallback_index is outside the candidate range")
        valid = self.resolved_candidate_valid_mask()
        if bool((~valid.gather(1, fallback[:, None]).squeeze(1)).any()):
            raise ValueError("the actor fallback candidate must be valid")
        if self.action_mask is not None and bool(
            (~self.action_mask.to(torch.bool).any(dim=-1) & valid).any()
        ):
            raise ValueError("every valid candidate needs at least one action step")

    def resolved_fallback_index(self) -> torch.Tensor:
        if self.fallback_index is None:
            return torch.zeros(
                self.batch_size, dtype=torch.long, device=self.actions.device
            )
        return self.fallback_index.to(device=self.actions.device, dtype=torch.long)

    def resolved_candidate_valid_mask(self) -> torch.Tensor:
        if self.candidate_valid_mask is None:
            return torch.ones(
                (self.batch_size, self.num_candidates),
                dtype=torch.bool,
                device=self.actions.device,
            )
        return self.candidate_valid_mask.to(device=self.actions.device, dtype=torch.bool)


class PolicyAdapter(ABC):
    """Base interface from a policy's native action chunks to model actions."""

    def __init__(
        self,
        *,
        name: str,
        policy_id: int,
        calibrated: bool = False,
        calibration_id: str | None = None,
        policy_bridge_verification_sha256: str | None = None,
        action_mapping_binding_sha256: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("policy adapter name must not be empty")
        if policy_id < 0:
            raise ValueError("policy_id must be non-negative")
        self.name = name
        self.policy_id = policy_id
        self.calibrated = calibrated
        self.calibration_id = calibration_id
        self.policy_bridge_verification_sha256 = policy_bridge_verification_sha256
        self.action_mapping_binding_sha256 = action_mapping_binding_sha256

    @abstractmethod
    def to_model_actions(
        self,
        native_actions: torch.Tensor,
        embodiment: EmbodimentSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return canonical actions and a same-shaped feature-valid mask."""

    def action_distance_scale(
        self, canonical_actions: torch.Tensor, embodiment: EmbodimentSpec
    ) -> torch.Tensor:
        """Return a positive broadcastable scale for actor-distance guards."""

        del embodiment
        return torch.ones(
            canonical_actions.shape[-1],
            dtype=canonical_actions.dtype,
            device=canonical_actions.device,
        )

    def adapt(
        self,
        native_actions: torch.Tensor,
        embodiment: EmbodimentSpec,
        *,
        action_mask: torch.Tensor | None = None,
        fallback_index: torch.Tensor | None = None,
        candidate_valid_mask: torch.Tensor | None = None,
        candidate_distance: torch.Tensor | None = None,
        duration_steps: torch.Tensor | None = None,
    ) -> CandidateBatch:
        """Build a candidate batch while preserving native execution actions."""

        if native_actions.ndim != 4:
            raise ValueError("native_actions must be [B,C,H,A_native]")
        model_actions, feature_mask = self.to_model_actions(native_actions, embodiment)
        if model_actions.ndim != 4 or feature_mask.shape != model_actions.shape:
            raise ValueError("adapter outputs must be matching [B,C,H,A] tensors")
        batch, candidates, horizon = model_actions.shape[:3]
        if action_mask is None:
            action_mask = torch.ones(
                (batch, candidates, horizon),
                dtype=torch.bool,
                device=model_actions.device,
            )
        else:
            action_mask = action_mask.to(device=model_actions.device, dtype=torch.bool)
        if fallback_index is None:
            fallback_index = torch.zeros(batch, dtype=torch.long, device=model_actions.device)
        else:
            fallback_index = fallback_index.to(
                device=model_actions.device, dtype=torch.long
            )
        if candidate_distance is None:
            if bool(((fallback_index < 0) | (fallback_index >= candidates)).any()):
                raise ValueError("fallback_index is outside the candidate range")
            gather_index = fallback_index[:, None, None, None].expand(
                -1, 1, horizon, model_actions.shape[-1]
            )
            reference = model_actions.gather(1, gather_index)
            scale = self.action_distance_scale(model_actions, embodiment).to(
                device=model_actions.device, dtype=model_actions.dtype
            )
            if bool((~torch.isfinite(scale)).any()) or bool((scale <= 0).any()):
                raise ValueError("action distance scale must be positive and finite")
            difference = (model_actions - reference) / scale
            distance_mask = feature_mask.to(torch.bool) & action_mask[..., None]
            squared = difference.square() * distance_mask.to(difference.dtype)
            denominator = distance_mask.sum(dim=(-1, -2)).clamp_min(1)
            candidate_distance = torch.sqrt(
                squared.sum(dim=(-1, -2)) / denominator.to(squared.dtype)
            )
        policy_id = torch.full(
            (batch,), self.policy_id, dtype=torch.long, device=model_actions.device
        )
        return CandidateBatch(
            actions=model_actions,
            execution_actions=native_actions,
            policy_id=policy_id,
            policy_name=self.name,
            adapter_name=type(self).__name__,
            adapter_calibrated=self.calibrated,
            calibration_id=self.calibration_id,
            policy_bridge_verification_sha256=(
                self.policy_bridge_verification_sha256
            ),
            action_mapping_binding_sha256=self.action_mapping_binding_sha256,
            action_mask=action_mask,
            action_feature_mask=feature_mask.to(torch.bool),
            candidate_distance=candidate_distance,
            fallback_index=fallback_index,
            candidate_valid_mask=candidate_valid_mask,
            duration_steps=duration_steps,
        )


@dataclass(frozen=True)
class ScoringConfig:
    gamma: float = 0.99
    success_weight: float = 1.0
    event_value_weight: float = 1.0
    uncertainty_weight: float = 0.1
    distance_weight: float = 0.0
    event_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0,1]")
        for name in (
            "success_weight",
            "event_value_weight",
            "uncertainty_weight",
            "distance_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")


@dataclass(frozen=True)
class GuardConfig:
    minimum_score_margin: float = 0.05
    maximum_candidate_distance: float = 0.25
    maximum_total_uncertainty: float = 0.75
    require_calibrated_adapters: bool = True
    require_registered_contract: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_score_margin) or self.minimum_score_margin < 0:
            raise ValueError("minimum_score_margin must be non-negative and finite")
        for name in ("maximum_candidate_distance", "maximum_total_uncertainty"):
            value = getattr(self, name)
            if math.isnan(value) or value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CounterfactualDeployment:
    """Frozen validation choices loaded from an ensemble manifest."""

    format: str
    success_temperature: float
    duration_scale: float
    event_values: tuple[float, ...]
    event_weight: float
    duration_weight: float
    candidate_distance_weight: float
    guard_enabled: bool
    gain_margin: float | None
    uncertainty_threshold: float | None

    def __post_init__(self) -> None:
        if self.format != "etsf_counterfactual_ensemble_v1":
            raise ValueError(f"unsupported counterfactual format: {self.format}")
        if not math.isfinite(self.success_temperature) or self.success_temperature <= 0:
            raise ValueError("success temperature must be positive and finite")
        if not math.isfinite(self.duration_scale) or self.duration_scale <= 0:
            raise ValueError("duration scale must be positive and finite")
        if not self.event_values or any(not math.isfinite(x) for x in self.event_values):
            raise ValueError("manifest event values must be finite and non-empty")
        for name in ("event_weight", "duration_weight", "candidate_distance_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"manifest {name} must be non-negative and finite")
        if self.guard_enabled:
            if self.gain_margin is None or self.uncertainty_threshold is None:
                raise ValueError("enabled manifest guard lacks thresholds")
            if self.gain_margin < 0 or self.uncertainty_threshold < 0:
                raise ValueError("manifest guard thresholds must be non-negative")


@dataclass
class EnsemblePrediction:
    """Aggregated predictions and scores, retaining no per-member activations."""

    outputs: dict[str, torch.Tensor]
    member_count: int
    base_score: torch.Tensor
    score: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    state_adapter_name: str = "unregistered_raw_state"
    state_adapter_calibrated: bool = False
    state_policy_name: str = ""
    state_calibration_id: str | None = None
    state_contract_matched: bool = False
    policy_bridge_contract_matched: bool = False
    policy_bridge_verification_sha256: str | None = None
    state_feature_binding_sha256: str | None = None
    action_mapping_binding_sha256: str | None = None
    predicate_input_required: bool = False
    predicate_calibrated: bool = False
    predicate_contract_matched: bool = False
    predicate_source: str | None = None
    predicate_calibration_id: str | None = None
    observer_source: str | None = None
    observer_artifact_sha256: str | None = None
    observer_calibrated: bool = False
    observer_contract_matched: bool = False
    observer_valid_mask: torch.Tensor | None = None
    observer_confidence: torch.Tensor | None = None


@dataclass
class SelectionDecision:
    """Guarded candidate decision for every element of a state batch."""

    selected_index: torch.Tensor
    proposed_index: torch.Tensor
    fallback_index: torch.Tensor
    changed_from_actor: torch.Tensor
    guard_fallback_used: torch.Tensor
    fallback_reasons: tuple[tuple[str, ...], ...]
    score_margin: torch.Tensor
    selected_execution_actions: torch.Tensor
    prediction: EnsemblePrediction


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp_min(1e-8)
    return -(probability * probability.log()).sum(dim=-1)


def _bernoulli_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-8, 1.0 - 1e-8)
    return -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )


def _mixture_mutual_information(
    member_probability: torch.Tensor, *, categorical: bool
) -> torch.Tensor:
    """Mutual information across ensemble members, normalized to [0,1]."""

    mixture = member_probability.mean(dim=0)
    if categorical:
        classes = member_probability.shape[-1]
        normalizer = max(math.log(classes), 1e-8)
        information = _entropy(mixture) - _entropy(member_probability).mean(dim=0)
    else:
        normalizer = math.log(2.0)
        information = _bernoulli_entropy(mixture) - _bernoulli_entropy(
            member_probability
        ).mean(dim=0)
    return (information / normalizer).clamp(0.0, 1.0)


def _continuous_epistemic_ratio(
    member_mean: torch.Tensor, member_log_scale: torch.Tensor
) -> torch.Tensor:
    between = member_mean.var(dim=0, unbiased=False)
    within = torch.exp(2.0 * member_log_scale).mean(dim=0)
    return (between / (between + within + 1e-8)).mean(dim=-1).clamp(0.0, 1.0)


class EventCriticPlugin:
    """Frozen ensemble predictor plus uncertainty-aware candidate guard."""

    _CONTRACT_KEYS = (
        "cache_schema",
        "source_manifest_sha256",
        "events",
        "object_names",
        "object_target",
        "body_to_id",
        "policy_to_id",
        "state_contracts",
        POLICY_BRIDGE_CONTRACT_KEY,
        "counterfactual_ranking_contract",
        "predicate_contract",
        "candidate_contract",
    )

    def __init__(
        self,
        models: Sequence[ActionConditionedEventWorldModel],
        *,
        contract: Mapping[str, Any] | None = None,
        normalization: Mapping[str, Any] | None = None,
        checkpoint_paths: Sequence[Path] = (),
        device: torch.device | str = "cpu",
        counterfactual_deployment: CounterfactualDeployment | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        if len(models) < 2:
            raise ValueError("epistemic uncertainty requires at least two checkpoints")
        config = models[0].config
        if any(model.config != config for model in models[1:]):
            raise ValueError("ensemble checkpoint configs differ")
        self.device = torch.device(device)
        self.models = tuple(model.to(self.device).eval() for model in models)
        for model in self.models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.config = config
        self.contract = dict(contract or {})
        self.policy_feature_action_bridge = None
        if POLICY_BRIDGE_CONTRACT_KEY in self.contract:
            self.policy_feature_action_bridge = validate_checkpoint_policy_bridge_header(
                config.to_dict(), self.contract
            )
        raw_predicate_contract = self.contract.get("predicate_contract", {})
        self.predicate_contract = (
            dict(raw_predicate_contract)
            if isinstance(raw_predicate_contract, Mapping)
            else {}
        )
        self.state_contracts = self._validate_state_contracts(
            self.contract.get("state_contracts"),
            config=config,
            policy_to_id=self.contract.get("policy_to_id"),
        )
        self.normalization = dict(normalization or {})
        self.checkpoint_paths = tuple(Path(path) for path in checkpoint_paths)
        self.counterfactual_deployment = counterfactual_deployment
        self.manifest_path = manifest_path
        self.state_observer: StateHiddenEventPredicateObserver | None = None
        self._verified_policy_bridge_receipts: dict[str, dict[str, Any]] = {}

    def verify_policy_bridge(
        self, *, expected_policy: str, runtime_binding: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Verify and retain the exact runtime bridge used for actor override."""

        receipt = verify_checkpoint_policy_bridge(
            config=self.config.to_dict(),
            checkpoint_contract=self.contract,
            expected_policy=expected_policy,
            runtime_binding=runtime_binding,
        )
        self._verified_policy_bridge_receipts[expected_policy] = dict(receipt)
        return dict(receipt)

    def attach_state_observer(
        self, observer: StateHiddenEventPredicateObserver
    ) -> "EventCriticPlugin":
        """Attach a separately frozen observer without mutating model artifacts."""

        observer.validate_for_world_model(
            self.config, self.contract, self.predicate_contract
        )
        observer.to(self.device).eval()
        for parameter in observer.parameters():
            parameter.requires_grad_(False)
        self.state_observer = observer
        return self

    def observe_state(
        self,
        state_adapter: StateAdapter,
        native_history: torch.Tensor,
        *,
        history_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> HistoryState:
        """Build structured online inputs from policy hidden, without object truth.

        Monitor-only observers may call :meth:`predict`, but their validity mask
        is all false and :meth:`select` therefore cannot authorize an actor
        override.
        """

        if not self.config.structured_events:
            raise ValueError("state observer is only defined for structured checkpoints")
        if self.state_observer is None:
            raise RuntimeError("no state-hidden observer is attached")
        model_history = state_adapter.to_model_history(native_history).to(self.device)
        observer_mask = (
            None if history_mask is None else history_mask.to(self.device)
        )
        prediction = self.state_observer.observe(model_history, observer_mask)
        return HistoryState(
            hidden=model_history,
            current_event_id=prediction.current_event_id,
            history_mask=observer_mask,
            proprio=proprio,
            current_predicates=prediction.current_predicates,
            predicate_source=STATE_OBSERVER_SOURCE,
            predicate_calibrated=self.state_observer.rerank_enabled,
            predicate_calibration_id=self.state_observer.calibration_id,
            policy_name=state_adapter.policy_name,
            adapter_name=state_adapter.name,
            adapter_calibrated=state_adapter.calibrated,
            calibration_id=state_adapter.calibration_id,
            policy_bridge_verification_sha256=(
                state_adapter.policy_bridge_verification_sha256
            ),
            state_feature_binding_sha256=(
                state_adapter.state_feature_binding_sha256
            ),
            observer_source=STATE_OBSERVER_SOURCE,
            observer_artifact_sha256=self.state_observer.artifact_sha256,
            observer_calibrated=self.state_observer.rerank_enabled,
            observer_valid_mask=prediction.valid_for_rerank,
            observer_confidence=prediction.confidence,
        )

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Sequence[str | Path],
        *,
        device: torch.device | str = "cpu",
        strict_contract: bool = True,
    ) -> "EventCriticPlugin":
        paths = tuple(Path(path).expanduser().resolve() for path in checkpoint_paths)
        if len(paths) < 2:
            raise ValueError("provide at least two ensemble checkpoints")
        models: list[ActionConditionedEventWorldModel] = []
        contracts: list[dict[str, Any]] = []
        normalizations: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"checkpoint is not a mapping: {path}")
            if "model" not in checkpoint or "config" not in checkpoint:
                raise ValueError(f"checkpoint lacks model/config: {path}")
            config = EventWorldModelConfig.from_dict(checkpoint["config"])
            model = ActionConditionedEventWorldModel(config)
            model.load_state_dict(checkpoint["model"], strict=True)
            models.append(model)
            contracts.append(dict(checkpoint.get("contract", {})))
            normalizations.append(dict(checkpoint.get("normalization", {})))
        if any(model.config != models[0].config for model in models[1:]):
            raise ValueError("ensemble checkpoint configs differ")
        reference_contract = contracts[0]
        if strict_contract:
            for index, contract in enumerate(contracts[1:], start=1):
                for key in cls._CONTRACT_KEYS:
                    if reference_contract.get(key) != contract.get(key):
                        raise ValueError(
                            f"ensemble contract mismatch at member {index}: {key}"
                        )
                if not cls._json_equivalent(normalizations[0], normalizations[index]):
                    raise ValueError(
                        f"ensemble normalization mismatch at member {index}"
                    )
        return cls(
            models,
            contract=reference_contract,
            normalization=normalizations[0],
            checkpoint_paths=paths,
            device=device,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json_equivalent(left: Any, right: Any) -> bool:
        """Compare JSON contracts while treating tuples and lists identically."""

        def plain(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, Mapping):
                return {str(key): plain(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(child) for child in value]
            return value

        return json.dumps(plain(left), sort_keys=True) == json.dumps(
            plain(right), sort_keys=True
        )

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        text = str(value)
        return len(text) == 64 and all(
            character in "0123456789abcdefABCDEF" for character in text
        )

    @classmethod
    def _validate_state_contracts(
        cls,
        raw_contracts: Any,
        *,
        config: EventWorldModelConfig,
        policy_to_id: Any,
    ) -> dict[str, dict[str, Any]]:
        """Validate content-addressed policy-state provenance.

        The first supported frozen representation is SmolVLA's observation-only
        contextualized prefix state. Its calibration id binds the hook anchor,
        dimension, noise boundary, and both audited source files. Merely naming
        an identity adapter "calibrated" is therefore insufficient.
        """

        registered_policies = (
            {str(name) for name in policy_to_id}
            if isinstance(policy_to_id, Mapping)
            else set()
        )
        if raw_contracts is None:
            if "smolvla" in registered_policies:
                raise ValueError(
                    "SmolVLA checkpoint lacks a frozen shared-state contract"
                )
            return {}
        if not isinstance(raw_contracts, Mapping):
            raise ValueError("state_contracts must be a mapping")

        required = {
            "policy",
            "anchor",
            "source",
            "hidden_dim",
            "prefix_length",
            "noise_independence",
            "modeling_sha256",
            "bridge_sha256",
            "calibration_id",
        }
        validated: dict[str, dict[str, Any]] = {}
        for policy_name, raw_contract in raw_contracts.items():
            policy_name = str(policy_name)
            if not isinstance(raw_contract, Mapping):
                raise ValueError(f"state contract for {policy_name!r} must be a mapping")
            contract = dict(raw_contract)
            if set(contract) != required:
                missing = sorted(required - set(contract))
                extra = sorted(set(contract) - required)
                raise ValueError(
                    f"state contract fields differ for {policy_name!r}: "
                    f"missing={missing}, extra={extra}"
                )
            if str(contract["policy"]) != policy_name:
                raise ValueError("state contract policy key/value mismatch")
            if int(contract["hidden_dim"]) != config.state_input_dim:
                raise ValueError(
                    f"state contract dimension for {policy_name!r} does not match config"
                )
            if not cls._valid_sha256(contract["modeling_sha256"]) or not cls._valid_sha256(
                contract["bridge_sha256"]
            ):
                raise ValueError("state contract source hashes are invalid")
            if policy_name == "smolvla":
                expected_anchor = (
                    "contextualized_vlm_prefix_final_state_token_before_flow_noise_v1"
                )
                expected_source = (
                    "policy.model.vlm_with_expert.get_vlm_model().text_model.norm"
                )
                if contract["anchor"] != expected_anchor or contract["source"] != expected_source:
                    raise ValueError("unsupported SmolVLA shared-state hook anchor")
                if int(contract["prefix_length"]) != 0:
                    raise ValueError("SmolVLA shared-state contract requires prefix_length=0")
                if (
                    contract["noise_independence"]
                    != "bit_exact_at_group_intervention_query"
                ):
                    raise ValueError("unsupported SmolVLA noise-independence contract")

            content = {key: contract[key] for key in required - {"calibration_id"}}
            expected_id = hashlib.sha256(
                json.dumps(
                    content,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            if str(contract["calibration_id"]) != expected_id:
                raise ValueError("state contract calibration id does not bind its content")
            validated[policy_name] = contract
        if "smolvla" in registered_policies and "smolvla" not in validated:
            raise ValueError("registered SmolVLA policy lacks a shared-state contract")
        return validated

    @staticmethod
    def _resolve_manifest_artifact(manifest_path: Path, recorded: str) -> Path:
        candidate = Path(recorded).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        portable = manifest_path.parent / candidate.name
        if portable.is_file():
            return portable.resolve()
        raise FileNotFoundError(
            f"manifest artifact is unavailable at {candidate} or {portable}"
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        verify_sha256: bool = True,
    ) -> "EventCriticPlugin":
        """Load an embedded counterfactual ensemble and its frozen guard.

        The manifest, calibration temperature, scoring constants, and guard are
        a single validation-frozen deployment contract.  Callers must not pass a
        second hand-written scoring/guard configuration when using this loader.
        """

        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("ensemble manifest must contain a JSON object")
        format_name = str(manifest.get("format", ""))
        if format_name != "etsf_counterfactual_ensemble_v1":
            raise ValueError(f"unsupported counterfactual format: {format_name}")
        artifact = manifest.get("ensemble_checkpoint")
        if not isinstance(artifact, Mapping) or not artifact.get("path"):
            raise ValueError("manifest lacks ensemble_checkpoint.path")
        checkpoint_path = cls._resolve_manifest_artifact(path, str(artifact["path"]))
        if verify_sha256:
            expected = str(artifact.get("sha256", ""))
            if not expected or cls._sha256(checkpoint_path) != expected:
                raise ValueError("counterfactual ensemble SHA256 mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("counterfactual ensemble checkpoint must be a mapping")
        if str(checkpoint.get("format", "")) != format_name:
            raise ValueError("manifest/checkpoint format mismatch")
        if not cls._json_equivalent(manifest.get("config"), checkpoint.get("config")):
            raise ValueError("manifest/checkpoint config mismatch")
        if not cls._json_equivalent(manifest.get("contract"), checkpoint.get("contract")):
            raise ValueError("manifest/checkpoint contract mismatch")
        if not cls._json_equivalent(
            manifest.get("success_calibration"), checkpoint.get("success_calibration")
        ):
            raise ValueError("manifest/checkpoint success calibration mismatch")
        if not cls._json_equivalent(manifest.get("guard"), checkpoint.get("guard")):
            raise ValueError("manifest/checkpoint guard mismatch")
        if not cls._json_equivalent(
            manifest.get("scoring"), checkpoint.get("scoring")
        ):
            raise ValueError("manifest/checkpoint scoring mismatch")
        if (
            manifest.get("scoring_selection") is not None
            or checkpoint.get("scoring_selection") is not None
        ) and not cls._json_equivalent(
            manifest.get("scoring_selection"),
            checkpoint.get("scoring_selection"),
        ):
            raise ValueError("manifest/checkpoint scoring selection mismatch")
        if not cls._json_equivalent(
            manifest.get("normalization"), checkpoint.get("normalization")
        ):
            raise ValueError("manifest/checkpoint normalization mismatch")
        if float(manifest.get("duration_scale", -1.0)) != float(
            checkpoint.get("duration_scale", -2.0)
        ):
            raise ValueError("manifest/checkpoint duration scale mismatch")
        for mirrored_key in ("predicate_contract", "candidate_contract"):
            manifest_value = manifest.get(mirrored_key)
            checkpoint_value = checkpoint.get(mirrored_key)
            if (manifest_value is not None or checkpoint_value is not None) and not (
                manifest_value is not None
                and checkpoint_value is not None
                and cls._json_equivalent(manifest_value, checkpoint_value)
            ):
                raise ValueError(
                    f"manifest/checkpoint {mirrored_key} mirror mismatch"
                )

        config = EventWorldModelConfig.from_dict(manifest["config"])
        manifest_contract = dict(manifest.get("contract", {}))
        top_level_predicate_contract = manifest.get("predicate_contract")
        nested_predicate_contract = manifest_contract.get("predicate_contract")
        if (
            top_level_predicate_contract is not None
            and nested_predicate_contract is not None
            and not cls._json_equivalent(
                top_level_predicate_contract, nested_predicate_contract
            )
        ):
            raise ValueError("top-level/nested predicate_contract mismatch")
        predicate_contract = (
            top_level_predicate_contract
            if top_level_predicate_contract is not None
            else nested_predicate_contract
        )
        if config.structured_events:
            if not isinstance(predicate_contract, Mapping):
                raise ValueError(
                    "structured manifest lacks predicate derivation/calibration contract"
                )
            predicate_contract = dict(predicate_contract)
            if tuple(predicate_contract.get("names", ())) != config.predicate_names:
                raise ValueError("predicate contract vocabulary does not match config")
            if predicate_contract.get("derivation") != "derive_atomic_predicates_v1":
                raise ValueError("unsupported predicate derivation contract")
            if (
                predicate_contract.get("source")
                != "simulator_object_poses_at_query_step"
            ):
                raise ValueError("unsupported predicate observation source")
            event_spec_sha256 = str(predicate_contract.get("event_spec_sha256", ""))
            if len(event_spec_sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF"
                for character in event_spec_sha256
            ):
                raise ValueError("predicate contract lacks event-spec calibration hash")
            task_calibration = predicate_contract.get("task_calibration")
            if not isinstance(task_calibration, Mapping) or not task_calibration:
                raise ValueError(
                    "predicate contract task_calibration must be a non-empty mapping"
                )
            if predicate_contract.get("online_requires_explicit_predicates") is not True:
                raise ValueError("structured manifest must require explicit predicates")
            if predicate_contract.get("missing_policy") != "error":
                raise ValueError("structured manifest predicate missing-policy must be error")
            manifest_contract["predicate_contract"] = predicate_contract
        top_level_candidate_contract = manifest.get("candidate_contract")
        nested_candidate_contract = manifest_contract.get("candidate_contract")
        if (
            top_level_candidate_contract is not None
            and nested_candidate_contract is not None
            and not cls._json_equivalent(
                top_level_candidate_contract, nested_candidate_contract
            )
        ):
            raise ValueError("top-level/nested candidate_contract mismatch")
        candidate_contract = (
            top_level_candidate_contract
            if top_level_candidate_contract is not None
            else nested_candidate_contract
        )
        if candidate_contract is not None:
            if not isinstance(candidate_contract, Mapping):
                raise ValueError("candidate_contract must be a mapping")
            if candidate_contract.get("baseline_candidate_name") != "deterministic":
                raise ValueError("unsupported baseline candidate contract")
            if int(candidate_contract.get("fallback_index", -1)) != 0:
                raise ValueError("unsupported manifest fallback index")
            manifest_contract["candidate_contract"] = dict(candidate_contract)
        model_states = checkpoint.get("models")
        if not isinstance(model_states, Sequence) or len(model_states) < 2:
            raise ValueError("embedded ensemble must contain at least two model states")
        member_entries = manifest.get("members")
        member_seeds = checkpoint.get("member_seeds")
        if not isinstance(member_entries, Sequence) or len(member_entries) != len(
            model_states
        ):
            raise ValueError("manifest member provenance does not match embedded ensemble")
        if any(not isinstance(entry, Mapping) for entry in member_entries):
            raise ValueError("invalid member provenance entry")
        if not isinstance(member_seeds, Sequence) or len(member_seeds) != len(
            model_states
        ):
            raise ValueError("checkpoint member seeds do not match embedded ensemble")
        if [entry.get("seed") for entry in member_entries] != list(member_seeds):
            raise ValueError("manifest/checkpoint member seed mismatch")
        models = []
        for state in model_states:
            if not isinstance(state, Mapping):
                raise ValueError("embedded ensemble model state is not a mapping")
            model = ActionConditionedEventWorldModel(config)
            model.load_state_dict(state, strict=True)
            models.append(model)

        scoring = manifest.get("scoring")
        calibration = manifest.get("success_calibration")
        guard = manifest.get("guard")
        if not all(isinstance(value, Mapping) for value in (scoring, calibration, guard)):
            raise ValueError("manifest lacks scoring/calibration/guard mappings")
        scoring_selection = manifest.get("scoring_selection")
        if scoring_selection is not None:
            if not isinstance(scoring_selection, Mapping):
                raise ValueError("manifest scoring_selection must be a mapping")
            selected_id = scoring_selection.get("selected_candidate_id")
            if not selected_id or scoring.get("candidate_id") != selected_id:
                raise ValueError("frozen scoring does not match scoring selection audit")
            selection_rows = scoring_selection.get("candidates")
            if not isinstance(selection_rows, Sequence) or isinstance(
                selection_rows, (str, bytes)
            ):
                raise ValueError("scoring selection audit lacks candidate rows")
            selected_rows = [
                row
                for row in selection_rows
                if isinstance(row, Mapping) and row.get("candidate_id") == selected_id
            ]
            if len(selected_rows) != 1:
                raise ValueError("scoring selection audit has no unique selected row")
            for weight_name in (
                "event_weight",
                "duration_weight",
                "candidate_distance_weight",
            ):
                if float(scoring.get(weight_name, float("nan"))) != float(
                    selected_rows[0].get(weight_name, float("nan"))
                ):
                    raise ValueError(
                        "frozen scoring weights do not match scoring selection audit"
                    )
        if scoring.get("uncertainty") != (
            "success_epistemic_std_plus_mean_model_aleatoric"
        ):
            raise ValueError("unsupported manifest uncertainty contract")
        event_values = tuple(float(value) for value in scoring["event_values"])
        if len(event_values) != config.num_events:
            raise ValueError("manifest event vocabulary does not match model config")
        deployment = CounterfactualDeployment(
            format=format_name,
            success_temperature=float(calibration["temperature"]),
            duration_scale=float(manifest["duration_scale"]),
            event_values=event_values,
            event_weight=float(scoring["event_weight"]),
            duration_weight=float(scoring["duration_weight"]),
            candidate_distance_weight=float(scoring["candidate_distance_weight"]),
            guard_enabled=bool(guard["enabled"]),
            gain_margin=None
            if guard.get("gain_margin") is None
            else float(guard["gain_margin"]),
            uncertainty_threshold=None
            if guard.get("uncertainty_threshold") is None
            else float(guard["uncertainty_threshold"]),
        )

        # Member files are provenance records; the aggregate checkpoint remains
        # portable because it embeds every state.  Verify members when present.
        if verify_sha256:
            for member in member_entries:
                if not isinstance(member, Mapping) or not member.get("path"):
                    raise ValueError("invalid member provenance entry")
                recorded = Path(str(member["path"])).expanduser()
                portable = path.parent / recorded.name
                existing = recorded if recorded.is_file() else portable
                if existing.is_file() and cls._sha256(existing) != str(member.get("sha256", "")):
                    raise ValueError(f"ensemble member SHA256 mismatch: {existing}")
        return cls(
            models,
            contract=manifest_contract,
            normalization=manifest.get("normalization", {}),
            checkpoint_paths=(checkpoint_path,),
            device=device,
            counterfactual_deployment=deployment,
            manifest_path=path,
        )

    def _registered_embodiment(self, embodiment: EmbodimentSpec) -> bool:
        mapping = self.contract.get("body_to_id")
        return isinstance(mapping, Mapping) and mapping.get(embodiment.name) == embodiment.body_id

    def _registered_policy(self, candidates: CandidateBatch) -> bool:
        mapping = self.contract.get("policy_to_id")
        if not isinstance(mapping, Mapping) or candidates.policy_name not in mapping:
            return False
        expected = int(mapping[candidates.policy_name])
        return bool((candidates.policy_id == expected).all())

    def _policy_bridge_matches(
        self, state: HistoryState, candidates: CandidateBatch
    ) -> bool:
        """Bind adapted tensors to this plugin's verified runtime receipt."""

        policy = candidates.policy_name
        receipt = self._verified_policy_bridge_receipts.get(policy)
        bridge = self.policy_feature_action_bridge
        if (
            receipt is None
            or bridge is None
            or state.policy_name != policy
            or state.adapter_name != bridge["state_feature"]["adapter"]
            or candidates.adapter_name != bridge["action_mapping"]["adapter"]
        ):
            return False
        verification = str(receipt["verification_sha256"])
        return bool(
            state.policy_bridge_verification_sha256 == verification
            and candidates.policy_bridge_verification_sha256 == verification
            and state.state_feature_binding_sha256
            == receipt["state_feature_binding_sha256"]
            and candidates.action_mapping_binding_sha256
            == receipt["action_mapping_binding_sha256"]
            and receipt["bridge_contract_sha256"] == bridge["contract_sha256"]
            and self._registered_policy(candidates)
        )

    def _selection_policy_bridge_matches(
        self, prediction: EnsemblePrediction, candidates: CandidateBatch
    ) -> bool:
        receipt = self._verified_policy_bridge_receipts.get(candidates.policy_name)
        bridge = self.policy_feature_action_bridge
        if (
            receipt is None
            or bridge is None
            or not prediction.policy_bridge_contract_matched
            or prediction.state_policy_name != candidates.policy_name
            or prediction.state_adapter_name != bridge["state_feature"]["adapter"]
            or candidates.adapter_name != bridge["action_mapping"]["adapter"]
            or not self._registered_policy(candidates)
            or receipt["bridge_contract_sha256"] != bridge["contract_sha256"]
        ):
            return False
        verification = str(receipt["verification_sha256"])
        return bool(
            prediction.policy_bridge_verification_sha256 == verification
            and candidates.policy_bridge_verification_sha256 == verification
            and prediction.state_feature_binding_sha256
            == receipt["state_feature_binding_sha256"]
            and prediction.action_mapping_binding_sha256
            == receipt["action_mapping_binding_sha256"]
            and candidates.action_mapping_binding_sha256
            == receipt["action_mapping_binding_sha256"]
        )

    def _predicate_contract_matches(self, state: HistoryState) -> bool:
        if not self.config.structured_events:
            return True
        if state.observer_source is not None:
            return self._observer_contract_matches(state)
        if not self.predicate_contract or not state.predicate_calibrated:
            return False
        if state.predicate_source != str(self.predicate_contract.get("derivation", "")):
            return False
        expected_calibration = self.predicate_contract.get("calibration_id")
        if expected_calibration is None:
            expected_calibration = self.predicate_contract.get("calibration_sha256")
        if expected_calibration is None:
            expected_calibration = self.predicate_contract.get("event_spec_sha256")
        return bool(expected_calibration) and (
            state.predicate_calibration_id == str(expected_calibration)
        )

    def _observer_contract_matches(self, state: HistoryState) -> bool:
        observer = self.state_observer
        if observer is None:
            return False
        try:
            observer.validate_for_world_model(
                self.config, self.contract, self.predicate_contract
            )
        except ValueError:
            return False
        return observer.matches_state_provenance(
            source=state.observer_source,
            artifact_sha256=state.observer_artifact_sha256,
            calibration_id=state.predicate_calibration_id,
        )

    def _state_contract_matches(self, state: HistoryState) -> bool:
        """Return whether the adapter is bound to the checkpoint state source."""

        if not self.state_contracts:
            # Legacy OpenVLA checkpoints predate content-addressed state hooks.
            # This preserves monitor compatibility, but is not cross-policy proof.
            return True
        expected = self.state_contracts.get(state.policy_name)
        if expected is None:
            return False
        return bool(
            state.adapter_calibrated
            and state.calibration_id
            and state.calibration_id == str(expected["calibration_id"])
            and state.hidden.shape[-1] == int(expected["hidden_dim"])
        )

    def _metadata(
        self,
        state: HistoryState,
        candidates: CandidateBatch,
        embodiment: EmbodimentSpec,
    ) -> dict[str, torch.Tensor | None]:
        batch, count = candidates.actions.shape[:2]
        dt = candidates.duration_steps
        if dt is None:
            if candidates.action_mask is None:
                dt = torch.full(
                    (batch, count),
                    float(candidates.actions.shape[2]),
                    dtype=state.hidden.dtype,
                    device=self.device,
                )
            else:
                dt = candidates.action_mask.sum(dim=-1).to(state.hidden.dtype)
            dt = dt * embodiment.clock_step_scale
        return {
            "history_mask": None
            if state.history_mask is None
            else state.history_mask.to(self.device),
            "action_mask": None
            if candidates.action_mask is None
            else candidates.action_mask.to(self.device),
            "action_feature_mask": None
            if candidates.action_feature_mask is None
            else candidates.action_feature_mask.to(self.device),
            "proprio": None if state.proprio is None else state.proprio.to(self.device),
            "body_id": torch.full(
                (batch,), embodiment.body_id, dtype=torch.long, device=self.device
            ),
            "policy_id": candidates.policy_id.to(self.device),
            "current_event_id": state.current_event_id.to(self.device),
            # The online clock is anchored to the same canonical current event
            # used by the structured transition head.  Passing this explicitly
            # prevents future core defaults from changing the deployment contract.
            "clock_event_id": state.current_event_id.to(self.device),
            "current_predicates": None
            if state.current_predicates is None
            else state.current_predicates.to(self.device),
            "beta": torch.full(
                (batch,), embodiment.beta, dtype=state.hidden.dtype, device=self.device
            ),
            "dt": dt.to(device=self.device, dtype=state.hidden.dtype),
        }

    def _aggregate(
        self, member_outputs: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        temperature = (
            self.counterfactual_deployment.success_temperature
            if self.counterfactual_deployment is not None
            else 1.0
        )
        event_probability = torch.stack(
            [torch.softmax(output["next_event_logits"], dim=-1) for output in member_outputs]
        )
        reach_probability = torch.stack(
            [torch.sigmoid(output["reach_logit"]) for output in member_outputs]
        )
        success_probability = torch.stack(
            [
                torch.sigmoid(output["success_logit"] / temperature)
                for output in member_outputs
            ]
        )
        outcome_probability = torch.stack(
            [torch.softmax(output["outcome_logits"], dim=-1) for output in member_outputs]
        )
        aleatoric = torch.stack(
            [output["aleatoric_uncertainty"] for output in member_outputs]
        ).mean(dim=0)

        duration_mean = torch.stack(
            [output["duration_selected_log_mean"] for output in member_outputs]
        )
        duration_scale = torch.stack(
            [output["duration_selected_log_scale"] for output in member_outputs]
        )
        object_mean = torch.stack(
            [output["object_delta_mean"] for output in member_outputs]
        )
        object_scale = torch.stack(
            [output["object_delta_log_scale"] for output in member_outputs]
        )
        future_mean = torch.stack(
            [output["future_latent_mean"] for output in member_outputs]
        )
        future_scale = torch.stack(
            [output["future_latent_log_scale"] for output in member_outputs]
        )

        epistemic_components = {
            "epistemic_event": _mixture_mutual_information(
                event_probability, categorical=True
            ),
            "epistemic_reach": _mixture_mutual_information(
                reach_probability, categorical=False
            ),
            "epistemic_success": _mixture_mutual_information(
                success_probability, categorical=False
            ),
            "epistemic_outcome": _mixture_mutual_information(
                outcome_probability, categorical=True
            ),
            "epistemic_duration": _continuous_epistemic_ratio(
                duration_mean[..., None], duration_scale[..., None]
            ),
            "epistemic_object": _continuous_epistemic_ratio(
                object_mean, object_scale
            ),
            "epistemic_future_latent": _continuous_epistemic_ratio(
                future_mean, future_scale
            ),
        }
        factorized_epistemic = torch.stack(
            tuple(epistemic_components.values()), dim=-1
        ).mean(-1)
        success_epistemic_std = success_probability.std(dim=0, unbiased=False)
        # Counterfactual guard thresholds were calibrated using exactly this
        # uncertainty contract.  Factual-pretrain ensembles retain the broader
        # factorized disagreement diagnostic until their own guard is fitted.
        epistemic = (
            success_epistemic_std
            if self.counterfactual_deployment is not None
            else factorized_epistemic
        )

        normalized_object_mean = object_mean.mean(dim=0)
        raw_object_mean = normalized_object_mean
        object_center = self.normalization.get("object_delta_mean")
        object_std = self.normalization.get("object_delta_std")
        if object_center is not None and object_std is not None:
            center = torch.as_tensor(
                object_center,
                device=normalized_object_mean.device,
                dtype=normalized_object_mean.dtype,
            )
            scale = torch.as_tensor(
                object_std,
                device=normalized_object_mean.device,
                dtype=normalized_object_mean.dtype,
            )
            if center.shape != normalized_object_mean.shape[-1:] or scale.shape != center.shape:
                raise ValueError("object normalization does not match checkpoint output")
            raw_object_mean = normalized_object_mean * scale + center

        aggregate: dict[str, torch.Tensor] = {
            "next_event_probability": event_probability.mean(dim=0),
            "reach_probability": reach_probability.mean(dim=0),
            "success_probability": success_probability.mean(dim=0),
            "outcome_probability": outcome_probability.mean(dim=0),
            "duration_selected_log_mean": duration_mean.mean(dim=0),
            "duration_selected_log_scale": duration_scale.mean(dim=0),
            "object_delta_mean": raw_object_mean,
            "object_delta_mean_normalized": normalized_object_mean,
            "future_latent_mean": future_mean.mean(dim=0),
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "epistemic_success_std": success_epistemic_std,
            "epistemic_factorized": factorized_epistemic,
            "total_uncertainty": aleatoric + epistemic,
            **epistemic_components,
        }
        if "post_predicate_logits" in member_outputs[0]:
            predicate_probability = torch.stack(
                [
                    torch.sigmoid(output["post_predicate_logits"])
                    for output in member_outputs
                ]
            )
            relative_probability = torch.stack(
                [
                    torch.softmax(output["relative_transition_logits"], dim=-1)
                    for output in member_outputs
                ]
            )
            aggregate.update(
                {
                    "post_predicate_probability": predicate_probability.mean(0),
                    "relative_transition_probability": relative_probability.mean(0),
                    "epistemic_predicate": _mixture_mutual_information(
                        predicate_probability, categorical=False
                    ).mean(-1),
                    "epistemic_relative_transition": _mixture_mutual_information(
                        relative_probability, categorical=True
                    ),
                }
            )
        return aggregate

    @torch.inference_mode()
    def predict(
        self,
        state: HistoryState,
        candidates: CandidateBatch,
        embodiment: EmbodimentSpec,
        *,
        scoring: ScoringConfig | None = None,
    ) -> EnsemblePrediction:
        """Predict every candidate; this is allowed in monitor-only calibration."""

        state.validate(self.config)
        candidates.validate(self.config)
        if state.hidden.shape[0] != candidates.batch_size:
            raise ValueError("state and candidate batches differ")
        if state.policy_name and state.policy_name != candidates.policy_name:
            raise ValueError(
                "state and action adapters declare different policy contracts"
            )
        if embodiment.body_id >= self.config.num_bodies:
            raise ValueError("body_id exceeds checkpoint vocabulary")
        if bool((candidates.policy_id < 0).any()) or bool(
            (candidates.policy_id >= self.config.num_policies).any()
        ):
            raise ValueError("policy_id exceeds checkpoint vocabulary")
        predicate_contract_matched = self._predicate_contract_matches(state)
        observer_contract_matched = (
            self._observer_contract_matches(state)
            if state.observer_source is not None
            else False
        )
        state_contract_matched = self._state_contract_matches(state)
        policy_bridge_contract_matched = self._policy_bridge_matches(state, candidates)
        if self.counterfactual_deployment is not None and scoring is not None:
            raise ValueError(
                "manifest-loaded ensembles use their frozen scoring contract; "
                "do not pass ScoringConfig"
            )
        scoring = scoring or ScoringConfig()
        event_values = None
        if scoring.event_values is not None:
            event_values = torch.tensor(
                scoring.event_values, dtype=state.hidden.dtype, device=self.device
            )
        hidden = state.hidden.to(device=self.device, dtype=torch.float32)
        actions = candidates.actions.to(device=self.device, dtype=torch.float32)
        metadata = self._metadata(state, candidates, embodiment)
        member_outputs = tuple(
            model.predict_candidates(hidden, actions, **metadata) for model in self.models
        )
        aggregate = self._aggregate(member_outputs)
        if self.counterfactual_deployment is None:
            member_base_score = torch.stack(
                [
                    model.score_candidates(
                        output,
                        event_values=event_values,
                        gamma=scoring.gamma,
                        success_weight=scoring.success_weight,
                        event_value_weight=scoring.event_value_weight,
                        uncertainty_weight=0.0,
                    )
                    for model, output in zip(self.models, member_outputs)
                ]
            )
        else:
            deployment = self.counterfactual_deployment
            frozen_event_values = actions.new_tensor(deployment.event_values)
            scores = []
            for output in member_outputs:
                event_probability = torch.softmax(
                    output["next_event_logits"], dim=-1
                )
                event_progress = (
                    event_probability * frozen_event_values
                ).sum(dim=-1)
                predicted_duration = torch.expm1(
                    output["duration_selected_log_mean"].clamp(0.0, 12.0)
                )
                scores.append(
                    output["success_logit"] / deployment.success_temperature
                    + deployment.event_weight * event_progress
                    - deployment.duration_weight
                    * predicted_duration
                    / deployment.duration_scale
                )
            member_base_score = torch.stack(scores)
        base_score = member_base_score.mean(dim=0)
        total_uncertainty = aggregate["total_uncertainty"]
        if self.counterfactual_deployment is None:
            score = base_score - scoring.uncertainty_weight * total_uncertainty
            distance_weight = scoring.distance_weight
        else:
            # Uncertainty is a guard, not a second score penalty, because the
            # validation manifest fitted its gain threshold on this exact score.
            score = base_score
            distance_weight = self.counterfactual_deployment.candidate_distance_weight
            if distance_weight > 0 and candidates.candidate_distance is None:
                raise ValueError(
                    "manifest scoring requires candidate_distance from the policy adapter"
                )
        if candidates.candidate_distance is not None:
            score = score - distance_weight * candidates.candidate_distance.to(
                device=self.device, dtype=score.dtype
            )
        valid = candidates.resolved_candidate_valid_mask().to(self.device)
        score = score.masked_fill(~valid, -torch.inf)
        return EnsemblePrediction(
            outputs=aggregate,
            member_count=len(self.models),
            base_score=base_score,
            score=score,
            aleatoric_uncertainty=aggregate["aleatoric_uncertainty"],
            epistemic_uncertainty=aggregate["epistemic_uncertainty"],
            total_uncertainty=total_uncertainty,
            state_adapter_name=state.adapter_name,
            state_adapter_calibrated=state.adapter_calibrated,
            state_policy_name=state.policy_name,
            state_calibration_id=state.calibration_id,
            state_contract_matched=state_contract_matched,
            policy_bridge_contract_matched=policy_bridge_contract_matched,
            policy_bridge_verification_sha256=(
                state.policy_bridge_verification_sha256
                if policy_bridge_contract_matched
                else None
            ),
            state_feature_binding_sha256=(
                state.state_feature_binding_sha256
                if policy_bridge_contract_matched
                else None
            ),
            action_mapping_binding_sha256=(
                candidates.action_mapping_binding_sha256
                if policy_bridge_contract_matched
                else None
            ),
            predicate_input_required=self.config.structured_events,
            predicate_calibrated=state.predicate_calibrated,
            predicate_contract_matched=predicate_contract_matched,
            predicate_source=state.predicate_source,
            predicate_calibration_id=state.predicate_calibration_id,
            observer_source=state.observer_source,
            observer_artifact_sha256=state.observer_artifact_sha256,
            observer_calibrated=state.observer_calibrated,
            observer_contract_matched=observer_contract_matched,
            observer_valid_mask=(
                None
                if state.observer_valid_mask is None
                else state.observer_valid_mask.to(self.device)
            ),
            observer_confidence=(
                None
                if state.observer_confidence is None
                else state.observer_confidence.to(self.device)
            ),
        )

    def select(
        self,
        prediction: EnsemblePrediction,
        candidates: CandidateBatch,
        embodiment: EmbodimentSpec,
        *,
        guard: GuardConfig | None = None,
    ) -> SelectionDecision:
        """Apply margin, distance, uncertainty, and calibration guards."""

        candidates.validate(self.config)
        manifest_guard_disabled = False
        if self.counterfactual_deployment is not None:
            if guard is not None:
                raise ValueError(
                    "manifest-loaded ensembles use their frozen guard; "
                    "do not pass GuardConfig"
                )
            deployment = self.counterfactual_deployment
            manifest_guard_disabled = not deployment.guard_enabled
            guard = GuardConfig(
                minimum_score_margin=deployment.gain_margin or 0.0,
                maximum_candidate_distance=math.inf,
                maximum_total_uncertainty=(
                    deployment.uncertainty_threshold
                    if deployment.uncertainty_threshold is not None
                    else 0.0
                ),
            )
        else:
            guard = guard or GuardConfig()
        score = prediction.score
        if score.shape != (candidates.batch_size, candidates.num_candidates):
            raise ValueError("prediction score does not match candidate batch")
        fallback = candidates.resolved_fallback_index().to(score.device)
        proposed = score.argmax(dim=1)
        proposed_score = score.gather(1, proposed[:, None]).squeeze(1)
        fallback_score = score.gather(1, fallback[:, None]).squeeze(1)
        margin = proposed_score - fallback_score
        uncertainty = prediction.total_uncertainty.gather(
            1, proposed[:, None]
        ).squeeze(1)
        distances = candidates.candidate_distance
        selected = proposed.clone()
        all_reasons: list[tuple[str, ...]] = []
        body_registered = self._registered_embodiment(embodiment)
        policy_registered = self._registered_policy(candidates)
        for row in range(candidates.batch_size):
            reasons: list[str] = []
            changing = int(proposed[row]) != int(fallback[row])
            if changing:
                # This authorization is independent of the configurable guard:
                # a legacy or merely calibrated-looking checkpoint is monitor-only.
                if not self._selection_policy_bridge_matches(prediction, candidates):
                    reasons.append("policy_feature_action_bridge_not_verified")
                if manifest_guard_disabled:
                    reasons.append("manifest_guard_disabled")
                if not bool(torch.isfinite(proposed_score[row])):
                    reasons.append("nonfinite_candidate_score")
                if not bool(torch.isfinite(fallback_score[row])):
                    reasons.append("nonfinite_fallback_score")
                if not bool(margin[row] >= guard.minimum_score_margin):
                    reasons.append("score_margin_below_guard")
                if distances is None:
                    if math.isfinite(guard.maximum_candidate_distance):
                        reasons.append("missing_candidate_distance")
                else:
                    distance = distances[row, proposed[row]].to(score.device)
                    if not bool(torch.isfinite(distance)):
                        reasons.append("nonfinite_candidate_distance")
                    elif not bool(distance <= guard.maximum_candidate_distance):
                        reasons.append("candidate_distance_above_guard")
                if not bool(torch.isfinite(uncertainty[row])):
                    reasons.append("nonfinite_uncertainty")
                elif not bool(uncertainty[row] <= guard.maximum_total_uncertainty):
                    reasons.append("uncertainty_above_guard")
                # Observer authorization is not an optional adapter guard.  In
                # particular, setting require_calibrated_adapters=False must
                # never turn a monitor-only artifact into a reranker.
                if self.config.structured_events and prediction.observer_source is not None:
                    if not prediction.state_adapter_calibrated:
                        reasons.append("uncalibrated_state_adapter")
                    elif not prediction.state_contract_matched:
                        reasons.append("state_contract_mismatch")
                    if not prediction.observer_calibrated:
                        reasons.append("uncalibrated_state_observer")
                    elif not prediction.observer_contract_matched:
                        reasons.append("state_observer_contract_mismatch")
                    elif (
                        prediction.observer_valid_mask is None
                        or not bool(prediction.observer_valid_mask[row])
                    ):
                        reasons.append(
                            "state_observer_confidence_below_calibrated_gate"
                        )
                if guard.require_calibrated_adapters:
                    if prediction.observer_source is None:
                        if not prediction.state_adapter_calibrated:
                            reasons.append("uncalibrated_state_adapter")
                        elif not prediction.state_contract_matched:
                            reasons.append("state_contract_mismatch")
                    if (
                        self.config.structured_events
                        and prediction.observer_source is None
                    ):
                        if not prediction.predicate_calibrated:
                            reasons.append("uncalibrated_predicate_derivation")
                        elif not prediction.predicate_contract_matched:
                            reasons.append("predicate_contract_mismatch")
                    if not candidates.adapter_calibrated:
                        reasons.append("uncalibrated_policy_adapter")
                    if not embodiment.action_adapter_calibrated:
                        reasons.append("uncalibrated_embodiment_action_adapter")
                    if not embodiment.clock_calibrated:
                        reasons.append("uncalibrated_embodiment_clock")
                if guard.require_registered_contract:
                    if not policy_registered:
                        reasons.append("policy_not_registered_in_checkpoint")
                    if not body_registered:
                        reasons.append("embodiment_not_registered_in_checkpoint")
            if reasons:
                selected[row] = fallback[row]
            all_reasons.append(tuple(reasons))

        changed = selected != fallback
        fallback_used = (proposed != fallback) & ~changed
        execution = candidates.execution_actions.to(score.device)
        gather_index = selected[:, None, None, None].expand(
            -1, 1, execution.shape[2], execution.shape[3]
        )
        selected_execution = execution.gather(1, gather_index).squeeze(1)
        return SelectionDecision(
            selected_index=selected,
            proposed_index=proposed,
            fallback_index=fallback,
            changed_from_actor=changed,
            guard_fallback_used=fallback_used,
            fallback_reasons=tuple(all_reasons),
            score_margin=margin,
            selected_execution_actions=selected_execution,
            prediction=prediction,
        )

    def rerank(
        self,
        state: HistoryState,
        candidates: CandidateBatch,
        embodiment: EmbodimentSpec,
        *,
        scoring: ScoringConfig | None = None,
        guard: GuardConfig | None = None,
    ) -> SelectionDecision:
        """Predict and safely select in one inference-loop call."""

        prediction = self.predict(state, candidates, embodiment, scoring=scoring)
        return self.select(prediction, candidates, embodiment, guard=guard)


__all__ = [
    "CandidateBatch",
    "CounterfactualDeployment",
    "EmbodimentSpec",
    "EnsemblePrediction",
    "EventCriticPlugin",
    "GuardConfig",
    "HistoryState",
    "IdentityStateAdapter",
    "PolicyAdapter",
    "ScoringConfig",
    "SelectionDecision",
    "StateAdapter",
    "StateHiddenEventPredicateObserver",
]
