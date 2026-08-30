#!/usr/bin/env python3
"""Policy-independent runtime protocol for the five-member shared event critic.

This module owns only canonical tensor/provenance validation and ensemble
scoring.  Policy inference, action conversion, state observation and environment
execution are deliberately supplied by external structural adapters.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch


FORMAT = "etsf_shared_event_critic_plugin_protocol_v1"
AUTHORITY_FORMAT = "etsf_shared_event_critic_plugin_authority_v1"
CANONICAL_STATE_SCHEMA = "dual_ee_object_relative_state_27d_v2"
CANONICAL_ACTION_SCHEMA = "dual_ee_se3_gripper_delta_14d_v2"
STATE_DIM = 27
ACTION_DIM = 14
CANDIDATE_COUNT = 4
SUPPORTED_CANDIDATE_COUNTS = (4, 8, 16)
EXECUTED_PREFIX_STEPS = 5
ENSEMBLE_MEMBER_COUNT = 5
EPISTEMIC_RISK_WEIGHT = 0.25
MEMBER_SCORE_KEY = "candidate_rank_logit"


class SharedEventCriticProtocolError(RuntimeError):
    """A plugin component, authority or canonical tensor broke the contract."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise SharedEventCriticProtocolError(f"{name} must be a lowercase SHA-256")
    return str(value)


def _require_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SharedEventCriticProtocolError(f"{name} must be a non-empty normalized string")
    return value


@dataclass(frozen=True)
class AuthorityProvenance:
    """Content-addressed authority joining policy-native and canonical sides.

    A matching dimension is intentionally not evidence of matching semantics.
    An identity/native-canonical declaration therefore requires a separate
    semantic-evidence digest; otherwise an explicit effect adapter contract is
    the only accepted boundary.
    """

    policy_family: str
    native_action_schema: str
    native_action_semantics_evidence_sha256: str | None
    actor_checkpoint_sha256: str
    candidate_provider_implementation_sha256: str
    candidate_sampling_contract_sha256: str
    effect_adapter_source_action_schema: str
    effect_adapter_implementation_sha256: str
    effect_adapter_semantic_contract_sha256: str
    state_observer_implementation_sha256: str
    environment_executor_implementation_sha256: str
    environment_execution_contract_sha256: str
    task_event_contract_sha256: str
    critic_member_checkpoint_sha256: tuple[str, ...]
    canonical_state_schema: str = CANONICAL_STATE_SCHEMA
    canonical_action_schema: str = CANONICAL_ACTION_SCHEMA
    candidate_count: int = CANDIDATE_COUNT
    executed_prefix_steps: int = EXECUTED_PREFIX_STEPS
    candidate_zero_is_actor_baseline: bool = True
    same_ordered_candidate_set_for_baseline_and_critic: bool = True

    def __post_init__(self) -> None:
        _require_name(self.policy_family, "policy_family")
        _require_name(self.native_action_schema, "native_action_schema")
        _require_name(
            self.effect_adapter_source_action_schema,
            "effect_adapter_source_action_schema",
        )
        if self.effect_adapter_source_action_schema != self.native_action_schema:
            raise SharedEventCriticProtocolError(
                "effect adapter source schema must equal the provider native schema"
            )
        if self.canonical_state_schema != CANONICAL_STATE_SCHEMA:
            raise SharedEventCriticProtocolError("canonical state schema changed")
        if self.canonical_action_schema != CANONICAL_ACTION_SCHEMA:
            raise SharedEventCriticProtocolError("canonical action schema changed")
        if self.native_action_schema == self.canonical_action_schema:
            _require_sha256(
                self.native_action_semantics_evidence_sha256,
                "native_action_semantics_evidence_sha256",
            )
        elif self.native_action_semantics_evidence_sha256 is not None:
            _require_sha256(
                self.native_action_semantics_evidence_sha256,
                "native_action_semantics_evidence_sha256",
            )
        for name in (
            "actor_checkpoint_sha256",
            "candidate_provider_implementation_sha256",
            "candidate_sampling_contract_sha256",
            "effect_adapter_implementation_sha256",
            "effect_adapter_semantic_contract_sha256",
            "state_observer_implementation_sha256",
            "environment_executor_implementation_sha256",
            "environment_execution_contract_sha256",
            "task_event_contract_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        members = tuple(self.critic_member_checkpoint_sha256)
        if len(members) != ENSEMBLE_MEMBER_COUNT or len(set(members)) != len(members):
            raise SharedEventCriticProtocolError(
                "authority must bind five distinct critic member checkpoints"
            )
        for index, digest in enumerate(members):
            _require_sha256(digest, f"critic_member_checkpoint_sha256[{index}]")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count not in SUPPORTED_CANDIDATE_COUNTS
        ):
            raise SharedEventCriticProtocolError(
                "candidate count must be authority-bound to 4, 8 or 16"
            )
        if (
            isinstance(self.executed_prefix_steps, bool)
            or not isinstance(self.executed_prefix_steps, int)
            or self.executed_prefix_steps != EXECUTED_PREFIX_STEPS
        ):
            raise SharedEventCriticProtocolError("executed prefix is frozen to five steps")
        if self.candidate_zero_is_actor_baseline is not True:
            raise SharedEventCriticProtocolError("candidate zero must be the actor baseline")
        if self.same_ordered_candidate_set_for_baseline_and_critic is not True:
            raise SharedEventCriticProtocolError(
                "baseline and critic must use the same ordered candidate set"
            )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format": AUTHORITY_FORMAT,
            "policy_family": self.policy_family,
            "native_action_schema": self.native_action_schema,
            "native_action_semantics_evidence_sha256": (
                self.native_action_semantics_evidence_sha256
            ),
            "actor_checkpoint_sha256": self.actor_checkpoint_sha256,
            "candidate_provider_implementation_sha256": (
                self.candidate_provider_implementation_sha256
            ),
            "candidate_sampling_contract_sha256": (
                self.candidate_sampling_contract_sha256
            ),
            "effect_adapter_source_action_schema": (
                self.effect_adapter_source_action_schema
            ),
            "effect_adapter_implementation_sha256": (
                self.effect_adapter_implementation_sha256
            ),
            "effect_adapter_semantic_contract_sha256": (
                self.effect_adapter_semantic_contract_sha256
            ),
            "state_observer_implementation_sha256": (
                self.state_observer_implementation_sha256
            ),
            "environment_executor_implementation_sha256": (
                self.environment_executor_implementation_sha256
            ),
            "environment_execution_contract_sha256": (
                self.environment_execution_contract_sha256
            ),
            "task_event_contract_sha256": self.task_event_contract_sha256,
            "critic_member_checkpoint_sha256": list(
                self.critic_member_checkpoint_sha256
            ),
            "canonical_state_schema": self.canonical_state_schema,
            "canonical_action_schema": self.canonical_action_schema,
            "candidate_count": self.candidate_count,
            "executed_prefix_steps": self.executed_prefix_steps,
            "candidate_zero_is_actor_baseline": (
                self.candidate_zero_is_actor_baseline
            ),
            "same_ordered_candidate_set_for_baseline_and_critic": (
                self.same_ordered_candidate_set_for_baseline_and_critic
            ),
        }

    @property
    def logical_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        return {**value, "logical_sha256": canonical_sha256(value)}


@dataclass(frozen=True)
class CanonicalStateObservation:
    """One pre-candidate root observation in the shared-head state contract."""

    state: torch.Tensor
    current_event_id: int
    event_age_seconds: float
    remaining_action_budget: float
    planned_dt_seconds: float
    state_schema: str
    observer_implementation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, torch.Tensor) or self.state.shape != (STATE_DIM,):
            raise SharedEventCriticProtocolError("canonical root state must be [27]")
        if not self.state.is_floating_point() or not bool(torch.isfinite(self.state).all()):
            raise SharedEventCriticProtocolError("canonical root state must be finite floating point")
        if isinstance(self.current_event_id, bool) or not isinstance(
            self.current_event_id, int
        ) or not 0 <= self.current_event_id < 5:
            raise SharedEventCriticProtocolError("current event id must be an integer in 0..4")
        expected_event = torch.zeros(5, dtype=self.state.dtype, device=self.state.device)
        expected_event[self.current_event_id] = 1.0
        if not torch.equal(self.state[18:23], expected_event):
            raise SharedEventCriticProtocolError(
                "state27 event one-hot does not match current_event_id"
            )
        if bool(((self.state[23:27] < 0.0) | (self.state[23:27] > 1.0)).any()):
            raise SharedEventCriticProtocolError("state27 predicates must lie in [0,1]")
        for value, name, strictly_positive in (
            (self.event_age_seconds, "event_age_seconds", False),
            (self.remaining_action_budget, "remaining_action_budget", True),
            (self.planned_dt_seconds, "planned_dt_seconds", True),
        ):
            if not math.isfinite(float(value)) or (
                float(value) <= 0.0 if strictly_positive else float(value) < 0.0
            ):
                raise SharedEventCriticProtocolError(f"{name} is outside its physical domain")
        if self.state_schema != CANONICAL_STATE_SCHEMA:
            raise SharedEventCriticProtocolError("observer did not return canonical state27")
        _require_sha256(
            self.observer_implementation_sha256,
            "observer_implementation_sha256",
        )


@dataclass(frozen=True)
class CanonicalCandidateBatch:
    """One authority-bound ordered candidate decision for the shared event head."""

    state: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    action_available: torch.Tensor
    action_schema_id: torch.Tensor
    body_id: torch.Tensor
    dt: torch.Tensor
    current_event_id: torch.Tensor
    event_age_seconds: torch.Tensor
    remaining_action_budget: torch.Tensor
    candidate_ids: tuple[str, ...]
    baseline_candidate_index: int
    canonical_state_schema: str
    canonical_action_schema: str
    authority_logical_sha256: str
    ordered_native_candidate_set_sha256: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        tensors = {
            "state": self.state,
            "actions": self.actions,
            "action_mask": self.action_mask,
            "action_available": self.action_available,
            "action_schema_id": self.action_schema_id,
            "body_id": self.body_id,
            "dt": self.dt,
            "current_event_id": self.current_event_id,
            "event_age_seconds": self.event_age_seconds,
            "remaining_action_budget": self.remaining_action_budget,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise SharedEventCriticProtocolError("canonical batch fields must be torch tensors")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise SharedEventCriticProtocolError("canonical batch tensors must share one device")
        if self.state.ndim != 2 or self.state.shape[1] != STATE_DIM:
            raise SharedEventCriticProtocolError("canonical state must be [N,27]")
        candidate_count = int(self.state.shape[0])
        if candidate_count not in SUPPORTED_CANDIDATE_COUNTS:
            raise SharedEventCriticProtocolError(
                "canonical candidate axis must contain 4, 8 or 16 entries"
            )
        if (
            self.actions.ndim != 3
            or self.actions.shape[0] != candidate_count
            or self.actions.shape[1] < EXECUTED_PREFIX_STEPS
            or self.actions.shape[2] != ACTION_DIM
        ):
            raise SharedEventCriticProtocolError("canonical actions must be [N,H>=5,14]")
        horizon = self.actions.shape[1]
        if self.action_mask.shape != (candidate_count, horizon):
            raise SharedEventCriticProtocolError("action mask shape changed")
        vector_fields = (
            self.action_available,
            self.action_schema_id,
            self.body_id,
            self.dt,
            self.current_event_id,
            self.event_age_seconds,
            self.remaining_action_budget,
        )
        if any(value.shape != (candidate_count,) for value in vector_fields):
            raise SharedEventCriticProtocolError("canonical context fields must be [N]")
        for name, value in (("state", self.state), ("actions", self.actions)):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise SharedEventCriticProtocolError(f"{name} must be finite floating point")
        if self.action_mask.dtype != torch.bool or self.action_available.dtype != torch.bool:
            raise SharedEventCriticProtocolError("action masks/availability must use bool dtype")
        expected_mask = torch.arange(horizon, device=self.actions.device) < EXECUTED_PREFIX_STEPS
        expected_mask = expected_mask[None].expand(candidate_count, -1)
        if not torch.equal(self.action_mask, expected_mask):
            raise SharedEventCriticProtocolError("action mask must expose exactly the first five steps")
        if not bool(self.action_available.all()):
            raise SharedEventCriticProtocolError("all planned candidates must be available")
        integer_fields = (
            (self.action_schema_id, "action_schema_id"),
            (self.body_id, "body_id"),
            (self.current_event_id, "current_event_id"),
        )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        for value, name in integer_fields:
            if value.dtype not in integer_dtypes:
                raise SharedEventCriticProtocolError(f"{name} must use an integer dtype")
        if bool((self.action_schema_id != 0).any()) or bool((self.body_id != 0).any()):
            raise SharedEventCriticProtocolError("shared checkpoint requires schema/body row zero")
        for name, value in (
            ("dt", self.dt),
            ("event_age_seconds", self.event_age_seconds),
            ("remaining_action_budget", self.remaining_action_budget),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise SharedEventCriticProtocolError(f"{name} must be finite floating point")
            if not torch.equal(value, value[:1].expand_as(value)):
                raise SharedEventCriticProtocolError(f"all candidates must share one {name}")
        if bool((self.dt <= 0).any()) or bool((self.event_age_seconds < 0).any()) or bool(
            (self.remaining_action_budget <= 0).any()
        ):
            raise SharedEventCriticProtocolError("canonical physical context is outside its domain")
        if bool(((self.current_event_id < 0) | (self.current_event_id >= 5)).any()):
            raise SharedEventCriticProtocolError("current event id is outside 0..4")
        if not torch.equal(
            self.current_event_id, self.current_event_id[:1].expand_as(self.current_event_id)
        ):
            raise SharedEventCriticProtocolError("all candidates must share one current event")
        if not torch.equal(self.state, self.state[:1].expand_as(self.state)):
            raise SharedEventCriticProtocolError(
                "all candidates must share one bit-exact root state"
            )
        expected_event = torch.zeros(
            (candidate_count, 5), dtype=self.state.dtype, device=self.state.device
        )
        expected_event.scatter_(1, self.current_event_id[:, None].long(), 1.0)
        if not torch.equal(self.state[:, 18:23], expected_event):
            raise SharedEventCriticProtocolError(
                "state27 event one-hot does not match current_event_id"
            )
        if bool(((self.state[:, 23:27] < 0.0) | (self.state[:, 23:27] > 1.0)).any()):
            raise SharedEventCriticProtocolError("state27 predicates must lie in [0,1]")
        if len(self.candidate_ids) != candidate_count or (
            len(set(self.candidate_ids)) != candidate_count
        ):
            raise SharedEventCriticProtocolError(
                "candidate ids must contain N unique entries"
            )
        for candidate_id in self.candidate_ids:
            _require_name(candidate_id, "candidate_id")
        if self.baseline_candidate_index != 0:
            raise SharedEventCriticProtocolError("candidate zero must be the actor baseline")
        if self.canonical_state_schema != CANONICAL_STATE_SCHEMA:
            raise SharedEventCriticProtocolError("canonical state schema changed")
        if self.canonical_action_schema != CANONICAL_ACTION_SCHEMA:
            raise SharedEventCriticProtocolError("canonical action schema changed")
        _require_sha256(self.authority_logical_sha256, "authority_logical_sha256")
        _require_sha256(
            self.ordered_native_candidate_set_sha256,
            "ordered_native_candidate_set_sha256",
        )

    @property
    def candidate_count(self) -> int:
        """Return the validated runtime candidate-axis length."""

        return int(self.state.shape[0])

    def to_model_batch(self) -> Mapping[str, torch.Tensor]:
        return {
            "state": self.state,
            "actions": self.actions,
            "action_mask": self.action_mask,
            "action_available": self.action_available,
            "action_schema_id": self.action_schema_id,
            "body_id": self.body_id,
            "dt": self.dt,
            "current_event_id": self.current_event_id,
            "event_age_seconds": self.event_age_seconds,
            "remaining_action_budget": self.remaining_action_budget,
        }


@runtime_checkable
class PolicyCandidateProvider(Protocol):
    @property
    def policy_family(self) -> str: ...

    @property
    def native_action_schema(self) -> str: ...

    @property
    def actor_checkpoint_sha256(self) -> str: ...

    @property
    def implementation_sha256(self) -> str: ...

    @property
    def sampling_contract_sha256(self) -> str: ...

    def propose_candidates(
        self,
        observation: Any,
        instruction: str,
        *,
        query_seed: int,
        candidate_count: int,
    ) -> Any: ...


@runtime_checkable
class CanonicalEffectAdapter(Protocol):
    @property
    def source_action_schema(self) -> str: ...

    @property
    def target_action_schema(self) -> str: ...

    @property
    def implementation_sha256(self) -> str: ...

    @property
    def semantic_contract_sha256(self) -> str: ...

    def adapt_candidates(
        self,
        root_observation: Any,
        native_candidates: Any,
        canonical_state: CanonicalStateObservation,
        authority: AuthorityProvenance,
    ) -> CanonicalCandidateBatch: ...


@runtime_checkable
class CanonicalStateObserver(Protocol):
    @property
    def target_state_schema(self) -> str: ...

    @property
    def implementation_sha256(self) -> str: ...

    @property
    def task_event_contract_sha256(self) -> str: ...

    def observe_state(
        self,
        observation: Any,
        history: Any,
        task_context: Any,
    ) -> CanonicalStateObservation: ...


@runtime_checkable
class EnvironmentExecutor(Protocol):
    @property
    def native_action_schema(self) -> str: ...

    @property
    def implementation_sha256(self) -> str: ...

    @property
    def execution_contract_sha256(self) -> str: ...

    def reset(self, requested_seed: int) -> Any: ...

    def execute_candidate(
        self,
        native_candidate: Any,
        *,
        executed_prefix_steps: int,
    ) -> Any: ...


@runtime_checkable
class CriticMember(Protocol):
    """One loaded member whose runtime object is bound to a checkpoint digest."""

    @property
    def checkpoint_sha256(self) -> str: ...

    def __call__(
        self, batch: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]: ...


class BoundCriticMember(torch.nn.Module):
    """Bind one loaded eval-mode module to its independently verified file SHA.

    This wrapper avoids mutating the model with an ad-hoc provenance attribute.
    The caller must compute the digest from the exact checkpoint file before
    construction; the scorer then compares it with the ordered authority.
    """

    def __init__(self, member: torch.nn.Module, checkpoint_sha256: str) -> None:
        super().__init__()
        if not isinstance(member, torch.nn.Module):
            raise SharedEventCriticProtocolError(
                "bound critic member must wrap a torch.nn.Module"
            )
        self.member = member
        self._checkpoint_sha256 = _require_sha256(
            checkpoint_sha256, "checkpoint_sha256"
        )
        self.eval()

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        if self.training or self.member.training:
            raise SharedEventCriticProtocolError(
                "bound critic member must remain in eval mode"
            )
        return self.member(batch)


def validate_plugin_components(
    authority: AuthorityProvenance,
    *,
    candidate_provider: PolicyCandidateProvider,
    effect_adapter: CanonicalEffectAdapter,
    state_observer: CanonicalStateObserver,
    environment_executor: EnvironmentExecutor,
) -> None:
    """Structurally check components and bind every runtime identity to authority."""

    components = (
        (candidate_provider, PolicyCandidateProvider, "candidate_provider"),
        (effect_adapter, CanonicalEffectAdapter, "effect_adapter"),
        (state_observer, CanonicalStateObserver, "state_observer"),
        (environment_executor, EnvironmentExecutor, "environment_executor"),
    )
    for component, protocol, name in components:
        if not isinstance(component, protocol):
            raise SharedEventCriticProtocolError(f"{name} does not implement its runtime protocol")
    observed = {
        "policy_family": candidate_provider.policy_family,
        "native_action_schema": candidate_provider.native_action_schema,
        "actor_checkpoint_sha256": candidate_provider.actor_checkpoint_sha256,
        "candidate_provider_implementation_sha256": candidate_provider.implementation_sha256,
        "candidate_sampling_contract_sha256": candidate_provider.sampling_contract_sha256,
        "effect_adapter_source_action_schema": effect_adapter.source_action_schema,
        "canonical_action_schema": effect_adapter.target_action_schema,
        "effect_adapter_implementation_sha256": effect_adapter.implementation_sha256,
        "effect_adapter_semantic_contract_sha256": effect_adapter.semantic_contract_sha256,
        "canonical_state_schema": state_observer.target_state_schema,
        "state_observer_implementation_sha256": state_observer.implementation_sha256,
        "task_event_contract_sha256": state_observer.task_event_contract_sha256,
        "executor_native_action_schema": environment_executor.native_action_schema,
        "environment_executor_implementation_sha256": environment_executor.implementation_sha256,
        "environment_execution_contract_sha256": environment_executor.execution_contract_sha256,
    }
    expected = {
        "policy_family": authority.policy_family,
        "native_action_schema": authority.native_action_schema,
        "actor_checkpoint_sha256": authority.actor_checkpoint_sha256,
        "candidate_provider_implementation_sha256": authority.candidate_provider_implementation_sha256,
        "candidate_sampling_contract_sha256": authority.candidate_sampling_contract_sha256,
        "effect_adapter_source_action_schema": authority.effect_adapter_source_action_schema,
        "canonical_action_schema": authority.canonical_action_schema,
        "effect_adapter_implementation_sha256": authority.effect_adapter_implementation_sha256,
        "effect_adapter_semantic_contract_sha256": authority.effect_adapter_semantic_contract_sha256,
        "canonical_state_schema": authority.canonical_state_schema,
        "state_observer_implementation_sha256": authority.state_observer_implementation_sha256,
        "task_event_contract_sha256": authority.task_event_contract_sha256,
        "executor_native_action_schema": authority.native_action_schema,
        "environment_executor_implementation_sha256": authority.environment_executor_implementation_sha256,
        "environment_execution_contract_sha256": authority.environment_execution_contract_sha256,
    }
    mismatched = [name for name in expected if observed[name] != expected[name]]
    if mismatched:
        raise SharedEventCriticProtocolError(
            f"runtime components differ from authority: {mismatched}"
        )


@dataclass(frozen=True)
class SharedEventCriticScores:
    member_scores: torch.Tensor
    member_mean: torch.Tensor
    epistemic_population_std: torch.Tensor
    risk_adjusted_scores: torch.Tensor
    selected_candidate_index: int
    candidate_ids: tuple[str, ...]
    ordered_native_candidate_set_sha256: str
    authority_logical_sha256: str


class SharedEventCriticScorer:
    """Pure five-member scorer: utility mean minus 0.25 population std.

    It contains no fallback, threshold, guard, policy call or environment call.
    The selected index is simply ``argmax(risk_adjusted_scores)``.
    """

    def __init__(
        self,
        members: Sequence[CriticMember],
        *,
        authority: AuthorityProvenance,
    ) -> None:
        self.members = tuple(members)
        self.authority = authority
        if len(self.members) != ENSEMBLE_MEMBER_COUNT:
            raise SharedEventCriticProtocolError("shared critic requires exactly five members")
        if len({id(member) for member in self.members}) != ENSEMBLE_MEMBER_COUNT:
            raise SharedEventCriticProtocolError("critic members must be five distinct objects")
        for index, (member, expected_digest) in enumerate(
            zip(self.members, authority.critic_member_checkpoint_sha256, strict=True)
        ):
            if not isinstance(member, CriticMember):
                raise SharedEventCriticProtocolError(
                    f"critic member {index} lacks callable/checkpoint provenance"
                )
            observed_digest = _require_sha256(
                member.checkpoint_sha256,
                f"critic member {index} checkpoint_sha256",
            )
            if observed_digest != expected_digest:
                raise SharedEventCriticProtocolError(
                    f"critic member {index} checkpoint differs from authority"
                )

    def score(self, batch: CanonicalCandidateBatch) -> SharedEventCriticScores:
        if not isinstance(batch, CanonicalCandidateBatch):
            raise TypeError("score expects a CanonicalCandidateBatch")
        batch.validate()
        if batch.authority_logical_sha256 != self.authority.logical_sha256:
            raise SharedEventCriticProtocolError("canonical batch authority does not match scorer")
        if batch.candidate_count != self.authority.candidate_count:
            raise SharedEventCriticProtocolError(
                "canonical batch candidate count does not match authority"
            )
        if batch.canonical_state_schema != self.authority.canonical_state_schema or (
            batch.canonical_action_schema != self.authority.canonical_action_schema
        ):
            raise SharedEventCriticProtocolError("canonical batch schema does not match authority")
        model_batch = batch.to_model_batch()
        rows = []
        with torch.inference_mode():
            for index, member in enumerate(self.members):
                if isinstance(member, torch.nn.Module) and member.training:
                    raise SharedEventCriticProtocolError(
                        f"critic member {index} must be in eval mode"
                    )
                output = member(model_batch)
                if not isinstance(output, Mapping) or MEMBER_SCORE_KEY not in output:
                    raise SharedEventCriticProtocolError(
                        f"critic member {index} lacks {MEMBER_SCORE_KEY}"
                    )
                score = output[MEMBER_SCORE_KEY]
                if (
                    not isinstance(score, torch.Tensor)
                    or score.shape != (batch.candidate_count,)
                    or not score.is_floating_point()
                    or score.device != batch.actions.device
                    or not bool(torch.isfinite(score).all())
                ):
                    raise SharedEventCriticProtocolError(
                        f"critic member {index} score must be finite floating "
                        f"[{batch.candidate_count}] on batch device"
                    )
                rows.append(score)
            member_scores = torch.stack(rows, dim=0)
            member_mean = member_scores.mean(dim=0)
            epistemic_std = member_scores.std(dim=0, correction=0)
            risk_adjusted = member_mean - EPISTEMIC_RISK_WEIGHT * epistemic_std
            selected = int(torch.argmax(risk_adjusted).item())
        return SharedEventCriticScores(
            member_scores=member_scores,
            member_mean=member_mean,
            epistemic_population_std=epistemic_std,
            risk_adjusted_scores=risk_adjusted,
            selected_candidate_index=selected,
            candidate_ids=batch.candidate_ids,
            ordered_native_candidate_set_sha256=(
                batch.ordered_native_candidate_set_sha256
            ),
            authority_logical_sha256=batch.authority_logical_sha256,
        )


__all__ = [
    "ACTION_DIM",
    "AUTHORITY_FORMAT",
    "AuthorityProvenance",
    "BoundCriticMember",
    "CANONICAL_ACTION_SCHEMA",
    "CANONICAL_STATE_SCHEMA",
    "CANDIDATE_COUNT",
    "CanonicalCandidateBatch",
    "CanonicalEffectAdapter",
    "CanonicalStateObservation",
    "CanonicalStateObserver",
    "CriticMember",
    "ENSEMBLE_MEMBER_COUNT",
    "EPISTEMIC_RISK_WEIGHT",
    "EXECUTED_PREFIX_STEPS",
    "EnvironmentExecutor",
    "FORMAT",
    "PolicyCandidateProvider",
    "STATE_DIM",
    "SUPPORTED_CANDIDATE_COUNTS",
    "SharedEventCriticProtocolError",
    "SharedEventCriticScorer",
    "SharedEventCriticScores",
    "canonical_sha256",
    "validate_plugin_components",
]
