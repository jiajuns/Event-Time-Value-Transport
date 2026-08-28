#!/usr/bin/env python3
"""SmolVLA boundaries for the pluggable ETSF event critic.

The existing fixed-language HDF5 files expose candidate-specific 720-D action
expert hidden states.  Those are useful for the existing direct-Q baseline but
are not a shared observation state: flow-matching noise changes them across
candidates.  This module consequently accepts only a separately captured
pre-noise/shared state history.  It never selects candidate 0 hidden as a
shortcut and never pads 720 dimensions into an OpenVLA 4096-D checkpoint.
Structured checkpoints additionally require an explicit predicate vector from
the current query state, its derivation id, and matching calibration hash.

The audited LeRobot 0.4.4 collector uses the contextualized final state token
from the VLM prefix pass (960-D), before action-expert flow denoising.  A native
SmolVLA event checkpoint must be trained on that exact state definition.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from etsf_policy_feature_action_bridge import verify_checkpoint_policy_bridge
from example_openvla_event_critic_plugin import OpenVLAActionAdapter
from openvla_etsf_event_critic_plugin import (
    CandidateBatch,
    EmbodimentSpec,
    EventCriticPlugin,
    GuardConfig,
    HistoryState,
    IdentityStateAdapter,
    ScoringConfig,
    SelectionDecision,
)


class SmolVLAActionAdapter(OpenVLAActionAdapter):
    """Map SmolVLA's native continuous action chunk to model feature slots."""

    def __init__(
        self,
        *,
        policy_id: int,
        native_action_dim: int = 14,
        model_action_dim: int = 14,
        model_slots: Sequence[int] | None = None,
        distance_scale: Sequence[float] | None = None,
        calibrated: bool = False,
        calibration_id: str | None = None,
        policy_name: str = "smolvla",
        policy_bridge_verification_sha256: str | None = None,
        action_mapping_binding_sha256: str | None = None,
    ) -> None:
        super().__init__(
            policy_id=policy_id,
            native_action_dim=native_action_dim,
            model_action_dim=model_action_dim,
            model_slots=model_slots,
            distance_scale=distance_scale,
            calibrated=calibrated,
            calibration_id=calibration_id,
            policy_name=policy_name,
            policy_bridge_verification_sha256=policy_bridge_verification_sha256,
            action_mapping_binding_sha256=action_mapping_binding_sha256,
        )


class SmolVLAStateAdapter(IdentityStateAdapter):
    """Identity boundary for a SmolVLA-native event checkpoint.

    ``expected_dim`` must equal the state input dimension of a checkpoint that
    was trained on the same *shared observation-state* definition.  This class
    intentionally refuses dimensional projection; learned cross-policy state
    alignment needs its own calibrated checkpoint and validation evidence.
    """

    def __init__(
        self,
        *,
        expected_dim: int,
        calibrated: bool = False,
        calibration_id: str | None = None,
        policy_name: str = "smolvla",
        policy_bridge_verification_sha256: str | None = None,
        state_feature_binding_sha256: str | None = None,
    ) -> None:
        super().__init__(
            expected_dim=expected_dim,
            name="SmolVLAStateAdapter",
            policy_name=policy_name,
            calibrated=calibrated,
            calibration_id=calibration_id,
            policy_bridge_verification_sha256=policy_bridge_verification_sha256,
            state_feature_binding_sha256=state_feature_binding_sha256,
        )


def verify_smolvla_policy_bridge(
    *,
    checkpoint_config: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
    runtime_bridge_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless checkpoint, 960-D hook and adapters are all SmolVLA."""

    return verify_checkpoint_policy_bridge(
        config=checkpoint_config,
        checkpoint_contract=checkpoint_contract,
        expected_policy="smolvla",
        runtime_binding=runtime_bridge_binding,
    )


def adapt_smolvla_query(
    *,
    shared_state_history: torch.Tensor,
    candidate_action_chunks: torch.Tensor,
    current_event_id: torch.Tensor,
    proprio: torch.Tensor,
    embodiment: EmbodimentSpec,
    checkpoint_config: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
    runtime_bridge_binding: Mapping[str, Any],
    policy_id: int | None = None,
    current_predicates: torch.Tensor | None = None,
    predicate_source: str | None = None,
    predicate_calibrated: bool = False,
    predicate_calibration_id: str | None = None,
    history_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    fallback_index: torch.Tensor | None = None,
    state_adapter_calibrated: bool = False,
    action_adapter_calibrated: bool = False,
    state_calibration_id: str | None = None,
    action_calibration_id: str | None = None,
) -> tuple[HistoryState, CandidateBatch]:
    """Construct plugin inputs without importing LeRobot or loading the actor.

    For a structured checkpoint, pass ``current_predicates=[B,P]``,
    ``predicate_source='derive_atomic_predicates_v1'``, and the manifest
    ``event_spec_sha256`` as ``predicate_calibration_id``.  This only proves an
    input-contract match; SmolVLA still needs calibrated state/action adapters
    and same-policy validation before reranking is authorized.

    This detached constructor does not authorize a plugin and is therefore
    monitor-only on its own.  Guarded reranking must use
    :func:`rerank_smolvla_candidates`, which verifies the same plugin that
    consumes the returned tensors.  ``state_calibration_id`` must still match
    ``plugin.state_contracts['smolvla']['calibration_id']``; an arbitrary
    experiment label remains monitor-only.
    """

    bridge_receipt = verify_smolvla_policy_bridge(
        checkpoint_config=checkpoint_config,
        checkpoint_contract=checkpoint_contract,
        runtime_bridge_binding=runtime_bridge_binding,
    )
    bridge = checkpoint_contract["policy_feature_action_bridge"]
    bound_policy_row = int(bridge["policy_row"])
    if policy_id is not None and policy_id != bound_policy_row:
        raise ValueError("requested SmolVLA policy row differs from bridge contract")
    state_adapter = SmolVLAStateAdapter(
        expected_dim=int(checkpoint_config["state_input_dim"]),
        calibrated=state_adapter_calibrated,
        calibration_id=state_calibration_id,
        policy_bridge_verification_sha256=bridge_receipt["verification_sha256"],
        state_feature_binding_sha256=bridge_receipt[
            "state_feature_binding_sha256"
        ],
    )
    action_adapter = SmolVLAActionAdapter(
        policy_id=bound_policy_row,
        native_action_dim=candidate_action_chunks.shape[-1],
        model_action_dim=int(checkpoint_config["action_dim"]),
        model_slots=tuple(
            int(value) for value in bridge["action_mapping"]["model_slots"]
        ),
        calibrated=action_adapter_calibrated,
        calibration_id=action_calibration_id,
        policy_bridge_verification_sha256=bridge_receipt["verification_sha256"],
        action_mapping_binding_sha256=bridge_receipt[
            "action_mapping_binding_sha256"
        ],
    )
    state = state_adapter.adapt(
        shared_state_history,
        current_event_id=current_event_id,
        history_mask=history_mask,
        proprio=proprio,
        current_predicates=current_predicates,
        predicate_source=predicate_source,
        predicate_calibrated=predicate_calibrated,
        predicate_calibration_id=predicate_calibration_id,
    )
    candidates = action_adapter.adapt(
        candidate_action_chunks,
        embodiment,
        action_mask=action_mask,
        fallback_index=fallback_index,
    )
    return state, candidates


def rerank_smolvla_candidates(
    *,
    plugin: EventCriticPlugin,
    runtime_bridge_binding: Mapping[str, Any],
    shared_state_history: torch.Tensor,
    candidate_action_chunks: torch.Tensor,
    current_event_id: torch.Tensor,
    proprio: torch.Tensor,
    embodiment: EmbodimentSpec,
    current_predicates: torch.Tensor | None = None,
    predicate_source: str | None = None,
    predicate_calibrated: bool = False,
    predicate_calibration_id: str | None = None,
    history_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    fallback_index: torch.Tensor | None = None,
    policy_id: int | None = None,
    state_adapter_calibrated: bool = False,
    action_adapter_calibrated: bool = False,
    state_calibration_id: str | None = None,
    action_calibration_id: str | None = None,
    scoring: ScoringConfig | None = None,
    guard: GuardConfig | None = None,
) -> SelectionDecision:
    """Verify this exact plugin, adapt the query, then rerank without TOCTOU."""

    plugin.verify_policy_bridge(
        expected_policy="smolvla", runtime_binding=runtime_bridge_binding
    )
    state, candidates = adapt_smolvla_query(
        shared_state_history=shared_state_history,
        candidate_action_chunks=candidate_action_chunks,
        current_event_id=current_event_id,
        proprio=proprio,
        embodiment=embodiment,
        checkpoint_config=plugin.config.to_dict(),
        checkpoint_contract=plugin.contract,
        runtime_bridge_binding=runtime_bridge_binding,
        policy_id=policy_id,
        current_predicates=current_predicates,
        predicate_source=predicate_source,
        predicate_calibrated=predicate_calibrated,
        predicate_calibration_id=predicate_calibration_id,
        history_mask=history_mask,
        action_mask=action_mask,
        fallback_index=fallback_index,
        state_adapter_calibrated=state_adapter_calibrated,
        action_adapter_calibrated=action_adapter_calibrated,
        state_calibration_id=state_calibration_id,
        action_calibration_id=action_calibration_id,
    )
    return plugin.rerank(
        state, candidates, embodiment, scoring=scoring, guard=guard
    )


__all__ = [
    "SmolVLAActionAdapter",
    "SmolVLAStateAdapter",
    "adapt_smolvla_query",
    "rerank_smolvla_candidates",
    "verify_smolvla_policy_bridge",
]
