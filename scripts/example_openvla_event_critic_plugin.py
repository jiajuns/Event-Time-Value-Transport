#!/usr/bin/env python3
"""Minimal OpenVLA integration for the ETSF ensemble event critic.

The functions here intentionally accept tensors rather than importing a
particular OpenVLA fork.  An inference loop only needs to provide the hidden
history captured at policy query points and a batch of sampled native action
chunks.  The returned action chunk remains in the original OpenVLA execution
contract.

Setting ``calibrated=True`` is an authorization statement, not a convenience
flag.  It should be enabled only after the adapter/clock were calibrated on the
new policy or embodiment and the ensemble checkpoints register the same ids.
For a structured-event checkpoint the caller must also derive the current
predicate vector from the live query state with the manifest-recorded event
spec; this example never substitutes an all-zero vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from openvla_etsf_event_critic_plugin import (
    EmbodimentSpec,
    EventCriticPlugin,
    GuardConfig,
    IdentityStateAdapter,
    PolicyAdapter,
    ScoringConfig,
    SelectionDecision,
)


class OpenVLAStateAdapter(IdentityStateAdapter):
    """Validated identity adapter for the checkpoint's OpenVLA hidden space."""

    def __init__(
        self,
        *,
        expected_dim: int = 4096,
        calibrated: bool = False,
        calibration_id: str | None = None,
        policy_name: str = "openvla",
        policy_bridge_verification_sha256: str | None = None,
        state_feature_binding_sha256: str | None = None,
    ) -> None:
        super().__init__(
            expected_dim=expected_dim,
            name="OpenVLAStateAdapter",
            policy_name=policy_name,
            calibrated=calibrated,
            calibration_id=calibration_id,
            policy_bridge_verification_sha256=policy_bridge_verification_sha256,
            state_feature_binding_sha256=state_feature_binding_sha256,
        )


class OpenVLAActionAdapter(PolicyAdapter):
    """Pad/reindex native OpenVLA actions into a checkpoint action contract.

    ``model_slots[i]`` is the checkpoint feature receiving native action
    feature ``i``.  This makes the boundary explicit for robots whose recorder
    pads actions to a shared maximum dimension.  Coordinate transforms beyond
    reindexing (for example joint space to object-centric end-effector deltas)
    require an embodiment-specific subclass learned or calibrated on paired
    data; they must not be treated as zero-shot support.
    """

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
        policy_name: str = "openvla",
        policy_bridge_verification_sha256: str | None = None,
        action_mapping_binding_sha256: str | None = None,
    ) -> None:
        super().__init__(
            name=policy_name,
            policy_id=policy_id,
            calibrated=calibrated,
            calibration_id=calibration_id,
            policy_bridge_verification_sha256=policy_bridge_verification_sha256,
            action_mapping_binding_sha256=action_mapping_binding_sha256,
        )
        if native_action_dim < 1 or model_action_dim < 1:
            raise ValueError("action dimensions must be positive")
        slots = tuple(range(native_action_dim)) if model_slots is None else tuple(model_slots)
        if len(slots) != native_action_dim or len(set(slots)) != len(slots):
            raise ValueError("model_slots must uniquely map every native feature")
        if min(slots) < 0 or max(slots) >= model_action_dim:
            raise ValueError("model_slots fall outside model_action_dim")
        scale = (
            tuple(1.0 for _ in range(model_action_dim))
            if distance_scale is None
            else tuple(float(value) for value in distance_scale)
        )
        if len(scale) != model_action_dim or any(
            value <= 0 or not torch.isfinite(torch.tensor(value)) for value in scale
        ):
            raise ValueError("distance_scale must contain positive finite model scales")
        self.native_action_dim = native_action_dim
        self.model_action_dim = model_action_dim
        self.model_slots = slots
        self._distance_scale = scale

    def to_model_actions(
        self,
        native_actions: torch.Tensor,
        embodiment: EmbodimentSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del embodiment
        if native_actions.shape[-1] != self.native_action_dim:
            raise ValueError(
                f"expected {self.native_action_dim} native action features"
            )
        canonical = native_actions.new_zeros(
            (*native_actions.shape[:-1], self.model_action_dim)
        )
        feature_mask = torch.zeros_like(canonical, dtype=torch.bool)
        canonical[..., list(self.model_slots)] = native_actions
        feature_mask[..., list(self.model_slots)] = True
        return canonical, feature_mask

    def action_distance_scale(
        self, canonical_actions: torch.Tensor, embodiment: EmbodimentSpec
    ) -> torch.Tensor:
        del embodiment
        return canonical_actions.new_tensor(self._distance_scale)


def verify_openvla_policy_bridge(
    plugin: EventCriticPlugin,
    runtime_bridge_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless this is an exact OpenVLA-native checkpoint boundary."""

    return plugin.verify_policy_bridge(
        expected_policy="openvla", runtime_binding=runtime_bridge_binding
    )


def rerank_openvla_candidates(
    *,
    checkpoint_paths: Sequence[str | Path],
    runtime_bridge_binding: Mapping[str, Any],
    hidden_history: torch.Tensor,
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
    policy_id: int | None = None,
    policy_name: str = "openvla",
    policy_adapter_calibrated: bool = False,
    policy_calibration_id: str | None = None,
    state_adapter_calibrated: bool = False,
    state_calibration_id: str | None = None,
    fallback_index: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    scoring: ScoringConfig | None = None,
    guard: GuardConfig | None = None,
) -> SelectionDecision:
    """One-call integration for an OpenVLA sampling loop.

    Args:
        hidden_history: ``[B,T,4096]`` hidden states from all query points so
            far.  Passing only the current hidden changes the trained GRU state
            contract and is supported only for legacy ablations.
        candidate_action_chunks: Native OpenVLA actions ``[B,C,H,14]``.  The
            actor's deterministic/default candidate is identified explicitly by
            ``fallback_index``.
        current_predicates: Required as ``[B,P]`` for a structured checkpoint.
            Values must be derived from the current simulator/perception state
            with the exact event-spec calibration recorded by the manifest;
            omitting them is an error rather than an implicit all-zero state.
        predicate_source: Derivation identifier.  The current structured
            contract requires ``derive_atomic_predicates_v1``.
        predicate_calibration_id: The manifest ``event_spec_sha256`` for the
            calibration used to derive this online vector.  A learned
            perception predicate estimator needs its own validation evidence;
            simulator calibration does not make it zero-shot transferable.

    Returns:
        A decision containing ``selected_execution_actions`` in the unchanged
        native OpenVLA action contract, plus scores, uncertainties, and any
        guard fallback reasons.
    """

    plugin = EventCriticPlugin.from_checkpoints(checkpoint_paths, device=device)
    bridge_receipt = verify_openvla_policy_bridge(plugin, runtime_bridge_binding)
    bridge = plugin.policy_feature_action_bridge
    assert bridge is not None
    bound_policy_row = int(bridge["policy_row"])
    if policy_id is not None and policy_id != bound_policy_row:
        raise ValueError("requested OpenVLA policy row differs from bridge contract")
    model_slots = tuple(int(value) for value in bridge["action_mapping"]["model_slots"])
    adapter = OpenVLAActionAdapter(
        policy_id=bound_policy_row,
        native_action_dim=candidate_action_chunks.shape[-1],
        model_action_dim=plugin.config.action_dim,
        model_slots=model_slots,
        calibrated=policy_adapter_calibrated,
        calibration_id=policy_calibration_id,
        policy_name=policy_name,
        policy_bridge_verification_sha256=bridge_receipt["verification_sha256"],
        action_mapping_binding_sha256=bridge_receipt[
            "action_mapping_binding_sha256"
        ],
    )
    candidates = adapter.adapt(
        candidate_action_chunks,
        embodiment,
        action_mask=action_mask,
        fallback_index=fallback_index,
    )
    state_adapter = OpenVLAStateAdapter(
        expected_dim=plugin.config.state_input_dim,
        calibrated=state_adapter_calibrated,
        calibration_id=state_calibration_id,
        policy_name=policy_name,
        policy_bridge_verification_sha256=bridge_receipt["verification_sha256"],
        state_feature_binding_sha256=bridge_receipt[
            "state_feature_binding_sha256"
        ],
    )
    state = state_adapter.adapt(
        hidden_history,
        current_event_id=current_event_id,
        history_mask=history_mask,
        proprio=proprio,
        current_predicates=current_predicates,
        predicate_source=predicate_source,
        predicate_calibrated=predicate_calibrated,
        predicate_calibration_id=predicate_calibration_id,
    )
    return plugin.rerank(
        state,
        candidates,
        embodiment,
        scoring=scoring,
        guard=guard,
    )


__all__ = [
    "OpenVLAActionAdapter",
    "OpenVLAStateAdapter",
    "verify_openvla_policy_bridge",
    "rerank_openvla_candidates",
]
