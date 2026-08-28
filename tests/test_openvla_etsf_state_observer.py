from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_openvla_etsf_state_observer as trainer  # noqa: E402
from openvla_etsf_event_critic_plugin import (  # noqa: E402
    CandidateBatch,
    EmbodimentSpec,
    EventCriticPlugin,
    GuardConfig,
    IdentityStateAdapter,
)
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from openvla_etsf_state_observer import (  # noqa: E402
    FORMAT,
    SOURCE,
    StateHiddenEventPredicateObserver,
    StateObserverConfig,
    observer_artifact_payload,
    sha256,
)


EVENT_SPEC_SHA = "a" * 64


def world_config() -> EventWorldModelConfig:
    return EventWorldModelConfig(
        state_input_dim=8,
        action_dim=4,
        proprio_dim=0,
        semantic_dim=8,
        action_hidden_dim=7,
        transition_hidden_dim=10,
        clock_hidden_dim=6,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=True,
        dropout=0.0,
    )


def observer_metadata(
    config: EventWorldModelConfig,
    *,
    enabled: bool = False,
) -> tuple[dict, dict, dict]:
    contract = {
        "source": SOURCE,
        "state_source": "openvla_hidden_at_query",
        "label_derivation": trainer.LABEL_DERIVATION,
        "event_names": list(config.event_names),
        "predicate_names": list(config.predicate_names),
        "event_spec_sha256": EVENT_SPEC_SHA,
        "policy_to_id": {"openvla": 0},
        "body_to_id": {"piper": 0},
        "state_contracts": {},
        "train_groups": ["task|piper|1"],
        "validation_groups": ["task|piper|2"],
        "sealed_test_groups": ["task|piper|3"],
        "sealed_test_access": "identity_attrs_only_not_loaded_not_evaluated",
    }
    calibration = {
        "calibration_id": "observer-calibration-v1",
        "selection_data": (
            "independent_observer_calibration_no_world_model_sealed_test"
            if enabled
            else "observer_validation_monitor_only_not_world_model_sealed_test"
        ),
        "event_temperature": 1.0,
        "predicate_thresholds": [0.5] * config.num_predicates,
        "minimum_joint_confidence": 0.0,
    }
    deployment = {
        "rerank_enabled": enabled,
        "promotion_status": (
            "independent_validation_calibrated_and_explicitly_promoted"
            if enabled
            else "monitor_only_requires_independent_validation"
        ),
    }
    return contract, calibration, deployment


def make_observer(
    config: EventWorldModelConfig,
    *,
    artifact_sha256: str = "observer-checkpoint-sha",
) -> StateHiddenEventPredicateObserver:
    contract, calibration, deployment = observer_metadata(config)
    return StateHiddenEventPredicateObserver(
        StateObserverConfig(
            state_input_dim=config.state_input_dim,
            hidden_dim=8,
            event_names=config.event_names,
            predicate_names=config.predicate_names,
            dropout=0.0,
        ),
        contract=contract,
        calibration=calibration,
        deployment=deployment,
        artifact_sha256=artifact_sha256,
    )


def make_plugin(config: EventWorldModelConfig) -> EventCriticPlugin:
    models = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        models.append(ActionConditionedEventWorldModel(config))
    return EventCriticPlugin(
        models,
        contract={
            "body_to_id": {"piper": 0},
            "policy_to_id": {"openvla": 0},
            "predicate_contract": {
                "names": list(config.predicate_names),
                "derivation": "derive_atomic_predicates_v1",
                "event_spec_sha256": EVENT_SPEC_SHA,
                "calibration_id": EVENT_SPEC_SHA,
            },
        },
    )


def test_observer_uses_actual_last_valid_history_state() -> None:
    config = world_config()
    observer = make_observer(config).eval()
    hidden = torch.randn(2, 4, config.state_input_dim)
    # Deliberately non-contiguous: a sum(mask)-1 implementation would select
    # indices 1 and 0 instead of the actual final valid indices 3 and 2.
    mask = torch.tensor([[1, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool)
    history = observer(hidden, mask)
    direct = observer(torch.stack([hidden[0, 3], hidden[1, 2]]))
    assert torch.allclose(history["event_logits"], direct["event_logits"])
    assert torch.allclose(history["predicate_logits"], direct["predicate_logits"])


def test_monitor_only_observer_builds_online_state_and_cannot_rerank() -> None:
    config = world_config()
    observer = make_observer(config)
    plugin = make_plugin(config).attach_state_observer(observer)
    adapter = IdentityStateAdapter(
        expected_dim=config.state_input_dim,
        name="OpenVLAStateAdapter",
        policy_name="openvla",
        calibrated=True,
        calibration_id="openvla-state-v1",
    )
    hidden = torch.randn(1, 2, config.state_input_dim)
    state = plugin.observe_state(
        adapter,
        hidden,
        history_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert state.current_event_id.shape == (1,)
    assert state.current_predicates.shape == (1, config.num_predicates)
    assert state.predicate_source == SOURCE
    assert not state.observer_calibrated
    assert not bool(state.observer_valid_mask.any())

    actions = torch.randn(1, 2, 3, config.action_dim)
    candidates = CandidateBatch(
        actions=actions,
        execution_actions=actions.clone(),
        policy_id=torch.tensor([0]),
        policy_name="openvla",
        adapter_name="OpenVLAActionAdapter",
        adapter_calibrated=True,
        calibration_id="openvla-action-v1",
        action_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        candidate_distance=torch.tensor([[0.0, 0.1]]),
        fallback_index=torch.tensor([0]),
    )
    embodiment = EmbodimentSpec(
        name="piper",
        body_id=0,
        action_adapter_calibrated=True,
        clock_calibrated=True,
    )
    prediction = plugin.predict(state, candidates, embodiment)
    prediction.score = torch.tensor([[0.0, 1.0]])
    prediction.base_score = prediction.score.clone()
    prediction.total_uncertainty = torch.zeros_like(prediction.score)
    decision = plugin.select(
        prediction,
        candidates,
        embodiment,
        guard=GuardConfig(
            minimum_score_margin=0.0,
            maximum_candidate_distance=1.0,
            maximum_total_uncertainty=1.0,
            # Observer authorization remains mandatory even if a caller turns
            # off the legacy generic adapter guard for an ablation.
            require_calibrated_adapters=False,
        ),
    )
    assert decision.proposed_index.item() == 1
    assert decision.selected_index.item() == 0
    assert "uncalibrated_state_observer" in decision.fallback_reasons[0]


def test_promoted_observer_requires_content_addressed_state_contract() -> None:
    config = world_config()
    contract, calibration, deployment = observer_metadata(config, enabled=True)
    with pytest.raises(ValueError, match="state representation"):
        StateHiddenEventPredicateObserver(
            StateObserverConfig(
                state_input_dim=config.state_input_dim,
                hidden_dim=8,
                event_names=config.event_names,
                predicate_names=config.predicate_names,
            ),
            contract=contract,
            calibration=calibration,
            deployment=deployment,
            artifact_sha256="b" * 64,
        )


def test_manifest_binds_checkpoint_and_frozen_metadata(tmp_path: Path) -> None:
    config = world_config()
    observer = make_observer(config)
    payload = observer_artifact_payload(observer)
    checkpoint = tmp_path / "state_observer.pt"
    torch.save({**payload, "model": observer.state_dict()}, checkpoint)
    manifest = {
        **payload,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
    }
    manifest_path = tmp_path / "state_observer_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = StateHiddenEventPredicateObserver.from_manifest(manifest_path)
    assert loaded.artifact_sha256 == sha256(checkpoint)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

    manifest["calibration"]["event_temperature"] = 2.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration mismatch"):
        StateHiddenEventPredicateObserver.from_manifest(manifest_path)


def test_sealed_descriptors_do_not_cross_group_loader_boundary(monkeypatch) -> None:
    descriptors = [
        SimpleNamespace(logical_key="train"),
        SimpleNamespace(logical_key="validation"),
        SimpleNamespace(logical_key="sealed"),
    ]
    seen: list[str] = []

    def fake_loader(selected, *args, **kwargs):
        del args, kwargs
        seen.extend(row.logical_key for row in selected)
        return [SimpleNamespace(logical_key=row.logical_key) for row in selected]

    monkeypatch.setattr(trainer, "load_descriptor_groups", fake_loader)
    train_groups, validation_groups = trainer.load_observer_group_splits(
        descriptors=descriptors,
        splits={
            "train": ["train"],
            "validation": ["validation"],
            "test": ["sealed"],
        },
        world_config=None,
        object_names=["can"],
        body_to_id={"piper": 0},
        policy_to_id={"openvla": 0},
        calibrations={},
        event_spec_sha256=EVENT_SPEC_SHA,
    )
    assert seen == ["train", "validation"]
    assert [group.logical_key for group in train_groups] == ["train"]
    assert [group.logical_key for group in validation_groups] == ["validation"]


def test_cpu_supervised_training_emits_monitor_only_artifact() -> None:
    rng = np.random.default_rng(7)
    config = world_config()
    contract, _, _ = observer_metadata(config)
    train_hidden = rng.normal(size=(12, config.state_input_dim)).astype(np.float32)
    validation_hidden = rng.normal(size=(6, config.state_input_dim)).astype(np.float32)
    train_event = np.arange(12, dtype=np.int64) % config.num_events
    validation_event = np.arange(6, dtype=np.int64) % config.num_events
    train_predicates = (rng.random((12, config.num_predicates)) > 0.5).astype(
        np.float32
    )
    validation_predicates = (
        rng.random((6, config.num_predicates)) > 0.5
    ).astype(np.float32)
    observer, summary = trainer.train(
        train_hidden=train_hidden,
        train_event=train_event,
        train_predicates=train_predicates,
        validation_hidden=validation_hidden,
        validation_event=validation_event,
        validation_predicates=validation_predicates,
        config=StateObserverConfig(
            state_input_dim=config.state_input_dim,
            hidden_dim=8,
            event_names=config.event_names,
            predicate_names=config.predicate_names,
            dropout=0.0,
        ),
        contract=contract,
        seed=3,
        steps=2,
        batch_size=6,
        learning_rate=1e-3,
        weight_decay=0.0,
        eval_every=1,
        device=torch.device("cpu"),
    )
    assert summary["status"] == "complete"
    assert not observer.rerank_enabled
    assert observer.deployment["promotion_status"] == (
        "monitor_only_requires_independent_validation"
    )
    assert observer.calibration["selection_data"] == (
        "observer_validation_monitor_only_not_world_model_sealed_test"
    )
