from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from example_openvla_event_critic_plugin import OpenVLAActionAdapter  # noqa: E402
from example_smolvla_event_critic_adapter import (  # noqa: E402
    SmolVLAActionAdapter,
    SmolVLAStateAdapter,
    adapt_smolvla_query,
)
from etsf_policy_feature_action_bridge import (  # noqa: E402
    build_policy_feature_action_bridge_contract,
    build_runtime_policy_bridge_binding,
)
from openvla_etsf_event_critic_plugin import (  # noqa: E402
    EmbodimentSpec,
    EnsemblePrediction,
    EventCriticPlugin,
    GuardConfig,
    HistoryState,
    IdentityStateAdapter,
    ScoringConfig,
)
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


def make_config() -> EventWorldModelConfig:
    return EventWorldModelConfig(
        state_input_dim=16,
        action_dim=6,
        proprio_dim=4,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=6,
        object_delta_dim=5,
        num_bodies=2,
        num_policies=3,
        metadata_dim=4,
        dropout=0.0,
    )


def make_structured_plugin() -> EventCriticPlugin:
    values = make_config().to_dict()
    values["structured_events"] = True
    config = EventWorldModelConfig.from_dict(values)
    models = []
    for seed in range(3):
        torch.manual_seed(90 + seed)
        models.append(ActionConditionedEventWorldModel(config))
    predicate_contract = {
        "names": list(config.predicate_names),
        "derivation": "derive_atomic_predicates_v1",
        "source": "simulator_object_poses_at_query_step",
        "event_spec_sha256": "event-spec-sha",
        "calibration_id": "event-spec-sha",
        "task_calibration": {"move_can_pot": "frozen"},
        "online_requires_explicit_predicates": True,
        "missing_policy": "error",
    }
    return EventCriticPlugin(
        models,
        contract={
            "body_to_id": {"piper": 0},
            "policy_to_id": {"openvla": 0},
            "predicate_contract": predicate_contract,
        },
    )


def save_ensemble(tmp_path: Path, *, mismatch: bool = False) -> list[Path]:
    paths = []
    config = make_config()
    for seed in range(3):
        torch.manual_seed(20 + seed)
        model = ActionConditionedEventWorldModel(config)
        # Give the ensemble a controlled, non-zero success disagreement.
        with torch.no_grad():
            model.success_head.bias.fill_(float(seed - 1))
        contract = {
            "cache_schema": "unit-test",
            "source_manifest_sha256": "abc",
            "events": list(config.event_names),
            "object_names": ["can"],
            "object_target": "all_object_position_deltas_xyz",
            "body_to_id": {"piper": 0},
            "policy_to_id": {"openvla": 0},
        }
        if mismatch and seed == 2:
            contract["events"] = ["different"]
        path = tmp_path / f"seed_{seed}.pt"
        torch.save(
            {"model": model.state_dict(), "config": model.config_dict(), "contract": contract},
            path,
        )
        paths.append(path)
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def smolvla_state_contract(hidden_dim: int) -> dict[str, object]:
    base: dict[str, object] = {
        "policy": "smolvla",
        "anchor": "contextualized_vlm_prefix_final_state_token_before_flow_noise_v1",
        "source": "policy.model.vlm_with_expert.get_vlm_model().text_model.norm",
        "hidden_dim": hidden_dim,
        "prefix_length": 0,
        "noise_independence": "bit_exact_at_group_intervention_query",
        "modeling_sha256": "0" * 64,
        "bridge_sha256": "1" * 64,
    }
    calibration_id = hashlib.sha256(
        json.dumps(
            base, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return {**base, "calibration_id": calibration_id}


def strict_smolvla_bridge(policy_row: int = 1) -> tuple[dict, dict, dict]:
    bridge = build_policy_feature_action_bridge_contract(
        policy="smolvla",
        state_feature_source_sha256="2" * 64,
        policy_row=policy_row,
    )
    config = {
        "state_input_dim": 960,
        "action_dim": 14,
        "num_policies": max(2, policy_row + 1),
        "event_names": ["e0", "e12", "e3", "e4", "eK"],
        "structured_events": True,
        "predicate_names": ["moved", "lifted", "near_goal", "stationary", "success"],
        "relative_transition_names": ["stay", "advance", "skip", "regress"],
    }
    contract = {
        "policy_to_id": {"smolvla": policy_row},
        "policy_feature_action_bridge": bridge,
    }
    runtime = build_runtime_policy_bridge_binding(
        policy="smolvla",
        state_feature_source_sha256="2" * 64,
        state_feature_dimension=960,
        state_adapter="SmolVLAStateAdapter",
        action_adapter="SmolVLAActionAdapter",
        policy_row=policy_row,
    )
    return config, contract, runtime


def save_counterfactual_manifest(
    tmp_path: Path, *, guard_enabled: bool = True
) -> Path:
    member_paths = save_ensemble(tmp_path)
    members = [torch.load(path, map_location="cpu", weights_only=False) for path in member_paths]
    config = members[0]["config"]
    contract = members[0]["contract"]
    normalization_tensor = {
        "object_delta_mean": torch.zeros(5),
        "object_delta_std": torch.ones(5),
    }
    calibration = {"temperature": 2.0, "status": "fitted", "candidates": 24}
    guard = {
        "enabled": guard_enabled,
        "gain_margin": 0.2 if guard_enabled else None,
        "uncertainty_threshold": 0.4 if guard_enabled else None,
    }
    scoring = {
        "candidate_id": "full",
        "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "event_weight": 0.25,
        "duration_weight": 0.05,
        "candidate_distance_weight": 0.1,
        "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
    }
    scoring_selection = {
        "grid_version": "validation_scoring_grid_v1",
        "selection_rule": "fixed_validation_only_test_rule",
        "selected_candidate_id": "full",
        "candidates": [
            {
                "candidate_id": "full",
                "event_weight": 0.25,
                "duration_weight": 0.05,
                "candidate_distance_weight": 0.1,
            }
        ],
    }
    payload = {
        "format": "etsf_counterfactual_ensemble_v1",
        "models": [member["model"] for member in members],
        "member_seeds": [1, 2, 3],
        "config": config,
        "contract": contract,
        "normalization": normalization_tensor,
        "duration_scale": 20.0,
        "success_calibration": calibration,
        "guard": guard,
        "scoring": scoring,
        "scoring_selection": scoring_selection,
    }
    ensemble_path = tmp_path / "counterfactual_ensemble.pt"
    torch.save(payload, ensemble_path)
    manifest = {
        "format": payload["format"],
        "ensemble_checkpoint": {
            "path": str(ensemble_path),
            "sha256": file_sha256(ensemble_path),
        },
        "members": [
            {"path": str(path), "sha256": file_sha256(path), "seed": seed}
            for path, seed in zip(member_paths, [1, 2, 3])
        ],
        "config": config,
        "contract": contract,
        "normalization": {
            "object_delta_mean": [0.0] * 5,
            "object_delta_std": [1.0] * 5,
        },
        "duration_scale": 20.0,
        "success_calibration": calibration,
        "guard": guard,
        "scoring": scoring,
        "scoring_selection": scoring_selection,
    }
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def calibrated_piper() -> EmbodimentSpec:
    return EmbodimentSpec(
        name="piper",
        body_id=0,
        beta=0.1,
        action_contract="openvla_14d",
        action_adapter_calibrated=True,
        clock_calibrated=True,
        calibration_id="piper-cal-v1",
    )


def make_inputs(plugin: EventCriticPlugin):
    batch, candidates, horizon = 2, 3, 4
    state = HistoryState(
        hidden=torch.randn(batch, 3, plugin.config.state_input_dim),
        history_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        current_event_id=torch.tensor([0, 1]),
        proprio=torch.randn(batch, plugin.config.proprio_dim),
        policy_name="openvla",
        adapter_name="OpenVLAStateAdapter",
        adapter_calibrated=True,
        calibration_id="openvla-state-v1",
    )
    adapter = OpenVLAActionAdapter(
        policy_id=0,
        native_action_dim=4,
        model_action_dim=plugin.config.action_dim,
        model_slots=(0, 2, 3, 5),
        calibrated=True,
        calibration_id="openvla-piper-v1",
    )
    native = torch.randn(batch, candidates, horizon, 4)
    batch_candidates = adapter.adapt(native, calibrated_piper())
    return state, batch_candidates, native


def fake_prediction(scores: torch.Tensor, uncertainty: torch.Tensor) -> EnsemblePrediction:
    zeros = torch.zeros_like(scores)
    return EnsemblePrediction(
        outputs={},
        member_count=3,
        base_score=scores,
        score=scores,
        aleatoric_uncertainty=zeros,
        epistemic_uncertainty=uncertainty,
        total_uncertainty=uncertainty,
        state_adapter_name="OpenVLAStateAdapter",
        state_adapter_calibrated=True,
        state_policy_name="openvla",
        state_calibration_id="openvla-state-v1",
        state_contract_matched=True,
    )


def test_load_three_checkpoints_and_predict_on_cpu(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_checkpoints(save_ensemble(tmp_path), device="cpu")
    assert len(plugin.models) == 3
    assert all(not model.training for model in plugin.models)
    assert all(not parameter.requires_grad for model in plugin.models for parameter in model.parameters())
    state, candidates, _ = make_inputs(plugin)
    prediction = plugin.predict(
        state,
        candidates,
        calibrated_piper(),
        scoring=ScoringConfig(uncertainty_weight=0.2, distance_weight=0.1),
    )
    assert prediction.member_count == 3
    assert prediction.score.shape == (2, 3)
    assert prediction.outputs["next_event_probability"].shape == (2, 3, 5)
    assert prediction.outputs["object_delta_mean"].shape == (2, 3, 5)
    assert torch.isfinite(prediction.score).all()
    assert bool((prediction.aleatoric_uncertainty >= 0).all())
    assert bool((prediction.epistemic_uncertainty > 0).all())
    assert torch.allclose(
        prediction.total_uncertainty,
        prediction.aleatoric_uncertainty + prediction.epistemic_uncertainty,
    )


def test_checkpoint_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contract mismatch"):
        EventCriticPlugin.from_checkpoints(save_ensemble(tmp_path, mismatch=True))


def test_openvla_adapter_preserves_execution_actions_and_masks_padding(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_checkpoints(save_ensemble(tmp_path))
    _, candidates, native = make_inputs(plugin)
    assert candidates.actions.shape == (2, 3, 4, 6)
    assert torch.equal(candidates.execution_actions, native)
    assert candidates.action_feature_mask is not None
    assert candidates.action_feature_mask[0, 0, 0].tolist() == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert candidates.candidate_distance is not None
    assert torch.equal(candidates.candidate_distance[:, 0], torch.zeros(2))


def test_legacy_checkpoint_is_monitor_only_even_with_safe_margin(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_checkpoints(save_ensemble(tmp_path))
    state, candidates, native = make_inputs(plugin)
    candidates.candidate_distance = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])
    prediction = fake_prediction(
        torch.tensor([[0.1, 0.9, 0.2], [0.1, 0.2, 0.8]]),
        torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
    )
    plugin.predict = lambda *args, **kwargs: prediction  # type: ignore[method-assign]
    decision = plugin.rerank(
        state,
        candidates,
        calibrated_piper(),
        guard=GuardConfig(
            minimum_score_margin=0.2,
            maximum_candidate_distance=0.25,
            maximum_total_uncertainty=0.5,
            require_calibrated_adapters=False,
            require_registered_contract=False,
        ),
    )
    assert decision.selected_index.tolist() == [0, 0]
    assert decision.changed_from_actor.tolist() == [False, False]
    assert all(
        "policy_feature_action_bridge_not_verified" in reasons
        for reasons in decision.fallback_reasons
    )
    assert torch.equal(decision.selected_execution_actions, native[:, 0])


def test_each_guard_can_force_actor_fallback(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_checkpoints(save_ensemble(tmp_path))
    _, candidates, native = make_inputs(plugin)
    candidates.candidate_distance = torch.tensor([[0.0, 0.4, 0.1], [0.0, 0.1, 0.1]])
    prediction = fake_prediction(
        torch.tensor([[0.2, 0.9, 0.1], [0.2, 0.9, 0.1]]),
        torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.9, 0.1]]),
    )
    decision = plugin.select(
        prediction,
        candidates,
        calibrated_piper(),
        guard=GuardConfig(
            minimum_score_margin=0.5,
            maximum_candidate_distance=0.25,
            maximum_total_uncertainty=0.5,
        ),
    )
    assert decision.selected_index.tolist() == [0, 0]
    assert decision.guard_fallback_used.tolist() == [True, True]
    assert "candidate_distance_above_guard" in decision.fallback_reasons[0]
    assert "uncertainty_above_guard" in decision.fallback_reasons[1]
    assert torch.equal(decision.selected_execution_actions, native[:, 0])


def test_uncalibrated_or_unregistered_transfer_is_monitor_only(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_checkpoints(save_ensemble(tmp_path))
    _, candidates, _ = make_inputs(plugin)
    candidates.adapter_calibrated = False
    candidates.policy_name = "new_policy"
    candidates.candidate_distance = torch.tensor([[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]])
    uncalibrated_body = EmbodimentSpec(name="new_robot", body_id=1)
    prediction = fake_prediction(
        torch.tensor([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1]]),
        torch.full((2, 3), 0.1),
    )
    decision = plugin.select(prediction, candidates, uncalibrated_body)
    assert decision.selected_index.tolist() == [0, 0]
    for reasons in decision.fallback_reasons:
        assert "uncalibrated_policy_adapter" in reasons
        assert "uncalibrated_embodiment_action_adapter" in reasons
        assert "uncalibrated_embodiment_clock" in reasons
        assert "policy_not_registered_in_checkpoint" in reasons
        assert "embodiment_not_registered_in_checkpoint" in reasons


def test_counterfactual_manifest_loads_frozen_scoring_and_uncertainty(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_manifest(save_counterfactual_manifest(tmp_path))
    assert plugin.counterfactual_deployment is not None
    assert plugin.counterfactual_deployment.success_temperature == 2.0
    state, candidates, _ = make_inputs(plugin)
    prediction = plugin.predict(state, candidates, calibrated_piper())
    assert torch.equal(
        prediction.epistemic_uncertainty,
        prediction.outputs["epistemic_success_std"],
    )
    assert torch.allclose(
        prediction.total_uncertainty,
        prediction.outputs["epistemic_success_std"]
        + prediction.aleatoric_uncertainty,
    )
    with pytest.raises(ValueError, match="frozen scoring contract"):
        plugin.predict(
            state,
            candidates,
            calibrated_piper(),
            scoring=ScoringConfig(),
        )


def test_counterfactual_manifest_guard_is_mandatory(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_manifest(save_counterfactual_manifest(tmp_path))
    _, candidates, native = make_inputs(plugin)
    candidates.candidate_distance = torch.tensor([[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]])
    safe = fake_prediction(
        torch.tensor([[0.0, 0.8, 0.1], [0.0, 0.8, 0.1]]),
        torch.full((2, 3), 0.1),
    )
    decision = plugin.select(safe, candidates, calibrated_piper())
    assert decision.selected_index.tolist() == [0, 0]
    assert torch.equal(decision.selected_execution_actions, native[:, 0])
    assert all(
        "policy_feature_action_bridge_not_verified" in row
        for row in decision.fallback_reasons
    )

    unsafe = fake_prediction(
        safe.score,
        torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]),
    )
    fallback = plugin.select(unsafe, candidates, calibrated_piper())
    assert fallback.selected_index.tolist() == [0, 0]
    assert all("uncertainty_above_guard" in row for row in fallback.fallback_reasons)
    with pytest.raises(ValueError, match="frozen guard"):
        plugin.select(safe, candidates, calibrated_piper(), guard=GuardConfig())


def test_disabled_manifest_guard_never_replaces_actor(tmp_path: Path) -> None:
    plugin = EventCriticPlugin.from_manifest(
        save_counterfactual_manifest(tmp_path, guard_enabled=False)
    )
    _, candidates, _ = make_inputs(plugin)
    candidates.candidate_distance = torch.tensor([[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]])
    prediction = fake_prediction(
        torch.tensor([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1]]),
        torch.zeros(2, 3),
    )
    decision = plugin.select(prediction, candidates, calibrated_piper())
    assert decision.selected_index.tolist() == [0, 0]
    assert all("manifest_guard_disabled" in row for row in decision.fallback_reasons)


def test_counterfactual_manifest_sha256_is_verified(tmp_path: Path) -> None:
    path = save_counterfactual_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["ensemble_checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        EventCriticPlugin.from_manifest(path)


def test_counterfactual_manifest_scoring_cannot_be_tampered_independently(
    tmp_path: Path,
) -> None:
    path = save_counterfactual_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["scoring"]["event_weight"] = 99.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scoring mismatch"):
        EventCriticPlugin.from_manifest(path)


def test_counterfactual_scoring_selection_audit_cannot_be_tampered_independently(
    tmp_path: Path,
) -> None:
    path = save_counterfactual_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["scoring_selection"]["selected_candidate_id"] = "success_only"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scoring selection mismatch"):
        EventCriticPlugin.from_manifest(path)


def test_state_adapter_rejects_cross_policy_dimension_shortcut() -> None:
    adapter = IdentityStateAdapter(
        expected_dim=16,
        name="SmolVLAStateAdapter",
        policy_name="smolvla",
        calibrated=False,
    )
    with pytest.raises(ValueError, match="does not match checkpoint contract"):
        adapter.adapt(
            torch.randn(2, 3, 720),
            current_event_id=torch.zeros(2, dtype=torch.long),
        )


def test_smolvla_adapter_requires_shared_state_matching_its_checkpoint() -> None:
    body = calibrated_piper()
    native_actions = torch.randn(2, 4, 5, 14)
    config, contract, runtime = strict_smolvla_bridge()
    with pytest.raises(ValueError, match="does not match checkpoint contract"):
        adapt_smolvla_query(
            shared_state_history=torch.randn(2, 3, 720),
            candidate_action_chunks=native_actions,
            current_event_id=torch.zeros(2, dtype=torch.long),
            proprio=torch.randn(2, 4),
            embodiment=body,
            checkpoint_config=config,
            checkpoint_contract=contract,
            runtime_bridge_binding=runtime,
            policy_id=1,
        )

    state, candidates = adapt_smolvla_query(
        shared_state_history=torch.randn(2, 3, 960),
        candidate_action_chunks=native_actions,
        current_event_id=torch.zeros(2, dtype=torch.long),
        proprio=torch.randn(2, 4),
        embodiment=body,
        checkpoint_config=config,
        checkpoint_contract=contract,
        runtime_bridge_binding=runtime,
        policy_id=1,
        state_adapter_calibrated=True,
        action_adapter_calibrated=True,
    )
    assert state.hidden.shape == (2, 3, 960)
    assert state.policy_name == "smolvla"
    assert state.adapter_calibrated
    assert torch.equal(candidates.execution_actions, native_actions)
    assert candidates.policy_name == "smolvla"


def test_smolvla_plugin_binds_adapter_to_content_addressed_state_contract() -> None:
    config = make_config()
    models = [ActionConditionedEventWorldModel(config) for _ in range(2)]
    contract = smolvla_state_contract(config.state_input_dim)
    plugin = EventCriticPlugin(
        models,
        contract={
            "body_to_id": {"piper": 0},
            "policy_to_id": {"smolvla": 0},
            "state_contracts": {"smolvla": contract},
        },
    )
    native_actions = torch.randn(2, 3, 4, config.action_dim)
    state = SmolVLAStateAdapter(
        expected_dim=config.state_input_dim,
        calibrated=True,
        calibration_id=str(contract["calibration_id"]),
    ).adapt(
        torch.randn(2, 3, config.state_input_dim),
        current_event_id=torch.zeros(2, dtype=torch.long),
        proprio=torch.randn(2, config.proprio_dim),
    )
    candidates = SmolVLAActionAdapter(
        policy_id=0,
        native_action_dim=config.action_dim,
        model_action_dim=config.action_dim,
        calibrated=True,
    ).adapt(native_actions, calibrated_piper())
    prediction = plugin.predict(state, candidates, calibrated_piper())
    assert prediction.state_contract_matched

    state.calibration_id = "not-the-content-addressed-contract"
    mismatch = plugin.predict(state, candidates, calibrated_piper())
    assert not mismatch.state_contract_matched
    mismatch.score = torch.tensor([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1]])
    mismatch.total_uncertainty = torch.full((2, 3), 0.1)
    candidates.candidate_distance = torch.tensor(
        [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]]
    )
    decision = plugin.select(
        mismatch,
        candidates,
        calibrated_piper(),
        guard=GuardConfig(
            minimum_score_margin=0.1,
            maximum_candidate_distance=0.5,
            maximum_total_uncertainty=0.5,
        ),
    )
    assert decision.selected_index.tolist() == [0, 0]
    assert all("state_contract_mismatch" in row for row in decision.fallback_reasons)


def test_registered_smolvla_policy_requires_frozen_state_contract() -> None:
    config = make_config()
    models = [ActionConditionedEventWorldModel(config) for _ in range(2)]
    with pytest.raises(ValueError, match="lacks a frozen shared-state contract"):
        EventCriticPlugin(models, contract={"policy_to_id": {"smolvla": 0}})


def test_tampered_smolvla_state_source_hash_is_rejected() -> None:
    config = make_config()
    models = [ActionConditionedEventWorldModel(config) for _ in range(2)]
    contract = smolvla_state_contract(config.state_input_dim)
    contract["modeling_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not bind its content"):
        EventCriticPlugin(
            models,
            contract={
                "policy_to_id": {"smolvla": 0},
                "state_contracts": {"smolvla": contract},
            },
        )


def test_structured_plugin_rejects_missing_or_malformed_predicates() -> None:
    plugin = make_structured_plugin()
    state, candidates, _ = make_inputs(plugin)
    with pytest.raises(ValueError, match="requires explicit current_predicates"):
        plugin.predict(state, candidates, calibrated_piper())
    state.current_predicates = torch.zeros(2, plugin.config.num_predicates - 1)
    state.predicate_source = "derive_atomic_predicates_v1"
    with pytest.raises(ValueError, match="current_predicates must have shape"):
        plugin.predict(state, candidates, calibrated_piper())


def test_structured_metadata_passes_predicates_and_explicit_clock_event() -> None:
    plugin = make_structured_plugin()
    state, candidates, _ = make_inputs(plugin)
    state.current_predicates = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=torch.float32
    )
    state.predicate_source = "derive_atomic_predicates_v1"
    state.predicate_calibrated = True
    state.predicate_calibration_id = "event-spec-sha"
    metadata = plugin._metadata(state, candidates, calibrated_piper())
    assert torch.equal(metadata["clock_event_id"], state.current_event_id)
    assert torch.equal(metadata["current_predicates"], state.current_predicates)
    prediction = plugin.predict(state, candidates, calibrated_piper())
    assert prediction.predicate_contract_matched
    assert "post_predicate_probability" in prediction.outputs


def test_structured_guard_falls_back_on_predicate_contract_mismatch() -> None:
    plugin = make_structured_plugin()
    _, candidates, _ = make_inputs(plugin)
    candidates.candidate_distance = torch.tensor(
        [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]]
    )
    prediction = fake_prediction(
        torch.tensor([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1]]),
        torch.full((2, 3), 0.1),
    )
    prediction.predicate_input_required = True
    prediction.predicate_calibrated = True
    prediction.predicate_contract_matched = False
    decision = plugin.select(
        prediction,
        candidates,
        calibrated_piper(),
        guard=GuardConfig(
            minimum_score_margin=0.1,
            maximum_candidate_distance=0.5,
            maximum_total_uncertainty=0.5,
        ),
    )
    assert decision.selected_index.tolist() == [0, 0]
    assert all("predicate_contract_mismatch" in row for row in decision.fallback_reasons)


def test_structured_manifest_requires_predicate_derivation_contract(tmp_path: Path) -> None:
    manifest_path = save_counterfactual_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    ensemble_path = Path(manifest["ensemble_checkpoint"]["path"])
    checkpoint = torch.load(ensemble_path, map_location="cpu", weights_only=False)
    manifest["config"]["structured_events"] = True
    checkpoint["config"]["structured_events"] = True
    torch.save(checkpoint, ensemble_path)
    manifest["ensemble_checkpoint"]["sha256"] = file_sha256(ensemble_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predicate derivation/calibration contract"):
        EventCriticPlugin.from_manifest(manifest_path)
