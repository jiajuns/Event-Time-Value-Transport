from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from etsf_policy_feature_action_bridge import (  # noqa: E402
    PolicyBridgeContractError,
    build_policy_feature_action_bridge_contract,
    build_runtime_policy_bridge_binding,
    canonical_sha256,
    validate_policy_feature_action_bridge_contract,
    verify_checkpoint_file,
    verify_checkpoint_policy_bridge,
)
from example_openvla_event_critic_plugin import (  # noqa: E402
    OpenVLAActionAdapter,
    OpenVLAStateAdapter,
    verify_openvla_policy_bridge,
)
from example_smolvla_event_critic_adapter import (  # noqa: E402
    rerank_smolvla_candidates,
    verify_smolvla_policy_bridge,
)
from openvla_etsf_event_critic_plugin import (  # noqa: E402
    EmbodimentSpec,
    EventCriticPlugin,
    GuardConfig,
)
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


def bridge_fixture(policy: str, *, row: int = 0) -> tuple[dict, dict, dict]:
    state_dim = 960 if policy == "smolvla" else 4096
    state_adapter = "SmolVLAStateAdapter" if policy == "smolvla" else "OpenVLAStateAdapter"
    action_adapter = "SmolVLAActionAdapter" if policy == "smolvla" else "OpenVLAActionAdapter"
    bridge = build_policy_feature_action_bridge_contract(
        policy=policy,
        state_feature_source_sha256="1" * 64,
        policy_row=row,
    )
    config = {
        "state_input_dim": state_dim,
        "action_dim": 14,
        "num_policies": max(2, row + 1),
        "event_names": ["e0", "e12", "e3", "e4", "eK"],
        "structured_events": True,
        "predicate_names": ["moved", "lifted", "near_goal", "stationary", "success"],
        "relative_transition_names": ["stay", "advance", "skip", "regress"],
    }
    checkpoint_contract = {
        "policy_to_id": {policy: row},
        "policy_feature_action_bridge": bridge,
    }
    runtime = build_runtime_policy_bridge_binding(
        policy=policy,
        state_feature_source_sha256="1" * 64,
        state_feature_dimension=state_dim,
        state_adapter=state_adapter,
        action_adapter=action_adapter,
        policy_row=row,
    )
    return config, checkpoint_contract, runtime


@pytest.mark.parametrize("policy,state_dim", [("smolvla", 960), ("openvla", 4096)])
def test_exact_policy_bridges_share_only_canonical_interface(
    policy: str, state_dim: int
) -> None:
    config, contract, runtime = bridge_fixture(policy, row=1)
    receipt = verify_checkpoint_policy_bridge(
        config=config,
        checkpoint_contract=contract,
        expected_policy=policy,
        runtime_binding=runtime,
    )
    assert receipt["status"] == "verified_exact_policy_feature_action_bridge"
    assert receipt["state_feature_dimension"] == state_dim
    assert receipt["policy_row"] == 1
    assert receipt["cross_policy_latent_reuse_allowed"] is False


def test_direct_smolvla_960d_checkpoint_use_for_openvla_is_forbidden() -> None:
    smol_config, smol_contract, _ = bridge_fixture("smolvla")
    _, _, open_runtime = bridge_fixture("openvla")
    with pytest.raises(
        PolicyBridgeContractError,
        match="direct SmolVLA 960D checkpoint use for OpenVLA is forbidden",
    ):
        verify_checkpoint_policy_bridge(
            config=smol_config,
            checkpoint_contract=smol_contract,
            expected_policy="openvla",
            runtime_binding=open_runtime,
        )


@pytest.mark.parametrize("mismatch", ["state_source", "action_mapping"])
def test_runtime_provenance_or_mapping_mismatch_is_rejected(mismatch: str) -> None:
    config, contract, _ = bridge_fixture("openvla")
    kwargs = {
        "policy": "openvla",
        "state_feature_source_sha256": "1" * 64,
        "state_feature_dimension": 4096,
        "state_adapter": "OpenVLAStateAdapter",
        "action_adapter": "OpenVLAActionAdapter",
        "policy_row": 0,
        "model_slots": tuple(range(14)),
    }
    if mismatch == "state_source":
        kwargs["state_feature_source_sha256"] = "4" * 64
    elif mismatch == "action_mapping":
        kwargs["model_slots"] = tuple(reversed(range(14)))
    runtime = build_runtime_policy_bridge_binding(**kwargs)
    with pytest.raises(PolicyBridgeContractError, match="runtime state source"):
        verify_checkpoint_policy_bridge(
            config=config,
            checkpoint_contract=contract,
            expected_policy="openvla",
            runtime_binding=runtime,
        )


def test_adapter_name_and_actual_implementation_sha_are_fail_closed() -> None:
    with pytest.raises(PolicyBridgeContractError, match="local implementation bundle"):
        build_policy_feature_action_bridge_contract(
            policy="openvla",
            state_feature_source_sha256="1" * 64,
            state_adapter_sha256="2" * 64,
            policy_row=0,
        )
    with pytest.raises(PolicyBridgeContractError, match="adapter SHA256"):
        build_runtime_policy_bridge_binding(
            policy="smolvla",
            state_feature_source_sha256="1" * 64,
            state_feature_dimension=960,
            state_adapter="OpenVLAStateAdapter",
            action_adapter="OpenVLAActionAdapter",
            policy_row=0,
        )
    bridge = build_policy_feature_action_bridge_contract(
        policy="smolvla",
        state_feature_source_sha256="1" * 64,
        policy_row=0,
    )
    resigned = copy.deepcopy(bridge)
    resigned["state_feature"]["adapter"] = "OpenVLAStateAdapter"
    unsigned = dict(resigned)
    unsigned.pop("contract_sha256")
    resigned["contract_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(PolicyBridgeContractError, match="adapter SHA256"):
        validate_policy_feature_action_bridge_contract(resigned)


def test_policy_row_and_content_hash_are_fail_closed() -> None:
    config, contract, runtime = bridge_fixture("smolvla")
    wrong_row = copy.deepcopy(contract)
    wrong_row["policy_to_id"]["smolvla"] = 1
    with pytest.raises(PolicyBridgeContractError, match="policy row"):
        verify_checkpoint_policy_bridge(
            config=config,
            checkpoint_contract=wrong_row,
            expected_policy="smolvla",
            runtime_binding=runtime,
        )

    tampered = copy.deepcopy(contract)
    tampered["policy_feature_action_bridge"]["state_feature"]["source_sha256"] = "f" * 64
    with pytest.raises(PolicyBridgeContractError, match="contract SHA256 mismatch"):
        verify_checkpoint_policy_bridge(
            config=config,
            checkpoint_contract=tampered,
            expected_policy="smolvla",
            runtime_binding=runtime,
        )


def test_existing_openvla_and_smolvla_helpers_call_the_strict_verifier() -> None:
    open_config, open_contract, open_runtime = bridge_fixture("openvla")
    fake_plugin = SimpleNamespace(
        config=SimpleNamespace(to_dict=lambda: open_config),
        contract=open_contract,
    )
    fake_plugin.verify_policy_bridge = lambda *, expected_policy, runtime_binding: (
        verify_checkpoint_policy_bridge(
            config=open_config,
            checkpoint_contract=open_contract,
            expected_policy=expected_policy,
            runtime_binding=runtime_binding,
        )
    )
    assert verify_openvla_policy_bridge(fake_plugin, open_runtime)["policy"] == "openvla"

    smol_config, smol_contract, smol_runtime = bridge_fixture("smolvla")
    assert verify_smolvla_policy_bridge(
        checkpoint_config=smol_config,
        checkpoint_contract=smol_contract,
        runtime_bridge_binding=smol_runtime,
    )["policy"] == "smolvla"

    with pytest.raises(PolicyBridgeContractError, match="different policy"):
        verify_openvla_policy_bridge(fake_plugin, smol_runtime)


def test_reversible_interface_requires_structured_heads_and_exact_vocabularies() -> None:
    config, contract, runtime = bridge_fixture("openvla")
    unstructured = {**config, "structured_events": False}
    with pytest.raises(PolicyBridgeContractError, match="reversible event interface"):
        verify_checkpoint_policy_bridge(
            config=unstructured,
            checkpoint_contract=contract,
            expected_policy="openvla",
            runtime_binding=runtime,
        )
    wrong_predicates = {**config, "predicate_names": ["wrong"]}
    with pytest.raises(PolicyBridgeContractError, match="reversible event interface"):
        verify_checkpoint_policy_bridge(
            config=wrong_predicates,
            checkpoint_contract=contract,
            expected_policy="openvla",
            runtime_binding=runtime,
        )


def test_unstructured_checkpoint_is_explicitly_event_id_only() -> None:
    bridge = build_policy_feature_action_bridge_contract(
        policy="openvla",
        state_feature_source_sha256="1" * 64,
        policy_row=0,
        structured_events=False,
    )
    runtime = build_runtime_policy_bridge_binding(
        policy="openvla",
        state_feature_source_sha256="1" * 64,
        state_feature_dimension=4096,
        state_adapter="OpenVLAStateAdapter",
        action_adapter="OpenVLAActionAdapter",
        policy_row=0,
        structured_events=False,
    )
    receipt = verify_checkpoint_policy_bridge(
        config={
            "state_input_dim": 4096,
            "action_dim": 14,
            "num_policies": 1,
            "event_names": ["e0", "e12", "e3", "e4", "eK"],
            "structured_events": False,
        },
        checkpoint_contract={
            "policy_to_id": {"openvla": 0},
            "policy_feature_action_bridge": bridge,
        },
        expected_policy="openvla",
        runtime_binding=runtime,
    )
    assert receipt["canonical_event_interface"] == "canonical_event_id_only_v1"


def test_file_verifier_binds_checkpoint_and_runtime_files(tmp_path: Path) -> None:
    config, contract, runtime = bridge_fixture("openvla")
    checkpoint_path = tmp_path / "openvla_member.pt"
    runtime_path = tmp_path / "runtime_bridge.json"
    torch.save({"config": config, "contract": contract}, checkpoint_path)
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    receipt = verify_checkpoint_file(
        checkpoint_path=checkpoint_path,
        runtime_binding_path=runtime_path,
        expected_policy="openvla",
    )
    assert len(receipt["checkpoint_file_sha256"]) == 64
    assert len(receipt["runtime_binding_file_sha256"]) == 64


def calibrated_body() -> EmbodimentSpec:
    return EmbodimentSpec(
        name="piper",
        body_id=0,
        action_adapter_calibrated=True,
        clock_calibrated=True,
    )


def test_plugin_override_requires_its_own_verified_runtime_receipt() -> None:
    config = EventWorldModelConfig(
        state_input_dim=4096,
        action_dim=14,
        proprio_dim=14,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=10,
        clock_hidden_dim=4,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        structured_events=False,
        dropout=0,
    )
    bridge = build_policy_feature_action_bridge_contract(
        policy="openvla",
        state_feature_source_sha256="a" * 64,
        policy_row=0,
        structured_events=False,
    )
    contract = {
        "body_to_id": {"piper": 0},
        "policy_to_id": {"openvla": 0},
        "policy_feature_action_bridge": bridge,
    }
    runtime = build_runtime_policy_bridge_binding(
        policy="openvla",
        state_feature_source_sha256="a" * 64,
        state_feature_dimension=4096,
        state_adapter="OpenVLAStateAdapter",
        action_adapter="OpenVLAActionAdapter",
        policy_row=0,
        structured_events=False,
    )
    plugin = EventCriticPlugin(
        [ActionConditionedEventWorldModel(config) for _ in range(2)],
        contract=contract,
    )
    receipt = plugin.verify_policy_bridge(
        expected_policy="openvla", runtime_binding=runtime
    )
    state = OpenVLAStateAdapter(
        expected_dim=4096,
        calibrated=True,
        policy_bridge_verification_sha256=receipt["verification_sha256"],
        state_feature_binding_sha256=receipt["state_feature_binding_sha256"],
    ).adapt(
        torch.zeros(1, 1, 4096),
        current_event_id=torch.zeros(1, dtype=torch.long),
        proprio=torch.zeros(1, 14),
    )
    native = torch.zeros(1, 2, 1, 14)
    native[:, 1] = 0.01
    candidates = OpenVLAActionAdapter(
        policy_id=0,
        calibrated=True,
        policy_bridge_verification_sha256=receipt["verification_sha256"],
        action_mapping_binding_sha256=receipt["action_mapping_binding_sha256"],
    ).adapt(native, calibrated_body())
    prediction = plugin.predict(state, candidates, calibrated_body())
    assert prediction.policy_bridge_contract_matched
    prediction.score = torch.tensor([[0.0, 1.0]])
    prediction.total_uncertainty = torch.zeros(1, 2)
    decision = plugin.select(
        prediction,
        candidates,
        calibrated_body(),
        guard=GuardConfig(
            minimum_score_margin=0.1,
            maximum_candidate_distance=1.0,
            maximum_total_uncertainty=1.0,
        ),
    )
    assert decision.selected_index.tolist() == [1]

    candidates.action_mapping_binding_sha256 = "0" * 64
    rejected = plugin.select(
        prediction,
        candidates,
        calibrated_body(),
        guard=GuardConfig(
            minimum_score_margin=0.1,
            maximum_candidate_distance=1.0,
            maximum_total_uncertainty=1.0,
        ),
    )
    assert rejected.selected_index.tolist() == [0]
    assert "policy_feature_action_bridge_not_verified" in rejected.fallback_reasons[0]


def test_integrated_smolvla_rerank_reverifies_the_same_plugin() -> None:
    config = EventWorldModelConfig(
        state_input_dim=960,
        action_dim=14,
        proprio_dim=14,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=10,
        clock_hidden_dim=4,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        structured_events=False,
        dropout=0,
    )
    bridge = build_policy_feature_action_bridge_contract(
        policy="smolvla",
        state_feature_source_sha256="b" * 64,
        policy_row=0,
        structured_events=False,
    )
    state_base = {
        "policy": "smolvla",
        "anchor": "contextualized_vlm_prefix_final_state_token_before_flow_noise_v1",
        "source": "policy.model.vlm_with_expert.get_vlm_model().text_model.norm",
        "hidden_dim": 960,
        "prefix_length": 0,
        "noise_independence": "bit_exact_at_group_intervention_query",
        "modeling_sha256": "c" * 64,
        "bridge_sha256": "d" * 64,
    }
    import hashlib

    state_calibration = hashlib.sha256(
        json.dumps(
            state_base, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    contract = {
        "body_to_id": {"piper": 0},
        "policy_to_id": {"smolvla": 0},
        "policy_feature_action_bridge": bridge,
        "state_contracts": {
            "smolvla": {**state_base, "calibration_id": state_calibration}
        },
    }
    runtime = build_runtime_policy_bridge_binding(
        policy="smolvla",
        state_feature_source_sha256="b" * 64,
        state_feature_dimension=960,
        state_adapter="SmolVLAStateAdapter",
        action_adapter="SmolVLAActionAdapter",
        policy_row=0,
        structured_events=False,
    )
    plugin = EventCriticPlugin(
        [ActionConditionedEventWorldModel(config) for _ in range(2)],
        contract=contract,
    )
    original_predict = plugin.predict

    def deterministic_predict(*args, **kwargs):
        prediction = original_predict(*args, **kwargs)
        prediction.score = torch.tensor([[0.0, 1.0]])
        prediction.total_uncertainty = torch.zeros(1, 2)
        return prediction

    plugin.predict = deterministic_predict  # type: ignore[method-assign]
    native = torch.zeros(1, 2, 1, 14)
    native[:, 1] = 0.01
    decision = rerank_smolvla_candidates(
        plugin=plugin,
        runtime_bridge_binding=runtime,
        shared_state_history=torch.zeros(1, 1, 960),
        candidate_action_chunks=native,
        current_event_id=torch.zeros(1, dtype=torch.long),
        proprio=torch.zeros(1, 14),
        embodiment=calibrated_body(),
        state_adapter_calibrated=True,
        action_adapter_calibrated=True,
        state_calibration_id=state_calibration,
        guard=GuardConfig(
            minimum_score_margin=0.1,
            maximum_candidate_distance=1.0,
            maximum_total_uncertainty=1.0,
        ),
    )
    assert decision.selected_index.tolist() == [1]
    assert decision.prediction.policy_bridge_contract_matched
