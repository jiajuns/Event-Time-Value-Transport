from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from smolvla_piper_causal_event_observer_v1 import (  # noqa: E402
    MAX_HISTORY_STEPS,
    STATE_DIM,
    ActorVisibleCausalEventObserverV1,
    CausalObserverConfig,
    CausalObserverContractError,
    EVALUATION400_V4_TARGET,
    EmbodimentResidualAdapter,
    PROMOTION_EVIDENCE_FORMAT,
    build_causal_history_window,
    canonical_sha256,
    causal_history_contract,
    make_actor_adapter_contract,
    make_calibration,
    make_deployment,
    make_execution_receipt,
    make_image_receipt,
    make_input_receipt,
    training_supervision_contract,
    validate_promotion_evidence,
)


CORE_SHA = "a" * 64
EVENT_SPEC_SHA = "b" * 64
CALIBRATION_SPLIT_SHA = "c" * 64
DATASET_MANIFEST_SHA = "6" * 64
VALIDATION_SPLIT_SHA = "5" * 64
CHECKPOINT_SHA = "1" * 64
CONFIG_SHA = "2" * 64
ADAPTER_SET_SHA = "3" * 64
ADAPTER_CHECKPOINT_SET_SHA = "4" * 64
ACTORS = (
    {
        "actor_name": "piper",
        "policy_family": "smolvla",
        "state_feature_source_sha256": "d" * 64,
    },
    {
        "actor_name": "new_body",
        "policy_family": "smolvla",
        "state_feature_source_sha256": "e" * 64,
    },
)


def _promotion_evidence(training_sha: str, calibration_sha: str) -> dict:
    value = {
        "format": PROMOTION_EVIDENCE_FORMAT,
        "status": "independent_validation_passed_all_gates",
        "observer_core_file_sha256": CORE_SHA,
        "observer_checkpoint_file_sha256": CHECKPOINT_SHA,
        "observer_config_sha256": CONFIG_SHA,
        "training_supervision_contract_sha256": training_sha,
        "actor_adapter_set_sha256": ADAPTER_SET_SHA,
        "actor_adapter_checkpoint_set_sha256": ADAPTER_CHECKPOINT_SET_SHA,
        "calibration_sha256": calibration_sha,
        "independent_calibration_split_sha256": CALIBRATION_SPLIT_SHA,
        "independent_validation_split_sha256": VALIDATION_SPLIT_SHA,
        "actor_names": ["piper", "new_body"],
        "independent_validation_groups": 80,
        "per_actor_validation_groups": {"piper": 40, "new_body": 40},
        "event_macro_accuracy_lcb95": 0.80,
        "event_macro_f1_lcb95": 0.78,
        "predicate_macro_f1_lcb95": 0.75,
        "event_predicate_ontology_consistency": 0.97,
        "event_macro_f1_gain_over_train_frequency_lcb95": 0.10,
        "predicate_macro_f1_gain_over_train_constant_lcb95": 0.08,
        "maximum_event_ece": 0.05,
        "maximum_predicate_ece": 0.05,
        "low_confidence_false_accept_ucb95": 0.02,
        "future_feature_perturbation_invariant": True,
        "cross_branch_isolation_passed": True,
        "privileged_input_static_audit_passed": True,
        "calibration_group_disjoint": True,
    }
    return {**value, "promotion_receipt_sha256": canonical_sha256(value)}


def _observer(
    *,
    promoted: bool = False,
    minimum_confidence: float = 0.90,
    image_feature_dim: int = 0,
) -> ActorVisibleCausalEventObserverV1:
    torch.manual_seed(17)
    config = CausalObserverConfig(
        hidden_dim=24,
        adapter_rank=3,
        image_feature_dim=image_feature_dim,
    )
    training = training_supervision_contract(
        event_spec_sha256=EVENT_SPEC_SHA,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA,
        actor_registry=ACTORS,
    )
    states = {}
    contracts = {}
    for index, record in enumerate(ACTORS):
        adapter = EmbodimentResidualAdapter(config.hidden_dim, config.adapter_rank)
        if index == 1:
            with torch.no_grad():
                adapter.down.weight.fill_(0.05)
                adapter.up.weight.fill_(0.03)
        states[record["actor_name"]] = {
            name: tensor.detach().clone()
            for name, tensor in adapter.state_dict().items()
        }
        contracts[record["actor_name"]] = make_actor_adapter_contract(
            actor_name=record["actor_name"],
            policy_family=record["policy_family"],
            state_feature_source_sha256=record["state_feature_source_sha256"],
            observer_core_file_sha256=CORE_SHA,
            training_contract_sha256=training["contract_sha256"],
            image_feature_extractor_file_sha256=(
                "f" * 64 if image_feature_dim > 0 else None
            ),
            config=config,
            adapter_state=states[record["actor_name"]],
        )
    calibration = make_calibration(
        event_spec_sha256=EVENT_SPEC_SHA,
        independent_calibration_split_sha256=CALIBRATION_SPLIT_SHA,
        minimum_joint_confidence=minimum_confidence,
    )
    evidence = (
        _promotion_evidence(
            training["contract_sha256"], calibration["calibration_sha256"]
        )
        if promoted else None
    )
    observer = ActorVisibleCausalEventObserverV1(
        config,
        training_contract=training,
        observer_core_file_sha256=CORE_SHA,
        observer_checkpoint_file_sha256=CHECKPOINT_SHA,
        observer_config_sha256=CONFIG_SHA,
        actor_adapter_set_sha256=ADAPTER_SET_SHA,
        actor_adapter_checkpoint_set_sha256=ADAPTER_CHECKPOINT_SET_SHA,
        adapter_contracts=contracts,
        adapter_states=states,
        calibration=calibration,
        deployment=make_deployment(
            promotion_enabled=promoted,
            promotion_evidence=evidence,
            integration_target=(
                EVALUATION400_V4_TARGET if promoted else "monitor_only"
            ),
            promotion_validation_context=(
                {
                    "observer_core_file_sha256": CORE_SHA,
                    "observer_checkpoint_file_sha256": CHECKPOINT_SHA,
                    "observer_config_sha256": CONFIG_SHA,
                    "training_contract_sha256": training["contract_sha256"],
                    "actor_adapter_set_sha256": ADAPTER_SET_SHA,
                    "actor_adapter_checkpoint_set_sha256": (
                        ADAPTER_CHECKPOINT_SET_SHA
                    ),
                    "calibration_sha256": calibration["calibration_sha256"],
                    "actor_names": ["piper", "new_body"],
                }
                if promoted else None
            ),
        ),
    ).eval()
    return observer


def _batch(
    observer: ActorVisibleCausalEventObserverV1,
    *,
    actors: tuple[str, ...] = ("piper",),
    query_index: int = 1,
    with_image: bool = False,
    with_execution: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict], torch.Tensor | None]:
    generator = np.random.default_rng(23)
    valid = min(query_index + 1, MAX_HISTORY_STEPS)
    prefix = generator.normal(size=(query_index + 1, STATE_DIM)).astype(np.float32)
    history_np, mask_np = build_causal_history_window(prefix)
    history = torch.from_numpy(np.repeat(history_np[None], len(actors), axis=0))
    mask = torch.from_numpy(np.repeat(mask_np[None], len(actors), axis=0))
    proprio = torch.from_numpy(
        generator.normal(size=(len(actors), observer.config.proprio_dim)).astype(np.float32)
    )
    image = None
    if observer.config.image_feature_dim > 0:
        image = torch.zeros(len(actors), observer.config.image_feature_dim)
        if with_image:
            image.copy_(
                torch.from_numpy(
                    generator.normal(size=image.shape).astype(np.float32)
                )
            )
    receipts = []
    by_name = {item["actor_name"]: item for item in ACTORS}
    for row, actor in enumerate(actors):
        image_receipt = (
            make_image_receipt(
                image[row],
                extractor_file_sha256="f" * 64,
                frame_query_index=query_index,
            )
            if with_image and image is not None
            else None
        )
        execution_receipt = (
            make_execution_receipt(
                action_sha256="1" * 64,
                executed_control_steps=1,
                last_completed_query_index=query_index - 1,
                current_query_index=query_index,
            )
            if with_execution
            else None
        )
        record = by_name[actor]
        receipts.append(
            make_input_receipt(
                history=history[row],
                history_mask=mask[row],
                proprio=proprio[row],
                actor_name=actor,
                policy_family=record["policy_family"],
                state_feature_source_sha256=record["state_feature_source_sha256"],
                current_query_index=query_index,
                valid_history_steps=valid,
                image_feature_receipt=image_receipt,
                execution_receipt=execution_receipt,
            )
        )
    return history, mask, proprio, receipts, image


def _resign(receipt: dict, field: str, value) -> dict:
    changed = copy.deepcopy(receipt)
    changed[field] = value
    logical = dict(changed)
    logical.pop("receipt_sha256")
    changed["receipt_sha256"] = canonical_sha256(logical)
    return changed


def test_causal_window_is_future_invariant_padded_truncated_and_branch_local() -> None:
    first = np.arange(5 * STATE_DIM, dtype=np.float32).reshape(5, STATE_DIM)
    changed_future = np.concatenate(
        [first, np.full((3, STATE_DIM), 999, dtype=np.float32)], axis=0
    )
    other = first + np.float32(10_000)
    history, mask = build_causal_history_window(first)
    same_prefix, same_mask = build_causal_history_window(changed_future[:5])
    other_history, _ = build_causal_history_window(other)
    assert np.array_equal(history, same_prefix)
    assert np.array_equal(mask, same_mask)
    assert mask.tolist() == [True] * 5 + [False] * 3
    assert np.array_equal(history[5:], np.zeros((3, STATE_DIM), np.float32))
    assert not np.array_equal(history[:5], other_history[:5])

    long = np.arange(12 * STATE_DIM, dtype=np.float32).reshape(12, STATE_DIM)
    truncated, full_mask = build_causal_history_window(long)
    assert full_mask.all()
    assert np.array_equal(truncated, long[-MAX_HISTORY_STEPS:])


def test_training_contract_is_non_privileged_and_4096d_is_rejected() -> None:
    contract = training_supervision_contract(
        event_spec_sha256=EVENT_SPEC_SHA,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA,
        actor_registry=ACTORS,
    )
    assert contract["privileged_label_source_available_to_model_inputs"] is False
    assert contract["future_query_features_available_to_model_inputs"] is False
    assert "object_poses" in contract["forbidden_online_input_fields"]
    assert contract["history_contract"] == causal_history_contract()
    with pytest.raises(ValueError, match="960-D/8-step"):
        CausalObserverConfig(state_input_dim=4096)


def test_monitor_only_and_low_confidence_both_fail_closed() -> None:
    monitor = _observer(promoted=False, minimum_confidence=0.0)
    history, mask, proprio, receipts, image = _batch(monitor)
    with torch.no_grad():
        monitor.event_head.weight.zero_()
        monitor.event_head.bias[0] = 20
        monitor.predicate_head.weight.zero_()
        monitor.predicate_head.bias.fill_(20)
    prediction = monitor.observe(
        history, mask, proprio,
        actor_names=("piper",), receipts=receipts, image_features=image,
    )
    assert prediction.current_event_probability.shape == (1, 5)
    assert prediction.current_predicate_probability.shape == (1, 5)
    assert torch.allclose(prediction.current_event_probability.sum(1), torch.ones(1))
    assert prediction.applicability.tolist() == [False]
    assert prediction.applicability_reason == ("monitor_only_not_promoted",)

    promoted = _observer(promoted=True, minimum_confidence=0.9)
    history, mask, proprio, receipts, image = _batch(promoted)
    with torch.no_grad():
        promoted.event_head.weight.zero_(); promoted.event_head.bias.zero_()
        promoted.predicate_head.weight.zero_(); promoted.predicate_head.bias.zero_()
    prediction = promoted.observe(
        history, mask, proprio,
        actor_names=("piper",), receipts=receipts, image_features=image,
    )
    assert prediction.applicability.tolist() == [False]
    assert prediction.applicability_reason == ("low_confidence_fail_closed",)

    # A no-threshold calibration must reject even float32 probabilities that
    # saturate to exactly one; threshold=1 alone is not a reject-all sentinel.
    promoted.calibration = make_calibration(
        event_spec_sha256=EVENT_SPEC_SHA,
        independent_calibration_split_sha256=CALIBRATION_SPLIT_SHA,
        minimum_joint_confidence=1.0,
        reject_all=True,
    )
    with torch.no_grad():
        promoted.event_head.bias.fill_(-100)
        promoted.event_head.bias[0] = 100
        promoted.predicate_head.bias.fill_(100)
    prediction = promoted.observe(
        history, mask, proprio,
        actor_names=("piper",), receipts=receipts, image_features=image,
    )
    assert prediction.confidence.item() == 1.0
    assert prediction.applicability.tolist() == [False]
    assert prediction.applicability_reason == ("low_confidence_fail_closed",)


def test_runtime_confidence_tracks_the_emitted_thresholded_predicate() -> None:
    observer = _observer(promoted=False, minimum_confidence=0.0)
    observer.calibration = make_calibration(
        event_spec_sha256=EVENT_SPEC_SHA,
        independent_calibration_split_sha256=CALIBRATION_SPLIT_SHA,
        predicate_thresholds=[0.8] * 5,
        minimum_joint_confidence=0.0,
    )
    history, mask, proprio, receipts, image = _batch(observer)
    with torch.no_grad():
        observer.event_head.weight.zero_()
        observer.event_head.bias.zero_()
        observer.event_head.bias[0] = 20.0
        observer.predicate_head.weight.zero_()
        observer.predicate_head.bias.fill_(
            float(torch.logit(torch.tensor(0.6)))
        )
    prediction = observer.observe(
        history,
        mask,
        proprio,
        actor_names=("piper",),
        receipts=receipts,
        image_features=image,
    )
    assert prediction.current_predicates.tolist() == [[0.0] * 5]
    assert prediction.confidence.item() == pytest.approx(0.4, abs=1.0e-6)


def test_promoted_observer_accepts_only_high_confidence_with_bound_optional_inputs() -> None:
    observer = _observer(promoted=True, minimum_confidence=0.9, image_feature_dim=4)
    history, mask, proprio, receipts, image = _batch(
        observer, with_image=True, with_execution=True
    )
    with torch.no_grad():
        observer.event_head.weight.zero_(); observer.event_head.bias.fill_(-20)
        observer.event_head.bias[2] = 20
        observer.predicate_head.weight.zero_(); observer.predicate_head.bias.fill_(20)
    prediction = observer.observe(
        history, mask, proprio,
        actor_names=("piper",), receipts=receipts, image_features=image,
    )
    assert prediction.current_event_id.tolist() == [2]
    assert prediction.current_predicates.tolist() == [[1, 1, 1, 1, 1]]
    assert prediction.applicability.tolist() == [True]
    assert prediction.applicability_reason == (
        "applicable_promoted_evaluation400_v4_rerank",
    )
    assert observer.deployment["rerank_enabled"] is True
    assert observer.deployment["integration_status"] == (
        "integrated_frozen_observer_into_evaluation400_v4"
    )

    wrong_extractor = copy.deepcopy(receipts[0])
    nested = wrong_extractor["image_feature_receipt"]
    nested["extractor_file_sha256"] = "9" * 64
    nested_logical = dict(nested); nested_logical.pop("receipt_sha256")
    nested["receipt_sha256"] = canonical_sha256(nested_logical)
    root_logical = dict(wrong_extractor); root_logical.pop("receipt_sha256")
    wrong_extractor["receipt_sha256"] = canonical_sha256(root_logical)
    with pytest.raises(CausalObserverContractError, match="extractor provenance"):
        observer.observe(
            history, mask, proprio,
            actor_names=("piper",), receipts=[wrong_extractor],
            image_features=image,
        )


def test_input_receipt_rejects_privileged_future_extra_and_tensor_tampering() -> None:
    observer = _observer()
    history, mask, proprio, receipts, image = _batch(observer)

    privileged = _resign(receipts[0], "object_pose_fields_present", True)
    with pytest.raises(CausalObserverContractError, match="failed closed"):
        observer(history, mask, proprio, actor_names=("piper",), receipts=[privileged])

    future = _resign(receipts[0], "future_features_read", True)
    with pytest.raises(CausalObserverContractError, match="failed closed"):
        observer(history, mask, proprio, actor_names=("piper",), receipts=[future])

    extra = {**receipts[0], "object_poses": [[0, 0, 0]]}
    with pytest.raises(CausalObserverContractError, match="fields changed"):
        observer(history, mask, proprio, actor_names=("piper",), receipts=[extra])

    tampered_history = history.clone()
    tampered_history[0, 0, 0] = torch.nextafter(
        tampered_history[0, 0, 0], torch.tensor(float("inf"))
    )
    with pytest.raises(CausalObserverContractError, match="failed closed"):
        observer(
            tampered_history, mask, proprio,
            actor_names=("piper",), receipts=receipts,
        )

    bad_padding = history.clone(); bad_padding[0, 7, 0] = 1
    padded_receipt = make_input_receipt(
        history=bad_padding[0], history_mask=mask[0], proprio=proprio[0],
        actor_name="piper", policy_family="smolvla",
        state_feature_source_sha256="d" * 64,
        current_query_index=1, valid_history_steps=2,
    )
    with pytest.raises(CausalObserverContractError, match="failed closed"):
        observer(
            bad_padding, mask, proprio,
            actor_names=("piper",), receipts=[padded_receipt],
        )


def test_multi_actor_adapter_is_content_addressed_and_pluggable() -> None:
    observer = _observer(minimum_confidence=0.0)
    history, mask, proprio, receipts, _ = _batch(
        observer, actors=("piper", "new_body")
    )
    output = observer(
        history, mask, proprio,
        actor_names=("piper", "new_body"), receipts=receipts,
    )
    assert not torch.equal(output["event_logits"][0], output["event_logits"][1])
    assert set(observer.adapter_contracts) == {"piper", "new_body"}

    wrong_source = _resign(
        receipts[1], "state_feature_source_sha256", "9" * 64
    )
    with pytest.raises(CausalObserverContractError, match="failed closed"):
        observer(
            history, mask, proprio,
            actor_names=("piper", "new_body"),
            receipts=[receipts[0], wrong_source],
        )

    tampered_contracts = copy.deepcopy(observer.adapter_contracts)
    tampered_contracts["new_body"]["adapter_state_sha256"] = "0" * 64
    # Re-signing metadata cannot make it match the loaded adapter tensors.
    logical = dict(tampered_contracts["new_body"])
    logical.pop("adapter_contract_sha256")
    tampered_contracts["new_body"]["adapter_contract_sha256"] = canonical_sha256(logical)
    states = {
        name: {key: tensor.detach().clone() for key, tensor in adapter.state_dict().items()}
        for name, adapter in zip(observer.actor_names, observer.actor_adapters, strict=True)
    }
    with pytest.raises(CausalObserverContractError, match="adapter contract"):
        ActorVisibleCausalEventObserverV1(
            observer.config,
            training_contract=observer.training_contract,
            observer_core_file_sha256=CORE_SHA,
            observer_checkpoint_file_sha256=CHECKPOINT_SHA,
            observer_config_sha256=CONFIG_SHA,
            actor_adapter_set_sha256=ADAPTER_SET_SHA,
            actor_adapter_checkpoint_set_sha256=ADAPTER_CHECKPOINT_SET_SHA,
            adapter_contracts=tampered_contracts,
            adapter_states=states,
            calibration=observer.calibration,
            deployment=observer.deployment,
        )


def test_promotion_gate_rejects_re_signed_causal_or_support_failure() -> None:
    training = training_supervision_contract(
        event_spec_sha256=EVENT_SPEC_SHA,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA,
        actor_registry=ACTORS,
    )
    calibration = make_calibration(
        event_spec_sha256=EVENT_SPEC_SHA,
        independent_calibration_split_sha256=CALIBRATION_SPLIT_SHA,
        minimum_joint_confidence=0.9,
    )
    evidence = _promotion_evidence(
        training["contract_sha256"], calibration["calibration_sha256"]
    )
    validate_promotion_evidence(
        evidence,
        observer_core_file_sha256=CORE_SHA,
        observer_checkpoint_file_sha256=CHECKPOINT_SHA,
        observer_config_sha256=CONFIG_SHA,
        training_contract_sha256=training["contract_sha256"],
        actor_adapter_set_sha256=ADAPTER_SET_SHA,
        actor_adapter_checkpoint_set_sha256=ADAPTER_CHECKPOINT_SET_SHA,
        calibration_sha256=calibration["calibration_sha256"],
        actor_names=("piper", "new_body"),
    )
    for field, value in (
        ("future_feature_perturbation_invariant", False),
        ("per_actor_validation_groups", {"piper": 30, "new_body": 9}),
        ("low_confidence_false_accept_ucb95", 0.051),
    ):
        changed = copy.deepcopy(evidence)
        changed[field] = value
        logical = dict(changed); logical.pop("promotion_receipt_sha256")
        changed["promotion_receipt_sha256"] = canonical_sha256(logical)
        with pytest.raises(CausalObserverContractError, match="failed closed"):
            validate_promotion_evidence(
                changed,
                observer_core_file_sha256=CORE_SHA,
                observer_checkpoint_file_sha256=CHECKPOINT_SHA,
                observer_config_sha256=CONFIG_SHA,
                training_contract_sha256=training["contract_sha256"],
                actor_adapter_set_sha256=ADAPTER_SET_SHA,
                actor_adapter_checkpoint_set_sha256=ADAPTER_CHECKPOINT_SET_SHA,
                calibration_sha256=calibration["calibration_sha256"],
                actor_names=("piper", "new_body"),
            )


def test_only_validated_v4_target_can_enable_rerank() -> None:
    training = training_supervision_contract(
        event_spec_sha256=EVENT_SPEC_SHA,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA,
        actor_registry=ACTORS,
    )
    calibration = make_calibration(
        event_spec_sha256=EVENT_SPEC_SHA,
        independent_calibration_split_sha256=CALIBRATION_SPLIT_SHA,
        minimum_joint_confidence=0.9,
    )
    evidence = _promotion_evidence(
        training["contract_sha256"], calibration["calibration_sha256"]
    )
    context = {
        "observer_core_file_sha256": CORE_SHA,
        "observer_checkpoint_file_sha256": CHECKPOINT_SHA,
        "observer_config_sha256": CONFIG_SHA,
        "training_contract_sha256": training["contract_sha256"],
        "actor_adapter_set_sha256": ADAPTER_SET_SHA,
        "actor_adapter_checkpoint_set_sha256": ADAPTER_CHECKPOINT_SET_SHA,
        "calibration_sha256": calibration["calibration_sha256"],
        "actor_names": ["piper", "new_body"],
    }
    v4 = make_deployment(
        promotion_enabled=True,
        promotion_evidence=evidence,
        integration_target=EVALUATION400_V4_TARGET,
        promotion_validation_context=context,
    )
    assert v4["rerank_enabled"] is True
    monitor = make_deployment(
        promotion_enabled=True,
        promotion_evidence=evidence,
        integration_target="evaluation400_v3",
    )
    assert monitor["rerank_enabled"] is False
    tampered = copy.deepcopy(evidence)
    tampered["event_macro_accuracy_lcb95"] = 0.1
    logical = dict(tampered); logical.pop("promotion_receipt_sha256")
    tampered["promotion_receipt_sha256"] = canonical_sha256(logical)
    with pytest.raises(CausalObserverContractError, match="failed closed"):
        make_deployment(
            promotion_enabled=True,
            promotion_evidence=tampered,
            integration_target=EVALUATION400_V4_TARGET,
            promotion_validation_context=context,
        )
