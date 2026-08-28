from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import initialize_smolvla_schema5_native_event_core as initializer  # noqa: E402
import openvla_etsf_event_world_model as world_model  # noqa: E402
import smolvla_piper_causal_event_observer_v1 as observer  # noqa: E402
import smolvla_piper_evaluation400_production_root_predictor_runtime_v1 as runtime  # noqa: E402
import smolvla_piper_evaluation400_root_observed_contract_v1 as root_contract  # noqa: E402
import train_openvla_etsf_counterfactual as source_training  # noqa: E402
import train_smolvla_piper_schema6_embodiment_adapter as adapter_trainer  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def signed(base: dict[str, Any], field: str) -> dict[str, Any]:
    return {**base, field: runtime.canonical_sha256(base)}


def _production_source_checkpoint(seed: int) -> dict[str, Any]:
    digest = sha(f"fixture-{seed}")
    checkpoint = initializer._build_payload(
        event_spec=Path("synthetic_event_spec.json"),
        event_spec_sha256=digest,
        source_manifest=Path("synthetic_manifest.json"),
        source_manifest_sha256=digest,
        source_split=Path("synthetic_split.json"),
        source_split_sha256=digest,
        modeling_sha256=digest,
        bridge_sha256=digest,
        initialization_seed=seed,
    )
    config_raw = dict(checkpoint["config"])
    config_raw["action_rank_residual"] = True
    config_raw["action_rank_success_only"] = False
    config_raw["dropout"] = 0.0
    config = world_model.EventWorldModelConfig.from_dict(config_raw)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1000)
        model = world_model.ActionConditionedEventWorldModel(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(name.startswith("action_rank_head.") for name in incompatible.missing_keys)
    with torch.no_grad():
        model.action_encoder.body_embedding.weight[0].add_(0.125)
        model.action_encoder.policy_embedding.weight[0].sub_(0.25)
    checkpoint["config"] = config.to_dict()
    checkpoint["model"] = model.state_dict()
    checkpoint["contract"]["object_names"] = ["can"]
    checkpoint["contract"]["causal_history_contract"] = (
        source_training.causal_history_contract()
    )
    checkpoint["contract"]["action_rank_optimization"] = {
        "freeze_factual_core": False,
        "trainable_parameter_names": [
            "semantic.bridge.0.weight",
            "next_event_head.weight",
            "success_head.weight",
            "clock_cell.candidate.weight",
            "action_rank_head.0.weight",
        ],
    }
    rows = source_training.validate_reserved_target_rows(checkpoint, config)
    proof = source_training.reserved_rows_source_only_proof(
        model,
        rows,
        source_training_steps=3,
        source_training_groups=2,
        input_pretrained_checkpoint_sha256=sha("pretrained-source"),
        action_normalization=None,
    )
    checkpoint["reserved_target_rows_source_only_proof"] = proof
    checkpoint["contract"]["reserved_target_rows_source_only_proof"] = proof
    checkpoint["duration_scale"] = 25.0
    checkpoint["normalization"] = {
        "object_delta_mean": [0.01, -0.02, 0.03],
        "object_delta_std": [0.10, 0.20, 0.30],
    }
    adapter_trainer.validate_source_checkpoint(checkpoint)
    adapter_trainer.validate_production_source_rank_config(config)
    return checkpoint


def _adapter_payload(
    source: dict[str, Any], *, source_file_sha256: str, seed: int,
) -> tuple[dict[str, Any], str, str]:
    config = world_model.EventWorldModelConfig.from_dict(source["config"])
    rank = adapter_trainer.source_rank_score_contract(
        source, config, source_checkpoint_file_sha256=source_file_sha256
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 2000)
        native = world_model.ActionConditionedEventWorldModel(config)
        native.load_state_dict(source["model"], strict=True)
        model = adapter_trainer.SmolVLAPiperAdapter(
            native, state_rank=2, action_rank=2, source_rank_contract=rank
        )
        recovery = adapter_trainer.DetachedConditionalRecoveryAdapter(
            config.semantic_dim
        )
    payload = {
        "format": adapter_trainer.FORMAT,
        "source_checkpoint_sha256": source_file_sha256,
        "source_rank_score_contract": rank,
        "adapter_config": {
            "state_rank": 2,
            "action_rank": 2,
            "source_action_rank_residual_consumed": True,
            "source_action_rank_success_only": False,
            "deployment_success_logit": "base_factual_success_logit",
            "deployment_primary_candidate_score": "source_contract_rank_score",
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "source_rank_score_contract_sha256": rank["contract_sha256"],
        },
        "ranking_contract": {
            "candidate_prediction_api": "predict_grouped_candidates",
            "source_action_rank_success_only": False,
            "deployment_success_logit": "base_factual_success_logit",
            "deployment_primary_candidate_score": "source_contract_rank_score",
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "deployment_success_probability_selector_authorized": False,
        },
        "conditional_recovery_contract": {
            "trained": True,
            "shared_transition_stop_gradient": True,
        },
        "model": model.state_dict(),
        "conditional_recovery_adapter": recovery.state_dict(),
    }
    mean, std = adapter_trainer.object_normalization(
        source, config.object_delta_dim
    )
    normalization_sha = runtime._array_bundle_sha256({
        "object_delta_mean": mean.numpy(),
        "object_delta_std": std.numpy(),
    })
    return payload, rank["contract_sha256"], normalization_sha


@pytest.fixture(scope="module")
def artifact_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("real_root_predictor")
    source_paths: list[Path] = []
    adapter_paths: list[Path] = []
    source_shas: list[str] = []
    adapter_shas: list[str] = []
    rank_shas: list[str] = []
    normalization_shas: list[str] = []
    model_config_sha: str | None = None
    for index in range(runtime.MEMBER_COUNT):
        source = _production_source_checkpoint(31 + index)
        source_path = root / f"source_{index:02d}.pt"
        torch.save(source, source_path)
        source_sha = runtime.file_sha256(source_path)
        payload, rank_sha, normalization_sha = _adapter_payload(
            source, source_file_sha256=source_sha, seed=31 + index
        )
        adapter_path = root / f"adapter_{index:02d}.pt"
        torch.save(payload, adapter_path)
        source_paths.append(source_path)
        adapter_paths.append(adapter_path)
        source_shas.append(source_sha)
        adapter_shas.append(runtime.file_sha256(adapter_path))
        rank_shas.append(rank_sha)
        normalization_shas.append(normalization_sha)
        current_config_sha = runtime.canonical_sha256(source["config"])
        model_config_sha = model_config_sha or current_config_sha
        assert model_config_sha == current_config_sha
    assert len(set(normalization_shas)) == 1
    execution_base = {
        "format": runtime.EXECUTION_AUTHORITY_FORMAT,
        "status": "frozen_real_model_family_for_evaluation400_execution_only",
        "model_family": runtime.MODEL_FAMILY,
        "member_count": runtime.MEMBER_COUNT,
        "source_checkpoint_file_sha256": source_shas,
        "adapter_checkpoint_file_sha256": adapter_shas,
        "source_rank_score_contract_sha256": rank_shas,
        "model_config_sha256": model_config_sha,
        "object_source_normalization_sha256": normalization_shas,
        "calibration": {
            "source_calibration_file_sha256": sha("formal190-calibration-file"),
            "source_calibration_sha256": sha("formal190-calibration"),
            "root_group_ranker_sha256": sha("formal190-root-ranker"),
            "post_event_temperature": 1.1,
            "next_event_temperature": 1.2,
            "success_temperature": 1.3,
            "conditional_recovery_temperature": 1.4,
            "duration_scale_multiplier": 1.5,
            "object_scale_multiplier": 1.6,
            "object_error_robust_scale_m": 0.2,
            "maximum_total_uncertainty": 0.9,
            "all_six_heads_enabled": True,
            "formal190_selection_aware_gate_passed": True,
        },
        "implementation": runtime._implementation_contract(),
        "production_evaluation400_execution_authorized": True,
        "scientific_promotion_eligible": False,
        "compact_reference_model_allowed": False,
        "actor_visible_root_observation_required": True,
        "simulator_or_target_inputs_available_to_predictor": False,
    }
    execution = signed(execution_base, "execution_authority_sha256")
    execution_path = root / "execution_authority_input.json"
    execution_path.write_text(
        json.dumps(execution, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    artifacts = root / "artifacts"
    converted = runtime.convert_production_root_predictor_artifacts(
        source_checkpoint_paths=source_paths,
        adapter_checkpoint_paths=adapter_paths,
        execution_authority_path=execution_path,
        expected_execution_authority_file_sha256=runtime.file_sha256(execution_path),
        expected_execution_authority_sha256=execution["execution_authority_sha256"],
        output_directory=artifacts,
    )
    return {
        "root": root,
        "artifacts": artifacts,
        "source_paths": source_paths,
        "adapter_paths": adapter_paths,
        "execution": execution,
        "execution_path": execution_path,
        "converted": converted,
    }


PAIR_ID = sha("pair")
SNAPSHOT_SHA = sha("shared-snapshot")
PRE_ACTION_SHA = sha("pre-action-snapshot")
OBSERVER_AUTHORITY_SHA = sha("observer-authority")
OBSERVER_ADAPTER_SHA = sha("observer-piper-adapter")
ORDERED = [sha(f"candidate-{index}") for index in range(4)]
LEGAL = [False, True, True, True]


def observation_material() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]:
    history = torch.zeros(8, 960, dtype=torch.float32)
    history[0] = torch.arange(960, dtype=torch.float32) / 960.0
    history_mask = torch.tensor([True] + [False] * 7, dtype=torch.bool)
    proprio = torch.linspace(-1.0, 1.0, 14, dtype=torch.float32)
    input_receipt = observer.make_input_receipt(
        history=history,
        history_mask=history_mask,
        proprio=proprio,
        actor_name="piper",
        policy_family="smolvla",
        state_feature_source_sha256=sha("actor-visible-hidden-hook"),
        current_query_index=0,
        valid_history_steps=1,
    )
    receipt = signed({
        "format": "etsf_smolvla_piper_causal_observer_query_receipt_v4",
        "status": "actor_visible_promoted_observation_applicable",
        "authority_sha256": OBSERVER_AUTHORITY_SHA,
        "pair_id": PAIR_ID,
        "condition_id": root_contract.ROOT_SCOPE,
        "step_index": 0,
        "input_receipt_sha256": input_receipt["receipt_sha256"],
        "prediction_sha256": sha("observer-output"),
        "calibration_sha256": sha("observer-calibration"),
        "minimum_joint_confidence": 0.7,
        "current_event_id": 1,
        "current_predicates": {
            "moved": True,
            "lifted": False,
            "near_goal": False,
            "stationary": False,
            "success": False,
        },
        "confidence": 0.9,
        "applicable": True,
        "object_pose_fields_present": False,
        "simulator_privileged_state_read": False,
        "hardcoded_event_fallback_used": False,
    }, "receipt_sha256")
    inputs = {
        "actor_name": "piper",
        "policy_family": "smolvla",
        "state_feature_source_sha256": sha("actor-visible-hidden-hook"),
        "actor_adapter_contract_sha256": OBSERVER_ADAPTER_SHA,
        "history": history.numpy().copy(),
        "history_mask": history_mask.numpy().copy(),
        "proprio": proprio.numpy().copy(),
        "image_feature": None,
        "image_feature_extractor_file_sha256": None,
        "observer_input_receipt": input_receipt,
    }
    actions = (
        np.arange(4 * 3 * 14, dtype=np.float32).reshape(4, 3, 14) / 100.0
    )
    chronology = {
        "root_reset_calls": 1,
        "root_policy_query_calls": 1,
        "root_observer_calls": 1,
        "root_world_model_member_calls": 0,
        "simulator_step_calls": 0,
        "condition_started_count": 0,
        "target_read_calls": 0,
    }
    commitment = root_contract.build_root_observation_commitment(
        pair_id=PAIR_ID,
        pair_ordinal=0,
        shared_snapshot_sha256=SNAPSHOT_SHA,
        pre_action_snapshot_sha256=PRE_ACTION_SHA,
        observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
        observer_actor_adapter_contract_sha256=OBSERVER_ADAPTER_SHA,
        observer_output_receipt=receipt,
        actor_visible_inputs=inputs,
        ordered_candidate_sha256=ORDERED,
        candidate_legal=LEGAL,
        lowest_legal_original_candidate_index=1,
        mapped_actions=actions,
        chronology=chronology,
    )
    return commitment, inputs, receipt, actions


def predict_kwargs() -> dict[str, Any]:
    commitment, inputs, receipt, actions = observation_material()
    return {
        "observation_commitment": commitment,
        "expected_pair_id": PAIR_ID,
        "expected_pair_ordinal": 0,
        "expected_shared_snapshot_sha256": SNAPSHOT_SHA,
        "expected_pre_action_snapshot_sha256": PRE_ACTION_SHA,
        "expected_observer_authority_sha256": OBSERVER_AUTHORITY_SHA,
        "expected_observer_actor_adapter_contract_sha256": OBSERVER_ADAPTER_SHA,
        "observer_output_receipt": receipt,
        "actor_visible_inputs": inputs,
        "ordered_candidate_sha256": ORDERED,
        "candidate_legal": LEGAL,
        "lowest_legal_original_candidate_index": 1,
        "mapped_actions": actions,
    }


def load(bundle: dict[str, Any]) -> runtime.ProductionRootPredictorRuntimeV1:
    return runtime.load_production_root_predictor_runtime(
        bundle["artifacts"],
        expected_root_predictor_authority_sha256=bundle["converted"][
            "root_predictor_authority_sha256"
        ],
    )


def test_converter_loader_and_prediction_use_exact_real_five_member_family(
    artifact_bundle: dict[str, Any],
) -> None:
    predictor = load(artifact_bundle)
    assert predictor.model_family == runtime.MODEL_FAMILY
    assert "Compact" not in predictor.model_family
    assert predictor.production_evaluation400_execution_authorized is True
    assert predictor.scientific_promotion_eligible is False
    assert len(predictor._members) == 5
    for member in predictor._members:
        assert type(member.model) is adapter_trainer.SmolVLAPiperAdapter
        assert type(member.model.core) is world_model.ActionConditionedEventWorldModel
        assert type(member.recovery) is adapter_trainer.DetachedConditionalRecoveryAdapter
        assert all(parameter.requires_grad is False for parameter in member.model.parameters())
        assert all(parameter.requires_grad is False for parameter in member.recovery.parameters())

    result = predictor.predict(**predict_kwargs())
    assert result.member_call_count == 5
    assert set(result.raw_predictions) == {
        "post_event_logits", "next_event_logits", "duration_log_mean",
        "duration_log_scale", "success_logit", "object_mean", "object_log_scale",
    }
    assert result.raw_predictions["success_logit"].shape == (5, 3)
    assert result.raw_predictions["object_mean"].shape == (5, 3, 3)
    assert result.auxiliary_tensors["member_composite_rank_score"].shape == (5, 3)
    uncertainty = result.auxiliary_tensors[
        "candidate_structured_five_head_uncertainty"
    ]
    assert uncertainty.shape == (3,)
    assert np.isfinite(uncertainty).all()
    assert ((uncertainty >= 0.0) & (uncertainty <= 1.0)).all()


def test_converter_and_loader_always_request_weights_only(
    artifact_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    original = torch.load

    def guarded_load(
        checkpoint: Any, map_location: Any = None, *,
        weights_only: Any = None, **kwargs: Any,
    ) -> Any:
        calls.append(weights_only)
        assert weights_only is True
        return original(
            checkpoint, map_location=map_location,
            weights_only=weights_only, **kwargs,
        )

    monkeypatch.setattr(torch, "load", guarded_load)
    output = tmp_path / "converted"
    result = runtime.convert_production_root_predictor_artifacts(
        source_checkpoint_paths=artifact_bundle["source_paths"],
        adapter_checkpoint_paths=artifact_bundle["adapter_paths"],
        execution_authority_path=artifact_bundle["execution_path"],
        expected_execution_authority_file_sha256=runtime.file_sha256(
            artifact_bundle["execution_path"]
        ),
        expected_execution_authority_sha256=artifact_bundle["execution"][
            "execution_authority_sha256"
        ],
        output_directory=output,
    )
    runtime.load_production_root_predictor_runtime(
        output,
        expected_root_predictor_authority_sha256=result[
            "root_predictor_authority_sha256"
        ],
    )
    assert calls == [True] * 15


def test_loader_uses_locked_legacy_allowlist_when_safe_globals_is_absent(
    artifact_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialization = torch.serialization
    assert callable(getattr(serialization, "get_safe_globals", None))
    assert callable(getattr(serialization, "add_safe_globals", None))
    assert callable(getattr(serialization, "clear_safe_globals", None))
    before = tuple(serialization.get_safe_globals())
    monkeypatch.delattr(serialization, "safe_globals")
    predictor = load(artifact_bundle)
    assert predictor.model_family == runtime.MODEL_FAMILY
    after = tuple(serialization.get_safe_globals())
    assert len(after) == len(before)
    assert all(any(value is other for other in after) for value in before)
    assert all(any(value is other for other in before) for value in after)
    assert runtime.weights_only_compat._SAFE_GLOBALS_LOCK is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_promotion_eligible", True),
        ("compact_reference_model_allowed", True),
        ("model_family", "CompactRootPredictorV1"),
    ],
)
def test_resigned_execution_authority_cannot_cross_promotion_or_compact_boundary(
    artifact_bundle: dict[str, Any], field: str, value: Any,
) -> None:
    changed = copy.deepcopy(artifact_bundle["execution"])
    changed[field] = value
    base = {
        name: child
        for name, child in changed.items()
        if name != "execution_authority_sha256"
    }
    changed["execution_authority_sha256"] = runtime.canonical_sha256(base)
    with pytest.raises(runtime.ProductionRootPredictorError):
        runtime.validate_execution_authority(
            changed,
            expected_authority_sha256=changed["execution_authority_sha256"],
        )


def test_wrong_external_artifact_pin_and_post_load_artifact_tamper_fail_closed(
    artifact_bundle: dict[str, Any], tmp_path: Path,
) -> None:
    with pytest.raises(runtime.ProductionRootPredictorError, match="externally pinned"):
        runtime.load_production_root_predictor_runtime(
            artifact_bundle["artifacts"],
            expected_root_predictor_authority_sha256=sha("wrong-artifact"),
        )
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_bundle["artifacts"], copied)
    predictor = runtime.load_production_root_predictor_runtime(
        copied,
        expected_root_predictor_authority_sha256=artifact_bundle["converted"][
            "root_predictor_authority_sha256"
        ],
    )
    authority = copied / runtime.AUTHORITY_BASENAME
    authority.write_bytes(authority.read_bytes() + b" ")
    with pytest.raises(runtime.ProductionRootPredictorError, match="changed after load"):
        predictor.predict(**predict_kwargs())


def test_actor_adapter_mismatch_is_rejected_before_any_real_member_call(
    artifact_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = load(artifact_bundle)
    calls = 0

    def forbidden_call(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("real model must not run before actor-visible validation")

    monkeypatch.setattr(
        adapter_trainer.SmolVLAPiperAdapter,
        "predict_grouped_candidates",
        forbidden_call,
    )
    kwargs = predict_kwargs()
    kwargs["expected_observer_actor_adapter_contract_sha256"] = sha(
        "different-observer-adapter"
    )
    with pytest.raises(runtime.ProductionRootPredictorError, match="before real-member"):
        predictor.predict(**kwargs)
    assert calls == 0


def test_resigned_prediction_provenance_tamper_is_rejected(
    artifact_bundle: dict[str, Any],
) -> None:
    predictor = load(artifact_bundle)
    kwargs = predict_kwargs()
    result = predictor.predict(**kwargs)
    derivation = copy.deepcopy(result.derivation_commitment)
    derivation["root_predictor_authority_sha256"] = sha("other-root-authority")
    base = {
        name: child
        for name, child in derivation.items()
        if name != "derivation_commit_sha256"
    }
    derivation["derivation_commit_sha256"] = runtime.canonical_sha256(base)
    changed = runtime.ProductionRootPredictionResult(
        raw_predictions=result.raw_predictions,
        auxiliary_tensors=result.auxiliary_tensors,
        derivation_commitment=derivation,
        member_call_count=5,
    )
    with pytest.raises(runtime.ProductionRootPredictorError, match="provenance"):
        predictor.validate_prediction_result(
            changed, observation_commitment=kwargs["observation_commitment"]
        )
