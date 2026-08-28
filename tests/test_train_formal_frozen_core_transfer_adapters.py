from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from train_formal_frozen_core_transfer_adapters import (  # noqa: E402
    BODY_EMBEDDING,
    INPUT_FORMAT,
    INPUT_STATUS,
    POLICY_EMBEDDING,
    RESERVATION_FORMAT,
    RESERVATION_STATUS,
    FormalFrozenCoreTransferModel,
    FormalTrainingConfig,
    FormalTransferError,
    binary_support_gates,
    canonical_sha256,
    file_sha256,
    state_dict_sha256,
    structured_payload_sha256,
    tensor_sha256,
    train_formal_target_adapters,
    validate_dual_reservation,
    validate_monitor_checkpoint,
    validate_receipt,
    validate_structured_input,
)


def _config(*, bodies: int = 2, policies: int = 2, recovery: bool = True):
    return EventWorldModelConfig(
        state_input_dim=4096,
        action_dim=14,
        proprio_dim=14,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=8,
        clock_hidden_dim=4,
        object_delta_dim=6,
        num_bodies=bodies,
        num_policies=policies,
        metadata_dim=4,
        structured_events=True,
        recovery_supervised=recovery,
        dropout=0.0,
    )


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "source_manifest.json"
    split = tmp_path / "source_split.json"
    manifest.write_text('{"source_groups":100}', encoding="utf-8")
    split.write_text('{"train_groups":["s0","s1"]}', encoding="utf-8")
    return manifest, split


def _write_source(
    tmp_path: Path,
    *,
    bodies: int = 2,
    policies: int = 2,
    recovery: bool = True,
    with_proof: bool = True,
) -> tuple[Path, Path, Path]:
    torch.manual_seed(7)
    config = _config(bodies=bodies, policies=policies, recovery=recovery)
    model = ActionConditionedEventWorldModel(config)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    manifest, split = _source_files(tmp_path)
    payload: dict[str, object] = {
        "model": state,
        "config": config.to_dict(),
        "contract": {
            "body_to_id": (
                {"source_body": 0, "__reserved__piper_piper_0.6": 1}
                if bodies >= 2
                else {"piper_piper_0.6": 0}
            ),
            "policy_to_id": (
                {"OpenVLA": 0, "__reserved__smolvla": 1}
                if policies >= 2
                else {"OpenVLA": 0}
            ),
        },
        "normalization": {
            "object_delta_mean": [0.0] * config.object_delta_dim,
            "object_delta_std": [1.0] * config.object_delta_dim,
        },
    }
    if with_proof:
        proof: dict[str, object] = {
            "format": RESERVATION_FORMAT,
            "status": RESERVATION_STATUS,
            "target_body_name": "piper_piper_0.6",
            "target_body_id": 1,
            "target_body_row": 1,
            "target_policy_name": "smolvla",
            "target_policy_id": 1,
            "target_policy_row": 1,
            "body_embedding_parameter": BODY_EMBEDDING,
            "policy_embedding_parameter": POLICY_EMBEDDING,
            "reserved_body_row_sha256": tensor_sha256(state[BODY_EMBEDDING][1]),
            "reserved_policy_row_sha256": tensor_sha256(state[POLICY_EMBEDDING][1]),
            "source_manifest_sha256": file_sha256(manifest),
            "source_split_sha256": file_sha256(split),
            "source_training_steps": 10,
            "source_training_groups": 10,
            "target_data_read": False,
            "target_labels_read": False,
            "reserved_rows_used_in_source_batches": False,
            "reserved_rows_unchanged_during_source_training": True,
            "shared_core_retrained": True,
            "source_core_state_sha256": state_dict_sha256(state),
            "input_dual_expanded_checkpoint_sha256": "a" * 64,
        }
        proof["reservation_sha256"] = canonical_sha256(proof)
        payload["formal_target_reservation"] = proof
    source = tmp_path / "source_core.pt"
    torch.save(payload, source)
    return source, manifest, split


def _structured_payload(config: EventWorldModelConfig, *, samples: int = 8):
    torch.manual_seed(11)
    group_index = torch.arange(samples, dtype=torch.int64)
    binary = (group_index % 2).to(torch.float32)
    object_valid = torch.ones(samples, 2, dtype=torch.bool)
    arrays = {
        "state": torch.randn(samples, 960),
        "history_mask": torch.ones(samples, 1, dtype=torch.bool),
        "action_chunks": torch.randn(samples, 50, 14) * 0.05,
        "action_mask": torch.ones(samples, 50, dtype=torch.bool),
        "proprio": torch.randn(samples, config.proprio_dim) * 0.05,
        "current_event_id": group_index % config.num_events,
        "clock_event_id": group_index % config.num_events,
        "current_predicates": (torch.rand(samples, config.num_predicates) > 0.5).float(),
        "dt_decision_steps": torch.ones(samples),
        "next_event_id": (group_index + 1) % config.num_events,
        "next_event_mask": torch.ones(samples, dtype=torch.bool),
        "destination_event_id": (group_index + 2) % config.num_events,
        "destination_event_mask": torch.ones(samples, dtype=torch.bool),
        "duration_log1p_decision_steps": torch.full((samples,), 0.6931472),
        "duration_mask": torch.ones(samples, dtype=torch.bool),
        "post_predicates": (torch.rand(samples, config.num_predicates) > 0.5).float(),
        "predicate_mask": torch.ones(samples, config.num_predicates, dtype=torch.bool),
        "object_delta_physical": torch.randn(samples, config.object_delta_dim) * 0.01,
        "object_delta_supervision_valid": object_valid,
        "object_delta_invalid_reason_bitset": torch.zeros(samples, 2, dtype=torch.int64),
        "object_feature_object_index": torch.tensor([0, 0, 0, 1, 1, 1]),
        "success": binary,
        "success_mask": torch.ones(samples, dtype=torch.bool),
        "recovery": 1.0 - binary,
        "recovery_mask": torch.ones(samples, dtype=torch.bool),
        "sample_group_index": group_index,
    }
    payload: dict[str, object] = {
        "format": INPUT_FORMAT,
        "status": INPUT_STATUS,
        "evidence_scope": "nonfresh_target_adaptation_development_only",
        "schema_version": 6,
        "task": "move_can_pot",
        "target_actor_id": "smolvla_robotwin_aloha-trained__piper-zero-shot",
        "target_body": "piper_piper_0.6",
        "split_role": "target_adaptation",
        "fresh_or_confirmation_data_read": False,
        "event_spec_sha256": "b" * 64,
        "schema6_pose_quality": {
            "format": "etsf_schema6_pose_quality_v1",
            "object_registry_sha256": "c" * 64,
            "pose_integrity_spec_sha256": "d" * 64,
            "interval_mask_semantics": "all_destinations_valid_no_reset_or_teleport_crossed",
        },
        "logical_group_keys": [f"group-{index}" for index in range(samples)],
        "arrays": arrays,
    }
    payload["payload_sha256"] = structured_payload_sha256(payload)
    return payload


def _resign(payload: dict[str, object]) -> None:
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = structured_payload_sha256(payload)


def test_end_to_end_cpu_training_is_monitor_only_and_content_addressed(
    tmp_path: Path,
) -> None:
    source, manifest, split = _write_source(tmp_path)
    payload = _structured_payload(_config())
    structured = tmp_path / "structured_arrays.pt"
    torch.save(payload, structured)
    result = train_formal_target_adapters(
        source_checkpoint_path=source,
        source_manifest_path=manifest,
        source_split_path=split,
        input_path=structured,
        output_dir=tmp_path / "formal_output",
        config=FormalTrainingConfig(
            steps=1,
            batch_size=4,
            state_bottleneck_dim=4,
            min_binary_class_groups=50,
        ),
        device="cpu",
    )
    checkpoint_path = Path(result["checkpoint"])
    receipt_path = Path(result["receipt"])
    assert checkpoint_path.name == (
        f"formal_target_adapter_{result['checkpoint_payload_sha256']}.pt"
    )
    checkpoint = validate_monitor_checkpoint(checkpoint_path)
    receipt = validate_receipt(receipt_path)
    assert checkpoint["authorization"] == {
        "monitor_only": True,
        "selection_authorized": False,
        "action_ranking_authorized": False,
        "environment_execution_authorized": False,
        "transfer_claim_authorized": False,
        "shared_core_gradient_or_update_authorized": False,
    }
    assert checkpoint["training"]["immutable_core_audit"][
        "all_non_target_core_values_bit_exact"
    ] is True
    assert checkpoint["binary_support_gates"]["success"]["enabled"] is False
    assert checkpoint["binary_support_gates"]["recovery"]["enabled"] is False
    assert receipt["selection_authorized"] is False
    assert result["selection_authorized"] is False
    assert result["monitor_only"] is True
    assert checkpoint_path.stat().st_mode & 0o222 == 0
    assert receipt_path.stat().st_mode & 0o222 == 0


def test_binary_heads_require_independent_group_support() -> None:
    config = _config(recovery=True)
    validated = validate_structured_input(
        _structured_payload(config, samples=100), core_config=config
    )
    support = binary_support_gates(validated["arrays"], core_config=config, minimum=50)
    assert support["success"]["enabled"] is True
    assert support["recovery"]["enabled"] is True
    no_recovery = binary_support_gates(
        validated["arrays"], core_config=_config(recovery=False), minimum=50
    )
    assert no_recovery["recovery"]["enabled"] is False
    assert no_recovery["recovery"]["reason"] == "core_recovery_head_not_source_supervised"


def test_supported_success_and_recovery_losses_are_optimized(tmp_path: Path) -> None:
    source, manifest, split = _write_source(tmp_path)
    structured = tmp_path / "supported_structured_arrays.pt"
    torch.save(_structured_payload(_config()), structured)
    result = train_formal_target_adapters(
        source_checkpoint_path=source,
        source_manifest_path=manifest,
        source_split_path=split,
        input_path=structured,
        output_dir=tmp_path / "supported_output",
        config=FormalTrainingConfig(
            steps=1,
            batch_size=8,
            state_bottleneck_dim=4,
            min_binary_class_groups=4,
        ),
    )
    checkpoint = validate_monitor_checkpoint(Path(result["checkpoint"]))
    assert checkpoint["binary_support_gates"]["success"]["enabled"] is True
    assert checkpoint["binary_support_gates"]["recovery"]["enabled"] is True
    assert checkpoint["training"]["mean_losses"]["success"] > 0
    assert checkpoint["training"]["mean_losses"]["recovery"] > 0


def test_schema6_quality_and_payload_tampering_fail_closed() -> None:
    config = _config()
    payload = _structured_payload(config)
    tampered = copy.deepcopy(payload)
    tampered["arrays"]["object_delta_invalid_reason_bitset"][0, 0] = 1  # type: ignore[index]
    _resign(tampered)
    with pytest.raises(FormalTransferError, match="valid mask"):
        validate_structured_input(tampered, core_config=config)

    unsigned_tamper = copy.deepcopy(payload)
    unsigned_tamper["arrays"]["state"][0, 0] += 1  # type: ignore[index]
    with pytest.raises(FormalTransferError, match="payload SHA"):
        validate_structured_input(unsigned_tamper, core_config=config)


def test_group_labels_must_be_binary_and_consistent() -> None:
    config = _config()
    nonbinary = _structured_payload(config)
    nonbinary["arrays"]["success"][0] = 0.5  # type: ignore[index]
    _resign(nonbinary)
    with pytest.raises(FormalTransferError, match="exactly binary"):
        validate_structured_input(nonbinary, core_config=config)

    inconsistent = _structured_payload(config)
    inconsistent["arrays"]["sample_group_index"] = torch.tensor(  # type: ignore[index]
        [0, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int64
    )
    inconsistent["logical_group_keys"].pop()  # type: ignore[union-attr]
    _resign(inconsistent)
    with pytest.raises(FormalTransferError, match="consistent within"):
        validate_structured_input(inconsistent, core_config=config)


def test_authoritative_single_row_source_core_is_rejected_before_target_data(
    tmp_path: Path,
) -> None:
    source, manifest, split = _write_source(
        tmp_path, bodies=1, policies=1, with_proof=False
    )
    structured = tmp_path / "unread_structured.pt"
    torch.save({}, structured)
    with pytest.raises(FormalTransferError, match="num_bodies=1,num_policies=1"):
        train_formal_target_adapters(
            source_checkpoint_path=source,
            source_manifest_path=manifest,
            source_split_path=split,
            input_path=structured,
            output_dir=tmp_path / "must_not_exist",
            config=FormalTrainingConfig(steps=1),
        )
    assert not (tmp_path / "must_not_exist").exists()


def test_posthoc_or_unsigned_reservation_is_rejected(tmp_path: Path) -> None:
    source, manifest, split = _write_source(tmp_path, with_proof=False)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["transfer_source_core_expansion"] = {"axis": "policy"}
    with pytest.raises(FormalTransferError, match="post-hoc/single-axis"):
        validate_dual_reservation(
            checkpoint,
            source_manifest_sha256=file_sha256(manifest),
            source_split_sha256=file_sha256(split),
        )

    source, manifest, split = _write_source(tmp_path / "signed")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["formal_target_reservation"]["source_training_steps"] = 11
    with pytest.raises(FormalTransferError, match="signature mismatch"):
        validate_dual_reservation(
            checkpoint,
            source_manifest_sha256=file_sha256(manifest),
            source_split_sha256=file_sha256(split),
        )


def test_immutable_core_audit_detects_and_restores_shared_change() -> None:
    core = ActionConditionedEventWorldModel(_config())
    model = FormalFrozenCoreTransferModel(
        core,
        target_body_row=1,
        target_policy_row=1,
        state_bottleneck_dim=4,
    )
    with torch.no_grad():
        model.core.next_event_head.weight[0, 0] += 1
    with pytest.raises(FormalTransferError, match="shared core changed"):
        model.immutable_core_audit()
    model.enforce_frozen_core()
    assert model.immutable_core_audit()["all_non_target_core_values_bit_exact"] is True


def test_monitor_checkpoint_validator_rejects_tensor_tamper(tmp_path: Path) -> None:
    source, manifest, split = _write_source(tmp_path)
    structured = tmp_path / "structured_arrays.pt"
    torch.save(_structured_payload(_config()), structured)
    result = train_formal_target_adapters(
        source_checkpoint_path=source,
        source_manifest_path=manifest,
        source_split_path=split,
        input_path=structured,
        output_dir=tmp_path / "out",
        config=FormalTrainingConfig(steps=1, batch_size=4, state_bottleneck_dim=4),
    )
    original = Path(result["checkpoint"])
    payload = torch.load(original, map_location="cpu", weights_only=True)
    payload["adapter_state"]["clock_beta"] += 1
    tampered = tmp_path / original.name
    torch.save(payload, tampered)
    with pytest.raises(FormalTransferError, match="payload SHA"):
        validate_monitor_checkpoint(tampered)


def test_receipt_json_is_self_signed(tmp_path: Path) -> None:
    source, manifest, split = _write_source(tmp_path)
    structured = tmp_path / "structured_arrays.pt"
    torch.save(_structured_payload(_config()), structured)
    result = train_formal_target_adapters(
        source_checkpoint_path=source,
        source_manifest_path=manifest,
        source_split_path=split,
        input_path=structured,
        output_dir=tmp_path / "out_receipt",
        config=FormalTrainingConfig(steps=1, batch_size=4, state_bottleneck_dim=4),
    )
    receipt = Path(result["receipt"])
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["selection_authorized"] = True
    altered = tmp_path / "altered_receipt.json"
    altered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FormalTransferError, match="signature mismatch"):
        validate_receipt(altered)
