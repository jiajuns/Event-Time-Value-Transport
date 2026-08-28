from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from collect_smolvla_etsf_event_branches import (  # noqa: E402
    _synthetic_record,
    save_group,
)
from initialize_smolvla_schema5_native_event_core import (  # noqa: E402
    BODY_EMBEDDING,
    FORMAT,
    POLICY_EMBEDDING,
    STATUS,
    file_sha256,
    initialize_smolvla_schema5_native_core,
    state_dict_sha256,
    verify_initialized_core,
)
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
)
from train_openvla_etsf_counterfactual import (  # noqa: E402
    canonical_policy_mapping,
    load_descriptor_groups,
    load_pretrained,
    read_group_descriptor,
    validated_pretrained_policy_bridge,
)


MODELING_SHA = "0" * 64
BRIDGE_SHA = "1" * 64


def _protocol_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    event_spec = tmp_path / "event_spec.json"
    manifest = tmp_path / "source_manifest.json"
    split = tmp_path / "source_split.json"
    # These deliberately contain plausible label-looking keys.  The initializer
    # only hashes their bytes; it never parses or makes a split/model decision.
    event_spec.write_text('{"calibration":"opaque"}', encoding="utf-8")
    manifest.write_text('{"success":true,"groups":80}', encoding="utf-8")
    split.write_text('{"train":[1],"validation":[2],"test":[3]}', encoding="utf-8")
    return event_spec, manifest, split


def _initialize(tmp_path: Path, name: str = "native_core.pt") -> tuple[Path, dict]:
    event_spec, manifest, split = _protocol_inputs(tmp_path)
    output = tmp_path / name
    audit = initialize_smolvla_schema5_native_core(
        output=output,
        event_spec=event_spec,
        event_spec_sha256=file_sha256(event_spec),
        source_manifest=manifest,
        source_manifest_sha256=file_sha256(manifest),
        source_split=split,
        source_split_sha256=file_sha256(split),
        state_modeling_sha256=MODELING_SHA,
        state_bridge_sha256=BRIDGE_SHA,
        initialization_seed=20260828,
    )
    return output, audit


def _load(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    return value


def test_exact_architecture_identities_rows_and_untrained_contract(tmp_path: Path) -> None:
    output, audit = _initialize(tmp_path)
    payload = _load(output)
    config = payload["config"]
    contract = payload["contract"]
    assert payload["format"] == FORMAT
    assert audit["status"] == STATUS
    assert config["state_input_dim"] == 960
    assert config["action_dim"] == 14
    assert config["proprio_dim"] == 14
    assert config["object_delta_dim"] == 3
    assert config["structured_events"] is True
    assert config["num_bodies"] == 2
    assert config["num_policies"] == 2
    assert contract["body_to_id"] == {"aloha-agilex": 0, "__reserved__piper": 1}
    assert contract["policy_to_id"] == {"smolvla": 0, "__reserved__openvla": 1}
    bridge = validated_pretrained_policy_bridge(payload, required=True)
    assert bridge is not None
    assert bridge["policy"] == "smolvla"
    assert bridge["policy_row"] == 0
    assert bridge["state_feature"]["dimension"] == 960
    assert bridge["action_mapping"]["model_slots"] == list(range(14))
    assert audit["policy_feature_action_bridge_sha256"] == bridge[
        "contract_sha256"
    ]
    assert contract["source_identity_rows"]["body"] == {
        "mapping_name": "body_to_id",
        "identity": "aloha-agilex",
        "id": 0,
        "row": 0,
        "parameter": BODY_EMBEDDING,
        "tensor_sha256": contract["source_identity_rows"]["body"]["tensor_sha256"],
    }
    reserved = contract["reserved_target_rows"]
    assert reserved["body"]["mapping_name"] == "body_to_id"
    assert reserved["body"]["identity"] == "piper"
    assert reserved["body"]["row"] == 1
    assert reserved["body"]["parameter"] == BODY_EMBEDDING
    assert reserved["policy"]["mapping_name"] == "policy_to_id"
    assert reserved["policy"]["identity"] == "openvla"
    assert reserved["policy"]["row"] == 1
    assert reserved["policy"]["parameter"] == POLICY_EMBEDDING
    state = payload["model"]
    assert torch.equal(state[BODY_EMBEDDING][0], state[BODY_EMBEDDING][1])
    assert torch.equal(state[POLICY_EMBEDDING][0], state[POLICY_EMBEDDING][1])
    assert reserved["body"]["tensor_sha256"] == contract["source_identity_rows"]["body"][
        "tensor_sha256"
    ]
    assert reserved["policy"]["tensor_sha256"] == contract["source_identity_rows"][
        "policy"
    ]["tensor_sha256"]
    assert contract["protocol_input_access"] == "bytewise_sha256_only_files_not_parsed"
    for flag in (
        "source_rollout_containers_read",
        "source_labels_read",
        "target_data_read",
        "target_labels_read",
        "sealed_test_data_read",
        "training_performed",
        "shared_core_training_performed",
        "prediction_ready",
        "transfer_ready",
        "ready_for_protocol_freeze",
    ):
        assert contract[flag] is False
    assert contract["training_steps"] == 0
    assert contract["action_normalization"]["status"] == "identity_placeholder_unfitted"
    assert torch.equal(state["action_encoder.action_mean"], torch.zeros(14))
    assert torch.equal(state["action_encoder.action_std"], torch.ones(14))
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert not list(tmp_path.glob("*.partial"))


def test_initialization_is_logically_deterministic_and_preserves_caller_rng(
    tmp_path: Path,
) -> None:
    event_spec, manifest, split = _protocol_inputs(tmp_path)
    arguments = {
        "event_spec": event_spec,
        "event_spec_sha256": file_sha256(event_spec),
        "source_manifest": manifest,
        "source_manifest_sha256": file_sha256(manifest),
        "source_split": split,
        "source_split_sha256": file_sha256(split),
        "state_modeling_sha256": MODELING_SHA,
        "state_bridge_sha256": BRIDGE_SHA,
        "initialization_seed": 9173,
    }
    before = torch.random.get_rng_state().clone()
    initialize_smolvla_schema5_native_core(output=tmp_path / "a.pt", **arguments)
    after_first = torch.random.get_rng_state().clone()
    initialize_smolvla_schema5_native_core(output=tmp_path / "b.pt", **arguments)
    after_second = torch.random.get_rng_state().clone()
    assert torch.equal(before, after_first)
    assert torch.equal(before, after_second)
    left = _load(tmp_path / "a.pt")
    right = _load(tmp_path / "b.pt")
    assert left["config"] == right["config"]
    assert left["contract"] == right["contract"]
    assert state_dict_sha256(left["model"]) == state_dict_sha256(right["model"])
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])


def test_load_pretrained_and_schema5_loader_contracts_are_compatible(
    tmp_path: Path,
) -> None:
    output, _ = _initialize(tmp_path)
    checkpoint, config = load_pretrained(output)
    assert config.state_input_dim == 960
    assert config.object_delta_dim == 3
    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    policy_to_id = canonical_policy_mapping(checkpoint["contract"]["policy_to_id"])
    assert policy_to_id == {"smolvla": 0, "openvla": 1}

    group_path = tmp_path / "smolvla_group.hdf5"
    record = _synthetic_record(hidden_dim=960, chunk=4)
    record["event_spec_sha256"] = checkpoint["contract"]["event_spec_sha256"]
    save_group(group_path, record)
    calibration = {
        "move_can_pot": {
            "moving": "can",
            "anchor": "pot",
            "centers": [[1.0, 1.0, 1.0]],
            "offset": [0.0, 0.0, 0.0],
            "delta_move": 0.05,
            "delta_z": 0.1,
            "tau_d": 0.15,
            "tau_motion": 0.03,
            "stationary_steps": 2,
        }
    }
    descriptor = read_group_descriptor(group_path, {})
    groups = load_descriptor_groups(
        [descriptor],
        config,
        ["can"],
        checkpoint["contract"]["body_to_id"],
        policy_to_id,
        calibrations=calibration,
        regression_persistence_steps=2,
        expected_event_spec_sha256=checkpoint["contract"]["event_spec_sha256"],
    )
    assert len(groups) == 1
    assert groups[0].body_id == 0
    assert groups[0].policy_id == 0
    assert groups[0].hidden.shape == (2, 960)
    assert groups[0].object_delta.shape == (2, 3)
    assert groups[0].state_contract == checkpoint["contract"]["state_contracts"][
        "smolvla"
    ]


def test_hash_binding_sensitive_paths_symlinks_and_no_overwrite_fail_closed(
    tmp_path: Path,
) -> None:
    event_spec, manifest, split = _protocol_inputs(tmp_path)
    common = {
        "event_spec": event_spec,
        "event_spec_sha256": file_sha256(event_spec),
        "source_manifest": manifest,
        "source_manifest_sha256": file_sha256(manifest),
        "source_split": split,
        "source_split_sha256": file_sha256(split),
        "state_modeling_sha256": MODELING_SHA,
        "state_bridge_sha256": BRIDGE_SHA,
    }
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        initialize_smolvla_schema5_native_core(
            output=tmp_path / "mismatch.pt",
            **{**common, "event_spec_sha256": "f" * 64},
        )
    assert not (tmp_path / "mismatch.pt").exists()

    restricted = tmp_path / "Fresh50"
    restricted.mkdir()
    restricted_manifest = restricted / "manifest.json"
    restricted_manifest.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="Fresh/confirmation"):
        initialize_smolvla_schema5_native_core(
            output=tmp_path / "restricted.pt",
            **{
                **common,
                "source_manifest": restricted_manifest,
                "source_manifest_sha256": file_sha256(restricted_manifest),
            },
        )

    linked = tmp_path / "manifest_link.json"
    linked.symlink_to(manifest)
    with pytest.raises(ValueError, match="not a symlink"):
        initialize_smolvla_schema5_native_core(
            output=tmp_path / "linked.pt",
            **{
                **common,
                "source_manifest": linked,
                "source_manifest_sha256": file_sha256(manifest),
            },
        )

    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"user-owned")
    with pytest.raises(FileExistsError):
        initialize_smolvla_schema5_native_core(output=existing, **common)
    assert existing.read_bytes() == b"user-owned"


def test_verifier_rejects_state_tampering_even_if_contract_is_unchanged(
    tmp_path: Path,
) -> None:
    output, _ = _initialize(tmp_path)
    payload = _load(output)
    payload["model"]["success_head.bias"][0] += 1
    os.chmod(output, 0o644)
    torch.save(payload, output)
    with pytest.raises(ValueError, match="state/config audit"):
        verify_initialized_core(output)


def test_formal_trainer_rejects_missing_or_tampered_policy_bridge(
    tmp_path: Path,
) -> None:
    output, _ = _initialize(tmp_path)
    payload = _load(output)
    without_bridge = {**payload, "contract": dict(payload["contract"])}
    without_bridge["contract"].pop("policy_feature_action_bridge")
    with pytest.raises(RuntimeError, match="requires a strict"):
        validated_pretrained_policy_bridge(without_bridge, required=True)

    tampered = {**payload, "contract": dict(payload["contract"])}
    raw = dict(tampered["contract"]["policy_feature_action_bridge"])
    raw["policy_row"] = 1
    tampered["contract"]["policy_feature_action_bridge"] = raw
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validated_pretrained_policy_bridge(tampered, required=True)


def test_protocol_files_remain_byte_identical(tmp_path: Path) -> None:
    event_spec, manifest, split = _protocol_inputs(tmp_path)
    before = {path: path.read_bytes() for path in (event_spec, manifest, split)}
    output = tmp_path / "core.pt"
    initialize_smolvla_schema5_native_core(
        output=output,
        event_spec=event_spec,
        event_spec_sha256=file_sha256(event_spec),
        source_manifest=manifest,
        source_manifest_sha256=file_sha256(manifest),
        source_split=split,
        source_split_sha256=file_sha256(split),
        state_modeling_sha256=MODELING_SHA,
        state_bridge_sha256=BRIDGE_SHA,
    )
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads(json.dumps(verify_initialized_core(output)))["transfer_ready"] is False
