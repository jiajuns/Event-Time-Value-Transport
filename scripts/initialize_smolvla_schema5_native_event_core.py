#!/usr/bin/env python3
"""Create a deterministic, data-blind SmolVLA-native schema-v5 event core.

This is an *initialization* artifact, not a trained model.  It hashes the three
explicit protocol inputs byte-for-byte but never parses rollout descriptors or
opens rollout/label containers.  The source identities occupy row zero and the
target identities are data-blind, bit-exact clones in row one.  A later
source-only training stage must restore both reserved rows after every optimizer
step before this artifact can become eligible for target adaptation.

No OpenVLA, SmolVLA, RoboTwin, CUDA, network, or target-data dependency is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from etsf_policy_feature_action_bridge import (
    build_policy_feature_action_bridge_contract,
    validate_checkpoint_policy_bridge_header,
)


FORMAT = "etsf_smolvla_schema5_native_core_initialization_v2"
STATUS = "initialized_data_blind_untrained_not_transfer_ready"
DEFAULT_INITIALIZATION_SEED = 20260828
STATE_INPUT_DIM = 960
ACTION_DIM = 14
PROPRIO_DIM = 14
OBJECT_DELTA_DIM = 3
SOURCE_BODY = "aloha-agilex"
SOURCE_POLICY = "smolvla"
TARGET_BODY = "piper"
TARGET_POLICY = "openvla"
BODY_EMBEDDING = "action_encoder.body_embedding.weight"
POLICY_EMBEDDING = "action_encoder.policy_embedding.weight"
EVENT_NAMES = ("e0", "e12", "e3", "e4", "eK")
PREDICATE_NAMES = ("moved", "lifted", "near_goal", "stationary", "success")
RELATIVE_TRANSITION_NAMES = ("stay", "advance", "skip", "regress")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_json_bytes([str(value.dtype), list(value.shape)]))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(_json_bytes([name, str(value.dtype), list(value.shape)]))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be one lowercase SHA256 digest")
    return normalized


def _contains_sensitive_component(path: Path) -> bool:
    return any(
        token in component.lower()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def _resolve_protocol_file(path: str | Path, label: str) -> Path:
    supplied = Path(path).expanduser()
    supplied_absolute = Path(os.path.abspath(os.fspath(supplied)))
    if _contains_sensitive_component(supplied_absolute):
        raise ValueError(f"{label} path is forbidden by the Fresh/confirmation boundary")
    if supplied.is_symlink():
        raise ValueError(f"{label} must be a materialized regular file, not a symlink")
    resolved = supplied.resolve(strict=True)
    if _contains_sensitive_component(resolved):
        raise ValueError(f"{label} resolves into a forbidden Fresh/confirmation path")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _bind_protocol_file(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[Path, str]:
    resolved = _resolve_protocol_file(path, label)
    expected = _require_sha256(expected_sha256, f"expected {label} SHA256")
    actual = file_sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return resolved, actual


def smolvla_state_contract(
    *, modeling_sha256: str, bridge_sha256: str
) -> dict[str, Any]:
    """Reproduce the exact schema-v5 collector/loader prefix-state contract."""

    modeling = _require_sha256(modeling_sha256, "SmolVLA modeling SHA256")
    bridge = _require_sha256(bridge_sha256, "SmolVLA VLM/expert bridge SHA256")
    base: dict[str, Any] = {
        "policy": SOURCE_POLICY,
        "anchor": "contextualized_vlm_prefix_final_state_token_before_flow_noise_v1",
        "source": "policy.model.vlm_with_expert.get_vlm_model().text_model.norm",
        "hidden_dim": STATE_INPUT_DIM,
        "prefix_length": 0,
        "noise_independence": "bit_exact_at_group_intervention_query",
        "modeling_sha256": modeling,
        "bridge_sha256": bridge,
    }
    return {**base, "calibration_id": canonical_sha256(base)}


def make_config() -> EventWorldModelConfig:
    """Return the fixed SmolVLA-native schema-v5 architecture contract."""

    return EventWorldModelConfig(
        state_input_dim=STATE_INPUT_DIM,
        action_dim=ACTION_DIM,
        proprio_dim=PROPRIO_DIM,
        object_delta_dim=OBJECT_DELTA_DIM,
        num_bodies=2,
        num_policies=2,
        event_names=EVENT_NAMES,
        predicate_names=PREDICATE_NAMES,
        relative_transition_names=RELATIVE_TRANSITION_NAMES,
        structured_events=True,
        recovery_supervised=False,
        action_rank_residual=False,
        action_rank_success_only=False,
    )


def _deterministically_initialized_model(
    initialization_seed: int,
) -> ActionConditionedEventWorldModel:
    if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
        raise ValueError("initialization_seed must be an integer")
    if not 0 <= initialization_seed < 2**63:
        raise ValueError("initialization_seed must lie in [0, 2**63)")
    # fork_rng prevents this data-blind constructor from mutating caller RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        model = ActionConditionedEventWorldModel(make_config()).cpu()
    state = model.state_dict()
    with torch.no_grad():
        state[BODY_EMBEDDING][1].copy_(state[BODY_EMBEDDING][0])
        state[POLICY_EMBEDDING][1].copy_(state[POLICY_EMBEDDING][0])
    return model


def _row_contract(
    *,
    mapping_name: str,
    identity: str,
    row: int,
    parameter: str,
    tensor: torch.Tensor,
    source_identity: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mapping_name": mapping_name,
        "identity": identity,
        "id": row,
        "row": row,
        "parameter": parameter,
        "tensor_sha256": tensor_sha256(tensor),
    }
    if source_identity is not None:
        result.update(
            {
                "initializer": "bit_exact_clone_source_row_data_blind_v1",
                "source_identity": source_identity,
                "source_row": 0,
            }
        )
    return result


def _build_payload(
    *,
    event_spec: Path,
    event_spec_sha256: str,
    source_manifest: Path,
    source_manifest_sha256: str,
    source_split: Path,
    source_split_sha256: str,
    modeling_sha256: str,
    bridge_sha256: str,
    initialization_seed: int,
) -> dict[str, Any]:
    model = _deterministically_initialized_model(initialization_seed)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    config = model.config_dict()
    body = state[BODY_EMBEDDING]
    policy = state[POLICY_EMBEDDING]
    source_identity_rows = {
        "body": _row_contract(
            mapping_name="body_to_id",
            identity=SOURCE_BODY,
            row=0,
            parameter=BODY_EMBEDDING,
            tensor=body[0],
        ),
        "policy": _row_contract(
            mapping_name="policy_to_id",
            identity=SOURCE_POLICY,
            row=0,
            parameter=POLICY_EMBEDDING,
            tensor=policy[0],
        ),
    }
    reserved_target_rows = {
        "body": _row_contract(
            mapping_name="body_to_id",
            identity=TARGET_BODY,
            row=1,
            parameter=BODY_EMBEDDING,
            tensor=body[1],
            source_identity=SOURCE_BODY,
        ),
        "policy": _row_contract(
            mapping_name="policy_to_id",
            identity=TARGET_POLICY,
            row=1,
            parameter=POLICY_EMBEDDING,
            tensor=policy[1],
            source_identity=SOURCE_POLICY,
        ),
    }
    contract: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "compatible_rollout_schema_version": 5,
        "event_spec_path": str(event_spec),
        "event_spec_sha256": event_spec_sha256,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "source_split_path": str(source_split),
        "source_split_sha256": source_split_sha256,
        "events": list(EVENT_NAMES),
        "predicate_names": list(PREDICATE_NAMES),
        "relative_transition_names": list(RELATIVE_TRANSITION_NAMES),
        "object_target": "single_selected_object_xyz_delta",
        "body_to_id": {SOURCE_BODY: 0, f"__reserved__{TARGET_BODY}": 1},
        "policy_to_id": {SOURCE_POLICY: 0, f"__reserved__{TARGET_POLICY}": 1},
        "policy_feature_action_bridge": build_policy_feature_action_bridge_contract(
            policy=SOURCE_POLICY,
            state_feature_source_sha256=bridge_sha256,
            policy_row=0,
            native_action_dim=ACTION_DIM,
            model_action_dim=ACTION_DIM,
            model_slots=tuple(range(ACTION_DIM)),
            structured_events=True,
        ),
        "source_identity_rows": source_identity_rows,
        "reserved_target_rows": reserved_target_rows,
        "state_contracts": {
            SOURCE_POLICY: smolvla_state_contract(
                modeling_sha256=modeling_sha256,
                bridge_sha256=bridge_sha256,
            )
        },
        "initialization": {
            "algorithm": (
                "torch_cpu_manual_seed_default_initialization_then_"
                "reserved_embedding_rows_bit_exact_source_clone_v1"
            ),
            "seed": initialization_seed,
            "device": "cpu",
            # ``torch.__version__`` is a TorchVersion subclass on newer builds;
            # materialize a plain string so weights_only=True needs no allowlist.
            "torch_version": str(torch.__version__),
            "rng_scope": "torch.random.fork_rng_devices_empty",
            "model_state_sha256": state_dict_sha256(state),
            "config_sha256": canonical_sha256(config),
        },
        "action_normalization": {
            "status": "identity_placeholder_unfitted",
            "action_mean": [0.0] * ACTION_DIM,
            "action_std": [1.0] * ACTION_DIM,
        },
        "protocol_input_access": "bytewise_sha256_only_files_not_parsed",
        "source_rollout_containers_read": False,
        "source_labels_read": False,
        "target_data_read": False,
        "target_labels_read": False,
        "sealed_test_data_read": False,
        "training_performed": False,
        "shared_core_training_performed": False,
        "training_steps": 0,
        "prediction_ready": False,
        "transfer_ready": False,
        "ready_for_protocol_freeze": False,
        "required_next_stage": (
            "source_only_schema5_training_with_both_reserved_rows_restored_"
            "bit_exact_after_every_optimizer_step_then_independent_validation"
        ),
    }
    return {
        "format": FORMAT,
        "model": state,
        "config": config,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
    }


def _prepare_output(path: str | Path) -> Path:
    supplied = Path(path).expanduser()
    output = Path(os.path.abspath(os.fspath(supplied)))
    if _contains_sensitive_component(output):
        raise ValueError("output path is forbidden by the Fresh/confirmation boundary")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent != output.parent.resolve(strict=True):
        raise ValueError("output parent must be materialized and contain no symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    return output


def atomic_torch_publish_new(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a new read-only checkpoint without overwriting a race."""

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            # link(2) is an atomic no-overwrite publication within one directory.
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(path) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError("initialized checkpoint must contain a mapping")
    return dict(value)


def verify_initialized_core(path: str | Path) -> dict[str, Any]:
    """Fail closed on any architecture, provenance, row, or readiness drift."""

    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    payload = _safe_load(checkpoint_path)
    if payload.get("format") != FORMAT:
        raise ValueError("initialized checkpoint format is invalid")
    state = payload.get("model")
    config_value = payload.get("config")
    contract = payload.get("contract")
    if not all(isinstance(value, Mapping) for value in (state, config_value, contract)):
        raise ValueError("initialized checkpoint lacks model/config/contract mappings")
    state = dict(state)
    config_value = dict(config_value)
    contract = dict(contract)
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("initialized model state contains a non-tensor value")
    if payload.get("contract_sha256") != canonical_sha256(contract):
        raise ValueError("initialized checkpoint contract SHA256 is inconsistent")
    expected_config = make_config().to_dict()
    if config_value != expected_config:
        raise ValueError("initialized checkpoint architecture contract changed")
    with torch.random.fork_rng(devices=[]):
        model = ActionConditionedEventWorldModel(
            EventWorldModelConfig.from_dict(config_value)
        )
    model.load_state_dict(state, strict=True)
    if contract.get("status") != STATUS:
        raise ValueError("initialized checkpoint status changed")
    expected_false = (
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
    )
    if any(contract.get(name) is not False for name in expected_false):
        raise ValueError("untrained/data-blind/readiness flags changed")
    if contract.get("training_steps") != 0:
        raise ValueError("initialization artifact must have zero training steps")
    initialization = contract.get("initialization")
    seed = initialization.get("seed") if isinstance(initialization, Mapping) else None
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("torch_version") != str(torch.__version__)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or initialization.get("model_state_sha256") != state_dict_sha256(state)
        or initialization.get("config_sha256") != canonical_sha256(config_value)
    ):
        raise ValueError("initialized state/config audit is inconsistent")
    expected_state = _deterministically_initialized_model(seed).state_dict()
    if set(state) != set(expected_state) or any(
        not torch.equal(state[name], expected_state[name]) for name in state
    ):
        raise ValueError("model state differs from its deterministic initialization")
    if contract.get("body_to_id") != {
        SOURCE_BODY: 0,
        f"__reserved__{TARGET_BODY}": 1,
    } or contract.get("policy_to_id") != {
        SOURCE_POLICY: 0,
        f"__reserved__{TARGET_POLICY}": 1,
    }:
        raise ValueError("source/reserved identity registries changed")
    row_specs = {
        "body": (BODY_EMBEDDING, SOURCE_BODY, TARGET_BODY),
        "policy": (POLICY_EMBEDDING, SOURCE_POLICY, TARGET_POLICY),
    }
    source_rows = contract.get("source_identity_rows")
    target_rows = contract.get("reserved_target_rows")
    if not isinstance(source_rows, Mapping) or not isinstance(target_rows, Mapping):
        raise ValueError("checkpoint lacks auditable source/reserved row contracts")
    for axis, (parameter, source_identity, target_identity) in row_specs.items():
        tensor = state.get(parameter)
        source_spec = source_rows.get(axis)
        target_spec = target_rows.get(axis)
        mapping_name = "body_to_id" if axis == "body" else "policy_to_id"
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != 2
            or not isinstance(source_spec, Mapping)
            or not isinstance(target_spec, Mapping)
            or source_spec.get("mapping_name") != mapping_name
            or source_spec.get("identity") != source_identity
            or source_spec.get("id") != 0
            or source_spec.get("row") != 0
            or source_spec.get("parameter") != parameter
            or source_spec.get("tensor_sha256") != tensor_sha256(tensor[0])
            or target_spec.get("mapping_name") != mapping_name
            or target_spec.get("identity") != target_identity
            or target_spec.get("id") != 1
            or target_spec.get("row") != 1
            or target_spec.get("parameter") != parameter
            or target_spec.get("tensor_sha256") != tensor_sha256(tensor[1])
            or target_spec.get("initializer")
            != "bit_exact_clone_source_row_data_blind_v1"
            or target_spec.get("source_identity") != source_identity
            or target_spec.get("source_row") != 0
            or not torch.equal(tensor[0], tensor[1])
        ):
            raise ValueError(f"{axis} source/reserved row audit is inconsistent")
    for path_name, sha_name in (
        ("event_spec_path", "event_spec_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("source_split_path", "source_split_sha256"),
    ):
        frozen_path = _resolve_protocol_file(str(contract.get(path_name, "")), path_name)
        if file_sha256(frozen_path) != contract.get(sha_name):
            raise ValueError(f"frozen protocol input {path_name} changed")
    state_contracts = contract.get("state_contracts")
    if not isinstance(state_contracts, Mapping) or set(state_contracts) != {
        SOURCE_POLICY
    }:
        raise ValueError("SmolVLA state contract is missing or ambiguous")
    state_contract = state_contracts[SOURCE_POLICY]
    if not isinstance(state_contract, Mapping) or state_contract.get("hidden_dim") != 960:
        raise ValueError("SmolVLA state contract has the wrong hidden dimension")
    expected_state_contract = smolvla_state_contract(
        modeling_sha256=str(state_contract.get("modeling_sha256", "")),
        bridge_sha256=str(state_contract.get("bridge_sha256", "")),
    )
    if dict(state_contract) != expected_state_contract:
        raise ValueError("SmolVLA state contract is not content-addressed correctly")
    bridge_contract = validate_checkpoint_policy_bridge_header(
        config_value, contract
    )
    if (
        bridge_contract.get("policy") != SOURCE_POLICY
        or bridge_contract.get("policy_row") != 0
        or bridge_contract.get("state_feature", {}).get("source_sha256")
        != state_contract.get("bridge_sha256")
    ):
        raise ValueError("SmolVLA policy feature/action bridge changed")
    return {
        "format": FORMAT,
        "status": STATUS,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "contract_sha256": canonical_sha256(contract),
        "model_state_sha256": state_dict_sha256(state),
        "source_body_row_sha256": source_rows["body"]["tensor_sha256"],
        "reserved_body_row_sha256": target_rows["body"]["tensor_sha256"],
        "source_policy_row_sha256": source_rows["policy"]["tensor_sha256"],
        "reserved_policy_row_sha256": target_rows["policy"]["tensor_sha256"],
        "policy_feature_action_bridge_sha256": bridge_contract[
            "contract_sha256"
        ],
        "target_data_read": False,
        "target_labels_read": False,
        "training_performed": False,
        "transfer_ready": False,
    }


def initialize_smolvla_schema5_native_core(
    *,
    output: str | Path,
    event_spec: str | Path,
    event_spec_sha256: str,
    source_manifest: str | Path,
    source_manifest_sha256: str,
    source_split: str | Path,
    source_split_sha256: str,
    state_modeling_sha256: str,
    state_bridge_sha256: str,
    initialization_seed: int = DEFAULT_INITIALIZATION_SEED,
) -> dict[str, Any]:
    """Initialize, atomically publish, and independently verify one core."""

    event_path, event_digest = _bind_protocol_file(
        event_spec, event_spec_sha256, "event spec"
    )
    manifest_path, manifest_digest = _bind_protocol_file(
        source_manifest, source_manifest_sha256, "source manifest"
    )
    split_path, split_digest = _bind_protocol_file(
        source_split, source_split_sha256, "source split"
    )
    modeling_digest = _require_sha256(
        state_modeling_sha256, "SmolVLA modeling SHA256"
    )
    bridge_digest = _require_sha256(
        state_bridge_sha256, "SmolVLA VLM/expert bridge SHA256"
    )
    if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
        raise ValueError("initialization_seed must be an integer")
    output_path = _prepare_output(output)
    payload = _build_payload(
        event_spec=event_path,
        event_spec_sha256=event_digest,
        source_manifest=manifest_path,
        source_manifest_sha256=manifest_digest,
        source_split=split_path,
        source_split_sha256=split_digest,
        modeling_sha256=modeling_digest,
        bridge_sha256=bridge_digest,
        initialization_seed=initialization_seed,
    )
    atomic_torch_publish_new(output_path, payload)
    return verify_initialized_core(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--source-split-sha256", required=True)
    parser.add_argument("--state-modeling-sha256", required=True)
    parser.add_argument("--state-bridge-sha256", required=True)
    parser.add_argument(
        "--initialization-seed", type=int, default=DEFAULT_INITIALIZATION_SEED
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = initialize_smolvla_schema5_native_core(
        output=args.output,
        event_spec=args.event_spec,
        event_spec_sha256=args.event_spec_sha256,
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        source_split=args.source_split,
        source_split_sha256=args.source_split_sha256,
        state_modeling_sha256=args.state_modeling_sha256,
        state_bridge_sha256=args.state_bridge_sha256,
        initialization_seed=args.initialization_seed,
    )
    print(
        "SMOLVLA_SCHEMA5_NATIVE_CORE_INITIALIZED="
        + json.dumps(audit, sort_keys=True, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
