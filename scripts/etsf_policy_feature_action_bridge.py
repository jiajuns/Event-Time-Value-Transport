#!/usr/bin/env python3
"""Strict policy-feature/action bridge contracts for ETSF world models.

The event and action-effect interface is shared, but a policy's latent state is
not.  This module makes that boundary content-addressed and fail-closed for the
two currently implemented policy paths: SmolVLA and OpenVLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_policy_feature_action_bridge_v1"
RUNTIME_FORMAT = "etsf_policy_feature_action_runtime_binding_v1"
CONTRACT_KEY = "policy_feature_action_bridge"
CANONICAL_EVENTS = ("e0", "e12", "e3", "e4", "eK")
CANONICAL_PREDICATES = ("moved", "lifted", "near_goal", "stationary", "success")
CANONICAL_RELATIVE_TRANSITIONS = ("stay", "advance", "skip", "regress")
REVERSIBLE_EVENT_INTERFACE = "canonical_event_id_and_reversible_predicates_v1"
EVENT_ID_ONLY_INTERFACE = "canonical_event_id_only_v1"
ACTION_EFFECT_INTERFACE = "masked_canonical_action_chunk_and_feature_validity_v1"
ACTION_MAPPING = "native_feature_i_to_model_slot_i_no_coordinate_transform_v1"
SHA_CHARS = frozenset("0123456789abcdef")

POLICY_SPECS: dict[str, dict[str, Any]] = {
    "smolvla": {
        "checkpoint_family": "smolvla_native_event_world_model",
        "state_source": (
            "policy.model.vlm_with_expert.get_vlm_model().text_model.norm:"
            "contextualized_final_prefix_state_before_flow_noise"
        ),
        "state_dim": 960,
        "state_adapter": "SmolVLAStateAdapter",
        "action_adapter": "SmolVLAActionAdapter",
    },
    "openvla": {
        "checkpoint_family": "openvla_native_event_world_model",
        "state_source": (
            "language_model.forward:last_hidden_states[:,"
            "-action_dim*num_action_chunks-1]"
        ),
        "state_dim": 4096,
        "state_adapter": "OpenVLAStateAdapter",
        "action_adapter": "OpenVLAActionAdapter",
    },
}
POLICY_ADAPTER_IMPLEMENTATION_FILES = {
    "smolvla": (
        "example_smolvla_event_critic_adapter.py",
        "example_openvla_event_critic_plugin.py",
        "openvla_etsf_event_critic_plugin.py",
    ),
    "openvla": (
        "example_openvla_event_critic_plugin.py",
        "openvla_etsf_event_critic_plugin.py",
    ),
}


class PolicyBridgeContractError(ValueError):
    """A checkpoint/runtime policy boundary is absent or ambiguous."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _positive_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PolicyBridgeContractError(f"{role} must be a positive integer")
    return value


def _nonnegative_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyBridgeContractError(f"{role} must be a non-negative integer")
    return value


def _policy_spec(policy: Any) -> tuple[str, Mapping[str, Any]]:
    name = str(policy)
    if name not in POLICY_SPECS:
        raise PolicyBridgeContractError(f"unsupported policy bridge: {name!r}")
    return name, POLICY_SPECS[name]


def policy_adapter_implementation_sha256(policy: str) -> str:
    """Hash the exact local adapter implementation bundle, rejecting indirection."""

    name, _ = _policy_spec(policy)
    root = Path(__file__).resolve().parent
    rows: list[dict[str, str]] = []
    for filename in POLICY_ADAPTER_IMPLEMENTATION_FILES[name]:
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise PolicyBridgeContractError(
                f"{name} adapter implementation is unavailable or indirect: {filename}"
            )
        rows.append({"path": filename, "sha256": file_sha256(path)})
    return canonical_sha256(rows)


def _canonical_interface(
    model_action_dim: int, *, structured_events: bool
) -> dict[str, Any]:
    return {
        "events": list(CANONICAL_EVENTS),
        "event_interface": (
            REVERSIBLE_EVENT_INTERFACE
            if structured_events
            else EVENT_ID_ONLY_INTERFACE
        ),
        "predicate_names": list(CANONICAL_PREDICATES) if structured_events else [],
        "relative_transition_names": (
            list(CANONICAL_RELATIVE_TRANSITIONS) if structured_events else []
        ),
        "action_effect_interface": ACTION_EFFECT_INTERFACE,
        "model_action_dim": model_action_dim,
    }


def _action_mapping(
    *,
    native_action_dim: int,
    model_action_dim: int,
    model_slots: Sequence[int],
    adapter: str,
    adapter_sha256: str,
) -> dict[str, Any]:
    return {
        "mapping": ACTION_MAPPING,
        "native_action_dim": native_action_dim,
        "model_action_dim": model_action_dim,
        "model_slots": list(model_slots),
        "adapter": adapter,
        "adapter_sha256": adapter_sha256,
    }


def build_policy_feature_action_bridge_contract(
    *,
    policy: str,
    state_feature_source_sha256: str,
    state_adapter_sha256: str | None = None,
    action_adapter_sha256: str | None = None,
    policy_row: int,
    native_action_dim: int = 14,
    model_action_dim: int = 14,
    model_slots: Sequence[int] | None = None,
    structured_events: bool = True,
) -> dict[str, Any]:
    """Build a content-addressed checkpoint-side bridge contract."""

    name, spec = _policy_spec(policy)
    native_dim = _positive_int(native_action_dim, "native action dimension")
    model_dim = _positive_int(model_action_dim, "model action dimension")
    slots = tuple(range(native_dim)) if model_slots is None else tuple(model_slots)
    if not isinstance(structured_events, bool):
        raise PolicyBridgeContractError("structured_events must be bool")
    implementation_sha = policy_adapter_implementation_sha256(name)
    if state_adapter_sha256 not in (None, implementation_sha) or action_adapter_sha256 not in (
        None,
        implementation_sha,
    ):
        raise PolicyBridgeContractError(
            "declared adapter SHA256 does not match the local implementation bundle"
        )
    unsigned = {
        "format": FORMAT,
        "policy": name,
        "checkpoint_family": spec["checkpoint_family"],
        "canonical_interface": _canonical_interface(
            model_dim, structured_events=structured_events
        ),
        "state_feature": {
            "source": spec["state_source"],
            "source_sha256": state_feature_source_sha256,
            "dimension": spec["state_dim"],
            "adapter": spec["state_adapter"],
            "adapter_sha256": implementation_sha,
        },
        "action_mapping": _action_mapping(
            native_action_dim=native_dim,
            model_action_dim=model_dim,
            model_slots=slots,
            adapter=str(spec["action_adapter"]),
            adapter_sha256=implementation_sha,
        ),
        "policy_row": _nonnegative_int(policy_row, "policy row"),
    }
    value = {**unsigned, "contract_sha256": canonical_sha256(unsigned)}
    validate_policy_feature_action_bridge_contract(value)
    return value


def build_runtime_policy_bridge_binding(
    *,
    policy: str,
    state_feature_source_sha256: str,
    state_feature_dimension: int,
    state_adapter: str,
    state_adapter_sha256: str | None = None,
    action_adapter: str,
    action_adapter_sha256: str | None = None,
    policy_row: int,
    native_action_dim: int = 14,
    model_action_dim: int = 14,
    model_slots: Sequence[int] | None = None,
    structured_events: bool = True,
) -> dict[str, Any]:
    """Bind the actual runtime hook and adapter implementation."""

    name, spec = _policy_spec(policy)
    native_dim = _positive_int(native_action_dim, "runtime native action dimension")
    model_dim = _positive_int(model_action_dim, "runtime model action dimension")
    slots = tuple(range(native_dim)) if model_slots is None else tuple(model_slots)
    if not isinstance(structured_events, bool):
        raise PolicyBridgeContractError("runtime structured_events must be bool")
    implementation_sha = policy_adapter_implementation_sha256(name)
    if state_adapter_sha256 not in (None, implementation_sha) or action_adapter_sha256 not in (
        None,
        implementation_sha,
    ):
        raise PolicyBridgeContractError(
            "runtime adapter SHA256 does not match the local implementation bundle"
        )
    unsigned = {
        "format": RUNTIME_FORMAT,
        "policy": name,
        "canonical_interface": _canonical_interface(
            model_dim, structured_events=structured_events
        ),
        "state_feature": {
            "source": spec["state_source"],
            "source_sha256": state_feature_source_sha256,
            "dimension": _positive_int(
                state_feature_dimension, "runtime state feature dimension"
            ),
            "adapter": str(state_adapter),
            "adapter_sha256": implementation_sha,
        },
        "action_mapping": _action_mapping(
            native_action_dim=native_dim,
            model_action_dim=model_dim,
            model_slots=slots,
            adapter=str(action_adapter),
            adapter_sha256=implementation_sha,
        ),
        "policy_row": _nonnegative_int(policy_row, "runtime policy row"),
    }
    value = {**unsigned, "binding_sha256": canonical_sha256(unsigned)}
    _validate_runtime_binding(value)
    return value


def _validate_shared_payload(
    value: Mapping[str, Any], *, policy: str, spec: Mapping[str, Any], runtime: bool
) -> None:
    interface = value.get("canonical_interface")
    state = value.get("state_feature")
    action = value.get("action_mapping")
    if not isinstance(interface, Mapping) or set(interface) != {
        "events", "event_interface", "predicate_names", "relative_transition_names",
        "action_effect_interface", "model_action_dim",
    }:
        raise PolicyBridgeContractError("canonical policy interface fields changed")
    event_interface = interface["event_interface"]
    if event_interface not in (REVERSIBLE_EVENT_INTERFACE, EVENT_ID_ONLY_INTERFACE):
        raise PolicyBridgeContractError("canonical event interface is unsupported")
    claimed_structured = event_interface == REVERSIBLE_EVENT_INTERFACE
    if dict(interface) != _canonical_interface(
        _positive_int(interface["model_action_dim"], "interface model action dimension"),
        structured_events=claimed_structured,
    ):
        raise PolicyBridgeContractError("canonical event/action-effect interface changed")
    if not isinstance(state, Mapping) or set(state) != {
        "source", "source_sha256", "dimension", "adapter", "adapter_sha256"
    }:
        raise PolicyBridgeContractError("state feature bridge fields changed")
    if state["source"] != spec["state_source"]:
        raise PolicyBridgeContractError(f"{policy} state feature source changed")
    implementation_sha = policy_adapter_implementation_sha256(policy)
    if (
        not _is_sha256(state["source_sha256"])
        or state["adapter"] != spec["state_adapter"]
        or state["adapter_sha256"] != implementation_sha
    ):
        raise PolicyBridgeContractError("state feature/adapter SHA256 is invalid")
    state_dim = _positive_int(state["dimension"], "state feature dimension")
    if not runtime and state_dim != spec["state_dim"]:
        raise PolicyBridgeContractError(f"{policy} native state dimension changed")
    if not isinstance(action, Mapping) or set(action) != {
        "mapping", "native_action_dim", "model_action_dim", "model_slots",
        "adapter", "adapter_sha256",
    }:
        raise PolicyBridgeContractError("action bridge fields changed")
    native_dim = _positive_int(action["native_action_dim"], "native action dimension")
    model_dim = _positive_int(action["model_action_dim"], "model action dimension")
    slots = action["model_slots"]
    if (
        action["mapping"] != ACTION_MAPPING
        or not isinstance(slots, list)
        or len(slots) != native_dim
        or any(isinstance(slot, bool) or not isinstance(slot, int) for slot in slots)
        or len(set(slots)) != len(slots)
        or min(slots, default=-1) < 0
        or max(slots, default=model_dim) >= model_dim
        or interface["model_action_dim"] != model_dim
        or action["adapter"] != spec["action_adapter"]
        or action["adapter_sha256"] != implementation_sha
    ):
        raise PolicyBridgeContractError("action mapping is invalid or ambiguous")
    _nonnegative_int(value.get("policy_row"), "policy row")


def validate_policy_feature_action_bridge_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "format", "policy", "checkpoint_family", "canonical_interface",
        "state_feature", "action_mapping", "policy_row", "contract_sha256",
    }:
        raise PolicyBridgeContractError("policy bridge contract schema changed")
    name, spec = _policy_spec(value.get("policy"))
    if value["format"] != FORMAT or value["checkpoint_family"] != spec["checkpoint_family"]:
        raise PolicyBridgeContractError("policy bridge format/checkpoint family changed")
    _validate_shared_payload(value, policy=name, spec=spec, runtime=False)
    unsigned = dict(value)
    recorded = unsigned.pop("contract_sha256")
    if not _is_sha256(recorded) or recorded != canonical_sha256(unsigned):
        raise PolicyBridgeContractError("policy bridge contract SHA256 mismatch")
    return dict(value)


def _validate_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "format", "policy", "canonical_interface", "state_feature",
        "action_mapping", "policy_row", "binding_sha256",
    }:
        raise PolicyBridgeContractError("runtime policy binding schema changed")
    name, spec = _policy_spec(value.get("policy"))
    if value["format"] != RUNTIME_FORMAT:
        raise PolicyBridgeContractError("runtime policy binding format changed")
    _validate_shared_payload(value, policy=name, spec=spec, runtime=True)
    unsigned = dict(value)
    recorded = unsigned.pop("binding_sha256")
    if not _is_sha256(recorded) or recorded != canonical_sha256(unsigned):
        raise PolicyBridgeContractError("runtime policy binding SHA256 mismatch")
    return dict(value)


def _checkpoint_header(
    config: Mapping[str, Any], checkpoint_contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(config, Mapping) or not isinstance(checkpoint_contract, Mapping):
        raise PolicyBridgeContractError("checkpoint config/contract must be mappings")
    raw = checkpoint_contract.get(CONTRACT_KEY)
    if not isinstance(raw, Mapping):
        raise PolicyBridgeContractError("checkpoint lacks strict policy feature/action bridge")
    bridge = validate_policy_feature_action_bridge_contract(raw)
    state_dim = _positive_int(config.get("state_input_dim"), "checkpoint state dimension")
    action_dim = _positive_int(config.get("action_dim"), "checkpoint action dimension")
    num_policies = _positive_int(config.get("num_policies"), "checkpoint policy count")
    event_names = tuple(config.get("event_names", ()))
    mapping = checkpoint_contract.get("policy_to_id")
    policy = str(bridge["policy"])
    row = int(bridge["policy_row"])
    if (
        state_dim != int(bridge["state_feature"]["dimension"])
        or action_dim != int(bridge["action_mapping"]["model_action_dim"])
        or event_names != CANONICAL_EVENTS
        or row >= num_policies
        or not isinstance(mapping, Mapping)
        or mapping.get(policy) != row
    ):
        raise PolicyBridgeContractError(
            "checkpoint config/policy row does not match its policy bridge"
        )
    interface = bridge["canonical_interface"]
    if interface["event_interface"] == REVERSIBLE_EVENT_INTERFACE and (
        config.get("structured_events") is not True
        or tuple(config.get("predicate_names", ())) != CANONICAL_PREDICATES
        or tuple(config.get("relative_transition_names", ()))
        != CANONICAL_RELATIVE_TRANSITIONS
    ):
        raise PolicyBridgeContractError(
            "reversible event interface requires structured heads and exact vocabularies"
        )
    return bridge


def validate_checkpoint_policy_bridge_header(
    config: Mapping[str, Any], checkpoint_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the embedded bridge without authorizing a runtime."""

    return _checkpoint_header(config, checkpoint_contract)


def verify_checkpoint_policy_bridge(
    *,
    config: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
    expected_policy: str,
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify checkpoint and live hook/action adapter as one exact boundary."""

    expected, _ = _policy_spec(expected_policy)
    bridge = _checkpoint_header(config, checkpoint_contract)
    actual = str(bridge["policy"])
    if actual != expected:
        if actual == "smolvla" and expected == "openvla":
            raise PolicyBridgeContractError(
                "direct SmolVLA 960D checkpoint use for OpenVLA is forbidden"
            )
        raise PolicyBridgeContractError(
            f"checkpoint policy {actual!r} does not match runtime {expected!r}"
        )
    runtime = _validate_runtime_binding(runtime_binding)
    if runtime["policy"] != expected:
        raise PolicyBridgeContractError("runtime binding declares a different policy")
    shared_bridge = {
        key: bridge[key]
        for key in (
            "policy", "canonical_interface", "state_feature", "action_mapping",
            "policy_row",
        )
    }
    shared_runtime = {
        key: runtime[key]
        for key in (
            "policy", "canonical_interface", "state_feature", "action_mapping",
            "policy_row",
        )
    }
    if shared_bridge != shared_runtime:
        raise PolicyBridgeContractError(
            "runtime state source/dimension/action mapping/policy adapter differs from checkpoint"
        )
    receipt = {
        "status": "verified_exact_policy_feature_action_bridge",
        "policy": expected,
        "checkpoint_family": bridge["checkpoint_family"],
        "bridge_contract_sha256": bridge["contract_sha256"],
        "runtime_binding_sha256": runtime["binding_sha256"],
        "state_feature_source_sha256": bridge["state_feature"]["source_sha256"],
        "state_feature_dimension": bridge["state_feature"]["dimension"],
        "state_feature_binding_sha256": canonical_sha256(bridge["state_feature"]),
        "action_mapping": bridge["action_mapping"]["mapping"],
        "action_mapping_binding_sha256": canonical_sha256(bridge["action_mapping"]),
        "policy_row": bridge["policy_row"],
        "canonical_event_interface": bridge["canonical_interface"]["event_interface"],
        "canonical_action_effect_interface": ACTION_EFFECT_INTERFACE,
        "cross_policy_latent_reuse_allowed": False,
    }
    receipt["verification_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_checkpoint_file(
    *, checkpoint_path: Path, runtime_binding_path: Path, expected_policy: str
) -> dict[str, Any]:
    """File-oriented verifier used by the CLI and deployment preflights."""

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise PolicyBridgeContractError("checkpoint payload is not a mapping")
    runtime = json.loads(runtime_binding_path.read_text(encoding="utf-8"))
    receipt = verify_checkpoint_policy_bridge(
        config=checkpoint.get("config", {}),
        checkpoint_contract=checkpoint.get("contract", {}),
        expected_policy=expected_policy,
        runtime_binding=runtime,
    )
    return {
        **receipt,
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "runtime_binding_file_sha256": file_sha256(runtime_binding_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--expected-policy", choices=sorted(POLICY_SPECS), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_checkpoint_file(
                checkpoint_path=args.checkpoint.resolve(),
                runtime_binding_path=args.runtime_binding.resolve(),
                expected_policy=args.expected_policy,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
