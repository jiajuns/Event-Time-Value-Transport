#!/usr/bin/env python3
"""Formal monitor-only target-adapter trainer for a frozen ETSF core.

The trainer consumes a content-addressed structured-array artifact derived from
schema-v6 pose-quality data.  It trains only:

* a 960 -> 4096 state adapter;
* separate policy-action and body-action affine adapters;
* decision-step clock parameters; and
* one source-training-time-reserved policy row plus one reserved body row.

Every other source-core tensor/embedding row is restored after every optimizer
step and audited bit-exact at the end.  A checkpoint with merely one policy and
one body, a post-hoc vocabulary expansion, or no source-only retraining proof is
rejected.  Outputs are content-addressed and monitor-only; this module has no
selector, actor, simulator, HDF, Fresh, or execution interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from etsf_transfer_adapters import AffineActionEffectProjector, LowRankStateProjector
from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


FORMAT = "etsf_formal_frozen_core_target_adapter_checkpoint_v1"
RECEIPT_FORMAT = "etsf_formal_frozen_core_target_adapter_receipt_v1"
INPUT_FORMAT = "etsf_formal_transfer_structured_arrays_v1"
INPUT_STATUS = "complete_target_adaptation_structured_arrays"
RESERVATION_FORMAT = "etsf_formal_dual_target_reservation_v1"
RESERVATION_STATUS = "source_core_ready_for_formal_target_adaptation"
MODEL_FORMAT = "etsf_formal_frozen_core_transfer_model_v1"
TARGET_ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
TARGET_BODY = "piper_piper_0.6"
TARGET_STATE_DIM = 960
CORE_STATE_DIM = 4096
ACTION_DIM = 14
ACTION_CHUNK = 50
MIN_BINARY_CLASS_GROUPS = 50
BODY_EMBEDDING = "action_encoder.body_embedding.weight"
POLICY_EMBEDDING = "action_encoder.policy_embedding.weight"
ALLOWED_EMBEDDINGS = (BODY_EMBEDDING, POLICY_EMBEDDING)


class FormalTransferError(ValueError):
    """Fail-closed source, data, or immutable-core contract violation."""


@dataclass(frozen=True)
class FormalTrainingConfig:
    steps: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 20261004
    state_bottleneck_dim: int = 128
    next_event_weight: float = 1.0
    destination_weight: float = 1.0
    duration_weight: float = 0.2
    predicate_weight: float = 0.5
    object_weight: float = 0.2
    success_weight: float = 0.2
    recovery_weight: float = 0.2
    min_binary_class_groups: int = MIN_BINARY_CLASS_GROUPS

    def __post_init__(self) -> None:
        integer_fields = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "state_bottleneck_dim": self.state_bottleneck_dim,
            "min_binary_class_groups": self.min_binary_class_groups,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields.values()):
            raise FormalTransferError("training integer fields must be exact integers")
        if min(self.steps, self.batch_size, self.state_bottleneck_dim, self.min_binary_class_groups) < 1:
            raise FormalTransferError("training steps/batch/bottleneck/support must be positive")
        numeric = (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.next_event_weight,
            self.destination_weight,
            self.duration_weight,
            self.predicate_weight,
            self.object_weight,
            self.success_weight,
            self.recovery_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric):
            raise FormalTransferError("training numeric fields must be finite/non-negative")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise FormalTransferError("learning rate and gradient clip must be positive")


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
    # ``view(dtype)`` rejects a rank-0 tensor even though its storage is valid.
    return (
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
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


def structured_payload_sha256(value: Any, *, excluded_keys: set[str] | None = None) -> str:
    """Hash nested JSON-like values and tensors without torch serialization bytes."""

    excluded = excluded_keys or set()
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor:")
            digest.update(_json_bytes([str(tensor.dtype), list(tensor.shape)]))
            digest.update(_tensor_bytes(tensor))
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            keys = sorted(str(key) for key in item if str(key) not in excluded)
            digest.update(len(keys).to_bytes(8, "big"))
            for key in keys:
                update(key)
                update(item[key])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            digest.update(b"sequence:")
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif isinstance(item, str):
            raw = item.encode("utf-8")
            digest.update(b"string:" + len(raw).to_bytes(8, "big") + raw)
        elif isinstance(item, bytes):
            digest.update(b"bytes:" + len(item).to_bytes(8, "big") + item)
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, bool):
            digest.update(b"bool:1" if item else b"bool:0")
        elif isinstance(item, int):
            digest.update(f"int:{item}".encode("ascii"))
        elif isinstance(item, float):
            digest.update(b"float:" + struct.pack(">d", item))
        else:
            raise FormalTransferError(f"unsupported payload value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _reject_sensitive_path(path: str | Path, role: str, *, must_exist: bool = True) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise FormalTransferError(f"{role} must be absolute")
    resolved = raw.resolve(strict=False)
    if any(
        token in part.casefold()
        for part in (*raw.parts, *resolved.parts)
        for token in ("fresh", "confirmation")
    ):
        raise FormalTransferError(f"{role} must not reference Fresh/confirmation")
    if must_exist and not resolved.is_file():
        raise FormalTransferError(f"{role} is not an existing file")
    return resolved


def _signed(value: Mapping[str, Any], key: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise FormalTransferError(f"{role} signature mismatch")
    return str(recorded)


def _load_torch_mapping(path: Path, role: str) -> dict[str, Any]:
    try:
        numpy_globals = [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.float32)),
        ]
        with torch.serialization.safe_globals(numpy_globals):
            value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FormalTransferError(f"{role} is not a safe torch mapping") from exc
    if not isinstance(value, Mapping):
        raise FormalTransferError(f"{role} must contain a mapping")
    return dict(value)


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalTransferError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FormalTransferError(f"{role} must contain a JSON object")
    return value


def _immutable_core_sha256(
    state: Mapping[str, torch.Tensor], *, allowed_rows: Mapping[str, int]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        if name not in allowed_rows:
            digest.update(_tensor_bytes(tensor))
            continue
        row = allowed_rows[name]
        if tensor.ndim != 2 or row < 0 or row >= tensor.shape[0]:
            raise FormalTransferError(f"reserved row is invalid for {name}")
        if row:
            digest.update(_tensor_bytes(tensor[:row]))
        if row + 1 < tensor.shape[0]:
            digest.update(_tensor_bytes(tensor[row + 1 :]))
    return digest.hexdigest()


def validate_dual_reservation(
    checkpoint: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    source_split_sha256: str,
) -> dict[str, Any]:
    """Require both target rows to predate verified source-only retraining."""

    state = checkpoint.get("model")
    config = checkpoint.get("config")
    contract = checkpoint.get("contract")
    proof = checkpoint.get("formal_target_reservation")
    if not isinstance(state, Mapping) or not state or any(
        not torch.is_tensor(value) for value in state.values()
    ):
        raise FormalTransferError("source checkpoint lacks a tensor model state")
    if not isinstance(config, Mapping) or not isinstance(contract, Mapping):
        raise FormalTransferError("source checkpoint lacks config/contract mappings")
    if not isinstance(proof, Mapping):
        if checkpoint.get("transfer_source_core_expansion") is not None:
            raise FormalTransferError(
                "post-hoc/single-axis expansion is not trainable: reserve both target body and "
                "policy rows, then perform source-only retraining on the exact frozen source split"
            )
        raise FormalTransferError(
            "source core has no reserved target body/policy rows; data-blind expansion plus "
            "exact-source-split source-only retraining proof is required"
        )
    signature = _signed(proof, "reservation_sha256", "dual reservation proof")
    expected_fields = {
        "format",
        "status",
        "target_body_name",
        "target_body_id",
        "target_body_row",
        "target_policy_name",
        "target_policy_id",
        "target_policy_row",
        "body_embedding_parameter",
        "policy_embedding_parameter",
        "reserved_body_row_sha256",
        "reserved_policy_row_sha256",
        "source_manifest_sha256",
        "source_split_sha256",
        "source_training_steps",
        "source_training_groups",
        "target_data_read",
        "target_labels_read",
        "reserved_rows_used_in_source_batches",
        "reserved_rows_unchanged_during_source_training",
        "shared_core_retrained",
        "source_core_state_sha256",
        "input_dual_expanded_checkpoint_sha256",
        "reservation_sha256",
    }
    if set(proof) != expected_fields:
        raise FormalTransferError("dual reservation proof fields changed")
    body_id = proof["target_body_id"]
    body_row = proof["target_body_row"]
    policy_id = proof["target_policy_id"]
    policy_row = proof["target_policy_row"]
    if (
        proof["format"] != RESERVATION_FORMAT
        or proof["status"] != RESERVATION_STATUS
        or proof["target_body_name"] != TARGET_BODY
        or proof["target_policy_name"] != "smolvla"
        or isinstance(body_id, bool)
        or isinstance(body_row, bool)
        or isinstance(policy_id, bool)
        or isinstance(policy_row, bool)
        or not all(isinstance(value, int) for value in (body_id, body_row, policy_id, policy_row))
        or body_id != body_row
        or policy_id != policy_row
        or min(body_id, policy_id) < 1
        or proof["body_embedding_parameter"] != BODY_EMBEDDING
        or proof["policy_embedding_parameter"] != POLICY_EMBEDDING
        or proof["source_manifest_sha256"] != source_manifest_sha256
        or proof["source_split_sha256"] != source_split_sha256
        or isinstance(proof["source_training_steps"], bool)
        or not isinstance(proof["source_training_steps"], int)
        or proof["source_training_steps"] <= 0
        or isinstance(proof["source_training_groups"], bool)
        or not isinstance(proof["source_training_groups"], int)
        or proof["source_training_groups"] <= 0
        or proof["target_data_read"] is not False
        or proof["target_labels_read"] is not False
        or proof["reserved_rows_used_in_source_batches"] is not False
        or proof["reserved_rows_unchanged_during_source_training"] is not True
        or proof["shared_core_retrained"] is not True
        or not _is_sha(proof["input_dual_expanded_checkpoint_sha256"])
    ):
        raise FormalTransferError("dual reservation source-only retraining proof is invalid")
    if BODY_EMBEDDING not in state or POLICY_EMBEDDING not in state:
        raise FormalTransferError("source checkpoint lacks body/policy embeddings")
    body_embedding = state[BODY_EMBEDDING]
    policy_embedding = state[POLICY_EMBEDDING]
    if (
        body_embedding.ndim != 2
        or policy_embedding.ndim != 2
        or body_row >= body_embedding.shape[0]
        or policy_row >= policy_embedding.shape[0]
        or int(config.get("num_bodies", -1)) != body_embedding.shape[0]
        or int(config.get("num_policies", -1)) != policy_embedding.shape[0]
    ):
        raise FormalTransferError("reserved vocabulary/config shapes are invalid")
    body_registry = contract.get("body_to_id")
    policy_registry = contract.get("policy_to_id")
    if (
        not isinstance(body_registry, Mapping)
        or not isinstance(policy_registry, Mapping)
        or body_registry.get(f"__reserved__{TARGET_BODY}") != body_id
        or policy_registry.get("__reserved__smolvla") != policy_id
    ):
        raise FormalTransferError("contract registries do not identify both reserved rows")
    if (
        tensor_sha256(body_embedding[body_row]) != proof["reserved_body_row_sha256"]
        or tensor_sha256(policy_embedding[policy_row]) != proof["reserved_policy_row_sha256"]
        or state_dict_sha256(state) != proof["source_core_state_sha256"]
    ):
        raise FormalTransferError("reserved rows/source core differ from retraining proof")
    return {
        "reservation_sha256": signature,
        "target_body_id": body_id,
        "target_body_row": body_row,
        "target_policy_id": policy_id,
        "target_policy_row": policy_row,
        "source_core_state_sha256": proof["source_core_state_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_split_sha256": source_split_sha256,
        "source_only_retraining_verified": True,
    }


class FormalFrozenCoreTransferModel(nn.Module):
    """Frozen core with exactly five trainable target adaptation components."""

    def __init__(
        self,
        core: ActionConditionedEventWorldModel,
        *,
        target_body_row: int,
        target_policy_row: int,
        state_bottleneck_dim: int,
    ) -> None:
        super().__init__()
        if core.config.state_input_dim != CORE_STATE_DIM or core.config.action_dim != ACTION_DIM:
            raise FormalTransferError("formal SmolVLA transfer requires a 4096D/14D source core")
        if target_body_row >= core.config.num_bodies or target_policy_row >= core.config.num_policies:
            raise FormalTransferError("target rows are outside the pre-reserved vocabularies")
        self.core = core
        self.target_body_row = int(target_body_row)
        self.target_policy_row = int(target_policy_row)
        self.state_adapter = LowRankStateProjector(
            TARGET_STATE_DIM, CORE_STATE_DIM, state_bottleneck_dim
        )
        self.action_adapter = AffineActionEffectProjector(ACTION_DIM, ACTION_DIM)
        self.body_action_adapter = AffineActionEffectProjector(ACTION_DIM, ACTION_DIM)
        self.clock_beta = nn.Parameter(torch.zeros(()))
        self.clock_log_step_scale = nn.Parameter(torch.zeros(()))
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        body_embedding = self.core.action_encoder.body_embedding.weight
        policy_embedding = self.core.action_encoder.policy_embedding.weight
        body_embedding.requires_grad_(True)
        policy_embedding.requires_grad_(True)
        body_mask = torch.zeros_like(body_embedding)
        policy_mask = torch.zeros_like(policy_embedding)
        body_mask[self.target_body_row] = 1
        policy_mask[self.target_policy_row] = 1
        self.register_buffer("_body_gradient_mask", body_mask, persistent=False)
        self.register_buffer("_policy_gradient_mask", policy_mask, persistent=False)
        body_embedding.register_hook(lambda grad: grad * self._body_gradient_mask.to(grad))
        policy_embedding.register_hook(lambda grad: grad * self._policy_gradient_mask.to(grad))
        self._allowed_rows = {
            BODY_EMBEDDING: self.target_body_row,
            POLICY_EMBEDDING: self.target_policy_row,
        }
        self._source_core_state = {
            name: value.detach().cpu().clone() for name, value in self.core.state_dict().items()
        }
        self._immutable_before = _immutable_core_sha256(
            self._source_core_state, allowed_rows=self._allowed_rows
        )

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        history_mask: torch.Tensor | None,
        action_mask: torch.Tensor,
        proprio: torch.Tensor,
        current_event_id: torch.Tensor,
        clock_event_id: torch.Tensor,
        current_predicates: torch.Tensor | None,
        dt_decision_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        adapted_state = self.state_adapter(state)
        policy_actions = self.action_adapter(actions)
        body_actions = self.body_action_adapter(policy_actions)
        batch = state.shape[0]
        device = state.device
        body_id = torch.full(
            (batch,), self.target_body_row, dtype=torch.long, device=device
        )
        policy_id = torch.full(
            (batch,), self.target_policy_row, dtype=torch.long, device=device
        )
        beta = self.clock_beta.to(state).expand(batch)
        dt = dt_decision_steps * self.clock_log_step_scale.exp().to(dt_decision_steps)
        return self.core(
            adapted_state,
            body_actions,
            history_mask=history_mask,
            action_mask=action_mask,
            action_feature_mask=torch.ones_like(body_actions, dtype=torch.bool),
            proprio=proprio,
            body_id=body_id,
            policy_id=policy_id,
            current_event_id=current_event_id,
            clock_event_id=clock_event_id,
            current_predicates=current_predicates,
            beta=beta,
            dt=dt,
        )

    @torch.no_grad()
    def enforce_frozen_core(self) -> None:
        current = self.core.state_dict()
        target_values = {
            name: current[name][row].detach().clone()
            for name, row in self._allowed_rows.items()
        }
        for name, frozen in self._source_core_state.items():
            current[name].copy_(frozen.to(device=current[name].device, dtype=current[name].dtype))
        for name, row in self._allowed_rows.items():
            current[name][row].copy_(target_values[name])

    def immutable_core_audit(self) -> dict[str, Any]:
        current = self.core.state_dict()
        after = _immutable_core_sha256(current, allowed_rows=self._allowed_rows)
        if after != self._immutable_before:
            raise FormalTransferError("shared core changed outside the two reserved target rows")
        row_audit = {}
        for name, row in self._allowed_rows.items():
            row_audit[name] = {
                "target_row": row,
                "before_sha256": tensor_sha256(self._source_core_state[name][row]),
                "after_sha256": tensor_sha256(current[name][row]),
                "changed": not torch.equal(
                    self._source_core_state[name][row], current[name][row].detach().cpu()
                ),
            }
        return {
            "immutable_before_sha256": self._immutable_before,
            "immutable_after_sha256": after,
            "all_non_target_core_values_bit_exact": True,
            "allowed_target_rows": row_audit,
        }

    def trainable_parameter_audit(self) -> dict[str, Any]:
        names = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        expected_core = {f"core.{BODY_EMBEDDING}", f"core.{POLICY_EMBEDDING}"}
        unexpected_core = sorted(
            name for name in names if name.startswith("core.") and name not in expected_core
        )
        if unexpected_core or not expected_core.issubset(names):
            raise FormalTransferError(f"unexpected trainable core parameters: {unexpected_core}")
        effective = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == f"core.{BODY_EMBEDDING}":
                effective += parameter.shape[1]
            elif name == f"core.{POLICY_EMBEDDING}":
                effective += parameter.shape[1]
            else:
                effective += parameter.numel()
        return {
            "trainable_parameter_names": names,
            "effective_trainable_parameters": int(effective),
            "full_core_parameters_trainable": False,
            "only_reserved_core_rows_trainable": True,
        }


REQUIRED_ARRAYS = {
    "state",
    "history_mask",
    "action_chunks",
    "action_mask",
    "proprio",
    "current_event_id",
    "clock_event_id",
    "current_predicates",
    "dt_decision_steps",
    "next_event_id",
    "next_event_mask",
    "destination_event_id",
    "destination_event_mask",
    "duration_log1p_decision_steps",
    "duration_mask",
    "post_predicates",
    "predicate_mask",
    "object_delta_physical",
    "object_delta_supervision_valid",
    "object_delta_invalid_reason_bitset",
    "object_feature_object_index",
    "success",
    "success_mask",
    "recovery",
    "recovery_mask",
    "sample_group_index",
}


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise FormalTransferError(f"structured array {name} must be a tensor")
    return value.detach().cpu().contiguous()


def _finite_float(value: torch.Tensor, name: str) -> torch.Tensor:
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise FormalTransferError(f"{name} must be finite floating point")
    return value.to(torch.float32)


def _bool(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype != torch.bool:
        raise FormalTransferError(f"{name} must be explicit bool")
    return value


def _integer(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise FormalTransferError(f"{name} must be an integer tensor")
    return value.to(torch.int64)


def validate_structured_input(
    payload: Mapping[str, Any], *, core_config: EventWorldModelConfig
) -> dict[str, Any]:
    """Authenticate schema-v6 provenance and tensor shapes/masks."""

    expected_root = {
        "format",
        "status",
        "evidence_scope",
        "schema_version",
        "task",
        "target_actor_id",
        "target_body",
        "split_role",
        "fresh_or_confirmation_data_read",
        "event_spec_sha256",
        "schema6_pose_quality",
        "logical_group_keys",
        "arrays",
        "payload_sha256",
    }
    if set(payload) != expected_root:
        raise FormalTransferError("structured input root fields changed")
    recorded = payload["payload_sha256"]
    if not _is_sha(recorded) or recorded != structured_payload_sha256(
        payload, excluded_keys={"payload_sha256"}
    ):
        raise FormalTransferError("structured input payload SHA mismatch")
    if (
        payload["format"] != INPUT_FORMAT
        or payload["status"] != INPUT_STATUS
        or payload["evidence_scope"] != "nonfresh_target_adaptation_development_only"
        or payload["schema_version"] != 6
        or payload["task"] != "move_can_pot"
        or payload["target_actor_id"] != TARGET_ACTOR_ID
        or payload["target_body"] != TARGET_BODY
        or payload["split_role"] != "target_adaptation"
        or payload["fresh_or_confirmation_data_read"] is not False
        or not _is_sha(payload["event_spec_sha256"])
    ):
        raise FormalTransferError("structured input scope/schema contract changed")
    quality = payload["schema6_pose_quality"]
    if not isinstance(quality, Mapping) or quality != {
        "format": "etsf_schema6_pose_quality_v1",
        "object_registry_sha256": quality.get("object_registry_sha256"),
        "pose_integrity_spec_sha256": quality.get("pose_integrity_spec_sha256"),
        "interval_mask_semantics": "all_destinations_valid_no_reset_or_teleport_crossed",
    } or not _is_sha(quality.get("object_registry_sha256")) or not _is_sha(
        quality.get("pose_integrity_spec_sha256")
    ):
        raise FormalTransferError("schema-v6 pose-quality provenance is invalid")
    group_keys = payload["logical_group_keys"]
    if (
        not isinstance(group_keys, list)
        or not group_keys
        or any(not isinstance(key, str) or not key for key in group_keys)
        or len(set(group_keys)) != len(group_keys)
    ):
        raise FormalTransferError("logical_group_keys must be unique non-empty strings")
    raw_arrays = payload["arrays"]
    if not isinstance(raw_arrays, Mapping) or set(raw_arrays) != REQUIRED_ARRAYS:
        raise FormalTransferError("structured input array fields changed")
    arrays = {name: _tensor(raw_arrays[name], name) for name in REQUIRED_ARRAYS}
    state = _finite_float(arrays["state"], "state")
    if state.ndim not in (2, 3) or state.shape[-1] != TARGET_STATE_DIM:
        raise FormalTransferError("state must be [N,960] or [N,T,960]")
    count = state.shape[0]
    if count < 1:
        raise FormalTransferError("structured input is empty")
    history = _bool(arrays["history_mask"], "history_mask")
    expected_history = (count, 1) if state.ndim == 2 else state.shape[:2]
    if history.shape != expected_history or bool((~history.any(dim=1)).any()):
        raise FormalTransferError("history_mask does not cover every state row")
    actions = _finite_float(arrays["action_chunks"], "action_chunks")
    action_mask = _bool(arrays["action_mask"], "action_mask")
    if actions.shape != (count, ACTION_CHUNK, ACTION_DIM) or action_mask.shape != (
        count,
        ACTION_CHUNK,
    ) or bool((~action_mask.any(dim=1)).any()):
        raise FormalTransferError("SmolVLA actions/mask must be [N,50,14]/[N,50]")
    proprio = _finite_float(arrays["proprio"], "proprio")
    if proprio.shape != (count, core_config.proprio_dim):
        raise FormalTransferError("proprio shape differs from the source core")
    current_event = _integer(arrays["current_event_id"], "current_event_id")
    clock_event = _integer(arrays["clock_event_id"], "clock_event_id")
    next_event = _integer(arrays["next_event_id"], "next_event_id")
    destination = _integer(arrays["destination_event_id"], "destination_event_id")
    for name, value in (
        ("current_event_id", current_event),
        ("clock_event_id", clock_event),
        ("next_event_id", next_event),
        ("destination_event_id", destination),
    ):
        if value.shape != (count,) or bool(((value < 0) | (value >= core_config.num_events)).any()):
            raise FormalTransferError(f"{name} is outside the event vocabulary")
    predicates = _finite_float(arrays["current_predicates"], "current_predicates")
    post_predicates = _finite_float(arrays["post_predicates"], "post_predicates")
    predicate_mask = _bool(arrays["predicate_mask"], "predicate_mask")
    expected_predicate = (count, core_config.num_predicates)
    if (
        not core_config.structured_events
        or predicates.shape != expected_predicate
        or post_predicates.shape != expected_predicate
        or predicate_mask.shape != expected_predicate
        or bool(((predicates < 0) | (predicates > 1)).any())
        or bool(((post_predicates < 0) | (post_predicates > 1)).any())
    ):
        raise FormalTransferError("structured predicate arrays are invalid/incompatible")
    dt = _finite_float(arrays["dt_decision_steps"], "dt_decision_steps")
    duration = _finite_float(
        arrays["duration_log1p_decision_steps"], "duration_log1p_decision_steps"
    )
    if dt.shape != (count,) or duration.shape != (count,) or bool((dt <= 0).any()) or bool((duration < 0).any()):
        raise FormalTransferError("decision-step duration/dt arrays are invalid")
    masks = {}
    for name in (
        "next_event_mask",
        "destination_event_mask",
        "duration_mask",
        "success_mask",
        "recovery_mask",
    ):
        masks[name] = _bool(arrays[name], name)
        if masks[name].shape != (count,):
            raise FormalTransferError(f"{name} must be [N]")
    if not bool(masks["next_event_mask"].any()) or not bool(
        masks["destination_event_mask"].any()
    ) or not bool(masks["duration_mask"].any()):
        raise FormalTransferError("event/destination/duration each require observed supervision")
    success = _finite_float(arrays["success"], "success")
    recovery = _finite_float(arrays["recovery"], "recovery")
    if success.shape != (count,) or recovery.shape != (count,) or bool(
        ((success < 0) | (success > 1) | (recovery < 0) | (recovery > 1)).any()
    ):
        raise FormalTransferError("success/recovery targets must lie in [0,1]")
    object_delta = _finite_float(arrays["object_delta_physical"], "object_delta_physical")
    object_valid = _bool(
        arrays["object_delta_supervision_valid"], "object_delta_supervision_valid"
    )
    object_reason = _integer(
        arrays["object_delta_invalid_reason_bitset"], "object_delta_invalid_reason_bitset"
    )
    feature_object = _integer(
        arrays["object_feature_object_index"], "object_feature_object_index"
    )
    if object_delta.shape != (count, core_config.object_delta_dim):
        raise FormalTransferError("object delta width differs from the source core")
    if object_valid.ndim != 2 or object_valid.shape[0] != count or object_reason.shape != object_valid.shape:
        raise FormalTransferError("schema6 object quality arrays must align [N,O]")
    if bool((object_reason < 0).any()) or not torch.equal(object_valid, object_reason == 0):
        raise FormalTransferError("schema6 valid mask must equal reason_bitset==0")
    if feature_object.shape != (core_config.object_delta_dim,) or bool(
        ((feature_object < 0) | (feature_object >= object_valid.shape[1])).any()
    ):
        raise FormalTransferError("object feature-to-registry mapping is invalid")
    group_index = _integer(arrays["sample_group_index"], "sample_group_index")
    if group_index.shape != (count,) or bool(
        ((group_index < 0) | (group_index >= len(group_keys))).any()
    ) or set(group_index.tolist()) != set(range(len(group_keys))):
        raise FormalTransferError("sample_group_index must cover every logical group")
    for target_name, target, mask in (
        ("success", success, masks["success_mask"]),
        ("recovery", recovery, masks["recovery_mask"]),
    ):
        if bool(mask.any()) and bool(((target[mask] != 0) & (target[mask] != 1)).any()):
            raise FormalTransferError(f"{target_name} supervised targets must be exactly binary")
        for group in torch.unique(group_index[mask]):
            group_values = torch.unique(target[mask & (group_index == group)])
            if group_values.numel() > 1:
                raise FormalTransferError(
                    f"{target_name} labels must be consistent within each logical group"
                )
    normalized = {
        **arrays,
        "state": state,
        "history_mask": history,
        "action_chunks": actions,
        "action_mask": action_mask,
        "proprio": proprio,
        "current_event_id": current_event,
        "clock_event_id": clock_event,
        "current_predicates": predicates,
        "dt_decision_steps": dt,
        "next_event_id": next_event,
        "destination_event_id": destination,
        "duration_log1p_decision_steps": duration,
        "post_predicates": post_predicates,
        "predicate_mask": predicate_mask,
        "object_delta_physical": object_delta,
        "object_delta_supervision_valid": object_valid,
        "object_delta_invalid_reason_bitset": object_reason,
        "object_feature_object_index": feature_object,
        "success": success,
        "recovery": recovery,
        "sample_group_index": group_index,
        **masks,
    }
    return {
        "arrays": normalized,
        "logical_group_keys": list(group_keys),
        "samples": count,
        "groups": len(group_keys),
        "payload_sha256": recorded,
        "event_spec_sha256": payload["event_spec_sha256"],
        "schema6_pose_quality": dict(quality),
    }


def _group_class_support(
    target: torch.Tensor, mask: torch.Tensor, group_index: torch.Tensor
) -> dict[str, int]:
    result = {}
    for class_value, name in ((0, "negative_groups"), (1, "positive_groups")):
        selected = mask & (target == float(class_value))
        result[name] = len(set(group_index[selected].tolist()))
    return result


def binary_support_gates(
    arrays: Mapping[str, torch.Tensor], *, core_config: EventWorldModelConfig, minimum: int
) -> dict[str, Any]:
    success = _group_class_support(
        arrays["success"], arrays["success_mask"], arrays["sample_group_index"]
    )
    recovery = _group_class_support(
        arrays["recovery"], arrays["recovery_mask"], arrays["sample_group_index"]
    )
    success_enabled = min(success.values()) >= minimum
    recovery_enabled = min(recovery.values()) >= minimum and core_config.recovery_supervised
    return {
        "minimum_independent_groups_per_class": minimum,
        "success": {
            **success,
            "enabled": success_enabled,
            "reason": "support_gate_passed" if success_enabled else "insufficient_independent_group_support",
        },
        "recovery": {
            **recovery,
            "core_recovery_supervised": bool(core_config.recovery_supervised),
            "enabled": recovery_enabled,
            "reason": (
                "support_gate_passed"
                if recovery_enabled
                else "core_recovery_head_not_source_supervised"
                if not core_config.recovery_supervised
                else "insufficient_independent_group_support"
            ),
        },
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.to(dtype=torch.bool)
    if not bool(selected.any()):
        return value.sum() * 0.0
    return value[selected].mean()


def _laplace_nll(mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_scale = log_scale.clamp(-8.0, 5.0)
    return log_scale + (target - mean).abs() * torch.exp(-log_scale) + math.log(2.0)


def compute_losses(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    object_target_normalized: torch.Tensor,
    object_feature_mask: torch.Tensor,
    support: Mapping[str, Any],
    config: FormalTrainingConfig,
) -> dict[str, torch.Tensor]:
    next_per = F.cross_entropy(
        output["next_event_logits"], batch["next_event_id"], reduction="none"
    )
    destination_per = F.cross_entropy(
        output["next_reached_event_logits"],
        batch["destination_event_id"],
        reduction="none",
    )
    duration_per = _laplace_nll(
        output["duration_selected_log_mean"],
        output["duration_selected_log_scale"],
        batch["duration_log1p_decision_steps"],
    )
    predicate_per = F.binary_cross_entropy_with_logits(
        output["post_predicate_logits"], batch["post_predicates"], reduction="none"
    )
    object_per = _laplace_nll(
        output["object_delta_mean"],
        output["object_delta_log_scale"],
        object_target_normalized,
    )
    success_per = F.binary_cross_entropy_with_logits(
        output["success_logit"], batch["success"], reduction="none"
    )
    if support["recovery"]["enabled"]:
        if output["outcome_logits"].shape[-1] < 3:
            raise FormalTransferError("enabled recovery supervision requires a recovery outcome")
        recovery_logit = output["outcome_logits"][:, 2] - torch.logsumexp(
            output["outcome_logits"][:, :2], dim=-1
        )
        recovery_per = F.binary_cross_entropy_with_logits(
            recovery_logit, batch["recovery"], reduction="none"
        )
    else:
        recovery_per = output["outcome_logits"].sum(dim=-1) * 0.0
    losses = {
        "next_event": _masked_mean(next_per, batch["next_event_mask"]),
        "destination": _masked_mean(destination_per, batch["destination_event_mask"]),
        "duration": _masked_mean(duration_per, batch["duration_mask"]),
        "predicate": _masked_mean(predicate_per, batch["predicate_mask"]),
        "object": _masked_mean(object_per, object_feature_mask),
        "success": (
            _masked_mean(success_per, batch["success_mask"])
            if support["success"]["enabled"]
            else success_per.sum() * 0.0
        ),
        "recovery": (
            _masked_mean(recovery_per, batch["recovery_mask"])
            if support["recovery"]["enabled"]
            else recovery_per.sum() * 0.0
        ),
    }
    losses["total"] = (
        config.next_event_weight * losses["next_event"]
        + config.destination_weight * losses["destination"]
        + config.duration_weight * losses["duration"]
        + config.predicate_weight * losses["predicate"]
        + config.object_weight * losses["object"]
        + config.success_weight * losses["success"]
        + config.recovery_weight * losses["recovery"]
    )
    return losses


def _normalization(checkpoint: Mapping[str, Any], object_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise FormalTransferError("source checkpoint lacks object normalization")
    try:
        mean = torch.as_tensor(normalization["object_delta_mean"], dtype=torch.float32)
        std = torch.as_tensor(normalization["object_delta_std"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalTransferError("source object normalization is invalid") from exc
    if mean.shape != (object_dim,) or std.shape != mean.shape or not bool(
        torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
    ):
        raise FormalTransferError("source object normalization shape/value mismatch")
    return mean, std


def _batch(arrays: Mapping[str, torch.Tensor], index: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in arrays.items():
        if name == "object_feature_object_index":
            result[name] = value.to(device)
        else:
            result[name] = value[index].to(device)
    return result


def train_model(
    model: FormalFrozenCoreTransferModel,
    *,
    arrays: Mapping[str, torch.Tensor],
    object_mean: torch.Tensor,
    object_std: torch.Tensor,
    support: Mapping[str, Any],
    config: FormalTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run deterministic adapter-only optimization and return an audit summary."""

    torch.manual_seed(config.seed)
    model.to(device)
    trainable = model.trainable_parameter_audit()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sample_count = arrays["state"].shape[0]
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    order = torch.randperm(sample_count, generator=generator)
    cursor = 0
    totals = {name: 0.0 for name in ("total", "next_event", "destination", "duration", "predicate", "object", "success", "recovery")}
    model.train()
    model.core.eval()
    for _ in range(config.steps):
        if cursor + config.batch_size > sample_count:
            order = torch.randperm(sample_count, generator=generator)
            cursor = 0
        index = order[cursor : cursor + min(config.batch_size, sample_count)]
        cursor += len(index)
        batch = _batch(arrays, index, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["state"],
            batch["action_chunks"],
            history_mask=batch["history_mask"],
            action_mask=batch["action_mask"],
            proprio=batch["proprio"],
            current_event_id=batch["current_event_id"],
            clock_event_id=batch["clock_event_id"],
            current_predicates=batch["current_predicates"],
            dt_decision_steps=batch["dt_decision_steps"],
        )
        feature_object = batch["object_feature_object_index"]
        object_mask = batch["object_delta_supervision_valid"][:, feature_object]
        object_target = (
            batch["object_delta_physical"] - object_mean.to(device)
        ) / object_std.to(device)
        losses = compute_losses(
            output,
            batch,
            object_target_normalized=object_target,
            object_feature_mask=object_mask,
            support=support,
            config=config,
        )
        if not bool(torch.isfinite(losses["total"])):
            raise FormalTransferError("adapter training loss became non-finite")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
        optimizer.step()
        model.enforce_frozen_core()
        model.immutable_core_audit()
        for name in totals:
            totals[name] += float(losses[name].detach().cpu())
    immutable = model.immutable_core_audit()
    return {
        "steps": config.steps,
        "mean_losses": {name: value / config.steps for name, value in totals.items()},
        "trainable_parameter_audit": trainable,
        "immutable_core_audit": immutable,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "fixed_order_seed": config.seed,
        },
    }


def _adapter_state(model: FormalFrozenCoreTransferModel) -> dict[str, Any]:
    return {
        "state_adapter": {name: value.detach().cpu() for name, value in model.state_adapter.state_dict().items()},
        "action_adapter": {name: value.detach().cpu() for name, value in model.action_adapter.state_dict().items()},
        "body_action_adapter": {name: value.detach().cpu() for name, value in model.body_action_adapter.state_dict().items()},
        "clock_beta": model.clock_beta.detach().cpu(),
        "clock_log_step_scale": model.clock_log_step_scale.detach().cpu(),
        "target_body_embedding_row": model.core.state_dict()[BODY_EMBEDDING][model.target_body_row].detach().cpu(),
        "target_policy_embedding_row": model.core.state_dict()[POLICY_EMBEDDING][model.target_policy_row].detach().cpu(),
    }


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(value), partial)
    os.replace(partial, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def train_formal_target_adapters(
    *,
    source_checkpoint_path: Path,
    source_manifest_path: Path,
    source_split_path: Path,
    input_path: Path,
    output_dir: Path,
    config: FormalTrainingConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Authenticate, train, audit, and atomically publish monitor-only outputs."""

    source_path = _reject_sensitive_path(source_checkpoint_path, "source checkpoint")
    source_manifest = _reject_sensitive_path(source_manifest_path, "source manifest")
    source_split = _reject_sensitive_path(source_split_path, "source split")
    structured_path = _reject_sensitive_path(input_path, "structured input")
    output = _reject_sensitive_path(output_dir, "output directory", must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    if device != "cpu" and not (device == "cuda" and torch.cuda.is_available()):
        raise FormalTransferError("device must be cpu or an available cuda device")
    checkpoint = _load_torch_mapping(source_path, "source checkpoint")
    source_state = checkpoint.get("model")
    raw_config = checkpoint.get("config")
    if not isinstance(source_state, Mapping) or not isinstance(raw_config, Mapping):
        raise FormalTransferError("source checkpoint lacks model/config")
    core_config = EventWorldModelConfig.from_dict(raw_config)
    if (
        core_config.state_input_dim != CORE_STATE_DIM
        or core_config.action_dim != ACTION_DIM
        or core_config.num_bodies < 2
        or core_config.num_policies < 2
    ):
        raise FormalTransferError(
            "source core has no dual reserved vocabulary (current authoritative core is "
            "4096D/action14 but num_bodies=1,num_policies=1); data-blind expansion and "
            "exact-source-split source-only retraining are required"
        )
    if (
        not core_config.structured_events
        or core_config.outcome_names != ("failure", "success", "recovery")
    ):
        raise FormalTransferError(
            "formal target adaptation requires structured event heads and the exact "
            "failure/success/recovery outcome vocabulary"
        )
    reservation = validate_dual_reservation(
        checkpoint,
        source_manifest_sha256=file_sha256(source_manifest),
        source_split_sha256=file_sha256(source_split),
    )
    core = ActionConditionedEventWorldModel(core_config)
    core.load_state_dict(source_state, strict=True)
    structured_payload = _load_torch_mapping(structured_path, "structured input")
    validated = validate_structured_input(structured_payload, core_config=core_config)
    object_mean, object_std = _normalization(checkpoint, core_config.object_delta_dim)
    support = binary_support_gates(
        validated["arrays"],
        core_config=core_config,
        minimum=config.min_binary_class_groups,
    )
    model = FormalFrozenCoreTransferModel(
        core,
        target_body_row=reservation["target_body_row"],
        target_policy_row=reservation["target_policy_row"],
        state_bottleneck_dim=config.state_bottleneck_dim,
    )
    training = train_model(
        model,
        arrays=validated["arrays"],
        object_mean=object_mean,
        object_std=object_std,
        support=support,
        config=config,
        device=torch.device(device),
    )
    adapter_state = _adapter_state(model)
    adapter_state_sha = structured_payload_sha256(adapter_state)
    source_file_sha = file_sha256(source_path)
    checkpoint_payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete_monitor_only",
        "model_format": MODEL_FORMAT,
        "target_actor_id": TARGET_ACTOR_ID,
        "target_body": TARGET_BODY,
        "source_core": {
            "path": str(source_path),
            "file_sha256": source_file_sha,
            "state_dict_sha256": reservation["source_core_state_sha256"],
            "reservation_sha256": reservation["reservation_sha256"],
            "source_manifest_sha256": reservation["source_manifest_sha256"],
            "source_split_sha256": reservation["source_split_sha256"],
        },
        "structured_input": {
            "path": str(structured_path),
            "file_sha256": file_sha256(structured_path),
            "payload_sha256": validated["payload_sha256"],
            "event_spec_sha256": validated["event_spec_sha256"],
            "schema6_pose_quality": validated["schema6_pose_quality"],
            "samples": validated["samples"],
            "groups": validated["groups"],
        },
        "training_config": asdict(config),
        "binary_support_gates": support,
        "training": training,
        "adapter_state": adapter_state,
        "adapter_state_sha256": adapter_state_sha,
        "authorization": {
            "monitor_only": True,
            "selection_authorized": False,
            "action_ranking_authorized": False,
            "environment_execution_authorized": False,
            "transfer_claim_authorized": False,
            "shared_core_gradient_or_update_authorized": False,
        },
        "fresh_or_confirmation_data_read": False,
    }
    checkpoint_payload["checkpoint_payload_sha256"] = structured_payload_sha256(
        checkpoint_payload
    )
    payload_sha = checkpoint_payload["checkpoint_payload_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_output = output / f"formal_target_adapter_{payload_sha}.pt"
    atomic_torch(checkpoint_output, checkpoint_payload)
    receipt: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "status": "complete_monitor_only",
        "checkpoint": str(checkpoint_output),
        "checkpoint_file_sha256": file_sha256(checkpoint_output),
        "checkpoint_payload_sha256": payload_sha,
        "adapter_state_sha256": adapter_state_sha,
        "source_core_file_sha256": source_file_sha,
        "source_core_state_dict_sha256": reservation["source_core_state_sha256"],
        "source_reservation_sha256": reservation["reservation_sha256"],
        "structured_input_file_sha256": file_sha256(structured_path),
        "structured_input_payload_sha256": validated["payload_sha256"],
        "binary_support_gates": support,
        "immutable_core_audit": training["immutable_core_audit"],
        "selection_authorized": False,
        "monitor_only": True,
        "fresh_or_confirmation_data_read": False,
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_output = output / f"formal_target_adapter_receipt_{payload_sha}.json"
    atomic_json(receipt_output, receipt)
    os.chmod(checkpoint_output, 0o444)
    os.chmod(receipt_output, 0o444)
    return {
        "checkpoint": str(checkpoint_output),
        "receipt": str(receipt_output),
        "checkpoint_file_sha256": receipt["checkpoint_file_sha256"],
        "checkpoint_payload_sha256": payload_sha,
        "receipt_sha256": receipt["receipt_sha256"],
        "selection_authorized": False,
        "monitor_only": True,
    }


def validate_monitor_checkpoint(path: Path) -> dict[str, Any]:
    resolved = _reject_sensitive_path(path, "monitor checkpoint")
    payload = _load_torch_mapping(resolved, "monitor checkpoint")
    unsigned = dict(payload)
    recorded = unsigned.pop("checkpoint_payload_sha256", None)
    if not _is_sha(recorded) or recorded != structured_payload_sha256(unsigned):
        raise FormalTransferError("monitor checkpoint payload SHA mismatch")
    authorization = payload.get("authorization")
    if (
        payload.get("format") != FORMAT
        or payload.get("status") != "complete_monitor_only"
        or payload.get("model_format") != MODEL_FORMAT
        or not isinstance(authorization, Mapping)
        or authorization.get("monitor_only") is not True
        or any(
            authorization.get(key) is not False
            for key in (
                "selection_authorized",
                "action_ranking_authorized",
                "environment_execution_authorized",
                "transfer_claim_authorized",
                "shared_core_gradient_or_update_authorized",
            )
        )
        or payload.get("fresh_or_confirmation_data_read") is not False
        or structured_payload_sha256(payload.get("adapter_state"))
        != payload.get("adapter_state_sha256")
    ):
        raise FormalTransferError("monitor checkpoint authorization/content boundary changed")
    if resolved.name != f"formal_target_adapter_{recorded}.pt":
        raise FormalTransferError("monitor checkpoint filename is not content-addressed")
    return dict(payload)


def validate_receipt(path: Path) -> dict[str, Any]:
    resolved = _reject_sensitive_path(path, "monitor receipt")
    receipt = _load_json(resolved, "monitor receipt")
    _signed(receipt, "receipt_sha256", "monitor receipt")
    checkpoint = _reject_sensitive_path(str(receipt.get("checkpoint", "")), "receipt checkpoint")
    payload = validate_monitor_checkpoint(checkpoint)
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("status") != "complete_monitor_only"
        or receipt.get("selection_authorized") is not False
        or receipt.get("monitor_only") is not True
        or receipt.get("fresh_or_confirmation_data_read") is not False
        or file_sha256(checkpoint) != receipt.get("checkpoint_file_sha256")
        or payload.get("checkpoint_payload_sha256") != receipt.get("checkpoint_payload_sha256")
        or payload.get("adapter_state_sha256") != receipt.get("adapter_state_sha256")
    ):
        raise FormalTransferError("monitor receipt/checkpoint binding changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--state-bottleneck-dim", type=int, default=128)
    args = parser.parse_args()
    result = train_formal_target_adapters(
        source_checkpoint_path=args.source_checkpoint,
        source_manifest_path=args.source_manifest,
        source_split_path=args.source_split,
        input_path=args.input,
        output_dir=args.output_dir,
        config=FormalTrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            state_bottleneck_dim=args.state_bottleneck_dim,
        ),
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FormalFrozenCoreTransferModel",
    "FormalTrainingConfig",
    "FormalTransferError",
    "INPUT_FORMAT",
    "INPUT_STATUS",
    "MIN_BINARY_CLASS_GROUPS",
    "RESERVATION_FORMAT",
    "RESERVATION_STATUS",
    "binary_support_gates",
    "canonical_sha256",
    "compute_losses",
    "file_sha256",
    "state_dict_sha256",
    "structured_payload_sha256",
    "tensor_sha256",
    "train_formal_target_adapters",
    "train_model",
    "validate_dual_reservation",
    "validate_monitor_checkpoint",
    "validate_receipt",
    "validate_structured_input",
]
