#!/usr/bin/env python3
"""Counterfactual fine-tuning for the pluggable ETSF event world model.

The trainer consumes candidate *groups*, not independently shuffled branches.
Schema-v3 groups provide dense post-intervention supervision, while schema-v2
groups are deliberately restricted to terminal success/cost and ranking losses.
All train/validation/test decisions are made at the logical episode key
``(task, body, seed)``; a v3 group supersedes a v2 group for the same key.

One invocation may train several independently initialised fine-tuning members.
It writes member checkpoints plus a deployment manifest containing a shared
validation-only success temperature and a conservative fallback guard.  The
sealed test split is recorded but never loaded or evaluated by this script.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from etsf_policy_feature_action_bridge import (
    CONTRACT_KEY as POLICY_BRIDGE_CONTRACT_KEY,
    validate_checkpoint_policy_bridge_header,
)
from train_openvla_etsf_event_world_model import (
    derive_atomic_predicates,
    dynamic_event_ids,
    event_transition_target,
    relative_transition_ids,
)


SUPPORTED_SCHEMAS = (2, 3, 4, 5)
DEFAULT_EVENTS = ("e0", "e12", "e3", "e4", "eK")
SPLIT_SEED = 20260827
DEFAULT_REGRESSION_PERSISTENCE_STEPS = 3
CAUSAL_HISTORY_MAX_STEPS = 8
CAUSAL_HISTORY_FORMAT = "etsf_same_branch_causal_hidden_history_v1"
SCORING_GRID_VERSION = "validation_scoring_grid_v1"
GUARD_GRID_VERSION = "validation_guard_quantile_grid_v1"
SCORING_SELECTION_RULE = (
    "eligible_if_proposals_coverage_lcb_pass_then_lexicographic_"
    "lcb90_policy_success_mean_delta_conservative_grid_order"
)
GUARD_GAIN_QUANTILES = (0.0, 0.25, 0.5)
GUARD_UNCERTAINTY_QUANTILES = (0.5, 0.75, 1.0)
MEMBER_SELECTION_RULE = (
    "validation_only_pure_success_pair_lcb90_then_top1_uplift_then_event_then_total_v2"
)
DEFAULT_LOSS_WEIGHTS = {
    "success": 1.0,
    "outcome": 0.2,
    "pairwise": 0.75,
    "listwise": 0.5,
    "group_centered": 1.0,
    "baseline_contrast": 1.5,
    "event": 1.0,
    "relative": 0.5,
    "destination": 0.5,
    "predicate": 0.5,
    "reach": 0.75,
    "duration": 0.5,
    "object": 0.5,
    "latent": 0.5,
}


def causal_history_contract() -> dict[str, Any]:
    """Return the immutable hidden-history contract used by source and adapters."""

    value: dict[str, Any] = {
        "format": CAUSAL_HISTORY_FORMAT,
        "max_history_steps": CAUSAL_HISTORY_MAX_STEPS,
        "input_history": (
            "same_branch_query_hidden_prefix_ending_at_current_query"
        ),
        "post_hidden_target_history": (
            "same_input_prefix_plus_current_query_post_hidden"
        ),
        "padding": "right_zero_padding_with_false_mask",
        "truncation": "left_truncate_keep_most_recent",
        "root_candidate_effective_history_steps": 1,
        "cross_branch_or_group_history_allowed": False,
        "future_query_hidden_allowed": False,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


def fixed_causal_hidden_window(
    prefix: np.ndarray,
    *,
    max_steps: int = CAUSAL_HISTORY_MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad one already-causal prefix without consulting future rows."""

    values = np.asarray(prefix)
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps < 1
        or values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("causal hidden prefix is invalid")
    retained = values[-max_steps:]
    window = np.zeros((max_steps, values.shape[1]), dtype=values.dtype)
    mask = np.zeros(max_steps, dtype=np.bool_)
    window[: len(retained)] = retained
    mask[: len(retained)] = True
    return window, mask

RESERVED_ROW_PARAMETERS = {
    "body": ("body_to_id", "body_id", "action_encoder.body_embedding.weight"),
    "policy": (
        "policy_to_id",
        "policy_id",
        "action_encoder.policy_embedding.weight",
    ),
}
RESERVED_ROWS_PROOF_FORMAT = "etsf_dual_reserved_rows_source_only_counterfactual_v1"
ACTION_NORMALIZATION_FORMAT = "etsf_source_train_action_normalization_v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


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


@dataclass(frozen=True)
class ReservedTargetRow:
    """One initializer-reserved embedding row and its immutable reference."""

    axis: str
    mapping_name: str
    batch_id_field: str
    mapping_key: str
    identity: str
    target_id: int
    row: int
    parameter: str
    initial_tensor_sha256: str
    reference: torch.Tensor


def validate_reserved_target_rows(
    checkpoint: Mapping[str, Any],
    config: EventWorldModelConfig | None = None,
) -> dict[str, ReservedTargetRow] | None:
    """Validate both rows from the cold dual-reservation initializer.

    Absence means a legacy checkpoint and intentionally preserves the previous
    training path.  Presence is fail-closed: both axes, the registry entry,
    numeric id/row, exact parameter name, tensor shape, and row digest must all
    agree before any source labels are loaded.
    """

    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        return None
    raw_rows = contract.get("reserved_target_rows")
    if raw_rows is None:
        return None
    if not isinstance(raw_rows, Mapping) or set(raw_rows) != set(
        RESERVED_ROW_PARAMETERS
    ):
        raise RuntimeError("reserved_target_rows must contain exactly body and policy")
    state = checkpoint.get("model")
    config_value = checkpoint.get("config")
    if not isinstance(state, Mapping) or not isinstance(config_value, Mapping):
        raise RuntimeError("reserved-row checkpoint lacks model/config mappings")
    if config is None:
        config = EventWorldModelConfig.from_dict(config_value)

    result: dict[str, ReservedTargetRow] = {}
    for axis, (mapping_name, batch_id_field, parameter_name) in (
        RESERVED_ROW_PARAMETERS.items()
    ):
        spec = raw_rows.get(axis)
        registry = contract.get(mapping_name)
        tensor = state.get(parameter_name)
        if not isinstance(spec, Mapping) or not isinstance(registry, Mapping):
            raise RuntimeError(f"reserved {axis} row lacks its spec/registry")
        identity = spec.get("identity")
        target_id = spec.get("id")
        row = spec.get("row")
        digest = spec.get("tensor_sha256")
        if (
            spec.get("mapping_name") != mapping_name
            or not isinstance(identity, str)
            or not identity
            or isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or isinstance(row, bool)
            or not isinstance(row, int)
            or target_id != row
            or row < 0
            or spec.get("parameter") != parameter_name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"reserved {axis} row contract is malformed")
        mapping_key = f"__reserved__{identity}"
        normalized_registry: dict[str, int] = {}
        for raw_name, raw_id in registry.items():
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise RuntimeError(f"{mapping_name} must contain integer ids")
            normalized_registry[str(raw_name)] = raw_id
        if (
            normalized_registry.get(mapping_key) != target_id
            or sum(value == target_id for value in normalized_registry.values()) != 1
            or len(set(normalized_registry.values())) != len(normalized_registry)
        ):
            raise RuntimeError(f"reserved {axis} mapping/id is inconsistent")
        configured_rows = config.num_bodies if axis == "body" else config.num_policies
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != configured_rows
            or row >= tensor.shape[0]
            or tensor_sha256(tensor[row]) != digest
        ):
            raise RuntimeError(f"reserved {axis} parameter/tensor SHA is inconsistent")
        result[axis] = ReservedTargetRow(
            axis=axis,
            mapping_name=mapping_name,
            batch_id_field=batch_id_field,
            mapping_key=mapping_key,
            identity=identity,
            target_id=target_id,
            row=row,
            parameter=parameter_name,
            initial_tensor_sha256=digest,
            reference=tensor[row].detach().cpu().clone(),
        )
    return result


def assert_reserved_ids_absent(
    batch: Mapping[str, Any],
    rows: Mapping[str, ReservedTargetRow] | None,
) -> None:
    """Reject a source batch that addresses either target reservation."""

    if rows is None:
        return
    for axis, row in rows.items():
        values = batch.get(row.batch_id_field)
        if not torch.is_tensor(values):
            raise RuntimeError(f"source batch lacks {row.batch_id_field} for {axis} audit")
        if bool(torch.any(values == row.target_id)):
            raise RuntimeError(f"reserved target {axis} id entered a source batch")


def assert_source_groups_avoid_reserved_ids(
    groups: Sequence[BranchGroup],
    rows: Mapping[str, ReservedTargetRow] | None,
) -> None:
    """Fail before training if any loaded source group resolves to a target row."""

    if rows is None:
        return
    for group in groups:
        for axis, row in rows.items():
            value = group.body_id if axis == "body" else group.policy_id
            if int(value) == row.target_id:
                raise RuntimeError(
                    f"reserved target {axis} id resolved in source group {group.logical_key}"
                )


@torch.no_grad()
def restore_reserved_target_rows(
    model: nn.Module,
    rows: Mapping[str, ReservedTargetRow] | None,
) -> None:
    """Undo dense AdamW decay/update on both unused embedding rows."""

    if rows is None:
        return
    parameters = dict(model.named_parameters())
    for axis, row in rows.items():
        parameter = parameters.get(row.parameter)
        if (
            parameter is None
            or parameter.ndim != 2
            or row.row >= parameter.shape[0]
            or parameter.shape[1:] != row.reference.shape
        ):
            raise RuntimeError(f"model reserved {axis} parameter shape changed")
        parameter[row.row].copy_(row.reference.to(parameter))


def assert_reserved_target_rows_bit_exact(
    state_or_model: Mapping[str, torch.Tensor] | nn.Module,
    rows: Mapping[str, ReservedTargetRow] | None,
) -> None:
    """Recheck both rows immediately before validation or serialization."""

    if rows is None:
        return
    state = (
        state_or_model.state_dict()
        if isinstance(state_or_model, nn.Module)
        else state_or_model
    )
    for axis, row in rows.items():
        tensor = state.get(row.parameter)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or row.row >= tensor.shape[0]
            or not torch.equal(tensor[row.row].detach().cpu(), row.reference)
            or tensor_sha256(tensor[row.row]) != row.initial_tensor_sha256
        ):
            raise RuntimeError(f"reserved target {axis} row is not bit-exact")


def source_train_action_statistics(
    train_groups: Sequence[BranchGroup], action_dim: int
) -> dict[str, Any]:
    """Fit action normalization from valid train-group timesteps only."""

    parts: list[np.ndarray] = []
    candidate_samples = 0
    continuation_samples = 0
    for group in train_groups:
        actions = np.asarray(group.actions)
        mask = np.asarray(group.action_mask, dtype=bool)
        if actions.ndim != 3 or actions.shape[-1] != action_dim or mask.shape != actions.shape[:2]:
            raise RuntimeError("train-group action/mask shape is invalid")
        selected = np.asarray(actions[mask], dtype=np.float64)
        if selected.size:
            parts.append(selected)
            candidate_samples += len(selected)
        continuation = group.continuation
        if continuation is not None:
            auxiliary_actions = np.asarray(continuation["action_chunks"])
            auxiliary_mask = np.asarray(continuation["action_mask"], dtype=bool)
            if (
                auxiliary_actions.ndim != 3
                or auxiliary_actions.shape[-1] != action_dim
                or auxiliary_mask.shape != auxiliary_actions.shape[:2]
            ):
                raise RuntimeError("train continuation action/mask shape is invalid")
            auxiliary_selected = np.asarray(
                auxiliary_actions[auxiliary_mask], dtype=np.float64
            )
            if auxiliary_selected.size:
                parts.append(auxiliary_selected)
                continuation_samples += len(auxiliary_selected)
    if not parts:
        raise RuntimeError("source train groups contain no valid action samples")
    values = np.concatenate(parts, axis=0)
    if values.shape[1:] != (action_dim,) or not np.isfinite(values).all():
        raise RuntimeError("source train actions are non-finite or malformed")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(values.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    mean_tensor = torch.from_numpy(mean.copy())
    std_tensor = torch.from_numpy(std.copy())
    source_keys = sorted(str(group.logical_key) for group in train_groups)
    result: dict[str, Any] = {
        "format": ACTION_NORMALIZATION_FORMAT,
        "status": "fitted_source_train_only",
        "source_split": "train_groups_only",
        "source_group_count": len(train_groups),
        "source_group_keys_sha256": canonical_sha256(source_keys),
        "valid_action_sample_count": int(len(values)),
        "candidate_valid_action_sample_count": int(candidate_samples),
        "continuation_valid_action_sample_count": int(continuation_samples),
        "action_dim": int(action_dim),
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "action_mean_tensor_sha256": tensor_sha256(mean_tensor),
        "action_std_tensor_sha256": tensor_sha256(std_tensor),
        "validation_groups_used": 0,
        "sealed_test_groups_used": 0,
        "target_data_read": False,
        "target_labels_read": False,
    }
    result["statistics_sha256"] = canonical_sha256(result)
    return result


def resolve_source_action_normalization(
    checkpoint: Mapping[str, Any],
    train_groups: Sequence[BranchGroup],
    config: EventWorldModelConfig,
) -> dict[str, Any] | None:
    """Replace only the cold initializer's identity normalization placeholder."""

    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        return None
    placeholder = contract.get("action_normalization")
    if not isinstance(placeholder, Mapping) or placeholder.get("status") != (
        "identity_placeholder_unfitted"
    ):
        return None
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("action-normalization initializer lacks model state")
    mean = np.asarray(placeholder.get("action_mean"), dtype=np.float32)
    std = np.asarray(placeholder.get("action_std"), dtype=np.float32)
    state_mean = state.get("action_encoder.action_mean")
    state_std = state.get("action_encoder.action_std")
    if (
        mean.shape != (config.action_dim,)
        or std.shape != mean.shape
        or not np.array_equal(mean, np.zeros(config.action_dim, dtype=np.float32))
        or not np.array_equal(std, np.ones(config.action_dim, dtype=np.float32))
        or not isinstance(state_mean, torch.Tensor)
        or not isinstance(state_std, torch.Tensor)
        or not torch.equal(state_mean.detach().cpu(), torch.zeros(config.action_dim))
        or not torch.equal(state_std.detach().cpu(), torch.ones(config.action_dim))
    ):
        raise RuntimeError("identity action-normalization placeholder is inconsistent")
    return source_train_action_statistics(train_groups, config.action_dim)


def install_source_action_normalization(
    model: ActionConditionedEventWorldModel,
    normalization: Mapping[str, Any] | None,
) -> None:
    if normalization is None:
        return
    unsigned = dict(normalization)
    signature = unsigned.pop("statistics_sha256", None)
    if (
        normalization.get("format") != ACTION_NORMALIZATION_FORMAT
        or normalization.get("status") != "fitted_source_train_only"
        or normalization.get("target_data_read") is not False
        or normalization.get("target_labels_read") is not False
        or normalization.get("validation_groups_used") != 0
        or normalization.get("sealed_test_groups_used") != 0
        or signature != canonical_sha256(unsigned)
    ):
        raise RuntimeError("source action-normalization proof is invalid")
    mean = torch.as_tensor(normalization.get("action_mean"), dtype=torch.float32)
    std = torch.as_tensor(normalization.get("action_std"), dtype=torch.float32)
    if (
        tensor_sha256(mean) != normalization.get("action_mean_tensor_sha256")
        or tensor_sha256(std) != normalization.get("action_std_tensor_sha256")
    ):
        raise RuntimeError("source action-normalization statistics SHA changed")
    model.action_encoder.set_normalization(mean, std)
    assert_source_action_normalization_installed(model, normalization)


def assert_source_action_normalization_installed(
    state_or_model: Mapping[str, torch.Tensor] | nn.Module,
    normalization: Mapping[str, Any] | None,
) -> None:
    if normalization is None:
        return
    unsigned = dict(normalization)
    signature = unsigned.pop("statistics_sha256", None)
    mean = torch.as_tensor(normalization.get("action_mean"), dtype=torch.float32)
    std = torch.as_tensor(normalization.get("action_std"), dtype=torch.float32)
    if (
        normalization.get("format") != ACTION_NORMALIZATION_FORMAT
        or signature != canonical_sha256(unsigned)
        or tensor_sha256(mean) != normalization.get("action_mean_tensor_sha256")
        or tensor_sha256(std) != normalization.get("action_std_tensor_sha256")
    ):
        raise RuntimeError("source action-normalization proof changed")
    state = (
        state_or_model.state_dict()
        if isinstance(state_or_model, nn.Module)
        else state_or_model
    )
    state_mean = state.get("action_encoder.action_mean")
    state_std = state.get("action_encoder.action_std")
    if (
        not isinstance(state_mean, torch.Tensor)
        or not isinstance(state_std, torch.Tensor)
        or not torch.equal(state_mean.detach().cpu(), mean)
        or not torch.equal(state_std.detach().cpu(), std)
    ):
        raise RuntimeError("source action-normalization buffers are not bit-exact")


def reserved_rows_source_only_proof(
    model: nn.Module,
    rows: Mapping[str, ReservedTargetRow] | None,
    *,
    source_training_steps: int,
    source_training_groups: int,
    input_pretrained_checkpoint_sha256: str,
    action_normalization: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Create a signed, model-bound proof for a member best checkpoint."""

    if rows is None:
        return None
    if source_training_steps <= 0 or source_training_groups <= 0:
        raise RuntimeError("reserved-row proof requires positive source training")
    if (
        len(input_pretrained_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in input_pretrained_checkpoint_sha256
        )
    ):
        raise RuntimeError("reserved-row proof lacks the pretrained checkpoint SHA")
    assert_reserved_target_rows_bit_exact(model, rows)
    assert_source_action_normalization_installed(model, action_normalization)
    state = model.state_dict()
    proof_rows: dict[str, dict[str, Any]] = {}
    for axis, row in rows.items():
        final_digest = tensor_sha256(state[row.parameter][row.row])
        if final_digest != row.initial_tensor_sha256:
            raise RuntimeError(f"reserved target {axis} final tensor SHA changed")
        proof_rows[axis] = {
            "mapping_name": row.mapping_name,
            "mapping_key": row.mapping_key,
            "identity": row.identity,
            "id": row.target_id,
            "row": row.row,
            "parameter": row.parameter,
            "initial_tensor_sha256": row.initial_tensor_sha256,
            "final_tensor_sha256": final_digest,
            "bit_exact_unchanged": True,
        }
    proof: dict[str, Any] = {
        "format": RESERVED_ROWS_PROOF_FORMAT,
        "status": "complete_source_only",
        "source_training_steps": int(source_training_steps),
        "source_training_groups": int(source_training_groups),
        "input_pretrained_checkpoint_sha256": input_pretrained_checkpoint_sha256,
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_rows_used_in_source_batches": False,
        "reserved_rows_unchanged_during_source_training": True,
        "rows": proof_rows,
        "source_core_state_sha256": state_dict_sha256(state),
        "source_action_normalization": (
            dict(action_normalization)
            if action_normalization is not None
            else {"status": "pretrained_unchanged"}
        ),
    }
    proof["proof_sha256"] = canonical_sha256(proof)
    return proof


def validate_reserved_rows_source_only_proof(
    checkpoint: Mapping[str, Any],
    rows: Mapping[str, ReservedTargetRow] | None,
) -> dict[str, Any] | None:
    """Independently bind a saved proof to its checkpoint model and contract."""

    if rows is None:
        return None
    proof = checkpoint.get("reserved_target_rows_source_only_proof")
    contract = checkpoint.get("contract")
    if not isinstance(proof, Mapping) or not isinstance(contract, Mapping):
        raise RuntimeError("best checkpoint lacks its dual reserved-row proof")
    if contract.get("reserved_target_rows_source_only_proof") != proof:
        raise RuntimeError("checkpoint/contract reserved-row proofs differ")
    unsigned = dict(proof)
    signature = unsigned.pop("proof_sha256", None)
    if (
        proof.get("format") != RESERVED_ROWS_PROOF_FORMAT
        or proof.get("status") != "complete_source_only"
        or proof.get("target_data_read") is not False
        or proof.get("target_labels_read") is not False
        or proof.get("reserved_rows_used_in_source_batches") is not False
        or proof.get("reserved_rows_unchanged_during_source_training") is not True
        or signature != canonical_sha256(unsigned)
    ):
        raise RuntimeError("best checkpoint reserved-row proof is invalid")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("best checkpoint proof lacks model state")
    assert_reserved_target_rows_bit_exact(state, rows)
    if proof.get("source_core_state_sha256") != state_dict_sha256(state):
        raise RuntimeError("best checkpoint model differs from reserved-row proof")
    action_normalization = proof.get("source_action_normalization")
    if not isinstance(action_normalization, Mapping):
        raise RuntimeError("best checkpoint source action-normalization proof is invalid")
    if action_normalization.get("status") != "pretrained_unchanged":
        assert_source_action_normalization_installed(state, action_normalization)
    proof_rows = proof.get("rows")
    if not isinstance(proof_rows, Mapping) or set(proof_rows) != set(rows):
        raise RuntimeError("best checkpoint proof does not cover both reserved rows")
    for axis, row in rows.items():
        spec = proof_rows.get(axis)
        if (
            not isinstance(spec, Mapping)
            or spec.get("mapping_name") != row.mapping_name
            or spec.get("mapping_key") != row.mapping_key
            or spec.get("identity") != row.identity
            or spec.get("id") != row.target_id
            or spec.get("row") != row.row
            or spec.get("parameter") != row.parameter
            or spec.get("initial_tensor_sha256") != row.initial_tensor_sha256
            or spec.get("final_tensor_sha256") != row.initial_tensor_sha256
            or spec.get("bit_exact_unchanged") is not True
        ):
            raise RuntimeError(f"best checkpoint {axis} reserved-row proof changed")
    return dict(proof)


def content_addressed_state_contract(
    *,
    hidden_dim: int,
    modeling_sha256: str,
    bridge_sha256: str,
) -> dict[str, Any]:
    """Reconstruct the SmolVLA collector's frozen state representation id."""

    base: dict[str, Any] = {
        "policy": "smolvla",
        "anchor": (
            "contextualized_vlm_prefix_final_state_token_before_flow_noise_v1"
        ),
        "source": (
            "policy.model.vlm_with_expert.get_vlm_model().text_model.norm"
        ),
        "hidden_dim": int(hidden_dim),
        "prefix_length": 0,
        "noise_independence": "bit_exact_at_group_intervention_query",
        "modeling_sha256": str(modeling_sha256),
        "bridge_sha256": str(bridge_sha256),
    }
    encoded = json.dumps(
        base, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {**base, "calibration_id": hashlib.sha256(encoded).hexdigest()}


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


@dataclass
class BranchGroup:
    """One fixed-state intervention group with all candidate branches aligned."""

    path: str
    schema_version: int
    logical_key: str
    seed: int
    task: str
    body: str
    policy: str
    candidate_names: list[str]
    hidden: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    proprio: np.ndarray
    current_event_id: np.ndarray
    next_event_id: np.ndarray
    clock_event_id: np.ndarray
    next_reached_event_id: np.ndarray
    current_predicates: np.ndarray
    post_predicates: np.ndarray
    relative_transition_id: np.ndarray
    structured_mask: np.ndarray
    duration: np.ndarray
    duration_observed: np.ndarray
    success: np.ndarray
    outcome_id: np.ndarray
    trajectory_regress: np.ndarray
    trajectory_recovery: np.ndarray
    steps: np.ndarray
    object_delta: np.ndarray
    post_hidden: np.ndarray
    dense_mask: np.ndarray
    candidate_distance: np.ndarray
    state_contract: dict[str, Any] | None = None
    continuation: dict[str, np.ndarray] | None = None
    body_id: int = 0
    policy_id: int = 0
    history_hidden: np.ndarray | None = None
    history_mask: np.ndarray | None = None
    post_history_hidden: np.ndarray | None = None
    post_history_mask: np.ndarray | None = None

    @property
    def candidate_count(self) -> int:
        return int(len(self.success))


@dataclass(frozen=True)
class GroupDescriptor:
    """Label-free identity used to split groups before reading any targets."""

    path: str
    schema_version: int
    logical_key: str
    seed: int
    requested_seed: int
    task: str
    body: str
    policy: str
    metadata: Mapping[str, str]


def derive_regression_recovery(
    predicates: np.ndarray,
    dynamic_phase: np.ndarray,
    *,
    persistence_steps: int = DEFAULT_REGRESSION_PERSISTENCE_STEPS,
) -> tuple[bool, bool]:
    """Detect persistent phase regression and operational recovery.

    A regression is a dynamic phase below its pre-drop historical peak for at
    least ``persistence_steps`` consecutive simulator states.  Predicate-only
    down-flips (for example putting an object down while cumulative ``moved``
    keeps the phase at e12) and shorter threshold jitter are not regressions.
    Recovery requires a later return to the pre-drop peak for the same minimum
    persistence, or a later terminal success/eK observation.
    """

    predicates = np.asarray(predicates) > 0.5
    dynamic_phase = np.asarray(dynamic_phase, dtype=np.int64)
    if persistence_steps <= 0:
        raise ValueError("persistence_steps must be positive")
    if predicates.ndim != 2 or predicates.shape[1] < 5:
        raise ValueError(
            "predicates must include moved/lifted/near_goal/stationary/success"
        )
    if dynamic_phase.shape != (len(predicates),):
        raise ValueError("dynamic phase must align with predicate trajectory")
    if len(predicates) <= persistence_steps:
        return False, False

    regression = False
    terminal_success_step = (
        len(predicates) - 1 if bool(predicates[-1, 4]) else None
    )
    for drop_start in range(1, len(dynamic_phase)):
        pre_drop_peak = int(dynamic_phase[:drop_start].max())
        if int(dynamic_phase[drop_start]) >= pre_drop_peak:
            continue

        drop_end = drop_start
        while (
            drop_end < len(dynamic_phase)
            and int(dynamic_phase[drop_end]) < pre_drop_peak
        ):
            drop_end += 1
        if drop_end - drop_start < persistence_steps:
            continue
        regression = True

        # Success/eK is terminal by the predicate contract and is accepted even
        # though it cannot persist for another three recorded simulator states.
        if terminal_success_step is not None and terminal_success_step >= drop_end:
            return True, True

        restored = dynamic_phase[drop_end:] >= pre_drop_peak
        if len(restored) >= persistence_steps:
            run = np.convolve(
                restored.astype(np.int32),
                np.ones(persistence_steps, dtype=np.int32),
                mode="valid",
            )
            if bool((run >= persistence_steps).any()):
                return True, True
    return regression, False


def canonical_events_from_predicates(
    predicates: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Reconstruct the frozen e0/e12/e3/e4/eK milestone sequence.

    This independently binds canonical clock labels to simulator poses instead
    of trusting editable ``event_names/event_steps`` HDF datasets.  It matches
    the current structured event chain: e12 is emitted only when both move and
    lift have been observed, at their earlier first-hit step; later milestones
    that occur before the previous canonical milestone are skipped.
    """

    values = np.asarray(predicates) > 0.5
    if values.ndim != 2 or values.shape[1] != 5 or len(values) < 1:
        raise ValueError("structured predicates must be non-empty [T,5]")
    first_hits: dict[str, int] = {"e0": 0}
    moved = np.flatnonzero(values[:, 0])
    lifted = np.flatnonzero(values[:, 1])
    if moved.size and lifted.size:
        first_hits["e12"] = min(int(moved[0]), int(lifted[0]))
    for event_name, column in (("e3", 2), ("e4", 3), ("eK", 4)):
        hits = np.flatnonzero(values[:, column])
        if hits.size:
            first_hits[event_name] = int(hits[0])

    canonical: list[tuple[str, int]] = []
    previous = -1
    for event_name in DEFAULT_EVENTS:
        step = first_hits.get(event_name)
        if step is None or step < previous:
            continue
        canonical.append((event_name, step))
        previous = step
    return (
        [name for name, _ in canonical],
        np.asarray([step for _, step in canonical], dtype=np.int64),
    )


def canonical_policy_identity(value: Any) -> str:
    """Normalize checkpoint paths and collector aliases to one policy identity."""

    recorded = str(value)
    lowered = recorded.lower()
    if "smolvla" in lowered:
        return "smolvla"
    if "openvla" in lowered:
        return "openvla"
    return recorded


def canonical_policy_mapping(value: Any) -> dict[str, int]:
    """Canonicalize a checkpoint policy map without silently merging ids."""

    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("checkpoint policy_to_id must be a non-empty mapping")
    result: dict[str, int] = {}
    for raw_name, raw_id in value.items():
        name = canonical_policy_identity(raw_name)
        policy_id = int(raw_id)
        previous = result.get(name)
        if previous is not None and previous != policy_id:
            raise RuntimeError(
                f"checkpoint policy aliases collide for {name!r}: "
                f"{previous} != {policy_id}"
            )
        result[name] = policy_id
    return result


def _manifest_metadata(root: Path) -> dict[str, str]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    model_path = str(value.get("model_path", ""))
    recorded_policy = value.get("policy")
    if recorded_policy is None:
        lowered = model_path.lower()
        if "smolvla" in lowered:
            recorded_policy = "smolvla"
        elif "openvla" in lowered:
            recorded_policy = "openvla"
        else:
            recorded_policy = "unknown"
    return {
        "task": str(value.get("task", "unknown")),
        "body": str(value.get("body", "unknown")),
        "policy": canonical_policy_identity(recorded_policy),
    }


def discover_group_files(inputs: Sequence[Path]) -> list[tuple[Path, dict[str, str]]]:
    found: dict[str, tuple[Path, dict[str, str]]] = {}
    for source in inputs:
        source = source.resolve()
        if source.is_file():
            if source.suffix not in {".hdf5", ".h5"}:
                raise ValueError(f"not an HDF5 group: {source}")
            candidates = [source]
            metadata = _manifest_metadata(source.parent.parent)
        elif source.is_dir():
            metadata = _manifest_metadata(source)
            candidates = sorted((source / "groups").glob("*.hdf5"))
            if not candidates:
                candidates = sorted(source.glob("*.hdf5"))
        else:
            raise FileNotFoundError(source)
        for path in candidates:
            found[str(path.resolve())] = (path.resolve(), metadata)
    if not found:
        raise RuntimeError("no candidate group HDF5 files were found")
    return [found[key] for key in sorted(found)]


def read_group_descriptor(
    path: Path, metadata: Mapping[str, str]
) -> GroupDescriptor:
    """Read only identity attrs; no HDF5 label dataset is opened here."""

    with h5py.File(path, "r") as handle:
        schema = int(handle.attrs.get("schema_version", -1))
        if schema not in SUPPORTED_SCHEMAS:
            raise RuntimeError(f"unsupported schema {schema} in {path}")
        task = str(handle.attrs.get("task", metadata.get("task", "unknown")))
        body = str(handle.attrs.get("body", metadata.get("body", "unknown")))
        policy = canonical_policy_identity(
            handle.attrs.get("policy", metadata.get("policy", "openvla"))
        )
        seed = int(handle.attrs.get("resolved_seed", handle.attrs.get("seed", -1)))
        requested_seed = int(handle.attrs.get("requested_seed", seed))
    if seed < 0:
        raise RuntimeError(f"missing seed identity attr in {path}")
    if requested_seed < 0:
        raise RuntimeError(f"missing requested seed identity attr in {path}")
    logical_key = f"{task}|{body}|{seed}"
    return GroupDescriptor(
        path=str(path.resolve()),
        schema_version=schema,
        logical_key=logical_key,
        seed=seed,
        requested_seed=requested_seed,
        task=task,
        body=body,
        policy=policy,
        metadata=dict(metadata),
    )


def scan_group_descriptors(inputs: Sequence[Path]) -> list[GroupDescriptor]:
    """Resolve logical duplicates label-free, preferring the newest schema."""

    selected: dict[str, GroupDescriptor] = {}
    for path, metadata in discover_group_files(inputs):
        descriptor = read_group_descriptor(path, metadata)
        previous = selected.get(descriptor.logical_key)
        if previous is not None and previous.schema_version == descriptor.schema_version:
            raise RuntimeError(
                f"duplicate schema-{descriptor.schema_version} logical group "
                f"{descriptor.logical_key}: {previous.path}, {descriptor.path}"
            )
        if previous is None or descriptor.schema_version > previous.schema_version:
            selected[descriptor.logical_key] = descriptor
    return [selected[key] for key in sorted(selected)]


def _required(handle: h5py.File, names: Iterable[str], path: Path) -> None:
    missing = sorted(set(names) - set(handle.keys()))
    if missing:
        raise RuntimeError(f"missing fields {missing} in {path}")


def _finite(name: str, value: np.ndarray, path: Path) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"non-finite {name} in {path}")


def read_group(
    path: Path,
    metadata: Mapping[str, str],
    config: EventWorldModelConfig,
    object_names: Sequence[str],
    calibrations: Mapping[str, Mapping[str, Any]] | None = None,
    regression_persistence_steps: int = DEFAULT_REGRESSION_PERSISTENCE_STEPS,
    expected_event_spec_sha256: str | None = None,
) -> BranchGroup:
    """Read one group; v4 trajectory labels are derived and audited online."""

    with h5py.File(path, "r") as handle:
        schema = int(handle.attrs.get("schema_version", -1))
        if schema not in SUPPORTED_SCHEMAS:
            raise RuntimeError(f"unsupported schema {schema} in {path}")
        common = {
            "initial_hidden",
            "candidate_names",
            "candidate_actions",
            "success",
            "steps",
        }
        _required(handle, common, path)
        actions = handle["candidate_actions"][:].astype(np.float32)
        success = handle["success"][:].astype(np.float32)
        steps = handle["steps"][:].astype(np.float32)
        names = decode_strings(handle["candidate_names"][:])
        if actions.ndim != 3 or actions.shape[-1] != config.action_dim:
            raise RuntimeError(f"invalid action shape {actions.shape} in {path}")
        count, horizon = actions.shape[:2]
        if len(names) != count or success.shape != (count,) or steps.shape != (count,):
            raise RuntimeError(f"candidate fields are not aligned in {path}")
        if count < 2:
            raise RuntimeError(f"counterfactual group needs at least two candidates in {path}")
        if not names or names[0] != "deterministic" or names.count("deterministic") != 1:
            raise RuntimeError(
                f"candidate zero must be the unique deterministic fallback in {path}"
            )
        initial_hidden = handle["initial_hidden"][:].astype(np.float16)
        if initial_hidden.shape != (config.state_input_dim,):
            raise RuntimeError(f"invalid initial hidden shape in {path}")
        task = str(handle.attrs.get("task", metadata.get("task", "unknown")))
        body = str(handle.attrs.get("body", metadata.get("body", "unknown")))
        policy = canonical_policy_identity(
            handle.attrs.get("policy", metadata.get("policy", "openvla"))
        )
        recorded_event_spec_sha256 = str(
            handle.attrs.get("event_spec_sha256", "")
        )
        if recorded_event_spec_sha256 and expected_event_spec_sha256 is not None:
            if recorded_event_spec_sha256 != expected_event_spec_sha256:
                raise RuntimeError(
                    f"event-spec provenance mismatch in {path}: "
                    f"{recorded_event_spec_sha256} != {expected_event_spec_sha256}"
                )
        state_contract: dict[str, Any] | None = None
        if schema == 5 and policy == "smolvla":
            required_state_attrs = (
                "hidden_anchor",
                "shared_state_source",
                "shared_state_noise_independence",
                "shared_state_modeling_sha256",
                "shared_state_bridge_sha256",
                "shared_state_contract_id",
                "event_spec_sha256",
                "candidate_hidden_forbidden",
            )
            missing_state_attrs = [
                key for key in required_state_attrs if key not in handle.attrs
            ]
            if missing_state_attrs:
                raise RuntimeError(
                    f"SmolVLA schema-v5 state provenance missing "
                    f"{missing_state_attrs} in {path}"
                )
            if expected_event_spec_sha256 is None:
                raise RuntimeError(
                    "SmolVLA schema-v5 loading requires an event-spec SHA binding"
                )
            if recorded_event_spec_sha256 != expected_event_spec_sha256:
                raise RuntimeError(f"SmolVLA event-spec provenance mismatch in {path}")
            if "candidate_hidden" in handle or not bool(
                handle.attrs["candidate_hidden_forbidden"]
            ):
                raise RuntimeError(
                    "SmolVLA schema-v5 cannot contain candidate-specific expert hidden"
                )
            if (
                handle.attrs["shared_state_noise_independence"]
                != "bit_exact_across_explicit_noise_candidates"
            ):
                raise RuntimeError(
                    f"SmolVLA shared-state noise boundary changed in {path}"
                )
            for source_key in (
                "shared_state_modeling_sha256",
                "shared_state_bridge_sha256",
            ):
                digest = str(handle.attrs[source_key])
                if len(digest) != 64 or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in digest
                ):
                    raise RuntimeError(
                        f"SmolVLA source hash {source_key} is invalid in {path}"
                    )
            state_contract = content_addressed_state_contract(
                hidden_dim=int(initial_hidden.shape[0]),
                modeling_sha256=str(
                    handle.attrs["shared_state_modeling_sha256"]
                ),
                bridge_sha256=str(handle.attrs["shared_state_bridge_sha256"]),
            )
            if handle.attrs["hidden_anchor"] != state_contract["anchor"] or handle.attrs[
                "shared_state_source"
            ] != state_contract["source"]:
                raise RuntimeError(f"SmolVLA shared-state anchor changed in {path}")
            if str(handle.attrs["shared_state_contract_id"]) != state_contract[
                "calibration_id"
            ]:
                raise RuntimeError(
                    f"SmolVLA shared-state content hash is inconsistent in {path}"
                )
        seed = int(handle.attrs.get("resolved_seed", handle.attrs.get("seed", -1)))
        if seed < 0:
            raise RuntimeError(f"missing seed in {path}")
        logical_key = f"{task}|{body}|{seed}"
        distance = (
            handle["normalized_l2_from_baseline"][:].astype(np.float32)
            if "normalized_l2_from_baseline" in handle
            else np.zeros(count, dtype=np.float32)
        )
        if distance.shape != (count,):
            raise RuntimeError(f"invalid candidate distance in {path}")
        continuation_rows: dict[str, list[np.ndarray | float | int | bool]] = {
            key: []
            for key in (
                "hidden_t",
                "history_mask",
                "action_chunks",
                "action_mask",
                "proprio",
                "current_event_id",
                "next_event_id",
                "clock_event_id",
                "next_reached_event_id",
                "current_predicates",
                "post_predicates",
                "relative_transition_id",
                "duration",
                "duration_observed",
                "object_delta",
                "post_hidden",
                "post_history_mask",
                "trajectory_regress",
                "trajectory_recovery",
            )
        }

        if schema in (3, 4, 5):
            _required(
                handle,
                {
                    "pre_hidden",
                    "post_chunk_hidden",
                    "first_chunk_action_mask",
                    "pre_proprio",
                    "pre_event_id",
                    "next_event_id",
                    "duration",
                    "duration_observed",
                    "pre_object_poses",
                    "post_object_poses",
                    "object_names",
                },
                path,
            )
            hidden = handle["pre_hidden"][:].astype(np.float16)
            post_hidden = handle["post_chunk_hidden"][:].astype(np.float16)
            action_mask = handle["first_chunk_action_mask"][:].astype(bool)
            proprio = handle["pre_proprio"][:].astype(np.float32)
            if schema == 5 and policy == "smolvla":
                _required(
                    handle,
                    {
                        "shared_state_hook_calls",
                        "shared_state_max_abs_delta",
                        "noise_seeds",
                    },
                    path,
                )
                if not np.array_equal(
                    hidden, np.repeat(initial_hidden[None], count, axis=0)
                ):
                    raise RuntimeError(
                        f"SmolVLA candidate branches do not share the root state in {path}"
                    )
                if not np.array_equal(
                    handle["shared_state_hook_calls"][:],
                    np.ones(count, dtype=np.int16),
                ) or not np.array_equal(
                    handle["shared_state_max_abs_delta"][:],
                    np.zeros(count, dtype=np.float32),
                ):
                    raise RuntimeError(
                        f"SmolVLA shared-prefix noise-independence proof failed in {path}"
                    )
                if len(np.unique(handle["noise_seeds"][:])) != count:
                    raise RuntimeError(f"SmolVLA candidate noise seeds repeat in {path}")
                if not np.any(actions[1:] != actions[0]):
                    raise RuntimeError(
                        f"SmolVLA candidate action intervention has no effect in {path}"
                    )
            canonical_current_event = handle["pre_event_id"][:].astype(np.int64)
            next_reached_event = handle["next_event_id"][:].astype(np.int64)
            duration = handle["duration"][:].astype(np.float32)
            duration_observed = handle["duration_observed"][:].astype(np.float32)
            available_objects = decode_strings(handle["object_names"][:])
            missing_objects = sorted(set(object_names) - set(available_objects))
            if missing_objects:
                raise RuntimeError(
                    f"objects {missing_objects} absent in {path}; available={available_objects}"
                )
            indices = [available_objects.index(name) for name in object_names]
            pre_pose = handle["pre_object_poses"][:, indices, :3].astype(np.float32)
            post_pose = handle["post_object_poses"][:, indices, :3].astype(np.float32)
            object_delta = (post_pose - pre_pose).reshape(count, -1)
            dense_mask = np.ones(count, dtype=bool)
            current_event = canonical_current_event.copy()
            next_event = canonical_current_event.copy()
            current_predicates = np.zeros(
                (count, config.num_predicates), dtype=np.float32
            )
            post_predicates = np.zeros_like(current_predicates)
            relative_transition = np.zeros(count, dtype=np.int64)
            structured_mask = np.zeros(count, dtype=bool)
            trajectory_regress = np.zeros(count, dtype=bool)
            trajectory_recovery = np.zeros(count, dtype=bool)
            if schema == 3 and not config.structured_events:
                next_event = next_reached_event.copy()
            if schema >= 4:
                if not config.structured_events:
                    raise RuntimeError(
                        f"schema-v4 requires a structured-events checkpoint: {path}"
                    )
                if calibrations is None or task not in calibrations:
                    raise RuntimeError(
                        f"schema-v4 requires event calibration for task {task!r}"
                    )
                if "branches" not in handle or len(handle["branches"]) != count:
                    raise RuntimeError(f"schema-v4 branch trajectories missing in {path}")
                executed = handle["first_chunk_executed_length"][:].astype(np.int64)
                for index in range(count):
                    branch_name = f"candidate_{index:03d}"
                    if branch_name not in handle["branches"]:
                        raise RuntimeError(f"missing {branch_name} in {path}")
                    branch = handle["branches"][branch_name]
                    if any(
                        key not in branch
                        for key in ("object_poses", "proprio", "event_names", "event_steps")
                    ):
                        raise RuntimeError(f"incomplete trajectory in {path}/{branch_name}")
                    trajectory_pose = branch["object_poses"][:].astype(np.float32)
                    trajectory_proprio = branch["proprio"][:].astype(np.float32)
                    expected_length = int(steps[index]) + 1
                    if trajectory_pose.shape != (
                        expected_length,
                        len(available_objects),
                        7,
                    ):
                        raise RuntimeError(
                            f"invalid object trajectory {trajectory_pose.shape} in "
                            f"{path}/{branch_name}"
                        )
                    if trajectory_proprio.shape != (
                        expected_length,
                        config.proprio_dim,
                    ):
                        raise RuntimeError(
                            f"invalid proprio trajectory {trajectory_proprio.shape} in "
                            f"{path}/{branch_name}"
                        )
                    _finite("trajectory_object_poses", trajectory_pose, path)
                    _finite("trajectory_proprio", trajectory_proprio, path)
                    post_step = int(executed[index])
                    if not 0 <= post_step < expected_length:
                        raise RuntimeError(f"invalid post step in {path}/{branch_name}")
                    if not np.array_equal(
                        trajectory_pose[0], handle["pre_object_poses"][index]
                    ) or not np.array_equal(
                        trajectory_pose[post_step], handle["post_object_poses"][index]
                    ):
                        raise RuntimeError(
                            f"trajectory/object boundary mismatch in {path}/{branch_name}"
                        )
                    if not np.array_equal(
                        trajectory_proprio[0], handle["pre_proprio"][index]
                    ) or not np.array_equal(
                        trajectory_proprio[post_step], handle["post_proprio"][index]
                    ):
                        raise RuntimeError(
                            f"trajectory/proprio boundary mismatch in {path}/{branch_name}"
                        )
                    predicate_sequence = derive_atomic_predicates(
                        trajectory_pose,
                        available_objects,
                        bool(success[index]),
                        calibrations[task],
                    )
                    canonical_names = decode_strings(branch["event_names"][:])
                    canonical_steps = branch["event_steps"][:].astype(np.int64)
                    if schema == 5 and policy == "smolvla":
                        expected_canonical_names, expected_canonical_steps = (
                            canonical_events_from_predicates(predicate_sequence)
                        )
                        if canonical_names != expected_canonical_names or not np.array_equal(
                            canonical_steps, expected_canonical_steps
                        ):
                            raise RuntimeError(
                                f"canonical event/predicate provenance mismatch in "
                                f"{path}/{branch_name}"
                            )
                    if predicate_sequence.shape[1] != config.num_predicates:
                        raise RuntimeError(
                            "predicate vocabulary differs from structured checkpoint"
                        )
                    dynamic_phase = dynamic_event_ids(
                        predicate_sequence,
                        {name: idx for idx, name in enumerate(config.event_names)},
                    )
                    current_predicates[index] = predicate_sequence[0]
                    post_predicates[index] = predicate_sequence[post_step]
                    current_event[index] = dynamic_phase[0]
                    next_event[index] = dynamic_phase[post_step]
                    relative_transition[index] = relative_transition_ids(
                        dynamic_phase[[0]], dynamic_phase[[post_step]]
                    )[0]
                    regress, recovery = derive_regression_recovery(
                        predicate_sequence,
                        dynamic_phase,
                        persistence_steps=regression_persistence_steps,
                    )
                    trajectory_regress[index] = regress
                    trajectory_recovery[index] = recovery
                    if schema == 5:
                        if "queries" not in handle:
                            raise RuntimeError(f"schema-v5 root query counts missing in {path}")
                        query_fields = (
                            "query_steps",
                            "query_post_steps",
                            "query_hidden",
                            "query_post_hidden",
                            "query_actions",
                            "query_action_mask",
                        )
                        if any(key not in branch for key in query_fields):
                            raise RuntimeError(
                                f"schema-v5 continuation fields missing in {path}/{branch_name}"
                            )
                        query_steps = branch["query_steps"][:].astype(np.int64)
                        query_post_steps = branch["query_post_steps"][:].astype(np.int64)
                        query_hidden = branch["query_hidden"][:].astype(np.float16)
                        query_post_hidden = branch["query_post_hidden"][:].astype(np.float16)
                        query_actions = branch["query_actions"][:].astype(np.float32)
                        query_masks = branch["query_action_mask"][:].astype(bool)
                        query_count = len(query_steps)
                        expected_query_shapes = {
                            "post_steps": (query_count,),
                            "hidden": (query_count, config.state_input_dim),
                            "post_hidden": (query_count, config.state_input_dim),
                            "actions": (query_count, horizon, config.action_dim),
                            "masks": (query_count, horizon),
                        }
                        actual_query_shapes = {
                            "post_steps": query_post_steps.shape,
                            "hidden": query_hidden.shape,
                            "post_hidden": query_post_hidden.shape,
                            "actions": query_actions.shape,
                            "masks": query_masks.shape,
                        }
                        if query_count < 1 or actual_query_shapes != expected_query_shapes:
                            raise RuntimeError(
                                f"invalid schema-v5 query shapes in {path}/{branch_name}: "
                                f"{actual_query_shapes}"
                            )
                        if (
                            query_steps[0] != 0
                            or query_post_steps[0] != int(executed[index])
                            or query_post_steps[-1] != expected_length - 1
                            or query_count - 1 != int(handle["queries"][index])
                            or not np.array_equal(query_steps[1:], query_post_steps[:-1])
                            or not np.array_equal(query_hidden[1:], query_post_hidden[:-1])
                            or not np.array_equal(query_hidden[0], hidden[index])
                            or not np.array_equal(query_post_hidden[0], post_hidden[index])
                            or not np.array_equal(query_actions[0], actions[index])
                            or not np.array_equal(query_masks[0], action_mask[index])
                        ):
                            raise RuntimeError(
                                f"schema-v5 query chain contract failed in {path}/{branch_name}"
                            )
                        query_lengths = query_post_steps - query_steps
                        expected_masks = (
                            np.arange(horizon)[None, :] < query_lengths[:, None]
                        )
                        if (
                            np.any(query_lengths <= 0)
                            or np.any(query_lengths > horizon)
                            or not np.array_equal(query_masks, expected_masks)
                        ):
                            raise RuntimeError(
                                f"schema-v5 query action mask failed in {path}/{branch_name}"
                            )
                        for key, value in {
                            "query_hidden": query_hidden,
                            "query_post_hidden": query_post_hidden,
                            "query_actions": query_actions,
                        }.items():
                            _finite(key, value, path)
                        event_to_id = {
                            name: idx for idx, name in enumerate(config.event_names)
                        }
                        # Query zero is the already represented counterfactual
                        # candidate.  Only deterministic continuation queries
                        # become auxiliary factual transitions.
                        for query_index in range(1, query_count):
                            query_history, query_history_mask = (
                                fixed_causal_hidden_window(
                                    query_hidden[: query_index + 1]
                                )
                            )
                            query_post_history, query_post_history_mask = (
                                fixed_causal_hidden_window(
                                    np.concatenate(
                                        [
                                            query_hidden[: query_index + 1],
                                            query_post_hidden[
                                                query_index : query_index + 1
                                            ],
                                        ],
                                        axis=0,
                                    )
                                )
                            )
                            query_step = int(query_steps[query_index])
                            query_post_step = int(query_post_steps[query_index])
                            clock_event, reached_event, query_duration, observed = (
                                event_transition_target(
                                    query_step,
                                    query_post_step,
                                    expected_length - 1,
                                    canonical_names,
                                    canonical_steps,
                                    event_to_id,
                                )
                            )
                            current_phase = int(dynamic_phase[query_step])
                            post_phase = int(dynamic_phase[query_post_step])
                            query_regress, query_recovery = derive_regression_recovery(
                                predicate_sequence[query_step : query_post_step + 1],
                                dynamic_phase[query_step : query_post_step + 1],
                                persistence_steps=regression_persistence_steps,
                            )
                            auxiliary_values: dict[
                                str, np.ndarray | float | int | bool
                            ] = {
                                "hidden_t": query_history,
                                "history_mask": query_history_mask,
                                "action_chunks": query_actions[query_index],
                                "action_mask": query_masks[query_index],
                                "proprio": trajectory_proprio[query_step],
                                "current_event_id": current_phase,
                                "next_event_id": post_phase,
                                "clock_event_id": clock_event,
                                "next_reached_event_id": reached_event,
                                "current_predicates": predicate_sequence[query_step],
                                "post_predicates": predicate_sequence[query_post_step],
                                "relative_transition_id": relative_transition_ids(
                                    np.asarray([current_phase]),
                                    np.asarray([post_phase]),
                                )[0],
                                "duration": query_duration,
                                "duration_observed": observed,
                                "object_delta": (
                                    trajectory_pose[query_post_step, indices, :3]
                                    - trajectory_pose[query_step, indices, :3]
                                ).reshape(-1),
                                "post_hidden": query_post_history,
                                "post_history_mask": query_post_history_mask,
                                "trajectory_regress": query_regress,
                                "trajectory_recovery": query_recovery,
                            }
                            for key, value in auxiliary_values.items():
                                continuation_rows[key].append(value)
                structured_mask.fill(True)
        else:
            # Schema-v2 has no observation after do(action).  Only success/cost
            # and group ranking below may consume these branches.
            hidden = np.repeat(initial_hidden[None], count, axis=0)
            post_hidden = np.zeros_like(hidden)
            action_mask = np.ones((count, horizon), dtype=bool)
            proprio = np.zeros((count, config.proprio_dim), dtype=np.float32)
            current_event = np.zeros(count, dtype=np.int64)
            next_event = np.zeros(count, dtype=np.int64)
            canonical_current_event = np.zeros(count, dtype=np.int64)
            next_reached_event = np.zeros(count, dtype=np.int64)
            current_predicates = np.zeros(
                (count, config.num_predicates), dtype=np.float32
            )
            post_predicates = np.zeros_like(current_predicates)
            relative_transition = np.zeros(count, dtype=np.int64)
            structured_mask = np.zeros(count, dtype=bool)
            trajectory_regress = np.zeros(count, dtype=bool)
            trajectory_recovery = np.zeros(count, dtype=bool)
            duration = np.zeros(count, dtype=np.float32)
            duration_observed = np.zeros(count, dtype=np.float32)
            object_delta = np.zeros((count, config.object_delta_dim), dtype=np.float32)
            dense_mask = np.zeros(count, dtype=bool)

        # Root interventions share exactly one pre-action state.  Dense schemas
        # additionally expose the post-action hidden used by the future-latent
        # target; schema-v2 has no such observation and is never given a fake
        # post state.  Schema-v5 continuation histories were constructed inside
        # their branch loop above, before rows from different branches meet.
        root_histories: list[np.ndarray] = []
        root_history_masks: list[np.ndarray] = []
        root_post_histories: list[np.ndarray] = []
        root_post_history_masks: list[np.ndarray] = []
        for index in range(count):
            history, history_mask = fixed_causal_hidden_window(
                hidden[index : index + 1]
            )
            post_prefix = (
                np.stack([hidden[index], post_hidden[index]], axis=0)
                if bool(dense_mask[index])
                else hidden[index : index + 1]
            )
            post_history, post_history_mask = fixed_causal_hidden_window(
                post_prefix
            )
            root_histories.append(history)
            root_history_masks.append(history_mask)
            root_post_histories.append(post_history)
            root_post_history_masks.append(post_history_mask)
        history_hidden = np.stack(root_histories, axis=0)
        history_mask = np.stack(root_history_masks, axis=0)
        post_history_hidden = np.stack(root_post_histories, axis=0)
        post_history_mask = np.stack(root_post_history_masks, axis=0)

        outcome_id = success.astype(np.int64)
        outcome_id[trajectory_recovery] = config.outcome_names.index("recovery")
        continuation: dict[str, np.ndarray] | None = None
        if schema == 5:
            continuation_dtypes: dict[str, Any] = {
                "hidden_t": np.float16,
                "history_mask": np.bool_,
                "action_chunks": np.float32,
                "action_mask": np.bool_,
                "proprio": np.float32,
                "current_event_id": np.int64,
                "next_event_id": np.int64,
                "clock_event_id": np.int64,
                "next_reached_event_id": np.int64,
                "current_predicates": np.float32,
                "post_predicates": np.float32,
                "relative_transition_id": np.int64,
                "duration": np.float32,
                "duration_observed": np.float32,
                "object_delta": np.float32,
                "post_hidden": np.float16,
                "post_history_mask": np.bool_,
                "trajectory_regress": np.bool_,
                "trajectory_recovery": np.bool_,
            }
            continuation = {
                key: np.asarray(continuation_rows[key], dtype=dtype)
                for key, dtype in continuation_dtypes.items()
            }
            auxiliary_count = len(continuation["duration"])
            empty_shapes = {
                "hidden_t": (
                    0,
                    CAUSAL_HISTORY_MAX_STEPS,
                    config.state_input_dim,
                ),
                "history_mask": (0, CAUSAL_HISTORY_MAX_STEPS),
                "action_chunks": (0, horizon, config.action_dim),
                "action_mask": (0, horizon),
                "proprio": (0, config.proprio_dim),
                "current_predicates": (0, config.num_predicates),
                "post_predicates": (0, config.num_predicates),
                "object_delta": (0, config.object_delta_dim),
                "post_hidden": (
                    0,
                    CAUSAL_HISTORY_MAX_STEPS,
                    config.state_input_dim,
                ),
                "post_history_mask": (0, CAUSAL_HISTORY_MAX_STEPS),
            }
            for key, shape in empty_shapes.items():
                if auxiliary_count == 0:
                    continuation[key] = np.empty(
                        shape, dtype=continuation_dtypes[key]
                    )
            if any(len(value) != auxiliary_count for value in continuation.values()):
                raise RuntimeError(f"unaligned schema-v5 continuation rows in {path}")

        expected_shapes = {
            "hidden": (count, config.state_input_dim),
            "history_hidden": (
                count,
                CAUSAL_HISTORY_MAX_STEPS,
                config.state_input_dim,
            ),
            "history_mask": (count, CAUSAL_HISTORY_MAX_STEPS),
            "post_hidden": (count, config.state_input_dim),
            "post_history_hidden": (
                count,
                CAUSAL_HISTORY_MAX_STEPS,
                config.state_input_dim,
            ),
            "post_history_mask": (count, CAUSAL_HISTORY_MAX_STEPS),
            "action_mask": (count, horizon),
            "proprio": (count, config.proprio_dim),
            "current_event": (count,),
            "next_event": (count,),
            "clock_event": (count,),
            "next_reached_event": (count,),
            "current_predicates": (count, config.num_predicates),
            "post_predicates": (count, config.num_predicates),
            "relative_transition": (count,),
            "structured_mask": (count,),
            "outcome_id": (count,),
            "trajectory_regress": (count,),
            "trajectory_recovery": (count,),
            "duration": (count,),
            "duration_observed": (count,),
            "object_delta": (count, config.object_delta_dim),
        }
        actual = {
            "hidden": hidden.shape,
            "history_hidden": history_hidden.shape,
            "history_mask": history_mask.shape,
            "post_hidden": post_hidden.shape,
            "post_history_hidden": post_history_hidden.shape,
            "post_history_mask": post_history_mask.shape,
            "action_mask": action_mask.shape,
            "proprio": proprio.shape,
            "current_event": current_event.shape,
            "next_event": next_event.shape,
            "clock_event": canonical_current_event.shape,
            "next_reached_event": next_reached_event.shape,
            "current_predicates": current_predicates.shape,
            "post_predicates": post_predicates.shape,
            "relative_transition": relative_transition.shape,
            "structured_mask": structured_mask.shape,
            "outcome_id": outcome_id.shape,
            "trajectory_regress": trajectory_regress.shape,
            "trajectory_recovery": trajectory_recovery.shape,
            "duration": duration.shape,
            "duration_observed": duration_observed.shape,
            "object_delta": object_delta.shape,
        }
        if actual != expected_shapes:
            raise RuntimeError(f"shape contract failed in {path}: {actual} != {expected_shapes}")
        for key, value in {
            "actions": actions,
            "success": success,
            "steps": steps,
            "hidden": hidden,
            "proprio": proprio,
            "distance": distance,
        }.items():
            _finite(key, value, path)
        if schema in (3, 4, 5):
            for key, value in {
                "post_hidden": post_hidden,
                "duration": duration,
                "object_delta": object_delta,
            }.items():
                _finite(key, value, path)
        if bool(((success < 0) | (success > 1)).any()) or bool((steps <= 0).any()):
            raise RuntimeError(f"invalid terminal labels in {path}")
        if bool(((current_event < 0) | (current_event >= config.num_events)).any()):
            raise RuntimeError(f"current event outside model vocabulary in {path}")
        if bool(((next_event < 0) | (next_event >= config.num_events)).any()):
            raise RuntimeError(f"next event outside model vocabulary in {path}")
        if bool(
            ((next_reached_event < 0) | (next_reached_event >= config.num_events)).any()
        ):
            raise RuntimeError(f"next reached event outside model vocabulary in {path}")

    return BranchGroup(
        path=str(path),
        schema_version=schema,
        logical_key=logical_key,
        seed=seed,
        task=task,
        body=body,
        policy=policy,
        candidate_names=names,
        hidden=hidden,
        actions=actions,
        action_mask=action_mask,
        proprio=proprio,
        current_event_id=current_event,
        next_event_id=next_event,
        clock_event_id=canonical_current_event,
        next_reached_event_id=next_reached_event,
        current_predicates=current_predicates,
        post_predicates=post_predicates,
        relative_transition_id=relative_transition,
        structured_mask=structured_mask,
        duration=duration,
        duration_observed=duration_observed,
        success=success,
        outcome_id=outcome_id,
        trajectory_regress=trajectory_regress,
        trajectory_recovery=trajectory_recovery,
        steps=steps,
        object_delta=object_delta,
        post_hidden=post_hidden,
        dense_mask=dense_mask,
        candidate_distance=distance,
        state_contract=state_contract,
        continuation=continuation,
        history_hidden=history_hidden,
        history_mask=history_mask,
        post_history_hidden=post_history_hidden,
        post_history_mask=post_history_mask,
    )


def load_groups(
    inputs: Sequence[Path],
    config: EventWorldModelConfig,
    object_names: Sequence[str],
    body_to_id: Mapping[str, int],
    policy_to_id: Mapping[str, int],
    calibrations: Mapping[str, Mapping[str, Any]] | None = None,
    regression_persistence_steps: int = DEFAULT_REGRESSION_PERSISTENCE_STEPS,
    expected_event_spec_sha256: str | None = None,
) -> list[BranchGroup]:
    """Compatibility helper that intentionally loads every selected group."""

    return load_descriptor_groups(
        scan_group_descriptors(inputs),
        config,
        object_names,
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
        regression_persistence_steps=regression_persistence_steps,
        expected_event_spec_sha256=expected_event_spec_sha256,
    )


def load_descriptor_groups(
    descriptors: Sequence[GroupDescriptor],
    config: EventWorldModelConfig,
    object_names: Sequence[str],
    body_to_id: Mapping[str, int],
    policy_to_id: Mapping[str, int],
    calibrations: Mapping[str, Mapping[str, Any]] | None = None,
    regression_persistence_steps: int = DEFAULT_REGRESSION_PERSISTENCE_STEPS,
    expected_event_spec_sha256: str | None = None,
) -> list[BranchGroup]:
    """Strictly load labels for an already selected train/validation subset."""

    groups = [
        read_group(
            Path(descriptor.path),
            descriptor.metadata,
            config,
            object_names,
            calibrations=calibrations,
            regression_persistence_steps=regression_persistence_steps,
            expected_event_spec_sha256=expected_event_spec_sha256,
        )
        for descriptor in descriptors
    ]
    for group in groups:
        if group.body not in body_to_id:
            if group.schema_version < 5 and len(body_to_id) == 1:
                group.body_id = next(iter(body_to_id.values()))
            else:
                raise RuntimeError(f"unknown body {group.body!r} in {group.path}")
        else:
            group.body_id = int(body_to_id[group.body])
        if group.policy not in policy_to_id:
            if group.schema_version < 5 and len(policy_to_id) == 1:
                group.policy_id = next(iter(policy_to_id.values()))
            else:
                raise RuntimeError(f"unknown policy {group.policy!r} in {group.path}")
        else:
            group.policy_id = int(policy_to_id[group.policy])
    return groups


def make_group_splits(
    groups: Sequence[BranchGroup | GroupDescriptor],
    seed: int = SPLIT_SEED,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[str]]:
    """Deterministic logical-group split with a sealed remainder."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie in (0,1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be < 1")
    keys = sorted({group.logical_key for group in groups})
    if len(keys) < 3:
        raise RuntimeError("at least three logical groups are required for train/validation/test")
    generator = random.Random(seed)
    generator.shuffle(keys)
    train_count = max(1, min(len(keys) - 2, int(round(len(keys) * train_fraction))))
    validation_count = max(
        1,
        min(len(keys) - train_count - 1, int(round(len(keys) * validation_fraction))),
    )
    return {
        "train": sorted(keys[:train_count]),
        "validation": sorted(keys[train_count : train_count + validation_count]),
        "test": sorted(keys[train_count + validation_count :]),
    }


def read_split_manifest(
    path: Path, groups: Sequence[BranchGroup | GroupDescriptor]
) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    split_unit = value.get("split_unit", "resolved_seed_logical_group")
    if split_unit not in {
        "requested_seed_logical_group",
        "resolved_seed_logical_group",
    }:
        raise RuntimeError(f"unsupported split manifest unit {split_unit!r}")
    by_seed: dict[int, list[str]] = {}
    for group in groups:
        if split_unit == "requested_seed_logical_group":
            seed = int(getattr(group, "requested_seed", group.seed))
        else:
            seed = int(group.seed)
        by_seed.setdefault(seed, []).append(group.logical_key)
    result: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        entries = value.get(split, [])
        keys: list[str] = []
        for entry in entries:
            if isinstance(entry, str) and "|" in entry:
                keys.append(entry)
            else:
                seed = int(entry["seed"] if isinstance(entry, dict) else entry)
                keys.extend(by_seed.get(seed, []))
        result[split] = sorted(set(keys))
    known = {group.logical_key for group in groups}
    assigned = set().union(*map(set, result.values()))
    if assigned != known:
        raise RuntimeError(
            f"split manifest must assign every logical group exactly once; "
            f"missing={sorted(known-assigned)}, unknown={sorted(assigned-known)}"
        )
    if any(set(result[a]) & set(result[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("logical group leakage across split manifest")
    if not result["train"] or not result["validation"] or not result["test"]:
        raise RuntimeError("train, validation, and sealed test must all be non-empty")
    return result


class GroupDataset(Dataset[BranchGroup]):
    def __init__(self, groups: Sequence[BranchGroup]) -> None:
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> BranchGroup:
        return self.groups[index]


def collate_groups(
    groups: Sequence[BranchGroup],
    object_mean: np.ndarray,
    object_std: np.ndarray,
    include_auxiliary: bool = True,
) -> dict[str, Any]:
    """Flatten candidates but retain an exact group index for ranking losses."""

    tensors: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "hidden_t",
            "history_mask",
            "action_chunks",
            "action_mask",
            "proprio",
            "current_event_id",
            "next_event_id",
            "clock_event_id",
            "next_reached_event_id",
            "current_predicates",
            "post_predicates",
            "relative_transition_id",
            "structured_mask",
            "duration",
            "duration_observed",
            "success",
            "outcome_id",
            "trajectory_regress",
            "trajectory_recovery",
            "steps",
            "object_delta",
            "post_hidden",
            "post_history_mask",
            "dense_mask",
            "terminal_mask",
            "candidate_distance",
            "body_id",
            "policy_id",
            "group_index",
            "baseline_mask",
        )
    }
    group_keys = []
    candidate_names: list[str] = []
    for group_index, group in enumerate(groups):
        if any(
            value is None
            for value in (
                group.history_hidden,
                group.history_mask,
                group.post_history_hidden,
                group.post_history_mask,
            )
        ):
            raise RuntimeError(
                "branch group lacks the signed causal hidden-history fields"
            )
        count = group.candidate_count
        baseline = np.zeros(count, dtype=bool)
        deterministic = [
            index for index, name in enumerate(group.candidate_names) if name == "deterministic"
        ]
        baseline[deterministic[0] if deterministic else 0] = True
        normalized_object = (group.object_delta - object_mean) / object_std
        values: dict[str, np.ndarray] = {
            "hidden_t": group.history_hidden,
            "history_mask": group.history_mask,
            "action_chunks": group.actions,
            "action_mask": group.action_mask,
            "proprio": group.proprio,
            "current_event_id": group.current_event_id,
            "next_event_id": group.next_event_id,
            "clock_event_id": group.clock_event_id,
            "next_reached_event_id": group.next_reached_event_id,
            "current_predicates": group.current_predicates,
            "post_predicates": group.post_predicates,
            "relative_transition_id": group.relative_transition_id,
            "structured_mask": group.structured_mask,
            "duration": group.duration,
            "duration_observed": group.duration_observed,
            "success": group.success,
            "outcome_id": group.outcome_id,
            "trajectory_regress": group.trajectory_regress,
            "trajectory_recovery": group.trajectory_recovery,
            "steps": group.steps,
            "object_delta": normalized_object.astype(np.float32),
            "post_hidden": group.post_history_hidden,
            "post_history_mask": group.post_history_mask,
            "dense_mask": group.dense_mask,
            "terminal_mask": np.ones(count, dtype=bool),
            "candidate_distance": group.candidate_distance,
            "body_id": np.full(count, group.body_id, dtype=np.int64),
            "policy_id": np.full(count, group.policy_id, dtype=np.int64),
            "group_index": np.full(count, group_index, dtype=np.int64),
            "baseline_mask": baseline,
        }
        for key, value in values.items():
            tensors[key].append(torch.from_numpy(np.asarray(value)))
        group_keys.append(group.logical_key)
        candidate_names.extend(group.candidate_names)
        if include_auxiliary and group.continuation is not None:
            continuation = group.continuation
            auxiliary_count = len(continuation["duration"])
            if auxiliary_count:
                normalized_aux_object = (
                    continuation["object_delta"] - object_mean
                ) / object_std
                auxiliary_values: dict[str, np.ndarray] = {
                    "hidden_t": continuation["hidden_t"],
                    "history_mask": continuation["history_mask"],
                    "action_chunks": continuation["action_chunks"],
                    "action_mask": continuation["action_mask"],
                    "proprio": continuation["proprio"],
                    "current_event_id": continuation["current_event_id"],
                    "next_event_id": continuation["next_event_id"],
                    "clock_event_id": continuation["clock_event_id"],
                    "next_reached_event_id": continuation["next_reached_event_id"],
                    "current_predicates": continuation["current_predicates"],
                    "post_predicates": continuation["post_predicates"],
                    "relative_transition_id": continuation[
                        "relative_transition_id"
                    ],
                    "structured_mask": np.ones(auxiliary_count, dtype=bool),
                    "duration": continuation["duration"],
                    "duration_observed": continuation["duration_observed"],
                    # Terminal labels are deliberately masked below; these
                    # placeholders cannot repeat one branch outcome Q times.
                    "success": np.zeros(auxiliary_count, dtype=np.float32),
                    "outcome_id": np.zeros(auxiliary_count, dtype=np.int64),
                    "trajectory_regress": continuation["trajectory_regress"],
                    "trajectory_recovery": continuation["trajectory_recovery"],
                    "steps": np.ones(auxiliary_count, dtype=np.float32),
                    "object_delta": normalized_aux_object.astype(np.float32),
                    "post_hidden": continuation["post_hidden"],
                    "post_history_mask": continuation["post_history_mask"],
                    "dense_mask": np.ones(auxiliary_count, dtype=bool),
                    "terminal_mask": np.zeros(auxiliary_count, dtype=bool),
                    "candidate_distance": np.zeros(auxiliary_count, dtype=np.float32),
                    "body_id": np.full(auxiliary_count, group.body_id, dtype=np.int64),
                    "policy_id": np.full(auxiliary_count, group.policy_id, dtype=np.int64),
                    "group_index": np.full(auxiliary_count, -1, dtype=np.int64),
                    "baseline_mask": np.zeros(auxiliary_count, dtype=bool),
                }
                for key, value in auxiliary_values.items():
                    tensors[key].append(torch.from_numpy(np.asarray(value)))
                candidate_names.extend(
                    [f"continuation_{index}" for index in range(auxiliary_count)]
                )
    result: dict[str, Any] = {
        key: torch.cat(parts, dim=0) for key, parts in tensors.items()
    }
    result["group_keys"] = group_keys
    result["candidate_names"] = candidate_names
    return result


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    result["hidden_t"] = result["hidden_t"].float()
    result["post_hidden"] = result["post_hidden"].float()
    return result


def forward_model(
    model: ActionConditionedEventWorldModel, batch: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    return model(
        batch["hidden_t"],
        batch["action_chunks"],
        history_mask=batch["history_mask"],
        action_mask=batch["action_mask"],
        proprio=batch["proprio"],
        body_id=batch["body_id"],
        policy_id=batch["policy_id"],
        current_event_id=batch["current_event_id"],
        clock_event_id=(
            batch["clock_event_id"] if model.config.structured_events else None
        ),
        current_predicates=(
            batch["current_predicates"] if model.config.structured_events else None
        ),
        dt=batch["action_mask"].sum(-1).float().clamp_min(1.0),
    )


def lognormal_nll_per_item(
    log_mean: torch.Tensor,
    log_scale: torch.Tensor,
    duration: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    target = torch.log1p(duration.clamp_min(0.0))
    scale = torch.exp(log_scale.clamp(-5.0, 3.0)).clamp_min(1e-4)
    z = (target - log_mean) / scale
    observed_nll = 0.5 * z.square() + torch.log(scale) + 0.5 * math.log(2 * math.pi)
    censored_nll = -torch.special.log_ndtr(-z)
    return torch.where(observed.bool(), observed_nll, censored_nll)


def candidate_rank_score(
    output: Mapping[str, torch.Tensor],
    event_values: torch.Tensor,
    duration_scale: float,
    *,
    success_temperature: float = 1.0,
    event_weight: float = 0.25,
    duration_weight: float = 0.05,
) -> torch.Tensor:
    event_probability = torch.softmax(output["next_event_logits"], dim=-1)
    event_progress = (event_probability * event_values.to(event_probability)).sum(-1)
    predicted_duration = torch.expm1(
        output["duration_selected_log_mean"].clamp(0.0, 12.0)
    )
    normalized_duration = predicted_duration / max(float(duration_scale), 1.0)
    return (
        output["success_logit"] / max(float(success_temperature), 1e-4)
        + event_weight * event_progress
        - duration_weight * normalized_duration
    )


def group_action_rank_residual(
    model: ActionConditionedEventWorldModel,
    output: Mapping[str, torch.Tensor],
    group_index: torch.Tensor,
    baseline_mask: torch.Tensor,
    *,
    detach_features: bool = False,
) -> torch.Tensor:
    """Apply the optional action branch only to complete candidate groups."""

    base_logit = output["success_logit"]
    if model.action_rank_head is None:
        return torch.zeros_like(base_logit)
    if not (
        base_logit.shape == group_index.shape == baseline_mask.shape
    ):
        raise ValueError("action-rank group metadata must align with model output")
    baseline_rows = torch.arange(len(group_index), device=group_index.device)
    ranked = torch.zeros_like(group_index, dtype=torch.bool)
    for group_id in torch.unique(group_index, sorted=True):
        if int(group_id) < 0:
            continue
        rows = torch.nonzero(group_index == group_id, as_tuple=False).squeeze(-1)
        group_baseline = baseline_mask[rows].bool()
        if int(group_baseline.sum()) != 1:
            raise ValueError("every action-rank group needs one deterministic baseline")
        baseline_row = rows[torch.nonzero(group_baseline, as_tuple=False)[0, 0]]
        baseline_rows[rows] = baseline_row
        ranked[rows] = True
    semantic = output["semantic"]
    action_effect = output["action_effect"]
    if detach_features:
        # Ranking-only supervision is sparse and must not rewrite the factual
        # action encoder or shared event representation.  Gradients still flow
        # through the relative rank head applied below.
        semantic = semantic.detach()
        action_effect = action_effect.detach()
    baseline_action_effect = action_effect[baseline_rows]
    residual = model.relative_action_rank_logit(
        semantic, action_effect, baseline_action_effect
    )
    return torch.where(ranked, residual, torch.zeros_like(residual))


def counterfactual_success_logit(
    model: ActionConditionedEventWorldModel,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return output["success_logit"] + group_action_rank_residual(
        model, output, batch["group_index"], batch["baseline_mask"]
    )


def counterfactual_aleatoric_uncertainty(
    model: ActionConditionedEventWorldModel,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Keep flat grouped evaluation consistent with ``predict_candidates``.

    Training represents candidates as flattened rows, whereas the online
    plugin calls ``predict_candidates``.  Both paths must update the success
    entropy after adding the baseline-relative residual or the OOF guard would
    be tuned against a different uncertainty than deployment.
    """

    if model.action_rank_head is None:
        return output["aleatoric_uncertainty"]
    adjusted = counterfactual_success_logit(model, output, batch)
    probability = torch.sigmoid(adjusted).clamp(1e-7, 1.0 - 1e-7)
    adjusted_entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    ) / math.log(2.0)
    coefficient = 0.1 if model.config.structured_events else 0.2
    return (
        output["aleatoric_uncertainty"]
        + coefficient
        * (adjusted_entropy - output["aleatoric_success_entropy"])
    ).clamp_min(0.0)


def counterfactual_rank_score(
    model: ActionConditionedEventWorldModel,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    event_values: torch.Tensor,
    duration_scale: float,
    *,
    success_temperature: float = 1.0,
    event_weight: float = 0.25,
    duration_weight: float = 0.05,
    ranking_gradient_only: bool = False,
) -> torch.Tensor:
    if model.config.action_rank_success_only:
        base = output["success_logit"] / max(float(success_temperature), 1e-4)
    else:
        base = candidate_rank_score(
            output,
            event_values,
            duration_scale,
            success_temperature=success_temperature,
            event_weight=event_weight,
            duration_weight=duration_weight,
        )
    residual = group_action_rank_residual(
        model,
        output,
        batch["group_index"],
        batch["baseline_mask"],
        detach_features=ranking_gradient_only,
    )
    if ranking_gradient_only and model.action_rank_head is not None:
        base = base.detach()
    return base + residual / max(float(success_temperature), 1e-4)


def ranking_losses(
    scores: torch.Tensor,
    success: torch.Tensor,
    group_index: torch.Tensor,
    *,
    list_temperature: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rank only binary success within an intervention group.

    Duration is supervised by the dedicated time head.  Including terminal
    steps here creates preferences between candidates with the same outcome,
    which conflicts with the success-centered target and can overwhelm the
    comparatively rare success-changing action pairs.
    """

    pairwise_parts = []
    listwise_parts = []
    pair_count = 0
    for group_id in torch.unique(group_index, sorted=True):
        if int(group_id) < 0:
            continue
        mask = group_index == group_id
        group_score = scores[mask]
        if len(group_score) < 2:
            continue
        utility = 2.0 * success[mask]
        utility_delta = utility[:, None] - utility[None, :]
        score_delta = group_score[:, None] - group_score[None, :]
        upper = torch.triu(torch.ones_like(utility_delta, dtype=torch.bool), diagonal=1)
        comparable = upper & (utility_delta.abs() > 1e-7)
        if comparable.any():
            signs = torch.sign(utility_delta[comparable])
            pairwise_parts.append(F.softplus(-signs * score_delta[comparable]).mean())
            pair_count += int(comparable.sum())
        target = torch.softmax(utility / list_temperature, dim=0)
        listwise_parts.append(
            -(target * torch.log_softmax(group_score, dim=0)).sum()
            / math.log(len(group_score))
        )
    zero = scores.sum() * 0.0
    pairwise = torch.stack(pairwise_parts).mean() if pairwise_parts else zero
    listwise = torch.stack(listwise_parts).mean() if listwise_parts else zero
    return pairwise, listwise, pair_count


def success_pair_ranking_counts(
    scores: torch.Tensor,
    success: torch.Tensor,
    baseline_mask: torch.Tensor,
) -> dict[str, int]:
    """Count pure-success and deterministic-baseline ranking decisions."""

    if not (scores.shape == success.shape == baseline_mask.shape):
        raise ValueError("success pair metric inputs must be aligned vectors")
    if scores.ndim != 1:
        raise ValueError("success pair metrics require one intervention group")
    baseline_mask = baseline_mask.bool()
    if int(baseline_mask.sum()) != 1:
        raise ValueError("success pair metrics require one deterministic baseline")

    success_delta = success[:, None] - success[None, :]
    score_delta = scores[:, None] - scores[None, :]
    upper = torch.triu(
        torch.ones_like(success_delta, dtype=torch.bool), diagonal=1
    )
    comparable = upper & (success_delta.abs() > 1e-7)
    pure_total = int(comparable.sum())
    pure_correct = int(
        ((success_delta[comparable] * score_delta[comparable]) > 0).sum()
    )

    baseline = torch.nonzero(baseline_mask, as_tuple=False)[0, 0]
    alternatives = ~baseline_mask
    baseline_success_delta = success[alternatives] - success[baseline]
    baseline_score_delta = scores[alternatives] - scores[baseline]
    baseline_comparable = baseline_success_delta.abs() > 1e-7
    baseline_total = int(baseline_comparable.sum())
    baseline_correct = int(
        (
            baseline_success_delta[baseline_comparable]
            * baseline_score_delta[baseline_comparable]
            > 0
        ).sum()
    )
    return {
        "pure_success_correct": pure_correct,
        "pure_success_total": pure_total,
        "baseline_changing_correct": baseline_correct,
        "baseline_changing_total": baseline_total,
    }


def counterfactual_centered_losses(
    scores: torch.Tensor,
    success: torch.Tensor,
    group_index: torch.Tensor,
    baseline_mask: torch.Tensor,
    *,
    centered_target_margin: float = 2.0,
    baseline_score_margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Make action ranking identifiable after removing scene difficulty.

    Absolute success BCE can obtain a high global AUC by recognizing easy and
    hard scenes while ordering the four actions inside each scene backwards.
    The centered term removes each intervention group's common logit and also
    teaches homogeneous groups not to invent action differences.  The baseline
    term gives every success-changing candidate a direct comparison against the
    deterministic action actually used by the deployment fallback.
    """

    if not (
        scores.shape == success.shape == group_index.shape == baseline_mask.shape
    ):
        raise ValueError("centered ranking inputs must be aligned vectors")
    centered_parts: list[torch.Tensor] = []
    baseline_parts: list[torch.Tensor] = []
    baseline_pairs = 0
    for group_id in torch.unique(group_index, sorted=True):
        if int(group_id) < 0:
            continue
        mask = group_index == group_id
        group_score = scores[mask]
        group_success = success[mask]
        group_baseline = baseline_mask[mask].bool()
        if len(group_score) < 2:
            continue
        if int(group_baseline.sum()) != 1:
            raise ValueError("every intervention group needs one deterministic baseline")
        centered_parts.append(
            F.smooth_l1_loss(
                group_score - group_score.mean(),
                centered_target_margin * (group_success - group_success.mean()),
            )
        )
        baseline = torch.nonzero(group_baseline, as_tuple=False)[0, 0]
        alternatives = ~group_baseline
        success_delta = group_success[alternatives] - group_success[baseline]
        comparable = success_delta.abs() > 1e-7
        if comparable.any():
            score_delta = group_score[alternatives] - group_score[baseline]
            signs = torch.sign(success_delta[comparable])
            baseline_parts.append(
                F.softplus(
                    baseline_score_margin - signs * score_delta[comparable]
                ).mean()
            )
            baseline_pairs += int(comparable.sum())
    zero = scores.sum() * 0.0
    centered = torch.stack(centered_parts).mean() if centered_parts else zero
    baseline_contrast = (
        torch.stack(baseline_parts).mean() if baseline_parts else zero
    )
    return centered, baseline_contrast, baseline_pairs


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def compute_loss(
    model: ActionConditionedEventWorldModel,
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    success_pos_weight: torch.Tensor,
    event_class_weight: torch.Tensor,
    event_values: torch.Tensor,
    duration_scale: float,
    destination_class_weight: torch.Tensor | None = None,
    relative_class_weight: torch.Tensor | None = None,
    relative_supported: torch.Tensor | None = None,
    predicate_pos_weight: torch.Tensor | None = None,
    outcome_class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    output = forward_model(model, batch)
    dense = batch["dense_mask"].bool()
    structured = batch["structured_mask"].bool()
    terminal = batch["terminal_mask"].bool()
    success_per = F.binary_cross_entropy_with_logits(
        output["success_logit"],
        batch["success"],
        pos_weight=success_pos_weight,
        reduction="none",
    )
    success = _masked_mean(success_per, terminal)
    legacy_outcome_per = F.cross_entropy(
        output["outcome_logits"][:, :2], batch["success"].long(), reduction="none"
    )
    if model.config.recovery_supervised:
        structured_outcome_per = F.cross_entropy(
            output["outcome_logits"],
            batch["outcome_id"],
            weight=outcome_class_weight,
            reduction="none",
        )
        outcome_per = torch.where(structured, structured_outcome_per, legacy_outcome_per)
    else:
        # Until enough v4 recovery positives exist, never train the third class
        # and never turn legacy terminal labels into recovery negatives.
        outcome_per = legacy_outcome_per
    outcome = _masked_mean(outcome_per, terminal)
    composite_rank_score = counterfactual_rank_score(
        model,
        output,
        batch,
        event_values,
        duration_scale,
        ranking_gradient_only=True,
    )
    # Weak schema-v2 terminal labels may shape the success/ranking head, but
    # must not rewrite event or clock heads for transitions that were never
    # observed.  Structured v4 groups train the full composite score; legacy
    # v3 does not fabricate a reversible post-chunk event target.
    composite_mask = structured if model.config.structured_events else dense
    rank_score = output["success_logit"] + composite_mask.to(output["success_logit"]) * (
        composite_rank_score - output["success_logit"]
    )
    pairwise, listwise, _ = ranking_losses(
        rank_score,
        batch["success"],
        batch["group_index"],
    )
    group_centered, baseline_contrast, _ = counterfactual_centered_losses(
        rank_score,
        batch["success"],
        batch["group_index"],
        batch["baseline_mask"],
    )

    event_per = F.cross_entropy(
        output["next_event_logits"],
        batch["next_event_id"],
        weight=event_class_weight,
        reduction="none",
    )
    event_mask = structured if model.config.structured_events else dense
    event = _masked_mean(event_per, event_mask)
    zero = output["success_logit"].sum() * 0.0
    if model.config.structured_events:
        if (
            destination_class_weight is None
            or relative_class_weight is None
            or relative_supported is None
            or predicate_pos_weight is None
        ):
            raise ValueError("structured loss metadata is required")
        relative_target = batch["relative_transition_id"]
        relative_per = F.cross_entropy(
            output["relative_transition_logits"],
            relative_target,
            weight=relative_class_weight,
            reduction="none",
        )
        relative = _masked_mean(
            relative_per, structured & relative_supported[relative_target]
        )
        destination_per = F.cross_entropy(
            output["next_reached_event_logits"],
            batch["next_reached_event_id"],
            weight=destination_class_weight,
            reduction="none",
        )
        destination = _masked_mean(
            destination_per, dense & batch["duration_observed"].bool()
        )
        predicate_per = F.binary_cross_entropy_with_logits(
            output["post_predicate_logits"],
            batch["post_predicates"],
            pos_weight=predicate_pos_weight,
            reduction="none",
        ).mean(-1)
        predicate = _masked_mean(predicate_per, structured)
    else:
        relative = destination = predicate = zero
    reach_per = F.binary_cross_entropy_with_logits(
        output["reach_logit"], batch["duration_observed"], reduction="none"
    )
    reach = _masked_mean(reach_per, dense)
    duration_per = lognormal_nll_per_item(
        output["duration_selected_log_mean"],
        output["duration_selected_log_scale"],
        batch["duration"],
        batch["duration_observed"],
    )
    duration = _masked_mean(duration_per, dense)
    object_scale = torch.exp(output["object_delta_log_scale"].clamp(-5.0, 3.0))
    object_per = (
        0.5
        * torch.square(
            (batch["object_delta"] - output["object_delta_mean"])
            / object_scale.clamp_min(1e-4)
        )
        + torch.log(object_scale.clamp_min(1e-4))
    ).mean(-1)
    object_delta = _masked_mean(object_per, dense)
    if dense.any():
        with torch.no_grad():
            target_semantic = model.encode_state(
                batch["post_hidden"][dense],
                batch["post_history_mask"][dense],
            )
        latent_scale = torch.exp(output["future_latent_log_scale"][dense]).clamp_min(1e-4)
        latent_nll = (
            0.5
            * torch.square(
                (target_semantic.detach() - output["future_latent_mean"][dense])
                / latent_scale
            )
            + torch.log(latent_scale)
        ).mean()
        latent_cosine = (
            1.0
            - F.cosine_similarity(
                output["future_latent_mean"][dense], target_semantic.detach(), dim=-1
            )
        ).mean()
        latent = latent_nll + latent_cosine
    else:
        latent = output["future_latent_mean"].sum() * 0.0
    pieces = {
        "success": success,
        "outcome": outcome,
        "pairwise": pairwise,
        "listwise": listwise,
        "group_centered": group_centered,
        "baseline_contrast": baseline_contrast,
        "event": event,
        "relative": relative,
        "destination": destination,
        "predicate": predicate,
        "reach": reach,
        "duration": duration,
        "object": object_delta,
        "latent": latent,
    }
    effective_weights = {**DEFAULT_LOSS_WEIGHTS, **dict(weights)}
    unknown_weights = sorted(set(effective_weights) - set(DEFAULT_LOSS_WEIGHTS))
    if unknown_weights:
        raise ValueError(f"unknown loss weights: {unknown_weights}")
    total = sum(effective_weights[name] * value for name, value in pieces.items())
    pieces["total"] = total
    return total, pieces, output


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return result


def wilson_lower_bound(successes: int, total: int, z: float = 1.6448536269514722) -> float | None:
    """One-sided 90% Wilson lower bound used only on validation metrics."""

    if total <= 0:
        return None
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return float((center - radius) / denominator)


def member_validation_selection_key(validation: Mapping[str, Any]) -> tuple[float, ...]:
    """Validation-only member selection centered on deployable group ranking."""

    pair_lcb = validation.get("pure_success_pair_lcb90")
    pair_value = -1.0 if pair_lcb is None else float(pair_lcb)
    uplift = float(validation["top1_success_rate"]) - float(
        validation["baseline_success_rate"]
    )
    losses = validation["losses"]
    return (
        pair_value,
        uplift,
        -float(losses["event"]),
        -float(losses["total"]),
    )


@torch.no_grad()
def evaluate_model(
    model: ActionConditionedEventWorldModel,
    loader: DataLoader,
    device: torch.device,
    weights: Mapping[str, float],
    loss_metadata: Mapping[str, torch.Tensor],
    event_values: torch.Tensor,
    duration_scale: float,
    reserved_target_rows: Mapping[str, ReservedTargetRow] | None = None,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    candidates = 0
    success_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    pure_pair_correct = pure_pair_total = 0
    baseline_pair_correct = baseline_pair_total = 0
    baseline_successes = selected_successes = oracle_successes = groups_seen = 0
    dense_count = 0
    for raw in loader:
        assert_reserved_ids_absent(raw, reserved_target_rows)
        batch = move_batch(raw, device)
        loss, pieces, output = compute_loss(
            model,
            batch,
            weights,
            loss_metadata["success_pos"],
            loss_metadata["event"],
            event_values,
            duration_scale,
            destination_class_weight=loss_metadata["destination"],
            relative_class_weight=loss_metadata["relative"],
            relative_supported=loss_metadata["relative_supported"],
            predicate_pos_weight=loss_metadata["predicate_pos"],
            outcome_class_weight=loss_metadata["outcome"],
        )
        del loss
        count = len(batch["success"])
        candidates += count
        dense_count += int(batch["dense_mask"].sum())
        for key, value in pieces.items():
            totals[key] = totals.get(key, 0.0) + float(value) * count
        probability = torch.sigmoid(
            counterfactual_success_logit(model, output, batch)
        )
        terminal = batch["terminal_mask"].bool()
        success_rows.append(batch["success"][terminal].cpu().numpy())
        probability_rows.append(probability[terminal].cpu().numpy())
        rank_score = counterfactual_rank_score(
            model, output, batch, event_values, duration_scale
        )
        for group_id in torch.unique(batch["group_index"], sorted=True):
            if int(group_id) < 0:
                continue
            mask = batch["group_index"] == group_id
            group_success = batch["success"][mask]
            group_scores = rank_score[mask]
            group_baseline = batch["baseline_mask"][mask]
            baseline = torch.nonzero(group_baseline, as_tuple=False)[0, 0]
            selected = group_scores.argmax()
            baseline_successes += int(group_success[baseline].item())
            selected_successes += int(group_success[selected].item())
            oracle_successes += int(group_success.max().item())
            groups_seen += 1
            rank_counts = success_pair_ranking_counts(
                group_scores, group_success, group_baseline
            )
            pure_pair_correct += rank_counts["pure_success_correct"]
            pure_pair_total += rank_counts["pure_success_total"]
            baseline_pair_correct += rank_counts["baseline_changing_correct"]
            baseline_pair_total += rank_counts["baseline_changing_total"]
    labels = np.concatenate(success_rows)
    probabilities = np.concatenate(probability_rows)
    return {
        "groups": groups_seen,
        "candidates": candidates,
        "dense_candidates": dense_count,
        "losses": {key: value / max(candidates, 1) for key, value in totals.items()},
        "success_auc": binary_auc(labels, probabilities),
        "success_brier": float(np.mean(np.square(probabilities - labels))),
        "success_ece": expected_calibration_error(labels, probabilities),
        "pure_success_pair_accuracy": (
            pure_pair_correct / pure_pair_total if pure_pair_total else None
        ),
        "pure_success_pair_lcb90": wilson_lower_bound(
            pure_pair_correct, pure_pair_total
        ),
        "pure_success_comparable_pairs": pure_pair_total,
        "baseline_changing_pair_accuracy": (
            baseline_pair_correct / baseline_pair_total
            if baseline_pair_total
            else None
        ),
        "baseline_changing_pair_lcb90": wilson_lower_bound(
            baseline_pair_correct, baseline_pair_total
        ),
        "baseline_changing_pairs": baseline_pair_total,
        # Backward-readable aliases now have an explicit pure-success meaning.
        "pairwise_accuracy": (
            pure_pair_correct / pure_pair_total if pure_pair_total else None
        ),
        "pairwise_lcb90": wilson_lower_bound(
            pure_pair_correct, pure_pair_total
        ),
        "comparable_pairs": pure_pair_total,
        "baseline_success_rate": baseline_successes / max(groups_seen, 1),
        "top1_success_rate": selected_successes / max(groups_seen, 1),
        "top1_uplift_over_baseline": (
            selected_successes - baseline_successes
        ) / max(groups_seen, 1),
        "oracle_success_rate": oracle_successes / max(groups_seen, 1),
    }


def class_weights(
    groups: Sequence[BranchGroup],
    config: EventWorldModelConfig,
    device: torch.device,
    min_relative_support: int = 5,
) -> dict[str, torch.Tensor]:
    def balanced(values: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(values.astype(np.int64), minlength=classes).astype(np.float64)
        result = np.ones(classes, dtype=np.float32)
        present = counts > 0
        if present.any():
            result[present] = len(values) / (present.sum() * counts[present])
            result[present] /= result[present].mean()
        return result, counts

    successes = np.concatenate([group.success for group in groups])
    positive = float(successes.sum())
    success_pos = (len(successes) - positive) / max(positive, 1.0)
    event_parts: list[np.ndarray] = []
    destination_parts: list[np.ndarray] = []
    relative_parts: list[np.ndarray] = []
    predicate_parts: list[np.ndarray] = []
    for group in groups:
        event_mask = (
            group.structured_mask if config.structured_events else group.dense_mask
        )
        if event_mask.any():
            event_parts.append(group.next_event_id[event_mask])
        observed = group.dense_mask & (group.duration_observed > 0.5)
        if observed.any():
            destination_parts.append(group.next_reached_event_id[observed])
        if group.structured_mask.any():
            relative_parts.append(
                group.relative_transition_id[group.structured_mask]
            )
            predicate_parts.append(group.post_predicates[group.structured_mask])
        if group.continuation is not None and len(group.continuation["duration"]):
            continuation = group.continuation
            event_parts.append(continuation["next_event_id"])
            continuation_observed = continuation["duration_observed"] > 0.5
            if continuation_observed.any():
                destination_parts.append(
                    continuation["next_reached_event_id"][continuation_observed]
                )
            relative_parts.append(continuation["relative_transition_id"])
            predicate_parts.append(continuation["post_predicates"])
    event_values = (
        np.concatenate(event_parts) if event_parts else np.asarray([], dtype=np.int64)
    )
    event_weight, _ = balanced(event_values, config.num_events)
    observed_destination = (
        np.concatenate(destination_parts)
        if destination_parts
        else np.asarray([], dtype=np.int64)
    )
    destination_weight, _ = balanced(observed_destination, config.num_events)
    relative_values = (
        np.concatenate(relative_parts)
        if relative_parts
        else np.asarray([], dtype=np.int64)
    )
    relative_weight, relative_counts = balanced(
        relative_values, config.num_relative_transitions
    )
    relative_supported = relative_counts >= min_relative_support
    relative_weight[~relative_supported] = 0.0
    predicate_values = (
        np.concatenate(predicate_parts)
        if predicate_parts
        else np.zeros((1, config.num_predicates), dtype=np.float32)
    )
    predicate_positive = predicate_values.sum(0)
    predicate_negative = len(predicate_values) - predicate_positive
    predicate_pos = np.minimum(
        predicate_negative / np.maximum(predicate_positive, 1.0), 20.0
    ).astype(np.float32)
    outcomes = np.concatenate(
        [group.outcome_id[group.structured_mask] for group in groups if group.structured_mask.any()]
    ) if config.recovery_supervised and any(group.structured_mask.any() for group in groups) else np.asarray([], dtype=np.int64)
    outcome_weight, _ = balanced(outcomes, config.num_outcomes)
    return {
        "success_pos": torch.tensor(max(success_pos, 1.0), device=device),
        "event": torch.tensor(event_weight, device=device),
        "destination": torch.tensor(destination_weight, device=device),
        "relative": torch.tensor(relative_weight, device=device),
        "relative_supported": torch.tensor(relative_supported, device=device),
        "predicate_pos": torch.tensor(predicate_pos, device=device),
        "outcome": torch.tensor(outcome_weight, device=device),
    }


def fit_success_temperature(
    logits: np.ndarray, labels: np.ndarray, max_steps: int = 100
) -> dict[str, Any]:
    """Fit one ensemble temperature on validation candidates only."""

    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float64)
    if logits_tensor.ndim != 2 or labels_tensor.shape != logits_tensor.shape[1:]:
        raise ValueError("logits must be [members,candidates] with aligned labels")
    before_probability = torch.sigmoid(logits_tensor).mean(0)
    if len(torch.unique(labels_tensor)) < 2:
        temperature = 1.0
        status = "degenerate_validation_labels"
    else:
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature], lr=0.2, max_iter=max_steps, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature_value = log_temperature.exp().clamp(0.05, 20.0)
            probability = torch.sigmoid(logits_tensor / temperature_value).mean(0)
            loss = F.binary_cross_entropy(
                probability.clamp(1e-8, 1 - 1e-8), labels_tensor
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
        status = "fitted"
    after_probability = torch.sigmoid(logits_tensor / temperature).mean(0)
    before = before_probability.numpy()
    after = after_probability.numpy()
    raw_labels = labels_tensor.numpy()
    return {
        "temperature": temperature,
        "status": status,
        "candidates": int(len(labels)),
        "before": {
            "brier": float(np.mean(np.square(before - raw_labels))),
            "ece": expected_calibration_error(raw_labels, before),
        },
        "after": {
            "brier": float(np.mean(np.square(after - raw_labels))),
            "ece": expected_calibration_error(raw_labels, after),
        },
    }


@torch.no_grad()
def ensemble_group_predictions(
    models: Sequence[ActionConditionedEventWorldModel],
    groups: Sequence[BranchGroup],
    device: torch.device,
    object_mean: np.ndarray,
    object_std: np.ndarray,
    event_values: torch.Tensor,
    duration_scale: float,
    temperature: float,
    distance_weight: float,
    event_weight: float = 0.25,
    duration_weight: float = 0.05,
) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        raw = collate_groups(
            [group], object_mean, object_std, include_auxiliary=False
        )
        batch = move_batch(raw, device)
        member_logits = []
        member_scores = []
        member_aleatoric = []
        for model in models:
            model.eval()
            output = forward_model(model, batch)
            adjusted_success_logit = counterfactual_success_logit(
                model, output, batch
            )
            member_logits.append(adjusted_success_logit.cpu().numpy())
            score = counterfactual_rank_score(
                model,
                output,
                batch,
                event_values,
                duration_scale,
                success_temperature=temperature,
                event_weight=event_weight,
                duration_weight=duration_weight,
            ) - distance_weight * batch["candidate_distance"]
            member_scores.append(score.cpu().numpy())
            member_aleatoric.append(
                counterfactual_aleatoric_uncertainty(model, output, batch)
                .cpu()
                .numpy()
            )
        logits = np.stack(member_logits)
        calibrated_probability = 1.0 / (1.0 + np.exp(-logits / temperature))
        scores = np.stack(member_scores)
        uncertainty = calibrated_probability.std(0) + np.stack(member_aleatoric).mean(0)
        baseline_indices = [
            index for index, name in enumerate(group.candidate_names) if name == "deterministic"
        ]
        rows.append(
            {
                "logical_key": group.logical_key,
                "schema_version": group.schema_version,
                "success": group.success.copy(),
                "steps": group.steps.copy(),
                "baseline_index": baseline_indices[0] if baseline_indices else 0,
                "mean_score": scores.mean(0),
                "mean_success_probability": calibrated_probability.mean(0),
                "uncertainty": uncertainty,
            }
        )
    return rows


def predefined_scoring_grid(distance_weight: float) -> list[dict[str, Any]]:
    """Return the preregistered, deliberately small validation scoring grid."""

    distance_weight = float(distance_weight)
    if distance_weight < 0:
        raise ValueError("distance_weight must be non-negative")
    return [
        {
            "candidate_id": "success_only",
            "event_weight": 0.0,
            "duration_weight": 0.0,
            "candidate_distance_weight": 0.0,
        },
        {
            "candidate_id": "success_distance",
            "event_weight": 0.0,
            "duration_weight": 0.0,
            "candidate_distance_weight": distance_weight,
        },
        {
            "candidate_id": "progress_light",
            "event_weight": 0.10,
            "duration_weight": 0.0,
            "candidate_distance_weight": 0.0,
        },
        {
            "candidate_id": "progress",
            "event_weight": 0.25,
            "duration_weight": 0.0,
            "candidate_distance_weight": 0.0,
        },
        {
            "candidate_id": "progress_clock",
            "event_weight": 0.25,
            "duration_weight": 0.05,
            "candidate_distance_weight": 0.0,
        },
        {
            "candidate_id": "full_light",
            "event_weight": 0.10,
            "duration_weight": 0.02,
            "candidate_distance_weight": distance_weight,
        },
        {
            "candidate_id": "full",
            "event_weight": 0.25,
            "duration_weight": 0.05,
            "candidate_distance_weight": distance_weight,
        },
    ]


def _paired_delta_summary(deltas: np.ndarray) -> tuple[float | None, float | None]:
    if not len(deltas):
        return None, None
    mean = float(deltas.mean())
    if len(deltas) < 2:
        return mean, None
    standard_error = float(deltas.std(ddof=1) / math.sqrt(len(deltas)))
    return mean, mean - 1.645 * standard_error


def select_validation_scoring(
    candidate_rows: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    *,
    minimum_proposals: int = 10,
    minimum_coverage: float = 0.10,
    minimum_lcb: float = 0.0,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, Any]]:
    """Select once from a fixed scoring grid using validation outcomes only.

    The grid and lexicographic rule are preregistered.  Every candidate result
    is retained in the audit, and guard tuning happens only after this choice.
    """

    if minimum_proposals <= 0:
        raise ValueError("minimum_proposals must be positive")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must lie in (0,1]")
    if not candidate_rows:
        raise ValueError("scoring grid is empty")

    audits: list[dict[str, Any]] = []
    rows_by_id: dict[str, list[Mapping[str, Any]]] = {}
    specs_by_id: dict[str, dict[str, Any]] = {}
    for order, (raw_spec, raw_rows) in enumerate(candidate_rows):
        spec = dict(raw_spec)
        candidate_id = str(spec["candidate_id"])
        if candidate_id in rows_by_id:
            raise ValueError(f"duplicate scoring candidate id: {candidate_id}")
        rows = list(raw_rows)
        rows_by_id[candidate_id] = rows
        specs_by_id[candidate_id] = spec
        proposed_deltas = []
        policy_success = 0.0
        baseline_success = 0.0
        helpful = 0
        harmful = 0
        for row in rows:
            score = np.asarray(row["mean_score"])
            success = np.asarray(row["success"])
            baseline = int(row["baseline_index"])
            selected = int(np.argmax(score))
            baseline_success += float(success[baseline])
            policy_success += float(success[selected])
            if selected != baseline:
                delta = float(success[selected] - success[baseline])
                proposed_deltas.append(delta)
                helpful += int(delta > 0)
                harmful += int(delta < 0)
        total_groups = max(len(rows), 1)
        deltas = np.asarray(proposed_deltas, dtype=np.float64)
        mean_delta, lower_bound = _paired_delta_summary(deltas)
        proposals = len(deltas)
        coverage = proposals / total_groups
        eligible = (
            proposals >= minimum_proposals
            and coverage >= minimum_coverage
            and lower_bound is not None
            and lower_bound >= minimum_lcb
        )
        audits.append(
            {
                **spec,
                "grid_order": order,
                "validation_groups": len(rows),
                "nonbaseline_proposals": proposals,
                "coverage": coverage,
                "helpful_changes": helpful,
                "harmful_changes": harmful,
                "mean_proposal_paired_success_delta": mean_delta,
                "proposal_paired_success_delta_lcb90": lower_bound,
                "baseline_success_rate": baseline_success / total_groups,
                "unguarded_policy_success_rate": policy_success / total_groups,
                "passes_pre_guard_evidence_gate": eligible,
            }
        )

    eligible_audits = [
        audit for audit in audits if audit["passes_pre_guard_evidence_gate"]
    ]
    pool = eligible_audits if eligible_audits else audits

    def selection_key(audit: Mapping[str, Any]) -> tuple[float, float, float, int]:
        lower_bound = audit["proposal_paired_success_delta_lcb90"]
        mean_delta = audit["mean_proposal_paired_success_delta"]
        return (
            float(lower_bound) if lower_bound is not None else -math.inf,
            float(audit["unguarded_policy_success_rate"]),
            float(mean_delta) if mean_delta is not None else -math.inf,
            -int(audit["grid_order"]),
        )

    selected_audit = max(pool, key=selection_key)
    selected_id = str(selected_audit["candidate_id"])
    audit = {
        "grid_version": SCORING_GRID_VERSION,
        "selection_rule": SCORING_SELECTION_RULE,
        "minimum_proposals": minimum_proposals,
        "minimum_coverage": minimum_coverage,
        "minimum_lcb90": minimum_lcb,
        "selection_pool": (
            "pre_guard_evidence_eligible" if eligible_audits else "all_grid_candidates"
        ),
        "selected_candidate_id": selected_id,
        "candidates": audits,
    }
    return specs_by_id[selected_id], rows_by_id[selected_id], audit


def tune_guard(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_guarded_groups: int = 10,
    min_coverage: float = 0.10,
    minimum_lcb: float = 0.0,
    max_harmful_rate: float = 0.10,
) -> dict[str, Any]:
    """Choose validation-only gain/uncertainty gates with conservative fallback."""

    if min_guarded_groups <= 0:
        raise ValueError("min_guarded_groups must be positive")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must lie in (0,1]")
    if not 0 <= max_harmful_rate <= 1:
        raise ValueError("max_harmful_rate must lie in [0,1]")

    candidates = []
    for row in rows:
        score = np.asarray(row["mean_score"])
        uncertainty = np.asarray(row["uncertainty"])
        baseline = int(row["baseline_index"])
        best = int(np.argmax(score))
        if best == baseline:
            continue
        candidates.append(
            {
                "success_delta": float(row["success"][best] - row["success"][baseline]),
                "gain": float(score[best] - score[baseline]),
                "uncertainty": float(uncertainty[best]),
            }
        )
    if not candidates:
        return {
            "enabled": False,
            "reason": "no_nonbaseline_validation_selection",
            "gain_margin": None,
            "uncertainty_threshold": None,
            "coverage": 0.0,
            "grid_version": GUARD_GRID_VERSION,
            "minimum_guarded_groups": min_guarded_groups,
            "minimum_coverage": min_coverage,
            "minimum_lcb": minimum_lcb,
            "maximum_harmful_rate": max_harmful_rate,
            "threshold_candidates": [],
        }
    gains = np.asarray([row["gain"] for row in candidates])
    uncertainties = np.asarray([row["uncertainty"] for row in candidates])
    gain_grid = np.unique(np.quantile(gains, GUARD_GAIN_QUANTILES))
    uncertainty_grid = np.unique(
        np.quantile(uncertainties, GUARD_UNCERTAINTY_QUANTILES)
    )
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, int] | None = None
    total_groups = max(len(rows), 1)
    threshold_candidates: list[dict[str, Any]] = []
    for gain_margin in gain_grid:
        for uncertainty_threshold in uncertainty_grid:
            selected = [
                item
                for item in candidates
                if item["gain"] >= gain_margin
                and item["uncertainty"] <= uncertainty_threshold
            ]
            coverage = len(selected) / total_groups
            delta = np.asarray([item["success_delta"] for item in selected])
            mean, lower_bound = _paired_delta_summary(delta)
            harmful_changes = int((delta < 0).sum())
            harmful_rate = harmful_changes / max(len(selected), 1)
            rejection_reasons = []
            if len(selected) < min_guarded_groups:
                rejection_reasons.append("insufficient_guarded_groups")
            if coverage < min_coverage:
                rejection_reasons.append("insufficient_coverage")
            if lower_bound is None or lower_bound < minimum_lcb:
                rejection_reasons.append("lcb90_below_minimum")
            if harmful_rate > max_harmful_rate:
                rejection_reasons.append("harmful_rate_above_maximum")
            threshold_audit = {
                "gain_margin": float(gain_margin),
                "uncertainty_threshold": float(uncertainty_threshold),
                "guarded_groups": len(selected),
                "coverage": coverage,
                "mean_paired_success_delta": mean,
                "paired_success_delta_lcb90": lower_bound,
                "harmful_changes": harmful_changes,
                "harmful_rate": harmful_rate,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
            threshold_candidates.append(threshold_audit)
            if rejection_reasons:
                continue
            assert mean is not None and lower_bound is not None
            policy_success = sum(
                float(row["success"][int(np.argmax(row["mean_score"]))])
                if (
                    int(np.argmax(row["mean_score"])) != int(row["baseline_index"])
                    and float(np.max(row["mean_score"]) - row["mean_score"][int(row["baseline_index"])]) >= gain_margin
                    and float(row["uncertainty"][int(np.argmax(row["mean_score"]))]) <= uncertainty_threshold
                )
                else float(row["success"][int(row["baseline_index"])] )
                for row in rows
            ) / total_groups
            proposal = {
                "enabled": True,
                "gain_margin": float(gain_margin),
                "uncertainty_threshold": float(uncertainty_threshold),
                "guarded_groups": len(selected),
                "coverage": coverage,
                "mean_paired_success_delta": mean,
                "paired_success_delta_lcb90": lower_bound,
                "harmful_changes": harmful_changes,
                "harmful_rate": harmful_rate,
                "validation_policy_success_rate": policy_success,
            }
            key = (policy_success, lower_bound, len(selected))
            if best_key is None or key > best_key:
                best = proposal
                best_key = key
    if best is None:
        return {
            "enabled": False,
            "reason": "no_threshold_met_conservative_validation_gate",
            "gain_margin": None,
            "uncertainty_threshold": None,
            "coverage": 0.0,
            "grid_version": GUARD_GRID_VERSION,
            "minimum_lcb": minimum_lcb,
            "minimum_guarded_groups": min_guarded_groups,
            "minimum_coverage": min_coverage,
            "maximum_harmful_rate": max_harmful_rate,
            "threshold_candidates": threshold_candidates,
        }
    best["grid_version"] = GUARD_GRID_VERSION
    best["minimum_lcb"] = minimum_lcb
    best["minimum_guarded_groups"] = min_guarded_groups
    best["minimum_coverage"] = min_coverage
    best["maximum_harmful_rate"] = max_harmful_rate
    best["threshold_candidates"] = threshold_candidates
    return best


def load_pretrained(path: Path) -> tuple[dict[str, Any], EventWorldModelConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise RuntimeError("pretrained checkpoint must contain model and config")
    config = EventWorldModelConfig.from_dict(checkpoint["config"])
    return checkpoint, config


def validated_pretrained_policy_bridge(
    checkpoint: Mapping[str, Any], *, required: bool
) -> dict[str, Any] | None:
    """Validate an embedded bridge before any rollout label can be opened."""

    contract = checkpoint.get("contract")
    has_bridge = isinstance(contract, Mapping) and (
        POLICY_BRIDGE_CONTRACT_KEY in contract
    )
    if required and not has_bridge:
        raise RuntimeError(
            "formal native-policy training requires a strict policy feature/action bridge"
        )
    if not has_bridge:
        return None
    bridge = validate_checkpoint_policy_bridge_header(
        checkpoint.get("config", {}), contract
    )
    return dict(bridge)


def load_counterfactual_pretrained_state(
    model: ActionConditionedEventWorldModel,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Load a factual model while permitting only the new rank-head keys."""

    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = (
        {
            f"action_rank_head.{key}"
            for key in model.action_rank_head.state_dict()
        }
        if model.action_rank_head is not None
        else set()
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected or missing not in (set(), allowed_missing):
        raise RuntimeError(
            "pretrained/counterfactual architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def configure_action_rank_training(
    model: ActionConditionedEventWorldModel,
    *,
    freeze_factual_core: bool,
) -> dict[str, Any]:
    """Freeze the factual world model and expose only the relative rank head.

    The returned manifest is suitable for a training/provenance contract.  It
    is derived from ``named_parameters`` after the mutation so callers can
    independently assert the exact trainable state instead of trusting a flag.
    """

    if not freeze_factual_core:
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        return {
            "freeze_factual_core": False,
            "trainable_parameter_names": trainable,
            "trainable_parameter_count": int(
                sum(value.numel() for value in model.parameters() if value.requires_grad)
            ),
        }
    if model.action_rank_head is None:
        raise ValueError("freeze_factual_core requires action_rank_residual=True")
    if not model.config.action_rank_success_only:
        raise ValueError(
            "freeze_factual_core requires action_rank_success_only=True"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_rank_head.parameters():
        parameter.requires_grad_(True)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    expected = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("action_rank_head.")
    ]
    if trainable != expected or not trainable:
        raise RuntimeError("frozen-core mode exposed parameters outside action_rank_head")
    count = int(sum(value.numel() for value in model.parameters() if value.requires_grad))
    expected_count = 2 * model.config.semantic_dim
    if count != expected_count:
        raise RuntimeError(
            "low-capacity action rank head must have exactly 2*semantic_dim "
            f"parameters, found {count} instead of {expected_count}"
        )
    return {
        "freeze_factual_core": True,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": count,
        "expected_trainable_parameter_count": expected_count,
        "factual_core_trainable_parameters": 0,
        "ranking_utility": "success_logit_plus_baseline_relative_residual_only",
    }


def train_member(
    *,
    seed: int,
    pretrained: Mapping[str, Any],
    config: EventWorldModelConfig,
    train_groups: Sequence[BranchGroup],
    validation_groups: Sequence[BranchGroup],
    object_mean: np.ndarray,
    object_std: np.ndarray,
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reserved_target_rows = validate_reserved_target_rows(pretrained, config)
    source_action_normalization = resolve_source_action_normalization(
        pretrained, train_groups, config
    )
    recorded_action_normalization = contract.get("source_action_normalization")
    if source_action_normalization is not None and (
        not isinstance(recorded_action_normalization, Mapping)
        or dict(recorded_action_normalization) != source_action_normalization
    ):
        raise RuntimeError(
            "member source action normalization differs from the train-only contract"
        )
    model = ActionConditionedEventWorldModel(config)
    load_counterfactual_pretrained_state(model, pretrained["model"])
    install_source_action_normalization(model, source_action_normalization)
    assert_reserved_target_rows_bit_exact(model, reserved_target_rows)
    freeze_factual_core = bool(getattr(args, "freeze_factual_core", False))
    if freeze_factual_core and bool(getattr(args, "unfreeze_semantic", False)):
        raise ValueError("freeze_factual_core conflicts with unfreeze_semantic")
    if not freeze_factual_core and not args.unfreeze_semantic:
        model.freeze_semantic()
    optimization_contract = configure_action_rank_training(
        model, freeze_factual_core=freeze_factual_core
    )
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_name = args.amp
    if amp_name == "auto":
        amp_name = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "off"
        )
    amp_enabled = device.type == "cuda" and amp_name != "off"
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_name == "fp16")
    collate = partial(collate_groups, object_mean=object_mean, object_std=object_std)
    generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        GroupDataset(train_groups),
        batch_size=args.groups_per_batch,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        GroupDataset(validation_groups),
        batch_size=args.groups_per_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    loss_metadata = class_weights(
        train_groups,
        config,
        device,
        min_relative_support=args.min_relative_support,
    )
    event_values = torch.linspace(0.0, 1.0, config.num_events, device=device)
    duration_parts = [
        group.duration[group.dense_mask]
        for group in train_groups
        if group.dense_mask.any()
    ]
    duration_parts.extend(
        group.continuation["duration"]
        for group in train_groups
        if group.continuation is not None and len(group.continuation["duration"])
    )
    duration_values = (
        np.concatenate(duration_parts) if duration_parts else np.asarray([25.0])
    )
    duration_scale = float(max(np.mean(duration_values), 1.0))
    weights = {
        "success": args.success_weight,
        "outcome": args.outcome_weight,
        "pairwise": args.pairwise_weight,
        "listwise": args.listwise_weight,
        "group_centered": args.group_centered_weight,
        "baseline_contrast": args.baseline_contrast_weight,
        "event": args.event_weight,
        "relative": args.relative_weight,
        "destination": args.destination_weight,
        "predicate": args.predicate_weight,
        "reach": args.reach_weight,
        "duration": args.duration_weight,
        "object": args.object_weight,
        "latent": args.latent_weight,
    }
    member_dir = output / "members" / f"seed_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    log_path = member_dir / "train_log.jsonl"
    best_path = member_dir / "event_world_model_counterfactual_best.pt"
    step = best_step = 0
    best_score = math.inf
    best_selection_key: tuple[float, ...] | None = None
    iterator = iter(train_loader)
    started = time.time()
    while step < args.steps:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        assert_reserved_ids_absent(raw, reserved_target_rows)
        batch = move_batch(raw, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, pieces, _ = compute_loss(
                model,
                batch,
                weights,
                loss_metadata["success_pos"],
                loss_metadata["event"],
                event_values,
                duration_scale,
                destination_class_weight=loss_metadata["destination"],
                relative_class_weight=loss_metadata["relative"],
                relative_supported=loss_metadata["relative_supported"],
                predicate_pos_weight=loss_metadata["predicate_pos"],
                outcome_class_weight=loss_metadata["outcome"],
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at seed={seed}, step={step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        restore_reserved_target_rows(model, reserved_target_rows)
        scaler.update()
        step += 1
        row: dict[str, Any] = {
            "step": step,
            "wall_seconds": time.time() - started,
            **{f"train_{key}": float(value.detach()) for key, value in pieces.items()},
        }
        if step % args.eval_every == 0 or step == args.steps:
            assert_reserved_target_rows_bit_exact(model, reserved_target_rows)
            validation = evaluate_model(
                model,
                validation_loader,
                device,
                weights,
                loss_metadata,
                event_values,
                duration_scale,
                reserved_target_rows=reserved_target_rows,
            )
            score = float(validation["losses"]["total"])
            selection_key = member_validation_selection_key(validation)
            row["validation"] = validation
            row["validation_selection_score"] = score
            row["validation_selection_rule"] = MEMBER_SELECTION_RULE
            row["validation_selection_key"] = list(selection_key)
            if best_selection_key is None or selection_key > best_selection_key:
                assert_reserved_target_rows_bit_exact(model, reserved_target_rows)
                best_score = score
                best_selection_key = selection_key
                best_step = step
                source_only_proof = reserved_rows_source_only_proof(
                    model,
                    reserved_target_rows,
                    source_training_steps=step,
                    source_training_groups=len(train_groups),
                    input_pretrained_checkpoint_sha256=str(
                        contract.get("pretrained_sha256", "")
                    ),
                    action_normalization=source_action_normalization,
                )
                member_contract = {
                    **dict(contract),
                    "action_rank_optimization": optimization_contract,
                }
                if source_action_normalization is not None:
                    member_contract["source_action_normalization"] = dict(
                        source_action_normalization
                    )
                if source_only_proof is not None:
                    member_contract["reserved_target_rows_source_only_proof"] = dict(
                        source_only_proof
                    )
                checkpoint_payload: dict[str, Any] = {
                    "model": model.state_dict(),
                    "config": dataclasses.asdict(config),
                    "step": step,
                    "best_step": step,
                    "best_score": score,
                    "best_selection_rule": MEMBER_SELECTION_RULE,
                    "best_selection_key": list(selection_key),
                    "seed": seed,
                    "contract": member_contract,
                    "normalization": {
                        "object_delta_mean": object_mean,
                        "object_delta_std": object_std,
                    },
                    "duration_scale": duration_scale,
                    "loss_weights": weights,
                    "validation": validation,
                }
                if source_action_normalization is not None:
                    checkpoint_payload["normalization"].update(
                        {
                            "action_mean": np.asarray(
                                source_action_normalization["action_mean"],
                                dtype=np.float32,
                            ),
                            "action_std": np.asarray(
                                source_action_normalization["action_std"],
                                dtype=np.float32,
                            ),
                            "action_valid_sample_count": int(
                                source_action_normalization[
                                    "valid_action_sample_count"
                                ]
                            ),
                            "action_statistics_sha256": source_action_normalization[
                                "statistics_sha256"
                            ],
                        }
                    )
                if source_only_proof is not None:
                    checkpoint_payload["reserved_target_rows_source_only_proof"] = (
                        dict(source_only_proof)
                    )
                atomic_torch_save(
                    best_path,
                    checkpoint_payload,
                )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("COUNTERFACTUAL=" + json.dumps({"seed": seed, **row}, sort_keys=True), flush=True)
        if args.early_stopping_patience > 0 and step - best_step >= args.early_stopping_patience:
            break
    if not best_path.is_file():
        raise RuntimeError(f"member {seed} produced no best checkpoint")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--event-spec",
        type=Path,
        help="Stage-1 event specification; mandatory when schema-v4 is present.",
    )
    parser.add_argument(
        "--allow-legacy-only",
        action="store_true",
        help="Permit a compatibility/smoke run without schema-v4; never use for formal training.",
    )
    parser.add_argument("--object-names", nargs="+", default=["can"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260827, 20260828, 20260829])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--unfreeze-semantic", action="store_true")
    parser.add_argument(
        "--require-policy-feature-action-bridge",
        action="store_true",
        help=(
            "Require and propagate a strict checkpoint-side policy feature/action "
            "bridge. Formal native-policy source training must enable this."
        ),
    )
    parser.add_argument(
        "--freeze-factual-core",
        action="store_true",
        help=(
            "Freeze every pretrained world-model parameter and train only the "
            "low-capacity baseline-relative action_rank_head.  This mode also "
            "enforces success-only ranking at training and deployment."
        ),
    )
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=0.2)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--listwise-weight", type=float, default=0.5)
    parser.add_argument("--group-centered-weight", type=float, default=1.0)
    parser.add_argument("--baseline-contrast-weight", type=float, default=1.5)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=0.5)
    parser.add_argument("--destination-weight", type=float, default=0.5)
    parser.add_argument("--predicate-weight", type=float, default=0.5)
    parser.add_argument("--reach-weight", type=float, default=0.75)
    parser.add_argument("--duration-weight", type=float, default=0.5)
    parser.add_argument("--object-weight", type=float, default=0.5)
    parser.add_argument("--latent-weight", type=float, default=0.5)
    parser.add_argument("--distance-weight", type=float, default=0.02)
    parser.add_argument("--guard-min-groups", type=int, default=10)
    parser.add_argument("--guard-min-coverage", type=float, default=0.10)
    parser.add_argument("--guard-min-lcb", type=float, default=0.0)
    parser.add_argument("--guard-max-harmful-rate", type=float, default=0.10)
    parser.add_argument("--min-relative-support", type=int, default=5)
    parser.add_argument("--min-recovery-support", type=int, default=5)
    parser.add_argument(
        "--regression-persistence-steps",
        type=int,
        default=DEFAULT_REGRESSION_PERSISTENCE_STEPS,
        help=(
            "Minimum consecutive simulator states below the pre-drop dynamic-phase "
            "peak, and at/above that peak for non-terminal recovery."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.groups_per_batch <= 0 or args.eval_every <= 0:
        raise ValueError("steps, groups-per-batch, and eval-every must be positive")
    if (
        args.min_relative_support <= 0
        or args.min_recovery_support <= 0
        or args.regression_persistence_steps <= 0
    ):
        raise ValueError("minimum structured-label supports must be positive")
    if not 0 < args.guard_min_coverage <= 1:
        raise ValueError("guard-min-coverage must lie in (0,1]")
    if not 0 <= args.guard_max_harmful_rate <= 1:
        raise ValueError("guard-max-harmful-rate must lie in [0,1]")
    if args.distance_weight < 0:
        raise ValueError("distance-weight must be non-negative")
    if args.freeze_factual_core and args.unfreeze_semantic:
        raise ValueError("freeze-factual-core conflicts with unfreeze-semantic")
    loss_weight_names = (
        "success_weight",
        "outcome_weight",
        "pairwise_weight",
        "listwise_weight",
        "group_centered_weight",
        "baseline_contrast_weight",
        "event_weight",
        "relative_weight",
        "destination_weight",
        "predicate_weight",
        "reach_weight",
        "duration_weight",
        "object_weight",
        "latent_weight",
    )
    if any(float(getattr(args, name)) < 0 for name in loss_weight_names):
        raise ValueError("loss weights must be non-negative")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("ensemble seeds must be non-empty and unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    pretrained, config = load_pretrained(args.pretrained)
    # Validate the optional dual reservation before descriptor discovery or any
    # source label dataset is opened.  A legacy checkpoint has no such key and
    # follows the historical path unchanged.
    reserved_target_rows = validate_reserved_target_rows(pretrained, config)
    if config.object_delta_dim != 3 * len(args.object_names):
        raise RuntimeError(
            f"checkpoint object_delta_dim={config.object_delta_dim} but selected objects "
            f"require {3 * len(args.object_names)} xyz values"
        )
    checkpoint_contract = pretrained.get("contract", {})
    policy_bridge = validated_pretrained_policy_bridge(
        pretrained, required=args.require_policy_feature_action_bridge
    )
    raw_body_to_id = checkpoint_contract.get("body_to_id", {"unknown": 0})
    raw_policy_to_id = checkpoint_contract.get("policy_to_id", {"openvla": 0})
    body_to_id = {
        str(name): int(identity_id) for name, identity_id in raw_body_to_id.items()
    }
    policy_to_id = canonical_policy_mapping(
        raw_policy_to_id
    )
    calibrations: Mapping[str, Mapping[str, Any]] | None = None
    event_spec_sha256: str | None = None
    if args.event_spec is not None:
        event_spec = json.loads(args.event_spec.read_text(encoding="utf-8"))
        calibrations = event_spec.get("calibration")
        if not isinstance(calibrations, Mapping):
            raise RuntimeError("event spec has no calibration mapping")
        event_spec_sha256 = sha256(args.event_spec)
    # Identity-only scan and split happen before any label dataset is opened.
    # In particular, read_group is never invoked for sealed-test descriptors.
    descriptors = scan_group_descriptors(args.data)
    schema5_descriptors = [
        descriptor for descriptor in descriptors if descriptor.schema_version == 5
    ]
    if not schema5_descriptors and not args.allow_legacy_only:
        raise RuntimeError(
            "formal counterfactual training requires schema-v5 continuation data; use "
            "--allow-legacy-only only for compatibility tests"
        )
    splits = (
        read_split_manifest(args.split_manifest, descriptors)
        if args.split_manifest
        else make_group_splits(descriptors)
    )
    descriptor_map = {
        descriptor.logical_key: descriptor for descriptor in descriptors
    }
    train_descriptors = [descriptor_map[key] for key in splits["train"]]
    validation_descriptors = [
        descriptor_map[key] for key in splits["validation"]
    ]
    sealed_test_descriptors = [descriptor_map[key] for key in splits["test"]]
    loaded_groups = load_descriptor_groups(
        [*train_descriptors, *validation_descriptors],
        config,
        args.object_names,
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
        regression_persistence_steps=args.regression_persistence_steps,
        expected_event_spec_sha256=event_spec_sha256,
    )
    group_map = {group.logical_key: group for group in loaded_groups}
    train_groups = [group_map[key] for key in splits["train"]]
    validation_groups = [group_map[key] for key in splits["validation"]]
    assert_source_groups_avoid_reserved_ids(loaded_groups, reserved_target_rows)
    state_contracts: dict[str, dict[str, Any]] = {}
    for group in loaded_groups:
        if group.state_contract is None:
            if group.policy == "smolvla" and group.schema_version == 5:
                raise RuntimeError(
                    f"SmolVLA schema-v5 group lacks a shared-state contract: {group.path}"
                )
            continue
        previous_state_contract = state_contracts.get(group.policy)
        if previous_state_contract is not None and previous_state_contract != group.state_contract:
            raise RuntimeError(
                f"state representation changed within policy {group.policy!r}"
            )
        state_contracts[group.policy] = dict(group.state_contract)
    if state_contracts:
        pretrained_state_contracts = checkpoint_contract.get("state_contracts")
        if not isinstance(pretrained_state_contracts, Mapping):
            raise RuntimeError(
                "pretrained checkpoint lacks state_contracts; cannot prove that its "
                "semantic encoder was trained on the collected SmolVLA prefix state"
            )
        for policy_name, state_contract in state_contracts.items():
            if pretrained_state_contracts.get(policy_name) != state_contract:
                raise RuntimeError(
                    f"pretrained/collector state contract mismatch for {policy_name!r}"
                )
    pretrained_state_contracts = checkpoint_contract.get("state_contracts")
    if isinstance(pretrained_state_contracts, Mapping):
        for policy_name, state_contract in pretrained_state_contracts.items():
            if not isinstance(state_contract, Mapping):
                raise RuntimeError(
                    f"invalid pretrained state contract for {policy_name!r}"
                )
            state_contracts.setdefault(str(policy_name), dict(state_contract))
    train_recovery_count = sum(
        int((group.trajectory_recovery & group.structured_mask).sum())
        for group in train_groups
    )
    validation_recovery_count = sum(
        int((group.trajectory_recovery & group.structured_mask).sum())
        for group in validation_groups
    )
    train_regress_count = sum(
        int((group.trajectory_regress & group.structured_mask).sum())
        for group in train_groups
    )
    validation_regress_count = sum(
        int((group.trajectory_regress & group.structured_mask).sum())
        for group in validation_groups
    )
    recovery_supervised = train_recovery_count >= args.min_recovery_support
    config = dataclasses.replace(
        config,
        recovery_supervised=recovery_supervised,
        action_rank_residual=True,
        action_rank_success_only=bool(args.freeze_factual_core),
    )
    source_action_normalization = resolve_source_action_normalization(
        pretrained, train_groups, config
    )
    # The sealed-test files have only had identity attrs read.  Their datasets
    # are neither audited nor loaded by this training process.
    normalization = pretrained.get("normalization", {})
    if "object_delta_mean" in normalization and "object_delta_std" in normalization:
        object_mean = np.asarray(normalization["object_delta_mean"], dtype=np.float32)
        object_std = np.asarray(normalization["object_delta_std"], dtype=np.float32)
    else:
        object_parts = [
            group.object_delta[group.dense_mask]
            for group in train_groups
            if group.dense_mask.any()
        ]
        object_parts.extend(
            group.continuation["object_delta"]
            for group in train_groups
            if group.continuation is not None and len(group.continuation["duration"])
        )
        dense_objects = (
            np.concatenate(object_parts)
            if object_parts
            else np.zeros((1, config.object_delta_dim), dtype=np.float32)
        )
        object_mean = dense_objects.mean(0).astype(np.float32)
        object_std = np.maximum(dense_objects.std(0), 1e-4).astype(np.float32)
    if object_mean.shape != (config.object_delta_dim,) or object_std.shape != object_mean.shape:
        raise RuntimeError("checkpoint object normalization is incompatible")
    args.output.mkdir(parents=True, exist_ok=True)
    structured_tasks = sorted(
        {group.task for group in loaded_groups if group.schema_version >= 4}
    )
    if len(structured_tasks) > 1:
        raise RuntimeError(
            "one ensemble cannot expose a single online predicate calibration for "
            f"multiple tasks: {structured_tasks}"
        )
    task_calibration = (
        dict(calibrations[structured_tasks[0]])
        if structured_tasks and calibrations is not None
        else None
    )
    predicate_contract = {
        "names": list(config.predicate_names),
        "derivation": "derive_atomic_predicates_v1",
        "source": "simulator_object_poses_at_query_step",
        "event_spec_sha256": event_spec_sha256,
        "task_calibration": task_calibration,
        "online_requires_explicit_predicates": True,
        "missing_policy": "error",
    }
    candidate_contract = {
        "baseline_candidate_name": "deterministic",
        "fallback_index": 0,
    }
    contract = {
        "trainer": "schema_v2_to_v5_structured_counterfactual_v6",
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": sha256(args.pretrained),
        "events": list(config.event_names),
        "object_names": list(args.object_names),
        "body_to_id": (
            {str(name): int(value) for name, value in raw_body_to_id.items()}
            if reserved_target_rows is not None
            else dict(body_to_id)
        ),
        "policy_to_id": (
            {str(name): int(value) for name, value in raw_policy_to_id.items()}
            if reserved_target_rows is not None
            else dict(policy_to_id)
        ),
        "state_contracts": state_contracts,
        "causal_history_contract": causal_history_contract(),
        "train_groups": splits["train"],
        "validation_groups": splits["validation"],
        "sealed_test_groups": splits["test"],
        "sealed_test_access": "identity_attrs_and_raw_file_sha256_only_no_label_datasets",
        "sealed_test_files": [
            {
                "logical_key": descriptor.logical_key,
                "schema_version": descriptor.schema_version,
                "path": descriptor.path,
                "sha256": sha256(Path(descriptor.path)),
            }
            for descriptor in sealed_test_descriptors
        ],
        "schema_counts": {
            str(schema): sum(
                descriptor.schema_version == schema for descriptor in descriptors
            )
            for schema in SUPPORTED_SCHEMAS
        },
        "weak_schema_v2_contract": "success_only_ranking_no_step_utility",
        "dense_schema_v3_contract": "event_time_object_future_latent",
        "structured_schema_v4_contract": (
            "dynamic_predicates_post_phase_relative_transition_"
            "persistent_phase_regression_recovery"
        ),
        "auxiliary_schema_v5_contract": (
            "same_branch_causal_prefix_continuation_query_event_time_object_"
            "future_latent_no_rank_no_terminal_reweight"
        ),
        "formal_training_requires_schema_v5": True,
        "event_spec": str(args.event_spec.resolve()) if args.event_spec else None,
        "event_spec_sha256": event_spec_sha256,
        "recovery_supervised": recovery_supervised,
        "min_recovery_support": args.min_recovery_support,
        "predicate_contract": predicate_contract,
        "candidate_contract": candidate_contract,
        "counterfactual_ranking_contract": {
            "group_centering": "candidate_score_minus_intervention_group_mean",
            "centered_target": "2x_(success_minus_intervention_group_mean)",
            "baseline_contrast": (
                "success_changing_candidates_vs_deterministic_fallback_margin_1"
            ),
            "pairwise_target": (
                "success_changing_candidate_pairs_only_terminal_steps_excluded"
            ),
            "listwise_target": (
                "softmax_2x_binary_success_uniform_within_outcome_"
                "terminal_steps_excluded_normalized_by_log_candidate_count"
            ),
            "duration_supervision": "dedicated_duration_head_only_not_ranking_utility",
            "candidate_cardinality": {
                "variable_candidate_count_supported": True,
                "minimum_candidates_per_group": 2,
                "unique_baseline_name": "deterministic",
                "baseline_index": 0,
                "pairwise_reduction": "mean_pairs_then_mean_groups",
                "listwise_reduction": "cross_entropy_div_log_C_then_mean_groups",
                "train_count_histogram": {
                    str(count): sum(
                        group.candidate_count == count for group in train_groups
                    )
                    for count in sorted(
                        {group.candidate_count for group in train_groups}
                    )
                },
                "validation_count_histogram": {
                    str(count): sum(
                        group.candidate_count == count
                        for group in validation_groups
                    )
                    for count in sorted(
                        {group.candidate_count for group in validation_groups}
                    )
                },
            },
            "action_sensitivity": {
                "enabled": True,
                "architecture": (
                    "baseline_relative_diagonal_bilinear_2d_linear_v2"
                    if args.freeze_factual_core
                    else "baseline_relative_action_effect_residual_v1"
                ),
                "inputs": [
                    "action_effect_minus_deterministic_action_effect",
                    "shared_semantic_times_action_effect_delta",
                ],
                "baseline_residual": 0.0,
                "absolute_success_supervision": "base_world_success_logit_only",
                "ranking_gradient": (
                    "rank_head_only_semantic_and_action_effect_detached"
                    if args.freeze_factual_core
                    else "residual_branch_with_base_score_and_features_stop_gradient"
                ),
                "deployment": "predict_candidates_adds_residual_to_success_logit",
                "ranking_utility": (
                    "success_only_no_fixed_event_duration_or_distance_utility"
                    if args.freeze_factual_core
                    else "validation_selected_composite_utility"
                ),
                "freeze_factual_core": bool(args.freeze_factual_core),
                "trainable_rank_parameters": 2 * config.semantic_dim,
                "event_time_object_heads": (
                    "bit_exact_frozen"
                    if args.freeze_factual_core
                    else "trainable_by_dedicated_supervision"
                ),
            },
            "validation_metrics": {
                "member_selection_primary": "pure_success_pair_lcb90",
                "pure_success_pair": (
                    "all_unordered_within_group_pairs_with_different_binary_success"
                ),
                "baseline_changing_pair": (
                    "success_changing_candidates_vs_deterministic_fallback"
                ),
                "legacy_pairwise_alias": "pure_success_pair",
            },
            "member_selection_data": "validation_only_no_sealed_test",
            "member_selection_rule": MEMBER_SELECTION_RULE,
            "pairwise_confidence": "one_sided_wilson_lcb90",
            "loss_weights": {
                "pairwise": args.pairwise_weight,
                "listwise": args.listwise_weight,
                "group_centered": args.group_centered_weight,
                "baseline_contrast": args.baseline_contrast_weight,
            },
        },
        "regression_recovery_label_contract": {
            "phase_drop_persistence_simulator_states": (
                args.regression_persistence_steps
            ),
            "regression": (
                "dynamic_phase_below_pre_drop_peak_for_minimum_persistence"
            ),
            "recovery": (
                "later_at_or_above_pre_drop_peak_for_minimum_persistence_"
                "or_later_terminal_success_eK"
            ),
            "predicate_downflip_alone_is_regression": False,
        },
        "group_files": [
            {
                "logical_key": group.logical_key,
                "schema_version": group.schema_version,
                "path": group.path,
                "sha256": sha256(Path(group.path)),
            }
            for group in descriptors
        ],
    }
    if policy_bridge is not None:
        contract[POLICY_BRIDGE_CONTRACT_KEY] = dict(policy_bridge)
    if reserved_target_rows is not None:
        contract["reserved_target_rows"] = {
            str(axis): dict(spec)
            for axis, spec in checkpoint_contract["reserved_target_rows"].items()
        }
        source_identity_rows = checkpoint_contract.get("source_identity_rows")
        if isinstance(source_identity_rows, Mapping):
            contract["source_identity_rows"] = {
                str(axis): dict(spec)
                for axis, spec in source_identity_rows.items()
            }
    if source_action_normalization is not None:
        contract["source_action_normalization"] = dict(
            source_action_normalization
        )
        contract["action_normalization"] = dict(source_action_normalization)
    atomic_json(
        args.output / "split_manifest.json",
        {
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"],
            "test_policy": "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened",
        },
    )
    atomic_json(
        args.output / "data_audit.json",
        {
            "logical_groups": len(descriptors),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "sealed_test_groups_not_evaluated": len(splits["test"]),
            "train_candidates": sum(group.candidate_count for group in train_groups),
            "train_dense_candidates": sum(int(group.dense_mask.sum()) for group in train_groups),
            "validation_candidates": sum(group.candidate_count for group in validation_groups),
            "validation_dense_candidates": sum(int(group.dense_mask.sum()) for group in validation_groups),
            "train_auxiliary_continuation_transitions": sum(
                len(group.continuation["duration"])
                for group in train_groups
                if group.continuation is not None
            ),
            "validation_auxiliary_continuation_transitions": sum(
                len(group.continuation["duration"])
                for group in validation_groups
                if group.continuation is not None
            ),
            "train_structured_v4_candidates": sum(
                int(group.structured_mask.sum()) for group in train_groups
            ),
            "validation_structured_v4_candidates": sum(
                int(group.structured_mask.sum()) for group in validation_groups
            ),
            "train_trajectory_regress": train_regress_count,
            "validation_trajectory_regress": validation_regress_count,
            "train_trajectory_recovery": train_recovery_count,
            "validation_trajectory_recovery": validation_recovery_count,
            "regression_persistence_steps": args.regression_persistence_steps,
            "recovery_supervised": recovery_supervised,
            "recovery_support_status": (
                "supported" if recovery_supervised else "unsupported_binary_outcome_only"
            ),
            "contract": contract,
        },
    )
    member_paths = [
        train_member(
            seed=seed,
            pretrained=pretrained,
            config=config,
            train_groups=train_groups,
            validation_groups=validation_groups,
            object_mean=object_mean,
            object_std=object_std,
            output=args.output,
            device=device,
            args=args,
            contract=contract,
        )
        for seed in args.seeds
    ]
    models = []
    member_reserved_row_proofs: list[dict[str, Any]] = []
    for path in member_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_reserved_rows = validate_reserved_target_rows(checkpoint, config)
        saved_proof = validate_reserved_rows_source_only_proof(
            checkpoint, checkpoint_reserved_rows
        )
        if saved_proof is not None:
            member_reserved_row_proofs.append(saved_proof)
        model = ActionConditionedEventWorldModel(config).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        assert_reserved_target_rows_bit_exact(model, checkpoint_reserved_rows)
        model.eval()
        models.append(model)
    event_values = torch.linspace(0.0, 1.0, config.num_events, device=device)
    duration_scale = float(
        np.mean([torch.load(path, map_location="cpu", weights_only=False)["duration_scale"] for path in member_paths])
    )
    # Collect raw validation logits once for exact ensemble temperature scaling
    # without touching the sealed test groups.
    member_logits = []
    labels = np.concatenate([group.success for group in validation_groups])
    for model in models:
        logits = []
        for group in validation_groups:
            batch = move_batch(
                collate_groups(
                    [group], object_mean, object_std, include_auxiliary=False
                ),
                device,
            )
            assert_reserved_ids_absent(batch, reserved_target_rows)
            with torch.no_grad():
                output = forward_model(model, batch)
                logits.append(
                    counterfactual_success_logit(model, output, batch)
                    .cpu()
                    .numpy()
                )
        member_logits.append(np.concatenate(logits))
    calibration = fit_success_temperature(np.stack(member_logits), labels)
    scoring_candidates = predefined_scoring_grid(args.distance_weight)
    if config.action_rank_success_only:
        scoring_candidates = [
            candidate
            for candidate in scoring_candidates
            if candidate["candidate_id"] == "success_only"
        ]
    scoring_rows = [
        (
            candidate,
            ensemble_group_predictions(
                models,
                validation_groups,
                device,
                object_mean,
                object_std,
                event_values,
                duration_scale,
                temperature=float(calibration["temperature"]),
                distance_weight=float(candidate["candidate_distance_weight"]),
                event_weight=float(candidate["event_weight"]),
                duration_weight=float(candidate["duration_weight"]),
            ),
        )
        for candidate in scoring_candidates
    ]
    selected_scoring, rows, scoring_selection = select_validation_scoring(
        scoring_rows,
        minimum_proposals=args.guard_min_groups,
        minimum_coverage=args.guard_min_coverage,
        minimum_lcb=args.guard_min_lcb,
    )
    guard = tune_guard(
        rows,
        min_guarded_groups=args.guard_min_groups,
        min_coverage=args.guard_min_coverage,
        minimum_lcb=args.guard_min_lcb,
        max_harmful_rate=args.guard_max_harmful_rate,
    )
    scoring = {
        "candidate_id": selected_scoring["candidate_id"],
        "event_values": event_values.cpu().tolist(),
        "event_weight": selected_scoring["event_weight"],
        "duration_weight": selected_scoring["duration_weight"],
        "candidate_distance_weight": selected_scoring[
            "candidate_distance_weight"
        ],
        "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
    }
    ensemble_contract = {
        **contract,
        "scoring_selection_contract": {
            "grid_version": SCORING_GRID_VERSION,
            "grid_candidate_ids": [
                candidate["candidate_id"] for candidate in scoring_candidates
            ],
            "selection_rule": scoring_selection["selection_rule"],
            "selection_data": "validation_only_no_sealed_test",
            "guard_grid_version": GUARD_GRID_VERSION,
        },
    }
    ensemble_checkpoint = args.output / "counterfactual_ensemble.pt"
    ensemble_normalization: dict[str, Any] = {
        "object_delta_mean": object_mean,
        "object_delta_std": object_std,
    }
    if source_action_normalization is not None:
        ensemble_normalization.update(
            {
                "action_mean": np.asarray(
                    source_action_normalization["action_mean"], dtype=np.float32
                ),
                "action_std": np.asarray(
                    source_action_normalization["action_std"], dtype=np.float32
                ),
                "action_valid_sample_count": int(
                    source_action_normalization["valid_action_sample_count"]
                ),
                "action_statistics_sha256": source_action_normalization[
                    "statistics_sha256"
                ],
            }
        )
    ensemble_payload: dict[str, Any] = {
        "format": "etsf_counterfactual_ensemble_v1",
        "models": [model.state_dict() for model in models],
        "member_seeds": list(args.seeds),
        "config": dataclasses.asdict(config),
        "contract": ensemble_contract,
        "predicate_contract": predicate_contract,
        "candidate_contract": candidate_contract,
        "normalization": ensemble_normalization,
        "duration_scale": duration_scale,
        "success_calibration": calibration,
        "guard": guard,
        "scoring": scoring,
        "scoring_selection": scoring_selection,
    }
    if member_reserved_row_proofs:
        if len(member_reserved_row_proofs) != len(member_paths):
            raise RuntimeError("ensemble members disagree on reserved-row proof presence")
        ensemble_payload["member_reserved_target_rows_source_only_proofs"] = (
            member_reserved_row_proofs
        )
    for model in models:
        assert_reserved_target_rows_bit_exact(model, reserved_target_rows)
        assert_source_action_normalization_installed(
            model, source_action_normalization
        )
    atomic_torch_save(
        ensemble_checkpoint,
        ensemble_payload,
    )
    manifest = {
        "format": "etsf_counterfactual_ensemble_v1",
        "ensemble_checkpoint": {
            "path": str(ensemble_checkpoint.resolve()),
            "sha256": sha256(ensemble_checkpoint),
        },
        "members": [
            {"path": str(path.resolve()), "sha256": sha256(path), "seed": seed}
            for path, seed in zip(member_paths, args.seeds)
        ],
        "config": dataclasses.asdict(config),
        "contract": ensemble_contract,
        "predicate_contract": predicate_contract,
        "candidate_contract": candidate_contract,
        "normalization": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in ensemble_normalization.items()
        },
        "duration_scale": duration_scale,
        "success_calibration": calibration,
        "guard": guard,
        "scoring": scoring,
        "scoring_selection": scoring_selection,
        "test_policy": "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened",
    }
    if member_reserved_row_proofs:
        manifest["member_reserved_target_rows_source_only_proofs"] = (
            member_reserved_row_proofs
        )
    atomic_json(args.output / "ensemble_manifest.json", manifest)
    print("COUNTERFACTUAL_ENSEMBLE_COMPLETE=" + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
