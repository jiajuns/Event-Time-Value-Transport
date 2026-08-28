#!/usr/bin/env python3
"""Deterministically reserve one target policy/body row in an ETSF checkpoint.

The operation is CPU-only and data-blind.  Every existing tensor remains
bit-exact; the selected embedding matrix receives one additional row initialized
as the arithmetic mean of its existing rows.  The output records a content-
addressed parent lineage but is deliberately **not** ready for protocol freeze:
the same frozen source split must retrain the expanded model first, without ever
placing the reserved target id in a source batch.

This is vocabulary preparation, not target adaptation.  It reads no target
rollout or label and does not authorize reranking by itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


FORMAT = "etsf_transfer_source_core_expansion_v1"
RETRAIN_FORMAT = "etsf_reserved_source_core_retraining_v1"
AXIS_CONFIG = {
    "policy": {
        "embedding": "action_encoder.policy_embedding.weight",
        "count": "num_policies",
        "mapping": "policy_to_id",
    },
    "embodiment": {
        "embedding": "action_encoder.body_embedding.weight",
        "count": "num_bodies",
        "mapping": "body_to_id",
    },
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _safe_torch_mapping(path: Path) -> Any:
    """Load tensor/JSON/NumPy-array checkpoints without enabling arbitrary pickle."""

    numpy_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def _load(path: Path) -> dict[str, Any]:
    value = _safe_torch_mapping(path)
    if not isinstance(value, Mapping):
        raise ValueError("source checkpoint must contain a mapping")
    payload = dict(value)
    state = payload.get("model")
    config = payload.get("config")
    contract = payload.get("contract")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("source checkpoint must contain one model under 'model'")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("model state contains a non-tensor value")
    if not isinstance(config, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("source checkpoint must freeze config and contract")
    payload["model"] = dict(state)
    payload["config"] = dict(config)
    payload["contract"] = dict(contract)
    return payload


def expand_source_core(
    source: Path,
    output: Path,
    *,
    axis: str,
    target_name: str,
    source_manifest: Path,
    source_split: Path,
) -> dict[str, Any]:
    """Create a data-blind vocabulary initializer that still requires retraining."""

    source = source.expanduser().resolve()
    output = output.expanduser().absolute()
    source_manifest = source_manifest.expanduser().resolve()
    source_split = source_split.expanduser().resolve()
    for required_path in (source, source_manifest, source_split):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    if output.exists():
        raise FileExistsError(output)
    if axis not in AXIS_CONFIG:
        raise ValueError("axis must be policy or embodiment")
    if not target_name or target_name.startswith("__reserved__"):
        raise ValueError("target_name must be a non-reserved non-empty identity")
    payload = _load(source)
    original = copy.deepcopy(payload)
    state = payload["model"]
    config = payload["config"]
    contract = payload["contract"]
    spec = AXIS_CONFIG[axis]
    embedding_name = spec["embedding"]
    if embedding_name not in state:
        raise ValueError(f"source checkpoint lacks {embedding_name}")
    embedding = state[embedding_name]
    if embedding.ndim != 2 or not embedding.is_floating_point():
        raise ValueError("policy/body embedding must be a floating [N,D] tensor")
    count_name = spec["count"]
    mapping_name = spec["mapping"]
    if int(config.get(count_name, -1)) != embedding.shape[0]:
        raise ValueError("config vocabulary size does not match embedding rows")
    mapping = contract.get(mapping_name)
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"checkpoint contract lacks {mapping_name}")
    normalized_mapping = {str(key): int(value) for key, value in mapping.items()}
    if target_name in normalized_mapping:
        raise ValueError("target identity is already registered")
    ids = list(normalized_mapping.values())
    if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= embedding.shape[0]:
        raise ValueError("source registry ids are invalid or duplicated")
    # Every existing row contributes equally.  With a single source identity
    # this exactly clones that row; no target observation is needed.
    initialized_row = embedding.to(torch.float64).mean(dim=0).to(embedding.dtype)
    expanded = torch.cat([embedding, initialized_row[None]], dim=0)
    target_row = int(embedding.shape[0])
    reservation_name = f"__reserved__{target_name}"
    state[embedding_name] = expanded
    config[count_name] = target_row + 1
    normalized_mapping[reservation_name] = target_row
    contract[mapping_name] = normalized_mapping
    lineage = {
        "format": FORMAT,
        "axis": axis,
        "target_name": target_name,
        "reservation_name": reservation_name,
        "target_row": target_row,
        "embedding_parameter": embedding_name,
        "initializer": "mean_existing_embedding_rows_float64_v1",
        "parent_path": str(source),
        "parent_file_sha256": file_sha256(source),
        "parent_state_dict_sha256": state_sha256(original["model"]),
        "parent_config_sha256": json_sha256(original["config"]),
        "parent_contract_sha256": json_sha256(original["contract"]),
        # Freeze the exact source-only retraining inputs before a target id is
        # introduced.  A later proof naming any other manifest/split is rejected.
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_split_path": str(source_split),
        "source_split_sha256": file_sha256(source_split),
        "target_data_read": False,
        "target_labels_read": False,
        "shared_core_training_performed": False,
    }
    payload["transfer_source_core_expansion"] = lineage
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    audit = verify_expansion(source, output)
    return audit


def verify_expansion(source: Path, expanded_path: Path) -> dict[str, Any]:
    """Recheck lineage plus bit-exact preservation of the parent core."""

    source = source.expanduser().resolve()
    expanded_path = expanded_path.expanduser().resolve()
    before = _load(source)
    after = _load(expanded_path)
    lineage = after.get("transfer_source_core_expansion")
    if not isinstance(lineage, Mapping) or lineage.get("format") != FORMAT:
        raise ValueError("expanded checkpoint lacks transfer lineage")
    if (
        lineage.get("parent_file_sha256") != file_sha256(source)
        or lineage.get("parent_state_dict_sha256") != state_sha256(before["model"])
        or lineage.get("parent_config_sha256") != json_sha256(before["config"])
        or lineage.get("parent_contract_sha256") != json_sha256(before["contract"])
        or lineage.get("target_data_read") is not False
        or lineage.get("target_labels_read") is not False
        or lineage.get("shared_core_training_performed") is not False
    ):
        raise ValueError("expanded checkpoint lineage does not match its parent")
    for path_key, sha_key in (
        ("source_manifest_path", "source_manifest_sha256"),
        ("source_split_path", "source_split_sha256"),
    ):
        frozen_source_path = Path(str(lineage.get(path_key, ""))).expanduser()
        if (
            not frozen_source_path.is_file()
            or file_sha256(frozen_source_path) != lineage.get(sha_key)
        ):
            raise ValueError("frozen source retraining input is missing or changed")
    axis = str(lineage["axis"])
    if axis not in AXIS_CONFIG:
        raise ValueError("expanded checkpoint has an invalid transfer axis")
    spec = AXIS_CONFIG[axis]
    embedding_name = spec["embedding"]
    target_row = int(lineage["target_row"])
    before_state = before["model"]
    after_state = after["model"]
    if set(before_state) != set(after_state):
        raise ValueError("vocabulary expansion changed model state keys")
    for name, before_tensor in before_state.items():
        after_tensor = after_state[name]
        if name != embedding_name:
            if not torch.equal(before_tensor, after_tensor):
                raise ValueError(f"shared core tensor changed during expansion: {name}")
            continue
        if after_tensor.shape != (before_tensor.shape[0] + 1, before_tensor.shape[1]):
            raise ValueError("embedding expansion has the wrong shape")
        if not torch.equal(before_tensor, after_tensor[:-1]):
            raise ValueError("existing embedding rows changed during expansion")
        expected = before_tensor.to(torch.float64).mean(dim=0).to(before_tensor.dtype)
        if target_row != before_tensor.shape[0] or not torch.equal(expected, after_tensor[-1]):
            raise ValueError("reserved embedding row does not use the frozen initializer")
    expected_config = dict(before["config"])
    expected_config[spec["count"]] = int(expected_config[spec["count"]]) + 1
    if after["config"] != expected_config:
        raise ValueError("expansion changed config fields beyond the vocabulary size")
    expected_contract = dict(before["contract"])
    expected_mapping = {
        str(key): int(value)
        for key, value in expected_contract[spec["mapping"]].items()
    }
    expected_mapping[str(lineage["reservation_name"])] = target_row
    expected_contract[spec["mapping"]] = expected_mapping
    if after["contract"] != expected_contract:
        raise ValueError("expansion changed contract fields beyond the reserved id")
    return {
        "format": FORMAT,
        "status": "vocabulary_preparation_requires_source_retraining",
        "axis": axis,
        "target_name": lineage["target_name"],
        "reservation_name": lineage["reservation_name"],
        "target_row": target_row,
        "embedding_parameter": embedding_name,
        "initializer": lineage["initializer"],
        "parent_file_sha256": file_sha256(source),
        "expanded_file_sha256": file_sha256(expanded_path),
        "expanded_state_dict_sha256": state_sha256(after_state),
        "source_manifest_sha256": lineage["source_manifest_sha256"],
        "source_split_sha256": lineage["source_split_sha256"],
        "target_data_read": False,
        "target_labels_read": False,
        "shared_parent_tensors_preserved_bit_exact": True,
        "ready_for_protocol_freeze": False,
        "required_next_stage": (
            "source_retrain_with_reserved_row_on_identical_frozen_source_split"
        ),
    }


def verify_source_retraining(
    expanded_path: Path,
    retrained_path: Path,
    *,
    source_manifest: Path,
    source_split: Path,
) -> dict[str, Any]:
    """Verify source-only retraining before a prepared vocabulary may be frozen."""

    expanded_path = expanded_path.expanduser().resolve()
    retrained_path = retrained_path.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    source_split = source_split.expanduser().resolve()
    for path in (expanded_path, retrained_path, source_manifest, source_split):
        if not path.is_file():
            raise FileNotFoundError(path)
    expanded = _load(expanded_path)
    retrained = _load(retrained_path)
    lineage = expanded.get("transfer_source_core_expansion")
    proof = retrained.get("reserved_source_retraining")
    if not isinstance(lineage, Mapping) or lineage.get("format") != FORMAT:
        raise ValueError("input checkpoint is not a verified vocabulary preparation")
    if not isinstance(proof, Mapping) or proof.get("format") != RETRAIN_FORMAT:
        raise ValueError("retrained checkpoint lacks reserved-source provenance")
    required = {
        "format",
        "status",
        "input_expanded_checkpoint_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_split_path",
        "source_split_sha256",
        "source_training_steps",
        "source_training_groups",
        "target_data_read",
        "target_labels_read",
        "reserved_row_used_in_source_batches",
        "shared_core_retrained",
    }
    if set(proof) != required:
        raise ValueError("reserved-source retraining proof fields differ")
    if (
        proof["status"] != "complete_source_only"
        or proof["input_expanded_checkpoint_sha256"] != file_sha256(expanded_path)
        or Path(str(proof["source_manifest_path"])).resolve() != source_manifest
        or proof["source_manifest_sha256"] != file_sha256(source_manifest)
        or Path(str(proof["source_split_path"])).resolve() != source_split
        or proof["source_split_sha256"] != file_sha256(source_split)
        or Path(str(lineage.get("source_manifest_path", ""))).resolve()
        != source_manifest
        or lineage.get("source_manifest_sha256") != file_sha256(source_manifest)
        or Path(str(lineage.get("source_split_path", ""))).resolve() != source_split
        or lineage.get("source_split_sha256") != file_sha256(source_split)
        or int(proof["source_training_steps"]) <= 0
        or int(proof["source_training_groups"]) <= 0
        or proof["target_data_read"] is not False
        or proof["target_labels_read"] is not False
        or proof["reserved_row_used_in_source_batches"] is not False
        or proof["shared_core_retrained"] is not True
    ):
        raise ValueError("source-only retraining proof is invalid")
    if expanded["config"] != retrained["config"] or expanded["contract"] != retrained["contract"]:
        raise ValueError("source retraining changed the reserved config/contract")
    if set(expanded["model"]) != set(retrained["model"]):
        raise ValueError("source retraining changed model state keys")
    axis = str(lineage["axis"])
    embedding_name = AXIS_CONFIG[axis]["embedding"]
    target_row = int(lineage["target_row"])
    changed_nonreserved = False
    for name, before in expanded["model"].items():
        after = retrained["model"][name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError("source retraining changed tensor shape/dtype")
        if name == embedding_name:
            if not torch.equal(before[target_row], after[target_row]):
                raise ValueError("source-only retraining changed the reserved target row")
            keep = torch.ones(before.shape[0], dtype=torch.bool)
            keep[target_row] = False
            changed_nonreserved |= not torch.equal(before[keep], after[keep])
        else:
            changed_nonreserved |= not torch.equal(before, after)
    if not changed_nonreserved:
        raise ValueError("source retraining proof reports steps but no source parameter changed")
    return {
        "format": RETRAIN_FORMAT,
        "status": "source_core_ready_for_protocol_freeze",
        "expanded_checkpoint_sha256": file_sha256(expanded_path),
        "retrained_checkpoint_sha256": file_sha256(retrained_path),
        "retrained_state_dict_sha256": state_sha256(retrained["model"]),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_split_sha256": file_sha256(source_split),
        "source_training_steps": int(proof["source_training_steps"]),
        "source_training_groups": int(proof["source_training_groups"]),
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_target_row_unchanged": True,
        "source_parameters_changed": True,
        "ready_for_protocol_freeze": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    expand = sub.add_parser("expand")
    expand.add_argument("--source", type=Path, required=True)
    expand.add_argument("--output", type=Path, required=True)
    expand.add_argument("--axis", choices=tuple(AXIS_CONFIG), required=True)
    expand.add_argument("--target-name", required=True)
    expand.add_argument("--source-manifest", type=Path, required=True)
    expand.add_argument("--source-split", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--expanded", type=Path, required=True)
    retrain = sub.add_parser("verify-source-retraining")
    retrain.add_argument("--expanded", type=Path, required=True)
    retrain.add_argument("--retrained", type=Path, required=True)
    retrain.add_argument("--source-manifest", type=Path, required=True)
    retrain.add_argument("--source-split", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "expand":
        result = expand_source_core(
            args.source,
            args.output,
            axis=args.axis,
            target_name=args.target_name,
            source_manifest=args.source_manifest,
            source_split=args.source_split,
        )
    elif args.command == "verify":
        result = verify_expansion(args.source, args.expanded)
    else:
        result = verify_source_retraining(
            args.expanded,
            args.retrained,
            source_manifest=args.source_manifest,
            source_split=args.source_split,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
