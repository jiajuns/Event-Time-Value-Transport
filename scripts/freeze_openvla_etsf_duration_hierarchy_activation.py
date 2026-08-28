#!/usr/bin/env python3
"""Freeze an authenticated all-D250 duration-v2 prediction activation.

Inputs are limited to a signed R3 materialization bundle and a signed, passed
R5 duration OOF JSON/NPZ pair.  Source HDF5, Fresh data, model training, reward,
and candidate selection are deliberately outside this program.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import evaluate_openvla_etsf_duration_hierarchy_oof as r5_evaluator
from openvla_etsf_duration_hierarchy import (
    MINIMUM_APPLIED_SOURCE_SUPPORT,
    canonical_sha256,
    validate_duration_hierarchy_contract,
)
from openvla_etsf_duration_hierarchy_adapter import (
    ACTIVATION_FORMAT,
    DEVELOPMENT_GROUP_COUNT,
    EMPIRICAL_REGISTRY_FORMAT,
    RESIDUAL_MULTIPLIER,
    fit_final_duration_hierarchy,
    sha256_path,
    validate_duration_activation,
    validate_empirical_registry_contract,
)


R5_EXPECTED_IMPLEMENTATION_FILES = {
    "evaluate_openvla_etsf_duration_hierarchy_oof.py",
    "openvla_etsf_duration_hierarchy.py",
    "openvla_etsf_v8_structured_adapters.py",
    "train_openvla_etsf_v8_structured_adapters.py",
}
ACTIVATION_IMPLEMENTATION_FILES = R5_EXPECTED_IMPLEMENTATION_FILES | {
    "freeze_openvla_etsf_duration_hierarchy_activation.py",
    "openvla_etsf_duration_hierarchy_adapter.py",
}


def _reject_fresh_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    path = _reject_fresh_path(path, role=role)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must contain a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _authenticate_r3_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    manifest = _load_json(manifest_path, role="R3 materialization manifest")
    unsigned = dict(manifest)
    recorded = unsigned.pop("materialization_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("R3 materialization signature mismatch")
    groups = manifest.get("development_groups")
    folds = manifest.get("folds")
    if (
        manifest.get("format") != r5_evaluator.MATERIALIZATION_FORMAT
        or manifest.get("status") != "complete_development_only"
        or manifest.get("timing_scope")
        != "adaptive_development_only_designed_after_v7_collection_started"
        or manifest.get("prospective_claim_for_v8") is not False
        or manifest.get("fresh_confirmation_data_or_labels_read") is not False
        or manifest.get("authorization_guard_changed") is not False
        or not isinstance(groups, list)
        or len(groups) != DEVELOPMENT_GROUP_COUNT
        or groups != sorted(groups)
        or len(set(groups)) != DEVELOPMENT_GROUP_COUNT
        or r5_evaluator.logical_group_list_sha256(groups)
        != manifest.get("development_groups_sha256")
        or not isinstance(folds, list)
        or [row.get("outer_fold_id") for row in folds]
        != list(range(r5_evaluator.FOLD_COUNT))
    ):
        raise RuntimeError("R3 materialization D250/no-Fresh contract changed")
    owners: dict[str, int] = {}
    registry = set(groups)
    for fold_id, row in enumerate(folds):
        if not isinstance(row, Mapping):
            raise RuntimeError("R3 fold row must be a mapping")
        training = list(map(str, row.get("training_groups", [])))
        holdout = list(map(str, row.get("oof_holdout_groups", [])))
        if (
            r5_evaluator.logical_group_list_sha256(training)
            != row.get("training_groups_sha256")
            or r5_evaluator.logical_group_list_sha256(holdout)
            != row.get("oof_holdout_groups_sha256")
            or set(training) & set(holdout)
            or set(training) | set(holdout) != registry
        ):
            raise RuntimeError(f"R3 fold {fold_id} ownership changed")
        for group in holdout:
            if group in owners:
                raise RuntimeError("R3 D250 logical group has multiple holdout owners")
            owners[group] = fold_id
    if set(owners) != registry or len(owners) != DEVELOPMENT_GROUP_COUNT:
        raise RuntimeError("five R3 holdouts do not cover D250 exactly once")
    for key in ("base_checkpoint_sha256", "event_spec_sha256"):
        if not _is_sha256(manifest.get(key)):
            raise RuntimeError(f"R3 materialization lacks signed {key}")
    return manifest, groups


def _authenticate_r5_result(
    *,
    result_path: Path,
    rows_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result_path = _reject_fresh_path(result_path, role="R5 duration result")
    rows_path = _reject_fresh_path(rows_path, role="R5 duration row arrays")
    result = _load_json(result_path, role="R5 duration result")
    unsigned = dict(result)
    recorded = unsigned.pop("result_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("R5 duration result signature mismatch")
    source = result.get("source_materialization")
    row_contract = result.get("row_arrays")
    if (
        result.get("format") != r5_evaluator.FORMAT
        or result.get("status") != "passed"
        or result.get("passed") is not True
        or result.get("evidence_scope")
        != "R3_adaptive_development_OOF_only_not_prospective"
        or result.get("minimum_applied_source_support_gate") is not True
        or int(result.get("minimum_applied_source_support", -1))
        < MINIMUM_APPLIED_SOURCE_SUPPORT
        or result.get("fresh50_inputs_accepted") is not False
        or result.get("fresh50_labels_read") is not False
        or result.get("fresh50_confirmation_authorized") is not False
        or result.get("selector_authorized") is not False
        or result.get("prospective_claim_allowed") is not False
        or not isinstance(source, Mapping)
        or not isinstance(row_contract, Mapping)
        or source.get("path") != str(manifest_path.resolve())
        or source.get("file_sha256") != sha256_path(manifest_path)
        or source.get("materialization_sha256")
        != manifest.get("materialization_sha256")
        or source.get("ten_artifacts_authenticated") is not True
        or source.get("source_hdf5_read") is not False
        or Path(str(row_contract.get("path", ""))).resolve() != rows_path.resolve()
        or row_contract.get("file_sha256") != sha256_path(rows_path)
        or row_contract.get("alignment")
        != "owner_fold_id_logical_group_row_index"
    ):
        raise RuntimeError("R5 passed evidence or no-Fresh contract changed")
    implementation = result.get("implementation_files")
    scripts_root = Path(__file__).resolve().parent
    if not isinstance(implementation, Mapping) or set(implementation) != (
        R5_EXPECTED_IMPLEMENTATION_FILES
    ):
        raise RuntimeError("R5 implementation hash set changed")
    for filename in R5_EXPECTED_IMPLEMENTATION_FILES:
        if sha256_path(scripts_root / filename) != implementation.get(filename):
            raise RuntimeError(f"R5 implementation hash mismatch: {filename}")
    if result.get("factual_state_sha256") is None or not _is_sha256(
        result.get("factual_state_sha256")
    ):
        raise RuntimeError("R5 factual-state hash is missing")
    contracts = result.get("contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != {
        str(index) for index in range(r5_evaluator.FOLD_COUNT)
    }:
        raise RuntimeError("R5 fold hierarchy contracts are incomplete")
    for contract in contracts.values():
        validate_duration_hierarchy_contract(contract)
    with np.load(rows_path, allow_pickle=False) as loaded:
        expected_keys = set(map(str, row_contract.get("keys", [])))
        required_alignment = {
            "owner_fold_id",
            "logical_group",
            "row_index",
            "duration",
            "duration_observed",
            "dense_mask",
            "current_event_id",
            "body_id",
            "frozen_log_location",
        }
        if set(loaded.files) != expected_keys or not required_alignment <= expected_keys:
            raise RuntimeError("R5 NPZ key contract changed")
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    lengths = {len(value) for value in arrays.values() if value.ndim >= 1}
    if lengths != {int(row_contract.get("rows", -1))}:
        raise RuntimeError("R5 NPZ arrays are not row aligned")
    return result, arrays


def _require_equal(name: str, actual: np.ndarray, recorded: np.ndarray) -> None:
    if actual.dtype.kind in "f" or recorded.dtype.kind in "f":
        equal = np.array_equal(actual, recorded, equal_nan=False)
    else:
        equal = np.array_equal(actual, recorded)
    if not equal:
        raise RuntimeError(f"R5/R3 aligned row mismatch: {name}")


def _build_empirical_registry(
    metadata_rows: list[dict[str, Any]],
    *,
    development_groups: list[str],
    development_groups_sha256: str,
) -> dict[str, Any]:
    if len(metadata_rows) != DEVELOPMENT_GROUP_COUNT:
        raise RuntimeError("empirical registry requires one metadata row per D250 group")
    by_group = {str(row["logical_group_key"]): row for row in metadata_rows}
    if len(by_group) != DEVELOPMENT_GROUP_COUNT or set(by_group) != set(
        development_groups
    ):
        raise RuntimeError("empirical registry metadata does not cover D250 exactly once")
    body_groups: dict[tuple[str, int], list[str]] = {}
    policy_groups: dict[tuple[str, int], list[str]] = {}
    cell_groups: dict[tuple[str, int, str, int], list[str]] = {}
    for group, row in by_group.items():
        body_key = (str(row["body"]), int(row["body_id"]))
        policy_key = (str(row["policy"]), int(row["policy_id"]))
        cell_key = (*policy_key, *body_key)
        body_groups.setdefault(body_key, []).append(group)
        policy_groups.setdefault(policy_key, []).append(group)
        cell_groups.setdefault(cell_key, []).append(group)
    if len(body_groups) != 1 or len(policy_groups) != 1 or len(cell_groups) != 1:
        raise RuntimeError(
            "this activation protocol requires one empirical policy/body cell"
        )

    def group_binding(groups: list[str]) -> dict[str, Any]:
        groups = sorted(groups)
        return {
            "logical_group_count": len(groups),
            "logical_groups_canonical_sha256": canonical_sha256(groups),
        }

    (body, body_id), body_members = next(iter(body_groups.items()))
    (policy, policy_id), policy_members = next(iter(policy_groups.items()))
    (_, _, _, _), cell_members = next(iter(cell_groups.items()))
    registry: dict[str, Any] = {
        "format": EMPIRICAL_REGISTRY_FORMAT,
        "status": "authenticated_D250_observed_registry_one_cell_only",
        "logical_groups": DEVELOPMENT_GROUP_COUNT,
        "development_groups_sha256": development_groups_sha256,
        "one_cell_only": True,
        "cross_body_validated": False,
        "cross_policy_validated": False,
        "observed_bodies": [
            {
                "body": body,
                "body_id": body_id,
                **group_binding(body_members),
            }
        ],
        "observed_policies": [
            {
                "policy": policy,
                "policy_id": policy_id,
                **group_binding(policy_members),
            }
        ],
        "observed_policy_body_cells": [
            {
                "policy": policy,
                "policy_id": policy_id,
                "body": body,
                "body_id": body_id,
                **group_binding(cell_members),
            }
        ],
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    validate_empirical_registry_contract(registry)
    return registry


def freeze_duration_activation(
    *,
    materialization_manifest: Path,
    r5_result_json: Path,
    r5_rows_npz: Path,
) -> dict[str, Any]:
    manifest_path = _reject_fresh_path(
        materialization_manifest, role="R3 materialization manifest"
    )
    manifest, development_groups = _authenticate_r3_manifest(manifest_path)
    r5_result, r5_rows = _authenticate_r5_result(
        result_path=r5_result_json,
        rows_path=r5_rows_npz,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    trace: list[dict[str, Any]] = []
    parts: dict[str, list[np.ndarray]] = {}
    factual_states: set[str] = set()
    metadata_rows: list[dict[str, Any]] = []
    root = manifest_path.parent.resolve()
    for fold_id, fold_row in enumerate(manifest["folds"]):
        fold_contract = r5_result["contracts"][str(fold_id)]
        if fold_contract.get("outer_training_logical_groups_sha256") != (
            canonical_sha256(fold_row["training_groups"])
        ):
            raise RuntimeError(f"R5 fold {fold_id} contract owner changed")
        payload, _ = r5_evaluator._authenticate_artifact(
            manifest_root=root,
            fold_row=fold_row,
            fold_id=fold_id,
            role="holdout",
            trace=trace,
            signed_duration_contract=fold_contract,
        )
        provenance = payload.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("base_checkpoint_sha256")
            != manifest["base_checkpoint_sha256"]
            or provenance.get("event_spec_sha256")
            != manifest["event_spec_sha256"]
        ):
            raise RuntimeError(f"R3 fold {fold_id} factual/event provenance changed")
        arrays, state_sha = r5_evaluator._records_to_arrays(
            payload, fold_id=fold_id, role="holdout"
        )
        for record in payload["batches"]:
            metadata = record.get("group_metadata")
            batch = record.get("batch")
            if not isinstance(metadata, Mapping) or not isinstance(batch, Mapping):
                raise RuntimeError(
                    f"R3 fold {fold_id} group_metadata registry is missing"
                )
            group = str(record["logical_group_key"])
            if (
                metadata.get("logical_group_key") != group
                or not isinstance(metadata.get("body"), str)
                or not metadata["body"]
                or not isinstance(metadata.get("body_id"), int)
                or int(metadata["body_id"]) < 0
                or not isinstance(metadata.get("policy"), str)
                or not metadata["policy"]
                or not isinstance(metadata.get("policy_id"), int)
                or int(metadata["policy_id"]) < 0
            ):
                raise RuntimeError(f"R3 fold {fold_id} group_metadata is invalid")
            count = len(r5_evaluator._tensor_numpy(batch["duration"], name="duration"))
            body_values = r5_evaluator._tensor_numpy(
                batch.get("body_id"), name="body_id", length=count
            ).astype(np.int64)
            policy_values = r5_evaluator._tensor_numpy(
                batch.get("policy_id"), name="policy_id", length=count
            ).astype(np.int64)
            if np.any(body_values != int(metadata["body_id"])) or np.any(
                policy_values != int(metadata["policy_id"])
            ):
                raise RuntimeError(
                    f"R3 fold {fold_id} metadata/batch body-policy registry differs"
                )
            metadata_rows.append(
                {
                    "logical_group_key": group,
                    "body": str(metadata["body"]),
                    "body_id": int(metadata["body_id"]),
                    "policy": str(metadata["policy"]),
                    "policy_id": int(metadata["policy_id"]),
                }
            )
        factual_states.add(state_sha)
        count = len(arrays["duration"])
        arrays["owner_fold_id"] = np.full(count, fold_id, dtype=np.int64)
        for key, value in arrays.items():
            parts.setdefault(key, []).append(np.asarray(value))
    if factual_states != {r5_result["factual_state_sha256"]}:
        raise RuntimeError("R3/R5 frozen factual-state hash changed")
    rows = {key: np.concatenate(value) for key, value in parts.items()}
    alignment_fields = (
        "owner_fold_id",
        "logical_group",
        "row_index",
        "duration",
        "duration_observed",
        "dense_mask",
        "current_event_id",
        "body_id",
        "frozen_log_location",
    )
    for key in alignment_fields:
        if key not in r5_rows:
            raise RuntimeError(f"R5 NPZ lacks aligned field: {key}")
        _require_equal(key, rows[key], r5_rows[key])
    selected = rows["dense_mask"].astype(bool) & rows[
        "duration_observed"
    ].astype(bool)
    if int(selected.sum()) != int(r5_result.get("observed_rows", -1)):
        raise RuntimeError("R5 observed-row count differs from authenticated R3")
    final_hierarchy = fit_final_duration_hierarchy(
        duration=rows["duration"],
        duration_observed=rows["duration_observed"],
        dense_mask=rows["dense_mask"],
        current_event_id=rows["current_event_id"],
        body_id=rows["body_id"],
        logical_group=rows["logical_group"],
        development_groups=development_groups,
        materialization_groups_sha256=manifest["development_groups_sha256"],
    )
    empirical_registry = _build_empirical_registry(
        metadata_rows,
        development_groups=development_groups,
        development_groups_sha256=manifest["development_groups_sha256"],
    )
    observed_body = empirical_registry["observed_bodies"][0]
    observed_policy = empirical_registry["observed_policies"][0]
    scripts_root = Path(__file__).resolve().parent
    implementation_files = {
        filename: sha256_path(scripts_root / filename)
        for filename in sorted(ACTIVATION_IMPLEMENTATION_FILES)
    }
    activation: dict[str, Any] = {
        "format": ACTIVATION_FORMAT,
        "status": "activated_duration_prediction_only_development",
        "evidence_scope": "adaptive_development_only",
        "permissions": {
            "duration_prediction_adapter": True,
            "actor_control": False,
            "policy_modification": False,
            "reward_or_value": False,
            "candidate_ranking": False,
            "selector": False,
        },
        "interface_actor_policy_agnostic": True,
        "empirical_registry_contract": empirical_registry,
        "empirical_registry_contract_sha256": empirical_registry[
            "registry_sha256"
        ],
        "empirical_evidence_scope": {
            "policy": observed_policy["policy"],
            "policy_id": observed_policy["policy_id"],
            "body": observed_body["body"],
            "body_id": observed_body["body_id"],
            "one_cell_only": True,
            "cross_body_validated": False,
            "cross_policy_validated": False,
        },
        "transfer_claim_authorized": False,
        "duration_residual_multiplier": RESIDUAL_MULTIPLIER,
        "formula": "baseline+0.375*(frozen_duration_log_mean-baseline)",
        "final_hierarchy_contract": final_hierarchy,
        "final_hierarchy_contract_sha256": final_hierarchy["contract_sha256"],
        "development_coverage": {
            "logical_groups": DEVELOPMENT_GROUP_COUNT,
            "five_holdouts_cover_each_group_exactly_once": True,
            "development_groups_sha256": manifest[
                "development_groups_sha256"
            ],
            "materialized_rows": len(rows["duration"]),
            "dense_observed_rows": int(selected.sum()),
        },
        "evidence": {
            "factual_checkpoint_sha256": manifest["base_checkpoint_sha256"],
            "factual_state_sha256": next(iter(factual_states)),
            "event_spec_sha256": manifest["event_spec_sha256"],
            "materialization_sha256": manifest["materialization_sha256"],
            "materialization_file_sha256": sha256_path(manifest_path),
            "r5_result_sha256": r5_result["result_sha256"],
            "r5_result_file_sha256": sha256_path(r5_result_json),
            "r5_rows_file_sha256": sha256_path(r5_rows_npz),
        },
        "source_paths": {
            "materialization_manifest": str(manifest_path),
            "r5_result_json": str(Path(r5_result_json).resolve()),
            "r5_rows_npz": str(Path(r5_rows_npz).resolve()),
        },
        "implementation_files": implementation_files,
        "authentication_trace": trace,
        "source_hdf5_read": False,
        "model_training_performed": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "fresh50_confirmation_authorized": False,
        "selector_authorized": False,
        "prospective_claim_allowed": False,
    }
    activation["activation_sha256"] = canonical_sha256(activation)
    validate_duration_activation(activation)
    return activation


def write_activation(path: Path, activation: Mapping[str, Any]) -> Path:
    path = _reject_fresh_path(path, role="duration activation output")
    validate_duration_activation(activation)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(activation, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
        return path
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--r5-result-json", type=Path, required=True)
    parser.add_argument("--r5-rows-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activation = freeze_duration_activation(
        materialization_manifest=args.materialization_manifest,
        r5_result_json=args.r5_result_json,
        r5_rows_npz=args.r5_rows_npz,
    )
    output = write_activation(args.output, activation)
    print(
        json.dumps(
            {
                "status": activation["status"],
                "activation": str(output),
                "activation_file_sha256": sha256_path(output),
                "activation_sha256": activation["activation_sha256"],
                "duration_prediction_adapter": True,
                "fresh50_confirmation_authorized": False,
                "selector_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
