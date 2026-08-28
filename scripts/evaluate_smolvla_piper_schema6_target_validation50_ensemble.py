#!/usr/bin/env python3
"""Independent post-training evaluator for a frozen target-validation lane.

This lane is intentionally separate from the adapter trainer.  It requires five
already frozen adapter checkpoints and the authenticated external 60/20/50
split, opens exactly the bound target-validation HDF groups once, exports one common
label NPZ plus five prediction NPZ files, and freezes a calibrator input
authority.  It cannot accept or discover evaluation400 membership.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import calibrate_smolvla_piper_adapter_ensemble as calibrator
import train_smolvla_piper_schema6_embodiment_adapter as trainer
from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


INPUT_FORMAT = "etsf_smolvla_piper_schema6_target_validation_evaluator_input_v3"
INPUT_STATUS = "authorized_after_five_frozen_adapters_for_bound_target_validation_only"
RECEIPT_FORMAT = "etsf_smolvla_piper_schema6_target_validation_evaluator_receipt_v3"
RECEIPT_STATUS = "complete_bound_target_validation_five_member_predictions"
MEMBER_COUNT = 5
SUPPORTED_TARGET_VALIDATION_GROUPS = frozenset({50, 190})
SENSITIVE_TOKENS = ("fresh", "confirmation", "evaluation")
HDF_SUFFIXES = (".h5", ".hdf", ".hdf5")
SHA_CHARS = frozenset("0123456789abcdef")
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)


class TargetValidationEvaluatorError(RuntimeError):
    """The independent calibration lane cannot prove its exact scope."""


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _sensitive(path: PurePath) -> bool:
    return any(
        token in component.casefold()
        for component in path.parts
        for token in SENSITIVE_TOKENS
    )


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise TargetValidationEvaluatorError(f"{role} path is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(text)))
    resolved = lexical.resolve(strict=False)
    if _sensitive(PurePath(lexical)) or _sensitive(PurePath(resolved)):
        raise TargetValidationEvaluatorError(f"{role} is in a forbidden namespace")
    return resolved


def existing_file(
    value: str | os.PathLike[str],
    role: str,
    *,
    suffixes: set[str] | None = None,
) -> Path:
    path = safe_path(value, role)
    if path.is_symlink():
        raise TargetValidationEvaluatorError(f"{role} must not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise TargetValidationEvaluatorError(f"{role} is not a regular file")
    if suffixes is not None and resolved.suffix.casefold() not in suffixes:
        raise TargetValidationEvaluatorError(f"{role} suffix changed")
    return resolved


def file_sha256(path: Path, *, allow_hdf: bool = False) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES and not allow_hdf:
        raise TargetValidationEvaluatorError("metadata validator cannot hash HDF bytes")
    return trainer.file_sha256(path)


def load_json(path: Path, role: str) -> dict[str, Any]:
    path = existing_file(path, role, suffixes={".json"})
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TargetValidationEvaluatorError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise TargetValidationEvaluatorError(f"{role} must contain an object")
    return value


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != trainer.canonical_sha256(unsigned):
        raise TargetValidationEvaluatorError(f"{role} logical SHA mismatch")
    return str(recorded)


def freeze_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise TargetValidationEvaluatorError("output contains a symlink")
            item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise TargetValidationEvaluatorError("output contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def validate_input_authority(
    path: Path, expected_file_sha256: str
) -> dict[str, Any]:
    authority_path = existing_file(path, "target-validation50 input authority")
    if not _is_sha(expected_file_sha256) or file_sha256(authority_path) != expected_file_sha256:
        raise TargetValidationEvaluatorError("input authority file SHA mismatch")
    value = load_json(authority_path, "target-validation50 input authority")
    logical = verify_signed(value, "authority_sha256", "target-validation50 input authority")
    expected_fields = {
        "format", "status", "trainer_compatible_manifest",
        "expected_manifest_split_receipt", "canonical_event_spec",
        "members", "member_count", "target_validation_group_count",
        "adapter_training_complete_before_authority", "target_validation_open_authorized",
        "evaluation400_membership_present", "evaluation400_open_authorized",
        "fresh_or_confirmation_open_authorized", "source_rank_numeric_contract",
        "authority_sha256",
    }
    record_fields = {"path", "file_sha256", "logical_sha256"}
    file_fields = {"path", "file_sha256"}
    member_fields = {
        "member_index", "member_seed", "adapter_checkpoint",
        "source_checkpoint", "member_receipt",
        "training_manifest_sha256", "split_sha256",
        "source_ensemble_contract_sha256", "prediction_contract",
        "source_rank_score_contract", "source_rank_score_contract_sha256",
    }
    manifest_record = value.get("trainer_compatible_manifest")
    expected_record = value.get("expected_manifest_split_receipt")
    event_record = value.get("canonical_event_spec")
    members = value.get("members")
    if (
        set(value) != expected_fields
        or value.get("format") != INPUT_FORMAT
        or value.get("status") != INPUT_STATUS
        or value.get("member_count") != MEMBER_COUNT
        or value.get("target_validation_group_count")
        not in SUPPORTED_TARGET_VALIDATION_GROUPS
        or value.get("adapter_training_complete_before_authority") is not True
        or value.get("target_validation_open_authorized") is not True
        or value.get("evaluation400_membership_present") is not False
        or value.get("evaluation400_open_authorized") is not False
        or value.get("fresh_or_confirmation_open_authorized") is not False
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(manifest_record, Mapping)
        or set(manifest_record) != record_fields
        or not isinstance(expected_record, Mapping)
        or set(expected_record) != record_fields
        or not isinstance(event_record, Mapping)
        or set(event_record) != file_fields
        or not isinstance(members, list)
        or len(members) != MEMBER_COUNT
    ):
        raise TargetValidationEvaluatorError("input authority scope changed")
    decoded = []
    seeds: set[int] = set()
    contracts: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(members):
        if not isinstance(row, Mapping) or set(row) != member_fields:
            raise TargetValidationEvaluatorError("input member fields changed")
        prediction_contract = row.get("prediction_contract")
        source_rank_contract = row.get("source_rank_score_contract")
        try:
            normalized_source_rank_contract = trainer._validate_source_rank_score_contract(
                source_rank_contract
            )
        except (trainer.AdapterContractError, TypeError, ValueError) as error:
            raise TargetValidationEvaluatorError(
                "input member Source composite rank contract changed"
            ) from error
        if (
            row.get("member_index") != index
            or type(row.get("member_seed")) is not int
            or row["member_seed"] in seeds
            or not isinstance(prediction_contract, Mapping)
            or dict(source_rank_contract) != normalized_source_rank_contract
            or normalized_source_rank_contract.get(
                "source_rank_numeric_contract"
            ) != value.get("source_rank_numeric_contract")
            or row.get("source_rank_score_contract_sha256")
            != normalized_source_rank_contract["contract_sha256"]
        ):
            raise TargetValidationEvaluatorError("input member order/seed changed")
        seeds.add(row["member_seed"])
        contract_sha = trainer.canonical_sha256(prediction_contract)
        contracts.add(
            (
                str(row["training_manifest_sha256"]),
                str(row["split_sha256"]),
                str(row["source_ensemble_contract_sha256"]),
                contract_sha,
            )
        )
        adapter_record = row.get("adapter_checkpoint")
        source_record = row.get("source_checkpoint")
        member_receipt_record = row.get("member_receipt")
        if (
            not isinstance(adapter_record, Mapping)
            or set(adapter_record) != file_fields
            or not isinstance(source_record, Mapping)
            or set(source_record) != file_fields
            or not isinstance(member_receipt_record, Mapping)
            or set(member_receipt_record) != record_fields
            or not all(
                _is_sha(candidate)
                for candidate in (
                    member_receipt_record.get("file_sha256"),
                    member_receipt_record.get("logical_sha256"),
                    row.get("training_manifest_sha256"),
                    row.get("split_sha256"),
                    row.get("source_ensemble_contract_sha256"),
                    adapter_record.get("file_sha256"),
                    source_record.get("file_sha256"),
                    row.get("source_rank_score_contract_sha256"),
                )
            )
        ):
            raise TargetValidationEvaluatorError("input member bindings are invalid")
        adapter_path = existing_file(adapter_record["path"], f"adapter member {index}")
        source_path = existing_file(source_record["path"], f"source member {index}")
        member_receipt_path = existing_file(
            member_receipt_record["path"], f"adapter member {index} receipt"
        )
        member_receipt = load_json(
            member_receipt_path, f"adapter member {index} receipt"
        )
        member_receipt_logical = verify_signed(
            member_receipt,
            "receipt_sha256",
            f"adapter member {index} receipt",
        )
        if (
            normalized_source_rank_contract.get(
                "source_checkpoint_file_sha256"
            ) != source_record["file_sha256"]
            or
            file_sha256(adapter_path) != adapter_record["file_sha256"]
            or file_sha256(source_path) != source_record["file_sha256"]
            or file_sha256(member_receipt_path)
            != member_receipt_record["file_sha256"]
            or member_receipt_logical
            != member_receipt_record["logical_sha256"]
        ):
            raise TargetValidationEvaluatorError("input member checkpoint SHA changed")
        decoded.append(
            {
                **dict(row),
                "adapter_checkpoint": {**adapter_record, "path": str(adapter_path)},
                "source_checkpoint": {**source_record, "path": str(source_path)},
                "prediction_contract_sha256": contract_sha,
            }
        )
    if len(contracts) != 1:
        raise TargetValidationEvaluatorError("five members do not share one contract")
    bound_files = []
    for record, role in (
        (manifest_record, "trainer manifest"),
        (expected_record, "manifest/split receipt"),
        (event_record, "canonical event spec"),
    ):
        bound = existing_file(record["path"], role)
        if file_sha256(bound) != record["file_sha256"]:
            raise TargetValidationEvaluatorError(f"{role} file SHA changed")
        bound_files.append(bound)
    return {
        "path": str(authority_path),
        "file_sha256": expected_file_sha256,
        "logical_sha256": logical,
        "manifest_path": str(bound_files[0]),
        "manifest_logical_sha256": manifest_record["logical_sha256"],
        "expected_receipt_path": str(bound_files[1]),
        "expected_receipt_file_sha256": expected_record["file_sha256"],
        "event_spec_path": str(bound_files[2]),
        "members": decoded,
        "shared_contract": next(iter(contracts)),
        "target_validation_group_count": int(
            value["target_validation_group_count"]
        ),
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
    }


def _sample_identity(
    rows: Sequence[Mapping[str, Any]], *, expected_groups: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    occurrence: dict[str, int] = {}
    sample_ids = []
    group_ids = []
    for row in rows:
        group = str(row["logical_group_id"])
        ordinal = occurrence.get(group, 0)
        occurrence[group] = ordinal + 1
        sample_ids.append(
            "target-validation-"
            + trainer.canonical_sha256(
                {"logical_group_id": group, "row_ordinal": ordinal}
            )
        )
        group_ids.append(group)
    if (
        expected_groups not in SUPPORTED_TARGET_VALIDATION_GROUPS
        or len(occurrence) != expected_groups
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise TargetValidationEvaluatorError("target-validation sample identity changed")
    return np.asarray(sample_ids), np.asarray(group_ids)


def _labels(
    rows: Sequence[Mapping[str, Any]], sample_ids: np.ndarray, group_ids: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "sample_id": sample_ids,
        "group_id": group_ids,
        "group_row_ordinal": np.asarray(
            [row["group_row_ordinal"] for row in rows], dtype=np.int64
        ),
        "current_event": np.asarray(
            [row["current_event_id"] for row in rows], dtype=np.int64
        ),
        "post_event": np.asarray([row["post_event_id"] for row in rows], dtype=np.int64),
        "next_event": np.asarray([row["next_event_id"] for row in rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in rows], dtype=np.int64),
        "regress": np.asarray([row["regress"] for row in rows], dtype=bool),
        "recovery": np.asarray([row["recovery"] for row in rows], dtype=np.int64),
        "recovery_observed": np.asarray(
            [row["recovery_observed"] for row in rows], dtype=bool
        ),
        "duration": np.asarray([row["duration"] for row in rows], dtype=np.float64),
        "duration_observed": np.asarray([row["duration_observed"] for row in rows], dtype=bool),
        "object_target": np.stack(
            [np.asarray(row["object_delta"], dtype=np.float64) for row in rows]
        ),
        "object_observed": np.asarray(
            [bool(np.asarray(row["object_mask"], dtype=bool).all()) for row in rows],
            dtype=bool,
        ),
        "root_candidate": np.asarray(
            [row["root_candidate"] for row in rows], dtype=bool
        ),
        "candidate_index": np.asarray(
            [row["candidate_index"] for row in rows], dtype=np.int64
        ),
        "is_baseline": np.asarray(
            [row["is_baseline"] for row in rows], dtype=bool
        ),
        "candidate_final_success": np.asarray(
            [row["candidate_final_success"] for row in rows], dtype=np.int64
        ),
    }


@torch.no_grad()
def _predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    paired_groups: Sequence[Mapping[str, Any]],
    member: Mapping[str, Any],
    sample_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_path = Path(member["source_checkpoint"]["path"])
    adapter_path = Path(member["adapter_checkpoint"]["path"])
    source = trainer._load_torch(source_path, "source member checkpoint")
    source_audit = trainer.validate_source_checkpoint(source)
    payload = trainer._load_torch(adapter_path, "adapter member checkpoint")
    if (
        payload.get("format") != trainer.FORMAT
        or payload.get("source_checkpoint_sha256")
        != member["source_checkpoint"]["file_sha256"]
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("adapter_config"), Mapping)
        or not isinstance(payload.get("conditional_recovery_adapter"), Mapping)
        or not isinstance(payload.get("conditional_recovery_contract"), Mapping)
        or not isinstance(payload.get("source_rank_score_contract"), Mapping)
    ):
        raise TargetValidationEvaluatorError("adapter checkpoint lineage changed")
    config = EventWorldModelConfig.from_dict(source_audit["config"])
    core = ActionConditionedEventWorldModel(config)
    core.load_state_dict(source["model"], strict=True)
    adapter_config = payload["adapter_config"]
    model = trainer.SmolVLAPiperAdapter(
        core,
        state_rank=int(adapter_config["state_rank"]),
        action_rank=int(adapter_config["action_rank"]),
        source_rank_contract=payload["source_rank_score_contract"],
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.enforce_and_verify_frozen_core()
    model.eval()
    recovery_contract = payload["conditional_recovery_contract"]
    expected_recovery_trained = member["prediction_contract"].get(
        "recovery_head_trained"
    )
    if (
        recovery_contract.get("semantics")
        != "p(recovery_given_operational_regress)"
        or recovery_contract.get("shared_transition_stop_gradient") is not True
        or recovery_contract.get("enters_primary_utility_or_uncertainty_before_calibration")
        is not False
        or type(recovery_contract.get("trained")) is not bool
        or expected_recovery_trained is not recovery_contract.get("trained")
    ):
        raise TargetValidationEvaluatorError(
            "conditional recovery checkpoint contract changed"
        )
    recovery_adapter = trainer.DetachedConditionalRecoveryAdapter(
        config.semantic_dim
    ).to(device)
    recovery_adapter.load_state_dict(
        payload["conditional_recovery_adapter"], strict=True
    )
    recovery_adapter.eval()
    object_mean, object_std = trainer.object_normalization(
        source, config.object_delta_dim
    )
    object_mean = object_mean.to(device)
    object_std = object_std.to(device)
    loader = DataLoader(
        trainer.RowDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=trainer.collate,
    )
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "post_event_logits", "next_event_logits", "success_logit",
            "recovery_logit",
            "duration_log_mean", "duration_log_scale", "object_mean",
            "object_log_scale",
        )
    }
    for raw in loader:
        prediction = model(trainer.move(raw, device))
        values = {
            "post_event_logits": prediction["next_event_logits"],
            "next_event_logits": prediction["next_reached_event_logits"],
            "success_logit": prediction["success_logit"],
            "recovery_logit": recovery_adapter(prediction["transition"]),
            "duration_log_mean": prediction["duration_selected_log_mean"],
            "duration_log_scale": prediction["duration_selected_log_scale"],
            "object_mean": prediction["object_delta_mean"] * object_std + object_mean,
            "object_log_scale": prediction["object_delta_log_scale"] + object_std.log(),
        }
        for name, value in values.items():
            collected[name].append(value.detach().cpu().numpy())
    source_rank_contract = trainer._validate_source_rank_score_contract(
        payload["source_rank_score_contract"]
    )
    if (
        source_rank_contract != member["source_rank_score_contract"]
        or source_rank_contract.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or source_rank_contract["contract_sha256"]
        != member["source_rank_score_contract_sha256"]
    ):
        raise TargetValidationEvaluatorError(
            "checkpoint Source rank contract differs from formal authority"
        )
    root_lookup = {
        (str(row["logical_group_id"]), int(row["candidate_index"])): index
        for index, row in enumerate(rows)
        if bool(row["root_candidate"])
    }
    if len(root_lookup) != sum(bool(row["root_candidate"]) for row in rows):
        raise TargetValidationEvaluatorError("formal root lookup is not unique")
    root_scores = {
        name: np.zeros(len(rows), dtype=np.float32)
        for name in (
            "source_contract_base_rank_score",
            "source_action_rank_residual",
            "source_contract_rank_score",
        )
    }
    seen: set[tuple[str, int]] = set()
    ranking_loader = DataLoader(
        trainer.PairedGroupDataset(paired_groups),
        batch_size=max(1, batch_size // 4),
        shuffle=False,
        collate_fn=trainer.collate_ranking_groups,
    )
    for raw in ranking_loader:
        prediction = model.predict_grouped_candidates(trainer.move(raw, device))
        logical_ids = raw["ranking_logical_group_id"]
        candidate_indices = raw["ranking_candidate_index"].detach().cpu().tolist()
        values = {
            "source_contract_base_rank_score": prediction[
                "source_contract_base_rank_score"
            ].float().detach().cpu().numpy(),
            "source_action_rank_residual": prediction[
                "action_rank_residual"
            ].float().detach().cpu().numpy(),
            "source_contract_rank_score": prediction[
                "source_contract_rank_score"
            ].float().detach().cpu().numpy(),
        }
        for position, (logical, candidate_index) in enumerate(
            zip(logical_ids, candidate_indices, strict=True)
        ):
            key = (str(logical), int(candidate_index))
            if key not in root_lookup or key in seen:
                raise TargetValidationEvaluatorError(
                    "grouped Source rank prediction identity changed"
                )
            seen.add(key)
            destination = root_lookup[key]
            for name, array in values.items():
                root_scores[name][destination] = np.float32(array[position])
    if seen != set(root_lookup):
        raise TargetValidationEvaluatorError(
            "grouped Source rank prediction coverage is incomplete"
        )
    arrays = {
        "sample_id": sample_ids,
        **{name: np.concatenate(parts, axis=0) for name, parts in collected.items()},
        **root_scores,
    }
    return arrays, source_rank_contract


def run(
    authority_path: Path,
    expected_authority_file_sha256: str,
    output_root: Path,
    *,
    device_name: str = "cuda:0",
    batch_size: int = 64,
) -> dict[str, Any]:
    audit = validate_input_authority(authority_path, expected_authority_file_sha256)
    output = safe_path(output_root, "target-validation50 output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.resolve(strict=True)
    manifest, descriptors = trainer.scan_manifest(Path(audit["manifest_path"]))
    if manifest.get("manifest_sha256") != audit["manifest_logical_sha256"]:
        raise TargetValidationEvaluatorError("trainer manifest logical SHA changed")
    split, split_audit = trainer.validate_external_split_authority(
        expected_receipt_path=Path(audit["expected_receipt_path"]),
        expected_receipt_file_sha256=audit["expected_receipt_file_sha256"],
        manifest_path=Path(audit["manifest_path"]),
        manifest=manifest,
        descriptors=descriptors,
    )
    target_groups = audit["target_validation_group_count"]
    target_ids = split.get("test")
    if (
        not isinstance(target_ids, list)
        or len(target_ids) != target_groups
        or split.get("evaluation_groups_included") != 0
        or split_audit.get("sealed_test_group_count") != target_groups
        or split_audit.get("sealed_test_hdf5_files_opened") != 0
    ):
        raise TargetValidationEvaluatorError("target-validation split changed")
    by_id = {descriptor.logical_group_id: descriptor for descriptor in descriptors}
    if set(target_ids) - set(by_id):
        raise TargetValidationEvaluatorError("target-validation50 identity missing")
    event_spec = load_json(Path(audit["event_spec_path"]), "canonical event spec")
    calibration_by_task = event_spec.get("calibration")
    first_source = trainer._load_torch(
        Path(audit["members"][0]["source_checkpoint"]["path"]),
        "source member zero",
    )
    object_names = trainer.validate_source_checkpoint(first_source)["object_names"]
    object_dim = EventWorldModelConfig.from_dict(
        trainer.validate_source_checkpoint(first_source)["config"]
    ).object_delta_dim
    rows: list[dict[str, Any]] = []
    paired_groups: list[Mapping[str, Any]] = []
    for logical in target_ids:
        descriptor = by_id[logical]
        if (
            not isinstance(calibration_by_task, Mapping)
            or descriptor.task not in calibration_by_task
        ):
            raise TargetValidationEvaluatorError("event spec task calibration missing")
        group_rows, paired = trainer._read_group(
            descriptor,
            object_dim,
            object_names=object_names,
            canonical_calibration=calibration_by_task[descriptor.task],
            include_canonical_state=False,
        )
        root_metadata = {
            id(candidate["root_row"]): candidate
            for candidate in paired["candidates"]
        }
        if (
            paired.get("logical_group_id") != logical
            or len(root_metadata) != len(paired["candidates"])
        ):
            raise TargetValidationEvaluatorError(
                "formal root-candidate identity changed"
            )
        for ordinal, row in enumerate(group_rows):
            candidate = root_metadata.get(id(row))
            row["group_row_ordinal"] = ordinal
            row["root_candidate"] = candidate is not None
            row["candidate_index"] = (
                int(candidate["original_candidate_index"])
                if candidate is not None else -1
            )
            row["is_baseline"] = (
                bool(candidate["is_baseline"])
                if candidate is not None else False
            )
            row["candidate_final_success"] = (
                int(candidate["final_success"])
                if candidate is not None else -1
            )
        rows.extend(group_rows)
        paired_groups.append(paired)
    sample_ids, group_ids = _sample_identity(
        rows, expected_groups=target_groups
    )
    labels = _labels(rows, sample_ids, group_ids)
    output.mkdir(mode=0o755)
    labels_path = output / f"target_validation{target_groups}_labels.npz"
    trainer.atomic_npz_new(labels_path, labels)
    predictions = []
    device = torch.device(device_name)
    for member in audit["members"]:
        arrays, source_rank_contract = _predict(
            rows=rows,
            paired_groups=paired_groups,
            member=member,
            sample_ids=sample_ids,
            device=device,
            batch_size=batch_size,
        )
        prediction_path = output / (
            f"target_validation{target_groups}_member_{member['member_index']}.npz"
        )
        trainer.atomic_npz_new(prediction_path, arrays)
        predictions.append(
            {
                "member_index": member["member_index"],
                "member_seed": member["member_seed"],
                "checkpoint_path": member["adapter_checkpoint"]["path"],
                "checkpoint_file_sha256": member["adapter_checkpoint"]["file_sha256"],
                "validation_predictions_path": str(prediction_path),
                "validation_predictions_file_sha256": file_sha256(prediction_path),
                "source_rank_score_contract": source_rank_contract,
                "source_rank_score_contract_sha256": source_rank_contract[
                    "contract_sha256"
                ],
            }
        )
    shared_tuple = audit["shared_contract"]
    prediction_contract = dict(audit["members"][0]["prediction_contract"])
    shared = {
        "training_manifest_sha256": shared_tuple[0],
        "split_sha256": shared_tuple[1],
        "source_ensemble_contract_sha256": shared_tuple[2],
        "prediction_contract_sha256": shared_tuple[3],
    }
    calibrator_authority: dict[str, Any] = {
        "format": calibrator.INPUT_FORMAT,
        "status": calibrator.INPUT_STATUS,
        "lane": "validation_only",
        "member_count": MEMBER_COUNT,
        "shared_contract": shared,
        "prediction_contract": prediction_contract,
        "source_rank_numeric_contract": audit[
            "source_rank_numeric_contract"
        ],
        "validation_identity_set_sha256": trainer.canonical_sha256(
            sample_ids.astype(str).tolist()
        ),
        "labels_path": str(labels_path),
        "labels_file_sha256": file_sha256(labels_path),
        "members": [{**row, **shared} for row in predictions],
        "test_artifacts_read": False,
        "fresh_artifacts_read": False,
        "confirmation_artifacts_read": False,
    }
    calibrator_authority["input_authority_sha256"] = trainer.canonical_sha256(
        calibrator_authority
    )
    calibrator_authority_path = output / "calibration_input_authority.json"
    trainer.atomic_json_new(calibrator_authority_path, calibrator_authority)
    receipt: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "status": RECEIPT_STATUS,
        "input_authority_path": audit["path"],
        "input_authority_file_sha256": audit["file_sha256"],
        "input_authority_sha256": audit["logical_sha256"],
        "target_validation_groups": target_groups,
        "target_validation_samples": len(rows),
        "target_validation_hdf5_files_opened": target_groups,
        "target_validation_opened_after_five_adapters_frozen": True,
        "calibration_input_authority_path": str(calibrator_authority_path),
        "calibration_input_authority_file_sha256": file_sha256(
            calibrator_authority_path
        ),
        "calibration_input_authority_sha256": calibrator_authority[
            "input_authority_sha256"
        ],
        "source_rank_score_contract_sha256s": [
            row["source_rank_score_contract_sha256"] for row in predictions
        ],
        "source_rank_numeric_contract": audit[
            "source_rank_numeric_contract"
        ],
        "evaluation400_membership_present": False,
        "evaluation400_hdf5_or_label_files_opened": 0,
        "fresh_or_confirmation_files_opened": 0,
        "performance_or_transfer_claim_authorized": False,
    }
    receipt["receipt_sha256"] = trainer.canonical_sha256(receipt)
    trainer.atomic_json_new(output / "final_receipt.json", receipt)
    with (output / "run.exit").open("x", encoding="ascii") as handle:
        handle.write("0\n")
        handle.flush()
        os.fsync(handle.fileno())
    freeze_tree(output)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-authority", type=Path, required=True)
    parser.add_argument("--input-authority-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise TargetValidationEvaluatorError("batch size must be positive")
    receipt = run(
        args.input_authority,
        args.input_authority_file_sha256,
        args.output_root,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INPUT_FORMAT", "INPUT_STATUS", "RECEIPT_FORMAT", "RECEIPT_STATUS",
    "TargetValidationEvaluatorError", "run", "validate_input_authority",
]
