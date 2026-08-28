#!/usr/bin/env python3
"""Validation-only calibration for a five-member Piper adapter ensemble.

The module hashes, but never deserializes, adapter checkpoints.  It reads only
five frozen validation prediction NPZ files and one validation label NPZ file.
Any test/fresh/confirmation namespace is rejected before opening.  Thresholds,
temperature, support decisions and uncertainty scales are selected exclusively
from validation groups and are published through signed, read-only receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np

import smolvla_piper_deployment_uncertainty_v1 as deployment_uncertainty_v1


INPUT_FORMAT = "etsf_smolvla_piper_adapter_ensemble_validation_input_v2"
CALIBRATION_FORMAT = "etsf_smolvla_piper_adapter_ensemble_calibration_v2"
MANIFEST_FORMAT = "etsf_smolvla_piper_adapter_ensemble_manifest_v2"
RECEIPT_FORMAT = "etsf_smolvla_piper_adapter_ensemble_validation_receipt_v2"
HEAD_SUPPORT_FORMAT = "etsf_smolvla_piper_multitask_head_support_v2"
INPUT_STATUS = "frozen_five_member_validation_predictions_before_calibration"
RECEIPT_STATUS = "complete_validation_only_five_member_calibration"
MEMBER_COUNT = 5
FORMAL_ROOT_GROUP_COUNT = 190
ECE_BINS = 10
INTERVAL_MASS = 0.90
MINIMUM_HEAD_GROUPS_PER_SIDE = {
    "post_event": 10,
    "next_event": 10,
    "duration": 10,
    "success": 50,
    "recovery": 10,
    "object_effect": 50,
}
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_ALPHA = 0.05
CROSSFIT_FOLDS = 5
PERFORMANCE_GATE_PROTOCOL = (
    "five_fold_logical_group_crossfit_group_bootstrap_zero_gain_lcb_v1"
)
ROOT_MARGIN_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
ROOT_UNCERTAINTY_GRID = (0.25, 0.5, 0.75, 1.0)
ROOT_RECOVERY_UNCERTAINTY_POLICY = (
    deployment_uncertainty_v1.ROOT_RECOVERY_UNCERTAINTY_POLICY
)
ROOT_STRUCTURED_UNCERTAINTY_HEADS = deployment_uncertainty_v1.ROOT_INCLUDED_HEADS
MINIMUM_ROOT_CHANGED_GROUPS = 50
MINIMUM_ROOT_DISCORDANT_GROUPS = 20
MINIMUM_ROOT_CHANGED_GROUPS_PER_FOLD = 10
MINIMUM_ROOT_DISCORDANT_GROUPS_PER_FOLD = 4
MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES = 0.10
MINIMUM_RETAINED_GROUPS = 50
MINIMUM_RETAINED_COVERAGE = 0.50
MINIMUM_QUALITY_LCB = 0.60
THRESHOLD_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
SENSITIVE_TOKENS = ("fresh", "confirmation")
HDF_SUFFIXES = (".h5", ".hdf", ".hdf5")
SHA_CHARS = frozenset("0123456789abcdef")
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)
PREDICTION_KEYS = frozenset(
    {
        "sample_id",
        "post_event_logits",
        "next_event_logits",
        "success_logit",
        "source_contract_base_rank_score",
        "source_action_rank_residual",
        "source_contract_rank_score",
        "recovery_logit",
        "duration_log_mean",
        "duration_log_scale",
        "object_mean",
        "object_log_scale",
    }
)
LABEL_KEYS = frozenset(
    {
        "sample_id",
        "group_id",
        "group_row_ordinal",
        "current_event",
        "post_event",
        "next_event",
        "success",
        "regress",
        "recovery",
        "recovery_observed",
        "duration",
        "duration_observed",
        "object_target",
        "object_observed",
        "root_candidate",
        "candidate_index",
        "is_baseline",
        "candidate_final_success",
    }
)


class CalibrationError(RuntimeError):
    """A validation-only ensemble contract or numerical invariant failed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise CalibrationError(f"{role} logical SHA mismatch")
    return str(recorded)


def validate_source_rank_member_authority(
    authority: Mapping[str, Any], authority_sha256: str,
) -> list[float]:
    """Validate the exact ordered member authority used by float32 algebra."""

    members = authority.get("members") if isinstance(authority, Mapping) else None
    if (
        not isinstance(authority, Mapping)
        or set(authority) != {"source_rank_numeric_contract", "members"}
        or authority.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(members, list)
        or len(members) != MEMBER_COUNT
        or not _is_sha(authority_sha256)
        or authority_sha256 != canonical_sha256(authority)
    ):
        raise CalibrationError("Source rank member authority changed")
    expected_fields = {
        "member_index", "source_checkpoint_file_sha256",
        "source_rank_score_contract_sha256", "success_temperature",
    }
    temperatures: list[float] = []
    for index, row in enumerate(members):
        temperature = row.get("success_temperature") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_fields
            or type(row.get("member_index")) is not int
            or row["member_index"] != index
            or not _is_sha(row.get("source_checkpoint_file_sha256"))
            or not _is_sha(row.get("source_rank_score_contract_sha256"))
            or isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise CalibrationError("Source rank member authority row changed")
        temperature32 = np.float32(temperature)
        if not np.isfinite(temperature32) or temperature32 <= np.float32(0.0):
            raise CalibrationError(
                "Source rank member temperature is not usable as float32"
            )
        temperatures.append(float(temperature))
    return temperatures


def _sensitive(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if any(token in lowered for token in SENSITIVE_TOKENS):
            return True
        if lowered == "test" or lowered.startswith(("test_", "test-")):
            return True
    return False


def safe_existing(path: Path, role: str, *, allowed_suffixes: set[str] | None = None) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if _sensitive(PurePath(lexical)) or lexical.is_symlink():
        raise CalibrationError(f"{role} path is sensitive or a symlink")
    resolved = lexical.resolve(strict=True)
    if _sensitive(PurePath(resolved)) or not stat.S_ISREG(resolved.stat().st_mode):
        raise CalibrationError(f"{role} is not a safe regular file")
    suffix = resolved.suffix.casefold()
    if suffix in HDF_SUFFIXES:
        raise CalibrationError("HDF5 input is forbidden")
    if allowed_suffixes is not None and suffix not in allowed_suffixes:
        raise CalibrationError(f"{role} suffix is not allowed")
    return resolved


def safe_new_root(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if _sensitive(PurePath(lexical)):
        raise CalibrationError("output path is sensitive")
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError(lexical)
    parent = lexical.parent.resolve(strict=True)
    if _sensitive(PurePath(parent)) or not parent.is_dir():
        raise CalibrationError("output parent is invalid")
    return lexical


def load_json(path: Path, role: str) -> dict[str, Any]:
    path = safe_existing(path, role, allowed_suffixes={".json"})
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CalibrationError(f"{role} must contain an object")
    return value


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def immutable_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def freeze_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise CalibrationError("output contains a symlink")
            item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise CalibrationError("output contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def validate_input_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = verify_signed(value, "input_authority_sha256", "input authority")
    expected_root = {
        "format",
        "status",
        "lane",
        "member_count",
        "shared_contract",
        "prediction_contract",
        "source_rank_numeric_contract",
        "validation_identity_set_sha256",
        "labels_path",
        "labels_file_sha256",
        "members",
        "test_artifacts_read",
        "fresh_artifacts_read",
        "confirmation_artifacts_read",
        "input_authority_sha256",
    }
    shared_fields = {
        "training_manifest_sha256",
        "split_sha256",
        "source_ensemble_contract_sha256",
        "prediction_contract_sha256",
    }
    member_fields = {
        "member_index",
        "member_seed",
        "training_manifest_sha256",
        "split_sha256",
        "source_ensemble_contract_sha256",
        "prediction_contract_sha256",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "validation_predictions_path",
        "validation_predictions_file_sha256",
        "source_rank_score_contract",
        "source_rank_score_contract_sha256",
    }
    shared = value.get("shared_contract")
    prediction_contract = value.get("prediction_contract")
    members = value.get("members")
    if (
        set(value) != expected_root
        or value.get("format") != INPUT_FORMAT
        or value.get("status") != INPUT_STATUS
        or value.get("lane") != "validation_only"
        or value.get("member_count") != MEMBER_COUNT
        or not isinstance(shared, Mapping)
        or set(shared) != shared_fields
        or any(not _is_sha(shared.get(field)) for field in shared_fields)
        or not isinstance(prediction_contract, Mapping)
        or set(prediction_contract) != {
            "duration_target_transform",
            "next_event_observation_mask",
            "success_target",
            "recovery_target",
            "recovery_observation_mask",
            "recovery_shared_transition_stop_gradient",
            "recovery_enters_primary_before_calibration",
            "recovery_head_trained",
            "object_prediction_space",
            "object_source_normalization_sha256",
            "object_observed_policy",
        }
        or prediction_contract.get("duration_target_transform")
        != "log1p_decision_steps"
        or prediction_contract.get("next_event_observation_mask")
        != "duration_observed"
        or prediction_contract.get("success_target")
        != "eventual_final_branch_success_repeated_per_transition"
        or prediction_contract.get("recovery_target")
        != "conditional_recovery_given_operational_regress"
        or prediction_contract.get("recovery_observation_mask")
        != "recovery_observed_and_regress"
        or prediction_contract.get("recovery_shared_transition_stop_gradient")
        is not True
        or prediction_contract.get("recovery_enters_primary_before_calibration")
        is not False
        or type(prediction_contract.get("recovery_head_trained")) is not bool
        or prediction_contract.get("object_prediction_space")
        != "physical_delta_xyz_m"
        or not _is_sha(
            prediction_contract.get("object_source_normalization_sha256")
        )
        or prediction_contract.get("object_observed_policy")
        != "row_enabled_only_if_all_selected_xyz_are_valid"
        or canonical_sha256(prediction_contract)
        != shared.get("prediction_contract_sha256")
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not _is_sha(value.get("validation_identity_set_sha256"))
        or value.get("test_artifacts_read") is not False
        or value.get("fresh_artifacts_read") is not False
        or value.get("confirmation_artifacts_read") is not False
        or not isinstance(members, list)
        or len(members) != MEMBER_COUNT
    ):
        raise CalibrationError("input authority boundary is invalid")
    labels = safe_existing(
        Path(str(value["labels_path"])), "validation labels", allowed_suffixes={".npz"}
    )
    if file_sha256(labels) != value["labels_file_sha256"]:
        raise CalibrationError("validation label file SHA changed")
    decoded = []
    seeds: set[int] = set()
    checkpoint_hashes: set[str] = set()
    prediction_hashes: set[str] = set()
    for index, row in enumerate(members):
        if not isinstance(row, Mapping) or set(row) != member_fields:
            raise CalibrationError("adapter member fields changed")
        source_rank_contract = row.get("source_rank_score_contract")
        if (
            row["member_index"] != index
            or type(row["member_seed"]) is not int
            or row["member_seed"] in seeds
            or row["checkpoint_file_sha256"] in checkpoint_hashes
            or row["validation_predictions_file_sha256"] in prediction_hashes
            or any(row[field] != shared[field] for field in shared_fields)
            or not isinstance(source_rank_contract, Mapping)
            or source_rank_contract.get("format")
            != "etsf_source63_composite_candidate_rank_score_v1"
            or source_rank_contract.get("status")
            != "frozen_exact_source63_training_score_scientific_rank_only"
            or source_rank_contract.get("base_score") != "candidate_rank_score"
            or source_rank_contract.get("residual_combination")
            != "candidate_rank_score_plus_action_rank_residual"
            or source_rank_contract.get("source_action_rank_residual") is not True
            or source_rank_contract.get("source_rank_numeric_contract")
            != value.get("source_rank_numeric_contract")
            or source_rank_contract.get("source_contract_rank_score_is_success_logit")
            is not False
            or source_rank_contract.get(
                "source_contract_rank_score_is_success_probability"
            ) is not False
            or source_rank_contract.get(
                "deployment_success_probability_selector_authorized"
            ) is not False
            or row.get("source_rank_score_contract_sha256")
            != source_rank_contract.get("contract_sha256")
        ):
            raise CalibrationError("members do not share manifest/split/source contract")
        source_rank_unsigned = dict(source_rank_contract)
        source_rank_sha = source_rank_unsigned.pop("contract_sha256", None)
        source_temperature = source_rank_contract.get("success_temperature")
        if (
            not _is_sha(source_rank_sha)
            or source_rank_sha != canonical_sha256(source_rank_unsigned)
            or isinstance(source_temperature, bool)
            or not isinstance(source_temperature, (int, float))
            or not math.isfinite(float(source_temperature))
            or float(source_temperature) <= 0.0
            or not _is_sha(
                source_rank_contract.get("source_checkpoint_file_sha256")
            )
        ):
            raise CalibrationError("member Source composite rank contract changed")
        seeds.add(row["member_seed"])
        checkpoint_hashes.add(str(row["checkpoint_file_sha256"]))
        prediction_hashes.add(str(row["validation_predictions_file_sha256"]))
        checkpoint = safe_existing(Path(str(row["checkpoint_path"])), f"member {index} checkpoint")
        prediction = safe_existing(
            Path(str(row["validation_predictions_path"])),
            f"member {index} validation predictions",
            allowed_suffixes={".npz"},
        )
        if (
            file_sha256(checkpoint) != row["checkpoint_file_sha256"]
            or file_sha256(prediction) != row["validation_predictions_file_sha256"]
        ):
            raise CalibrationError(f"member {index} file SHA changed")
        decoded.append(
            {
                **dict(row),
                "checkpoint_path": str(checkpoint),
                "validation_predictions_path": str(prediction),
            }
        )
    source_rank_member_authority = {
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": row["member_index"],
                "source_checkpoint_file_sha256": row[
                    "source_rank_score_contract"
                ]["source_checkpoint_file_sha256"],
                "source_rank_score_contract_sha256": row[
                    "source_rank_score_contract_sha256"
                ],
                "success_temperature": row["source_rank_score_contract"][
                    "success_temperature"
                ],
            }
            for row in decoded
        ],
    }
    source_rank_member_authority_sha256 = canonical_sha256(
        source_rank_member_authority
    )
    validate_source_rank_member_authority(
        source_rank_member_authority, source_rank_member_authority_sha256
    )
    return {
        "logical_sha256": logical,
        "shared_contract": dict(shared),
        "prediction_contract": dict(prediction_contract),
        "validation_identity_set_sha256": value["validation_identity_set_sha256"],
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": source_rank_member_authority,
        "source_rank_member_authority_sha256": (
            source_rank_member_authority_sha256
        ),
        "labels_path": str(labels),
        "labels_file_sha256": value["labels_file_sha256"],
        "members": decoded,
    }


def _npz_arrays(path: Path, expected: frozenset[str], role: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise CalibrationError(f"{role} array fields changed")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise CalibrationError(f"{role} is not a safe NPZ") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise CalibrationError(f"{role} contains object/pickle arrays")
    return arrays


def _source_rank_float32_arrays(
    predictions: Sequence[Mapping[str, np.ndarray]],
    root_mask: np.ndarray,
    source_rank_member_authority: Mapping[str, Any],
    source_rank_member_authority_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the exact Source training arithmetic without dtype promotion."""

    source_rank_success_temperatures = validate_source_rank_member_authority(
        source_rank_member_authority, source_rank_member_authority_sha256
    )
    if len(predictions) != MEMBER_COUNT:
        raise CalibrationError("Source rank float32 numeric authority changed")
    mask = np.asarray(root_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise CalibrationError("Source rank root mask changed")
    bases: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    composites: list[np.ndarray] = []
    for index, (row, raw_temperature) in enumerate(
        zip(predictions, source_rank_success_temperatures, strict=True)
    ):
        if (
            isinstance(raw_temperature, bool)
            or not isinstance(raw_temperature, (int, float))
            or not math.isfinite(float(raw_temperature))
            or float(raw_temperature) <= 0.0
        ):
            raise CalibrationError(
                f"member {index} Source rank temperature is invalid"
            )
        temperature = np.float32(raw_temperature)
        if not np.isfinite(temperature) or temperature <= np.float32(0.0):
            raise CalibrationError(
                f"member {index} Source rank temperature is not usable as float32"
            )
        base = np.asarray(row["source_contract_base_rank_score"])
        residual = np.asarray(row["source_action_rank_residual"])
        composite = np.asarray(row["source_contract_rank_score"])
        if (
            base.dtype != np.float32
            or residual.dtype != np.float32
            or composite.dtype != np.float32
            or base.shape != mask.shape
            or residual.shape != mask.shape
            or composite.shape != mask.shape
            or not np.isfinite(base).all()
            or not np.isfinite(residual).all()
            or not np.isfinite(composite).all()
            or not np.array_equal(
                composite[mask],
                base[mask] + residual[mask] / temperature,
            )
            or not np.array_equal(
                base[~mask], np.zeros(int((~mask).sum()), dtype=np.float32)
            )
            or not np.array_equal(
                residual[~mask], np.zeros(int((~mask).sum()), dtype=np.float32)
            )
            or not np.array_equal(
                composite[~mask], np.zeros(int((~mask).sum()), dtype=np.float32)
            )
        ):
            raise CalibrationError(
                "Source composite rank-score float32 audit algebra changed"
            )
        bases.append(base)
        residuals.append(residual)
        composites.append(composite)
    return (
        np.stack(composites),
        np.stack(bases),
        np.stack(residuals),
    )


def load_validation_arrays(audit: Mapping[str, Any]) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    labels = _npz_arrays(Path(audit["labels_path"]), LABEL_KEYS, "validation labels")
    sample_ids = np.asarray(labels["sample_id"]).astype(str)
    group_ids = np.asarray(labels["group_id"]).astype(str)
    n = len(sample_ids)
    if (
        sample_ids.shape != (n,)
        or group_ids.shape != (n,)
        or n == 0
        or len(set(sample_ids.tolist())) != n
        or any(
            array.shape != (n,)
            for array in (
                labels["group_row_ordinal"], labels["current_event"],
                labels["post_event"], labels["next_event"], labels["success"],
                labels["regress"], labels["recovery"],
                labels["recovery_observed"], labels["duration"],
                labels["duration_observed"], labels["object_observed"],
                labels["root_candidate"], labels["candidate_index"],
                labels["is_baseline"], labels["candidate_final_success"],
            )
        )
        or labels["object_target"].ndim != 2
        or labels["object_target"].shape[0] != n
        or labels["object_target"].shape[1] < 1
    ):
        raise CalibrationError("validation label shapes are invalid")
    identity_sha = canonical_sha256(sample_ids.tolist())
    if identity_sha != audit["validation_identity_set_sha256"]:
        raise CalibrationError("validation identity-set SHA changed")
    for name in (
        "group_row_ordinal", "current_event", "post_event", "next_event",
        "success", "recovery", "candidate_index", "candidate_final_success",
    ):
        values = np.asarray(labels[name])
        allow_negative = name in {"candidate_index", "candidate_final_success"}
        if (
            not np.issubdtype(values.dtype, np.integer)
            or (not allow_negative and bool((values < 0).any()))
        ):
            raise CalibrationError(f"invalid categorical label: {name}")
    if not np.isin(labels["success"], [0, 1]).all() or not np.isin(
        labels["recovery"], [0, 1]
    ).all():
        raise CalibrationError("success/recovery labels must be binary")
    regress = np.asarray(labels["regress"], dtype=bool)
    recovery_observed = np.asarray(labels["recovery_observed"], dtype=bool)
    recovery = np.asarray(labels["recovery"], dtype=bool)
    if bool((recovery_observed & ~regress).any()) or bool(
        (recovery & ~recovery_observed).any()
    ):
        raise CalibrationError(
            "recovery labels must be observed only for operational regress rows"
        )
    if (
        not np.isfinite(labels["duration"]).all()
        or bool((labels["duration"] < 0).any())
        or not np.isfinite(labels["object_target"]).all()
    ):
        raise CalibrationError("continuous validation labels are invalid")
    root_candidate = np.asarray(labels["root_candidate"])
    is_baseline = np.asarray(labels["is_baseline"])
    candidate_index = np.asarray(labels["candidate_index"])
    candidate_success = np.asarray(labels["candidate_final_success"])
    if (
        root_candidate.dtype != np.bool_
        or is_baseline.dtype != np.bool_
        or bool((is_baseline & ~root_candidate).any())
        or bool(np.isin(candidate_index[root_candidate], [0, 1, 2, 3]).all())
        is not True
        or bool(np.isin(candidate_success[root_candidate], [0, 1]).all())
        is not True
        or bool((candidate_index[~root_candidate] != -1).any())
        or bool((candidate_success[~root_candidate] != -1).any())
        or bool(is_baseline[~root_candidate].any())
    ):
        raise CalibrationError("formal root-candidate metadata is invalid")
    row_ordinals = np.asarray(labels["group_row_ordinal"])
    for group in sorted(set(group_ids.tolist())):
        mask = group_ids == group
        if not np.array_equal(row_ordinals[mask], np.arange(int(mask.sum()))):
            raise CalibrationError("group row ordinals are not exact and contiguous")
        root = mask & root_candidate
        indices = candidate_index[root]
        if (
            len(indices) < 2
            or len(set(indices.tolist())) != len(indices)
            or int(is_baseline[root].sum()) != 1
            or int(indices[is_baseline[root]][0]) != int(indices.min())
        ):
            raise CalibrationError(
                "each formal group requires unique legal roots and lowest baseline"
            )
    predictions = []
    event_classes: int | None = None
    object_dim = labels["object_target"].shape[1]
    for member in audit["members"]:
        row = _npz_arrays(
            Path(member["validation_predictions_path"]),
            PREDICTION_KEYS,
            f"member {member['member_index']} validation predictions",
        )
        if not np.array_equal(np.asarray(row["sample_id"]).astype(str), sample_ids):
            raise CalibrationError("member prediction sample order changed")
        post, nxt = np.asarray(row["post_event_logits"]), np.asarray(row["next_event_logits"])
        if event_classes is None:
            event_classes = post.shape[1] if post.ndim == 2 else None
        vectors = (
            row["success_logit"], row["recovery_logit"],
            row["source_contract_base_rank_score"],
            row["source_action_rank_residual"],
            row["source_contract_rank_score"],
            row["duration_log_mean"], row["duration_log_scale"],
        )
        if (
            post.shape != (n, event_classes)
            or nxt.shape != (n, event_classes)
            or event_classes is None
            or event_classes < 2
            or any(np.asarray(array).shape != (n,) for array in vectors)
            or np.asarray(row["object_mean"]).shape != (n, object_dim)
            or np.asarray(row["object_log_scale"]).shape != (n, object_dim)
            or any(not np.isfinite(np.asarray(array, dtype=np.float64)).all() for name, array in row.items() if name != "sample_id")
        ):
            raise CalibrationError("member prediction shapes/values are invalid")
        predictions.append(row)
    _source_rank_float32_arrays(
        predictions,
        np.asarray(labels["root_candidate"]),
        audit["source_rank_member_authority"],
        audit["source_rank_member_authority_sha256"],
    )
    assert event_classes is not None
    if (
        bool((labels["current_event"] >= event_classes).any())
        or bool((labels["post_event"] >= event_classes).any())
        or bool((labels["next_event"] >= event_classes).any())
    ):
        raise CalibrationError("event label exceeds member class dimension")
    labels = {
        **labels,
        "sample_id": sample_ids,
        "group_id": group_ids,
        "prediction_contract": audit["prediction_contract"],
        "source_rank_numeric_contract": audit["source_rank_numeric_contract"],
        "source_rank_member_authority": dict(
            audit["source_rank_member_authority"]
        ),
        "source_rank_member_authority_sha256": audit[
            "source_rank_member_authority_sha256"
        ],
    }
    return predictions, labels


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def _ece(confidence: np.ndarray, correct: np.ndarray) -> float:
    total = len(confidence)
    value = 0.0
    for index in range(ECE_BINS):
        lower, upper = index / ECE_BINS, (index + 1) / ECE_BINS
        mask = (confidence >= lower) & (
            confidence <= upper if index == ECE_BINS - 1 else confidence < upper
        )
        if mask.any():
            value += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return value if total else math.nan


def _macro_f1(target: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    values = []
    for label in range(classes):
        tp = int(((target == label) & (prediction == label)).sum())
        fp = int(((target != label) & (prediction == label)).sum())
        fn = int(((target == label) & (prediction != label)).sum())
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def _logical_group_folds(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups).astype(str)
    unique = sorted(set(values.tolist()))
    mapping = {
        group: index % CROSSFIT_FOLDS for index, group in enumerate(unique)
    }
    return np.asarray([mapping[group] for group in values], dtype=np.int64)


def _equal_group_weighted_quantile(
    values: np.ndarray, groups: np.ndarray, quantile: float,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    valid = np.isfinite(array)
    if not bool(valid.any()) or not 0.0 <= quantile <= 1.0:
        return math.nan
    weights = _equal_group_row_weights(group_values, valid)
    order = np.argsort(array[valid], kind="stable")
    ordered_values = array[valid][order]
    ordered_weights = weights[valid][order]
    cumulative = np.cumsum(ordered_weights)
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _equal_group_row_weights(groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(groups).astype(str)
    observed = np.asarray(mask, dtype=bool)
    weights = np.zeros(len(values), dtype=np.float64)
    retained = sorted(set(values[observed].tolist()))
    if not retained:
        return weights
    for group in retained:
        rows = observed & (values == group)
        weights[rows] = 1.0 / (len(retained) * int(rows.sum()))
    return weights


def _group_mean_vector(
    values: np.ndarray, groups: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    observed = np.asarray(mask, dtype=bool) & np.isfinite(array)
    names = np.asarray(sorted(set(group_values[observed].tolist())))
    means = np.asarray(
        [array[observed & (group_values == name)].mean() for name in names],
        dtype=np.float64,
    )
    return names, means


def _group_bootstrap_mean_interval(
    values: np.ndarray, *, samples: int, seed_offset: int = 0,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 2 or type(samples) is not int or samples < 100:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, BOOTSTRAP_ALPHA)),
        float(np.quantile(means, 1.0 - BOOTSTRAP_ALPHA)),
    )


def _additive_gain_gate(
    model_loss: np.ndarray, baseline_loss: np.ndarray,
    groups: np.ndarray, mask: np.ndarray, *, samples: int, role: str,
) -> dict[str, Any]:
    names, gain = _group_mean_vector(
        np.asarray(baseline_loss, dtype=np.float64)
        - np.asarray(model_loss, dtype=np.float64),
        groups,
        mask,
    )
    point, lower, upper = _group_bootstrap_mean_interval(
        gain, samples=samples,
        seed_offset=int.from_bytes(hashlib.sha256(role.encode()).digest()[:2], "big"),
    )
    return {
        "comparison": role,
        "equal_logical_group_count": int(len(names)),
        "model_minus_baseline_orientation": "positive_is_model_improvement",
        "gain": point,
        "group_bootstrap_gain_lcb95": lower,
        "group_bootstrap_gain_ucb95": upper,
        "passed_zero_gain_lcb": bool(math.isfinite(lower) and lower >= 0.0),
    }


def _coverage_gain_gate(
    model_covered: np.ndarray, baseline_covered: np.ndarray,
    groups: np.ndarray, mask: np.ndarray, *, samples: int, role: str,
) -> dict[str, Any]:
    model_names, model_group = _group_mean_vector(
        model_covered, groups, mask
    )
    baseline_names, baseline_group = _group_mean_vector(
        baseline_covered, groups, mask
    )
    point = lower = upper = math.nan
    if (
        np.array_equal(model_names, baseline_names)
        and len(model_names) >= 2
        and type(samples) is int
        and samples >= 100
    ):
        point = float(
            abs(float(baseline_group.mean()) - INTERVAL_MASS)
            - abs(float(model_group.mean()) - INTERVAL_MASS)
        )
        rng = np.random.default_rng(
            BOOTSTRAP_SEED
            + int.from_bytes(hashlib.sha256(role.encode()).digest()[:2], "big")
        )
        draws = rng.integers(
            0, len(model_names), size=(samples, len(model_names))
        )
        model_draw = model_group[draws].mean(axis=1)
        baseline_draw = baseline_group[draws].mean(axis=1)
        gain = np.abs(baseline_draw - INTERVAL_MASS) - np.abs(
            model_draw - INTERVAL_MASS
        )
        lower = float(np.quantile(gain, BOOTSTRAP_ALPHA))
        upper = float(np.quantile(gain, 1.0 - BOOTSTRAP_ALPHA))
    return {
        "comparison": role,
        "target_coverage": INTERVAL_MASS,
        "equal_logical_group_count": int(len(model_names)),
        "gain": point,
        "group_bootstrap_gain_lcb95": lower,
        "group_bootstrap_gain_ucb95": upper,
        "passed_zero_gain_lcb": bool(math.isfinite(lower) and lower >= 0.0),
    }


def _temperature_grid() -> np.ndarray:
    return np.exp(np.linspace(math.log(0.05), math.log(20.0), 161))


def _fit_multiclass_temperature(
    logits: np.ndarray, target: np.ndarray, groups: np.ndarray,
    observed: np.ndarray, *, required_classes: Sequence[int] | None = None,
) -> float | None:
    mask = np.asarray(observed, dtype=bool)
    classes = logits.shape[-1]
    required = set(range(classes) if required_classes is None else required_classes)
    if not bool(mask.any()) or not required <= set(np.asarray(target)[mask].tolist()):
        return None
    weights = _equal_group_row_weights(groups, mask)
    losses = []
    for temperature in _temperature_grid():
        probability = _softmax(logits / temperature).mean(axis=0)
        row = -np.log(
            np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0)
        )
        losses.append(float(np.sum(weights * row)))
    return float(_temperature_grid()[int(np.argmin(losses))])


def _fit_binary_temperature_grouped(
    logits: np.ndarray, target: np.ndarray, groups: np.ndarray,
    observed: np.ndarray,
) -> float | None:
    mask = np.asarray(observed, dtype=bool)
    binary = np.asarray(target, dtype=np.int64)
    if not bool(mask.any()) or set(binary[mask].tolist()) != {0, 1}:
        return None
    weights = _equal_group_row_weights(groups, mask)
    losses = []
    for temperature in _temperature_grid():
        member = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
        probability = member.mean(axis=0)
        row = -(
            binary * np.log(np.clip(probability, 1e-12, 1.0))
            + (1 - binary) * np.log(np.clip(1 - probability, 1e-12, 1.0))
        )
        losses.append(float(np.sum(weights * row)))
    return float(_temperature_grid()[int(np.argmin(losses))])


def _weighted_average_precision(
    target: np.ndarray, score: np.ndarray, weights: np.ndarray,
) -> float:
    binary = np.asarray(target, dtype=np.int64)
    probability = np.asarray(score, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    positive_weight = float(weight[binary == 1].sum())
    if positive_weight <= 0 or float(weight[binary == 0].sum()) <= 0:
        return math.nan
    order = np.argsort(-probability, kind="stable")
    ordered_target = binary[order]
    ordered_weight = weight[order]
    cumulative_positive = np.cumsum(ordered_weight * ordered_target)
    cumulative_total = np.cumsum(ordered_weight)
    precision = np.divide(
        cumulative_positive, cumulative_total,
        out=np.zeros_like(cumulative_positive), where=cumulative_total > 0,
    )
    return float(
        np.sum(precision * ordered_weight * ordered_target) / positive_weight
    )


def _bootstrap_ap_gain(
    target: np.ndarray, model: np.ndarray, baseline: np.ndarray,
    groups: np.ndarray, observed: np.ndarray, *, samples: int, role: str,
) -> dict[str, Any]:
    group_values = np.asarray(groups).astype(str)
    mask = np.asarray(observed, dtype=bool)
    names = np.asarray(sorted(set(group_values[mask].tolist())))
    if len(names) < 2 or samples < 100:
        return {
            "comparison": role, "equal_logical_group_count": int(len(names)),
            "gain": math.nan, "group_bootstrap_gain_lcb95": math.nan,
            "group_bootstrap_gain_ucb95": math.nan,
            "passed_zero_gain_lcb": False,
        }

    def one(sampled: Sequence[str]) -> float:
        target_parts = []
        model_parts = []
        baseline_parts = []
        weight_parts = []
        for group in sampled:
            rows = np.flatnonzero(mask & (group_values == group))
            target_parts.append(target[rows])
            model_parts.append(model[rows])
            baseline_parts.append(baseline[rows])
            weight_parts.append(np.full(len(rows), 1.0 / len(rows)))
        weights = np.concatenate(weight_parts)
        return _weighted_average_precision(
            np.concatenate(target_parts), np.concatenate(model_parts), weights
        ) - _weighted_average_precision(
            np.concatenate(target_parts), np.concatenate(baseline_parts), weights
        )

    point = one(names.tolist())
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + int.from_bytes(hashlib.sha256(role.encode()).digest()[:2], "big")
    )
    draws = rng.integers(0, len(names), size=(samples, len(names)))
    gains = np.asarray([one(names[row].tolist()) for row in draws])
    gains = gains[np.isfinite(gains)]
    lower = float(np.quantile(gains, BOOTSTRAP_ALPHA)) if len(gains) else math.nan
    upper = float(np.quantile(gains, 1.0 - BOOTSTRAP_ALPHA)) if len(gains) else math.nan
    return {
        "comparison": role,
        "equal_logical_group_count": int(len(names)),
        "gain": float(point),
        "group_bootstrap_gain_lcb95": lower,
        "group_bootstrap_gain_ucb95": upper,
        "passed_zero_gain_lcb": bool(math.isfinite(lower) and lower >= 0.0),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def uncertainty_performance_gate(
    uncertainty: np.ndarray, error: np.ndarray, groups: np.ndarray,
    observed: np.ndarray, *, samples: int, role: str,
) -> dict[str, Any]:
    names_u, group_uncertainty = _group_mean_vector(
        uncertainty, groups, observed
    )
    names_e, group_error = _group_mean_vector(error, groups, observed)
    if not np.array_equal(names_u, names_e) or len(names_u) < 2:
        return {
            "status": "disabled_insufficient_equal_groups",
            "passed": False, "equal_logical_group_count": int(len(names_u)),
        }

    def statistics(indices: np.ndarray) -> tuple[float, float]:
        u = group_uncertainty[indices]
        e = group_error[indices]
        ordered_error = e[np.argsort(u, kind="stable")]
        risk = np.cumsum(ordered_error) / np.arange(1, len(e) + 1)
        aurc_gain = float(e.mean() - risk.mean())
        rank_u, rank_e = _rankdata(u), _rankdata(e)
        if float(rank_u.std()) == 0.0 or float(rank_e.std()) == 0.0:
            correlation = math.nan
        else:
            correlation = float(np.corrcoef(rank_u, rank_e)[0, 1])
        return aurc_gain, correlation

    point_gain, point_correlation = statistics(np.arange(len(names_u)))
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + int.from_bytes(hashlib.sha256((role + ":uncertainty").encode()).digest()[:2], "big")
    )
    draws = rng.integers(0, len(names_u), size=(samples, len(names_u)))
    sampled = np.asarray([statistics(row) for row in draws])
    valid_gain = sampled[np.isfinite(sampled[:, 0]), 0]
    valid_correlation = sampled[np.isfinite(sampled[:, 1]), 1]
    gain_lcb = float(np.quantile(valid_gain, BOOTSTRAP_ALPHA)) if len(valid_gain) else math.nan
    correlation_lcb = (
        float(np.quantile(valid_correlation, BOOTSTRAP_ALPHA))
        if len(valid_correlation) else math.nan
    )
    passed = bool(
        math.isfinite(gain_lcb) and gain_lcb >= 0.0
        and math.isfinite(correlation_lcb) and correlation_lcb >= 0.0
    )
    return {
        "status": "complete_equal_group_aurc_rank_correlation_gate",
        "passed": passed,
        "equal_logical_group_count": int(len(names_u)),
        "aurc_gain_over_unranked": point_gain,
        "aurc_gain_group_bootstrap_lcb95": gain_lcb,
        "uncertainty_error_rank_correlation": point_correlation,
        "rank_correlation_group_bootstrap_lcb95": correlation_lcb,
        "zero_gain_lcb_required": True,
    }


def event_metrics(
    member_logits: np.ndarray,
    target: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = _softmax(member_logits)
    ensemble = probabilities.mean(axis=0)
    prediction = ensemble.argmax(axis=-1)
    confidence = ensemble.max(axis=-1)
    correct = prediction == target
    aleatoric = _entropy(probabilities).mean(axis=0)
    total = _entropy(ensemble)
    epistemic = np.maximum(total - aleatoric, 0.0)
    observed = (
        np.ones(len(target), dtype=bool)
        if observation_mask is None
        else np.asarray(observation_mask, dtype=bool)
    )
    if observed.shape != (len(target),) or not bool(observed.any()):
        raise CalibrationError("event metric observation mask has no valid rows")
    metrics = {
        "nll": float(
            -np.log(
                np.clip(
                    ensemble[np.arange(len(target))[observed], target[observed]],
                    1e-12,
                    1,
                )
            ).mean()
        ),
        "macro_f1": _macro_f1(
            target[observed], prediction[observed], ensemble.shape[1]
        ),
        "ece_10_equal_width": _ece(
            confidence[observed], correct[observed].astype(np.float64)
        ),
        "accuracy": float(correct[observed].mean()),
        "mean_aleatoric_entropy": float(aleatoric[observed].mean()),
        "mean_epistemic_mutual_information": float(epistemic[observed].mean()),
        "mean_total_predictive_entropy": float(total[observed].mean()),
        "observed_rows": int(observed.sum()),
        "censored_rows_excluded": int((~observed).sum()),
        "metric_mask": (
            "all_rows" if observation_mask is None else "duration_observed_only"
        ),
    }
    return metrics, {
        "probability": ensemble,
        "correct": np.where(observed, correct.astype(float), np.nan),
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
    }


def crossfit_event_calibration(
    member_logits: np.ndarray, target: np.ndarray, current_event: np.ndarray,
    groups: np.ndarray, *, observation_mask: np.ndarray | None,
    baseline: str, bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    logits = np.asarray(member_logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    current = np.asarray(current_event, dtype=np.int64)
    group_values = np.asarray(groups).astype(str)
    observed = (
        np.ones(len(target), dtype=bool)
        if observation_mask is None else np.asarray(observation_mask, dtype=bool)
    )
    classes = logits.shape[-1]
    required_classes = list(range(classes)) if baseline == "persistence" else list(
        range(1, classes)
    )
    folds = _logical_group_folds(group_values)
    probability = np.full((len(target), classes), np.nan, dtype=np.float64)
    member_probability = np.full(logits.shape, np.nan, dtype=np.float64)
    baseline_probability = np.full_like(probability, np.nan)
    fold_temperatures: list[float] = []
    fold_support: list[dict[str, Any]] = []
    for fold in range(CROSSFIT_FOLDS):
        heldout_all = folds == fold
        heldout = (folds == fold) & observed
        training = (folds != fold) & observed
        temperature = _fit_multiclass_temperature(
            logits, target, group_values, training,
            required_classes=required_classes,
        )
        support_row = {
            "fold": fold,
            "training_groups": len(set(group_values[training].tolist())),
            "heldout_groups": len(set(group_values[heldout].tolist())),
            "all_required_training_classes_present": temperature is not None,
            "heldout_rows": int(heldout.sum()),
        }
        fold_support.append(support_row)
        if temperature is None or not bool(heldout_all.any()):
            continue
        fold_temperatures.append(temperature)
        calibrated_member = _softmax(logits[:, heldout_all, :] / temperature)
        member_probability[:, heldout_all, :] = calibrated_member
        probability[heldout_all] = calibrated_member.mean(axis=0)
        if baseline == "persistence":
            baseline_probability[heldout_all] = (1.0 - 0.99) / (classes - 1)
            baseline_probability[
                np.flatnonzero(heldout_all), current[heldout_all]
            ] = 0.99
        elif baseline == "prior":
            weights = _equal_group_row_weights(group_values, training)
            prior = np.asarray(
                [weights[(target == label) & training].sum() for label in range(classes)]
            )
            prior = np.clip(prior, 1e-9, None)
            prior /= prior.sum()
            baseline_probability[heldout_all] = prior
        else:
            raise CalibrationError("event baseline is not preregistered")
    crossfit_complete = bool(
        len(fold_temperatures) == CROSSFIT_FOLDS
        and np.isfinite(probability[observed]).all()
        and np.isfinite(baseline_probability[observed]).all()
    )
    if crossfit_complete:
        model_nll = -np.log(
            np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0)
        )
        baseline_nll = -np.log(
            np.clip(
                baseline_probability[np.arange(len(target)), target], 1e-12, 1.0
            )
        )
        model_error = (probability.argmax(axis=1) != target).astype(np.float64)
        baseline_error = (
            baseline_probability.argmax(axis=1) != target
        ).astype(np.float64)
        nll_gate = _additive_gain_gate(
            model_nll, baseline_nll, group_values, observed,
            samples=bootstrap_samples, role=f"event_{baseline}_nll",
        )
        accuracy_gate = _additive_gain_gate(
            model_error, baseline_error, group_values, observed,
            samples=bootstrap_samples, role=f"event_{baseline}_accuracy",
        )
        aleatoric = _entropy(member_probability)
        mean_aleatoric = np.full(len(target), np.nan, dtype=np.float64)
        mean_aleatoric[observed] = aleatoric[:, observed].mean(axis=0)
        total = _entropy(probability)
        epistemic = np.maximum(total - mean_aleatoric, 0.0)
        normalized_uncertainty = total / math.log(classes)
        uncertainty_gate = uncertainty_performance_gate(
            normalized_uncertainty, model_error, group_values, observed,
            samples=bootstrap_samples, role=f"event_{baseline}",
        )
    else:
        model_nll = baseline_nll = model_error = np.full(len(target), np.nan)
        nll_gate = accuracy_gate = {
            "equal_logical_group_count": 0,
            "group_bootstrap_gain_lcb95": math.nan,
            "passed_zero_gain_lcb": False,
        }
        uncertainty_gate = {
            "status": "disabled_incomplete_group_crossfit", "passed": False,
            "equal_logical_group_count": 0,
        }
        mean_aleatoric = total = epistemic = np.full(len(target), np.nan)
    performance_passed = bool(
        crossfit_complete
        and nll_gate["passed_zero_gain_lcb"]
        and accuracy_gate["passed_zero_gain_lcb"]
        and uncertainty_gate["passed"]
    )
    deployment_temperature = _fit_multiclass_temperature(
        logits, target, group_values, observed,
        required_classes=required_classes,
    ) if performance_passed else None
    metrics = {
        "calibration_protocol": PERFORMANCE_GATE_PROTOCOL,
        "baseline": baseline,
        "crossfit_folds": CROSSFIT_FOLDS,
        "fold_support": fold_support,
        "crossfit_complete": crossfit_complete,
        "all_row_oof_inference_complete": bool(
            np.isfinite(probability).all()
            and np.isfinite(member_probability).all()
        ),
        "fold_temperatures": fold_temperatures,
        "deployment_temperature": (
            float(deployment_temperature) if deployment_temperature is not None else 1.0
        ),
        "equal_group_model_nll": float(
            _group_mean_vector(model_nll, group_values, observed)[1].mean()
        ) if crossfit_complete else None,
        "equal_group_model_accuracy": float(
            1.0 - _group_mean_vector(model_error, group_values, observed)[1].mean()
        ) if crossfit_complete else None,
        "nll_gain_gate": nll_gate,
        "accuracy_gain_gate": accuracy_gate,
        "uncertainty_gate": uncertainty_gate,
        "performance_gate_passed": performance_passed,
        "metric_weighting": "equal_logical_group",
        "observed_rows": int(observed.sum()),
    }
    return metrics, {
        "probability": probability,
        "correct": np.where(observed, 1.0 - model_error, np.nan),
        "aleatoric": mean_aleatoric,
        "epistemic": epistemic,
        "total": total,
        "observation_mask": observed,
    }


def crossfit_binary_calibration(
    member_logits: np.ndarray, target: np.ndarray, groups: np.ndarray, *,
    observation_mask: np.ndarray | None, bootstrap_samples: int,
    head_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    logits = np.asarray(member_logits, dtype=np.float64)
    binary = np.asarray(target, dtype=np.int64)
    group_values = np.asarray(groups).astype(str)
    observed = (
        np.ones(len(binary), dtype=bool)
        if observation_mask is None else np.asarray(observation_mask, dtype=bool)
    )
    folds = _logical_group_folds(group_values)
    probability = np.full(len(binary), np.nan)
    baseline_probability = np.full(len(binary), np.nan)
    member_probability = np.full(logits.shape, np.nan)
    temperatures: list[float] = []
    fold_support: list[dict[str, Any]] = []
    for fold in range(CROSSFIT_FOLDS):
        heldout_all = folds == fold
        heldout = (folds == fold) & observed
        training = (folds != fold) & observed
        temperature = _fit_binary_temperature_grouped(
            logits, binary, group_values, training
        )
        weights = _equal_group_row_weights(group_values, training)
        prevalence = float(np.sum(weights * binary)) if bool(training.any()) else math.nan
        fold_support.append({
            "fold": fold,
            "training_groups": len(set(group_values[training].tolist())),
            "heldout_groups": len(set(group_values[heldout].tolist())),
            "training_positive_and_negative_present": temperature is not None,
            "heldout_rows": int(heldout.sum()),
        })
        if (
            temperature is None
            or not bool(heldout_all.any())
            or not 0.0 < prevalence < 1.0
        ):
            continue
        temperatures.append(temperature)
        calibrated_member = 1.0 / (
            1.0
            + np.exp(-np.clip(logits[:, heldout_all] / temperature, -40, 40))
        )
        member_probability[:, heldout_all] = calibrated_member
        probability[heldout_all] = calibrated_member.mean(axis=0)
        baseline_probability[heldout_all] = prevalence
    complete = bool(
        len(temperatures) == CROSSFIT_FOLDS
        and np.isfinite(probability[observed]).all()
        and np.isfinite(baseline_probability[observed]).all()
    )
    if complete:
        model_nll = -(
            binary * np.log(np.clip(probability, 1e-12, 1.0))
            + (1 - binary) * np.log(np.clip(1 - probability, 1e-12, 1.0))
        )
        baseline_nll = -(
            binary * np.log(np.clip(baseline_probability, 1e-12, 1.0))
            + (1 - binary)
            * np.log(np.clip(1 - baseline_probability, 1e-12, 1.0))
        )
        model_brier = np.square(probability - binary)
        baseline_brier = np.square(baseline_probability - binary)
        nll_gate = _additive_gain_gate(
            model_nll, baseline_nll, group_values, observed,
            samples=bootstrap_samples, role=f"{head_name}_prevalence_nll",
        )
        brier_gate = _additive_gain_gate(
            model_brier, baseline_brier, group_values, observed,
            samples=bootstrap_samples, role=f"{head_name}_prevalence_brier",
        )
        ap_gate = _bootstrap_ap_gain(
            binary, probability, baseline_probability, group_values, observed,
            samples=bootstrap_samples, role=f"{head_name}_prevalence_ap",
        )
        aleatoric = np.nanmean(
            member_probability * (1.0 - member_probability), axis=0
        )
        epistemic = np.nanvar(member_probability, axis=0)
        total = aleatoric + epistemic
        uncertainty_gate = uncertainty_performance_gate(
            total / 0.25, model_brier, group_values, observed,
            samples=bootstrap_samples, role=head_name,
        )
    else:
        model_nll = model_brier = np.full(len(binary), np.nan)
        disabled = {
            "equal_logical_group_count": 0,
            "group_bootstrap_gain_lcb95": math.nan,
            "passed_zero_gain_lcb": False,
        }
        nll_gate = brier_gate = ap_gate = disabled
        uncertainty_gate = {
            "status": "disabled_incomplete_group_crossfit", "passed": False,
            "equal_logical_group_count": 0,
        }
        aleatoric = epistemic = total = np.full(len(binary), np.nan)
    passed = bool(
        complete and nll_gate["passed_zero_gain_lcb"]
        and brier_gate["passed_zero_gain_lcb"]
        and ap_gate["passed_zero_gain_lcb"]
        and uncertainty_gate["passed"]
    )
    deployment_temperature = _fit_binary_temperature_grouped(
        logits, binary, group_values, observed
    ) if passed else None
    metrics = {
        "calibration_protocol": PERFORMANCE_GATE_PROTOCOL,
        "baseline": "training_fold_equal_group_prevalence",
        "crossfit_folds": CROSSFIT_FOLDS,
        "fold_support": fold_support,
        "crossfit_complete": complete,
        "all_row_oof_inference_complete": bool(
            np.isfinite(probability).all()
            and np.isfinite(member_probability).all()
        ),
        "fold_temperatures": temperatures,
        "deployment_temperature": (
            float(deployment_temperature) if deployment_temperature is not None else 1.0
        ),
        "equal_group_model_nll": float(
            _group_mean_vector(model_nll, group_values, observed)[1].mean()
        ) if complete else None,
        "equal_group_model_brier": float(
            _group_mean_vector(model_brier, group_values, observed)[1].mean()
        ) if complete else None,
        "nll_gain_gate": nll_gate,
        "brier_gain_gate": brier_gate,
        "average_precision_gain_gate": ap_gate,
        "uncertainty_gate": uncertainty_gate,
        "performance_gate_passed": passed,
        "metric_weighting": "equal_logical_group",
        "observed_rows": int(observed.sum()),
    }
    return metrics, {
        "probability": probability,
        "correct": np.where(
            observed, ((probability >= 0.5) == binary).astype(float), np.nan
        ),
        "aleatoric": np.where(observed, aleatoric, np.nan),
        "epistemic": np.where(observed, epistemic, np.nan),
        "total": np.where(observed, total, np.nan),
        "observation_mask": observed,
    }


def fit_success_temperature(
    member_logits: np.ndarray,
    target: np.ndarray,
    *,
    enabled: bool = True,
    observation_mask: np.ndarray | None = None,
    head_name: str = "success",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit a validation-only binary temperature with an optional label mask.

    Recovery is defined only after an observed operational regression.  Rows
    outside that condition are inapplicable, not negative examples, and must
    not affect either temperature fitting or deployment-threshold quality.
    """

    logits = np.asarray(member_logits, dtype=np.float64)
    binary_target = np.asarray(target, dtype=np.int64)
    if logits.ndim != 2 or binary_target.shape != (logits.shape[1],):
        raise CalibrationError(f"{head_name} binary calibration shape changed")
    observed = (
        np.ones(len(binary_target), dtype=bool)
        if observation_mask is None
        else np.asarray(observation_mask, dtype=bool)
    )
    if observed.shape != binary_target.shape or not np.isin(
        binary_target, [0, 1]
    ).all():
        raise CalibrationError(f"{head_name} binary calibration labels are invalid")
    temperatures = (
        np.exp(np.linspace(math.log(0.05), math.log(20.0), 401))
        if enabled and bool(observed.any())
        else np.asarray([1.0])
    )
    losses = []
    for temperature in temperatures:
        member_probability = 1.0 / (
            1.0 + np.exp(-np.clip(logits / temperature, -40, 40))
        )
        probability = member_probability.mean(axis=0)
        if bool(observed.any()):
            losses.append(
                float(
                    -(
                        binary_target[observed]
                        * np.log(np.clip(probability[observed], 1e-12, 1))
                        + (1 - binary_target[observed])
                        * np.log(np.clip(1 - probability[observed], 1e-12, 1))
                    ).mean()
                )
            )
        else:
            losses.append(math.inf)
    temperature = float(temperatures[int(np.argmin(losses))])
    member_probability = 1.0 / (
        1.0 + np.exp(-np.clip(logits / temperature, -40, 40))
    )
    probability = member_probability.mean(axis=0)
    aleatoric = (member_probability * (1 - member_probability)).mean(axis=0)
    epistemic = member_probability.var(axis=0)
    total = aleatoric + epistemic
    if bool(observed.any()):
        nll = float(
            -(
                binary_target[observed]
                * np.log(np.clip(probability[observed], 1e-12, 1))
                + (1 - binary_target[observed])
                * np.log(np.clip(1 - probability[observed], 1e-12, 1))
            ).mean()
        )
        brier = float(
            np.square(probability[observed] - binary_target[observed]).mean()
        )
        ece = _ece(probability[observed], binary_target[observed].astype(float))
        mean_aleatoric = float(aleatoric[observed].mean())
        mean_epistemic = float(epistemic[observed].mean())
        mean_total = float(total[observed].mean())
    else:
        nll = brier = ece = None
        mean_aleatoric = mean_epistemic = mean_total = None
    metrics = {
        "calibration_status": (
            "fitted_on_validation_only"
            if enabled and bool(observed.any())
            else (
                "disabled_no_observed_conditional_rows_temperature_fixed_at_one"
                if not bool(observed.any())
                else "disabled_head_support_insufficient_temperature_fixed_at_one"
            )
        ),
        "temperature": temperature,
        "nll": nll,
        "brier": brier,
        "ece_10_equal_width": ece,
        "observed_rows": int(observed.sum()),
        "inapplicable_or_unobserved_rows": int((~observed).sum()),
        "mean_aleatoric_variance": mean_aleatoric,
        "mean_epistemic_variance": mean_epistemic,
        "mean_total_variance": mean_total,
    }
    return metrics, {
        "probability": probability,
        "correct": np.where(
            observed,
            ((probability >= 0.5) == binary_target).astype(float),
            np.nan,
        ),
        "aleatoric": np.where(observed, aleatoric, np.nan),
        "epistemic": np.where(observed, epistemic, np.nan),
        "total": np.where(observed, total, np.nan),
        "observation_mask": observed,
    }


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    output = np.fromiter((0.5 * (1.0 + math.erf(float(item) / math.sqrt(2.0))) for item in flat), dtype=np.float64, count=len(flat))
    return output.reshape(np.asarray(value).shape)


def _mixture_lognormal_quantile(means: np.ndarray, scales: np.ndarray, probability: float) -> np.ndarray:
    lower = np.min(means - 10.0 * scales, axis=0)
    upper = np.max(means + 10.0 * scales, axis=0)
    for _ in range(70):
        middle = (lower + upper) / 2.0
        cdf = _normal_cdf((middle[None, :] - means) / scales).mean(axis=0)
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    # The adapter clock is supervised on log(1 + D).  Therefore 1 + D is
    # lognormal and the physical decision-step duration is shifted by -1.
    return np.maximum(np.exp(np.clip((lower + upper) / 2.0, -30, 30)) - 1.0, 0.0)


def duration_metrics(means: np.ndarray, log_scales: np.ndarray, target: np.ndarray, observed: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scales = np.exp(np.clip(log_scales, -8, 5))
    median = _mixture_lognormal_quantile(means, scales, 0.5)
    lower = _mixture_lognormal_quantile(means, scales, (1 - INTERVAL_MASS) / 2)
    upper = _mixture_lognormal_quantile(means, scales, 1 - (1 - INTERVAL_MASS) / 2)
    member_mean = np.exp(np.clip(means + 0.5 * scales**2, -30, 30)) - 1.0
    member_var = (np.exp(np.clip(scales**2, 0, 30)) - 1) * np.exp(np.clip(2 * means + scales**2, -30, 30))
    aleatoric = member_var.mean(axis=0)
    epistemic = member_mean.var(axis=0)
    total = aleatoric + epistemic
    covered = (target >= lower) & (target <= upper)
    observed_count = int(observed.sum())
    metrics = {
        "mixture_median_mae_observed": float(np.abs(median[observed] - target[observed]).mean()) if observed_count else None,
        "central_90_interval_coverage_observed": float(covered[observed].mean()) if observed_count else None,
        "observed_count": observed_count,
        "censored_count": int((~observed).sum()),
        "median_prediction_mean": float(median.mean()),
        "mean_aleatoric_variance": float(aleatoric.mean()),
        "mean_epistemic_variance": float(epistemic.mean()),
        "mean_total_variance": float(total.mean()),
    }
    quality = np.where(observed, covered.astype(float), np.nan)
    relative_std = np.sqrt(np.maximum(total, 0)) / np.maximum(median, 1e-8)
    uncertainty = relative_std / (1 + relative_std)
    return metrics, {"median": median, "quality": quality, "aleatoric": aleatoric, "epistemic": epistemic, "total": total, "uncertainty": uncertainty}


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)),
        axis=axis,
    )


def _duration_nll(
    means: np.ndarray, log_scales: np.ndarray, target: np.ndarray,
    scale_multiplier: float,
) -> np.ndarray:
    log_target = np.log1p(target)
    adjusted_log_scale = np.clip(
        log_scales + math.log(scale_multiplier), -8, 5
    )
    scale = np.exp(adjusted_log_scale)
    log_pdf = (
        -0.5 * np.square((log_target[None, :] - means) / scale)
        - adjusted_log_scale
        - 0.5 * math.log(2.0 * math.pi)
        - np.log1p(target)[None, :]
    )
    return -(_logsumexp(log_pdf, axis=0) - math.log(means.shape[0]))


def _fit_duration_scale(
    means: np.ndarray, log_scales: np.ndarray, target: np.ndarray,
    groups: np.ndarray, observed: np.ndarray,
) -> float | None:
    mask = np.asarray(observed, dtype=bool)
    if len(set(np.asarray(groups).astype(str)[mask].tolist())) < 2:
        return None
    weights = _equal_group_row_weights(groups, mask)
    grid = np.exp(np.linspace(-2.0, 2.0, 81))
    losses = [
        float(np.sum(weights * _duration_nll(means, log_scales, target, float(scale))))
        for scale in grid
    ]
    return float(grid[int(np.argmin(losses))])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    if len(cumulative) == 0 or cumulative[-1] <= 0:
        return math.nan
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def crossfit_duration_calibration(
    means: np.ndarray, log_scales: np.ndarray, target: np.ndarray,
    current_event: np.ndarray, groups: np.ndarray, observed: np.ndarray, *,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    group_values = np.asarray(groups).astype(str)
    event = np.asarray(current_event, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    folds = _logical_group_folds(group_values)
    median = np.full(len(target), np.nan)
    lower = np.full(len(target), np.nan)
    upper = np.full(len(target), np.nan)
    model_nll = np.full(len(target), np.nan)
    baseline_median = np.full(len(target), np.nan)
    baseline_nll = np.full(len(target), np.nan)
    baseline_lower = np.full(len(target), np.nan)
    baseline_upper = np.full(len(target), np.nan)
    aleatoric = np.full(len(target), np.nan)
    epistemic = np.full(len(target), np.nan)
    fold_scales: list[float] = []
    fold_support: list[dict[str, Any]] = []
    for fold in range(CROSSFIT_FOLDS):
        training = (folds != fold) & observed
        heldout_all = folds == fold
        heldout = (folds == fold) & observed
        scale_multiplier = _fit_duration_scale(
            means, log_scales, target, group_values, training
        )
        training_events = set(event[training].tolist())
        heldout_events = set(event[heldout].tolist())
        complete_events = heldout_events <= training_events
        fold_support.append({
            "fold": fold,
            "training_groups": len(set(group_values[training].tolist())),
            "heldout_groups": len(set(group_values[heldout].tolist())),
            "heldout_current_events_present_in_training": complete_events,
            "scale_fitted": scale_multiplier is not None,
        })
        if (
            scale_multiplier is None
            or not bool(heldout_all.any())
            or not complete_events
        ):
            continue
        fold_scales.append(scale_multiplier)
        scaled_log = log_scales[:, heldout_all] + math.log(scale_multiplier)
        scales = np.exp(np.clip(scaled_log, -8, 5))
        median[heldout_all] = _mixture_lognormal_quantile(
            means[:, heldout_all], scales, 0.5
        )
        lower[heldout_all] = _mixture_lognormal_quantile(
            means[:, heldout_all], scales, (1 - INTERVAL_MASS) / 2
        )
        upper[heldout_all] = _mixture_lognormal_quantile(
            means[:, heldout_all], scales, 1 - (1 - INTERVAL_MASS) / 2
        )
        model_nll[heldout] = _duration_nll(
            means[:, heldout], log_scales[:, heldout], target[heldout],
            scale_multiplier,
        )
        member_mean = np.exp(
            np.clip(means[:, heldout_all] + 0.5 * scales**2, -30, 30)
        ) - 1.0
        member_var = (
            np.exp(np.clip(scales**2, 0, 30)) - 1
        ) * np.exp(
            np.clip(2 * means[:, heldout_all] + scales**2, -30, 30)
        )
        aleatoric[heldout_all] = member_var.mean(axis=0)
        epistemic[heldout_all] = member_mean.var(axis=0)
        training_weights = _equal_group_row_weights(group_values, training)
        for current in sorted(heldout_events):
            train_event = training & (event == current)
            heldout_event = heldout & (event == current)
            weights = training_weights[train_event]
            center = _weighted_median(target[train_event], weights)
            residual = np.abs(target[train_event] - center)
            robust_scale = max(_weighted_median(residual, weights), 1e-6)
            baseline_median[heldout_event] = center
            baseline_nll[heldout_event] = (
                np.log(2.0 * robust_scale)
                + np.abs(target[heldout_event] - center) / robust_scale
            )
            width = -robust_scale * math.log(1.0 - INTERVAL_MASS)
            baseline_lower[heldout_event] = np.maximum(center - width, 0.0)
            baseline_upper[heldout_event] = center + width
    complete = bool(
        len(fold_scales) == CROSSFIT_FOLDS
        and np.isfinite(median[observed]).all()
        and np.isfinite(baseline_median[observed]).all()
    )
    if complete:
        absolute_error = np.abs(median - target)
        baseline_error = np.abs(baseline_median - target)
        mae_gate = _additive_gain_gate(
            absolute_error, baseline_error, group_values, observed,
            samples=bootstrap_samples, role="duration_event_median_mae",
        )
        nll_gate = _additive_gain_gate(
            model_nll, baseline_nll, group_values, observed,
            samples=bootstrap_samples, role="duration_event_median_nll",
        )
        covered = (target >= lower) & (target <= upper)
        baseline_covered = (target >= baseline_lower) & (target <= baseline_upper)
        coverage_gate = _coverage_gain_gate(
            covered, baseline_covered, group_values, observed,
            samples=bootstrap_samples, role="duration_event_median_coverage",
        )
        total = aleatoric + epistemic
        relative = np.sqrt(np.maximum(total, 0.0)) / np.maximum(median, 1e-8)
        uncertainty = relative / (1.0 + relative)
        uncertainty_gate = uncertainty_performance_gate(
            uncertainty, absolute_error, group_values, observed,
            samples=bootstrap_samples, role="duration",
        )
    else:
        absolute_error = np.full(len(target), np.nan)
        covered = np.zeros(len(target), dtype=bool)
        disabled = {"passed_zero_gain_lcb": False, "group_bootstrap_gain_lcb95": math.nan}
        mae_gate = nll_gate = coverage_gate = disabled
        uncertainty = np.full(len(target), np.nan)
        uncertainty_gate = {"status": "disabled_incomplete_group_crossfit", "passed": False}
    passed = bool(
        complete and mae_gate["passed_zero_gain_lcb"]
        and nll_gate["passed_zero_gain_lcb"]
        and coverage_gate["passed_zero_gain_lcb"]
        and uncertainty_gate["passed"]
    )
    deployment_scale = _fit_duration_scale(
        means, log_scales, target, group_values, observed
    ) if passed else None
    metrics = {
        "calibration_protocol": PERFORMANCE_GATE_PROTOCOL,
        "baseline": "training_fold_current_event_median_laplace",
        "crossfit_folds": CROSSFIT_FOLDS,
        "fold_support": fold_support,
        "crossfit_complete": complete,
        "all_row_oof_inference_complete": bool(
            np.isfinite(median).all()
            and np.isfinite(aleatoric).all()
            and np.isfinite(epistemic).all()
        ),
        "fold_scale_multipliers": fold_scales,
        "deployment_scale_multiplier": (
            float(deployment_scale) if deployment_scale is not None else 1.0
        ),
        "mae_gain_gate": mae_gate,
        "nll_gain_gate": nll_gate,
        "coverage_gain_gate": coverage_gate,
        "equal_group_central_90_interval_coverage": float(
            _group_mean_vector(covered, group_values, observed)[1].mean()
        ) if complete else None,
        "equal_group_baseline_central_90_interval_coverage": float(
            _group_mean_vector(
                baseline_covered, group_values, observed
            )[1].mean()
        ) if complete else None,
        "uncertainty_gate": uncertainty_gate,
        "performance_gate_passed": passed,
        "metric_weighting": "equal_logical_group",
        "observed_rows": int(observed.sum()),
    }
    return metrics, {
        "median": median,
        "quality": np.where(observed, covered.astype(float), np.nan),
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": aleatoric + epistemic,
        "uncertainty": uncertainty,
        "observation_mask": observed,
    }


def object_metrics(means: np.ndarray, log_scales: np.ndarray, target: np.ndarray, observed: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    variances = np.exp(np.clip(2 * log_scales, -20, 20))
    ensemble_mean = means.mean(axis=0)
    aleatoric = variances.mean(axis=0)
    epistemic = means.var(axis=0)
    total = aleatoric + epistemic
    z90 = 1.6448536269514722
    lower, upper = ensemble_mean - z90 * np.sqrt(total), ensemble_mean + z90 * np.sqrt(total)
    covered = (target >= lower) & (target <= upper)
    count = int(observed.sum())
    centered = target[observed] - np.median(target[observed], axis=0) if count else np.empty((0, target.shape[1]))
    robust_scale = float(np.median(np.linalg.norm(centered, axis=1))) if count else 0.0
    robust_scale = max(robust_scale, 1e-8)
    total_std = np.sqrt(np.maximum(total.mean(axis=1), 0))
    uncertainty = total_std / (robust_scale + total_std)
    metrics = {
        "central_90_marginal_coverage_observed": float(covered[observed].mean()) if count else None,
        "central_90_joint_coverage_observed": float(covered[observed].all(axis=1).mean()) if count else None,
        "observed_count": count,
        "missing_count": int((~observed).sum()),
        "robust_target_scale": robust_scale,
        "mean_aleatoric_variance_per_dimension": aleatoric.mean(axis=0).tolist(),
        "mean_epistemic_variance_per_dimension": epistemic.mean(axis=0).tolist(),
        "mean_total_variance_per_dimension": total.mean(axis=0).tolist(),
    }
    quality = np.where(observed, covered.all(axis=1).astype(float), np.nan)
    return metrics, {"mean": ensemble_mean, "quality": quality, "aleatoric": aleatoric, "epistemic": epistemic, "total": total, "uncertainty": uncertainty}


def _object_nll(
    means: np.ndarray, log_scales: np.ndarray, target: np.ndarray,
    scale_multiplier: float,
) -> np.ndarray:
    adjusted = np.clip(log_scales + math.log(scale_multiplier), -10, 10)
    scale = np.exp(adjusted)
    log_pdf = (
        -0.5 * np.square((target[None, :, :] - means) / scale)
        - adjusted
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(axis=2)
    return -(_logsumexp(log_pdf, axis=0) - math.log(means.shape[0]))


def _fit_object_scale(
    means: np.ndarray, log_scales: np.ndarray, target: np.ndarray,
    groups: np.ndarray, observed: np.ndarray,
) -> float | None:
    mask = np.asarray(observed, dtype=bool)
    if len(set(np.asarray(groups).astype(str)[mask].tolist())) < 2:
        return None
    weights = _equal_group_row_weights(groups, mask)
    grid = np.exp(np.linspace(-2.0, 2.0, 81))
    losses = [
        float(np.sum(weights * _object_nll(means, log_scales, target, float(scale))))
        for scale in grid
    ]
    return float(grid[int(np.argmin(losses))])


def crossfit_object_calibration(
    means: np.ndarray, log_scales: np.ndarray, target: np.ndarray,
    groups: np.ndarray, observed: np.ndarray, *, bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    group_values = np.asarray(groups).astype(str)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    folds = _logical_group_folds(group_values)
    prediction = means.mean(axis=0)
    robust_baseline = np.full_like(target, np.nan)
    model_nll = np.full(len(target), np.nan)
    aleatoric = np.full_like(target, np.nan)
    epistemic = means.var(axis=0)
    covered = np.zeros_like(target, dtype=bool)
    oof_uncertainty = np.full(len(target), np.nan)
    fold_scales: list[float] = []
    fold_uncertainty_robust_scales: list[float] = []
    fold_support: list[dict[str, Any]] = []
    for fold in range(CROSSFIT_FOLDS):
        training = (folds != fold) & observed
        heldout_all = folds == fold
        heldout = (folds == fold) & observed
        scale_multiplier = _fit_object_scale(
            means, log_scales, target, group_values, training
        )
        fold_support.append({
            "fold": fold,
            "training_groups": len(set(group_values[training].tolist())),
            "heldout_groups": len(set(group_values[heldout].tolist())),
            "scale_fitted": scale_multiplier is not None,
        })
        if scale_multiplier is None or not bool(heldout_all.any()):
            continue
        fold_scales.append(scale_multiplier)
        training_weights = _equal_group_row_weights(group_values, training)
        training_center = np.asarray(
            [
                _weighted_median(
                    target[training, dimension], training_weights[training]
                )
                for dimension in range(target.shape[1])
            ],
            dtype=np.float64,
        )
        training_error = np.linalg.norm(
            target - training_center[None, :], axis=1
        )
        _training_names, training_group_error = _group_mean_vector(
            training_error, group_values, training
        )
        fold_robust_scale = max(
            float(np.median(training_group_error)), 1e-8
        )
        fold_uncertainty_robust_scales.append(fold_robust_scale)
        for dimension in range(target.shape[1]):
            robust_baseline[heldout, dimension] = _weighted_median(
                target[training, dimension], training_weights[training]
            )
        model_nll[heldout] = _object_nll(
            means[:, heldout], log_scales[:, heldout], target[heldout],
            scale_multiplier,
        )
        variances = np.exp(
            np.clip(
                2.0
                * (
                    log_scales[:, heldout_all]
                    + math.log(scale_multiplier)
                ),
                -20,
                20,
            )
        )
        aleatoric[heldout_all] = variances.mean(axis=0)
        total_all = aleatoric[heldout_all] + epistemic[heldout_all]
        total_std_all = np.sqrt(
            np.maximum(total_all.mean(axis=1), 0.0)
        )
        oof_uncertainty[heldout_all] = total_std_all / (
            fold_robust_scale + total_std_all
        )
        total = aleatoric[heldout] + epistemic[heldout]
        radius = 1.6448536269514722 * np.sqrt(np.maximum(total, 0.0))
        covered[heldout] = (
            (target[heldout] >= prediction[heldout] - radius)
            & (target[heldout] <= prediction[heldout] + radius)
        )
    complete = bool(
        len(fold_scales) == CROSSFIT_FOLDS
        and np.isfinite(robust_baseline[observed]).all()
        and np.isfinite(model_nll[observed]).all()
    )
    model_error = np.linalg.norm(prediction - target, axis=1)
    zero_error = np.linalg.norm(target, axis=1)
    robust_error = np.linalg.norm(robust_baseline - target, axis=1)
    deployment_uncertainty_robust_scale = 1.0
    if complete:
        zero_gate = _additive_gain_gate(
            model_error, zero_error, group_values, observed,
            samples=bootstrap_samples, role="object_zero_l2",
        )
        robust_gate = _additive_gain_gate(
            model_error, robust_error, group_values, observed,
            samples=bootstrap_samples, role="object_robust_median_l2",
        )
        total_variance = aleatoric + epistemic
        total_std = np.sqrt(np.maximum(total_variance.mean(axis=1), 0.0))
        _robust_names, robust_group_error = _group_mean_vector(
            robust_error, group_values, observed
        )
        robust_scale = max(
            float(np.median(robust_group_error)), 1e-8
        )
        deployment_uncertainty_robust_scale = robust_scale
        uncertainty = oof_uncertainty
        uncertainty_gate = uncertainty_performance_gate(
            uncertainty, model_error, group_values, observed,
            samples=bootstrap_samples, role="object_effect",
        )
    else:
        zero_gate = robust_gate = {
            "passed_zero_gain_lcb": False,
            "group_bootstrap_gain_lcb95": math.nan,
        }
        uncertainty = np.full(len(target), np.nan)
        uncertainty_gate = {
            "status": "disabled_incomplete_group_crossfit", "passed": False
        }
    passed = bool(
        complete and zero_gate["passed_zero_gain_lcb"]
        and robust_gate["passed_zero_gain_lcb"]
        and uncertainty_gate["passed"]
    )
    deployment_scale = _fit_object_scale(
        means, log_scales, target, group_values, observed
    ) if passed else None
    metrics = {
        "calibration_protocol": PERFORMANCE_GATE_PROTOCOL,
        "baselines": ["zero_delta", "training_fold_coordinate_robust_median"],
        "crossfit_folds": CROSSFIT_FOLDS,
        "fold_support": fold_support,
        "crossfit_complete": complete,
        "all_row_oof_inference_complete": bool(
            np.isfinite(aleatoric).all()
            and np.isfinite(epistemic).all()
            and np.isfinite(oof_uncertainty).all()
        ),
        "fold_scale_multipliers": fold_scales,
        "fold_uncertainty_robust_scales_m": fold_uncertainty_robust_scales,
        "deployment_scale_multiplier": (
            float(deployment_scale) if deployment_scale is not None else 1.0
        ),
        "deployment_object_error_robust_scale_m": (
            float(deployment_uncertainty_robust_scale) if passed else 1.0
        ),
        "zero_baseline_gain_gate": zero_gate,
        "robust_median_gain_gate": robust_gate,
        "uncertainty_gate": uncertainty_gate,
        "performance_gate_passed": passed,
        "metric_weighting": "equal_logical_group",
        "observed_rows": int(observed.sum()),
        "equal_group_model_nll": float(
            _group_mean_vector(model_nll, group_values, observed)[1].mean()
        ) if complete else None,
        "central_90_marginal_coverage": float(covered[observed].mean())
        if complete else None,
    }
    return metrics, {
        "mean": prediction,
        "quality": np.where(observed, covered.all(axis=1).astype(float), np.nan),
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": aleatoric + epistemic,
        "uncertainty": uncertainty,
        "observation_mask": observed,
    }


def deployment_root_structured_uncertainty(
    *,
    post_logits: np.ndarray,
    next_logits: np.ndarray,
    success_logits: np.ndarray,
    recovery_logits: np.ndarray,
    duration_means: np.ndarray,
    duration_log_scales: np.ndarray,
    object_means: np.ndarray,
    object_log_scales: np.ndarray,
    labels: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recompute the exact online pre-action uncertainty with full-refit params."""

    root = np.asarray(labels["root_candidate"], dtype=bool)
    current_event = np.asarray(labels["current_event"], dtype=np.int64)
    if not bool(root.any()) or bool((current_event[root] != 0).any()):
        raise CalibrationError("formal root candidates must be pre-action e0 rows")
    post_temperature = float(metrics["post_event"]["deployment_temperature"])
    next_temperature = float(metrics["next_event"]["deployment_temperature"])
    success_temperature = float(metrics["success"]["deployment_temperature"])
    recovery_temperature = float(
        metrics["conditional_recovery"]["deployment_temperature"]
    )
    duration_multiplier = float(
        metrics["duration_lognormal_mixture"]["deployment_scale_multiplier"]
    )
    object_multiplier = float(
        metrics["object_total_variance"]["deployment_scale_multiplier"]
    )
    object_robust_scale = float(
        metrics["object_total_variance"][
            "deployment_object_error_robust_scale_m"
        ]
    )
    numeric = (
        post_temperature, next_temperature, success_temperature,
        recovery_temperature, duration_multiplier, object_multiplier,
        object_robust_scale,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise CalibrationError("deployment uncertainty parameter is invalid")

    duration_deployment_log_scales = np.clip(
        duration_log_scales + math.log(duration_multiplier), -8.0, 5.0
    )
    object_deployment_log_scales = np.clip(
        object_log_scales + math.log(object_multiplier), -10.0, 10.0
    )
    try:
        head_uncertainty = deployment_uncertainty_v1.deployment_uncertainty_components(
            predictions={
                "post_event_logits": post_logits,
                "next_event_logits": next_logits,
                "success_logit": success_logits,
                "recovery_logit": recovery_logits,
                "duration_log_mean": duration_means,
                "duration_log_scale": duration_deployment_log_scales,
                "object_mean": object_means,
                "object_log_scale": object_deployment_log_scales,
            },
            parameters={
                "post_event_temperature": post_temperature,
                "next_event_temperature": next_temperature,
                "success_temperature": success_temperature,
                "conditional_recovery_temperature": recovery_temperature,
                "object_error_robust_scale_m": object_robust_scale,
            },
        )
        shared_combined = (
            deployment_uncertainty_v1.combine_initial_e0_root_uncertainty(
                head_uncertainty
            )
        )
        combined = np.ones(len(root), dtype=np.float64)
        combined[root] = shared_combined[root]
    except deployment_uncertainty_v1.DeploymentUncertaintyError as exc:
        raise CalibrationError(str(exc)) from exc
    implementation_path = Path(deployment_uncertainty_v1.__file__).resolve()
    contract: dict[str, Any] = {
        "format": deployment_uncertainty_v1.FORMAT,
        "status": "frozen_full_formal190_refit_parameters_online_reproducible",
        "shared_implementation_path": str(implementation_path),
        "shared_implementation_file_sha256": file_sha256(implementation_path),
        "performance_gate_uncertainty_source": "five_fold_group_oof_predictions",
        "selector_uncertainty_source": "full_formal190_refit_deployment_parameters",
        "included_root_heads": list(ROOT_STRUCTURED_UNCERTAINTY_HEADS),
        "excluded_root_heads": ["recovery"],
        "root_structured_uncertainty_head_count": len(
            ROOT_STRUCTURED_UNCERTAINTY_HEADS
        ),
        "root_recovery_uncertainty_policy": ROOT_RECOVERY_UNCERTAINTY_POLICY,
        "post_event_temperature": post_temperature,
        "next_event_temperature": next_temperature,
        "success_temperature": success_temperature,
        "conditional_recovery_temperature": recovery_temperature,
        "duration_scale_multiplier": duration_multiplier,
        "object_scale_multiplier": object_multiplier,
        "object_error_robust_scale_m": object_robust_scale,
        "object_prediction_space": "physical_xyz_m",
        "duration_and_object_log_scale_multiplier_application": "add_log_multiplier_exactly_once",
        "shared_function_input_scale_state": "duration_and_object_deployment_multiplier_already_applied_exactly_once",
        "uncertainty_range": [0.0, 1.0],
        "formal_root_row_count": int(root.sum()),
        "evaluation400_outcomes_read": False,
    }
    contract["deployment_uncertainty_contract_sha256"] = canonical_sha256(contract)
    return combined, contract


def outer_nested_root_calibration_inputs(
    *,
    post_logits: np.ndarray,
    next_logits: np.ndarray,
    success_logits: np.ndarray,
    duration_means: np.ndarray,
    duration_log_scales: np.ndarray,
    object_means: np.ndarray,
    object_log_scales: np.ndarray,
    labels: Mapping[str, np.ndarray],
    all_six_head_gate_passed: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one root-calibration parameter set per outer training partition.

    For outer fold k every temperature/scale/robust normalization and every
    threshold-selection quality value is derived from groups != k.  The same
    frozen parameter set then produces the heldout fold's uncertainty once.
    """

    groups = np.asarray(labels["group_id"]).astype(str)
    folds = _logical_group_folds(groups)
    row_count = len(groups)
    root = np.asarray(labels["root_candidate"], dtype=bool)
    post_target = np.asarray(labels["post_event"], dtype=np.int64)
    next_target = np.asarray(labels["next_event"], dtype=np.int64)
    success_target = np.asarray(labels["success"], dtype=np.int64)
    duration_target = np.asarray(labels["duration"], dtype=np.float64)
    duration_observed = np.asarray(labels["duration_observed"], dtype=bool)
    object_target = np.asarray(labels["object_target"], dtype=np.float64)
    object_observed = np.asarray(labels["object_observed"], dtype=bool)
    classes = int(post_logits.shape[-1])
    outer_uncertainty = np.ones(
        (CROSSFIT_FOLDS, row_count), dtype=np.float64
    )
    outer_quality = np.full(
        (CROSSFIT_FOLDS, row_count), np.nan, dtype=np.float64
    )
    fold_parameters: list[dict[str, Any]] = []
    for outer_fold in range(CROSSFIT_FOLDS):
        training_groups = folds != outer_fold
        heldout_groups = folds == outer_fold
        training_root = training_groups & root
        post_temperature = _fit_multiclass_temperature(
            post_logits,
            post_target,
            groups,
            training_groups,
            required_classes=list(range(classes)),
        )
        next_temperature = _fit_multiclass_temperature(
            next_logits,
            next_target,
            groups,
            training_groups & duration_observed,
            required_classes=list(range(1, classes)),
        )
        success_temperature = _fit_binary_temperature_grouped(
            success_logits,
            success_target,
            groups,
            training_groups,
        )
        duration_multiplier = _fit_duration_scale(
            duration_means,
            duration_log_scales,
            duration_target,
            groups,
            training_groups & duration_observed,
        )
        object_multiplier = _fit_object_scale(
            object_means,
            object_log_scales,
            object_target,
            groups,
            training_groups & object_observed,
        )
        object_training = training_groups & object_observed
        object_weights = _equal_group_row_weights(groups, object_training)
        object_center = np.asarray(
            [
                _weighted_median(
                    object_target[object_training, dimension],
                    object_weights[object_training],
                )
                for dimension in range(object_target.shape[1])
            ],
            dtype=np.float64,
        )
        object_error = np.linalg.norm(
            object_target - object_center[None, :], axis=1
        )
        _object_names, object_group_error = _group_mean_vector(
            object_error, groups, object_training
        )
        object_robust_scale = (
            max(float(np.median(object_group_error)), 1e-8)
            if len(object_group_error)
            else None
        )
        parameters_complete = bool(
            all_six_head_gate_passed
            and post_temperature is not None
            and next_temperature is not None
            and success_temperature is not None
            and duration_multiplier is not None
            and object_multiplier is not None
            and object_robust_scale is not None
        )
        parameter_row: dict[str, Any] = {
            "outer_fold": outer_fold,
            "training_logical_group_count": len(
                set(groups[training_groups].tolist())
            ),
            "heldout_logical_group_count": len(
                set(groups[heldout_groups].tolist())
            ),
            "training_logical_group_ids_sha256": canonical_sha256(
                sorted(set(groups[training_groups].tolist()))
            ),
            "heldout_logical_group_ids_sha256": canonical_sha256(
                sorted(set(groups[heldout_groups].tolist()))
            ),
            "post_event_temperature": post_temperature,
            "next_event_temperature": next_temperature,
            "success_temperature": success_temperature,
            "duration_scale_multiplier": duration_multiplier,
            "object_scale_multiplier": object_multiplier,
            "object_uncertainty_robust_scale_m": object_robust_scale,
            "parameters_complete": parameters_complete,
            "heldout_labels_used_for_parameters_or_training_quality": False,
        }
        parameter_row["fold_parameter_sha256"] = canonical_sha256(parameter_row)
        fold_parameters.append(parameter_row)
        if not parameters_complete:
            continue
        duration_scaled = np.asarray(duration_log_scales, dtype=np.float64) + (
            math.log(float(duration_multiplier))
        )
        object_scaled = np.asarray(object_log_scales, dtype=np.float64) + (
            math.log(float(object_multiplier))
        )
        components = {
            "post_event": deployment_uncertainty_v1.event_total_uncertainty(
                post_logits, post_temperature
            ),
            "next_event": deployment_uncertainty_v1.event_total_uncertainty(
                next_logits, next_temperature
            ),
            "success": deployment_uncertainty_v1.binary_total_uncertainty(
                success_logits, success_temperature
            ),
            "duration": (
                deployment_uncertainty_v1.shifted_lognormal_duration_uncertainty(
                    duration_means, duration_scaled
                )
            ),
            "object_effect": deployment_uncertainty_v1.physical_object_uncertainty(
                object_means, object_scaled, object_robust_scale
            ),
        }
        structured = np.stack(
            [components[name] for name in ROOT_STRUCTURED_UNCERTAINTY_HEADS],
            axis=0,
        ).mean(axis=0)
        if (
            not np.isfinite(structured).all()
            or bool(((structured < 0.0) | (structured > 1.0)).any())
        ):
            continue
        outer_uncertainty[outer_fold] = structured

        post_probability = _softmax(post_logits / post_temperature).mean(axis=0)
        next_probability = _softmax(next_logits / next_temperature).mean(axis=0)
        success_probability = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        success_logits / success_temperature, -40.0, 40.0
                    )
                )
            )
        ).mean(axis=0)
        duration_scales = np.exp(np.clip(duration_scaled, -8.0, 5.0))
        duration_lower = _mixture_lognormal_quantile(
            duration_means,
            duration_scales,
            (1.0 - INTERVAL_MASS) / 2.0,
        )
        duration_upper = _mixture_lognormal_quantile(
            duration_means,
            duration_scales,
            1.0 - (1.0 - INTERVAL_MASS) / 2.0,
        )
        object_prediction = object_means.mean(axis=0)
        object_total = (
            np.exp(np.clip(2.0 * object_scaled, -20.0, 20.0)).mean(axis=0)
            + object_means.var(axis=0)
        )
        object_radius = 1.6448536269514722 * np.sqrt(
            np.maximum(object_total, 0.0)
        )
        quality_components = np.full((5, row_count), np.nan)
        quality_components[0, training_root] = (
            post_probability.argmax(axis=1)[training_root]
            == post_target[training_root]
        ).astype(float)
        next_quality_mask = training_root & duration_observed
        quality_components[1, next_quality_mask] = (
            next_probability.argmax(axis=1)[next_quality_mask]
            == next_target[next_quality_mask]
        ).astype(float)
        quality_components[2, training_root] = (
            (success_probability[training_root] >= 0.5)
            == success_target[training_root]
        ).astype(float)
        duration_quality_mask = training_root & duration_observed
        quality_components[3, duration_quality_mask] = (
            (duration_target[duration_quality_mask]
             >= duration_lower[duration_quality_mask])
            & (duration_target[duration_quality_mask]
               <= duration_upper[duration_quality_mask])
        ).astype(float)
        object_quality_mask = training_root & object_observed
        quality_components[4, object_quality_mask] = (
            (
                object_target[object_quality_mask]
                >= object_prediction[object_quality_mask]
                - object_radius[object_quality_mask]
            )
            & (
                object_target[object_quality_mask]
                <= object_prediction[object_quality_mask]
                + object_radius[object_quality_mask]
            )
        ).all(axis=1).astype(float)
        valid_count = np.isfinite(quality_components).sum(axis=0)
        outer_quality[outer_fold] = np.divide(
            np.nansum(quality_components, axis=0),
            valid_count,
            out=np.full(row_count, np.nan),
            where=valid_count > 0,
        )
        outer_quality[outer_fold, heldout_groups] = np.nan

    complete = bool(
        all(row["parameters_complete"] for row in fold_parameters)
        and np.isfinite(outer_uncertainty).all()
        and all(
            np.isfinite(
                outer_quality[fold, (folds != fold) & root]
            ).all()
            for fold in range(CROSSFIT_FOLDS)
        )
    )
    contract: dict[str, Any] = {
        "format": "etsf_smolvla_piper_formal190_complete_root_outer_nesting_v1",
        "status": (
            "complete_outer_heldout_isolation"
            if complete
            else "incomplete_fail_closed"
        ),
        "outer_crossfit_folds": CROSSFIT_FOLDS,
        "fold_assignment": "lexicographic_logical_group_index_modulo_five",
        "fold_parameters": fold_parameters,
        "outer_heldout_labels_used_for_any_parameter_or_selection": False,
        "same_outer_training_parameters_used_for_training_and_heldout_inference": True,
        "next_duration_observation_masks_used_only_for_parameter_fitting_and_quality": True,
        "object_robust_scale_fit_on_outer_training_groups_only": True,
        "complete_root_pipeline_outer_nesting": complete,
        "upstream_predictions_already_group_crossfit": complete,
        "evaluation400_outcomes_read": False,
    }
    contract["root_outer_nesting_contract_sha256"] = canonical_sha256(contract)
    return outer_uncertainty, outer_quality, contract


def _group_counts(labels: np.ndarray, groups: np.ndarray, classes: Sequence[int]) -> dict[str, int]:
    return {str(label): len(set(groups[labels == label].tolist())) for label in classes}


def head_support(labels: Mapping[str, np.ndarray], event_classes: int | None = None) -> dict[str, Any]:
    groups = labels["group_id"]
    class_count = event_classes or max(
        int(labels["post_event"].max()), int(labels["next_event"].max())
    ) + 1
    post_classes = list(range(class_count))
    # e0 is the structural step-0 initial state and cannot be an observed
    # future milestone.  Censored collector self-loops are also not labels.
    next_classes = list(range(1, class_count))
    post_counts = _group_counts(labels["post_event"], groups, post_classes)
    duration_observed = labels["duration_observed"].astype(bool)
    next_counts = _group_counts(
        labels["next_event"][duration_observed],
        groups[duration_observed],
        next_classes,
    )
    success_counts = _group_counts(labels["success"], groups, [1, 0])
    recovery_observed = labels["recovery_observed"].astype(bool)
    recovery_counts = _group_counts(
        labels["recovery"][recovery_observed],
        groups[recovery_observed],
        [1, 0],
    )
    object_observed = labels["object_observed"].astype(bool)
    object_nonzero = np.linalg.norm(labels["object_target"], axis=1) > 1e-6
    raw = {
        "post_event": (min(post_counts.values()), min(post_counts.values()), "minimum_validation_groups_per_event_class"),
        "next_event": (
            min(next_counts.values()),
            min(next_counts.values()),
            "minimum_duration_observed_validation_groups_per_structurally_reachable_event_class_e12_e3_e4_eK",
        ),
        "duration": (len(set(groups[duration_observed].tolist())), len(set(groups[~duration_observed].tolist())), "validation_observed_and_censored_groups"),
        "success": (success_counts["1"], success_counts["0"], "validation_positive_and_negative_groups"),
        "recovery": (
            recovery_counts["1"],
            recovery_counts["0"],
            "validation_positive_and_negative_groups_conditioned_on_observed_operational_regress",
        ),
        "object_effect": (len(set(groups[object_observed & object_nonzero].tolist())), len(set(groups[object_observed & ~object_nonzero].tolist())), "validation_nonzero_and_near_zero_effect_groups"),
    }
    heads = {}
    for name, (positive, negative, source) in raw.items():
        minimum = MINIMUM_HEAD_GROUPS_PER_SIDE[name]
        heads[name] = {
            "enabled_for_primary": positive >= minimum and negative >= minimum,
            "support_threshold_met": positive >= minimum and negative >= minimum,
            "performance_gate_passed": False,
            "uncertainty_gate_passed": False,
            "independent_positive_or_observed_groups": positive,
            "independent_negative_or_censored_groups": negative,
            "minimum_required_per_side": minimum,
            "support_source": source,
        }
    receipt: dict[str, Any] = {
        "format": HEAD_SUPPORT_FORMAT,
        "status": "frozen_from_training_and_validation_only_before_paired_development",
        "heads": heads,
        "paired_development_outcomes_read": False,
        "sealed_evaluation_reserve_outcomes_read": False,
    }
    receipt["head_support_sha256"] = canonical_sha256(receipt)
    return receipt


def _root_harmful_rate_interval(
    harmful: np.ndarray, *, samples: int, role: str,
) -> tuple[float, float]:
    values = np.asarray(harmful, dtype=np.float64)
    if len(values) < 2 or type(samples) is not int or samples < 100:
        return math.nan, math.nan
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + int.from_bytes(hashlib.sha256(role.encode()).digest()[:2], "big")
    )
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    rates = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(rates, 1.0 - BOOTSTRAP_ALPHA))


def _root_bootstrap_draws(group_count: int, samples: int) -> np.ndarray:
    """Return the preregistered draw matrix shared by every root candidate.

    The same seed and the same logical-group ordering are used for gain and
    harmful-rate inference.  Candidate-specific seeds would make grid ranking
    depend on avoidable Monte Carlo noise.
    """

    if (
        type(group_count) is not int
        or group_count < 2
        or group_count > np.iinfo(np.uint16).max
        or type(samples) is not int
        or samples < 100
    ):
        raise CalibrationError("root bootstrap draw dimensions are invalid")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(
        0,
        group_count,
        size=(samples, group_count),
        dtype=np.uint16,
    )


def _root_bootstrap_draw_descriptor(draws: np.ndarray) -> dict[str, Any]:
    values = np.asarray(draws)
    if values.ndim != 2 or values.dtype != np.uint16:
        raise CalibrationError("root bootstrap draws changed dtype or shape")
    little_endian = np.ascontiguousarray(values.astype("<u2", copy=False))
    descriptor: dict[str, Any] = {
        "algorithm": "numpy_pcg64_fixed_seed_logical_group_indices_v1",
        "seed": BOOTSTRAP_SEED,
        "dtype": "little_endian_uint16",
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "draws_sha256": hashlib.sha256(little_endian.tobytes(order="C")).hexdigest(),
    }
    descriptor["descriptor_sha256"] = canonical_sha256(descriptor)
    return descriptor


def _root_decision_arrays(
    group_records: Sequence[Mapping[str, Any]],
    *,
    margin: float,
    maximum_pair_uncertainty: float,
    maximum_global_candidate_uncertainty: float,
    uncertainty_namespace: str,
) -> dict[str, np.ndarray]:
    pair_field = (
        f"best_composite_rank_candidate_{uncertainty_namespace}"
        "structured_uncertainty"
    )
    candidate_field = (
        f"best_composite_rank_candidate_{uncertainty_namespace}"
        "global_uncertainty"
    )
    changed = np.asarray(
        [
            row["best_group_relative_composite_rank_score_margin"] > margin
            and row[pair_field]
            <= maximum_pair_uncertainty
            and row[candidate_field]
            <= maximum_global_candidate_uncertainty
            for row in group_records
        ],
        dtype=bool,
    )
    baseline_outcome = np.asarray(
        [row["baseline_final_success"] for row in group_records],
        dtype=np.int64,
    )
    alternative_outcome = np.asarray(
        [
            next(
                candidate["final_success"]
                for candidate in row["candidates"]
                if candidate["candidate_index"]
                == row["best_composite_rank_candidate_index"]
            )
            for row in group_records
        ],
        dtype=np.int64,
    )
    selected_outcome = np.where(changed, alternative_outcome, baseline_outcome)
    paired_gain = selected_outcome - baseline_outcome
    helpful = changed & (paired_gain > 0)
    harmful = changed & (paired_gain < 0)
    return {
        "changed": changed,
        "baseline_outcome": baseline_outcome,
        "alternative_outcome": alternative_outcome,
        "selected_outcome": selected_outcome,
        "paired_gain": paired_gain,
        "helpful": helpful,
        "harmful": harmful,
        "discordant": helpful | harmful,
    }


def _root_bootstrap_evidence(
    decisions: Mapping[str, np.ndarray], draws: np.ndarray,
) -> dict[str, float]:
    paired_gain = np.asarray(decisions["paired_gain"], dtype=np.float64)
    changed = np.asarray(decisions["changed"], dtype=bool)
    harmful = np.asarray(decisions["harmful"], dtype=bool)
    if draws.shape[1] != len(paired_gain):
        raise CalibrationError("root bootstrap draw width changed")
    sampled_gain = paired_gain[draws].mean(axis=1)
    sampled_changed = changed[draws].sum(axis=1)
    sampled_harmful = harmful[draws].sum(axis=1)
    valid_harm = sampled_changed > 0
    harmful_bootstrap = np.divide(
        sampled_harmful[valid_harm], sampled_changed[valid_harm]
    )
    harmful_rate = (
        float(harmful.sum() / changed.sum()) if bool(changed.any()) else math.nan
    )
    return {
        "paired_gain": float(paired_gain.mean()),
        "paired_gain_group_bootstrap_lcb95": float(
            np.quantile(sampled_gain, BOOTSTRAP_ALPHA)
        ),
        "paired_gain_group_bootstrap_ucb95": float(
            np.quantile(sampled_gain, 1.0 - BOOTSTRAP_ALPHA)
        ),
        "harmful_rate_among_executed_changes": harmful_rate,
        "harmful_rate_group_bootstrap_ucb95": (
            float(np.quantile(harmful_bootstrap, 1.0 - BOOTSTRAP_ALPHA))
            if len(harmful_bootstrap)
            else math.nan
        ),
    }


def _root_candidate_grid(
    group_records: Sequence[Mapping[str, Any]],
    record_folds: np.ndarray,
    *,
    support_fold_ids: Sequence[int],
    upstream_six_head_gate_passed: bool,
    global_abstain_threshold_enabled: bool,
    maximum_global_candidate_uncertainty: float,
    bootstrap_samples: int,
    uncertainty_namespace: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    draws = _root_bootstrap_draws(len(group_records), bootstrap_samples)
    draw_descriptor = _root_bootstrap_draw_descriptor(draws)
    required_changed = MINIMUM_ROOT_CHANGED_GROUPS_PER_FOLD * len(
        support_fold_ids
    )
    required_discordant = MINIMUM_ROOT_DISCORDANT_GROUPS_PER_FOLD * len(
        support_fold_ids
    )
    candidates: list[dict[str, Any]] = []
    for margin in ROOT_MARGIN_GRID:
        for maximum_uncertainty in ROOT_UNCERTAINTY_GRID:
            decisions = _root_decision_arrays(
                group_records,
                margin=margin,
                maximum_pair_uncertainty=maximum_uncertainty,
                maximum_global_candidate_uncertainty=(
                    maximum_global_candidate_uncertainty
                ),
                uncertainty_namespace=uncertainty_namespace,
            )
            bootstrap = _root_bootstrap_evidence(decisions, draws)
            fold_support = []
            for fold in support_fold_ids:
                in_fold = record_folds == fold
                changed_count = int((decisions["changed"] & in_fold).sum())
                discordant_count = int(
                    (decisions["discordant"] & in_fold).sum()
                )
                fold_support.append({
                    "fold": int(fold),
                    "logical_groups": int(in_fold.sum()),
                    "changed_groups": changed_count,
                    "discordant_groups": discordant_count,
                    "minimum_changed_groups": (
                        MINIMUM_ROOT_CHANGED_GROUPS_PER_FOLD
                    ),
                    "minimum_discordant_groups": (
                        MINIMUM_ROOT_DISCORDANT_GROUPS_PER_FOLD
                    ),
                    "support_passed": bool(
                        changed_count >= MINIMUM_ROOT_CHANGED_GROUPS_PER_FOLD
                        and discordant_count
                        >= MINIMUM_ROOT_DISCORDANT_GROUPS_PER_FOLD
                    ),
                })
            selected_outcome = decisions["selected_outcome"]
            baseline_outcome = decisions["baseline_outcome"]
            row: dict[str, Any] = {
                "minimum_group_relative_composite_rank_score_margin": margin,
                "maximum_structured_pair_uncertainty": maximum_uncertainty,
                "maximum_global_candidate_uncertainty": (
                    maximum_global_candidate_uncertainty
                ),
                "changed_group_count": int(decisions["changed"].sum()),
                "change_coverage": float(decisions["changed"].mean()),
                "helpful_group_count": int(decisions["helpful"].sum()),
                "harmful_group_count": int(decisions["harmful"].sum()),
                "discordant_group_count": int(
                    decisions["discordant"].sum()
                ),
                "selected_success_count": int(selected_outcome.sum()),
                "selected_success_rate": float(selected_outcome.mean()),
                "baseline_success_count": int(baseline_outcome.sum()),
                "baseline_success_rate": float(baseline_outcome.mean()),
                **bootstrap,
                "fold_support": fold_support,
            }
            row["eligible"] = bool(
                upstream_six_head_gate_passed
                and global_abstain_threshold_enabled
                and row["changed_group_count"] >= required_changed
                and row["discordant_group_count"] >= required_discordant
                and all(item["support_passed"] for item in fold_support)
                and math.isfinite(
                    row["paired_gain_group_bootstrap_lcb95"]
                )
                and row["paired_gain_group_bootstrap_lcb95"] > 0.0
                and math.isfinite(
                    row["harmful_rate_group_bootstrap_ucb95"]
                )
                and row["harmful_rate_group_bootstrap_ucb95"]
                <= MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES
            )
            candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["paired_gain_group_bootstrap_lcb95"],
                row["selected_success_rate"],
                -row["harmful_rate_group_bootstrap_ucb95"],
                row["change_coverage"],
                row["minimum_group_relative_composite_rank_score_margin"],
                -row["maximum_structured_pair_uncertainty"],
            ),
        )
        if eligible
        else None
    )
    return candidates, selected, draw_descriptor


def _root_selected_decisions(
    group_records: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    *,
    outer_fold: int | None = None,
    uncertainty_namespace: str,
) -> list[dict[str, Any]]:
    pair_field = (
        f"best_composite_rank_candidate_{uncertainty_namespace}"
        "structured_uncertainty"
    )
    candidate_field = (
        f"best_composite_rank_candidate_{uncertainty_namespace}"
        "global_uncertainty"
    )
    result = []
    for row in group_records:
        changed = bool(
            selected is not None
            and row["best_group_relative_composite_rank_score_margin"]
            > selected["minimum_group_relative_composite_rank_score_margin"]
            and row[pair_field]
            <= selected["maximum_structured_pair_uncertainty"]
            and row[candidate_field]
            <= selected["maximum_global_candidate_uncertainty"]
        )
        alternative = next(
            candidate
            for candidate in row["candidates"]
            if candidate["candidate_index"]
            == row["best_composite_rank_candidate_index"]
        )
        baseline_outcome = int(row["baseline_final_success"])
        selected_outcome = (
            int(alternative["final_success"]) if changed else baseline_outcome
        )
        decision: dict[str, Any] = {
            "logical_group_id": row["logical_group_id"],
            "selection_available": selected is not None,
            "changed_from_baseline": changed,
            "selected_candidate_index": (
                row["best_composite_rank_candidate_index"]
                if changed
                else row["baseline_candidate_index"]
            ),
            "baseline_final_success": baseline_outcome,
            "selected_final_success": selected_outcome,
            "paired_gain": selected_outcome - baseline_outcome,
        }
        if outer_fold is not None:
            decision["outer_fold"] = outer_fold
        result.append(decision)
    return result


def calibrate_root_group_ranker(
    member_source_contract_rank_scores: np.ndarray,
    member_source_contract_base_rank_scores: np.ndarray,
    member_source_action_rank_residuals: np.ndarray,
    deployment_structured_uncertainty: np.ndarray,
    labels: Mapping[str, np.ndarray],
    *,
    source_rank_member_authority: Mapping[str, Any],
    source_rank_member_authority_sha256: str,
    upstream_six_head_gate_passed: bool,
    global_abstain_threshold_enabled: bool,
    maximum_global_candidate_uncertainty: float,
    global_quality: np.ndarray,
    outer_fold_structured_uncertainty: np.ndarray,
    outer_fold_training_quality: np.ndarray,
    root_outer_nesting_contract: Mapping[str, Any],
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Freeze the formal190 group-relative root decision rule.

    Candidate outcomes are used only in this formal validation calibration lane.
    The eventual evaluation400 lane remains absent from every input and output.
    """

    validate_source_rank_member_authority(
        source_rank_member_authority, source_rank_member_authority_sha256
    )
    rank_scores = np.asarray(member_source_contract_rank_scores)
    base_rank_scores = np.asarray(member_source_contract_base_rank_scores)
    residuals = np.asarray(member_source_action_rank_residuals)
    uncertainty = np.asarray(
        deployment_structured_uncertainty, dtype=np.float64
    )
    outer_uncertainty = np.asarray(
        outer_fold_structured_uncertainty, dtype=np.float64
    )
    quality = np.asarray(global_quality, dtype=np.float64)
    outer_quality = np.asarray(
        outer_fold_training_quality, dtype=np.float64
    )
    groups = np.asarray(labels["group_id"]).astype(str)
    root = np.asarray(labels["root_candidate"], dtype=bool)
    candidate_index = np.asarray(labels["candidate_index"], dtype=np.int64)
    baseline = np.asarray(labels["is_baseline"], dtype=bool)
    final_success = np.asarray(labels["candidate_final_success"], dtype=np.int64)
    outer_nesting_contract = dict(root_outer_nesting_contract)
    outer_nesting_contract_sha = outer_nesting_contract.get(
        "root_outer_nesting_contract_sha256"
    )
    unsigned_outer_nesting_contract = dict(outer_nesting_contract)
    unsigned_outer_nesting_contract.pop(
        "root_outer_nesting_contract_sha256", None
    )
    if (
        not _is_sha(outer_nesting_contract_sha)
        or canonical_sha256(unsigned_outer_nesting_contract)
        != outer_nesting_contract_sha
    ):
        raise CalibrationError("root outer-nesting contract changed")
    upstream_predictions_already_group_crossfit = bool(
        outer_nesting_contract.get(
            "upstream_predictions_already_group_crossfit"
        )
        is True
        and outer_nesting_contract.get("outer_crossfit_folds")
        == CROSSFIT_FOLDS
        and outer_nesting_contract.get(
            "outer_heldout_labels_used_for_any_parameter_or_selection"
        )
        is False
        and outer_nesting_contract.get(
            "complete_root_pipeline_outer_nesting"
        )
        is True
    )
    if (
        rank_scores.ndim != 2
        or rank_scores.shape != (MEMBER_COUNT, len(groups))
        or base_rank_scores.shape != rank_scores.shape
        or residuals.shape != rank_scores.shape
        or rank_scores.dtype != np.float32
        or base_rank_scores.dtype != np.float32
        or residuals.dtype != np.float32
        or uncertainty.shape != (len(groups),)
        or outer_uncertainty.shape != (CROSSFIT_FOLDS, len(groups))
        or outer_quality.shape != (CROSSFIT_FOLDS, len(groups))
        or quality.shape != (len(groups),)
        or not np.isfinite(rank_scores).all()
        or not np.isfinite(base_rank_scores).all()
        or not np.isfinite(residuals).all()
        or not np.isfinite(uncertainty).all()
        or not np.isfinite(outer_uncertainty).all()
        or not math.isfinite(maximum_global_candidate_uncertainty)
        or maximum_global_candidate_uncertainty < 0.0
    ):
        raise CalibrationError("formal root ranker arrays are invalid")

    group_records: list[dict[str, Any]] = []
    group_input_rows: list[np.ndarray] = []
    group_names = sorted(set(groups.tolist()))
    for group in group_names:
        rows = np.flatnonzero((groups == group) & root)
        rows = rows[np.argsort(candidate_index[rows], kind="stable")]
        baseline_rows = rows[baseline[rows]]
        if (
            len(rows) < 2
            or len(baseline_rows) != 1
            or len(set(candidate_index[rows].tolist())) != len(rows)
            or int(candidate_index[baseline_rows[0]]) != int(candidate_index[rows].min())
            or not np.isin(final_success[rows], [0, 1]).all()
        ):
            raise CalibrationError("formal group legal root set changed")
        baseline_row = int(baseline_rows[0])
        relative_score = rank_scores[:, rows].astype(np.float64) - rank_scores[
            :, baseline_row
        ].astype(np.float64)[:, None]
        pair_uncertainty = np.maximum(
            uncertainty[rows], uncertainty[baseline_row]
        )
        candidate_rows = []
        for position, row in enumerate(rows):
            candidate_rows.append({
                "candidate_index": int(candidate_index[row]),
                "is_lowest_legal_baseline": bool(row == baseline_row),
                "final_success": int(final_success[row]),
                "member_group_relative_composite_rank_scores": relative_score[
                    :, position
                ].tolist(),
                "member_source_contract_base_rank_scores": base_rank_scores[
                    :, row
                ].tolist(),
                "member_source_action_rank_residuals": residuals[:, row].tolist(),
                "member_source_contract_rank_scores": rank_scores[:, row].tolist(),
                "ensemble_group_relative_composite_rank_score": float(
                    relative_score[:, position].mean()
                ),
                "structured_candidate_uncertainty": float(uncertainty[row]),
                "structured_pair_uncertainty": float(pair_uncertainty[position]),
            })
        alternatives = [row for row in candidate_rows if not row["is_lowest_legal_baseline"]]
        best = max(
            alternatives,
            key=lambda row: (
                row["ensemble_group_relative_composite_rank_score"],
                -row["candidate_index"],
            ),
        )
        baseline_record = next(
            row for row in candidate_rows if row["is_lowest_legal_baseline"]
        )
        group_records.append({
            "logical_group_id": group,
            "baseline_candidate_index": baseline_record["candidate_index"],
            "baseline_final_success": baseline_record["final_success"],
            "best_composite_rank_candidate_index": best["candidate_index"],
            "best_group_relative_composite_rank_score_margin": best[
                "ensemble_group_relative_composite_rank_score"
            ],
            "best_composite_rank_candidate_structured_uncertainty": best[
                "structured_pair_uncertainty"
            ],
            "best_composite_rank_candidate_global_uncertainty": best[
                "structured_candidate_uncertainty"
            ],
            "candidates": candidate_rows,
        })
        group_input_rows.append(rows.copy())

    record_folds = _logical_group_folds(np.asarray(group_names))
    candidates, selected, development_draws = _root_candidate_grid(
        group_records,
        record_folds,
        support_fold_ids=tuple(range(CROSSFIT_FOLDS)),
        upstream_six_head_gate_passed=upstream_six_head_gate_passed,
        global_abstain_threshold_enabled=global_abstain_threshold_enabled,
        maximum_global_candidate_uncertainty=(
            maximum_global_candidate_uncertainty
        ),
        bootstrap_samples=bootstrap_samples,
        uncertainty_namespace="",
    )
    if len(group_records) != FORMAL_ROOT_GROUP_COUNT:
        selected = None
    selected_decisions = _root_selected_decisions(
        group_records, selected, uncertainty_namespace=""
    )

    outer_folds: list[dict[str, Any]] = []
    stitched_decisions: list[dict[str, Any]] = []
    row_folds = _logical_group_folds(groups)
    for outer_fold in range(CROSSFIT_FOLDS):
        train_record_mask = record_folds != outer_fold
        heldout_record_mask = ~train_record_mask
        fold_group_records: list[dict[str, Any]] = []
        for record_index, record in enumerate(group_records):
            input_rows = group_input_rows[record_index]
            baseline_input_row = int(input_rows[baseline[input_rows]][0])
            best_input_row = int(
                input_rows[
                    candidate_index[input_rows]
                    == record["best_composite_rank_candidate_index"]
                ][0]
            )
            fold_record = dict(record)
            fold_record[
                "best_composite_rank_candidate_selection_aware_structured_uncertainty"
            ] = float(
                max(
                    outer_uncertainty[outer_fold, best_input_row],
                    outer_uncertainty[outer_fold, baseline_input_row],
                )
            )
            fold_record[
                "best_composite_rank_candidate_selection_aware_global_uncertainty"
            ] = float(outer_uncertainty[outer_fold, best_input_row])
            fold_group_records.append(fold_record)
        train_records = [
            row
            for index, row in enumerate(fold_group_records)
            if bool(train_record_mask[index])
        ]
        heldout_records = [
            row
            for index, row in enumerate(fold_group_records)
            if bool(heldout_record_mask[index])
        ]
        train_record_folds = record_folds[train_record_mask]
        train_row_mask = root & (row_folds != outer_fold)
        fold_abstain = select_abstain_threshold(
            outer_uncertainty[outer_fold, train_row_mask],
            outer_quality[outer_fold, train_row_mask],
            groups[train_row_mask],
            bootstrap_samples=bootstrap_samples,
        )
        fold_candidates, fold_selected, fold_draws = _root_candidate_grid(
            train_records,
            train_record_folds,
            support_fold_ids=tuple(
                fold for fold in range(CROSSFIT_FOLDS) if fold != outer_fold
            ),
            upstream_six_head_gate_passed=upstream_six_head_gate_passed,
            global_abstain_threshold_enabled=bool(fold_abstain["enabled"]),
            maximum_global_candidate_uncertainty=float(
                fold_abstain["maximum_total_uncertainty"]
            ),
            bootstrap_samples=bootstrap_samples,
            uncertainty_namespace="selection_aware_",
        )
        fold_decisions = _root_selected_decisions(
            heldout_records,
            fold_selected,
            outer_fold=outer_fold,
            uncertainty_namespace="selection_aware_",
        )
        stitched_decisions.extend(fold_decisions)
        training_ids = [row["logical_group_id"] for row in train_records]
        heldout_ids = [row["logical_group_id"] for row in heldout_records]
        outer_folds.append({
            "outer_fold": outer_fold,
            "training_logical_group_count": len(train_records),
            "heldout_logical_group_count": len(heldout_records),
            "training_logical_group_ids_sha256": canonical_sha256(training_ids),
            "heldout_logical_group_ids_sha256": canonical_sha256(heldout_ids),
            "training_global_abstain_threshold": fold_abstain,
            "training_root_candidate_grid": fold_candidates,
            "training_root_candidate_grid_sha256": canonical_sha256(
                fold_candidates
            ),
            "training_bootstrap_draws": fold_draws,
            "selected_training_candidate": fold_selected,
            "selection_available": fold_selected is not None,
            "heldout_outcomes_used_for_training_selection": False,
            "heldout_decisions": fold_decisions,
        })

    stitched_decisions.sort(key=lambda row: row["logical_group_id"])
    coverage: dict[str, int] = {}
    for row in stitched_decisions:
        group = str(row["logical_group_id"])
        coverage[group] = coverage.get(group, 0) + 1
    exact_once = bool(
        len(coverage) == FORMAL_ROOT_GROUP_COUNT
        and set(coverage) == set(group_names)
        and all(count == 1 for count in coverage.values())
    )
    oof_arrays = {
        "changed": np.asarray(
            [row["changed_from_baseline"] for row in stitched_decisions],
            dtype=bool,
        ),
        "baseline_outcome": np.asarray(
            [row["baseline_final_success"] for row in stitched_decisions],
            dtype=np.int64,
        ),
        "selected_outcome": np.asarray(
            [row["selected_final_success"] for row in stitched_decisions],
            dtype=np.int64,
        ),
    }
    oof_arrays["paired_gain"] = (
        oof_arrays["selected_outcome"] - oof_arrays["baseline_outcome"]
    )
    oof_arrays["helpful"] = oof_arrays["changed"] & (
        oof_arrays["paired_gain"] > 0
    )
    oof_arrays["harmful"] = oof_arrays["changed"] & (
        oof_arrays["paired_gain"] < 0
    )
    oof_arrays["discordant"] = oof_arrays["helpful"] | oof_arrays["harmful"]
    oof_draws = _root_bootstrap_draws(
        len(stitched_decisions), bootstrap_samples
    )
    oof_bootstrap = _root_bootstrap_evidence(oof_arrays, oof_draws)
    oof_fold_support = []
    stitched_folds = np.asarray(
        [row["outer_fold"] for row in stitched_decisions], dtype=np.int64
    )
    for fold in range(CROSSFIT_FOLDS):
        in_fold = stitched_folds == fold
        changed_count = int((oof_arrays["changed"] & in_fold).sum())
        discordant_count = int((oof_arrays["discordant"] & in_fold).sum())
        oof_fold_support.append({
            "fold": fold,
            "logical_groups": int(in_fold.sum()),
            "changed_groups": changed_count,
            "discordant_groups": discordant_count,
            "minimum_changed_groups": MINIMUM_ROOT_CHANGED_GROUPS_PER_FOLD,
            "minimum_discordant_groups": (
                MINIMUM_ROOT_DISCORDANT_GROUPS_PER_FOLD
            ),
            "support_passed": bool(
                changed_count >= MINIMUM_ROOT_CHANGED_GROUPS_PER_FOLD
                and discordant_count
                >= MINIMUM_ROOT_DISCORDANT_GROUPS_PER_FOLD
            ),
        })
    oof_passed = bool(
        upstream_six_head_gate_passed
        and upstream_predictions_already_group_crossfit
        and len(group_records) == FORMAL_ROOT_GROUP_COUNT
        and exact_once
        and all(row["selection_available"] for row in outer_folds)
        and int(oof_arrays["changed"].sum()) >= MINIMUM_ROOT_CHANGED_GROUPS
        and int(oof_arrays["discordant"].sum())
        >= MINIMUM_ROOT_DISCORDANT_GROUPS
        and all(row["support_passed"] for row in oof_fold_support)
        and math.isfinite(oof_bootstrap["paired_gain_group_bootstrap_lcb95"])
        and oof_bootstrap["paired_gain_group_bootstrap_lcb95"] > 0.0
        and math.isfinite(
            oof_bootstrap["harmful_rate_group_bootstrap_ucb95"]
        )
        and oof_bootstrap["harmful_rate_group_bootstrap_ucb95"]
        <= MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES
    )
    oof_evidence: dict[str, Any] = {
        "format": "etsf_smolvla_piper_formal190_root_selection_oof_evidence_v1",
        "status": (
            "passed_selection_aware_root_gate"
            if oof_passed
            else "failed_selection_aware_root_gate"
        ),
        "passed_for_primary": oof_passed,
        "outer_crossfit_folds": CROSSFIT_FOLDS,
        "outer_fold_assignment": (
            "lexicographic_logical_group_index_modulo_five"
        ),
        "root_selection_nested_within_outer_training_groups": True,
        "global_abstain_threshold_nested_within_outer_training_groups": True,
        "upstream_predictions_already_group_crossfit": (
            upstream_predictions_already_group_crossfit
        ),
        "complete_temperature_scale_and_root_double_nesting": (
            upstream_predictions_already_group_crossfit
        ),
        "scope": (
            "complete_outer_fold_temperature_scale_robust_normalization_"
            "quality_threshold_grid_then_once_heldout_evaluation"
        ),
        "root_outer_nesting_contract": outer_nesting_contract,
        "root_outer_nesting_contract_sha256": outer_nesting_contract_sha,
        "formal_logical_group_count": len(group_records),
        "stitched_decision_count": len(stitched_decisions),
        "unique_stitched_logical_group_count": len(coverage),
        "every_formal_logical_group_scored_exactly_once": exact_once,
        "outer_folds": outer_folds,
        "stitched_group_decisions": stitched_decisions,
        "changed_group_count": int(oof_arrays["changed"].sum()),
        "change_coverage": float(oof_arrays["changed"].mean()),
        "helpful_group_count": int(oof_arrays["helpful"].sum()),
        "harmful_group_count": int(oof_arrays["harmful"].sum()),
        "discordant_group_count": int(oof_arrays["discordant"].sum()),
        "selected_success_count": int(oof_arrays["selected_outcome"].sum()),
        "selected_success_rate": float(oof_arrays["selected_outcome"].mean()),
        "baseline_success_count": int(oof_arrays["baseline_outcome"].sum()),
        "baseline_success_rate": float(oof_arrays["baseline_outcome"].mean()),
        **oof_bootstrap,
        "fold_support": oof_fold_support,
        "minimum_changed_groups": MINIMUM_ROOT_CHANGED_GROUPS,
        "minimum_discordant_groups": MINIMUM_ROOT_DISCORDANT_GROUPS,
        "maximum_harmful_rate_among_executed_changes": (
            MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES
        ),
        "paired_gain_lcb_must_be_strictly_positive": True,
        "bootstrap_unit": "logical_group",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": bootstrap_samples,
        "shared_bootstrap_draws": _root_bootstrap_draw_descriptor(oof_draws),
        "evaluation400_outcomes_read": False,
    }
    oof_evidence["selection_aware_oof_evidence_sha256"] = canonical_sha256(
        oof_evidence
    )
    primary_enabled = bool(
        upstream_six_head_gate_passed
        and global_abstain_threshold_enabled
        and selected is not None
        and oof_passed
    )
    artifact: dict[str, Any] = {
        "format": "etsf_smolvla_piper_formal190_root_group_ranker_v1",
        "status": (
            "enabled_source_composite_primary_ranker"
            if primary_enabled
            else "disabled_formal_gate_not_met"
        ),
        "enabled_for_primary": primary_enabled,
        "formal_logical_group_count": len(group_records),
        "member_count": MEMBER_COUNT,
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": dict(source_rank_member_authority),
        "source_rank_member_authority_sha256": (
            source_rank_member_authority_sha256
        ),
        "score_semantics": "source_contract_rank_score_minus_same_group_lowest_legal_baseline_then_five_member_mean",
        "score_is_success_logit": False,
        "score_is_success_probability": False,
        "factual_success_head_used_only_for_independent_six_head_calibration": True,
        "structured_uncertainty_semantics": "maximum_of_candidate_and_same_group_baseline_five_applicable_head_initial_e0_uncertainty",
        "root_recovery_uncertainty_policy": ROOT_RECOVERY_UNCERTAINTY_POLICY,
        "root_structured_uncertainty_heads": list(
            ROOT_STRUCTURED_UNCERTAINTY_HEADS
        ),
        "root_structured_uncertainty_head_count": len(
            ROOT_STRUCTURED_UNCERTAINTY_HEADS
        ),
        "candidate_grid": candidates,
        "selected_candidate": selected,
        "selected_group_decisions": selected_decisions,
        "full_formal190_development_metrics_are_in_sample": True,
        "full_formal190_deployment_refit_candidate_available": (
            selected is not None
        ),
        "development_bootstrap_draws": development_draws,
        "selection_aware_oof_evidence": oof_evidence,
        "selection_aware_oof_evidence_sha256": oof_evidence[
            "selection_aware_oof_evidence_sha256"
        ],
        "primary_activation_requires_selection_aware_oof_evidence": True,
        "primary_gate_components": {
            "all_six_heads_support_performance_uncertainty_gate_passed": (
                upstream_six_head_gate_passed
            ),
            "full_formal190_deployment_candidate_available": selected is not None,
            "selection_aware_oof_evidence_passed": oof_passed,
        },
        "upstream_predictions_already_group_crossfit": (
            upstream_predictions_already_group_crossfit
        ),
        "complete_temperature_scale_and_root_double_nesting": (
            upstream_predictions_already_group_crossfit
        ),
        "groups": group_records,
        "selection_precedence": [
            "maximum_paired_gain_lcb95",
            "maximum_selected_success_rate",
            "minimum_harmful_rate_ucb95",
            "maximum_change_coverage",
            "more_conservative_margin_then_uncertainty_threshold",
        ],
        "minimum_changed_groups": MINIMUM_ROOT_CHANGED_GROUPS,
        "minimum_discordant_groups": MINIMUM_ROOT_DISCORDANT_GROUPS,
        "maximum_harmful_rate_among_executed_changes": (
            MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES
        ),
        "paired_gain_lcb_must_be_strictly_positive": True,
        "zero_gain_lcb_authorizes_only_noninferiority_not_primary": True,
        "margin_comparison": "strict_greater_than_baseline_plus_margin",
        "zero_margin_tie_selects_lowest_legal_baseline": True,
        "fold_support_required": True,
        "global_abstention_applied_during_formal_gain_calibration": True,
        "maximum_global_candidate_uncertainty": (
            maximum_global_candidate_uncertainty
        ),
        "metric_weighting": "equal_logical_group",
        "bootstrap_unit": "logical_group",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": bootstrap_samples,
        "evaluation400_outcomes_read": False,
        "paired_development_outcomes_read": False,
    }
    artifact["root_group_ranker_sha256"] = canonical_sha256(artifact)
    return artifact


def select_abstain_threshold(uncertainty: np.ndarray, quality: np.ndarray, groups: np.ndarray, *, bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    valid_quality = np.isfinite(quality) & np.isfinite(uncertainty)
    unique_groups = np.asarray(sorted(set(group_values.tolist())))
    if len(unique_groups) < MINIMUM_RETAINED_GROUPS or not valid_quality.any():
        return {"status": "disabled_insufficient_validation_groups", "enabled": False, "maximum_total_uncertainty": 0.0, "candidates": [], "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": bootstrap_samples}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    candidates = []
    for quantile in THRESHOLD_QUANTILES:
        threshold = _equal_group_weighted_quantile(
            uncertainty[valid_quality], group_values[valid_quality], quantile
        )
        retained = (uncertainty <= threshold) & valid_quality
        group_coverage = np.asarray([
            float(retained[group_values == group].mean())
            for group in unique_groups
        ])
        retained_names = unique_groups[group_coverage > 0]
        retained_quality = np.asarray([
            float(quality[retained & (group_values == group)].mean())
            for group in retained_names
        ])
        retained_groups = len(retained_names)
        if retained_groups >= 2:
            draws = rng.integers(
                0, retained_groups, size=(bootstrap_samples, retained_groups)
            )
            bootstrap_quality = retained_quality[draws].mean(axis=1)
            quality_lcb = float(np.quantile(bootstrap_quality, BOOTSTRAP_ALPHA))
        else:
            quality_lcb = math.nan
        row = {
            "quantile": quantile,
            "threshold": threshold,
            "retained_sample_coverage": float(group_coverage.mean()),
            "retained_group_count": retained_groups,
            "mean_quality": float(retained_quality.mean())
            if retained_groups else 0.0,
            "group_bootstrap_quality_lcb95": quality_lcb,
        }
        row["eligible"] = bool(
            row["retained_sample_coverage"] >= MINIMUM_RETAINED_COVERAGE
            and retained_groups >= MINIMUM_RETAINED_GROUPS
            and row["group_bootstrap_quality_lcb95"] >= MINIMUM_QUALITY_LCB
        )
        candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["retained_sample_coverage"], row["threshold"])) if eligible else None
    return {
        "status": "frozen_validation_group_bootstrap_lcb" if selected else "disabled_no_threshold_meets_validation_lcb",
        "enabled": selected is not None,
        "maximum_total_uncertainty": selected["threshold"] if selected else 0.0,
        "selected_quantile": selected["quantile"] if selected else None,
        "selected_quality_lcb95": selected["group_bootstrap_quality_lcb95"] if selected else None,
        "candidates": candidates,
        "bootstrap_unit": "validation_group",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": bootstrap_samples,
        "minimum_quality_lcb": MINIMUM_QUALITY_LCB,
        "minimum_retained_groups": MINIMUM_RETAINED_GROUPS,
        "minimum_retained_coverage": MINIMUM_RETAINED_COVERAGE,
        "threshold_quantile_weighting": "equal_logical_group_then_equal_row_within_group",
        "coverage_weighting": "equal_logical_group",
        "quality_weighting": "equal_retained_logical_group",
        "test_or_paired_outcomes_used_for_selection": False,
    }


def calibrate_arrays(
    predictions: Sequence[Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    *,
    source_rank_member_authority: Mapping[str, Any],
    source_rank_member_authority_sha256: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(predictions) != MEMBER_COUNT:
        raise CalibrationError("exactly five prediction members are required")
    post_logits = np.stack([row["post_event_logits"] for row in predictions]).astype(np.float64)
    next_logits = np.stack([row["next_event_logits"] for row in predictions]).astype(np.float64)
    success_logits = np.stack([row["success_logit"] for row in predictions]).astype(np.float64)
    (
        source_contract_rank_scores,
        source_contract_base_rank_scores,
        source_action_rank_residuals,
    ) = _source_rank_float32_arrays(
        predictions,
        np.asarray(labels["root_candidate"]),
        source_rank_member_authority,
        source_rank_member_authority_sha256,
    )
    recovery_logits = np.stack(
        [row["recovery_logit"] for row in predictions]
    ).astype(np.float64)
    duration_mean = np.stack([row["duration_log_mean"] for row in predictions]).astype(np.float64)
    duration_scale = np.stack([row["duration_log_scale"] for row in predictions]).astype(np.float64)
    object_mean = np.stack([row["object_mean"] for row in predictions]).astype(np.float64)
    object_scale = np.stack([row["object_log_scale"] for row in predictions]).astype(np.float64)
    support = head_support(labels, post_logits.shape[-1])
    recovery_checkpoint_trained = bool(
        isinstance(labels.get("prediction_contract"), Mapping)
        and labels["prediction_contract"].get("recovery_head_trained") is True
    )
    recovery_support = support["heads"]["recovery"]
    recovery_support["all_member_recovery_heads_trained"] = (
        recovery_checkpoint_trained
    )
    groups = labels["group_id"].astype(str)
    post_metric, post = crossfit_event_calibration(
        post_logits,
        labels["post_event"].astype(int),
        labels["current_event"].astype(int),
        groups,
        observation_mask=None,
        baseline="persistence",
        bootstrap_samples=bootstrap_samples,
    )
    next_metric, nxt = crossfit_event_calibration(
        next_logits,
        labels["next_event"].astype(int),
        labels["current_event"].astype(int),
        groups,
        observation_mask=labels["duration_observed"].astype(bool),
        baseline="prior",
        bootstrap_samples=bootstrap_samples,
    )
    success_metric, success = crossfit_binary_calibration(
        success_logits,
        labels["success"].astype(int),
        groups,
        observation_mask=None,
        bootstrap_samples=bootstrap_samples,
        head_name="success",
    )
    recovery_metric, recovery = crossfit_binary_calibration(
        recovery_logits,
        labels["recovery"].astype(int),
        groups,
        observation_mask=labels["recovery_observed"].astype(bool),
        bootstrap_samples=bootstrap_samples,
        head_name="conditional_recovery",
    )
    duration_metric, duration = crossfit_duration_calibration(
        duration_mean,
        duration_scale,
        labels["duration"].astype(float),
        labels["current_event"].astype(int),
        groups,
        labels["duration_observed"].astype(bool),
        bootstrap_samples=bootstrap_samples,
    )
    object_metric, obj = crossfit_object_calibration(
        object_mean,
        object_scale,
        labels["object_target"].astype(float),
        groups,
        labels["object_observed"].astype(bool),
        bootstrap_samples=bootstrap_samples,
    )
    metric_by_head = {
        "post_event": post_metric,
        "next_event": next_metric,
        "success": success_metric,
        "recovery": recovery_metric,
        "duration": duration_metric,
        "object_effect": object_metric,
    }
    for name, row in support["heads"].items():
        metric = metric_by_head[name]
        row["performance_gate_passed"] = bool(
            metric.get("performance_gate_passed") is True
        )
        uncertainty_gate = metric.get("uncertainty_gate")
        row["uncertainty_gate_passed"] = bool(
            isinstance(uncertainty_gate, Mapping)
            and uncertainty_gate.get("passed") is True
        )
        trained = recovery_checkpoint_trained if name == "recovery" else True
        row["enabled_for_primary"] = bool(
            row["support_threshold_met"]
            and row["performance_gate_passed"]
            and row["uncertainty_gate_passed"]
            and trained
        )
    enabled = {
        name: bool(row["enabled_for_primary"])
        for name, row in support["heads"].items()
    }
    quality_components = []
    for name, payload, uncertainty in (
        ("post_event", post, post["total"] / math.log(post_logits.shape[-1])),
        ("next_event", nxt, nxt["total"] / math.log(next_logits.shape[-1])),
        ("success", success, success["total"] / 0.25),
        ("recovery", recovery, recovery["total"] / 0.25),
        ("duration", duration, duration["uncertainty"]),
        ("object_effect", obj, obj["uncertainty"]),
    ):
        if enabled[name]:
            quality_components.append(payload["correct"] if "correct" in payload else payload["quality"])
    if not quality_components:
        combined_quality = np.full(len(labels["group_id"]), np.nan)
    else:
        stacked_quality = np.stack(quality_components)
        valid_count = np.isfinite(stacked_quality).sum(axis=0)
        combined_quality = np.divide(np.nansum(stacked_quality, axis=0), valid_count, out=np.full(len(valid_count), np.nan), where=valid_count > 0)
    deployment_metrics = {
        "post_event": post_metric,
        "next_event": next_metric,
        "success": success_metric,
        "conditional_recovery": recovery_metric,
        "duration_lognormal_mixture": duration_metric,
        "object_total_variance": object_metric,
    }
    outer_root_uncertainty, outer_root_quality, root_outer_nesting_contract = (
        outer_nested_root_calibration_inputs(
            post_logits=post_logits,
            next_logits=next_logits,
            success_logits=success_logits,
            duration_means=duration_mean,
            duration_log_scales=duration_scale,
            object_means=object_mean,
            object_log_scales=object_scale,
            labels=labels,
            all_six_head_gate_passed=all(enabled.values()),
        )
    )
    deployment_uncertainty, deployment_uncertainty_contract = (
        deployment_root_structured_uncertainty(
            post_logits=post_logits,
            next_logits=next_logits,
            success_logits=success_logits,
            recovery_logits=recovery_logits,
            duration_means=duration_mean,
            duration_log_scales=duration_scale,
            object_means=object_mean,
            object_log_scales=object_scale,
            labels=labels,
            metrics=deployment_metrics,
        )
    )
    root_mask = labels["root_candidate"].astype(bool)
    threshold = select_abstain_threshold(
        deployment_uncertainty[root_mask],
        combined_quality[root_mask],
        labels["group_id"].astype(str)[root_mask],
        bootstrap_samples=bootstrap_samples,
    )
    threshold["uncertainty_source"] = (
        "full_formal190_refit_deployment_root_structured_uncertainty"
    )
    threshold["deployment_uncertainty_contract_sha256"] = (
        deployment_uncertainty_contract["deployment_uncertainty_contract_sha256"]
    )
    all_six_enabled = all(enabled.values())
    root_ranker = calibrate_root_group_ranker(
        source_contract_rank_scores,
        source_contract_base_rank_scores,
        source_action_rank_residuals,
        deployment_uncertainty,
        labels,
        source_rank_member_authority=source_rank_member_authority,
        source_rank_member_authority_sha256=(
            source_rank_member_authority_sha256
        ),
        upstream_six_head_gate_passed=all_six_enabled,
        global_abstain_threshold_enabled=bool(threshold["enabled"]),
        maximum_global_candidate_uncertainty=float(
            threshold["maximum_total_uncertainty"]
        ),
        global_quality=combined_quality,
        outer_fold_structured_uncertainty=outer_root_uncertainty,
        outer_fold_training_quality=outer_root_quality,
        root_outer_nesting_contract=root_outer_nesting_contract,
        bootstrap_samples=bootstrap_samples,
    )
    root_ranker.pop("root_group_ranker_sha256", None)
    root_ranker["deployment_uncertainty_contract_sha256"] = (
        deployment_uncertainty_contract[
            "deployment_uncertainty_contract_sha256"
        ]
    )
    root_ranker["shared_uncertainty_implementation_path"] = (
        deployment_uncertainty_contract["shared_implementation_path"]
    )
    root_ranker["shared_uncertainty_implementation_file_sha256"] = (
        deployment_uncertainty_contract["shared_implementation_file_sha256"]
    )
    root_ranker["root_group_ranker_sha256"] = canonical_sha256(root_ranker)
    calibration: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "status": "complete_validation_only_metrics_and_threshold_freeze",
        "member_count": MEMBER_COUNT,
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": dict(source_rank_member_authority),
        "source_rank_member_authority_sha256": (
            source_rank_member_authority_sha256
        ),
        "metrics": {
            "post_event": post_metric,
            "next_event": next_metric,
            "success": success_metric,
            "conditional_recovery": recovery_metric,
            "duration_lognormal_mixture": duration_metric,
            "object_total_variance": object_metric,
        },
        "uncertainty_decomposition": {
            "event": "predictive_entropy=mean_member_entropy+mutual_information",
            "success": "total_bernoulli_variance=mean_member_aleatoric+variance_member_probability",
            "conditional_recovery": "conditional_on_observed_operational_regress;total_bernoulli_variance=mean_member_aleatoric+variance_member_probability",
            "duration": "total_variance=mean_member_lognormal_variance+variance_member_lognormal_mean",
            "object": "total_variance=mean_member_gaussian_variance+variance_member_mean",
            "structured_total_uncertainty": "mean_of_enabled_dimensionless_head_uncertainties",
        },
        "deployment_root_structured_uncertainty_contract": (
            deployment_uncertainty_contract
        ),
        "head_enabled_for_primary": enabled,
        "head_performance_gate_protocol": PERFORMANCE_GATE_PROTOCOL,
        "all_six_heads_support_performance_uncertainty_gate_passed": all_six_enabled,
        "prediction_contract": dict(
            labels.get("prediction_contract", {})
        ),
        "success_temperature_fitted_on_validation_only": enabled["success"],
        "recovery_temperature_fitted_on_validation_only": enabled["recovery"],
        "duration_scale_fitted_by_group_crossfit": enabled["duration"],
        "object_scale_fitted_by_group_crossfit": enabled["object_effect"],
        "recovery_enters_primary_only_if_support_and_calibration_pass": True,
        "abstain_threshold": threshold,
        "root_group_ranker": root_ranker,
        "root_group_ranker_enabled_for_primary": root_ranker[
            "enabled_for_primary"
        ],
        "deployment_uncertainty_contract_sha256": (
            deployment_uncertainty_contract[
                "deployment_uncertainty_contract_sha256"
            ]
        ),
        "validation_groups": len(set(labels["group_id"].astype(str).tolist())),
        "validation_samples": len(labels["group_id"]),
        "test_artifacts_read": False,
        "test_hdf5_files_opened": 0,
        "fresh_artifacts_read": False,
        "confirmation_artifacts_read": False,
        "paired_development_outcomes_read": False,
        "performance_claim_authorized": False,
    }
    calibration["calibration_sha256"] = canonical_sha256(calibration)
    support.pop("head_support_sha256", None)
    support["head_support_sha256"] = canonical_sha256(support)
    return calibration, support


def run(input_authority_path: Path, expected_input_file_sha256: str, output_root: Path, *, bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    authority_path = safe_existing(input_authority_path, "input authority", allowed_suffixes={".json"})
    if not _is_sha(expected_input_file_sha256) or file_sha256(authority_path) != expected_input_file_sha256:
        raise CalibrationError("input authority file SHA mismatch")
    authority = load_json(authority_path, "input authority")
    audit = validate_input_authority(authority)
    predictions, labels = load_validation_arrays(audit)
    calibration, support = calibrate_arrays(
        predictions,
        labels,
        source_rank_member_authority=audit["source_rank_member_authority"],
        source_rank_member_authority_sha256=audit[
            "source_rank_member_authority_sha256"
        ],
        bootstrap_samples=bootstrap_samples,
    )
    root = safe_new_root(output_root)
    root.mkdir(mode=0o755)
    immutable_json(root / "calibration.json", calibration)
    immutable_json(root / "paired_head_support.json", support)
    root_ranker = calibration["root_group_ranker"]
    immutable_json(root / "formal190_root_group_ranker.json", root_ranker)
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "status": "frozen_validation_only_five_member_deployment_contract",
        "member_count": MEMBER_COUNT,
        "source_rank_numeric_contract": audit[
            "source_rank_numeric_contract"
        ],
        "source_rank_member_authority": audit[
            "source_rank_member_authority"
        ],
        "source_rank_member_authority_sha256": audit[
            "source_rank_member_authority_sha256"
        ],
        "members": [
            {
                "member_index": row["member_index"],
                "member_seed": row["member_seed"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_file_sha256": row["checkpoint_file_sha256"],
                "source_rank_score_contract": row[
                    "source_rank_score_contract"
                ],
                "source_rank_score_contract_sha256": row[
                    "source_rank_score_contract_sha256"
                ],
            }
            for row in audit["members"]
        ],
        "shared_contract": audit["shared_contract"],
        "prediction_contract": audit["prediction_contract"],
        "deployment_root_structured_uncertainty_contract": calibration[
            "deployment_root_structured_uncertainty_contract"
        ],
        "deployment_uncertainty_contract_sha256": calibration[
            "deployment_uncertainty_contract_sha256"
        ],
        "post_event_temperature": calibration["metrics"]["post_event"][
            "deployment_temperature"
        ] if calibration["head_enabled_for_primary"]["post_event"] else 1.0,
        "next_event_temperature": calibration["metrics"]["next_event"][
            "deployment_temperature"
        ] if calibration["head_enabled_for_primary"]["next_event"] else 1.0,
        "success_temperature": calibration["metrics"]["success"]["deployment_temperature"] if calibration["head_enabled_for_primary"]["success"] else 1.0,
        "conditional_recovery_temperature": (
            calibration["metrics"]["conditional_recovery"]["deployment_temperature"]
            if calibration["head_enabled_for_primary"]["recovery"]
            else 1.0
        ),
        "duration_scale_multiplier": calibration["metrics"][
            "duration_lognormal_mixture"
        ]["deployment_scale_multiplier"] if calibration[
            "head_enabled_for_primary"
        ]["duration"] else 1.0,
        "object_scale_multiplier": calibration["metrics"][
            "object_total_variance"
        ]["deployment_scale_multiplier"] if calibration[
            "head_enabled_for_primary"
        ]["object_effect"] else 1.0,
        "object_error_robust_scale_m": calibration["metrics"][
            "object_total_variance"
        ]["deployment_object_error_robust_scale_m"] if calibration[
            "head_enabled_for_primary"
        ]["object_effect"] else 1.0,
        "conditional_recovery_semantics": "p(recovery_given_operational_regress)",
        "conditional_recovery_activation_requires_observed_regress": True,
        "head_enabled_for_primary": calibration["head_enabled_for_primary"],
        "all_six_heads_support_performance_uncertainty_gate_passed": calibration[
            "all_six_heads_support_performance_uncertainty_gate_passed"
        ],
        "root_group_ranker": {
            "path": str(root / "formal190_root_group_ranker.json"),
            "file_sha256": file_sha256(
                root / "formal190_root_group_ranker.json"
            ),
            "logical_sha256": root_ranker["root_group_ranker_sha256"],
            "enabled_for_primary": root_ranker["enabled_for_primary"],
        },
        "maximum_total_uncertainty": calibration["abstain_threshold"]["maximum_total_uncertainty"],
        "abstain_threshold_enabled": calibration["abstain_threshold"]["enabled"],
        "calibration_sha256": calibration["calibration_sha256"],
        "head_support_sha256": support["head_support_sha256"],
        "root_group_ranker_path": str(
            root / "formal190_root_group_ranker.json"
        ),
        "root_group_ranker_file_sha256": file_sha256(
            root / "formal190_root_group_ranker.json"
        ),
        "root_group_ranker_sha256": root_ranker[
            "root_group_ranker_sha256"
        ],
        "root_group_ranker_enabled_for_primary": root_ranker[
            "enabled_for_primary"
        ],
        "validation_identity_set_sha256": audit["validation_identity_set_sha256"],
        "test_artifacts_read": False,
        "test_hdf5_files_opened": 0,
        "fresh_artifacts_read": False,
        "confirmation_artifacts_read": False,
        "paired_development_outcomes_read": False,
    }
    manifest["ensemble_manifest_sha256"] = canonical_sha256(manifest)
    immutable_json(root / "ensemble_manifest.json", manifest)
    receipt: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "status": RECEIPT_STATUS,
        "input_authority_path": str(authority_path),
        "input_authority_file_sha256": expected_input_file_sha256,
        "input_authority_sha256": audit["logical_sha256"],
        "member_count": MEMBER_COUNT,
        "validation_only": True,
        "shared_contract": audit["shared_contract"],
        "prediction_contract_sha256": audit["shared_contract"][
            "prediction_contract_sha256"
        ],
        "source_rank_numeric_contract": audit[
            "source_rank_numeric_contract"
        ],
        "source_rank_member_authority": audit[
            "source_rank_member_authority"
        ],
        "source_rank_member_authority_sha256": audit[
            "source_rank_member_authority_sha256"
        ],
        "calibration_path": str(root / "calibration.json"),
        "calibration_file_sha256": file_sha256(root / "calibration.json"),
        "calibration_sha256": calibration["calibration_sha256"],
        "head_support_path": str(root / "paired_head_support.json"),
        "head_support_file_sha256": file_sha256(root / "paired_head_support.json"),
        "head_support_sha256": support["head_support_sha256"],
        "root_group_ranker_path": str(
            root / "formal190_root_group_ranker.json"
        ),
        "root_group_ranker_file_sha256": file_sha256(
            root / "formal190_root_group_ranker.json"
        ),
        "root_group_ranker_sha256": root_ranker[
            "root_group_ranker_sha256"
        ],
        "root_group_ranker_enabled_for_primary": root_ranker[
            "enabled_for_primary"
        ],
        "deployment_uncertainty_contract_sha256": calibration[
            "deployment_uncertainty_contract_sha256"
        ],
        "ensemble_manifest_path": str(root / "ensemble_manifest.json"),
        "ensemble_manifest_file_sha256": file_sha256(root / "ensemble_manifest.json"),
        "ensemble_manifest_sha256": manifest["ensemble_manifest_sha256"],
        "abstain_threshold_enabled": calibration["abstain_threshold"]["enabled"],
        "test_artifacts_read": False,
        "test_hdf5_files_opened": 0,
        "fresh_paths_accepted": False,
        "confirmation_artifacts_read": False,
        "paired_development_outcomes_read": False,
        "performance_or_transfer_claim_authorized": False,
        "artifacts_frozen_read_only": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    immutable_json(root / "final_receipt.json", receipt)
    immutable_text(root / "run.exit", "0\n")
    freeze_tree(root)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-authority", type=Path, required=True)
    parser.add_argument("--input-authority-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = run(args.input_authority, args.input_authority_file_sha256, args.output_root)
    print("ADAPTER_ENSEMBLE_VALIDATION_RECEIPT=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
