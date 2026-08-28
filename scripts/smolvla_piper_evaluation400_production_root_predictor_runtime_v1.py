#!/usr/bin/env python3
"""Freeze and run the real five-member SmolVLA/Piper root predictor.

This module is intentionally separate from the compact reference runtime.  A
member is always the production family used by the evaluation400 v3 backend:
``ActionConditionedEventWorldModel`` wrapped by ``SmolVLAPiperAdapter`` plus a
detached ``DetachedConditionalRecoveryAdapter``.  The converter reads only
already-frozen checkpoints and an externally pinned execution authority; it
never reads rollouts, HDF5 files, simulator state, targets, or outcomes.

The resulting authority permits execution inside evaluation400 but cannot by
itself promote a scientific task-success claim.  Such promotion remains an
outcome-level decision outside this runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import etsf_torch_weights_only_compat_v1 as weights_only_compat
import openvla_etsf_event_world_model as event_world_model
import smolvla_piper_deployment_uncertainty_v1 as deployment_uncertainty
import smolvla_piper_evaluation400_root_observed_contract_v1 as root_contract
import train_smolvla_piper_schema6_embodiment_adapter as adapter_trainer


MEMBER_COUNT = 5
MODEL_FAMILY = (
    "ActionConditionedEventWorldModel+SmolVLAPiperAdapter+"
    "DetachedConditionalRecoveryAdapter"
)
EXECUTION_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_production_root_execution_authority_v1"
)
ARTIFACT_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_production_root_artifact_authority_v1"
)
RUNTIME_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_production_root_runtime_authority_v1"
)
MEMBER_CHECKPOINT_FORMAT = (
    "etsf_smolvla_piper_production_root_member_checkpoint_v1"
)
CALIBRATION_FORMAT = "etsf_smolvla_piper_production_root_calibration_v1"
UNCERTAINTY_FORMAT = "etsf_smolvla_piper_production_root_uncertainty_v1"

AUTHORITY_BASENAME = "authority_manifest.json"
RUNTIME_AUTHORITY_BASENAME = "runtime_authority.json"
EXECUTION_AUTHORITY_BASENAME = "execution_authority.json"
CALIBRATION_BASENAME = "calibration.json"
UNCERTAINTY_BASENAME = "uncertainty_contract.json"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
SOURCE_RANK_NUMERIC_CONTRACT = adapter_trainer.SOURCE_RANK_NUMERIC_CONTRACT


class ProductionRootPredictorError(RuntimeError):
    """A frozen production artifact or actor-visible inference failed closed."""


def canonical_sha256(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProductionRootPredictorError("value is not canonical JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha(value: Any, role: str) -> str:
    if not _is_sha(value):
        raise ProductionRootPredictorError(f"{role} must be exact SHA-256")
    return str(value)


def _exact(value: Any, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProductionRootPredictorError(f"{role} fields changed")
    return value


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(base)
    return {**value, field: canonical_sha256(value)}


def _verify_signed(
    value: Any, *, fields: set[str], digest_field: str, role: str,
) -> str:
    item = _exact(value, fields | {digest_field}, role)
    digest = _sha(item[digest_field], f"{role} logical SHA")
    unsigned = {name: child for name, child in item.items() if name != digest_field}
    if digest != canonical_sha256(unsigned):
        raise ProductionRootPredictorError(f"{role} logical SHA changed")
    return digest


def _positive(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ProductionRootPredictorError(f"{role} must be finite and positive")
    return float(value)


def _read_bytes(path: Path, *, maximum: int, role: str) -> bytes:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ProductionRootPredictorError(f"{role} cannot be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(supplied, flags)
    except OSError as error:
        raise ProductionRootPredictorError(f"{role} cannot be opened") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise ProductionRootPredictorError(f"{role} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProductionRootPredictorError(f"{role} was truncated while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    raw = _read_bytes(path, maximum=MAX_CHECKPOINT_BYTES, role=f"file {path.name}")
    return hashlib.sha256(raw).hexdigest()


def _duplicate_reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ProductionRootPredictorError("JSON contains duplicate keys")
        result[name] = value
    return result


def _parse_json(raw: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_reject,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProductionRootPredictorError(f"{role} contains {token}")
            ),
        )
    except ProductionRootPredictorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionRootPredictorError(f"{role} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProductionRootPredictorError(f"{role} must contain one object")
    return value


def _read_hashed_json(path: Path, expected_sha: str, role: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_bytes(path, maximum=MAX_JSON_BYTES, role=role)
    if hashlib.sha256(raw).hexdigest() != _sha(expected_sha, f"{role} file SHA"):
        raise ProductionRootPredictorError(f"{role} file SHA changed")
    return _parse_json(raw, role), raw


def _safe_torch_bytes(raw: bytes, role: str) -> dict[str, Any]:
    try:
        value = weights_only_compat.load_numpy_weights_only(io.BytesIO(raw))
    except Exception as error:
        raise ProductionRootPredictorError(
            f"{role} is not a weights-only torch mapping"
        ) from error
    if not isinstance(value, Mapping):
        raise ProductionRootPredictorError(f"{role} must contain one mapping")
    return dict(value)


def _tensor_digest(name: str, tensor: torch.Tensor) -> str:
    if type(tensor) is not torch.Tensor:
        raise ProductionRootPredictorError(f"tensor {name} changed type")
    value = tensor.detach().cpu().contiguous()
    if value.is_sparse or not bool(torch.isfinite(value).all()):
        raise ProductionRootPredictorError(f"tensor {name} is invalid")
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_sha256(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    # ``view(dtype)`` rejects a zero-dimensional float tensor.  The real
    # adapter state contains scalar clock parameters, so flatten before taking
    # the byte view while retaining the original shape in the digest above.
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(value: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(value, Mapping) or not value:
        raise ProductionRootPredictorError("state dict is empty")
    names = sorted(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise ProductionRootPredictorError("state dict parameter names changed")
    return canonical_sha256(
        [{"name": name, "tensor_sha256": _tensor_digest(name, value[name])} for name in names]
    )


def _array_bundle_sha256(value: Mapping[str, Any]) -> str:
    records = []
    for name in sorted(value):
        array = np.ascontiguousarray(np.asarray(value[name]))
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ProductionRootPredictorError(f"array {name} is invalid")
        digest = hashlib.sha256()
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        records.append({"name": name, "tensor_sha256": digest.hexdigest()})
    return canonical_sha256(records)


def _module_sha(module: Any, role: str) -> str:
    source = inspect.getsourcefile(module)
    if not source:
        raise ProductionRootPredictorError(f"{role} implementation is unavailable")
    return file_sha256(Path(source).resolve())


def _runtime_sha() -> str:
    return file_sha256(Path(__file__).resolve())


def _atomic_bytes(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_bytes(path, raw)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    _atomic_bytes(path, buffer.getvalue())


def _implementation_contract() -> dict[str, str]:
    return {
        "event_world_model_file_sha256": _module_sha(
            event_world_model, "event world model"
        ),
        "piper_adapter_file_sha256": _module_sha(
            adapter_trainer, "Piper adapter"
        ),
        "deployment_uncertainty_file_sha256": _module_sha(
            deployment_uncertainty, "deployment uncertainty"
        ),
        "root_observed_contract_file_sha256": _module_sha(
            root_contract, "root observed contract"
        ),
        "weights_only_compat_file_sha256": _module_sha(
            weights_only_compat, "weights-only compatibility layer"
        ),
        "production_runtime_file_sha256": _runtime_sha(),
    }


def validate_execution_authority(
    value: Mapping[str, Any], *, expected_authority_sha256: str,
) -> dict[str, Any]:
    fields = {
        "format", "status", "model_family", "member_count",
        "source_checkpoint_file_sha256", "adapter_checkpoint_file_sha256",
        "source_rank_score_contract_sha256", "model_config_sha256",
        "object_source_normalization_sha256", "calibration",
        "implementation", "production_evaluation400_execution_authorized",
        "scientific_promotion_eligible", "compact_reference_model_allowed",
        "actor_visible_root_observation_required",
        "simulator_or_target_inputs_available_to_predictor",
    }
    logical = _verify_signed(
        value,
        fields=fields,
        digest_field="execution_authority_sha256",
        role="production execution authority",
    )
    item = dict(value)
    lists = (
        "source_checkpoint_file_sha256",
        "adapter_checkpoint_file_sha256",
        "source_rank_score_contract_sha256",
        "object_source_normalization_sha256",
    )
    calibration_fields = {
        "source_calibration_file_sha256", "source_calibration_sha256",
        "root_group_ranker_sha256", "post_event_temperature",
        "next_event_temperature", "success_temperature",
        "conditional_recovery_temperature", "duration_scale_multiplier",
        "object_scale_multiplier", "object_error_robust_scale_m",
        "maximum_total_uncertainty", "all_six_heads_enabled",
        "formal190_selection_aware_gate_passed",
    }
    calibration = item.get("calibration")
    implementation = item.get("implementation")
    if (
        logical != _sha(expected_authority_sha256, "expected execution authority")
        or item.get("format") != EXECUTION_AUTHORITY_FORMAT
        or item.get("status")
        != "frozen_real_model_family_for_evaluation400_execution_only"
        or item.get("model_family") != MODEL_FAMILY
        or item.get("member_count") != MEMBER_COUNT
        or type(item.get("member_count")) is not int
        or any(
            not isinstance(item.get(name), list)
            or len(item[name]) != MEMBER_COUNT
            or any(not _is_sha(child) for child in item[name])
            for name in lists
        )
        or not _is_sha(item.get("model_config_sha256"))
        or not isinstance(calibration, Mapping)
        or set(calibration) != calibration_fields
        or not isinstance(implementation, Mapping)
        or dict(implementation) != _implementation_contract()
        or item.get("production_evaluation400_execution_authorized") is not True
        or item.get("scientific_promotion_eligible") is not False
        or item.get("compact_reference_model_allowed") is not False
        or item.get("actor_visible_root_observation_required") is not True
        or item.get("simulator_or_target_inputs_available_to_predictor") is not False
    ):
        raise ProductionRootPredictorError("production execution authority changed")
    for name in (
        "source_calibration_file_sha256", "source_calibration_sha256",
        "root_group_ranker_sha256",
    ):
        _sha(calibration[name], f"calibration {name}")
    for name in (
        "post_event_temperature", "next_event_temperature", "success_temperature",
        "conditional_recovery_temperature", "duration_scale_multiplier",
        "object_scale_multiplier", "object_error_robust_scale_m",
    ):
        _positive(calibration[name], f"calibration {name}")
    maximum = calibration["maximum_total_uncertainty"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or not 0.0 <= float(maximum) <= 1.0
        or calibration["all_six_heads_enabled"] is not True
        or calibration["formal190_selection_aware_gate_passed"] is not True
    ):
        raise ProductionRootPredictorError("production calibration authority changed")
    item["execution_authority_sha256"] = logical
    return item


def _validate_source_and_adapter(
    *, source_raw: bytes, adapter_raw: bytes, source_file_sha: str,
    adapter_file_sha: str, member_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _safe_torch_bytes(source_raw, f"source member {member_index}")
    payload = _safe_torch_bytes(adapter_raw, f"adapter member {member_index}")
    try:
        audit = adapter_trainer.validate_source_checkpoint(source)
        config = event_world_model.EventWorldModelConfig.from_dict(audit["config"])
        adapter_trainer.validate_production_source_rank_config(config)
        rank = adapter_trainer._validate_source_rank_score_contract(
            payload.get("source_rank_score_contract")
        )
    except Exception as error:
        raise ProductionRootPredictorError(
            f"member {member_index} source/rank validation failed"
        ) from error
    adapter_config = payload.get("adapter_config")
    ranking = payload.get("ranking_contract")
    recovery_contract = payload.get("conditional_recovery_contract")
    if (
        payload.get("format") != adapter_trainer.FORMAT
        or payload.get("source_checkpoint_sha256") != source_file_sha
        or rank.get("source_checkpoint_file_sha256") != source_file_sha
        or not isinstance(adapter_config, Mapping)
        or type(adapter_config.get("state_rank")) is not int
        or int(adapter_config["state_rank"]) < 1
        or type(adapter_config.get("action_rank")) is not int
        or int(adapter_config["action_rank"]) < 1
        or adapter_config.get("source_action_rank_residual_consumed") is not True
        or adapter_config.get("source_action_rank_success_only") is not False
        or adapter_config.get("deployment_success_logit")
        != "base_factual_success_logit"
        or adapter_config.get("deployment_primary_candidate_score")
        != "source_contract_rank_score"
        or adapter_config.get("source_contract_rank_score_is_success_logit") is not False
        or adapter_config.get("source_contract_rank_score_is_success_probability") is not False
        or adapter_config.get("source_rank_score_contract_sha256")
        != rank["contract_sha256"]
        or not isinstance(ranking, Mapping)
        or ranking.get("candidate_prediction_api") != "predict_grouped_candidates"
        or ranking.get("source_action_rank_success_only") is not False
        or ranking.get("deployment_success_logit") != "base_factual_success_logit"
        or ranking.get("deployment_primary_candidate_score")
        != "source_contract_rank_score"
        or ranking.get("source_contract_rank_score_is_success_logit") is not False
        or ranking.get("source_contract_rank_score_is_success_probability") is not False
        or ranking.get("deployment_success_probability_selector_authorized") is not False
        or not isinstance(recovery_contract, Mapping)
        or recovery_contract.get("trained") is not True
        or recovery_contract.get("shared_transition_stop_gradient") is not True
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("conditional_recovery_adapter"), Mapping)
    ):
        raise ProductionRootPredictorError(
            f"member {member_index} adapter/recovery lineage changed"
        )
    try:
        object_mean, object_std = adapter_trainer.object_normalization(
            source, config.object_delta_dim
        )
        native = event_world_model.ActionConditionedEventWorldModel(config)
        native.load_state_dict(source["model"], strict=True)
        model = adapter_trainer.SmolVLAPiperAdapter(
            native,
            state_rank=int(adapter_config["state_rank"]),
            action_rank=int(adapter_config["action_rank"]),
            source_rank_contract=rank,
        )
        model.load_state_dict(payload["model"], strict=True)
        immutable = model.enforce_and_verify_frozen_core()
        if immutable.get("all_core_tensors_except_piper_body_row_bit_exact") is not True:
            raise ProductionRootPredictorError("adapter changed frozen factual core")
        recovery = adapter_trainer.DetachedConditionalRecoveryAdapter(
            config.semantic_dim
        )
        recovery.load_state_dict(payload["conditional_recovery_adapter"], strict=True)
        if recovery.parameter_audit().get("shared_transition_stop_gradient") is not True:
            raise ProductionRootPredictorError("recovery adapter is not detached")
    except ProductionRootPredictorError:
        raise
    except Exception as error:
        raise ProductionRootPredictorError(
            f"member {member_index} real model family cannot be reconstructed"
        ) from error
    model.eval()
    recovery.eval()
    for parameter in (*model.parameters(), *recovery.parameters()):
        parameter.requires_grad_(False)
    model_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    recovery_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in recovery.state_dict().items()
    }
    normalization = {
        "object_delta_mean": object_mean.detach().cpu().numpy().astype(np.float32),
        "object_delta_std": object_std.detach().cpu().numpy().astype(np.float32),
    }
    checkpoint_base = {
        "format": MEMBER_CHECKPOINT_FORMAT,
        "status": "frozen_real_production_model_family_weights_only",
        "model_family": MODEL_FAMILY,
        "member_index": member_index,
        "source_checkpoint_file_sha256": source_file_sha,
        "adapter_checkpoint_file_sha256": adapter_file_sha,
        "model_config": config.to_dict(),
        "model_config_sha256": canonical_sha256(config.to_dict()),
        "state_rank": int(adapter_config["state_rank"]),
        "action_rank": int(adapter_config["action_rank"]),
        "source_rank_score_contract": rank,
        "source_rank_score_contract_sha256": rank["contract_sha256"],
        "object_delta_mean": torch.from_numpy(normalization["object_delta_mean"]),
        "object_delta_std": torch.from_numpy(normalization["object_delta_std"]),
        "object_source_normalization_sha256": _array_bundle_sha256(normalization),
        "model_state_dict": model_state,
        "model_state_dict_sha256": state_dict_sha256(model_state),
        "recovery_state_dict": recovery_state,
        "recovery_state_dict_sha256": state_dict_sha256(recovery_state),
        "recovery_shared_transition_stop_gradient": True,
    }
    logical_base = {
        name: child for name, child in checkpoint_base.items()
        if name not in {
            "model_state_dict", "recovery_state_dict",
            "object_delta_mean", "object_delta_std",
        }
    }
    checkpoint = {
        **checkpoint_base,
        "checkpoint_logical_sha256": canonical_sha256(logical_base),
    }
    record = {
        "member_index": member_index,
        "source_checkpoint_file_sha256": source_file_sha,
        "adapter_checkpoint_file_sha256": adapter_file_sha,
        "source_rank_score_contract_sha256": rank["contract_sha256"],
        "object_source_normalization_sha256": checkpoint[
            "object_source_normalization_sha256"
        ],
        "model_config_sha256": checkpoint["model_config_sha256"],
        "converted_checkpoint_logical_sha256": checkpoint[
            "checkpoint_logical_sha256"
        ],
    }
    return checkpoint, record, audit


def _calibration_document(authority: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(authority["calibration"])
    base = {
        "format": CALIBRATION_FORMAT,
        "status": "formal190_frozen_execution_parameters_no_scientific_promotion",
        **source,
        "scientific_promotion_eligible": False,
        "parameter_application": (
            "temperature_then_duration_object_scale_exactly_once"
        ),
    }
    return _signed(base, "calibration_sha256")


def _uncertainty_document(calibration: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        "format": UNCERTAINTY_FORMAT,
        "status": "frozen_five_head_root_gate_recovery_ablation_only",
        "root_included_heads": list(deployment_uncertainty.ROOT_INCLUDED_HEADS),
        "root_head_count": deployment_uncertainty.ROOT_HEAD_COUNT,
        "root_recovery_policy": deployment_uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY,
        "calibration_sha256": calibration["calibration_sha256"],
        "implementation_file_sha256": _module_sha(
            deployment_uncertainty, "deployment uncertainty"
        ),
    }
    return _signed(base, "uncertainty_contract_sha256")


def convert_production_root_predictor_artifacts(
    *, source_checkpoint_paths: Sequence[Path],
    adapter_checkpoint_paths: Sequence[Path],
    execution_authority_path: Path,
    expected_execution_authority_file_sha256: str,
    expected_execution_authority_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Convert five original checkpoints into a sealed weights-only runtime."""

    if len(source_checkpoint_paths) != MEMBER_COUNT or len(adapter_checkpoint_paths) != MEMBER_COUNT:
        raise ProductionRootPredictorError("converter requires exactly five member pairs")
    authority, authority_raw = _read_hashed_json(
        Path(execution_authority_path), expected_execution_authority_file_sha256,
        "execution authority",
    )
    authority = validate_execution_authority(
        authority, expected_authority_sha256=expected_execution_authority_sha256
    )
    unresolved = Path(output_directory)
    if unresolved.is_symlink():
        raise ProductionRootPredictorError("output directory cannot be a symlink")
    output = unresolved.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ProductionRootPredictorError("output directory must be absent or empty")

    converted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    config_shas: list[str] = []
    normalization_shas: list[str] = []
    for index, (source_path, adapter_path) in enumerate(
        zip(source_checkpoint_paths, adapter_checkpoint_paths, strict=True)
    ):
        source_raw = _read_bytes(
            Path(source_path), maximum=MAX_CHECKPOINT_BYTES,
            role=f"source member {index}",
        )
        adapter_raw = _read_bytes(
            Path(adapter_path), maximum=MAX_CHECKPOINT_BYTES,
            role=f"adapter member {index}",
        )
        source_sha = hashlib.sha256(source_raw).hexdigest()
        adapter_sha = hashlib.sha256(adapter_raw).hexdigest()
        if (
            source_sha != authority["source_checkpoint_file_sha256"][index]
            or adapter_sha != authority["adapter_checkpoint_file_sha256"][index]
        ):
            raise ProductionRootPredictorError(
                f"member {index} input checkpoint differs from execution authority"
            )
        checkpoint, record, _audit = _validate_source_and_adapter(
            source_raw=source_raw,
            adapter_raw=adapter_raw,
            source_file_sha=source_sha,
            adapter_file_sha=adapter_sha,
            member_index=index,
        )
        if record["source_rank_score_contract_sha256"] != authority[
            "source_rank_score_contract_sha256"
        ][index]:
            raise ProductionRootPredictorError(f"member {index} rank authority changed")
        converted.append((checkpoint, record))
        config_shas.append(record["model_config_sha256"])
        normalization_shas.append(record["object_source_normalization_sha256"])
    if (
        len(set(config_shas)) != 1
        or config_shas[0] != authority["model_config_sha256"]
        or normalization_shas != authority["object_source_normalization_sha256"]
        or len(set(normalization_shas)) != 1
    ):
        raise ProductionRootPredictorError(
            "five-member config/object normalization is not the frozen shared contract"
        )

    output.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output / EXECUTION_AUTHORITY_BASENAME, authority_raw)
    calibration = _calibration_document(authority)
    uncertainty = _uncertainty_document(calibration)
    _atomic_json(output / CALIBRATION_BASENAME, calibration)
    _atomic_json(output / UNCERTAINTY_BASENAME, uncertainty)
    member_records: list[dict[str, Any]] = []
    checkpoint_file_shas: list[str] = []
    rank_shas: list[str] = []
    for index, (checkpoint, source_record) in enumerate(converted):
        basename = f"member_{index:02d}.pt"
        path = output / basename
        _atomic_torch(path, checkpoint)
        file_sha = file_sha256(path)
        checkpoint_file_shas.append(file_sha)
        rank_shas.append(source_record["source_rank_score_contract_sha256"])
        member_records.append({
            **source_record,
            "converted_checkpoint_basename": basename,
            "converted_checkpoint_file_sha256": file_sha,
        })
    runtime_base = {
        "format": RUNTIME_AUTHORITY_FORMAT,
        "status": "real_model_family_frozen_for_evaluation400_execution",
        "model_family": MODEL_FAMILY,
        "member_count": MEMBER_COUNT,
        "members": member_records,
        "converted_checkpoint_file_set_sha256": canonical_sha256(
            checkpoint_file_shas
        ),
        "source_rank_contract_set_sha256": canonical_sha256(rank_shas),
        "execution_authority": {
            "basename": EXECUTION_AUTHORITY_BASENAME,
            "file_sha256": hashlib.sha256(authority_raw).hexdigest(),
            "logical_sha256": authority["execution_authority_sha256"],
        },
        "calibration": {
            "basename": CALIBRATION_BASENAME,
            "file_sha256": file_sha256(output / CALIBRATION_BASENAME),
            "logical_sha256": calibration["calibration_sha256"],
        },
        "uncertainty_contract": {
            "basename": UNCERTAINTY_BASENAME,
            "file_sha256": file_sha256(output / UNCERTAINTY_BASENAME),
            "logical_sha256": uncertainty["uncertainty_contract_sha256"],
        },
        "implementation": _implementation_contract(),
        "production_evaluation400_execution_authorized": True,
        "scientific_promotion_eligible": False,
        "compact_reference_model_allowed": False,
        "online_input_contract": (
            "validated_actor_visible_root_observation_and_candidate_actions_only"
        ),
        "member_call_contract": "exactly_five_real_members_once_each",
    }
    runtime_authority = _signed(runtime_base, "runtime_authority_sha256")
    _atomic_json(output / RUNTIME_AUTHORITY_BASENAME, runtime_authority)
    roles = {
        EXECUTION_AUTHORITY_BASENAME: "execution_authority",
        CALIBRATION_BASENAME: "calibration",
        UNCERTAINTY_BASENAME: "uncertainty_contract",
        RUNTIME_AUTHORITY_BASENAME: "runtime_authority",
        **{f"member_{index:02d}.pt": "real_member_checkpoint" for index in range(MEMBER_COUNT)},
    }
    inventory = [
        {
            "basename": basename,
            "role": roles[basename],
            "file_sha256": file_sha256(output / basename),
        }
        for basename in sorted(roles)
    ]
    manifest_base = {
        "format": ARTIFACT_AUTHORITY_FORMAT,
        "status": "production_evaluation400_execution_not_scientific_promotion",
        "model_family": MODEL_FAMILY,
        "member_count": MEMBER_COUNT,
        "artifact_inventory": inventory,
        "runtime_authority_sha256": runtime_authority[
            "runtime_authority_sha256"
        ],
        "converted_checkpoint_file_set_sha256": runtime_authority[
            "converted_checkpoint_file_set_sha256"
        ],
        "source_rank_contract_set_sha256": runtime_authority[
            "source_rank_contract_set_sha256"
        ],
        "production_evaluation400_execution_authorized": True,
        "scientific_promotion_eligible": False,
    }
    manifest = _signed(manifest_base, "root_predictor_authority_sha256")
    _atomic_json(output / AUTHORITY_BASENAME, manifest)
    return {
        "artifact_root": str(output),
        "root_predictor_authority_sha256": manifest[
            "root_predictor_authority_sha256"
        ],
        "runtime_authority_sha256": runtime_authority[
            "runtime_authority_sha256"
        ],
        "production_evaluation400_execution_authorized": True,
        "scientific_promotion_eligible": False,
    }


@dataclass(frozen=True)
class ProductionRootPredictionResult:
    raw_predictions: Mapping[str, np.ndarray]
    auxiliary_tensors: Mapping[str, np.ndarray]
    derivation_commitment: Mapping[str, Any]
    member_call_count: int


@dataclass(frozen=True)
class _Member:
    model: adapter_trainer.SmolVLAPiperAdapter
    recovery: adapter_trainer.DetachedConditionalRecoveryAdapter
    object_mean: torch.Tensor
    object_std: torch.Tensor


def _validate_converted_member(
    value: Mapping[str, Any], *, member_record: Mapping[str, Any],
) -> _Member:
    fields = {
        "format", "status", "model_family", "member_index",
        "source_checkpoint_file_sha256", "adapter_checkpoint_file_sha256",
        "model_config", "model_config_sha256", "state_rank", "action_rank",
        "source_rank_score_contract", "source_rank_score_contract_sha256",
        "object_delta_mean", "object_delta_std",
        "object_source_normalization_sha256", "model_state_dict",
        "model_state_dict_sha256", "recovery_state_dict",
        "recovery_state_dict_sha256", "recovery_shared_transition_stop_gradient",
        "checkpoint_logical_sha256",
    }
    item = _exact(value, fields, f"converted member {member_record['member_index']}")
    model_state = item["model_state_dict"]
    recovery_state = item["recovery_state_dict"]
    logical_base = {
        name: child for name, child in item.items()
        if name not in {
            "model_state_dict", "recovery_state_dict",
            "object_delta_mean", "object_delta_std",
            "checkpoint_logical_sha256",
        }
    }
    if (
        item["format"] != MEMBER_CHECKPOINT_FORMAT
        or item["status"] != "frozen_real_production_model_family_weights_only"
        or item["model_family"] != MODEL_FAMILY
        or item["member_index"] != member_record["member_index"]
        or type(item["member_index"]) is not int
        or type(item["state_rank"]) is not int
        or item["state_rank"] < 1
        or type(item["action_rank"]) is not int
        or item["action_rank"] < 1
        or item["source_checkpoint_file_sha256"]
        != member_record["source_checkpoint_file_sha256"]
        or item["adapter_checkpoint_file_sha256"]
        != member_record["adapter_checkpoint_file_sha256"]
        or item["model_config_sha256"] != canonical_sha256(item["model_config"])
        or item["model_config_sha256"] != member_record["model_config_sha256"]
        or item["source_rank_score_contract_sha256"]
        != item["source_rank_score_contract"].get("contract_sha256")
        or item["source_rank_score_contract_sha256"]
        != member_record["source_rank_score_contract_sha256"]
        or item["object_source_normalization_sha256"]
        != member_record["object_source_normalization_sha256"]
        or not isinstance(model_state, Mapping)
        or item["model_state_dict_sha256"] != state_dict_sha256(model_state)
        or not isinstance(recovery_state, Mapping)
        or item["recovery_state_dict_sha256"] != state_dict_sha256(recovery_state)
        or item["recovery_shared_transition_stop_gradient"] is not True
        or item["checkpoint_logical_sha256"] != canonical_sha256(logical_base)
        or item["checkpoint_logical_sha256"]
        != member_record["converted_checkpoint_logical_sha256"]
    ):
        raise ProductionRootPredictorError("converted production member changed")
    try:
        config = event_world_model.EventWorldModelConfig.from_dict(item["model_config"])
        adapter_trainer.validate_production_source_rank_config(config)
        rank = adapter_trainer._validate_source_rank_score_contract(
            item["source_rank_score_contract"]
        )
        if rank["source_checkpoint_file_sha256"] != item["source_checkpoint_file_sha256"]:
            raise ProductionRootPredictorError("member rank/source binding changed")
        native = event_world_model.ActionConditionedEventWorldModel(config)
        model = adapter_trainer.SmolVLAPiperAdapter(
            native,
            state_rank=int(item["state_rank"]),
            action_rank=int(item["action_rank"]),
            source_rank_contract=rank,
        )
        model.load_state_dict(model_state, strict=True)
        recovery = adapter_trainer.DetachedConditionalRecoveryAdapter(
            config.semantic_dim
        )
        recovery.load_state_dict(recovery_state, strict=True)
    except ProductionRootPredictorError:
        raise
    except Exception as error:
        raise ProductionRootPredictorError(
            "converted checkpoint does not reconstruct the real model family"
        ) from error
    mean = torch.as_tensor(item["object_delta_mean"], dtype=torch.float32)
    std = torch.as_tensor(item["object_delta_std"], dtype=torch.float32)
    normalization = {
        "object_delta_mean": mean.numpy(), "object_delta_std": std.numpy()
    }
    if (
        mean.shape != (config.object_delta_dim,)
        or std.shape != mean.shape
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(std).all())
        or bool((std <= 0).any())
        or _array_bundle_sha256(normalization)
        != item["object_source_normalization_sha256"]
    ):
        raise ProductionRootPredictorError("converted object normalization changed")
    model.eval()
    recovery.eval()
    for parameter in (*model.parameters(), *recovery.parameters()):
        parameter.requires_grad_(False)
    return _Member(model=model, recovery=recovery, object_mean=mean, object_std=std)


class ProductionRootPredictorRuntimeV1:
    """Pinned real-family runtime over actor-visible q0 evidence."""

    def __init__(
        self, *, artifact_root: Path, authority_sha256: str,
        authority_file_sha256: str,
        file_shas: Mapping[str, str], runtime_authority: Mapping[str, Any],
        calibration: Mapping[str, Any], uncertainty: Mapping[str, Any],
        members: Sequence[_Member],
    ) -> None:
        self._artifact_root = artifact_root
        self._authority_sha256 = authority_sha256
        self._authority_file_sha256 = authority_file_sha256
        self._file_shas = dict(file_shas)
        self._runtime_authority = dict(runtime_authority)
        self._calibration = dict(calibration)
        self._uncertainty = dict(uncertainty)
        self._members = tuple(members)

    @property
    def authority_sha256(self) -> str:
        return self._authority_sha256

    @property
    def production_evaluation400_execution_authorized(self) -> bool:
        return True

    @property
    def scientific_promotion_eligible(self) -> bool:
        return False

    @property
    def model_family(self) -> str:
        return MODEL_FAMILY

    def _assert_unchanged(self) -> None:
        if len(self._members) != MEMBER_COUNT or any(
            type(member.model) is not adapter_trainer.SmolVLAPiperAdapter
            or type(member.model.core)
            is not event_world_model.ActionConditionedEventWorldModel
            or type(member.recovery)
            is not adapter_trainer.DetachedConditionalRecoveryAdapter
            or member.model.training
            or member.model.core.training
            or member.recovery.training
            or any(
                parameter.requires_grad
                for parameter in (*member.model.parameters(), *member.recovery.parameters())
            )
            for member in self._members
        ):
            raise ProductionRootPredictorError(
                "runtime no longer contains the exact frozen real model family"
            )
        expected_names = {AUTHORITY_BASENAME, *self._file_shas}
        if {child.name for child in self._artifact_root.iterdir()} != expected_names:
            raise ProductionRootPredictorError("artifact inventory changed after load")
        if (
            file_sha256(self._artifact_root / AUTHORITY_BASENAME)
            != self._authority_file_sha256
        ):
            raise ProductionRootPredictorError(
                "artifact authority changed after load"
            )
        for basename, expected in self._file_shas.items():
            if file_sha256(self._artifact_root / basename) != expected:
                raise ProductionRootPredictorError(f"artifact {basename} changed after load")
        if _implementation_contract() != self._runtime_authority["implementation"]:
            raise ProductionRootPredictorError("production implementation changed after load")

    def predict(
        self, observation_commitment: Mapping[str, Any], *,
        expected_pair_id: str, expected_pair_ordinal: int,
        expected_shared_snapshot_sha256: str,
        expected_pre_action_snapshot_sha256: str,
        expected_observer_authority_sha256: str,
        expected_observer_actor_adapter_contract_sha256: str,
        observer_output_receipt: Mapping[str, Any],
        actor_visible_inputs: Mapping[str, Any],
        ordered_candidate_sha256: Sequence[str], candidate_legal: Sequence[bool],
        lowest_legal_original_candidate_index: int, mapped_actions: Any,
    ) -> ProductionRootPredictionResult:
        self._assert_unchanged()
        if len(self._members) != MEMBER_COUNT:
            raise ProductionRootPredictorError("runtime does not contain five members")
        try:
            root_contract.validate_root_observation_commitment(
                observation_commitment,
                expected_pair_id=expected_pair_id,
                expected_pair_ordinal=expected_pair_ordinal,
                expected_shared_snapshot_sha256=expected_shared_snapshot_sha256,
                expected_pre_action_snapshot_sha256=expected_pre_action_snapshot_sha256,
                expected_observer_authority_sha256=expected_observer_authority_sha256,
                expected_observer_actor_adapter_contract_sha256=(
                    expected_observer_actor_adapter_contract_sha256
                ),
                observer_output_receipt=observer_output_receipt,
                actor_visible_inputs=actor_visible_inputs,
                ordered_candidate_sha256=ordered_candidate_sha256,
                candidate_legal=candidate_legal,
                lowest_legal_original_candidate_index=(
                    lowest_legal_original_candidate_index
                ),
                mapped_actions=mapped_actions,
            )
        except root_contract.RootObservedContractError as error:
            raise ProductionRootPredictorError(
                "actor-visible root contract failed before real-member inference"
            ) from error
        history = np.asarray(actor_visible_inputs["history"])
        history_mask = np.asarray(actor_visible_inputs["history_mask"])
        proprio = np.asarray(actor_visible_inputs["proprio"])
        actions = np.asarray(mapped_actions)
        legal = np.asarray(candidate_legal)
        legal_indices = np.flatnonzero(legal).astype(np.int64)
        config = self._members[0].model.core.config
        if (
            history.dtype != np.float32
            or history.shape != (root_contract.HISTORY_STEPS, config.state_input_dim)
            or history_mask.dtype != np.bool_
            or history_mask.shape != (root_contract.HISTORY_STEPS,)
            or proprio.dtype != np.float32
            or proprio.shape != (config.proprio_dim,)
            or actions.dtype != np.float32
            or actions.ndim != 3
            or actions.shape[0] != len(candidate_legal)
            or actions.shape[2] != config.action_dim
            or len(legal_indices) < 1
            or not np.isfinite(history).all()
            or not np.isfinite(proprio).all()
            or not np.isfinite(actions).all()
        ):
            raise ProductionRootPredictorError("production root tensors changed")
        count = len(legal_indices)
        device = next(self._members[0].model.parameters()).device
        event_id = int(observer_output_receipt["current_event_id"])
        predicates = np.asarray(
            [
                float(observer_output_receipt["current_predicates"][name])
                for name in root_contract.PREDICATE_NAMES
            ],
            dtype=np.float32,
        )
        batch = {
            "state": torch.as_tensor(
                np.repeat(history[None], count, axis=0), device=device
            ),
            "actions": torch.as_tensor(actions[legal_indices], device=device),
            "action_mask": torch.as_tensor(
                np.arange(actions.shape[1])[None] == 0, device=device
            ).expand(count, -1),
            "proprio": torch.as_tensor(
                np.repeat(proprio[None], count, axis=0), device=device
            ),
            "current_event_id": torch.full(
                (count,), event_id, dtype=torch.long, device=device
            ),
            "current_predicates": torch.as_tensor(
                np.repeat(predicates[None], count, axis=0), device=device
            ),
            "history_mask": torch.as_tensor(
                np.repeat(history_mask[None], count, axis=0), device=device
            ),
            "dt": torch.ones(count, dtype=torch.float32, device=device),
            "ranking_group_index": torch.zeros(count, dtype=torch.long, device=device),
            "ranking_candidate_index": torch.as_tensor(
                legal_indices, dtype=torch.long, device=device
            ),
            "ranking_baseline_mask": torch.as_tensor(
                legal_indices == lowest_legal_original_candidate_index,
                dtype=torch.bool, device=device,
            ),
            "ranking_group_count": 1,
            "ranking_logical_group_id": [f"evaluation400:{expected_pair_id}:root"] * count,
            "logical_group_id": [f"evaluation400:{expected_pair_id}:root"] * count,
        }
        outputs: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
        with torch.inference_mode():
            for member in self._members:
                output = dict(member.model.predict_grouped_candidates(batch))
                output["duration_log_mean"] = output["duration_selected_log_mean"]
                output["duration_log_scale"] = (
                    output["duration_selected_log_scale"]
                    + math.log(float(self._calibration["duration_scale_multiplier"]))
                )
                output["object_mean"] = (
                    output["object_delta_mean"] * member.object_std.to(device)
                    + member.object_mean.to(device)
                )
                output["object_log_scale"] = (
                    output["object_delta_log_scale"]
                    + member.object_std.to(device).log()
                    + math.log(float(self._calibration["object_scale_multiplier"]))
                )
                recovery = member.recovery(output["transition"])
                checked = (
                    "next_event_logits", "next_reached_event_logits",
                    "duration_log_mean", "duration_log_scale", "success_logit",
                    "object_mean", "object_log_scale", "source_contract_rank_score",
                )
                if any(not bool(torch.isfinite(output[name]).all()) for name in checked) \
                   or not bool(torch.isfinite(recovery).all()):
                    raise ProductionRootPredictorError("real member produced non-finite tensors")
                outputs.append((output, recovery))
        if len(outputs) != MEMBER_COUNT:
            raise ProductionRootPredictorError("five-member execution count changed")

        def stacked(name: str) -> np.ndarray:
            value = torch.stack([row[0][name] for row in outputs])
            return np.ascontiguousarray(value.cpu().numpy().astype(np.float32, copy=False))

        raw_predictions = {
            "post_event_logits": stacked("next_event_logits"),
            "next_event_logits": stacked("next_reached_event_logits"),
            "duration_log_mean": stacked("duration_log_mean"),
            "duration_log_scale": stacked("duration_log_scale"),
            "success_logit": stacked("success_logit"),
            "object_mean": stacked("object_mean"),
            "object_log_scale": stacked("object_log_scale"),
        }
        recovery_logits = np.ascontiguousarray(
            torch.stack([row[1] for row in outputs]).cpu().numpy().astype(np.float32)
        )
        uncertainty_inputs = {**raw_predictions, "recovery_logit": recovery_logits}
        parameters = {
            name: self._calibration[name]
            for name in (
                "post_event_temperature", "next_event_temperature",
                "success_temperature", "conditional_recovery_temperature",
                "object_error_robust_scale_m",
            )
        }
        try:
            components = deployment_uncertainty.root_components(
                predictions=uncertainty_inputs, parameters=parameters
            )
        except deployment_uncertainty.DeploymentUncertaintyError as error:
            raise ProductionRootPredictorError("root uncertainty failed") from error
        calibrated_success = 1.0 / (
            1.0 + np.exp(-np.clip(
                raw_predictions["success_logit"]
                / float(self._calibration["success_temperature"]), -40.0, 40.0
            ))
        )
        auxiliary = {
            "member_calibrated_success_probability": np.ascontiguousarray(
                calibrated_success.astype(np.float32)
            ),
            "member_composite_rank_score": stacked("source_contract_rank_score"),
            "candidate_structured_five_head_uncertainty": np.ascontiguousarray(
                np.asarray(components["structured_five_head"], dtype=np.float32)
            ),
        }
        chronology = dict(observation_commitment["chronology"])
        chronology["root_world_model_member_calls"] = MEMBER_COUNT
        derivation = root_contract.build_root_prediction_derivation_commitment(
            observation_commitment=observation_commitment,
            raw_predictions=raw_predictions,
            auxiliary_tensors=auxiliary,
            root_predictor_authority_sha256=self.authority_sha256,
            calibration_sha256=self._calibration["calibration_sha256"],
            source_rank_contract_set_sha256=self._runtime_authority[
                "source_rank_contract_set_sha256"
            ],
            uncertainty_contract_sha256=self._uncertainty[
                "uncertainty_contract_sha256"
            ],
            derivation_implementation_file_sha256=self._runtime_authority[
                "implementation"
            ]["root_observed_contract_file_sha256"],
            chronology=chronology,
        )
        result = ProductionRootPredictionResult(
            raw_predictions=raw_predictions,
            auxiliary_tensors=auxiliary,
            derivation_commitment=derivation,
            member_call_count=MEMBER_COUNT,
        )
        self.validate_prediction_result(result, observation_commitment=observation_commitment)
        return result

    def validate_prediction_result(
        self, result: ProductionRootPredictionResult, *,
        observation_commitment: Mapping[str, Any],
    ) -> str:
        self._assert_unchanged()
        if (
            type(result) is not ProductionRootPredictionResult
            or result.member_call_count != MEMBER_COUNT
            or type(result.member_call_count) is not int
        ):
            raise ProductionRootPredictorError("production result member count changed")
        try:
            return root_contract.validate_root_prediction_derivation_commitment(
                result.derivation_commitment,
                observation_commitment=observation_commitment,
                raw_predictions=result.raw_predictions,
                auxiliary_tensors=result.auxiliary_tensors,
                expected_root_predictor_authority_sha256=self.authority_sha256,
                expected_calibration_sha256=self._calibration["calibration_sha256"],
                expected_source_rank_contract_set_sha256=self._runtime_authority[
                    "source_rank_contract_set_sha256"
                ],
                expected_uncertainty_contract_sha256=self._uncertainty[
                    "uncertainty_contract_sha256"
                ],
                expected_derivation_implementation_file_sha256=self._runtime_authority[
                    "implementation"
                ]["root_observed_contract_file_sha256"],
            )
        except root_contract.RootObservedContractError as error:
            raise ProductionRootPredictorError(
                "production prediction provenance or tensors changed"
            ) from error


def load_production_root_predictor_runtime(
    artifact_root: Path, *, expected_root_predictor_authority_sha256: str,
) -> ProductionRootPredictorRuntimeV1:
    expected = _sha(
        expected_root_predictor_authority_sha256,
        "expected production root predictor authority",
    )
    root = Path(artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ProductionRootPredictorError("artifact root must be a real directory")
    root = root.resolve()
    manifest_raw = _read_bytes(
        root / AUTHORITY_BASENAME, maximum=MAX_JSON_BYTES, role="artifact authority"
    )
    manifest = _parse_json(manifest_raw, "artifact authority")
    manifest_fields = {
        "format", "status", "model_family", "member_count", "artifact_inventory",
        "runtime_authority_sha256", "converted_checkpoint_file_set_sha256",
        "source_rank_contract_set_sha256",
        "production_evaluation400_execution_authorized",
        "scientific_promotion_eligible",
    }
    logical = _verify_signed(
        manifest, fields=manifest_fields,
        digest_field="root_predictor_authority_sha256", role="artifact authority",
    )
    if (
        logical != expected
        or manifest["format"] != ARTIFACT_AUTHORITY_FORMAT
        or manifest["status"]
        != "production_evaluation400_execution_not_scientific_promotion"
        or manifest["model_family"] != MODEL_FAMILY
        or manifest["member_count"] != MEMBER_COUNT
        or type(manifest["member_count"]) is not int
        or manifest["production_evaluation400_execution_authorized"] is not True
        or manifest["scientific_promotion_eligible"] is not False
    ):
        raise ProductionRootPredictorError("artifact authority is not externally pinned")
    expected_roles = {
        EXECUTION_AUTHORITY_BASENAME: "execution_authority",
        CALIBRATION_BASENAME: "calibration",
        UNCERTAINTY_BASENAME: "uncertainty_contract",
        RUNTIME_AUTHORITY_BASENAME: "runtime_authority",
        **{f"member_{index:02d}.pt": "real_member_checkpoint" for index in range(MEMBER_COUNT)},
    }
    inventory = manifest["artifact_inventory"]
    if not isinstance(inventory, list) or len(inventory) != len(expected_roles):
        raise ProductionRootPredictorError("artifact inventory size changed")
    records: dict[str, Mapping[str, Any]] = {}
    for raw_record in inventory:
        record = _exact(raw_record, {"basename", "role", "file_sha256"}, "artifact record")
        basename = record["basename"]
        if (
            not isinstance(basename, str)
            or basename in records
            or basename not in expected_roles
            or Path(basename).name != basename
            or record["role"] != expected_roles[basename]
            or not _is_sha(record["file_sha256"])
        ):
            raise ProductionRootPredictorError("artifact inventory record changed")
        records[basename] = record
    if (
        set(records) != set(expected_roles)
        or [record["basename"] for record in inventory] != sorted(expected_roles)
        or {child.name for child in root.iterdir()}
        != {AUTHORITY_BASENAME, *expected_roles}
    ):
        raise ProductionRootPredictorError("artifact inventory names changed")
    file_shas = {name: str(records[name]["file_sha256"]) for name in records}
    for basename, file_sha in file_shas.items():
        if file_sha256(root / basename) != file_sha:
            raise ProductionRootPredictorError(f"artifact {basename} file SHA changed")
    runtime_authority, _ = _read_hashed_json(
        root / RUNTIME_AUTHORITY_BASENAME,
        file_shas[RUNTIME_AUTHORITY_BASENAME], "runtime authority",
    )
    runtime_fields = {
        "format", "status", "model_family", "member_count", "members",
        "converted_checkpoint_file_set_sha256", "source_rank_contract_set_sha256",
        "execution_authority", "calibration", "uncertainty_contract",
        "implementation", "production_evaluation400_execution_authorized",
        "scientific_promotion_eligible", "compact_reference_model_allowed",
        "online_input_contract", "member_call_contract",
    }
    runtime_sha = _verify_signed(
        runtime_authority, fields=runtime_fields,
        digest_field="runtime_authority_sha256", role="runtime authority",
    )
    if (
        runtime_sha != manifest["runtime_authority_sha256"]
        or runtime_authority["format"] != RUNTIME_AUTHORITY_FORMAT
        or runtime_authority["status"]
        != "real_model_family_frozen_for_evaluation400_execution"
        or runtime_authority["model_family"] != MODEL_FAMILY
        or runtime_authority["member_count"] != MEMBER_COUNT
        or type(runtime_authority["member_count"]) is not int
        or runtime_authority["implementation"] != _implementation_contract()
        or runtime_authority["production_evaluation400_execution_authorized"] is not True
        or runtime_authority["scientific_promotion_eligible"] is not False
        or runtime_authority["compact_reference_model_allowed"] is not False
        or runtime_authority["online_input_contract"]
        != "validated_actor_visible_root_observation_and_candidate_actions_only"
        or runtime_authority["member_call_contract"]
        != "exactly_five_real_members_once_each"
    ):
        raise ProductionRootPredictorError("runtime authority changed")
    execution_authority, _ = _read_hashed_json(
        root / EXECUTION_AUTHORITY_BASENAME,
        file_shas[EXECUTION_AUTHORITY_BASENAME], "execution authority",
    )
    execution_reference = runtime_authority["execution_authority"]
    if not isinstance(execution_reference, Mapping) or execution_reference != {
        "basename": EXECUTION_AUTHORITY_BASENAME,
        "file_sha256": file_shas[EXECUTION_AUTHORITY_BASENAME],
        "logical_sha256": execution_authority.get("execution_authority_sha256"),
    }:
        raise ProductionRootPredictorError("execution authority reference changed")
    execution_authority = validate_execution_authority(
        execution_authority,
        expected_authority_sha256=execution_reference["logical_sha256"],
    )
    calibration, _ = _read_hashed_json(
        root / CALIBRATION_BASENAME, file_shas[CALIBRATION_BASENAME], "calibration"
    )
    calibration_fields = {
        "format", "status", "source_calibration_file_sha256",
        "source_calibration_sha256", "root_group_ranker_sha256",
        "post_event_temperature", "next_event_temperature", "success_temperature",
        "conditional_recovery_temperature", "duration_scale_multiplier",
        "object_scale_multiplier", "object_error_robust_scale_m",
        "maximum_total_uncertainty", "all_six_heads_enabled",
        "formal190_selection_aware_gate_passed", "scientific_promotion_eligible",
        "parameter_application",
    }
    calibration_sha = _verify_signed(
        calibration, fields=calibration_fields,
        digest_field="calibration_sha256", role="calibration",
    )
    if (
        calibration != _calibration_document(execution_authority)
        or runtime_authority["calibration"] != {
            "basename": CALIBRATION_BASENAME,
            "file_sha256": file_shas[CALIBRATION_BASENAME],
            "logical_sha256": calibration_sha,
        }
    ):
        raise ProductionRootPredictorError("calibration cross-binding changed")
    uncertainty, _ = _read_hashed_json(
        root / UNCERTAINTY_BASENAME,
        file_shas[UNCERTAINTY_BASENAME], "uncertainty contract",
    )
    uncertainty_fields = {
        "format", "status", "root_included_heads", "root_head_count",
        "root_recovery_policy", "calibration_sha256",
        "implementation_file_sha256",
    }
    uncertainty_sha = _verify_signed(
        uncertainty, fields=uncertainty_fields,
        digest_field="uncertainty_contract_sha256", role="uncertainty contract",
    )
    if (
        uncertainty != _uncertainty_document(calibration)
        or runtime_authority["uncertainty_contract"] != {
            "basename": UNCERTAINTY_BASENAME,
            "file_sha256": file_shas[UNCERTAINTY_BASENAME],
            "logical_sha256": uncertainty_sha,
        }
    ):
        raise ProductionRootPredictorError("uncertainty cross-binding changed")
    member_records = runtime_authority["members"]
    if not isinstance(member_records, list) or len(member_records) != MEMBER_COUNT:
        raise ProductionRootPredictorError("runtime authority lacks five members")
    members: list[_Member] = []
    checkpoint_shas: list[str] = []
    rank_shas: list[str] = []
    config_shas: list[str] = []
    normalization_shas: list[str] = []
    member_fields = {
        "member_index", "source_checkpoint_file_sha256",
        "adapter_checkpoint_file_sha256", "source_rank_score_contract_sha256",
        "object_source_normalization_sha256", "model_config_sha256",
        "converted_checkpoint_logical_sha256", "converted_checkpoint_basename",
        "converted_checkpoint_file_sha256",
    }
    for index, raw_record in enumerate(member_records):
        record = _exact(raw_record, member_fields, f"runtime member {index}")
        basename = f"member_{index:02d}.pt"
        if (
            record["member_index"] != index
            or type(record["member_index"]) is not int
            or record["converted_checkpoint_basename"] != basename
            or record["converted_checkpoint_file_sha256"] != file_shas[basename]
            or record["source_checkpoint_file_sha256"]
            != execution_authority["source_checkpoint_file_sha256"][index]
            or record["adapter_checkpoint_file_sha256"]
            != execution_authority["adapter_checkpoint_file_sha256"][index]
            or record["source_rank_score_contract_sha256"]
            != execution_authority["source_rank_score_contract_sha256"][index]
            or record["object_source_normalization_sha256"]
            != execution_authority["object_source_normalization_sha256"][index]
            or record["model_config_sha256"] != execution_authority["model_config_sha256"]
        ):
            raise ProductionRootPredictorError(f"runtime member {index} binding changed")
        raw = _read_bytes(
            root / basename, maximum=MAX_CHECKPOINT_BYTES,
            role=f"converted member {index}",
        )
        checkpoint = _safe_torch_bytes(raw, f"converted member {index}")
        members.append(_validate_converted_member(checkpoint, member_record=record))
        checkpoint_shas.append(file_shas[basename])
        rank_shas.append(record["source_rank_score_contract_sha256"])
        config_shas.append(record["model_config_sha256"])
        normalization_shas.append(record["object_source_normalization_sha256"])
    if (
        canonical_sha256(checkpoint_shas)
        != runtime_authority["converted_checkpoint_file_set_sha256"]
        or runtime_authority["converted_checkpoint_file_set_sha256"]
        != manifest["converted_checkpoint_file_set_sha256"]
        or canonical_sha256(rank_shas)
        != runtime_authority["source_rank_contract_set_sha256"]
        or runtime_authority["source_rank_contract_set_sha256"]
        != manifest["source_rank_contract_set_sha256"]
        or len(set(config_shas)) != 1
        or len(set(normalization_shas)) != 1
    ):
        raise ProductionRootPredictorError("five-member set contract changed")
    return ProductionRootPredictorRuntimeV1(
        artifact_root=root,
        authority_sha256=logical,
        authority_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        file_shas=file_shas,
        runtime_authority=runtime_authority,
        calibration=calibration,
        uncertainty=uncertainty,
        members=members,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--execution-authority-file-sha256", required=True)
    parser.add_argument("--execution-authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = convert_production_root_predictor_artifacts(
        source_checkpoint_paths=args.source_checkpoint,
        adapter_checkpoint_paths=args.adapter_checkpoint,
        execution_authority_path=args.execution_authority,
        expected_execution_authority_file_sha256=(
            args.execution_authority_file_sha256
        ),
        expected_execution_authority_sha256=args.execution_authority_sha256,
        output_directory=args.output,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_AUTHORITY_FORMAT",
    "CALIBRATION_FORMAT",
    "EXECUTION_AUTHORITY_FORMAT",
    "MEMBER_CHECKPOINT_FORMAT",
    "MEMBER_COUNT",
    "MODEL_FAMILY",
    "ProductionRootPredictionResult",
    "ProductionRootPredictorError",
    "ProductionRootPredictorRuntimeV1",
    "RUNTIME_AUTHORITY_FORMAT",
    "UNCERTAINTY_FORMAT",
    "canonical_sha256",
    "convert_production_root_predictor_artifacts",
    "file_sha256",
    "load_production_root_predictor_runtime",
    "state_dict_sha256",
    "validate_execution_authority",
]
