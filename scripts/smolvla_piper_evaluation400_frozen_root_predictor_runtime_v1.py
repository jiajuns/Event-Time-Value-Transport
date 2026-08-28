#!/usr/bin/env python3
"""Reference-only frozen five-member predictor over actor-visible evidence.

The runtime accepts an externally pinned, content-addressed artifact directory.
It never reads evaluation targets, object poses, trajectories, or simulator
state.  Every online prediction starts by rebuilding the root-observation
contract from the original actor-visible arrays, then executes exactly one
forward call for each of five frozen members.  Raw and derived tensors are
committed and immediately rebuilt before a result can leave this module.

CompactRootPredictorV1 is deliberately a small contract/integration test model;
it is not ActionConditionedEventWorldModel or PiperAdaptedWorldModel and is
cryptographically marked promotion-ineligible throughout its authority chain.
Production evaluation must add and require a separate real-model-family loader.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import smolvla_piper_causal_event_observer_v1 as observer
import smolvla_piper_deployment_uncertainty_v1 as deployment_uncertainty
import smolvla_piper_evaluation400_root_observed_contract_v1 as root_contract


MANIFEST_FORMAT = "etsf_smolvla_piper_frozen_root_predictor_authority_v1"
RUNTIME_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_frozen_root_predictor_runtime_authority_v1"
)
EVENT_SPEC_FORMAT = "etsf_smolvla_piper_root_predictor_event_spec_v1"
CALIBRATION_FORMAT = "etsf_smolvla_piper_root_predictor_calibration_v1"
UNCERTAINTY_FORMAT = "etsf_smolvla_piper_root_predictor_uncertainty_contract_v1"
RANK_CONTRACT_FORMAT = "etsf_smolvla_piper_root_source_rank_contract_v1"
CHECKPOINT_FORMAT = "etsf_smolvla_piper_compact_root_predictor_checkpoint_v1"

MEMBER_COUNT = 5
MODEL_FAMILY = "CompactRootPredictorV1_reference_test_only"
AUTHORITY_MANIFEST_BASENAME = "authority_manifest.json"
RUNTIME_AUTHORITY_BASENAME = "runtime_authority.json"
EVENT_SPEC_BASENAME = "event_spec.json"
CALIBRATION_BASENAME = "calibration.json"
UNCERTAINTY_BASENAME = "uncertainty_contract.json"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
PREDICATE_NAMES = root_contract.PREDICATE_NAMES


class FrozenRootPredictorError(RuntimeError):
    """Frozen artifact or pre-condition inference evidence failed closed."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FrozenRootPredictorError("authority is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha(value: Any, role: str) -> str:
    if not _is_sha(value):
        raise FrozenRootPredictorError(f"{role} must be exact SHA-256")
    return str(value)


def _exact(value: Any, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FrozenRootPredictorError(f"{role} fields changed")
    return value


def _integer(
    value: Any, role: str, *, minimum: int = 0, maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise FrozenRootPredictorError(f"{role} is outside the frozen range")
    return value


def _positive(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise FrozenRootPredictorError(f"{role} must be finite and positive")
    return float(value)


def _finite_nonnegative(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise FrozenRootPredictorError(f"{role} must be finite and non-negative")
    return float(value)


def _signed_sha(
    value: Any, *, fields: set[str], digest_field: str, role: str,
) -> str:
    item = _exact(value, fields | {digest_field}, role)
    digest = _sha(item[digest_field], f"{role} logical SHA")
    unsigned = {name: child for name, child in item.items() if name != digest_field}
    if digest != canonical_sha256(unsigned):
        raise FrozenRootPredictorError(f"{role} logical SHA changed")
    return digest


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise FrozenRootPredictorError("JSON contains duplicate object keys")
        result[name] = value
    return result


def _parse_json_bytes(raw: bytes, role: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FrozenRootPredictorError(f"{role} contains {value}")
            ),
        )
    except FrozenRootPredictorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenRootPredictorError(f"{role} is not strict UTF-8 JSON") from error


def _read_bytes(path: Path, *, maximum: int, role: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FrozenRootPredictorError(f"{role} must be one regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise FrozenRootPredictorError(f"{role} file size escaped its frozen bound")
    return path.read_bytes()


def _read_hashed_json(path: Path, expected_file_sha: str, role: str) -> Any:
    raw = _read_bytes(path, maximum=MAX_JSON_BYTES, role=role)
    if hashlib.sha256(raw).hexdigest() != expected_file_sha:
        raise FrozenRootPredictorError(f"{role} file SHA changed")
    return _parse_json_bytes(raw, role)


def _module_file_sha(module: Any, role: str) -> str:
    path = Path(module.__file__).resolve()
    if not path.is_file():
        raise FrozenRootPredictorError(f"{role} implementation is unavailable")
    return file_sha256(path)


def _runtime_file_sha() -> str:
    path = Path(__file__).resolve()
    if not path.is_file():
        raise FrozenRootPredictorError("runtime implementation is unavailable")
    return file_sha256(path)


def _tensor_digest(name: str, tensor: torch.Tensor) -> str:
    if (
        type(tensor) is not torch.Tensor
        or tensor.layout != torch.strided
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all())
    ):
        raise FrozenRootPredictorError(
            f"checkpoint tensor {name} must be finite contiguous float32"
        )
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"ETSF/evaluation400/compact-root-state-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0float32\0")
    digest.update(canonical_sha256(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(value: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(value, Mapping) or not value:
        raise FrozenRootPredictorError("checkpoint state_dict is missing")
    names = sorted(value)
    if any(not isinstance(name, str) or not name or not name.isascii() for name in names):
        raise FrozenRootPredictorError("checkpoint state_dict names are invalid")
    return canonical_sha256(
        [{"name": name, "tensor_sha256": _tensor_digest(name, value[name])} for name in names]
    )


class CompactRootPredictorV1(nn.Module):
    """Small fixed reference architecture; never a production model substitute."""

    def __init__(
        self, *, input_dim: int, hidden_dim: int, event_count: int, object_dim: int,
    ) -> None:
        super().__init__()
        self.trunk = nn.Linear(input_dim, hidden_dim)
        self.post_event = nn.Linear(hidden_dim, event_count)
        self.next_event = nn.Linear(hidden_dim, event_count)
        self.duration_mean = nn.Linear(hidden_dim, 1)
        self.duration_log_scale = nn.Linear(hidden_dim, 1)
        self.success = nn.Linear(hidden_dim, 1)
        self.recovery = nn.Linear(hidden_dim, 1)
        self.object_mean = nn.Linear(hidden_dim, object_dim)
        self.object_log_scale = nn.Linear(hidden_dim, object_dim)
        self.action_rank_residual = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = torch.tanh(self.trunk(features))
        return {
            "post_event_logits": self.post_event(hidden),
            "next_event_logits": self.next_event(hidden),
            "duration_log_mean": self.duration_mean(hidden).squeeze(-1),
            "duration_log_scale": self.duration_log_scale(hidden).squeeze(-1),
            "success_logit": self.success(hidden).squeeze(-1),
            "recovery_logit": self.recovery(hidden).squeeze(-1),
            "object_mean": self.object_mean(hidden),
            "object_log_scale": self.object_log_scale(hidden),
            "action_rank_residual": self.action_rank_residual(hidden).squeeze(-1),
        }


@dataclass(frozen=True)
class FrozenRootPredictionResult:
    raw_predictions: Mapping[str, np.ndarray]
    auxiliary_tensors: Mapping[str, np.ndarray]
    derivation_commitment: Mapping[str, Any]
    member_call_count: int


@dataclass(frozen=True)
class _Member:
    model: CompactRootPredictorV1
    rank_contract: Mapping[str, Any]
    checkpoint_file_sha256: str


def _validate_event_spec(value: Any) -> dict[str, Any]:
    fields = {
        "format", "status", "event_names", "history_steps", "state_dim",
        "proprio_dim", "image_feature_dim", "action_horizon", "action_dim",
        "object_dim", "object_output_space", "feature_order",
        "feature_numeric_contract",
    }
    digest = _signed_sha(
        value, fields=fields, digest_field="event_spec_sha256", role="event spec"
    )
    item = dict(value)
    if (
        item["format"] != EVENT_SPEC_FORMAT
        or item["status"] != "frozen_before_evaluation400"
        or item["event_names"] != list(observer.EXPECTED_EVENTS)
        or _integer(item["history_steps"], "history steps", minimum=1)
        != root_contract.HISTORY_STEPS
        or _integer(item["state_dim"], "state dimension", minimum=1)
        != root_contract.STATE_DIM
        or _integer(item["proprio_dim"], "proprio dimension", minimum=1)
        != root_contract.PROPRIO_DIM
        or _integer(item["image_feature_dim"], "image feature dimension", maximum=65536)
        < 0
        or _integer(item["action_horizon"], "action horizon", minimum=1, maximum=4096)
        < 1
        or _integer(item["action_dim"], "action dimension", minimum=1, maximum=4096)
        < 1
        or _integer(item["object_dim"], "object dimension", minimum=1, maximum=1024)
        < 1
        or item["object_output_space"] != "physical_delta_xyz_m"
        or item["feature_order"] != [
            "masked_history_float64_mean_then_float32",
            "proprio",
            "image_feature_if_frozen",
            "observer_event_onehot",
            "observer_predicates_in_frozen_name_order",
            "mapped_action_flat_c_order",
        ]
        or item["feature_numeric_contract"]
        != "native_float32_actor_visible_q0_plus_candidate_action"
    ):
        raise FrozenRootPredictorError("event spec semantics changed")
    item["event_spec_sha256"] = digest
    return item


def _validate_calibration(value: Any, *, event_spec_sha: str) -> dict[str, Any]:
    fields = {
        "format", "status", "event_spec_sha256", "post_event_temperature",
        "next_event_temperature", "success_temperature",
        "conditional_recovery_temperature", "duration_scale_multiplier",
        "object_scale_multiplier", "object_error_robust_scale_m",
        "duration_and_object_scale_application",
    }
    digest = _signed_sha(
        value, fields=fields, digest_field="calibration_sha256", role="calibration"
    )
    item = dict(value)
    if (
        item["format"] != CALIBRATION_FORMAT
        or item["status"] != "formal_validation_frozen_deployment_parameters"
        or item["event_spec_sha256"] != event_spec_sha
        or item["duration_and_object_scale_application"]
        != "add_log_multiplier_exactly_once_inside_frozen_runtime"
    ):
        raise FrozenRootPredictorError("calibration semantics changed")
    for name in (
        "post_event_temperature", "next_event_temperature", "success_temperature",
        "conditional_recovery_temperature", "duration_scale_multiplier",
        "object_scale_multiplier", "object_error_robust_scale_m",
    ):
        _positive(item[name], f"calibration {name}")
    item["calibration_sha256"] = digest
    return item


def _validate_uncertainty(
    value: Any, *, event_spec_sha: str, calibration_sha: str,
) -> dict[str, Any]:
    fields = {
        "format", "status", "event_spec_sha256", "calibration_sha256",
        "root_included_heads", "root_head_count", "root_recovery_policy",
        "algorithm", "implementation_file_sha256",
    }
    digest = _signed_sha(
        value, fields=fields, digest_field="uncertainty_contract_sha256",
        role="uncertainty contract",
    )
    item = dict(value)
    actual_sha = _module_file_sha(deployment_uncertainty, "uncertainty")
    if (
        item["format"] != UNCERTAINTY_FORMAT
        or item["status"] != "formal_validation_frozen_five_head_root_uncertainty"
        or item["event_spec_sha256"] != event_spec_sha
        or item["calibration_sha256"] != calibration_sha
        or item["root_included_heads"]
        != list(deployment_uncertainty.ROOT_INCLUDED_HEADS)
        or item["root_head_count"] != deployment_uncertainty.ROOT_HEAD_COUNT
        or type(item["root_head_count"]) is not int
        or item["root_recovery_policy"]
        != deployment_uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY
        or item["algorithm"]
        != "mean_of_five_deployment_dimensionless_head_uncertainties"
        or item["implementation_file_sha256"] != actual_sha
    ):
        raise FrozenRootPredictorError("uncertainty contract semantics changed")
    item["uncertainty_contract_sha256"] = digest
    return item


def _validate_rank_contract(
    value: Any, *, member_index: int, checkpoint_file_sha: str,
    event_spec_sha: str,
) -> dict[str, Any]:
    fields = {
        "format", "status", "member_index", "source_checkpoint_file_sha256",
        "event_spec_sha256", "base_score", "source_action_rank_residual",
        "source_action_rank_success_only", "residual_combination",
        "event_names", "event_values", "duration_scale", "success_temperature",
        "event_weight", "duration_weight", "duration_unit", "numeric_contract",
        "score_is_success_logit", "score_is_success_probability",
    }
    digest = _signed_sha(
        value, fields=fields, digest_field="contract_sha256",
        role=f"member {member_index} rank contract",
    )
    item = dict(value)
    event_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    if (
        item["format"] != RANK_CONTRACT_FORMAT
        or item["status"] != "frozen_source_training_composite_rank"
        or item["member_index"] != member_index
        or type(item["member_index"]) is not int
        or item["source_checkpoint_file_sha256"] != checkpoint_file_sha
        or item["event_spec_sha256"] != event_spec_sha
        or item["base_score"] != "candidate_rank_score"
        or item["source_action_rank_residual"] is not True
        or item["source_action_rank_success_only"] is not False
        or item["residual_combination"]
        != "candidate_rank_score_plus_action_rank_residual_div_success_temperature"
        or item["event_names"] != list(observer.EXPECTED_EVENTS)
        or item["event_values"] != event_values
        or item["duration_unit"] != "decision_steps"
        or item["numeric_contract"]
        != "native_ieee754_float32_training_order"
        or item["score_is_success_logit"] is not False
        or item["score_is_success_probability"] is not False
    ):
        raise FrozenRootPredictorError(f"member {member_index} rank semantics changed")
    if _positive(item["duration_scale"], "rank duration scale") < 1.0:
        raise FrozenRootPredictorError("rank duration scale must be at least one step")
    _positive(item["success_temperature"], "rank success temperature")
    _finite_nonnegative(item["event_weight"], "rank event weight")
    _finite_nonnegative(item["duration_weight"], "rank duration weight")
    item["contract_sha256"] = digest
    return item


def _checkpoint_logical_base(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: child for name, child in value.items()
        if name not in {"state_dict", "checkpoint_logical_sha256"}
    }


def _load_checkpoint(
    raw: bytes, *, member_index: int, event_spec: Mapping[str, Any],
) -> tuple[CompactRootPredictorV1, str]:
    try:
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise FrozenRootPredictorError(
            f"member {member_index} checkpoint cannot be safely loaded"
        ) from error
    fields = {
        "format", "status", "model_family", "promotion_eligible", "member_index",
        "event_spec_sha256", "architecture",
        "input_dim", "hidden_dim", "event_count", "object_dim", "state_dict",
        "state_dict_sha256", "checkpoint_logical_sha256",
    }
    item = _exact(value, fields, f"member {member_index} checkpoint")
    state = item["state_dict"]
    input_dim = (
        int(event_spec["state_dim"])
        + int(event_spec["proprio_dim"])
        + int(event_spec["image_feature_dim"])
        + int(event_spec["action_horizon"]) * int(event_spec["action_dim"])
        + len(event_spec["event_names"])
        + len(PREDICATE_NAMES)
    )
    hidden_dim = _integer(
        item["hidden_dim"], "checkpoint hidden dimension", minimum=1, maximum=4096
    )
    if (
        item["format"] != CHECKPOINT_FORMAT
        or item["status"] != "reference_test_only_frozen_state_dict"
        or item["model_family"] != MODEL_FAMILY
        or item["promotion_eligible"] is not False
        or item["member_index"] != member_index
        or type(item["member_index"]) is not int
        or item["event_spec_sha256"] != event_spec["event_spec_sha256"]
        or item["architecture"] != "CompactRootPredictorV1"
        or item["input_dim"] != input_dim
        or type(item["input_dim"]) is not int
        or item["event_count"] != len(event_spec["event_names"])
        or type(item["event_count"]) is not int
        or item["object_dim"] != event_spec["object_dim"]
        or type(item["object_dim"]) is not int
        or not isinstance(state, Mapping)
        or item["state_dict_sha256"] != state_dict_sha256(state)
        or item["checkpoint_logical_sha256"]
        != canonical_sha256(_checkpoint_logical_base(item))
    ):
        raise FrozenRootPredictorError(f"member {member_index} checkpoint changed")
    model = CompactRootPredictorV1(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_count=len(event_spec["event_names"]),
        object_dim=int(event_spec["object_dim"]),
    )
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        type(state[name]) is not torch.Tensor
        or state[name].dtype != torch.float32
        or tuple(state[name].shape) != tuple(expected[name].shape)
        for name in expected
    ):
        raise FrozenRootPredictorError(f"member {member_index} state shape changed")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise FrozenRootPredictorError(f"member {member_index} state is invalid") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, str(item["checkpoint_logical_sha256"])


class FrozenRootPredictorRuntime:
    """Loaded, pinned five-member runtime; all public failures are fail-closed."""

    def __init__(
        self, *, artifact_root: Path, root_predictor_authority_sha256: str,
        manifest_file_sha256: str, inventory_file_sha256: Mapping[str, str],
        event_spec: Mapping[str, Any], calibration: Mapping[str, Any],
        uncertainty_contract: Mapping[str, Any],
        runtime_authority: Mapping[str, Any], members: Sequence[_Member],
    ) -> None:
        self._artifact_root = artifact_root
        self._root_predictor_authority_sha256 = root_predictor_authority_sha256
        self._manifest_file_sha256 = manifest_file_sha256
        self._inventory_file_sha256 = dict(inventory_file_sha256)
        self._event_spec = dict(event_spec)
        self._calibration = dict(calibration)
        self._uncertainty_contract = dict(uncertainty_contract)
        self._runtime_authority = dict(runtime_authority)
        self._members = tuple(members)

    @property
    def authority_sha256(self) -> str:
        return self._root_predictor_authority_sha256

    @property
    def calibration_sha256(self) -> str:
        return str(self._calibration["calibration_sha256"])

    @property
    def promotion_eligible(self) -> bool:
        """Reference artifacts can never authorize a production condition."""

        return False

    def _assert_frozen_files_unchanged(self) -> None:
        expected_names = {AUTHORITY_MANIFEST_BASENAME, *self._inventory_file_sha256}
        actual_names = {child.name for child in self._artifact_root.iterdir()}
        if actual_names != expected_names:
            raise FrozenRootPredictorError("frozen artifact inventory changed after load")
        expected = {
            AUTHORITY_MANIFEST_BASENAME: self._manifest_file_sha256,
            **self._inventory_file_sha256,
        }
        for basename, digest in expected.items():
            path = self._artifact_root / basename
            raw = _read_bytes(
                path,
                maximum=(MAX_CHECKPOINT_BYTES if basename.endswith(".pt") else MAX_JSON_BYTES),
                role=f"frozen artifact {basename}",
            )
            if hashlib.sha256(raw).hexdigest() != digest:
                raise FrozenRootPredictorError(
                    f"frozen artifact {basename} changed after load"
                )
        if (
            _runtime_file_sha()
            != self._runtime_authority["runtime_implementation_file_sha256"]
            or _module_file_sha(root_contract, "root observed contract")
            != self._runtime_authority[
                "root_observed_contract_implementation_file_sha256"
            ]
            or _module_file_sha(deployment_uncertainty, "uncertainty")
            != self._runtime_authority[
                "deployment_uncertainty_implementation_file_sha256"
            ]
        ):
            raise FrozenRootPredictorError("bound inference implementation changed")

    def _validate_observation(
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
    ) -> str:
        try:
            return root_contract.validate_root_observation_commitment(
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
            raise FrozenRootPredictorError(
                "root observation failed before member inference"
            ) from error

    def _features(
        self, *, actor_visible_inputs: Mapping[str, Any],
        observer_output_receipt: Mapping[str, Any], candidate_legal: Sequence[bool],
        mapped_actions: Any,
    ) -> torch.Tensor:
        history = np.asarray(actor_visible_inputs["history"])
        mask = np.asarray(actor_visible_inputs["history_mask"])
        proprio = np.asarray(actor_visible_inputs["proprio"])
        image = actor_visible_inputs["image_feature"]
        actions = np.asarray(mapped_actions)
        specification = self._event_spec
        expected_action_shape = (
            len(candidate_legal), specification["action_horizon"],
            specification["action_dim"],
        )
        if (
            history.dtype != np.float32
            or history.shape != (specification["history_steps"], specification["state_dim"])
            or mask.dtype != np.bool_
            or mask.shape != (specification["history_steps"],)
            or proprio.dtype != np.float32
            or proprio.shape != (specification["proprio_dim"],)
            or actions.dtype != np.float32
            or actions.shape != expected_action_shape
            or not history.flags.c_contiguous
            or not mask.flags.c_contiguous
            or not proprio.flags.c_contiguous
            or not actions.flags.c_contiguous
        ):
            raise FrozenRootPredictorError("actor-visible predictor tensor shape changed")
        image_dim = int(specification["image_feature_dim"])
        if image_dim == 0:
            if image is not None:
                raise FrozenRootPredictorError("unfrozen image feature appeared")
            image_array = np.empty((0,), dtype=np.float32)
        else:
            image_array = np.asarray(image)
            if (
                image_array.dtype != np.float32
                or image_array.shape != (image_dim,)
                or not image_array.flags.c_contiguous
                or not np.isfinite(image_array).all()
            ):
                raise FrozenRootPredictorError("image feature differs from event spec")
        if not np.isfinite(history).all() or not np.isfinite(proprio).all() or not np.isfinite(actions).all():
            raise FrozenRootPredictorError("actor-visible predictor tensors are non-finite")
        valid = np.asarray(candidate_legal, dtype=np.bool_)
        legal_indices = np.flatnonzero(valid)
        if len(legal_indices) < 1:
            raise FrozenRootPredictorError("no legal root candidate remains")
        masked_state = np.sum(
            history.astype(np.float64) * mask[:, None], axis=0
        ) / float(mask.sum())
        current_state = masked_state.astype(np.float32)
        event_id = observer_output_receipt["current_event_id"]
        event_onehot = np.zeros(len(specification["event_names"]), dtype=np.float32)
        event_onehot[event_id] = np.float32(1.0)
        predicates = np.asarray(
            [float(observer_output_receipt["current_predicates"][name]) for name in PREDICATE_NAMES],
            dtype=np.float32,
        )
        shared = np.concatenate(
            [current_state, proprio, image_array, event_onehot, predicates]
        ).astype(np.float32, copy=False)
        rows = [
            np.concatenate([shared, actions[index].reshape(-1)]).astype(
                np.float32, copy=False
            )
            for index in legal_indices
        ]
        feature_array = np.ascontiguousarray(np.stack(rows, axis=0))
        return torch.from_numpy(feature_array)

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
    ) -> FrozenRootPredictionResult:
        self._assert_frozen_files_unchanged()
        if len(self._members) != MEMBER_COUNT:
            raise FrozenRootPredictorError("runtime does not contain exactly five members")
        self._validate_observation(
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
            lowest_legal_original_candidate_index=lowest_legal_original_candidate_index,
            mapped_actions=mapped_actions,
        )
        features = self._features(
            actor_visible_inputs=actor_visible_inputs,
            observer_output_receipt=observer_output_receipt,
            candidate_legal=candidate_legal,
            mapped_actions=mapped_actions,
        )
        rows: list[dict[str, torch.Tensor]] = []
        calibrated_success: list[torch.Tensor] = []
        composite_rank: list[torch.Tensor] = []
        member_calls = 0
        with torch.inference_mode():
            for member in self._members:
                output = member.model(features)
                member_calls += 1
                calibration = self._calibration
                output["duration_log_scale"] = output["duration_log_scale"].clamp(-8.0, 5.0) + math.log(
                    float(calibration["duration_scale_multiplier"])
                )
                output["object_log_scale"] = output["object_log_scale"].clamp(-8.0, 5.0) + math.log(
                    float(calibration["object_scale_multiplier"])
                )
                rank = member.rank_contract
                event_values = torch.tensor(
                    rank["event_values"], dtype=torch.float32
                )
                event_progress = (
                    torch.softmax(output["next_event_logits"], dim=-1) * event_values
                ).sum(-1)
                duration = torch.expm1(output["duration_log_mean"].clamp(0.0, 12.0))
                base_rank = (
                    output["success_logit"] / np.float32(rank["success_temperature"])
                    + np.float32(rank["event_weight"]) * event_progress
                    - np.float32(rank["duration_weight"])
                    * duration / np.float32(rank["duration_scale"])
                )
                source_rank = base_rank + output["action_rank_residual"] / np.float32(
                    rank["success_temperature"]
                )
                success_probability = torch.sigmoid(
                    output["success_logit"]
                    / np.float32(calibration["success_temperature"])
                )
                if any(not bool(torch.isfinite(value).all()) for value in output.values()):
                    raise FrozenRootPredictorError("member produced non-finite tensors")
                rows.append(output)
                calibrated_success.append(success_probability)
                composite_rank.append(source_rank)
        if member_calls != MEMBER_COUNT or len(rows) != MEMBER_COUNT:
            raise FrozenRootPredictorError("five-member call chronology changed")

        def stacked(name: str) -> np.ndarray:
            value = torch.stack([row[name] for row in rows], dim=0)
            return np.ascontiguousarray(value.cpu().numpy().astype(np.float32, copy=False))

        raw_predictions = {
            name: stacked(name)
            for name in (
                "post_event_logits", "next_event_logits", "duration_log_mean",
                "duration_log_scale", "success_logit", "object_mean",
                "object_log_scale",
            )
        }
        recovery = stacked("recovery_logit")
        uncertainty_inputs = {**raw_predictions, "recovery_logit": recovery}
        uncertainty_parameters = {
            "post_event_temperature": self._calibration["post_event_temperature"],
            "next_event_temperature": self._calibration["next_event_temperature"],
            "success_temperature": self._calibration["success_temperature"],
            "conditional_recovery_temperature": self._calibration[
                "conditional_recovery_temperature"
            ],
            "object_error_robust_scale_m": self._calibration[
                "object_error_robust_scale_m"
            ],
        }
        try:
            components = deployment_uncertainty.root_components(
                predictions=uncertainty_inputs, parameters=uncertainty_parameters
            )
        except deployment_uncertainty.DeploymentUncertaintyError as error:
            raise FrozenRootPredictorError("uncertainty derivation failed") from error
        auxiliary = {
            "member_calibrated_success_probability": np.ascontiguousarray(
                torch.stack(calibrated_success).cpu().numpy().astype(np.float32, copy=False)
            ),
            "member_composite_rank_score": np.ascontiguousarray(
                torch.stack(composite_rank).cpu().numpy().astype(np.float32, copy=False)
            ),
            "candidate_structured_five_head_uncertainty": np.ascontiguousarray(
                np.asarray(components["structured_five_head"], dtype=np.float32)
            ),
        }
        chronology = dict(observation_commitment["chronology"])
        chronology["root_world_model_member_calls"] = member_calls
        try:
            derivation = root_contract.build_root_prediction_derivation_commitment(
                observation_commitment=observation_commitment,
                raw_predictions=raw_predictions,
                auxiliary_tensors=auxiliary,
                root_predictor_authority_sha256=self.authority_sha256,
                calibration_sha256=self.calibration_sha256,
                source_rank_contract_set_sha256=self._runtime_authority[
                    "source_rank_contract_set_sha256"
                ],
                uncertainty_contract_sha256=self._uncertainty_contract[
                    "uncertainty_contract_sha256"
                ],
                derivation_implementation_file_sha256=self._runtime_authority[
                    "root_observed_contract_implementation_file_sha256"
                ],
                chronology=chronology,
            )
            root_contract.validate_root_prediction_derivation_commitment(
                derivation,
                observation_commitment=observation_commitment,
                raw_predictions=raw_predictions,
                auxiliary_tensors=auxiliary,
                expected_root_predictor_authority_sha256=self.authority_sha256,
                expected_calibration_sha256=self.calibration_sha256,
                expected_source_rank_contract_set_sha256=self._runtime_authority[
                    "source_rank_contract_set_sha256"
                ],
                expected_uncertainty_contract_sha256=self._uncertainty_contract[
                    "uncertainty_contract_sha256"
                ],
                expected_derivation_implementation_file_sha256=(
                    self._runtime_authority[
                        "root_observed_contract_implementation_file_sha256"
                    ]
                ),
            )
        except root_contract.RootObservedContractError as error:
            raise FrozenRootPredictorError("derived tensor commitment failed closed") from error
        result = FrozenRootPredictionResult(
            raw_predictions=raw_predictions,
            auxiliary_tensors=auxiliary,
            derivation_commitment=derivation,
            member_call_count=member_calls,
        )
        self.validate_prediction_result(result, observation_commitment=observation_commitment)
        return result

    def validate_prediction_result(
        self, result: FrozenRootPredictionResult, *,
        observation_commitment: Mapping[str, Any],
    ) -> str:
        self._assert_frozen_files_unchanged()
        if (
            type(result) is not FrozenRootPredictionResult
            or result.member_call_count != MEMBER_COUNT
            or type(result.member_call_count) is not int
            or result.derivation_commitment.get("root_predictor_authority_sha256")
            != self.authority_sha256
            or result.derivation_commitment.get("calibration_sha256")
            != self.calibration_sha256
        ):
            raise FrozenRootPredictorError("runtime prediction result authority changed")
        try:
            return root_contract.validate_root_prediction_derivation_commitment(
                result.derivation_commitment,
                observation_commitment=observation_commitment,
                raw_predictions=result.raw_predictions,
                auxiliary_tensors=result.auxiliary_tensors,
                expected_root_predictor_authority_sha256=self.authority_sha256,
                expected_calibration_sha256=self.calibration_sha256,
                expected_source_rank_contract_set_sha256=self._runtime_authority[
                    "source_rank_contract_set_sha256"
                ],
                expected_uncertainty_contract_sha256=self._uncertainty_contract[
                    "uncertainty_contract_sha256"
                ],
                expected_derivation_implementation_file_sha256=(
                    self._runtime_authority[
                        "root_observed_contract_implementation_file_sha256"
                    ]
                ),
            )
        except root_contract.RootObservedContractError as error:
            raise FrozenRootPredictorError(
                "runtime prediction tensors or derived vectors changed"
            ) from error


def load_frozen_root_predictor_runtime(
    artifact_root: Path, *, expected_root_predictor_authority_sha256: str,
) -> FrozenRootPredictorRuntime:
    expected_authority = _sha(
        expected_root_predictor_authority_sha256, "expected root predictor authority"
    )
    supplied_root = Path(artifact_root)
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise FrozenRootPredictorError("artifact root must be one non-symlink directory")
    frozen_root = supplied_root.resolve()
    manifest_path = frozen_root / AUTHORITY_MANIFEST_BASENAME
    manifest_raw = _read_bytes(
        manifest_path, maximum=MAX_JSON_BYTES, role="authority manifest"
    )
    manifest_file_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = _parse_json_bytes(manifest_raw, "authority manifest")
    manifest_fields = {
        "format", "status", "member_count", "artifact_inventory",
        "runtime_authority", "checkpoint_file_set_sha256",
        "source_rank_contract_set_sha256",
    }
    logical = _signed_sha(
        manifest, fields=manifest_fields,
        digest_field="root_predictor_authority_sha256", role="authority manifest",
    )
    if (
        logical != expected_authority
        or manifest["format"] != MANIFEST_FORMAT
        or manifest["status"]
        != "frozen_reference_test_authority_not_deployment_promoted"
        or manifest["member_count"] != MEMBER_COUNT
        or type(manifest["member_count"]) is not int
    ):
        raise FrozenRootPredictorError("root predictor authority is not externally pinned")
    inventory = manifest["artifact_inventory"]
    expected_basenames = {
        EVENT_SPEC_BASENAME, CALIBRATION_BASENAME, UNCERTAINTY_BASENAME,
        RUNTIME_AUTHORITY_BASENAME,
        *(f"member_{index:02d}.pt" for index in range(MEMBER_COUNT)),
        *(f"rank_contract_{index:02d}.json" for index in range(MEMBER_COUNT)),
    }
    if not isinstance(inventory, list) or len(inventory) != len(expected_basenames):
        raise FrozenRootPredictorError("authority artifact inventory size changed")
    records: dict[str, Mapping[str, Any]] = {}
    for record in inventory:
        item = _exact(record, {"basename", "role", "file_sha256"}, "artifact record")
        basename = item["basename"]
        if (
            not isinstance(basename, str)
            or basename in records
            or basename not in expected_basenames
            or Path(basename).name != basename
        ):
            raise FrozenRootPredictorError("artifact basename inventory changed")
        _sha(item["file_sha256"], f"artifact {basename}")
        records[basename] = item
    if set(records) != expected_basenames or [row["basename"] for row in inventory] != sorted(expected_basenames):
        raise FrozenRootPredictorError("artifact inventory must be exact and sorted")
    actual_names = {child.name for child in frozen_root.iterdir()}
    if actual_names != {AUTHORITY_MANIFEST_BASENAME, *expected_basenames}:
        raise FrozenRootPredictorError("artifact root contains missing or extra files")
    expected_roles = {
        EVENT_SPEC_BASENAME: "event_spec",
        CALIBRATION_BASENAME: "calibration",
        UNCERTAINTY_BASENAME: "uncertainty_contract",
        RUNTIME_AUTHORITY_BASENAME: "runtime_authority",
        **{f"member_{index:02d}.pt": "member_checkpoint" for index in range(MEMBER_COUNT)},
        **{f"rank_contract_{index:02d}.json": "source_rank_contract" for index in range(MEMBER_COUNT)},
    }
    if any(records[name]["role"] != expected_roles[name] for name in records):
        raise FrozenRootPredictorError("artifact roles changed")
    inventory_shas = {
        name: str(records[name]["file_sha256"]) for name in expected_basenames
    }
    event_spec = _validate_event_spec(
        _read_hashed_json(
            frozen_root / EVENT_SPEC_BASENAME,
            inventory_shas[EVENT_SPEC_BASENAME], "event spec",
        )
    )
    calibration = _validate_calibration(
        _read_hashed_json(
            frozen_root / CALIBRATION_BASENAME,
            inventory_shas[CALIBRATION_BASENAME], "calibration",
        ),
        event_spec_sha=event_spec["event_spec_sha256"],
    )
    uncertainty_contract = _validate_uncertainty(
        _read_hashed_json(
            frozen_root / UNCERTAINTY_BASENAME,
            inventory_shas[UNCERTAINTY_BASENAME], "uncertainty contract",
        ),
        event_spec_sha=event_spec["event_spec_sha256"],
        calibration_sha=calibration["calibration_sha256"],
    )
    runtime_authority = _read_hashed_json(
        frozen_root / RUNTIME_AUTHORITY_BASENAME,
        inventory_shas[RUNTIME_AUTHORITY_BASENAME], "runtime authority",
    )
    runtime_fields = {
        "format", "status", "member_count", "event_spec", "calibration",
        "uncertainty_contract", "members", "checkpoint_file_set_sha256",
        "source_rank_contract_set_sha256",
        "runtime_implementation_file_sha256",
        "root_observed_contract_implementation_file_sha256",
        "deployment_uncertainty_implementation_file_sha256",
        "model_family", "promotion_eligible", "production_compatibility_status",
        "online_input_contract", "member_call_contract", "condition_boundary",
    }
    runtime_sha = _signed_sha(
        runtime_authority, fields=runtime_fields,
        digest_field="runtime_authority_sha256", role="runtime authority",
    )
    runtime_reference = _exact(
        manifest["runtime_authority"],
        {"basename", "file_sha256", "runtime_authority_sha256"},
        "manifest runtime authority reference",
    )
    if (
        runtime_reference != {
            "basename": RUNTIME_AUTHORITY_BASENAME,
            "file_sha256": inventory_shas[RUNTIME_AUTHORITY_BASENAME],
            "runtime_authority_sha256": runtime_sha,
        }
        or runtime_authority["format"] != RUNTIME_AUTHORITY_FORMAT
        or runtime_authority["status"]
        != "actor_visible_reference_inference_before_synthetic_condition_only"
        or runtime_authority["member_count"] != MEMBER_COUNT
        or type(runtime_authority["member_count"]) is not int
        or runtime_authority["model_family"] != MODEL_FAMILY
        or runtime_authority["promotion_eligible"] is not False
        or runtime_authority["production_compatibility_status"]
        != "must_not_replace_ActionConditionedEventWorldModel_or_PiperAdaptedWorldModel"
        or runtime_authority["event_spec"] != {
            "basename": EVENT_SPEC_BASENAME,
            "file_sha256": inventory_shas[EVENT_SPEC_BASENAME],
            "event_spec_sha256": event_spec["event_spec_sha256"],
        }
        or runtime_authority["calibration"] != {
            "basename": CALIBRATION_BASENAME,
            "file_sha256": inventory_shas[CALIBRATION_BASENAME],
            "calibration_sha256": calibration["calibration_sha256"],
        }
        or runtime_authority["uncertainty_contract"] != {
            "basename": UNCERTAINTY_BASENAME,
            "file_sha256": inventory_shas[UNCERTAINTY_BASENAME],
            "uncertainty_contract_sha256": uncertainty_contract[
                "uncertainty_contract_sha256"
            ],
        }
        or runtime_authority["runtime_implementation_file_sha256"]
        != _runtime_file_sha()
        or runtime_authority["root_observed_contract_implementation_file_sha256"]
        != _module_file_sha(root_contract, "root observed contract")
        or runtime_authority[
            "deployment_uncertainty_implementation_file_sha256"
        ] != _module_file_sha(deployment_uncertainty, "uncertainty")
        or runtime_authority["online_input_contract"]
        != "validated_root_observation_commitment_and_original_actor_visible_tensors_only"
        or runtime_authority["member_call_contract"]
        != "exactly_five_members_once_each_vectorized_over_all_legal_candidates"
        or runtime_authority["condition_boundary"]
        != "all_validation_and_derivation_complete_before_synthetic_test_condition"
    ):
        raise FrozenRootPredictorError("runtime authority cross-binding changed")
    member_records = runtime_authority["members"]
    if not isinstance(member_records, list) or len(member_records) != MEMBER_COUNT:
        raise FrozenRootPredictorError("runtime authority must bind exactly five members")
    members: list[_Member] = []
    checkpoint_file_shas: list[str] = []
    rank_contract_shas: list[str] = []
    for index, member_record in enumerate(member_records):
        record = _exact(
            member_record,
            {
                "member_index", "checkpoint_basename", "checkpoint_file_sha256",
                "checkpoint_logical_sha256", "rank_contract_basename",
                "rank_contract_file_sha256", "rank_contract_sha256",
            },
            f"runtime member {index}",
        )
        checkpoint_basename = f"member_{index:02d}.pt"
        rank_basename = f"rank_contract_{index:02d}.json"
        if (
            record["member_index"] != index
            or type(record["member_index"]) is not int
            or record["checkpoint_basename"] != checkpoint_basename
            or record["rank_contract_basename"] != rank_basename
            or record["checkpoint_file_sha256"] != inventory_shas[checkpoint_basename]
            or record["rank_contract_file_sha256"] != inventory_shas[rank_basename]
        ):
            raise FrozenRootPredictorError(f"runtime member {index} file binding changed")
        checkpoint_raw = _read_bytes(
            frozen_root / checkpoint_basename,
            maximum=MAX_CHECKPOINT_BYTES,
            role=f"member {index} checkpoint",
        )
        if hashlib.sha256(checkpoint_raw).hexdigest() != inventory_shas[checkpoint_basename]:
            raise FrozenRootPredictorError(f"member {index} checkpoint file SHA changed")
        model, checkpoint_logical = _load_checkpoint(
            checkpoint_raw, member_index=index, event_spec=event_spec
        )
        rank_contract = _validate_rank_contract(
            _read_hashed_json(
                frozen_root / rank_basename,
                inventory_shas[rank_basename], f"member {index} rank contract",
            ),
            member_index=index,
            checkpoint_file_sha=inventory_shas[checkpoint_basename],
            event_spec_sha=event_spec["event_spec_sha256"],
        )
        if (
            record["checkpoint_logical_sha256"] != checkpoint_logical
            or record["rank_contract_sha256"] != rank_contract["contract_sha256"]
        ):
            raise FrozenRootPredictorError(f"runtime member {index} logical binding changed")
        checkpoint_file_shas.append(inventory_shas[checkpoint_basename])
        rank_contract_shas.append(rank_contract["contract_sha256"])
        members.append(
            _Member(
                model=model,
                rank_contract=rank_contract,
                checkpoint_file_sha256=inventory_shas[checkpoint_basename],
            )
        )
    checkpoint_set_sha = canonical_sha256(checkpoint_file_shas)
    rank_set_sha = canonical_sha256(rank_contract_shas)
    if (
        runtime_authority["checkpoint_file_set_sha256"] != checkpoint_set_sha
        or manifest["checkpoint_file_set_sha256"] != checkpoint_set_sha
        or runtime_authority["source_rank_contract_set_sha256"] != rank_set_sha
        or manifest["source_rank_contract_set_sha256"] != rank_set_sha
    ):
        raise FrozenRootPredictorError("five-member checkpoint/rank set binding changed")
    return FrozenRootPredictorRuntime(
        artifact_root=frozen_root,
        root_predictor_authority_sha256=logical,
        manifest_file_sha256=manifest_file_sha,
        inventory_file_sha256=inventory_shas,
        event_spec=event_spec,
        calibration=calibration,
        uncertainty_contract=uncertainty_contract,
        runtime_authority=runtime_authority,
        members=members,
    )


__all__ = [
    "AUTHORITY_MANIFEST_BASENAME",
    "CALIBRATION_FORMAT",
    "CHECKPOINT_FORMAT",
    "CompactRootPredictorV1",
    "EVENT_SPEC_FORMAT",
    "FrozenRootPredictionResult",
    "FrozenRootPredictorError",
    "FrozenRootPredictorRuntime",
    "MANIFEST_FORMAT",
    "MEMBER_COUNT",
    "MODEL_FAMILY",
    "RANK_CONTRACT_FORMAT",
    "RUNTIME_AUTHORITY_FORMAT",
    "UNCERTAINTY_FORMAT",
    "canonical_sha256",
    "file_sha256",
    "load_frozen_root_predictor_runtime",
    "state_dict_sha256",
]
