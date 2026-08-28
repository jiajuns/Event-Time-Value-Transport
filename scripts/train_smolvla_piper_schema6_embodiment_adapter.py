#!/usr/bin/env python3
"""Train the strictly isolated SmolVLA->Piper schema-6 adapter.

This entry point intentionally does not import RoboTwin or LeRobot.  It consumes
materialized, non-Fresh schema-6 HDF5 groups and a source-only native-960 event
core.  Group membership is frozen from label-free manifest metadata before any
HDF5 file is opened; test-group HDF5 files are never opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from collect_smolvla_piper_schema6_dense_event_branches import (
    FORMAT as SCHEMA6_GROUP_FORMAT,
    INTERVENTION as SCHEMA6_INTERVENTION,
    validate_schema6_group_file,
)
from etsf_schema6_pose_quality import load_object_delta_supervision_v6
from initialize_smolvla_schema5_native_event_core import (
    BODY_EMBEDDING,
    POLICY_EMBEDDING,
    tensor_sha256,
)
from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from train_openvla_etsf_counterfactual import (
    CAUSAL_HISTORY_MAX_STEPS,
    candidate_rank_score,
    causal_history_contract,
    fixed_causal_hidden_window,
    group_action_rank_residual,
    state_dict_sha256,
    validate_reserved_rows_source_only_proof,
    validate_reserved_target_rows,
)


FORMAT = "etsf_smolvla_piper_schema6_embodiment_adapter_v1"
BODY_AGNOSTIC_FORMAT = (
    "etsf_smolvla_piper_schema6_body_agnostic_adapter_v1"
)
MANIFEST_FORMAT = "etsf_smolvla_piper_schema6_training_manifest_v1"
SPLIT_FORMAT = "etsf_smolvla_piper_schema6_group_split_v1"
EXTERNAL_SPLIT_FORMAT = "etsf_smolvla_piper_schema6_external_group_split_v2"
TARGET_PARTITION_FORMAT = "etsf_smolvla_piper_schema6_target_partition_v2"
EXPECTED_MANIFEST_SPLIT_FORMAT = "etsf_smolvla_piper_schema6_expected_manifest_split_v2"
EXTERNAL_SPLIT_FORMAT_V3 = "etsf_smolvla_piper_schema6_external_group_split_v3"
TARGET_PARTITION_FORMAT_V3 = "etsf_smolvla_piper_schema6_target_partition_v3"
EXPECTED_MANIFEST_SPLIT_FORMAT_V3 = "etsf_smolvla_piper_schema6_expected_manifest_split_v3"
HISTORICAL_SPLIT_PROFILE = "historical_v2"
DEVELOPMENT300_SPLIT_PROFILE = "development300_v3"
STATE_DIM = 960
ACTION_DIM = 14
SOURCE_BODY = "aloha-agilex"
TARGET_BODY = "piper"
SOURCE_POLICY = "smolvla"
RESERVED_POLICY = "openvla"
PROVIDER_VARIANTS = ("body_conditioned_adapter", "body_agnostic_adapter")
DEFAULT_PROVIDER_VARIANT = "body_conditioned_adapter"
FORBIDDEN_PATH_TOKENS = ("fresh", "confirmation")
EXPECTED_EVENTS = ("e0", "e12", "e3", "e4", "eK")
# ``dense_event_targets`` includes e0 only as the initial state at step zero;
# an observed future milestone is necessarily one of the remaining events.
# Censored rows carry current-event placeholders and are excluded below.
EXPECTED_NEXT_REACHED_EVENTS = ("e12", "e3", "e4", "eK")
EXPECTED_PREDICATES = ("moved", "lifted", "near_goal", "stationary", "success")
DEFAULT_MIN_TRAIN_GROUPS = 30
DEFAULT_MIN_VALIDATION_GROUPS = 20
DEFAULT_MIN_TEST_GROUPS = 50
DEFAULT_MIN_OUTCOME_GROUPS = 5
DEFAULT_MIN_DISCORDANT_GROUPS = 5
DEFAULT_MIN_EVENT_ROWS = 5
DEFAULT_MIN_DURATION_ROWS = 5
DEFAULT_MIN_OBJECT_ROWS = 5
FORMAL_TRAIN_GROUPS = 60
FORMAL_VALIDATION_GROUPS = 20
FORMAL_SEALED_TEST_GROUPS = 50
DEVELOPMENT300_TRAIN_GROUPS = 80
DEVELOPMENT300_VALIDATION_GROUPS = 30
DEVELOPMENT300_SEALED_TEST_GROUPS = 190
RECOVERY_PERSISTENCE_STATES = 3
MIN_RECOVERY_GROUPS_PER_CLASS = 10
PAIRED_BOOTSTRAP_SEED = 20260828
PAIRED_BOOTSTRAP_SAMPLES = 20_000
PAIRED_BOOTSTRAP_CONFIDENCE = 0.95
SOURCE_RANK_SCORE_FORMAT = "etsf_source63_composite_candidate_rank_score_v1"
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)
SOURCE_RANK_SUCCESS_TEMPERATURE = 1.0
SOURCE_RANK_EVENT_WEIGHT = 0.25
SOURCE_RANK_DURATION_WEIGHT = 0.05


class AdapterContractError(RuntimeError):
    """A protocol or immutable-state contract failed closed."""


def provider_artifact_format(provider_variant: str) -> str:
    if provider_variant == "body_conditioned_adapter":
        return FORMAT
    if provider_variant == "body_agnostic_adapter":
        return BODY_AGNOSTIC_FORMAT
    raise AdapterContractError(
        f"provider variant must be one of {PROVIDER_VARIANTS}"
    )


def validate_production_source_rank_config(
    config: EventWorldModelConfig,
) -> dict[str, Any]:
    """Require the exact rank configuration emitted by the Source63 launcher."""

    if (
        config.action_rank_residual is not True
        or config.action_rank_success_only is not False
    ):
        raise AdapterContractError(
            "production Piper transfer requires Source action_rank_residual=true "
            "and action_rank_success_only=false"
        )
    return {
        "source_action_rank_residual_consumed": True,
        "source_action_rank_success_only": False,
        "source_launcher_freeze_factual_core": False,
    }


def source_rank_score_contract(
    checkpoint: Mapping[str, Any],
    config: EventWorldModelConfig,
    *,
    source_checkpoint_file_sha256: str,
) -> dict[str, Any]:
    """Freeze the exact composite score on which Source63 trained residuals."""

    validate_production_source_rank_config(config)
    if not _is_sha(source_checkpoint_file_sha256):
        raise AdapterContractError("Source rank contract checkpoint SHA256 is invalid")
    duration_scale = checkpoint.get("duration_scale")
    if (
        isinstance(duration_scale, bool)
        or not isinstance(duration_scale, (int, float))
        or not math.isfinite(float(duration_scale))
        or float(duration_scale) < 1.0
    ):
        raise AdapterContractError(
            "Source rank contract lacks its training-derived duration_scale"
        )
    source_contract = checkpoint.get("contract")
    optimization = (
        source_contract.get("action_rank_optimization")
        if isinstance(source_contract, Mapping)
        else None
    )
    trainable_names = (
        optimization.get("trainable_parameter_names")
        if isinstance(optimization, Mapping)
        else None
    )
    required_trainable_prefixes = (
        "semantic.",
        "next_event_head.",
        "success_head.",
        "clock_cell.",
        "action_rank_head.",
    )
    if (
        not isinstance(optimization, Mapping)
        or optimization.get("freeze_factual_core") is not False
        or not isinstance(trainable_names, list)
        or not trainable_names
        or any(not isinstance(name, str) or not name for name in trainable_names)
        or any(
            not any(name.startswith(prefix) for name in trainable_names)
            for prefix in required_trainable_prefixes
        )
    ):
        raise AdapterContractError(
            "Source rank optimization does not prove full-core composite training"
        )
    event_values = torch.linspace(0.0, 1.0, config.num_events).tolist()
    if tuple(config.event_names) != EXPECTED_EVENTS or event_values != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise AdapterContractError("Source rank event-value authority changed")
    result: dict[str, Any] = {
        "format": SOURCE_RANK_SCORE_FORMAT,
        "status": "frozen_exact_source63_training_score_scientific_rank_only",
        "source_checkpoint_file_sha256": source_checkpoint_file_sha256,
        "source_action_rank_residual": True,
        "source_action_rank_success_only": False,
        "source_freeze_factual_core": False,
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "base_score": "candidate_rank_score",
        "event_names": list(EXPECTED_EVENTS),
        "event_values": event_values,
        "event_values_authority": "source_trainer_linspace_0_1_in_checkpoint_event_order",
        "duration_scale": float(duration_scale),
        "duration_scale_authority": "source_member_checkpoint.duration_scale",
        "duration_scale_scope": "per_source_member_not_ensemble_mean",
        "duration_unit": "decision_steps",
        "success_temperature": SOURCE_RANK_SUCCESS_TEMPERATURE,
        "event_weight": SOURCE_RANK_EVENT_WEIGHT,
        "duration_weight": SOURCE_RANK_DURATION_WEIGHT,
        "residual_combination": "candidate_rank_score_plus_action_rank_residual",
        "score_variant": "source_member_training_objective_defaults",
        "source_ensemble_validation_selected_scoring_consumed": False,
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "cross_embodiment_duration_scale_calibrated": False,
        "deployment_success_probability_selector_authorized": False,
    }
    result["contract_sha256"] = canonical_sha256(result)
    return result


def _validate_source_rank_score_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError("Source composite rank-score contract is missing")
    expected_fields = {
        "format", "status", "source_checkpoint_file_sha256",
        "source_action_rank_residual", "source_action_rank_success_only",
        "source_freeze_factual_core", "source_rank_numeric_contract",
        "base_score", "event_names", "event_values",
        "event_values_authority", "duration_scale", "duration_scale_authority",
        "duration_scale_scope", "duration_unit", "success_temperature",
        "event_weight", "duration_weight", "residual_combination", "score_variant",
        "source_ensemble_validation_selected_scoring_consumed",
        "source_contract_rank_score_is_success_logit",
        "source_contract_rank_score_is_success_probability",
        "cross_embodiment_duration_scale_calibrated",
        "deployment_success_probability_selector_authorized", "contract_sha256",
    }
    unsigned = dict(value)
    logical = unsigned.pop("contract_sha256", None)
    numeric = ("duration_scale", "success_temperature", "event_weight", "duration_weight")
    if (
        set(value) != expected_fields
        or value.get("format") != SOURCE_RANK_SCORE_FORMAT
        or value.get("status")
        != "frozen_exact_source63_training_score_scientific_rank_only"
        or not _is_sha(value.get("source_checkpoint_file_sha256"))
        or value.get("source_action_rank_residual") is not True
        or value.get("source_action_rank_success_only") is not False
        or value.get("source_freeze_factual_core") is not False
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or value.get("base_score") != "candidate_rank_score"
        or value.get("event_names") != list(EXPECTED_EVENTS)
        or not isinstance(value.get("event_values"), list)
        or len(value["event_values"]) != 5
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value["event_values"]
        )
        or [float(item) for item in value["event_values"]]
        != [0.0, 0.25, 0.5, 0.75, 1.0]
        or value.get("event_values_authority")
        != "source_trainer_linspace_0_1_in_checkpoint_event_order"
        or value.get("duration_scale_authority")
        != "source_member_checkpoint.duration_scale"
        or value.get("duration_scale_scope")
        != "per_source_member_not_ensemble_mean"
        or value.get("duration_unit") != "decision_steps"
        or value.get("residual_combination")
        != "candidate_rank_score_plus_action_rank_residual"
        or value.get("score_variant")
        != "source_member_training_objective_defaults"
        or value.get("source_ensemble_validation_selected_scoring_consumed") is not False
        or value.get("source_contract_rank_score_is_success_logit") is not False
        or value.get("source_contract_rank_score_is_success_probability") is not False
        or value.get("cross_embodiment_duration_scale_calibrated") is not False
        or value.get("deployment_success_probability_selector_authorized") is not False
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), (int, float))
            or not math.isfinite(float(value[name]))
            for name in numeric
        )
        or float(value["duration_scale"]) < 1.0
        or float(value["success_temperature"]) != SOURCE_RANK_SUCCESS_TEMPERATURE
        or float(value["event_weight"]) != SOURCE_RANK_EVENT_WEIGHT
        or float(value["duration_weight"]) != SOURCE_RANK_DURATION_WEIGHT
        or not _is_sha(logical)
        or logical != canonical_sha256(unsigned)
    ):
        raise AdapterContractError("Source composite rank-score contract changed")
    return dict(value)


@dataclass(frozen=True)
class ExternalSplitProfile:
    """Exact, receipt-selected physical split contract.

    A profile is selected only from a signed expected-receipt format.  It is
    never inferred from CLI arguments or from the lengths of identity lists.
    """

    name: str
    version: int
    expected_receipt_format: str
    target_partition_format: str
    external_split_format: str
    train_groups: int
    internal_validation_groups: int
    sealed_test_groups: int
    explicit_profile_field: bool

    @property
    def adaptation_groups(self) -> int:
        return self.train_groups + self.internal_validation_groups

    @property
    def required_trainer_group_counts(self) -> dict[str, int]:
        return {
            "train": self.train_groups,
            "validation": self.internal_validation_groups,
            "test": self.sealed_test_groups,
        }


HISTORICAL_V2_PROFILE = ExternalSplitProfile(
    name=HISTORICAL_SPLIT_PROFILE,
    version=2,
    expected_receipt_format=EXPECTED_MANIFEST_SPLIT_FORMAT,
    target_partition_format=TARGET_PARTITION_FORMAT,
    external_split_format=EXTERNAL_SPLIT_FORMAT,
    train_groups=FORMAL_TRAIN_GROUPS,
    internal_validation_groups=FORMAL_VALIDATION_GROUPS,
    sealed_test_groups=FORMAL_SEALED_TEST_GROUPS,
    explicit_profile_field=False,
)
DEVELOPMENT300_V3_PROFILE = ExternalSplitProfile(
    name=DEVELOPMENT300_SPLIT_PROFILE,
    version=3,
    expected_receipt_format=EXPECTED_MANIFEST_SPLIT_FORMAT_V3,
    target_partition_format=TARGET_PARTITION_FORMAT_V3,
    external_split_format=EXTERNAL_SPLIT_FORMAT_V3,
    train_groups=DEVELOPMENT300_TRAIN_GROUPS,
    internal_validation_groups=DEVELOPMENT300_VALIDATION_GROUPS,
    sealed_test_groups=DEVELOPMENT300_SEALED_TEST_GROUPS,
    explicit_profile_field=True,
)
SPLIT_PROFILES_BY_EXPECTED_FORMAT = {
    profile.expected_receipt_format: profile
    for profile in (HISTORICAL_V2_PROFILE, DEVELOPMENT300_V3_PROFILE)
}
SPLIT_PROFILES_BY_NAME = {
    profile.name: profile
    for profile in (HISTORICAL_V2_PROFILE, DEVELOPMENT300_V3_PROFILE)
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def schema6_causal_history_at_query(
    branch_hidden: np.ndarray, query_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct one history from one branch only, ending at ``query_index``."""

    values = np.asarray(branch_hidden)
    if (
        values.ndim != 2
        or values.shape[1] != STATE_DIM
        or values.shape[0] < 1
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or isinstance(query_index, bool)
        or not isinstance(query_index, int)
        or query_index < 0
        or query_index >= values.shape[0]
    ):
        raise AdapterContractError("schema6 causal branch prefix is invalid")
    history, mask = fixed_causal_hidden_window(values[: query_index + 1])
    if not np.array_equal(history[int(mask.sum()) - 1], values[query_index]):
        raise AdapterContractError("schema6 causal history lost its current query")
    return history, mask


def schema6_causal_history_application_contract() -> dict[str, Any]:
    """Describe exactly which part of the Source history contract Piper uses."""

    value: dict[str, Any] = {
        "format": "etsf_schema6_piper_causal_history_application_v1",
        "source_causal_history_contract_sha256": causal_history_contract()[
            "contract_sha256"
        ],
        "input_history": (
            "same_branch_query_hidden_prefix_ending_at_current_query"
        ),
        "max_history_steps": CAUSAL_HISTORY_MAX_STEPS,
        "root_candidate_effective_history_steps": 1,
        "post_hidden_target_supervised": False,
        "post_hidden_target_history": (
            "not_supervised_schema6_has_no_query_post_hidden"
        ),
        "oracle_or_fabricated_post_hidden_allowed": False,
        "cross_branch_or_group_history_allowed": False,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


def _require_nonnegative_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdapterContractError(f"{role} must be a non-negative integer")
    return value


def _require_finite(value: Any, role: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AdapterContractError(f"{role} must be finite")
    return result


def configure_determinism(seed: int) -> dict[str, Any]:
    """Seed every RNG before adapter construction and fail on nondeterminism."""

    seed = _require_nonnegative_int(seed, "training seed")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "training_seed": seed,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_all_seeded_if_available": True,
        "deterministic_algorithms_required": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def reject_sensitive_path(path: str | Path, role: str, *, file: bool = True) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise AdapterContractError(f"{role} must be absolute")
    leaf_is_symlink = supplied.is_symlink()
    resolved = supplied.resolve(strict=False)
    if any(token in part.casefold() for part in (*supplied.parts, *resolved.parts) for token in FORBIDDEN_PATH_TOKENS):
        raise AdapterContractError(f"{role} references Fresh/confirmation")
    if file and (leaf_is_symlink or not resolved.is_file()):
        raise AdapterContractError(f"{role} must be a materialized regular file")
    return resolved


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterContractError(f"invalid {role} JSON") from exc
    if not isinstance(value, dict):
        raise AdapterContractError(f"{role} must contain an object")
    return value


def _load_torch(path: Path, role: str) -> dict[str, Any]:
    try:
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.float32))]):
            value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AdapterContractError(f"{role} is not a safe torch mapping") from exc
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{role} must contain a mapping")
    return dict(value)


def _validate_source_row(spec: Any, *, identity: str, row: int, parameter: str, tensor: torch.Tensor) -> None:
    # ``tensor_sha256`` in source_identity_rows records cold initialization.
    # Source-only training may legitimately update row zero, so only the
    # identity/position contract is immutable here; the current row is bound by
    # source_core_state_sha256 and recorded separately below.
    if not isinstance(spec, Mapping) or spec.get("identity") != identity or spec.get("row") != row or spec.get("id") != row or spec.get("parameter") != parameter or not _is_sha(spec.get("tensor_sha256")):
        raise AdapterContractError(f"source identity row {identity!r} is invalid")


def validate_source_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the exact dual-reserved proof while selecting SmolVLA row zero."""

    config_raw, state, contract = checkpoint.get("config"), checkpoint.get("model"), checkpoint.get("contract")
    if not isinstance(config_raw, Mapping) or not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise AdapterContractError("source checkpoint lacks config/model/contract")
    config = EventWorldModelConfig.from_dict(config_raw)
    if config.state_input_dim != STATE_DIM or config.action_dim != ACTION_DIM or config.num_bodies != 2 or config.num_policies != 2 or tuple(config.event_names) != EXPECTED_EVENTS or tuple(config.predicate_names) != EXPECTED_PREDICATES or not config.structured_events:
        raise AdapterContractError("source core is not the native-960 structured schema-5 core")
    if contract.get("body_to_id") != {SOURCE_BODY: 0, "__reserved__piper": 1}:
        raise AdapterContractError("Piper must be the pre-existing reserved body row one")
    if contract.get("policy_to_id") != {SOURCE_POLICY: 0, "__reserved__openvla": 1}:
        raise AdapterContractError("policy registry is not the exact SmolVLA/OpenVLA reservation")
    history_contract = contract.get("causal_history_contract")
    if history_contract != causal_history_contract():
        raise AdapterContractError(
            "source checkpoint lacks the exact causal hidden-history contract"
        )
    try:
        rows = validate_reserved_target_rows(checkpoint, config)
        proof = validate_reserved_rows_source_only_proof(checkpoint, rows)
    except RuntimeError as exc:
        raise AdapterContractError("dual-reserved source-only proof is invalid") from exc
    if rows is None or proof is None or rows["body"].identity != TARGET_BODY or rows["body"].row != 1 or rows["policy"].identity != RESERVED_POLICY or rows["policy"].row != 1:
        raise AdapterContractError("dual-reserved proof has the wrong identities")
    body = state.get(BODY_EMBEDDING)
    policy = state.get(POLICY_EMBEDDING)
    if not torch.is_tensor(body) or not torch.is_tensor(policy):
        raise AdapterContractError("embedding tensors are missing")
    source_rows = contract.get("source_identity_rows")
    if not isinstance(source_rows, Mapping):
        raise AdapterContractError("source identity proof is missing")
    _validate_source_row(source_rows.get("body"), identity=SOURCE_BODY, row=0, parameter=BODY_EMBEDDING, tensor=body)
    _validate_source_row(source_rows.get("policy"), identity=SOURCE_POLICY, row=0, parameter=POLICY_EMBEDDING, tensor=policy)
    if proof.get("source_core_state_sha256") != state_dict_sha256(state):
        raise AdapterContractError("source-only proof is not bound to model state")
    object_names = contract.get("object_names")
    if (
        not isinstance(object_names, list)
        or not object_names
        or len(set(map(str, object_names))) != len(object_names)
        or config.object_delta_dim != 3 * len(object_names)
    ):
        raise AdapterContractError("source object head lacks an exact selected-object registry")
    return {
        "config": config.to_dict(),
        "target_body_row": 1,
        "policy_row": 0,
        "reserved_openvla_policy_row": 1,
        "source_core_state_sha256": proof["source_core_state_sha256"],
        "proof_sha256": proof["proof_sha256"],
        "source_smolvla_policy_row_0_sha256": tensor_sha256(policy[0]),
        "reserved_openvla_policy_row_1_sha256": tensor_sha256(policy[1]),
        "object_names": [str(name) for name in object_names],
        "causal_history_contract": dict(history_contract),
        "policy_selection": "source_smolvla_row_0_not_reserved_openvla_row_1",
    }


class ResidualLowRankStateAdapter(nn.Module):
    """960-D identity plus a zero-initialized low-rank residual."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("state rank must be positive")
        self.down = nn.Linear(STATE_DIM, rank, bias=False)
        self.up = nn.Linear(rank, STATE_DIM, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.up(self.down(state))


class IdentityLowRankDiagonalActionAdapter(nn.Module):
    """14-D diagonal identity plus a zero-initialized low-rank residual."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("action rank must be positive")
        self.diagonal = nn.Parameter(torch.ones(ACTION_DIM))
        self.down = nn.Linear(ACTION_DIM, rank, bias=False)
        self.up = nn.Linear(rank, ACTION_DIM, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.diagonal + self.up(self.down(actions))


class DetachedConditionalRecoveryAdapter(nn.Module):
    """Independent ``p(recovery | operational regress)`` probability head.

    Detaching inside the public forward boundary is intentional: recovery
    supervision must not rewrite the frozen source core *or* the embodiment
    adapter.  This head owns a separate optimizer and calibration contract.
    """

    def __init__(self, transition_dim: int) -> None:
        super().__init__()
        if transition_dim < 1:
            raise ValueError("recovery transition dimension must be positive")
        self.transition_dim = int(transition_dim)
        self.head = nn.Linear(self.transition_dim, 1)

    def initialize_prior(self, prevalence: float) -> None:
        if not math.isfinite(prevalence) or not 0.0 < prevalence < 1.0:
            raise AdapterContractError(
                "conditional recovery prior requires both supervised classes"
            )
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        if (
            transition.ndim != 2
            or transition.shape[1] != self.transition_dim
            or not bool(torch.isfinite(transition).all())
        ):
            raise AdapterContractError(
                "conditional recovery transition features are invalid"
            )
        return self.head(transition.detach()).squeeze(-1)

    def parameter_audit(self) -> dict[str, Any]:
        names = sorted(name for name, _ in self.named_parameters())
        if names != ["head.bias", "head.weight"]:
            raise AdapterContractError("conditional recovery parameter set changed")
        return {
            "names": names,
            "parameter_tensor_count": 2,
            "shared_transition_stop_gradient": True,
            "optimizer_isolated_from_embodiment_adapter": True,
        }


class SmolVLAPiperAdapter(nn.Module):
    """Frozen native core with exactly four effective target components."""

    def __init__(
        self,
        core: ActionConditionedEventWorldModel,
        *,
        state_rank: int = 16,
        action_rank: int = 4,
        source_rank_contract: Mapping[str, Any] | None = None,
        provider_variant: str = DEFAULT_PROVIDER_VARIANT,
    ) -> None:
        super().__init__()
        if core.config.state_input_dim != STATE_DIM or core.config.action_dim != ACTION_DIM:
            raise AdapterContractError("adapter requires the native 960D/14D core")
        if provider_variant not in PROVIDER_VARIANTS:
            raise AdapterContractError(
                f"provider variant must be one of {PROVIDER_VARIANTS}"
            )
        self.provider_variant = provider_variant
        self.core = core
        self.state_adapter = ResidualLowRankStateAdapter(state_rank)
        self.action_adapter = IdentityLowRankDiagonalActionAdapter(action_rank)
        self.clock_beta = nn.Parameter(torch.zeros(()))
        self.clock_log_step_scale = nn.Parameter(torch.zeros(()))
        self.target_body_row = 1
        self.prediction_body_row = (
            self.target_body_row
            if provider_variant == "body_conditioned_adapter"
            else 0
        )
        self.policy_row = 0
        for parameter in core.parameters():
            parameter.requires_grad_(False)
        body = core.action_encoder.body_embedding.weight
        mask = torch.zeros_like(body)
        if provider_variant == "body_conditioned_adapter":
            body.requires_grad_(True)
            mask[self.target_body_row] = 1
        else:
            self.clock_beta.requires_grad_(False)
            self.clock_log_step_scale.requires_grad_(False)
        self.register_buffer("_body_gradient_mask", mask, persistent=False)
        if provider_variant == "body_conditioned_adapter":
            body.register_hook(
                lambda gradient: gradient * self._body_gradient_mask.to(gradient)
            )
        self._source_state = {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}
        self._source_sha = state_dict_sha256(self._source_state)
        self._smolvla_policy_sha = tensor_sha256(self._source_state[POLICY_EMBEDDING][0])
        self._openvla_policy_sha = tensor_sha256(self._source_state[POLICY_EMBEDDING][1])
        self._source_rank_contract = (
            None
            if source_rank_contract is None
            else _validate_source_rank_score_contract(source_rank_contract)
        )

    def train(self, mode: bool = True) -> "SmolVLAPiperAdapter":
        # Frozen includes stochastic state: source dropout must not silently
        # become target-training noise when the adapter enters train mode.
        super().train(mode)
        self.core.eval()
        return self

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict factual targets, including the unadjusted success logit.

        Dense per-transition success BCE intentionally stays on this base logit,
        matching source-core factual supervision. Root-candidate ranking must use
        :meth:`predict_grouped_candidates`, which adds the frozen source model's
        baseline-relative action-rank residual.
        """
        state = self.state_adapter(batch["state"])
        actions = self.action_adapter(batch["actions"])
        count = state.shape[0]
        if self.provider_variant == "body_agnostic_adapter":
            self._verify_body_agnostic_clock()
            beta = state.new_zeros(count)
            dt = batch["dt"]
        else:
            beta = self.clock_beta.to(state).expand(count)
            dt = batch["dt"] * self.clock_log_step_scale.exp().to(state)
        return self.core(
            state,
            actions,
            history_mask=batch.get("history_mask"),
            action_mask=batch["action_mask"],
            action_feature_mask=torch.ones_like(actions, dtype=torch.bool),
            proprio=batch["proprio"],
            body_id=torch.full((count,), self.prediction_body_row, dtype=torch.long, device=state.device),
            policy_id=torch.full((count,), self.policy_row, dtype=torch.long, device=state.device),
            current_event_id=batch["current_event_id"],
            clock_event_id=batch["current_event_id"],
            current_predicates=batch["current_predicates"],
            beta=beta,
            dt=dt,
        )

    def _verify_body_agnostic_clock(self) -> None:
        if self.provider_variant != "body_agnostic_adapter":
            return
        if (
            self.clock_beta.requires_grad
            or self.clock_log_step_scale.requires_grad
            or not torch.equal(
                self.clock_beta.detach(), torch.zeros_like(self.clock_beta)
            )
            or not torch.equal(
                self.clock_log_step_scale.detach(),
                torch.zeros_like(self.clock_log_step_scale),
            )
        ):
            raise AdapterContractError(
                "body-agnostic adapter clock parameters must remain frozen at exact zero"
            )

    def predict_grouped_candidates(
        self, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Predict same-root candidates with the Source-core residual contract.

        This consumes the exact Source63 training score: factual success term,
        canonical event-progress term and source-duration normalization, plus
        ``group_action_rank_residual``. The Piper loader permits candidate-row
        permutation, so the strict metadata validator locates the unique
        lowest-legal baseline per logical group. Unlike the core's historical
        convenience API, the composite score is never relabelled as a success
        logit or probability here.
        """

        _validate_grouped_candidate_batch(batch)
        validate_production_source_rank_config(self.core.config)
        if self.core.action_rank_head is None:
            raise AdapterContractError(
                "grouped Piper candidate prediction requires the production "
                "Source action-rank residual head"
            )
        output = dict(self.forward(batch))
        base_success_logit = output["success_logit"]
        try:
            residual = group_action_rank_residual(
                self.core,
                output,
                batch["ranking_group_index"],
                batch["ranking_baseline_mask"],
                detach_features=False,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AdapterContractError(
                "Source action-rank residual rejected grouped candidates"
            ) from error
        contract = _validate_source_rank_score_contract(self._source_rank_contract)
        event_values = torch.as_tensor(
            contract["event_values"],
            dtype=base_success_logit.dtype,
            device=base_success_logit.device,
        )
        base_rank_score = candidate_rank_score(
            output,
            event_values,
            float(contract["duration_scale"]),
            success_temperature=float(contract["success_temperature"]),
            event_weight=float(contract["event_weight"]),
            duration_weight=float(contract["duration_weight"]),
        )
        # The Formal190 and online selectors require the exact same numerical
        # quantity.  Fail closed under autocast/float64 instead of silently
        # creating a score that a downstream float32 verifier cannot reproduce.
        if base_rank_score.dtype != torch.float32 or residual.dtype != torch.float32:
            raise AdapterContractError(
                "Source rank algebra requires native IEEE float32 tensors"
            )
        temperature32 = torch.as_tensor(
            contract["success_temperature"],
            dtype=torch.float32,
            device=base_rank_score.device,
        )
        if not bool(torch.isfinite(temperature32)) or not bool(temperature32 > 0):
            raise AdapterContractError("Source rank temperature is invalid in float32")
        scaled_residual = torch.div(residual, temperature32)
        source_contract_rank_score = torch.add(base_rank_score, scaled_residual)
        if not (
            base_success_logit.ndim == 1
            and base_success_logit.shape == residual.shape
            and base_rank_score.shape == residual.shape
            and source_contract_rank_score.dtype == torch.float32
            and bool(torch.isfinite(source_contract_rank_score).all())
        ):
            raise AdapterContractError("grouped candidate Source rank scores are invalid")
        output["base_success_logit"] = base_success_logit
        output["action_rank_residual"] = residual
        output["source_contract_base_rank_score"] = base_rank_score
        output["source_contract_rank_score"] = source_contract_rank_score
        # ``success_logit`` deliberately remains the factual dense head. The
        # Source63 composite score is neither a logit nor a probability.
        return output

    @torch.no_grad()
    def enforce_and_verify_frozen_core(self) -> dict[str, Any]:
        current = self.core.state_dict()
        learned_body = (
            current[BODY_EMBEDDING][self.target_body_row].detach().clone()
            if self.provider_variant == "body_conditioned_adapter"
            else None
        )
        for name, frozen in self._source_state.items():
            current[name].copy_(frozen.to(current[name]))
        if learned_body is not None:
            current[BODY_EMBEDDING][self.target_body_row].copy_(learned_body)
        for name, value in current.items():
            reference = self._source_state[name].to(value)
            if (
                name == BODY_EMBEDDING
                and self.provider_variant == "body_conditioned_adapter"
            ):
                if not torch.equal(value[0], reference[0]):
                    raise AdapterContractError("source body row changed")
                continue
            if not torch.equal(value, reference):
                raise AdapterContractError(f"frozen core tensor changed: {name}")
        policy = current[POLICY_EMBEDDING]
        if tensor_sha256(policy[0]) != self._smolvla_policy_sha or tensor_sha256(policy[1]) != self._openvla_policy_sha:
            raise AdapterContractError("SmolVLA/OpenVLA policy rows are not bit-exact")
        audit = {
            "all_core_tensors_except_piper_body_row_bit_exact": True,
            "smolvla_policy_row_0_sha256": self._smolvla_policy_sha,
            "reserved_openvla_policy_row_1_sha256": self._openvla_policy_sha,
            "piper_body_row_sha256": tensor_sha256(current[BODY_EMBEDDING][1]),
        }
        if self.provider_variant == "body_agnostic_adapter":
            self._verify_body_agnostic_clock()
            audit.update(
                {
                    "provider_variant": "body_agnostic_adapter",
                    "prediction_body_row": 0,
                    "reserved_target_body_row": 1,
                    "all_core_tensors_bit_exact": True,
                    "reserved_target_body_row_bit_exact": True,
                    "clock_beta_fixed_exact_zero": True,
                    "clock_log_step_scale_fixed_exact_zero": True,
                }
            )
        return audit

    def trainable_parameter_audit(self) -> dict[str, Any]:
        named = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        expected = {
            "state_adapter.down.weight",
            "state_adapter.up.weight",
            "action_adapter.diagonal",
            "action_adapter.down.weight",
            "action_adapter.up.weight",
        }
        if self.provider_variant == "body_conditioned_adapter":
            expected.update(
                {
                    "clock_beta",
                    "clock_log_step_scale",
                    f"core.{BODY_EMBEDDING}",
                }
            )
        if set(named) != expected:
            raise AdapterContractError(
                "trainable parameter set changed: "
                f"missing={sorted(expected-set(named))}, extra={sorted(set(named)-expected)}"
            )
        count = sum(parameter.numel() for parameter in named.values())
        effective = (
            count - self.core.config.metadata_dim
            if self.provider_variant == "body_conditioned_adapter"
            else count
        )
        audit = {
            "names": sorted(named),
            "parameter_tensor_count": len(named),
            "optimizer_parameter_scalars": count,
            "effective_target_parameter_scalars": effective,
            "effective_core_rows": (
                {BODY_EMBEDDING: [1]}
                if self.provider_variant == "body_conditioned_adapter"
                else {}
            ),
            "policy_rows_trainable": [],
            "only_authorized_components": True,
        }
        if self.provider_variant == "body_agnostic_adapter":
            self._verify_body_agnostic_clock()
            audit.update(
                {
                    "provider_variant": "body_agnostic_adapter",
                    "prediction_body_row": 0,
                    "reserved_target_body_row": 1,
                    "clock_parameters_trainable": False,
                    "core_parameters_trainable": False,
                }
            )
        return audit


@dataclass(frozen=True)
class GroupDescriptor:
    logical_group_id: str
    requested_seed: int
    resolved_seed: int
    task: str
    body: str
    policy: str
    path: Path
    file_sha256: str


def scan_manifest(path: Path) -> tuple[dict[str, Any], list[GroupDescriptor]]:
    """Read only label-free JSON/file metadata; never open an HDF5 container."""

    manifest = _load_json(reject_sensitive_path(path, "schema6 manifest"), "schema6 manifest")
    unsigned = dict(manifest)
    signature = unsigned.pop("manifest_sha256", None)
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("status") != "complete" or signature != canonical_sha256(unsigned):
        raise AdapterContractError("schema6 training manifest signature/status is invalid")
    if manifest.get("fresh_inputs_used") is not False or manifest.get("sealed_test_labels_disclosed") is not False:
        raise AdapterContractError("manifest does not prove non-Fresh label secrecy")
    rows = manifest.get("groups")
    if not isinstance(rows, list) or len(rows) < 3:
        raise AdapterContractError("schema6 manifest requires at least three groups")
    root = path.parent.resolve()
    result = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {"logical_group_id", "requested_seed", "resolved_seed", "task", "body", "policy", "path", "file_sha256"}:
            raise AdapterContractError("manifest group identity fields changed")
        logical = str(raw["logical_group_id"])
        task = str(raw["task"])
        if not logical or logical.strip() != logical or logical in seen:
            raise AdapterContractError("duplicate/empty logical group")
        seen.add(logical)
        requested = _require_nonnegative_int(raw["requested_seed"], f"group {logical} requested seed")
        resolved = _require_nonnegative_int(raw["resolved_seed"], f"group {logical} resolved seed")
        if not task or task.strip() != task:
            raise AdapterContractError("group task must be a non-empty canonical string")
        if raw["body"] != TARGET_BODY or raw["policy"] != SOURCE_POLICY or not _is_sha(raw["file_sha256"]):
            raise AdapterContractError("group is not Piper with the same SmolVLA policy")
        raw_path = Path(str(raw["path"]))
        group_path = reject_sensitive_path(
            raw_path if raw_path.is_absolute() else root / raw_path,
            f"group {logical}",
        )
        # Relative legacy manifests remain confined to their manifest root.
        # Formal v2 aggregation emits absolute paths because the completed
        # collection authority tree is already frozen and cannot contain the
        # later manifest output.  Absolute paths are accepted only through the
        # externally SHA-bound manifest/split receipt validated below.
        if not raw_path.is_absolute() and group_path.parent != root and root not in group_path.parents:
            raise AdapterContractError("relative group escapes manifest root")
        result.append(GroupDescriptor(logical, requested, resolved, task, TARGET_BODY, SOURCE_POLICY, group_path, str(raw["file_sha256"])))
    if len({item.requested_seed for item in result}) != len(result) or len({item.resolved_seed for item in result}) != len(result):
        raise AdapterContractError("requested/resolved seeds must both be unique")
    return manifest, result


def _verify_json_signature(
    value: Mapping[str, Any], key: str, role: str
) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise AdapterContractError(f"{role} logical SHA256 mismatch")
    return str(recorded)


def _bound_json_record(value: Any, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "file_sha256", "logical_sha256"
    }:
        raise AdapterContractError(f"{role} binding fields changed")
    result = {key: str(value[key]) for key in value}
    if not _is_sha(result["file_sha256"]) or not _is_sha(result["logical_sha256"]):
        raise AdapterContractError(f"{role} binding SHA256 is invalid")
    return result


def validate_external_split_authority(
    *,
    expected_receipt_path: Path,
    expected_receipt_file_sha256: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    descriptors: Sequence[GroupDescriptor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a receipt-selected exact split profile before any HDF access."""

    receipt_path = reject_sensitive_path(
        expected_receipt_path, "expected manifest/split receipt"
    )
    if not _is_sha(expected_receipt_file_sha256) or file_sha256(
        receipt_path
    ) != expected_receipt_file_sha256:
        raise AdapterContractError("expected manifest/split receipt file SHA256 mismatch")
    receipt = _load_json(receipt_path, "expected manifest/split receipt")
    receipt_logical = _verify_json_signature(
        receipt, "expected_receipt_sha256", "expected manifest/split receipt"
    )
    receipt_format = receipt.get("format")
    profile = (
        SPLIT_PROFILES_BY_EXPECTED_FORMAT.get(receipt_format)
        if isinstance(receipt_format, str)
        else None
    )
    if profile is None:
        raise AdapterContractError("expected manifest/split receipt profile is unsupported")
    expected_fields = {
        "format", "status", "trainer_compatible_manifest", "target_partition",
        "external_split", "bound_trainer_implementation",
        "required_trainer_group_counts",
        "direct_bound_trainer_execution_authorized", "hdf5_content_files_opened",
        "labels_read", "expected_receipt_sha256",
    }
    if profile.explicit_profile_field:
        expected_fields.add("split_profile")
    if (
        set(receipt) != expected_fields
        or receipt.get("format") != profile.expected_receipt_format
        or (
            profile.explicit_profile_field
            and receipt.get("split_profile") != profile.name
        )
        or receipt.get("status")
        != "complete_external_manifest_and_split_expectations"
        or receipt.get("required_trainer_group_counts")
        != profile.required_trainer_group_counts
        or receipt.get("direct_bound_trainer_execution_authorized") is not True
        or receipt.get("hdf5_content_files_opened") != 0
        or receipt.get("labels_read") is not False
    ):
        raise AdapterContractError("expected manifest/split receipt scope changed")

    trainer_binding = receipt.get("bound_trainer_implementation")
    if not isinstance(trainer_binding, Mapping) or set(trainer_binding) != {
        "path", "file_sha256"
    }:
        raise AdapterContractError("bound trainer implementation fields changed")
    trainer_path = reject_sensitive_path(
        Path(str(trainer_binding["path"])), "bound trainer implementation"
    )
    this_trainer = Path(__file__).resolve()
    if (
        trainer_path != this_trainer
        or not _is_sha(trainer_binding.get("file_sha256"))
        or file_sha256(this_trainer) != trainer_binding["file_sha256"]
    ):
        raise AdapterContractError("formal trainer implementation SHA/path mismatch")

    manifest_binding = _bound_json_record(
        receipt.get("trainer_compatible_manifest"), "trainer manifest"
    )
    external_binding_raw = receipt.get("external_split")
    external_logical = (
        external_binding_raw.get("logical_sha256")
        if isinstance(external_binding_raw, Mapping)
        else None
    )
    bound_manifest_path = reject_sensitive_path(
        Path(manifest_binding["path"]), "bound trainer manifest"
    )
    if (
        bound_manifest_path != manifest_path.resolve()
        or file_sha256(bound_manifest_path) != manifest_binding["file_sha256"]
        or manifest.get("manifest_sha256") != manifest_binding["logical_sha256"]
        or manifest.get("expected_external_split_sha256")
        != external_logical
    ):
        raise AdapterContractError("trainer manifest differs from external expectation")

    partition_binding = _bound_json_record(
        receipt.get("target_partition"), "target partition"
    )
    split_binding = _bound_json_record(receipt.get("external_split"), "external split")
    partition_path = reject_sensitive_path(
        Path(partition_binding["path"]), "target partition"
    )
    split_path = reject_sensitive_path(Path(split_binding["path"]), "external split")
    if (
        file_sha256(partition_path) != partition_binding["file_sha256"]
        or file_sha256(split_path) != split_binding["file_sha256"]
    ):
        raise AdapterContractError("external partition/split file SHA256 mismatch")
    partition = _load_json(partition_path, "target partition")
    split = _load_json(split_path, "external split")
    partition_logical = _verify_json_signature(
        partition, "partition_sha256", "target partition"
    )
    split_logical = _verify_json_signature(split, "split_sha256", "external split")
    if (
        partition_logical != partition_binding["logical_sha256"]
        or split_logical != split_binding["logical_sha256"]
    ):
        raise AdapterContractError("external partition/split logical SHA256 mismatch")

    partition_fields = {
        "format", "status", "adaptation", "validation", "evaluation",
        "evaluation_groups_included", "hdf5_files_opened_before_partition_freeze",
        "labels_read", "partition_sha256",
    }
    split_fields = {
        "format", "status", "algorithm", "seed", "train", "validation", "test",
        "source_partition_sha256",
        "target_validation_used_for_training_or_internal_validation",
        "evaluation_groups_included", "hdf5_files_opened_before_split_freeze",
        "labels_read", "split_sha256",
    }
    if profile.explicit_profile_field:
        partition_fields.update({"split_profile", "required_group_counts"})
        split_fields.update({"split_profile", "required_trainer_group_counts"})
    if (
        set(partition) != partition_fields
        or partition.get("format") != profile.target_partition_format
        or (
            profile.explicit_profile_field
            and (
                partition.get("split_profile") != profile.name
                or partition.get("required_group_counts")
                != {
                    "adaptation": profile.adaptation_groups,
                    "formal_target_validation": profile.sealed_test_groups,
                }
            )
        )
        or partition.get("status")
        != "frozen_from_target_seed_manifest_before_hdf_access"
        or partition.get("evaluation") != []
        or partition.get("evaluation_groups_included") != 0
        or partition.get("hdf5_files_opened_before_partition_freeze") != 0
        or partition.get("labels_read") is not False
        or set(split) != split_fields
        or split.get("format") != profile.external_split_format
        or (
            profile.explicit_profile_field
            and (
                split.get("split_profile") != profile.name
                or split.get("required_trainer_group_counts")
                != profile.required_trainer_group_counts
            )
        )
        or split.get("status") != "frozen_label_blind_before_hdf_access"
        or split.get("source_partition_sha256") != partition_logical
        or split.get("target_validation_used_for_training_or_internal_validation")
        is not False
        or split.get("evaluation_groups_included") != 0
        or split.get("hdf5_files_opened_before_split_freeze") != 0
        or split.get("labels_read") is not False
    ):
        raise AdapterContractError("external partition/split scope changed")

    adaptation = partition.get("adaptation")
    target_validation = partition.get("validation")
    train_ids, validation_ids, test_ids = (
        split.get("train"), split.get("validation"), split.get("test")
    )
    sequences = (adaptation, target_validation, train_ids, validation_ids, test_ids)
    if any(
        not isinstance(values, list)
        or any(not isinstance(item, str) or not item for item in values)
        or len(values) != len(set(values))
        for values in sequences
    ):
        raise AdapterContractError("external identity lists are malformed/duplicated")
    train_set, internal_validation_set, test_set = map(
        set, (train_ids, validation_ids, test_ids)
    )
    adaptation_set, target_validation_set = set(adaptation), set(target_validation)
    manifest_ids = {item.logical_group_id for item in descriptors}
    if (
        len(adaptation) != profile.adaptation_groups
        or len(target_validation) != profile.sealed_test_groups
        or len(train_ids) != profile.train_groups
        or len(validation_ids) != profile.internal_validation_groups
        or len(test_ids) != profile.sealed_test_groups
        or adaptation_set & target_validation_set
        or train_set & internal_validation_set
        or train_set & test_set
        or internal_validation_set & test_set
        or train_set | internal_validation_set != adaptation_set
        or test_set != target_validation_set
        or train_set | internal_validation_set | test_set != manifest_ids
    ):
        raise AdapterContractError(
            "external split is not exact disjoint "
            f"{profile.train_groups}/{profile.internal_validation_groups}/"
            f"{profile.sealed_test_groups} full coverage for {profile.name}"
        )
    return dict(split), {
        "split_profile": profile.name,
        "split_profile_version": profile.version,
        "split_profile_binding": (
            "signed_explicit_field_and_versioned_formats"
            if profile.explicit_profile_field
            else "signed_historical_v2_format_identity"
        ),
        "required_trainer_group_counts": profile.required_trainer_group_counts,
        "adaptation_group_count": profile.adaptation_groups,
        "formal_target_validation_group_count": profile.sealed_test_groups,
        "expected_receipt_path": str(receipt_path),
        "expected_receipt_file_sha256": expected_receipt_file_sha256,
        "expected_receipt_logical_sha256": receipt_logical,
        "trainer_implementation_sha256": trainer_binding["file_sha256"],
        "target_partition_sha256": partition_logical,
        "external_split_sha256": split_logical,
        "sealed_test_group_count": len(test_ids),
        "sealed_test_hdf5_files_opened": 0,
        "sealed_test_labels_opened": 0,
        "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_labels_opened_before_five_adapters_frozen": 0,
    }


def freeze_group_split(groups: Sequence[GroupDescriptor], *, seed: int, validation_fraction: float, test_fraction: float) -> dict[str, Any]:
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise AdapterContractError("invalid split fractions")
    strata: dict[tuple[str, str, str], list[GroupDescriptor]] = {}
    for group in groups:
        strata.setdefault((group.body, group.policy, group.task), []).append(group)
    assignments = {"train": [], "validation": [], "test": []}
    for stratum, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda group: hashlib.sha256(
                f"{seed}:{stratum}:{group.requested_seed}:{group.resolved_seed}:{group.logical_group_id}".encode()
            ).hexdigest(),
        )
        n = len(ordered)
        n_test = max(1, round(n * test_fraction))
        n_validation = max(1, round(n * validation_fraction))
        if n_test + n_validation >= n:
            raise AdapterContractError(
                f"stratum {stratum!r} needs at least three groups for train/validation/test"
            )
        assignments["test"].extend(group.logical_group_id for group in ordered[:n_test])
        assignments["validation"].extend(
            group.logical_group_id for group in ordered[n_test : n_test + n_validation]
        )
        assignments["train"].extend(
            group.logical_group_id for group in ordered[n_test + n_validation :]
        )
    split = {
        "format": SPLIT_FORMAT,
        "algorithm": "body_policy_task_stratified_sha256(seed:requested:resolved:logical)_v1",
        "seed": int(seed),
        "train": sorted(assignments["train"]),
        "validation": sorted(assignments["validation"]),
        "test": sorted(assignments["test"]),
        "hdf5_files_opened_before_split_freeze": 0,
    }
    split["split_sha256"] = canonical_sha256(split)
    return split


def reconstruct_pose_predicates(
    poses: np.ndarray,
    names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
    stored_events: np.ndarray,
) -> np.ndarray:
    """Derive reversible predicates from poses and bind dense event semantics."""

    from train_multibody_canonical_event_world_model import derive_predicates_and_events

    poses = np.asarray(poses, dtype=np.float32)
    stored_events = np.asarray(stored_events, dtype=np.int64)
    if (
        poses.ndim != 3
        or poses.shape[0] < 1
        or poses.shape[1] != len(names)
        or poses.shape[2] != 7
        or not np.isfinite(poses).all()
    ):
        raise AdapterContractError("schema6 object poses are not finite [T,O,7]")
    predicates, events = derive_predicates_and_events(
        poses, names, bool(success), calibration
    )
    if (
        predicates.shape != (len(poses), len(EXPECTED_PREDICATES))
        or not np.isfinite(predicates).all()
        or not np.array_equal(events, stored_events)
    ):
        raise AdapterContractError(
            "event-spec pose reconstruction disagrees with stored dense events"
        )
    return predicates.astype(np.float32)


def final_branch_success_targets(
    trajectory_success: np.ndarray,
    *,
    final_success: bool,
    steps: int,
) -> np.ndarray:
    """Bind every action-conditioned row to the eventual branch outcome.

    The source event core's success head and the root-candidate ranking loss both
    predict eventual task success.  ``trajectory_success`` is only a cumulative
    simulator diagnostic (normally false until the terminal transition), so
    using ``trajectory_success[1:]`` here would silently train a different,
    immediate-success target on almost every dense row.
    """

    observed = np.asarray(trajectory_success, dtype=bool)
    if (
        observed.shape != (steps + 1,)
        or bool(observed[0])
        or bool((observed[:-1] & ~observed[1:]).any())
        or bool(observed[-1]) != bool(final_success)
    ):
        raise AdapterContractError(
            "trajectory success is not a monotone diagnostic bound to final branch success"
        )
    return np.full(steps, float(bool(final_success)), dtype=np.float32)


def derive_conditional_recovery_targets(
    events: np.ndarray,
    *,
    right_censored: bool,
    persistence_states: int = RECOVERY_PERSISTENCE_STATES,
) -> dict[str, np.ndarray]:
    """Derive flicker-resistant recovery labels from dynamic canonical events.

    A transition is an operational regression only when the next three saved
    simulator states remain below the old trajectory peak.  Recovery is then a
    later three-state return to that peak, or a terminal eK.  A right-censored
    regression without an observed recovery has an unknown label, not a hard
    negative.  Only these conditional rows supervise the detached head.
    """

    values = np.asarray(events, dtype=np.int64)
    if (
        values.ndim != 1
        or len(values) < 2
        or isinstance(persistence_states, bool)
        or not isinstance(persistence_states, int)
        or persistence_states < 2
        or np.any(values < 0)
        or np.any(values >= len(EXPECTED_EVENTS))
    ):
        raise AdapterContractError("recovery derivation received invalid event states")
    steps = len(values) - 1
    regress = np.zeros(steps, dtype=bool)
    recovery = np.zeros(steps, dtype=np.float32)
    observed = np.zeros(steps, dtype=bool)
    terminal_success = EXPECTED_EVENTS.index("eK")
    for index in range(steps):
        old_peak = int(values[: index + 1].max())
        start = index + 1
        stop = start + persistence_states
        if old_peak <= 0 or stop > len(values):
            continue
        if not bool((values[start:stop] < old_peak).all()):
            continue
        regress[index] = True
        recovered = False
        for later in range(stop, len(values)):
            if int(values[later]) == terminal_success:
                recovered = True
                break
            later_stop = later + persistence_states
            if later_stop <= len(values) and bool(
                (values[later:later_stop] >= old_peak).all()
            ):
                recovered = True
                break
        if recovered:
            recovery[index] = 1.0
            observed[index] = True
        elif not right_censored:
            observed[index] = True
    if bool((recovery.astype(bool) & ~regress).any()) or bool(
        (observed & ~regress).any()
    ):
        raise AdapterContractError("recovery supervision escaped conditional regress")
    return {
        "regress": regress,
        "recovery": recovery,
        "recovery_observed": observed,
    }


def _read_group(
    descriptor: GroupDescriptor,
    object_dim: int,
    *,
    object_names: Sequence[str],
    canonical_calibration: Mapping[str, Any],
    include_canonical_state: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # This bytewise check happens only after the split receipt is durably
    # frozen.  The training process never invokes this function for test rows.
    if file_sha256(descriptor.path) != descriptor.file_sha256:
        raise AdapterContractError("group file SHA mismatch")
    validate_schema6_group_file(descriptor.path)
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    with h5py.File(descriptor.path, "r") as handle:
        if handle.attrs.get("format") != SCHEMA6_GROUP_FORMAT or handle.attrs.get("intervention") != SCHEMA6_INTERVENTION:
            raise AdapterContractError("schema6 group contract changed")
        baseline_index = int(handle.attrs["baseline_original_candidate_index"])
        for branch_name in sorted(handle["branches"]):
            branch = handle["branches"][branch_name]
            steps = int(branch.attrs["steps"])
            if steps < 1:
                continue
            selected = branch["query_selected_original_candidate_index"][:steps].astype(int)
            mapped = branch["query_mapped_actions"][:steps].astype(np.float32)
            feasibility = branch["query_feasibility_mask"][:steps].astype(bool)
            masks = branch["query_executed_action_mask"][:steps].astype(bool)
            if np.any(selected < 0) or not np.all([feasibility[i, selected[i]] for i in range(steps)]) or not np.all(masks.sum(1) == 1) or not np.all(masks[:, 0]):
                raise AdapterContractError("selected chunk/H1 feasibility contract changed")
            actions = np.stack([mapped[i, selected[i]] for i in range(steps)])
            events = branch["trajectory_event_id"][:].astype(np.int64)
            post = events[1:]
            next_event = branch["transition_next_event_id"][:].astype(np.int64)
            duration = branch["transition_duration_decision_steps"][:].astype(np.float32)
            observed = branch["transition_duration_observed"][:].astype(bool)
            censored = branch["transition_duration_censored"][:].astype(bool)
            final_success = bool(branch.attrs["success_diagnostic_only"])
            success = final_branch_success_targets(
                branch["trajectory_success"][:],
                final_success=final_success,
                steps=steps,
            )
            recovery = derive_conditional_recovery_targets(
                events,
                right_censored=bool(branch.attrs["right_censored"]),
            )
            if (
                not (len(events) == steps + 1 and len(next_event) == len(duration) == len(success) == steps)
                or not np.array_equal(~observed, censored)
                or not np.isfinite(mapped).all()
                or not np.isfinite(actions).all()
                or not np.isfinite(duration).all()
                or not np.isfinite(success).all()
                or np.any(events < 0)
                or np.any(events >= len(EXPECTED_EVENTS))
                or np.any(next_event < 0)
                or np.any(next_event >= len(EXPECTED_EVENTS))
            ):
                raise AdapterContractError("dense event/duration/success lengths changed")
            supervision = load_object_delta_supervision_v6(branch, start_steps=np.arange(steps), end_steps=np.arange(1, steps + 1), expected_registry_sha256=str(handle.attrs["object_registry_sha256"]), expected_spec_sha256=str(handle.attrs["pose_integrity_spec_sha256"]))
            quality = branch["pose_quality_v6"]
            raw_registry = quality["object_registry_json"][()]
            if isinstance(raw_registry, bytes):
                raw_registry = raw_registry.decode("utf-8")
            registry = json.loads(str(raw_registry))
            registry_names = [str(item["name"]) for item in registry["objects"]]
            if any(name not in registry_names for name in object_names):
                raise AdapterContractError("source selected object is absent from schema6 registry")
            object_indices = [registry_names.index(name) for name in object_names]
            delta = supervision["object_delta_xyz_m"][:, object_indices].reshape(steps, -1).astype(np.float32)
            valid = np.repeat(
                supervision["object_delta_supervision_valid"][:, object_indices], 3, axis=1
            ).reshape(steps, -1)
            if delta.shape != (steps, object_dim) or valid.shape != delta.shape:
                raise AdapterContractError("selected schema6 object target differs from source head")
            if not np.isfinite(delta).all():
                raise AdapterContractError("schema6 object supervision contains non-finite values")
            from train_multibody_canonical_event_world_model import canonical_state_vector

            names = registry_names
            poses = branch["object_poses"][:].astype(np.float32)
            derived_predicates = reconstruct_pose_predicates(
                poses,
                names,
                bool(branch.attrs["success_diagnostic_only"]),
                canonical_calibration,
                events,
            )
            predicates = derived_predicates[:-1].astype(np.float32)
            canonical_states: np.ndarray | None = None
            if include_canonical_state:
                canonical_states = np.stack(
                    [
                        canonical_state_vector(
                            poses, names, index, descriptor.task,
                            canonical_calibration, derived_predicates, int(events[index]),
                        )
                        for index in range(steps)
                    ]
                ).astype(np.float32)
            hidden = branch["query_hidden"][:steps].astype(np.float32)
            proprio = branch["query_processed_state"][:steps].astype(np.float32)
            if not np.isfinite(hidden).all() or not np.isfinite(proprio).all():
                raise AdapterContractError("schema6 state/proprio contains non-finite values")
            for index in range(steps):
                state_history, history_mask = schema6_causal_history_at_query(
                    hidden, index
                )
                row = {
                    "state": state_history,
                    "history_mask": history_mask,
                    "actions": actions[index], "action_mask": masks[index],
                    "proprio": proprio[index], "current_event_id": events[index], "current_predicates": predicates[index],
                    "post_event_id": post[index], "next_event_id": next_event[index], "duration": duration[index],
                    "duration_observed": observed[index], "success": success[index], "object_delta": delta[index],
                    "object_mask": valid[index], "logical_group_id": descriptor.logical_group_id,
                    "regress": recovery["regress"][index],
                    "recovery": recovery["recovery"][index],
                    "recovery_observed": recovery["recovery_observed"][index],
                    "causal_branch_id": branch_name,
                    "causal_query_index": index,
                }
                if canonical_states is not None:
                    row["canonical_state27"] = canonical_states[index]
                rows.append(row)
            candidates.append({
                "original_candidate_index": int(branch.attrs["original_candidate_index"]),
                "is_baseline": bool(branch.attrs["is_feasibility_baseline"]),
                "final_success": int(final_success),
                "root_row": rows[-steps],
            })
        if not candidates or sum(item["is_baseline"] for item in candidates) != 1:
            raise AdapterContractError("paired root baseline is missing or ambiguous")
        indices = [item["original_candidate_index"] for item in candidates]
        if (
            len(set(indices)) != len(indices)
            or any(index not in range(4) for index in indices)
            or len(candidates) < 2
        ):
            raise AdapterContractError("paired root candidate registry changed")
        baseline = next(item for item in candidates if item["is_baseline"])
        if baseline["original_candidate_index"] != baseline_index or baseline_index != min(item["original_candidate_index"] for item in candidates):
            raise AdapterContractError("paired baseline is not lowest legal")
        root_reference = candidates[0]["root_row"]
        for candidate in candidates[1:]:
            root = candidate["root_row"]
            if (
                not np.array_equal(root["state"], root_reference["state"])
                or not np.array_equal(
                    root["history_mask"], root_reference["history_mask"]
                )
                or not np.array_equal(root["proprio"], root_reference["proprio"])
                or root["current_event_id"] != root_reference["current_event_id"]
                or not np.array_equal(
                    root["current_predicates"], root_reference["current_predicates"]
                )
            ):
                raise AdapterContractError("paired candidates do not share one root state")
    return rows, {"logical_group_id": descriptor.logical_group_id, "candidates": candidates}


def read_train_and_internal_validation_groups(
    *,
    split: Mapping[str, Sequence[str]],
    descriptors: Sequence[GroupDescriptor],
    calibration_by_task: Mapping[str, Mapping[str, Any]],
    object_delta_dim: int,
    object_names: Sequence[str],
    include_canonical_state: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Open only adaptation train/internal groups; never dereference test IDs.

    Keeping this access boundary in one function makes the 50/190 sealed lane
    auditable and prevents profile expansion from accidentally adding a third
    HDF5-reading loop.
    """

    by_id = {item.logical_group_id: item for item in descriptors}
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_pairs: list[dict[str, Any]] = []
    validation_pairs: list[dict[str, Any]] = []
    lane_outputs = (
        ("train", train_rows, train_pairs),
        ("validation", validation_rows, validation_pairs),
    )
    for lane, output_rows, output_pairs in lane_outputs:
        for logical in split[lane]:
            descriptor = by_id.get(logical)
            if descriptor is None:
                raise AdapterContractError(f"{lane} group is absent from manifest")
            calibration = calibration_by_task[descriptor.task]
            rows, paired = _read_group(
                descriptor,
                object_delta_dim,
                object_names=object_names,
                canonical_calibration=calibration,
                include_canonical_state=include_canonical_state,
            )
            output_rows.extend(rows)
            output_pairs.append(paired)
    return train_rows, validation_rows, train_pairs, validation_pairs


class RowDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            raise AdapterContractError("split has no transition rows")
        self.rows = list(rows)
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def _logical_group_row_indices(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        logical = row.get("logical_group_id")
        if not isinstance(logical, str) or not logical:
            raise AdapterContractError("dense row logical group id is invalid")
        grouped.setdefault(logical, []).append(index)
    if not grouped or any(not indices for indices in grouped.values()):
        raise AdapterContractError("dense logical-group registry is empty")
    return {logical: grouped[logical] for logical in sorted(grouped)}


class LogicalGroupEqualizedSampler(Sampler[int]):
    """Draw exactly one transition per logical group in each deterministic epoch.

    Within each group, a frozen seeded row permutation is traversed cyclically.
    Thus long trajectories provide more distinct transitions over time but never
    receive more optimizer mass per epoch than short trajectories.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AdapterContractError("dense group sampler seed is invalid")
        self.seed = seed
        self.grouped = _logical_group_row_indices(rows)
        generator = np.random.default_rng(seed)
        self.row_cycles = {
            logical: [indices[int(offset)] for offset in generator.permutation(len(indices))]
            for logical, indices in self.grouped.items()
        }
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.grouped)

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        generator = np.random.default_rng(self.seed + epoch)
        logical_ids = list(self.grouped)
        for offset in generator.permutation(len(logical_ids)):
            logical = logical_ids[int(offset)]
            cycle = self.row_cycles[logical]
            yield cycle[epoch % len(cycle)]

    def audit(self) -> dict[str, Any]:
        counts = {logical: len(indices) for logical, indices in self.grouped.items()}
        return {
            "algorithm": "one_transition_per_logical_group_per_epoch_seeded_cyclic_v1",
            "seed": self.seed,
            "logical_group_count": len(counts),
            "samples_per_epoch": len(counts),
            "minimum_rows_per_group": min(counts.values()),
            "maximum_rows_per_group": max(counts.values()),
            "row_count_by_logical_group_sha256": canonical_sha256(counts),
            "trajectory_length_changes_group_sampling_mass": False,
            "repeated_success_targets_change_group_sampling_mass": False,
            "within_group_transition_coverage": "seeded_cycle_without_row_reweighting",
            "row_level_observation_masks_preserved": [
                "duration_observed_for_next_event_and_duration",
                "object_mask_for_object_delta",
                "recovery_observed_and_regress_for_recovery",
            ],
        }


class LogicalGroupBatchSampler(Sampler[list[int]]):
    """Yield one complete logical group per validation batch."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.grouped = _logical_group_row_indices(rows)

    def __len__(self) -> int:
        return len(self.grouped)

    def __iter__(self):
        for indices in self.grouped.values():
            yield list(indices)


class PairedGroupDataset(Dataset):
    def __init__(self, groups: Sequence[Mapping[str, Any]]) -> None:
        if not groups:
            raise AdapterContractError("split has no paired candidate groups")
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.groups[index]


def collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    floats = (
        "state", "actions", "proprio", "current_predicates", "duration",
        "success", "recovery", "object_delta",
    )
    integers = ("current_event_id", "post_event_id", "next_event_id")
    output: dict[str, Any] = {name: torch.as_tensor(np.stack([row[name] for row in rows]), dtype=torch.float32) for name in floats}
    if any(not bool(torch.isfinite(output[name]).all()) for name in floats):
        raise AdapterContractError("training batch contains non-finite floating values")
    if bool((output["duration"] < 0).any()) or bool(
        ((output["success"] != 0) & (output["success"] != 1)).any()
    ):
        raise AdapterContractError("duration/success target is outside its strict domain")
    output.update({name: torch.as_tensor([row[name] for row in rows], dtype=torch.long) for name in integers})
    output["action_mask"] = torch.as_tensor(np.stack([row["action_mask"] for row in rows]), dtype=torch.bool)
    output["duration_observed"] = torch.as_tensor([row["duration_observed"] for row in rows], dtype=torch.bool)
    output["regress"] = torch.as_tensor(
        [row["regress"] for row in rows], dtype=torch.bool
    )
    output["recovery_observed"] = torch.as_tensor(
        [row["recovery_observed"] for row in rows], dtype=torch.bool
    )
    if bool((output["recovery"].bool() & ~output["regress"]).any()) or bool(
        (output["recovery_observed"] & ~output["regress"]).any()
    ):
        raise AdapterContractError("recovery batch is not conditional on regress")
    output["object_mask"] = torch.as_tensor(np.stack([row["object_mask"] for row in rows]), dtype=torch.bool)
    output["history_mask"] = torch.as_tensor(
        np.stack([row["history_mask"] for row in rows]), dtype=torch.bool
    )
    if output["state"].shape != (
        len(rows),
        CAUSAL_HISTORY_MAX_STEPS,
        STATE_DIM,
    ) or output["history_mask"].shape != (
        len(rows),
        CAUSAL_HISTORY_MAX_STEPS,
    ):
        raise AdapterContractError("causal state history shape changed")
    for index, row in enumerate(rows):
        branch_id = row.get("causal_branch_id")
        query_index = row.get("causal_query_index")
        if (
            not isinstance(branch_id, str)
            or not branch_id
            or isinstance(query_index, bool)
            or not isinstance(query_index, int)
            or query_index < 0
        ):
            raise AdapterContractError("causal branch/query identity is invalid")
        expected_length = min(query_index + 1, CAUSAL_HISTORY_MAX_STEPS)
        expected_mask = torch.arange(CAUSAL_HISTORY_MAX_STEPS) < expected_length
        if not torch.equal(output["history_mask"][index], expected_mask):
            raise AdapterContractError("causal history mask is not a prefix mask")
        if bool(
            (output["state"][index][~output["history_mask"][index]] != 0).any()
        ):
            raise AdapterContractError("causal history padding is not exact zero")
    output["dt"] = torch.ones(len(rows), dtype=torch.float32)
    output["logical_group_id"] = [str(row["logical_group_id"]) for row in rows]
    teacher_presence = ["canonical_state27" in row for row in rows]
    if any(teacher_presence):
        if not all(teacher_presence):
            raise AdapterContractError("canonical teacher state is only partially present")
        output["canonical_state27"] = torch.as_tensor(
            np.stack([row["canonical_state27"] for row in rows]), dtype=torch.float32
        )
    return output


def collate_ranking_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    group_index: list[int] = []
    candidate_index: list[int] = []
    baseline_mask: list[bool] = []
    final_success: list[float] = []
    logical_ids: list[str] = []
    seen_logical_ids: set[str] = set()
    for index, group in enumerate(groups):
        logical_group_id = group.get("logical_group_id")
        if not isinstance(logical_group_id, str) or not logical_group_id:
            raise AdapterContractError("ranking logical group id is invalid")
        if logical_group_id in seen_logical_ids:
            raise AdapterContractError("ranking logical group appears more than once")
        seen_logical_ids.add(logical_group_id)
        candidates = group.get("candidates")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or len(candidates) < 2
        ):
            raise AdapterContractError("ranking group requires at least two candidates")
        if any(not isinstance(candidate, Mapping) for candidate in candidates):
            raise AdapterContractError("ranking candidate must be a mapping")
        if any(type(candidate.get("is_baseline")) is not bool for candidate in candidates):
            raise AdapterContractError("ranking candidate baseline flag must be boolean")
        if sum(bool(candidate.get("is_baseline")) for candidate in candidates) != 1:
            raise AdapterContractError("ranking group baseline is missing or ambiguous")
        original_indices = [
            candidate.get("original_candidate_index") for candidate in candidates
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in range(4)
            for value in original_indices
        ):
            raise AdapterContractError("candidate original index must be in [0,3]")
        if len(set(original_indices)) != len(original_indices):
            raise AdapterContractError(
                "ranking candidate original indices are not unique"
            )
        baseline_original_index = next(
            candidate.get("original_candidate_index")
            for candidate in candidates
            if bool(candidate.get("is_baseline"))
        )
        if baseline_original_index != min(original_indices):
            raise AdapterContractError(
                "ranking baseline is not the lowest legal candidate"
            )
        for candidate in candidates:
            outcome = candidate.get("final_success")
            original_index = candidate.get("original_candidate_index")
            if outcome not in (0, 1) or isinstance(outcome, bool):
                raise AdapterContractError("candidate final outcome must be integer 0/1")
            root_row = candidate.get("root_row")
            if not isinstance(root_row, Mapping):
                raise AdapterContractError("ranking candidate root row is invalid")
            if root_row.get("logical_group_id") != logical_group_id:
                raise AdapterContractError(
                    "ranking candidate crosses its declared logical group"
                )
            rows.append(root_row)
            group_index.append(index)
            candidate_index.append(original_index)
            baseline_mask.append(bool(candidate["is_baseline"]))
            final_success.append(float(outcome))
            logical_ids.append(logical_group_id)
    result = collate(rows)
    result.update(
        {
            "ranking_group_index": torch.as_tensor(group_index, dtype=torch.long),
            "ranking_candidate_index": torch.as_tensor(candidate_index, dtype=torch.long),
            "ranking_baseline_mask": torch.as_tensor(baseline_mask, dtype=torch.bool),
            "ranking_final_success": torch.as_tensor(final_success, dtype=torch.float32),
            "ranking_group_count": len(groups),
            "ranking_logical_group_id": logical_ids,
        }
    )
    return result


def _validate_grouped_candidate_batch(batch: Mapping[str, Any]) -> None:
    """Fail closed before applying a baseline-relative Source rank head."""

    required_tensors = {
        "ranking_group_index": torch.long,
        "ranking_candidate_index": torch.long,
        "ranking_baseline_mask": torch.bool,
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, dtype in required_tensors.items():
        value = batch.get(name)
        if not torch.is_tensor(value) or value.ndim != 1 or value.dtype != dtype:
            raise AdapterContractError(f"grouped candidate {name} is invalid")
        tensors[name] = value
    group_index = tensors["ranking_group_index"]
    candidate_index = tensors["ranking_candidate_index"]
    baseline_mask = tensors["ranking_baseline_mask"]
    row_count = int(group_index.numel())
    if (
        row_count < 2
        or candidate_index.shape != group_index.shape
        or baseline_mask.shape != group_index.shape
    ):
        raise AdapterContractError("grouped candidate metadata is not row aligned")
    group_count = batch.get("ranking_group_count")
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count < 1
    ):
        raise AdapterContractError("grouped candidate group count is invalid")
    if (
        group_index.device != candidate_index.device
        or group_index.device != baseline_mask.device
    ):
        raise AdapterContractError("grouped candidate metadata devices differ")
    unique_groups = torch.unique(group_index.detach().cpu(), sorted=True).tolist()
    if unique_groups != list(range(group_count)):
        raise AdapterContractError("grouped candidate indices are not canonical")
    logical_ids = batch.get("ranking_logical_group_id")
    row_logical_ids = batch.get("logical_group_id")
    if (
        not isinstance(logical_ids, list)
        or not isinstance(row_logical_ids, list)
        or len(logical_ids) != row_count
        or row_logical_ids != logical_ids
        or any(not isinstance(value, str) or not value for value in logical_ids)
    ):
        raise AdapterContractError("grouped candidate logical ids are invalid")
    group_names: set[str] = set()
    root_fields = (
        "state",
        "history_mask",
        "proprio",
        "current_event_id",
        "current_predicates",
    )
    for group in range(group_count):
        rows = torch.nonzero(group_index == group, as_tuple=False).squeeze(-1)
        if int(rows.numel()) < 2 or int(baseline_mask[rows].sum()) != 1:
            raise AdapterContractError(
                "every grouped candidate set needs one deterministic baseline"
            )
        row_numbers = rows.detach().cpu().tolist()
        names = {logical_ids[int(row)] for row in row_numbers}
        if len(names) != 1 or next(iter(names)) in group_names:
            raise AdapterContractError("grouped candidates cross logical groups")
        group_names.update(names)
        indices = candidate_index[rows]
        if (
            bool(((indices < 0) | (indices > 3)).any())
            or len(torch.unique(indices)) != len(indices)
        ):
            raise AdapterContractError("grouped candidate indices are invalid")
        baseline_index = indices[baseline_mask[rows]][0]
        if int(baseline_index) != int(indices.min()):
            raise AdapterContractError(
                "grouped baseline is not the lowest legal candidate"
            )
        anchor_row = int(rows[0])
        for field in root_fields:
            value = batch.get(field)
            if not torch.is_tensor(value) or value.shape[0] != row_count:
                raise AdapterContractError(
                    f"grouped candidate shared-root field {field} is invalid"
                )
            if not all(
                torch.equal(value[int(row)], value[anchor_row])
                for row in row_numbers
            ):
                raise AdapterContractError(
                    f"grouped candidates do not share root field {field}"
                )


def group_weighted_ranking_loss(
    logits: torch.Tensor, batch: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Baseline-relative pairwise/listwise losses, averaged equally by group."""

    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise AdapterContractError("candidate ranking logits must be finite [N]")
    group_index = batch["ranking_group_index"]
    baseline_mask = batch["ranking_baseline_mask"]
    outcomes = batch["ranking_final_success"].to(logits)
    if not (logits.shape == group_index.shape == baseline_mask.shape == outcomes.shape):
        raise AdapterContractError("candidate ranking tensors are not aligned")
    pair_losses: list[torch.Tensor] = []
    list_losses: list[torch.Tensor] = []
    discordant = 0
    group_count = int(batch["ranking_group_count"])
    for group in range(group_count):
        mask = group_index == group
        group_logits = logits[mask]
        group_outcomes = outcomes[mask]
        group_baseline = baseline_mask[mask]
        if len(group_logits) < 2 or int(group_baseline.sum()) != 1:
            raise AdapterContractError("candidate ranking group structure changed")
        baseline_logit = group_logits[group_baseline][0]
        baseline_outcome = group_outcomes[group_baseline][0]
        comparable = (~group_baseline) & (group_outcomes != baseline_outcome)
        if bool(comparable.any()):
            targets = (group_outcomes[comparable] > baseline_outcome).to(logits)
            pair_losses.append(
                F.binary_cross_entropy_with_logits(
                    group_logits[comparable] - baseline_logit, targets
                )
            )
        if bool((group_outcomes != group_outcomes[0]).any()):
            discordant += 1
            relative_logits = group_logits - baseline_logit
            target = group_outcomes / group_outcomes.sum()
            list_losses.append(-(target * F.log_softmax(relative_logits, dim=0)).sum())
    zero = logits.sum() * 0
    pairwise = torch.stack(pair_losses).mean() if pair_losses else zero
    listwise = torch.stack(list_losses).mean() if list_losses else zero
    return pairwise, listwise, {
        "groups": group_count,
        "discordant_groups": discordant,
        "pairwise_informative_groups": len(pair_losses),
    }


@dataclass(frozen=True)
class SupportGate:
    min_train_groups: int = DEFAULT_MIN_TRAIN_GROUPS
    min_validation_groups: int = DEFAULT_MIN_VALIDATION_GROUPS
    min_test_groups: int = DEFAULT_MIN_TEST_GROUPS
    min_outcome_groups: int = DEFAULT_MIN_OUTCOME_GROUPS
    min_discordant_groups: int = DEFAULT_MIN_DISCORDANT_GROUPS
    min_event_rows: int = DEFAULT_MIN_EVENT_ROWS
    min_duration_rows: int = DEFAULT_MIN_DURATION_ROWS
    min_object_rows: int = DEFAULT_MIN_OBJECT_ROWS
    min_candidate_index_groups: int = 1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _require_nonnegative_int(value, f"support gate {name}")

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)

    def require_formal_minimums(self) -> None:
        formal = SupportGate()
        lowered = {
            name: {"configured": value, "formal_minimum": getattr(formal, name)}
            for name, value in self.__dict__.items()
            if value < getattr(formal, name)
        }
        if lowered:
            raise AdapterContractError(
                f"formal training cannot lower support gates: {lowered}"
            )


def validate_group_count_gate(
    split: Mapping[str, Sequence[str]], gate: SupportGate,
    groups: Sequence[GroupDescriptor] | None = None,
    *, exact_profile: ExternalSplitProfile | None = None,
) -> dict[str, Any]:
    counts = {name: len(split[name]) for name in ("train", "validation", "test")}
    if exact_profile is not None and counts != exact_profile.required_trainer_group_counts:
        raise AdapterContractError(
            f"{exact_profile.name} exact group count gate failed: "
            f"actual={counts}, required={exact_profile.required_trainer_group_counts}"
        )
    required = {
        "train": gate.min_train_groups,
        "validation": gate.min_validation_groups,
        "test": gate.min_test_groups,
    }
    failed = {
        name: {"actual": counts[name], "required": required[name]}
        for name in counts
        if counts[name] < required[name]
    }
    if failed:
        raise AdapterContractError(f"independent group support gate failed: {failed}")
    per_stratum: dict[str, dict[str, int]] = {}
    if groups is not None:
        by_id = {group.logical_group_id: group for group in groups}
        for name in ("train", "validation", "test"):
            for logical in split[name]:
                group = by_id[logical]
                key = "|".join((group.body, group.policy, group.task))
                per_stratum.setdefault(
                    key, {"train": 0, "validation": 0, "test": 0}
                )[name] += 1
        stratum_failed = {
            key: {
                name: {"actual": values[name], "required": required[name]}
                for name in values
                if values[name] < required[name]
            }
            for key, values in per_stratum.items()
        }
        stratum_failed = {key: value for key, value in stratum_failed.items() if value}
        if stratum_failed:
            raise AdapterContractError(
                f"per-stratum independent group support gate failed: {stratum_failed}"
            )
    return {
        "total": counts,
        "per_stratum": per_stratum,
        "exact_split_profile": None if exact_profile is None else exact_profile.name,
    }


def split_supervision_support(
    rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    event_counts = {
        name: sum(int(row["post_event_id"]) == index for row in rows)
        for index, name in enumerate(EXPECTED_EVENTS)
    }
    observed_next_event_counts = {
        name: sum(
            bool(row["duration_observed"])
            and int(row["next_event_id"]) == index
            for row in rows
        )
        for index, name in enumerate(EXPECTED_EVENTS)
    }
    outcomes = [
        [int(candidate["final_success"]) for candidate in group["candidates"]]
        for group in paired
    ]
    candidate_counts = {
        str(index): sum(
            any(
                int(candidate["original_candidate_index"]) == index
                for candidate in group["candidates"]
            )
            for group in paired
        )
        for index in range(4)
    }
    return {
        "transition_rows": len(rows),
        "groups": len(paired),
        "positive_groups": sum(any(values) for values in outcomes),
        "negative_groups": sum(not all(values) for values in outcomes),
        "discordant_groups": sum(len(set(values)) > 1 for values in outcomes),
        "event_post_rows": event_counts,
        "event_next_observed_rows": observed_next_event_counts,
        "duration_observed_rows": sum(bool(row["duration_observed"]) for row in rows),
        "duration_censored_rows": sum(not bool(row["duration_observed"]) for row in rows),
        "object_supervised_rows": sum(bool(np.asarray(row["object_mask"]).any()) for row in rows),
        "object_supervised_features": sum(int(np.asarray(row["object_mask"]).sum()) for row in rows),
        "conditional_recovery": conditional_recovery_group_support(rows),
        "candidate_index_group_support": candidate_counts,
    }


def conditional_recovery_group_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_groups_per_class: int = MIN_RECOVERY_GROUPS_PER_CLASS,
) -> dict[str, Any]:
    if (
        isinstance(minimum_groups_per_class, bool)
        or not isinstance(minimum_groups_per_class, int)
        or minimum_groups_per_class < 1
    ):
        raise AdapterContractError("recovery group support minimum must be positive")
    positive_groups: set[str] = set()
    negative_groups: set[str] = set()
    observed_rows = positive_rows = 0
    for row in rows:
        regress = bool(row["regress"])
        observed = bool(row["recovery_observed"])
        label = float(row["recovery"])
        if label not in (0.0, 1.0) or (observed and not regress) or (label == 1.0 and not observed):
            raise AdapterContractError("conditional recovery row contract is invalid")
        if not observed:
            continue
        observed_rows += 1
        group = str(row["logical_group_id"])
        if label == 1.0:
            positive_rows += 1
            positive_groups.add(group)
        else:
            negative_groups.add(group)
    positive = len(positive_groups)
    negative = len(negative_groups)
    enabled = min(positive, negative) >= minimum_groups_per_class
    return {
        "status": (
            "enabled_independent_group_support_passed"
            if enabled
            else "disabled_insufficient_independent_group_support"
        ),
        "enabled": enabled,
        "conditional_on": "operational_regress_persistent_three_states",
        "positive_independent_groups": positive,
        "negative_independent_groups": negative,
        "minimum_groups_per_class": minimum_groups_per_class,
        "observed_rows": observed_rows,
        "positive_rows": positive_rows,
        "negative_rows": observed_rows - positive_rows,
        "right_censored_nonrecoveries_are_unobserved": True,
    }


def validate_supervision_support(
    support: Mapping[str, Any], gate: SupportGate, *, split_name: str
) -> None:
    checks = {
        "positive_groups": gate.min_outcome_groups,
        "negative_groups": gate.min_outcome_groups,
        "discordant_groups": gate.min_discordant_groups,
        "duration_observed_rows": gate.min_duration_rows,
        "duration_censored_rows": gate.min_duration_rows,
        "object_supervised_rows": gate.min_object_rows,
    }
    failed = {
        name: {"actual": int(support[name]), "required": required}
        for name, required in checks.items()
        if int(support[name]) < required
    }
    post_event_failed = {
        name: int(count)
        for name, count in support["event_post_rows"].items()
        if int(count) < gate.min_event_rows
    }
    next_event_failed = {
        name: int(count)
        for name, count in support["event_next_observed_rows"].items()
        if name in EXPECTED_NEXT_REACHED_EVENTS and int(count) < gate.min_event_rows
    }
    candidate_failed = {
        name: int(count)
        for name, count in support["candidate_index_group_support"].items()
        if int(count) < gate.min_candidate_index_groups
    }
    if failed or post_event_failed or next_event_failed or candidate_failed:
        raise AdapterContractError(
            f"{split_name} supervision support gate failed: "
            f"scalar={failed}, post_events={post_event_failed}, "
            f"observed_next_events={next_event_failed}, candidates={candidate_failed}"
        )


def object_normalization(checkpoint: Mapping[str, Any], object_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = checkpoint.get("normalization")
    if not isinstance(raw, Mapping):
        raise AdapterContractError("source checkpoint lacks object normalization")
    mean = torch.as_tensor(raw.get("object_delta_mean"), dtype=torch.float32)
    std = torch.as_tensor(raw.get("object_delta_std"), dtype=torch.float32)
    if mean.shape != (object_dim,) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std <= 0).any()):
        raise AdapterContractError("source object normalization is invalid")
    return mean, std


def censored_lognormal_nll(mean: torch.Tensor, log_scale: torch.Tensor, duration: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    target = torch.log1p(duration.clamp_min(0))
    scale = log_scale.clamp(-8, 5).exp()
    z = (target - mean) / scale
    density = 0.5 * z.square() + torch.log(scale) + 0.5 * math.log(2 * math.pi)
    survival = -torch.log((0.5 * torch.erfc(z / math.sqrt(2))).clamp_min(1e-8))
    return torch.where(observed, density, survival)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return value.sum() * 0
    return value[mask].mean()


def compute_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], *, object_mean: torch.Tensor, object_std: torch.Tensor, semantic_target: torch.Tensor | None = None, semantic_weight: float = 0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    post = F.cross_entropy(output["next_event_logits"], batch["post_event_id"])
    # The collector stores current-event as a placeholder when the next event is
    # right-censored.  It is not an observed self-loop and must not supervise the
    # destination head (the source-core trainer uses the same mask).
    next_per_row = F.cross_entropy(
        output["next_reached_event_logits"], batch["next_event_id"], reduction="none"
    )
    nxt = masked_mean(next_per_row, batch["duration_observed"])
    duration = censored_lognormal_nll(output["duration_selected_log_mean"], output["duration_selected_log_scale"], batch["duration"], batch["duration_observed"]).mean()
    success = F.binary_cross_entropy_with_logits(output["success_logit"], batch["success"])
    target = (batch["object_delta"] - object_mean) / object_std
    scale = output["object_delta_log_scale"].clamp(-8, 5)
    object_nll = 0.5 * ((target - output["object_delta_mean"]) * torch.exp(-scale)).square() + scale + 0.5 * math.log(2 * math.pi)
    obj = masked_mean(object_nll, batch["object_mask"])
    alignment = output["semantic"].sum() * 0 if semantic_target is None else F.mse_loss(output["semantic"], semantic_target)
    losses = {"post_event": post, "next_event": nxt, "duration": duration, "success": success, "object": obj, "semantic_alignment": alignment}
    total = post + 0.5 * nxt + 0.5 * duration + success + obj + semantic_weight * alignment
    losses["total"] = total
    return total, losses


def conditional_recovery_loss(
    adapter: DetachedConditionalRecoveryAdapter,
    transition: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = adapter(transition)
    mask = batch["recovery_observed"].bool()
    if bool((mask & ~batch["regress"].bool()).any()):
        raise AdapterContractError("recovery loss mask is not conditional on regress")
    if not enabled or not bool(mask.any()):
        return logits.sum() * 0.0, logits
    loss = F.binary_cross_entropy_with_logits(
        logits[mask], batch["recovery"][mask].to(logits)
    )
    return loss, logits


class CanonicalTeacher:
    """Optional, frozen 27D->96D semantic teacher with a strict contract."""

    def __init__(self, checkpoint: Mapping[str, Any], *, event_spec_sha256: str) -> None:
        from train_multibody_canonical_event_world_model import FORMAT as TEACHER_FORMAT, ModelConfig, MultibodyCanonicalEventWorldModel
        if checkpoint.get("format") != TEACHER_FORMAT or not isinstance(checkpoint.get("config"), Mapping) or not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("contract"), Mapping):
            raise AdapterContractError("canonical teacher contract is incomplete")
        contract = checkpoint["contract"]
        body_to_id = contract.get("body_to_id")
        input_sha = contract.get("input_sha256")
        if (
            contract.get("format") != TEACHER_FORMAT
            or contract.get("event_spec_sha256") != event_spec_sha256
            or not isinstance(body_to_id, Mapping)
            or TARGET_BODY not in body_to_id
            or not isinstance(input_sha, Mapping)
            or input_sha.get("event_spec") != event_spec_sha256
        ):
            raise AdapterContractError("canonical teacher is bound to a different event spec")
        config = ModelConfig(**checkpoint["config"])
        if config.semantic_dim != 96 or config.object_delta_dim != 6 or config.action_schema_count != 3:
            raise AdapterContractError("canonical teacher dimensions changed")
        self.model = MultibodyCanonicalEventWorldModel(config)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, state27: torch.Tensor) -> torch.Tensor:
        if state27.ndim != 2 or state27.shape[1] != 27:
            raise AdapterContractError("teacher alignment requires canonical 27D state")
        return self.model.semantic(state27)


def move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def paired_group_bootstrap_interval(
    differences: Sequence[int],
    *,
    seed: int = PAIRED_BOOTSTRAP_SEED,
    samples: int = PAIRED_BOOTSTRAP_SAMPLES,
    confidence: float = PAIRED_BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Fixed, equal-logical-group bootstrap CI for paired success gain."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AdapterContractError("paired bootstrap seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1_000:
        raise AdapterContractError("paired bootstrap sample count is too small")
    if not isinstance(confidence, float) or not 0.0 < confidence < 1.0:
        raise AdapterContractError("paired bootstrap confidence is invalid")
    values = np.asarray(list(differences), dtype=np.int8)
    if values.ndim != 1 or len(values) < 1 or not np.isin(values, (-1, 0, 1)).all():
        raise AdapterContractError("paired bootstrap differences must be nonempty {-1,0,1}")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), size=(samples, len(values)))
    bootstrap_gain = values[draws].mean(axis=1, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_gain, [alpha, 1.0 - alpha])
    return {
        "method": "equal_logical_group_nonparametric_bootstrap_percentile",
        "seed": seed,
        "samples": samples,
        "confidence": confidence,
        "group_count": int(len(values)),
        "point_gain": float(values.mean(dtype=np.float64)),
        "lower_confidence_bound": float(lower),
        "upper_confidence_bound": float(upper),
        "resampling_unit": "logical_group",
        "within_group_candidate_pairs_treated_as_independent": False,
    }


def adapter_checkpoint_selection_key(
    metrics: Mapping[str, Any], step: int
) -> tuple[float, float, float, float, int]:
    """Preregistered lexicographic adapter checkpoint ordering."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise AdapterContractError("adapter checkpoint step is invalid")
    names = (
        "paired_success_gain_lcb",
        "paired_success_gain",
        "model_success_rate",
        "dense_loss",
    )
    values: dict[str, float] = {}
    for name in names:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdapterContractError(f"adapter selection metric {name} is invalid")
        values[name] = float(value)
        if not math.isfinite(values[name]):
            raise AdapterContractError(f"adapter selection metric {name} is non-finite")
    return (
        -values["paired_success_gain_lcb"],
        -values["paired_success_gain"],
        -values["model_success_rate"],
        values["dense_loss"],
        step,
    )


@torch.no_grad()
def validation_metrics(
    model: SmolVLAPiperAdapter,
    recovery_adapter: DetachedConditionalRecoveryAdapter,
    loader: DataLoader,
    paired: Sequence[Mapping[str, Any]],
    device: torch.device,
    object_mean: torch.Tensor,
    object_std: torch.Tensor,
    *,
    recovery_enabled: bool,
    pairwise_weight: float,
    listwise_weight: float,
) -> dict[str, Any]:
    model.eval()
    totals: list[float] = []
    post_group_accuracy: list[float] = []
    next_group_accuracy: list[float] = []
    rows = next_observed_rows = 0
    recovery_probabilities: list[np.ndarray] = []
    recovery_targets: list[np.ndarray] = []
    for raw in loader:
        batch = move(raw, device)
        logical_ids = batch.get("logical_group_id")
        if (
            not isinstance(logical_ids, list)
            or not logical_ids
            or len(set(logical_ids)) != 1
        ):
            raise AdapterContractError(
                "validation dense batches must contain exactly one logical group"
            )
        output = model(batch)
        loss, _ = compute_loss(output, batch, object_mean=object_mean, object_std=object_std)
        value = float(loss)
        if not math.isfinite(value):
            raise AdapterContractError("validation dense loss is non-finite")
        totals.append(value)
        rows += len(batch["success"])
        post_group_accuracy.append(
            float(
                (output["next_event_logits"].argmax(1) == batch["post_event_id"])
                .float()
                .mean()
            )
        )
        observed = batch["duration_observed"]
        next_observed_rows += int(observed.sum())
        if bool(observed.any()):
            next_group_accuracy.append(
                float(
                    (
                        output["next_reached_event_logits"].argmax(1)[observed]
                        == batch["next_event_id"][observed]
                    )
                    .float()
                    .mean()
                )
            )
        if recovery_enabled:
            recovery_logits = recovery_adapter(output["transition"])
            recovery_mask = batch["recovery_observed"]
            if bool(recovery_mask.any()):
                recovery_probabilities.append(
                    torch.sigmoid(recovery_logits[recovery_mask]).cpu().numpy()
                )
                recovery_targets.append(
                    batch["recovery"][recovery_mask].cpu().numpy()
                )
    if rows < 1 or next_observed_rows < 1 or not post_group_accuracy or not next_group_accuracy:
        raise AdapterContractError(
            "validation requires transition rows and observed next-event targets"
        )
    ranking_batch = move(collate_ranking_groups(paired), device)
    ranking_output = model.predict_grouped_candidates(ranking_batch)
    source_rank_scores = ranking_output["source_contract_rank_score"].detach().cpu()
    source_base_rank_scores = ranking_output[
        "source_contract_base_rank_score"
    ].detach().cpu()
    base_logits = ranking_output["base_success_logit"].detach().cpu()
    residuals = ranking_output["action_rank_residual"].detach().cpu()
    flat_group_index = ranking_batch["ranking_group_index"].detach().cpu()
    baseline_success = model_success = changed = 0
    paired_differences: list[int] = []
    details = []
    for group_index, group in enumerate(paired):
        positions = torch.nonzero(
            flat_group_index == group_index, as_tuple=False
        ).squeeze(-1).tolist()
        if len(positions) != len(group["candidates"]):
            raise AdapterContractError("validation grouped candidate alignment changed")
        scored = []
        candidate_scores = []
        for position, candidate in zip(positions, group["candidates"]):
            source_rank_score = float(source_rank_scores[position])
            source_base_rank_score = float(source_base_rank_scores[position])
            base_logit = float(base_logits[position])
            residual = float(residuals[position])
            score = source_rank_score
            if not all(
                math.isfinite(value)
                for value in (
                    source_rank_score,
                    source_base_rank_score,
                    base_logit,
                    residual,
                )
            ):
                raise AdapterContractError("validation candidate score is non-finite")
            scored.append(
                (score, -candidate["original_candidate_index"], candidate)
            )
            candidate_scores.append(
                {
                    "original_candidate_index": candidate["original_candidate_index"],
                    "is_baseline": bool(candidate["is_baseline"]),
                    "base_success_logit": base_logit,
                    "action_rank_residual": residual,
                    "source_contract_base_rank_score": source_base_rank_score,
                    "source_contract_rank_score": source_rank_score,
                    "source_contract_rank_score_is_success_logit": False,
                    "source_contract_rank_score_is_success_probability": False,
                }
            )
        selected = max(scored)[2]
        baseline = next(item for item in group["candidates"] if item["is_baseline"])
        baseline_success += baseline["final_success"]
        model_success += selected["final_success"]
        changed += selected["original_candidate_index"] != baseline["original_candidate_index"]
        paired_differences.append(
            int(selected["final_success"] - baseline["final_success"])
        )
        details.append(
            {
                "logical_group_id": group["logical_group_id"],
                "baseline_original_candidate_index": baseline["original_candidate_index"],
                "selected_original_candidate_index": selected["original_candidate_index"],
                "baseline_success": baseline["final_success"],
                "selected_success": selected["final_success"],
                "paired_success_difference": paired_differences[-1],
                "candidate_scores": candidate_scores,
            }
        )
    groups = len(paired)
    pairwise, listwise, ranking_audit = group_weighted_ranking_loss(
        ranking_output["source_contract_rank_score"], ranking_batch
    )
    paired_interval = paired_group_bootstrap_interval(paired_differences)
    dense_loss = float(np.mean(totals))
    ranking_loss = pairwise_weight * float(pairwise) + listwise_weight * float(listwise)
    if recovery_enabled and recovery_targets:
        recovery_probability = np.concatenate(recovery_probabilities).astype(np.float64)
        recovery_target = np.concatenate(recovery_targets).astype(np.float64)
        clipped = np.clip(recovery_probability, 1e-12, 1 - 1e-12)
        recovery_metrics: dict[str, Any] = {
            "status": "descriptive_uncalibrated_validation_available",
            "conditional_on": "operational_regress_persistent_three_states",
            "observed_rows": int(len(recovery_target)),
            "positive_rows": int(recovery_target.sum()),
            "negative_rows": int(len(recovery_target) - recovery_target.sum()),
            "binary_nll": float(
                -np.mean(
                    recovery_target * np.log(clipped)
                    + (1 - recovery_target) * np.log(1 - clipped)
                )
            ),
            "brier": float(np.square(recovery_probability - recovery_target).mean()),
            "accuracy_at_0_5": float(
                ((recovery_probability >= 0.5) == recovery_target).mean()
            ),
            "calibrated": False,
            "enters_primary_utility_or_uncertainty": False,
        }
    else:
        recovery_metrics = {
            "status": (
                "disabled_training_group_support"
                if not recovery_enabled
                else "disabled_no_observed_validation_regress_rows"
            ),
            "observed_rows": 0,
            "calibrated": False,
            "enters_primary_utility_or_uncertainty": False,
        }
    return {
        "loss": dense_loss + ranking_loss,
        "dense_loss": dense_loss,
        "pairwise_ranking_loss": float(pairwise),
        "listwise_ranking_loss": float(listwise),
        "ranking_audit": ranking_audit,
        "rows": rows, "post_event_accuracy": float(np.mean(post_group_accuracy)),
        "next_event_accuracy": float(np.mean(next_group_accuracy)),
        "dense_metric_aggregation": "equal_logical_group",
        "dense_validation_logical_groups": len(post_group_accuracy),
        "next_event_validation_logical_groups_with_observations": len(next_group_accuracy),
        "next_event_observed_rows": next_observed_rows,
        "next_event_censored_rows_excluded": rows - next_observed_rows,
        "next_event_metric_mask": "duration_observed_only",
        "conditional_recovery": recovery_metrics,
        "paired_root_groups": groups,
        "baseline_success_rate": baseline_success / groups, "model_success_rate": model_success / groups,
        "paired_success_gain": (model_success - baseline_success) / groups,
        "paired_success_gain_lcb": paired_interval["lower_confidence_bound"],
        "paired_success_gain_ci": paired_interval,
        "changed_groups": int(changed), "paired_details": details,
        "candidate_score_contract": "source_candidate_rank_score_plus_action_rank_residual",
        "candidate_score_is_success_logit": False,
        "candidate_score_is_success_probability": False,
    }


@torch.no_grad()
def evaluate_conditional_recovery_adapter(
    *,
    model: SmolVLAPiperAdapter,
    recovery_adapter: DetachedConditionalRecoveryAdapter,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    recovery_adapter.eval()
    group_nll: list[float] = []
    group_brier: list[float] = []
    group_accuracy: list[float] = []
    observed_rows = positive_rows = 0
    groups: list[str] = []
    for raw in loader:
        batch = move(raw, device)
        logical_ids = batch.get("logical_group_id")
        if (
            not isinstance(logical_ids, list)
            or not logical_ids
            or len(set(logical_ids)) != 1
        ):
            raise AdapterContractError(
                "conditional recovery validation must batch one logical group"
            )
        prediction = model(batch)
        logits = recovery_adapter(prediction["transition"])
        mask = batch["recovery_observed"]
        if bool(mask.any()):
            probability = torch.sigmoid(logits[mask]).cpu().numpy().astype(np.float64)
            target = batch["recovery"][mask].cpu().numpy().astype(np.float64)
            clipped = np.clip(probability, 1e-12, 1 - 1e-12)
            group_nll.append(
                float(
                    -np.mean(
                        target * np.log(clipped)
                        + (1 - target) * np.log(1 - clipped)
                    )
                )
            )
            group_brier.append(float(np.square(probability - target).mean()))
            group_accuracy.append(
                float(((probability >= 0.5) == target).mean())
            )
            observed_rows += len(target)
            positive_rows += int(target.sum())
            groups.append(logical_ids[0])
    if not group_nll:
        return {
            "status": "disabled_no_observed_validation_regress_rows",
            "observed_rows": 0,
            "enters_primary_utility_or_uncertainty": False,
        }
    return {
        "status": "available_uncalibrated_conditional_recovery",
        "conditional_on": "operational_regress_persistent_three_states",
        "observed_rows": observed_rows,
        "positive_rows": positive_rows,
        "negative_rows": observed_rows - positive_rows,
        "independent_groups": len(set(groups)),
        "binary_nll": float(np.mean(group_nll)),
        "brier": float(np.mean(group_brier)),
        "accuracy_at_0_5": float(np.mean(group_accuracy)),
        "metric_aggregation": "equal_logical_group_with_observed_recovery",
        "calibrated": False,
        "enters_primary_utility_or_uncertainty": False,
    }


def fit_detached_conditional_recovery_adapter(
    *,
    model: SmolVLAPiperAdapter,
    recovery_adapter: DetachedConditionalRecoveryAdapter,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    training_support: Mapping[str, Any],
    validation_support: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    steps: int,
    eval_every: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Fit recovery only after the selected embodiment adapter is frozen."""

    authorized = bool(training_support["enabled"] and validation_support["enabled"])
    if not authorized:
        with torch.no_grad():
            recovery_adapter.head.weight.zero_()
            recovery_adapter.head.bias.zero_()
        return {
            "status": "disabled_insufficient_independent_group_support",
            "trained": False,
            "training_support": dict(training_support),
            "validation_support": dict(validation_support),
            "best_step": None,
            "validation_metrics": {
                "status": "disabled_independent_group_support",
                "enters_primary_utility_or_uncertainty": False,
            },
        }
    observed_train_rows = [
        row for row in train_rows if bool(row["recovery_observed"])
    ]
    observed_validation_rows = [
        row for row in validation_rows if bool(row["recovery_observed"])
    ]
    grouped_observed = _logical_group_row_indices(observed_train_rows)
    prevalence = float(
        np.mean(
            [
                np.mean(
                    [float(observed_train_rows[index]["recovery"]) for index in indices]
                )
                for indices in grouped_observed.values()
            ]
        )
    )
    recovery_adapter.initialize_prior(prevalence)
    optimizer = torch.optim.AdamW(
        recovery_adapter.parameters(), lr=learning_rate, weight_decay=0
    )
    recovery_sampler = LogicalGroupEqualizedSampler(
        observed_train_rows, seed=seed
    )
    train_loader = DataLoader(
        RowDataset(observed_train_rows), batch_size=batch_size, shuffle=False,
        sampler=recovery_sampler, collate_fn=collate,
    )
    validation_loader = DataLoader(
        RowDataset(observed_validation_rows),
        batch_sampler=LogicalGroupBatchSampler(observed_validation_rows),
        collate_fn=collate,
    )
    iterator = iter(train_loader)
    best_key: tuple[float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    model.eval()
    for step in range(1, steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        batch = move(raw, device)
        with torch.no_grad():
            transition = model(batch)["transition"]
        recovery_adapter.train()
        loss, _ = conditional_recovery_loss(
            recovery_adapter, transition, batch, enabled=True
        )
        if not torch.isfinite(loss):
            raise AdapterContractError("non-finite detached recovery training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(recovery_adapter.parameters(), 2.0)
        optimizer.step()
        if step % eval_every and step != steps:
            continue
        metrics = evaluate_conditional_recovery_adapter(
            model=model, recovery_adapter=recovery_adapter,
            loader=validation_loader, device=device,
        )
        if metrics.get("status") != "available_uncalibrated_conditional_recovery":
            raise AdapterContractError(
                "enabled recovery head lacks validation observations"
            )
        key = (float(metrics["binary_nll"]), step)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in recovery_adapter.state_dict().items()
            }
    if best_state is None or best_metrics is None or best_key is None:
        raise AdapterContractError("no detached recovery checkpoint was selected")
    recovery_adapter.load_state_dict(best_state, strict=True)
    return {
        "status": "complete_detached_conditional_recovery_training",
        "trained": True,
        "training_support": dict(training_support),
        "validation_support": dict(validation_support),
        "best_step": int(best_key[1]),
        "validation_selection_rule": "minimum_conditional_binary_nll_then_earliest_step",
        "validation_metrics": best_metrics,
        "training_sampling_contract": recovery_sampler.audit(),
        "training_prior": "mean_of_within_group_recovery_prevalence",
        "shared_transition_stop_gradient": True,
        "separate_optimizer": True,
    }


def atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_replace(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def atomic_npz_new(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Create one non-pickle NPZ without filename-suffix surprises."""

    if path.exists():
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def array_bundle_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Logical array digest independent of NPZ ZIP timestamps/metadata."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        if array.dtype.hasobject:
            raise AdapterContractError("validation artifact contains object arrays")
        header = json.dumps(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def export_internal_validation_artifacts(
    *,
    split_profile: ExternalSplitProfile,
    model: SmolVLAPiperAdapter,
    recovery_adapter: DetachedConditionalRecoveryAdapter,
    recovery_fit: Mapping[str, Any],
    recovery_training_support: Mapping[str, Any],
    recovery_validation_support: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    object_mean: torch.Tensor,
    object_std: torch.Tensor,
    output: Path,
) -> dict[str, Any]:
    """Export only the already-opened adaptation-derived internal validation.

    The profile's external 50/190-group formal target-validation partition
    remains the trainer's sealed test set and is never passed to this function.
    """

    if not rows:
        raise AdapterContractError("internal validation export has no rows")
    occurrence: dict[str, int] = {}
    sample_ids: list[str] = []
    group_ids: list[str] = []
    for row in rows:
        group = str(row["logical_group_id"])
        ordinal = occurrence.get(group, 0)
        occurrence[group] = ordinal + 1
        sample_ids.append(
            "internal-validation-"
            + canonical_sha256({"logical_group_id": group, "row_ordinal": ordinal})
        )
        group_ids.append(group)
    if len(set(sample_ids)) != len(sample_ids):
        raise AdapterContractError("internal validation sample identities collide")
    if len(occurrence) != split_profile.internal_validation_groups:
        raise AdapterContractError(
            f"{split_profile.name} internal validation artifact group count changed"
        )

    labels: dict[str, np.ndarray] = {
        "sample_id": np.asarray(sample_ids),
        "group_id": np.asarray(group_ids),
        "post_event": np.asarray([row["post_event_id"] for row in rows], dtype=np.int64),
        "next_event": np.asarray([row["next_event_id"] for row in rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in rows], dtype=np.int64),
        "regress": np.asarray([row["regress"] for row in rows], dtype=bool),
        "recovery": np.asarray([row["recovery"] for row in rows], dtype=np.int64),
        "recovery_observed": np.asarray(
            [row["recovery_observed"] for row in rows], dtype=bool
        ),
        "duration": np.asarray([row["duration"] for row in rows], dtype=np.float64),
        "duration_observed": np.asarray(
            [row["duration_observed"] for row in rows], dtype=bool
        ),
        "object_target": np.stack(
            [np.asarray(row["object_delta"], dtype=np.float64) for row in rows]
        ),
        "object_observed": np.asarray(
            [bool(np.asarray(row["object_mask"], dtype=bool).all()) for row in rows],
            dtype=bool,
        ),
    }
    loader = DataLoader(
        RowDataset(rows), batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "post_event_logits",
            "next_event_logits",
            "success_logit",
            "recovery_logit",
            "duration_log_mean",
            "duration_log_scale",
            "object_mean",
            "object_log_scale",
        )
    }
    model.eval()
    recovery_adapter.eval()
    raw_object_mean = object_mean.detach().to(device)
    raw_object_std = object_std.detach().to(device)
    normalization_arrays = {
        "object_delta_mean": raw_object_mean.detach().cpu().numpy(),
        "object_delta_std": raw_object_std.detach().cpu().numpy(),
    }
    for raw in loader:
        prediction = model(move(raw, device))
        values = {
            "post_event_logits": prediction["next_event_logits"],
            "next_event_logits": prediction["next_reached_event_logits"],
            "success_logit": prediction["success_logit"],
            "recovery_logit": recovery_adapter(prediction["transition"]),
            # The trainer models log(1 + duration), not log(duration).
            "duration_log_mean": prediction["duration_selected_log_mean"],
            "duration_log_scale": prediction["duration_selected_log_scale"],
            # The frozen source head is normalized; calibration operates in
            # physical object-delta units and therefore needs de-normalization.
            "object_mean": prediction["object_delta_mean"] * raw_object_std
            + raw_object_mean,
            "object_log_scale": prediction["object_delta_log_scale"]
            + raw_object_std.log(),
        }
        for name, value in values.items():
            collected[name].append(value.detach().cpu().numpy())
    predictions: dict[str, np.ndarray] = {
        "sample_id": np.asarray(sample_ids),
        **{name: np.concatenate(parts, axis=0) for name, parts in collected.items()},
    }
    if any(len(array) != len(rows) for array in labels.values()) or any(
        len(array) != len(rows) for array in predictions.values()
    ):
        raise AdapterContractError("internal validation artifact lengths changed")
    label_path = output / "internal_validation_labels.npz"
    prediction_path = output / "internal_validation_predictions.npz"
    atomic_npz_new(label_path, labels)
    atomic_npz_new(prediction_path, predictions)
    receipt = {
        "lane": "adaptation_derived_internal_validation_only",
        "split_profile": split_profile.name,
        "split_profile_version": split_profile.version,
        "required_trainer_group_counts": (
            split_profile.required_trainer_group_counts
        ),
        "validation_group_count": len(occurrence),
        "required_internal_validation_group_count": (
            split_profile.internal_validation_groups
        ),
        "sealed_formal_target_validation_group_count": (
            split_profile.sealed_test_groups
        ),
        "validation_sample_count": len(rows),
        "validation_identity_set_sha256": canonical_sha256(sample_ids),
        "causal_history_contract": causal_history_contract(),
        "causal_history_application_contract": (
            schema6_causal_history_application_contract()
        ),
        "labels_path": str(label_path),
        "labels_file_sha256": file_sha256(label_path),
        "labels_logical_sha256": array_bundle_sha256(labels),
        "predictions_path": str(prediction_path),
        "predictions_file_sha256": file_sha256(prediction_path),
        "predictions_logical_sha256": array_bundle_sha256(predictions),
        "duration_target_transform": "log1p_decision_steps",
        "next_event_observation_mask": "duration_observed",
        "success_target": "eventual_final_branch_success_repeated_per_transition",
        "recovery_target": "conditional_recovery_given_operational_regress",
        "recovery_observation_mask": "recovery_observed_and_regress",
        "recovery_persistence_states": RECOVERY_PERSISTENCE_STATES,
        "recovery_training_support": dict(recovery_training_support),
        "recovery_validation_support": dict(recovery_validation_support),
        "recovery_head_trained": bool(recovery_fit["trained"]),
        "recovery_fit_status": str(recovery_fit["status"]),
        "recovery_internal_validation_metrics": dict(
            recovery_fit["validation_metrics"]
        ),
        "recovery_enters_primary_utility_or_uncertainty": False,
        "recovery_calibration_required_before_activation": True,
        "recovery_shared_transition_stop_gradient": True,
        "object_prediction_space": "physical_delta_xyz_m",
        "object_source_normalization_sha256": array_bundle_sha256(
            normalization_arrays
        ),
        "object_observed_policy": "row_enabled_only_if_all_selected_xyz_are_valid",
        "sealed_formal_target_validation_hdf5_files_opened": 0,
        "sealed_test_labels_opened": 0,
        "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_labels_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_release_condition": (
            "external_authority_after_all_five_adapter_checkpoints_are_frozen"
        ),
    }
    # Preserve the historical v2 post-collection verifier without describing
    # the 190-group v3 lane as though it contained 50 groups.
    if split_profile is HISTORICAL_V2_PROFILE:
        receipt["target_validation50_hdf5_files_opened"] = 0
    return receipt


def train(args: argparse.Namespace) -> dict[str, Any]:
    provider_variant = getattr(
        args, "provider_variant", DEFAULT_PROVIDER_VARIANT
    )
    artifact_format = provider_artifact_format(provider_variant)
    determinism = configure_determinism(args.training_seed)
    gate = SupportGate(
        min_train_groups=args.min_train_groups,
        min_validation_groups=args.min_validation_groups,
        min_test_groups=args.min_test_groups,
        min_outcome_groups=args.min_outcome_groups,
        min_discordant_groups=args.min_discordant_groups,
        min_event_rows=args.min_event_rows,
        min_duration_rows=args.min_duration_rows,
        min_object_rows=args.min_object_rows,
        min_candidate_index_groups=args.min_candidate_index_groups,
    )
    gate.require_formal_minimums()
    determinism["dense_dataloader_seed"] = args.training_seed
    determinism["ranking_dataloader_seed"] = args.training_seed + 1
    determinism["conditional_recovery_dataloader_seed"] = args.training_seed + 2
    source_path = reject_sensitive_path(args.source_checkpoint, "source checkpoint")
    manifest_path = reject_sensitive_path(args.schema6_manifest, "schema6 manifest")
    expected_receipt_path = reject_sensitive_path(
        args.expected_manifest_split_receipt, "expected manifest/split receipt"
    )
    event_spec_path = reject_sensitive_path(args.canonical_event_spec, "canonical event spec")
    output = reject_sensitive_path(args.output, "output", file=False)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = _load_torch(source_path, "source checkpoint")
    source_audit = validate_source_checkpoint(checkpoint)
    config = EventWorldModelConfig.from_dict(source_audit["config"])
    source_rank_config_audit = validate_production_source_rank_config(config)
    frozen_source_rank_contract = source_rank_score_contract(
        checkpoint,
        config,
        source_checkpoint_file_sha256=file_sha256(source_path),
    )
    manifest, descriptors = scan_manifest(manifest_path)
    split, external_split_audit = validate_external_split_authority(
        expected_receipt_path=expected_receipt_path,
        expected_receipt_file_sha256=args.expected_manifest_split_receipt_file_sha256,
        manifest_path=manifest_path,
        manifest=manifest,
        descriptors=descriptors,
    )
    split_profile = SPLIT_PROFILES_BY_NAME[external_split_audit["split_profile"]]
    event_spec_sha = file_sha256(event_spec_path)
    if manifest.get("event_spec_sha256") != event_spec_sha:
        raise AdapterContractError("schema6 manifest is not bound to canonical event spec")
    event_spec = _load_json(event_spec_path, "canonical event spec")
    calibration_by_task = event_spec.get("calibration")
    if not isinstance(calibration_by_task, Mapping) or any(
        descriptor.task not in calibration_by_task
        or not isinstance(calibration_by_task[descriptor.task], Mapping)
        for descriptor in descriptors
    ):
        raise AdapterContractError("canonical event spec lacks a schema6 task calibration")
    teacher: CanonicalTeacher | None = None
    teacher_receipt: dict[str, Any] = {"enabled": False}
    if args.canonical_teacher_checkpoint is not None:
        teacher_path = reject_sensitive_path(args.canonical_teacher_checkpoint, "canonical teacher")
        teacher_checkpoint = _load_torch(teacher_path, "canonical teacher")
        teacher = CanonicalTeacher(teacher_checkpoint, event_spec_sha256=event_spec_sha)
        teacher_receipt = {
            "enabled": True,
            "checkpoint_sha256": file_sha256(teacher_path),
            "event_spec_sha256": event_spec_sha,
            "teacher_frozen": True,
            "state_contract": "same_object_pose_canonical_27D_to_96D",
        }
    group_counts = validate_group_count_gate(
        split, gate, descriptors, exact_profile=split_profile
    )
    output.mkdir(parents=False)
    split_receipt = {
        **split,
        "schema6_manifest_file_sha256": file_sha256(manifest_path),
        "schema6_manifest_logical_sha256": manifest["manifest_sha256"],
        "canonical_event_spec_file_sha256": event_spec_sha,
        "external_split_authority": external_split_audit,
        "split_profile": split_profile.name,
        "split_profile_version": split_profile.version,
        "required_trainer_group_counts": (
            split_profile.required_trainer_group_counts
        ),
        "support_gate": gate.to_dict(),
        "group_counts": group_counts,
    }
    split_receipt["receipt_sha256"] = canonical_sha256(split_receipt)
    atomic_json_new(output / "frozen_group_split.json", split_receipt)
    # The split receipt now exists durably.  Only train/validation labels may open.
    (
        train_rows,
        validation_rows,
        train_pairs,
        validation_pairs,
    ) = read_train_and_internal_validation_groups(
        split=split,
        descriptors=descriptors,
        calibration_by_task=calibration_by_task,
        object_delta_dim=config.object_delta_dim,
        object_names=source_audit["object_names"],
        include_canonical_state=teacher is not None,
    )
    train_support = split_supervision_support(train_rows, train_pairs)
    validation_support = split_supervision_support(validation_rows, validation_pairs)
    validate_supervision_support(train_support, gate, split_name="train")
    validate_supervision_support(validation_support, gate, split_name="validation")
    object_mean, object_std = object_normalization(checkpoint, config.object_delta_dim)
    device = torch.device(args.device)
    core = ActionConditionedEventWorldModel(config)
    core.load_state_dict(checkpoint["model"], strict=True)
    model = SmolVLAPiperAdapter(
        core,
        state_rank=args.state_rank,
        action_rank=args.action_rank,
        source_rank_contract=frozen_source_rank_contract,
        provider_variant=provider_variant,
    ).to(device)
    recovery_training_support = train_support["conditional_recovery"]
    recovery_validation_support = validation_support["conditional_recovery"]
    recovery_adapter = DetachedConditionalRecoveryAdapter(
        # The public transition representation is projected to semantic_dim;
        # transition_hidden_dim is only the private width inside the core MLP.
        config.semantic_dim
    ).to(device)
    with torch.no_grad():
        recovery_adapter.head.weight.zero_()
        recovery_adapter.head.bias.zero_()
    if teacher is not None:
        teacher.model.to(device)
    if model._source_sha != source_audit["source_core_state_sha256"]:
        raise AdapterContractError("loaded core differs from source proof")
    trainable = model.trainable_parameter_audit()
    recovery_parameter_audit = recovery_adapter.parameter_audit()
    adapter_config = {
        "state_input_dim": STATE_DIM,
        "state_adapter": "identity_plus_zero_initialized_low_rank_residual",
        "state_rank": args.state_rank,
        "action_dim": ACTION_DIM,
        "action_adapter": "identity_diagonal_plus_zero_initialized_low_rank_residual",
        "action_rank": args.action_rank,
        "target_body_row": 1,
        "policy_row": 0,
        "reserved_openvla_policy_row": 1,
        "source_action_rank_residual_consumed": True,
        "source_action_rank_success_only": False,
        "source_rank_config_audit": source_rank_config_audit,
        "deployment_success_logit": "base_factual_success_logit",
        "dense_success_uses_base_logit": True,
        "deployment_primary_candidate_score": "source_contract_rank_score",
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "source_rank_score_contract_sha256": frozen_source_rank_contract[
            "contract_sha256"
        ],
        "causal_history_contract": source_audit["causal_history_contract"],
        "causal_history_application_contract": (
            schema6_causal_history_application_contract()
        ),
    }
    if provider_variant == "body_agnostic_adapter":
        adapter_config.update(
            {
                "provider_variant": "body_agnostic_adapter",
                "prediction_body_row": 0,
                "reserved_target_body_row": 1,
                "shared_source_body_row_0_used": True,
                "target_body_row_1_trainable": False,
                "clock_beta_fixed_exact_zero": True,
                "clock_log_step_scale_fixed_exact_zero": True,
            }
        )
    supervision_support = {
        "split_profile": split_profile.name,
        "required_trainer_group_counts": (
            split_profile.required_trainer_group_counts
        ),
        "gate": gate.to_dict(),
        "group_counts": group_counts,
        "train": train_support,
        "validation": validation_support,
        "sealed_test_group_count": split_profile.sealed_test_groups,
        "sealed_test_hdf5_files_opened": 0,
        "sealed_test_labels_opened": 0,
    }
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate, weight_decay=0)
    ranking_generator = torch.Generator().manual_seed(args.training_seed + 1)
    dense_sampler = LogicalGroupEqualizedSampler(
        train_rows, seed=args.training_seed
    )
    dense_sampling_audit = dense_sampler.audit()
    train_loader = DataLoader(
        RowDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=False,
        sampler=dense_sampler,
        collate_fn=collate,
    )
    ranking_loader = DataLoader(
        PairedGroupDataset(train_pairs),
        batch_size=min(args.ranking_batch_groups, len(train_pairs)),
        shuffle=True,
        generator=ranking_generator,
        collate_fn=collate_ranking_groups,
    )
    val_loader = DataLoader(
        RowDataset(validation_rows),
        batch_sampler=LogicalGroupBatchSampler(validation_rows),
        collate_fn=collate,
    )
    iterator = iter(train_loader)
    ranking_iterator = iter(ranking_loader)
    object_mean = object_mean.to(device); object_std = object_std.to(device)
    best_key = None; best_metrics = None; best_step = 0
    best_path = output / "best.pt"
    for step in range(1, args.steps + 1):
        try: raw = next(iterator)
        except StopIteration: iterator = iter(train_loader); raw = next(iterator)
        try: raw_ranking = next(ranking_iterator)
        except StopIteration:
            ranking_iterator = iter(ranking_loader)
            raw_ranking = next(ranking_iterator)
        batch = move(raw, device)
        ranking_batch = move(raw_ranking, device)
        model.train(); prediction = model(batch)
        semantic_target = None if teacher is None else teacher.encode(batch["canonical_state27"])
        dense_loss, _ = compute_loss(
            prediction, batch, object_mean=object_mean, object_std=object_std,
            semantic_target=semantic_target, semantic_weight=args.semantic_alignment_weight,
        )
        ranking_prediction = model.predict_grouped_candidates(ranking_batch)
        pairwise_loss, listwise_loss, _ranking_audit = group_weighted_ranking_loss(
            ranking_prediction["source_contract_rank_score"], ranking_batch
        )
        loss = (
            dense_loss
            + args.pairwise_ranking_weight * pairwise_loss
            + args.listwise_ranking_weight * listwise_loss
        )
        if not torch.isfinite(loss): raise AdapterContractError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0); optimizer.step()
        immutable = model.enforce_and_verify_frozen_core()
        if step % args.eval_every and step != args.steps: continue
        metrics = validation_metrics(
            model, recovery_adapter, val_loader, validation_pairs, device,
            object_mean, object_std,
            recovery_enabled=False,
            pairwise_weight=args.pairwise_ranking_weight,
            listwise_weight=args.listwise_ranking_weight,
        )
        # Preregistered equal-logical-group uncertainty-aware selection. Candidate
        # pairs within a group are never treated as independent observations.
        key = adapter_checkpoint_selection_key(metrics, step)
        if best_key is None or key < best_key:
            best_key, best_metrics, best_step = key, metrics, step
            checkpoint_payload = {
                "format": artifact_format, "model": model.state_dict(), "source_checkpoint_sha256": file_sha256(source_path),
                "source_audit": source_audit, "split_receipt": split_receipt, "trainable_parameter_audit": trainable,
                "adapter_config": adapter_config, "supervision_support": supervision_support,
                "immutable_core_audit": immutable, "best_step": step, "validation": metrics,
                "canonical_teacher": teacher_receipt,
                "determinism": determinism,
                "dense_sampling_contract": dense_sampling_audit,
                "source_rank_score_contract": frozen_source_rank_contract,
                "causal_history_contract": source_audit[
                    "causal_history_contract"
                ],
                "causal_history_application_contract": (
                    schema6_causal_history_application_contract()
                ),
                "ranking_contract": {
                    "pairwise_weight": args.pairwise_ranking_weight,
                    "listwise_weight": args.listwise_ranking_weight,
                    "group_weighting": "equal_per_logical_group",
                    "pairwise_anchor": "lowest_legal_feasibility_baseline",
                    "target": "final_branch_success_from_root_intervention",
                    "source_action_rank_residual_consumed": True,
                    "source_action_rank_success_only": False,
                    "deployment_success_logit": "base_factual_success_logit",
                    "dense_success_uses_base_logit": True,
                    "deployment_primary_candidate_score": "source_contract_rank_score",
                    "source_contract_rank_score_is_success_logit": False,
                    "source_contract_rank_score_is_success_probability": False,
                    "deployment_success_probability_selector_authorized": False,
                    "candidate_prediction_api": "predict_grouped_candidates",
                    "checkpoint_selection_primary": "paired_success_gain_lcb",
                    "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
                    "paired_bootstrap_samples": PAIRED_BOOTSTRAP_SAMPLES,
                    "paired_bootstrap_resampling_unit": "logical_group",
                },
                "validation_selection_rule": "maximize equal_group_bootstrap_paired_success_gain_lcb; maximize paired_success_gain; maximize model_success_rate; minimize dense_validation_loss; earliest_step",
                "test_hdf5_files_opened": 0,
            }
            if provider_variant == "body_agnostic_adapter":
                checkpoint_payload["provider_variant"] = (
                    "body_agnostic_adapter"
                )
            atomic_torch_replace(best_path, checkpoint_payload)
    if best_metrics is None:
        raise AdapterContractError("no validation checkpoint was selected")
    try:
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        best_payload = torch.load(best_path, map_location=device)
    if (
        not isinstance(best_payload, Mapping)
        or best_payload.get("format") != artifact_format
        or not isinstance(best_payload.get("model"), Mapping)
    ):
        raise AdapterContractError("best adapter checkpoint cannot be reloaded")
    model.load_state_dict(best_payload["model"], strict=True)
    immutable = model.enforce_and_verify_frozen_core()
    if not immutable["all_core_tensors_except_piper_body_row_bit_exact"]:
        raise AdapterContractError("best adapter reload changed the frozen core")
    recovery_fit = fit_detached_conditional_recovery_adapter(
        model=model,
        recovery_adapter=recovery_adapter,
        train_rows=train_rows,
        validation_rows=validation_rows,
        training_support=recovery_training_support,
        validation_support=recovery_validation_support,
        device=device,
        batch_size=args.batch_size,
        steps=args.recovery_steps,
        eval_every=args.recovery_eval_every,
        learning_rate=args.recovery_learning_rate,
        seed=args.training_seed + 2,
    )
    recovery_contract = {
        "semantics": "p(recovery_given_operational_regress)",
        "operational_regress": "event_below_old_peak_persists_three_saved_states",
        "operational_recovery": "return_to_old_peak_persists_three_saved_states_or_terminal_eK",
        "right_censored_nonrecovery_supervised": False,
        "source_core_recovery_supervised": bool(config.recovery_supervised),
        "source_core_recovery_assumed": False,
        "shared_transition_stop_gradient": True,
        "separate_optimizer": True,
        "parameter_audit": recovery_parameter_audit,
        "fit": recovery_fit,
        "trained": bool(recovery_fit["trained"]),
        "enters_primary_utility_or_uncertainty_before_calibration": False,
    }
    best_metrics = {
        **dict(best_metrics),
        "conditional_recovery": dict(recovery_fit["validation_metrics"]),
    }
    best_payload = {
        **dict(best_payload),
        "validation": best_metrics,
        "conditional_recovery_adapter": recovery_adapter.state_dict(),
        "conditional_recovery_contract": recovery_contract,
    }
    atomic_torch_replace(best_path, best_payload)
    validation_artifacts = export_internal_validation_artifacts(
        split_profile=split_profile,
        model=model,
        recovery_adapter=recovery_adapter,
        recovery_fit=recovery_fit,
        recovery_training_support=recovery_training_support,
        recovery_validation_support=recovery_validation_support,
        rows=validation_rows,
        device=device,
        batch_size=args.batch_size,
        object_mean=object_mean,
        object_std=object_std,
        output=output,
    )
    summary = {
        "format": artifact_format, "status": "complete", "best_checkpoint": str(best_path), "best_checkpoint_sha256": file_sha256(best_path),
        "best_step": best_step, "best_validation": best_metrics, "train_groups": len(split["train"]), "validation_groups": len(split["validation"]),
        "split_profile": split_profile.name,
        "split_profile_version": split_profile.version,
        "required_trainer_group_counts": split_profile.required_trainer_group_counts,
        "sealed_test_groups": len(split["test"]), "test_hdf5_files_opened": 0, "shared_core_heads_frozen": True,
        "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_labels_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_release_condition": "external_authority_after_all_five_adapter_checkpoints_are_frozen",
        "smolvla_policy_row_0_frozen": True, "reserved_openvla_policy_row_1_frozen": True,
        "canonical_teacher": teacher_receipt,
        "canonical_event_spec_file_sha256": event_spec_sha,
        "determinism": determinism,
        "dense_sampling_contract": dense_sampling_audit,
        "source_rank_score_contract": frozen_source_rank_contract,
        "causal_history_contract": source_audit["causal_history_contract"],
        "causal_history_application_contract": (
            schema6_causal_history_application_contract()
        ),
        "ranking_contract": {
            "pairwise_weight": args.pairwise_ranking_weight,
            "listwise_weight": args.listwise_ranking_weight,
            "group_weighting": "equal_per_logical_group",
            "pairwise_anchor": "lowest_legal_feasibility_baseline",
            "target": "final_branch_success_from_root_intervention",
            "source_action_rank_residual_consumed": True,
            "source_action_rank_success_only": False,
            "deployment_success_logit": "base_factual_success_logit",
            "dense_success_uses_base_logit": True,
            "deployment_primary_candidate_score": "source_contract_rank_score",
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "deployment_success_probability_selector_authorized": False,
            "candidate_prediction_api": "predict_grouped_candidates",
            "checkpoint_selection_primary": "paired_success_gain_lcb",
            "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            "paired_bootstrap_samples": PAIRED_BOOTSTRAP_SAMPLES,
            "paired_bootstrap_resampling_unit": "logical_group",
        },
        "adapter_config": adapter_config, "supervision_support": supervision_support,
        "conditional_recovery_contract": recovery_contract,
        "validation_artifacts": validation_artifacts,
        "external_split_sha256": external_split_audit["external_split_sha256"],
        "schema6_training_manifest_sha256": manifest["manifest_sha256"],
        "source_checkpoint_sha256": file_sha256(source_path),
    }
    if provider_variant == "body_agnostic_adapter":
        summary["provider_variant"] = "body_agnostic_adapter"
    summary["summary_sha256"] = canonical_sha256(summary)
    atomic_json_new(output / "training_summary.json", summary)
    return summary


def synthetic_smoke(
    provider_variant: str = DEFAULT_PROVIDER_VARIANT,
) -> dict[str, Any]:
    torch.manual_seed(7)
    config = EventWorldModelConfig(state_input_dim=STATE_DIM, action_dim=ACTION_DIM, proprio_dim=14, semantic_dim=96, action_hidden_dim=32, transition_hidden_dim=48, clock_hidden_dim=16, object_delta_dim=6, num_bodies=2, num_policies=2, structured_events=True, dropout=0)
    core = ActionConditionedEventWorldModel(config)
    model = SmolVLAPiperAdapter(
        core,
        state_rank=4,
        action_rank=2,
        provider_variant=provider_variant,
    )
    root_state = torch.randn(3, STATE_DIM); actions = torch.randn(3, 5, ACTION_DIM)
    state = torch.zeros(3, CAUSAL_HISTORY_MAX_STEPS, STATE_DIM)
    state[:, 0] = root_state
    history_mask = torch.zeros(3, CAUSAL_HISTORY_MAX_STEPS, dtype=torch.bool)
    history_mask[:, 0] = True
    assert torch.equal(model.state_adapter(state), state) and torch.equal(model.action_adapter(actions), actions)
    batch = {"state": state, "history_mask": history_mask, "actions": actions, "action_mask": torch.tensor([[1,0,0,0,0]]*3, dtype=torch.bool), "proprio": torch.zeros(3,14), "current_event_id": torch.tensor([0,1,2]), "current_predicates": torch.zeros(3,5), "dt": torch.ones(3), "post_event_id": torch.tensor([1,2,2]), "next_event_id": torch.tensor([1,3,2]), "duration": torch.tensor([1.,2.,3.]), "duration_observed": torch.tensor([1,0,1], dtype=torch.bool), "success": torch.tensor([0.,0.,1.]), "object_delta": torch.zeros(3,6), "object_mask": torch.ones(3,6, dtype=torch.bool)}
    output = model(batch); loss, _ = compute_loss(output, batch, object_mean=torch.zeros(6), object_std=torch.ones(6)); loss.backward()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0); optimizer.step()
    audit = model.enforce_and_verify_frozen_core()
    result = {"loss_finite": math.isfinite(float(loss.detach())), "identity_initialization": True, "policy_rows_bit_exact": bool(audit["all_core_tensors_except_piper_body_row_bit_exact"]), "policy_row_used": 0, "reserved_openvla_row_not_used": 1}
    if provider_variant == "body_agnostic_adapter":
        result.update(
            {
                "provider_variant": "body_agnostic_adapter",
                "prediction_body_row": 0,
                "reserved_target_body_row_bit_exact": audit[
                    "reserved_target_body_row_bit_exact"
                ],
                "clock_parameters_fixed_exact_zero": True,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "synthetic-smoke"), default="train")
    parser.add_argument(
        "--provider-variant",
        choices=PROVIDER_VARIANTS,
        default=DEFAULT_PROVIDER_VARIANT,
    )
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--schema6-manifest", type=Path)
    parser.add_argument("--expected-manifest-split-receipt", type=Path)
    parser.add_argument("--expected-manifest-split-receipt-file-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--canonical-teacher-checkpoint", type=Path)
    parser.add_argument("--canonical-event-spec", type=Path)
    parser.add_argument("--semantic-alignment-weight", type=float, default=0.1)
    parser.add_argument("--pairwise-ranking-weight", type=float, default=0.75)
    parser.add_argument("--listwise-ranking-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ranking-batch-groups", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--recovery-steps", type=int, default=1000)
    parser.add_argument("--recovery-eval-every", type=int, default=25)
    parser.add_argument("--recovery-learning-rate", type=float, default=3e-4)
    parser.add_argument("--state-rank", type=int, default=16)
    parser.add_argument("--action-rank", type=int, default=4)
    parser.add_argument("--training-seed", type=int, default=1702)
    parser.add_argument("--min-train-groups", type=int, default=DEFAULT_MIN_TRAIN_GROUPS)
    parser.add_argument("--min-validation-groups", type=int, default=DEFAULT_MIN_VALIDATION_GROUPS)
    parser.add_argument("--min-test-groups", type=int, default=DEFAULT_MIN_TEST_GROUPS)
    parser.add_argument("--min-outcome-groups", type=int, default=DEFAULT_MIN_OUTCOME_GROUPS)
    parser.add_argument("--min-discordant-groups", type=int, default=DEFAULT_MIN_DISCORDANT_GROUPS)
    parser.add_argument("--min-event-rows", type=int, default=DEFAULT_MIN_EVENT_ROWS)
    parser.add_argument("--min-duration-rows", type=int, default=DEFAULT_MIN_DURATION_ROWS)
    parser.add_argument("--min-object-rows", type=int, default=DEFAULT_MIN_OBJECT_ROWS)
    parser.add_argument("--min-candidate-index-groups", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "train" and any(value is None for value in (
        args.source_checkpoint, args.schema6_manifest,
        args.expected_manifest_split_receipt,
        args.expected_manifest_split_receipt_file_sha256,
        args.output, args.canonical_event_spec,
    )):
        parser.error(
            "train mode requires source/schema6 manifest, externally expected "
            "manifest/split receipt + file SHA, canonical event spec and output"
        )
    positive_ints = (
        args.steps, args.eval_every, args.batch_size, args.ranking_batch_groups,
        args.state_rank, args.action_rank, args.recovery_steps,
        args.recovery_eval_every,
    )
    gate_ints = (
        args.min_train_groups, args.min_validation_groups, args.min_test_groups,
        args.min_outcome_groups, args.min_discordant_groups, args.min_event_rows,
        args.min_duration_rows, args.min_object_rows, args.min_candidate_index_groups,
        args.training_seed,
    )
    finite_nonnegative = (
        args.semantic_alignment_weight,
        args.pairwise_ranking_weight,
        args.listwise_ranking_weight,
    )
    if (
        min(positive_ints) < 1
        or any(value < 0 for value in gate_ints)
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.recovery_learning_rate)
        or args.recovery_learning_rate <= 0
        or any(not math.isfinite(value) or value < 0 for value in finite_nonnegative)
    ):
        parser.error("positive training dimensions/steps are required")
    return args


def main() -> None:
    args = parse_args()
    result = (
        synthetic_smoke(args.provider_variant)
        if args.mode == "synthetic-smoke"
        else train(args)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
