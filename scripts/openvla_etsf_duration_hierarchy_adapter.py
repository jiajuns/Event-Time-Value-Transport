#!/usr/bin/env python3
"""Development-only, policy-agnostic duration-v2 deployment adapter.

The adapter exposes duration prediction only.  It has no actor, reward,
candidate-ranking, success, or selector interface.  Its sole learned input is
the already-frozen factual duration log-location supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from openvla_etsf_duration_hierarchy import (
    DURATION_HIERARCHY_FORMAT,
    DURATION_HIERARCHY_PROTOCOL_V2,
    LOOKUP_ORDER,
    MINIMUM_APPLIED_SOURCE_SUPPORT,
    apply_duration_hierarchy,
    canonical_sha256,
    fit_duration_hierarchy,
    validate_duration_hierarchy_contract,
)


FINAL_HIERARCHY_FORMAT = "etsf_duration_hierarchy_final_d250_contract_v1"
ACTIVATION_FORMAT = "etsf_duration_v2_prediction_activation_v1"
EMPIRICAL_REGISTRY_FORMAT = "etsf_duration_empirical_body_policy_registry_v1"
RESIDUAL_MULTIPLIER = 0.375
DEVELOPMENT_GROUP_COUNT = 250


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_fresh_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a one-dimensional aligned array")
    return result


def _boolean(value: Any, *, name: str, length: int) -> np.ndarray:
    raw = _vector(value, name=name, length=length)
    if raw.dtype == np.bool_:
        return raw.astype(bool)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be binary") from error
    if not np.isfinite(numeric).all() or np.any(
        (numeric != 0.0) & (numeric != 1.0)
    ):
        raise ValueError(f"{name} must be binary")
    return numeric.astype(bool)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _final_as_v2(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Build an ephemeral v2 validator/applicator view of final source tables."""

    value: dict[str, Any] = {
        "format": DURATION_HIERARCHY_FORMAT,
        "protocol": DURATION_HIERARCHY_PROTOCOL_V2,
        "status": "fitted_outer_training_only",
        # The final contract has no OOF owner.  This value only satisfies the
        # legacy v2 source-table container and is never persisted or exposed.
        "owner_fold_id": 0,
        "fit_scope": "outer_training_only",
        "fit_label_scope": "observed_duration_only",
        "current_event_field": "current_event_id",
        "clock_event_proxy_allowed": False,
        "target_transform": "log1p_duration",
        "lookup_order": list(LOOKUP_ORDER),
        "minimum_applied_source_support": MINIMUM_APPLIED_SOURCE_SUPPORT,
        "support_unit": "observed_outer_training_rows",
        "outer_training_rows": int(contract["development_rows"]),
        "outer_training_observed_rows": int(contract["dense_observed_rows"]),
        "outer_training_logical_groups": list(contract["development_groups"]),
        "outer_training_logical_groups_sha256": contract[
            "development_groups_canonical_sha256"
        ],
        "sources": contract["sources"],
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def fit_final_duration_hierarchy(
    *,
    duration: Any,
    duration_observed: Any,
    dense_mask: Any,
    current_event_id: Any,
    body_id: Any,
    logical_group: Any,
    development_groups: Sequence[str],
    materialization_groups_sha256: str,
) -> dict[str, Any]:
    """Fit the immutable all-D250 source table from dense observed rows."""

    duration_array = _vector(duration, name="duration").astype(np.float64)
    length = len(duration_array)
    if length == 0 or not np.isfinite(duration_array).all() or np.any(
        duration_array < 0.0
    ):
        raise ValueError("duration must be finite, non-negative, and non-empty")
    observed = _boolean(
        duration_observed, name="duration_observed", length=length
    )
    dense = _boolean(dense_mask, name="dense_mask", length=length)
    groups = np.asarray(
        [str(item) for item in _vector(logical_group, name="logical_group", length=length)],
        dtype=object,
    )
    registry = list(map(str, development_groups))
    if (
        len(registry) != DEVELOPMENT_GROUP_COUNT
        or registry != sorted(registry)
        or len(set(registry)) != DEVELOPMENT_GROUP_COUNT
        or set(groups.tolist()) != set(registry)
    ):
        raise RuntimeError("final duration hierarchy requires exact D250 coverage")
    selected = observed & dense
    generated = fit_duration_hierarchy(
        duration=duration_array,
        duration_observed=selected,
        current_event_id=current_event_id,
        body_id=body_id,
        logical_group=groups,
        split_role=np.repeat("outer_training", length),
        owner_fold_id=0,
    )
    contract: dict[str, Any] = {
        "format": FINAL_HIERARCHY_FORMAT,
        "protocol": DURATION_HIERARCHY_PROTOCOL_V2,
        "status": "fitted_all_d250_development_only",
        "fit_scope": "all_D250_dense_and_observed_development_only",
        "future_or_fresh_labels_used": False,
        "current_event_field": "current_event_id",
        "clock_event_proxy_allowed": False,
        "target_transform": "log1p_duration",
        "lookup_order": list(LOOKUP_ORDER),
        "minimum_applied_source_support": MINIMUM_APPLIED_SOURCE_SUPPORT,
        "support_unit": "dense_observed_development_rows",
        "development_rows": length,
        "dense_observed_rows": int(selected.sum()),
        "development_groups": registry,
        "development_groups_canonical_sha256": canonical_sha256(registry),
        "materialization_groups_sha256": str(materialization_groups_sha256),
        "sources": generated["sources"],
        "source_table_generated_by": DURATION_HIERARCHY_PROTOCOL_V2,
        "source_generator_contract_sha256": generated["contract_sha256"],
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    validate_final_duration_hierarchy(contract)
    return contract


def validate_final_duration_hierarchy(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(contract)
    recorded = unsigned.pop("contract_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("final duration hierarchy signature mismatch")
    groups = contract.get("development_groups")
    if (
        contract.get("format") != FINAL_HIERARCHY_FORMAT
        or contract.get("protocol") != DURATION_HIERARCHY_PROTOCOL_V2
        or contract.get("status") != "fitted_all_d250_development_only"
        or contract.get("fit_scope")
        != "all_D250_dense_and_observed_development_only"
        or contract.get("future_or_fresh_labels_used") is not False
        or contract.get("current_event_field") != "current_event_id"
        or contract.get("clock_event_proxy_allowed") is not False
        or contract.get("target_transform") != "log1p_duration"
        or contract.get("lookup_order") != list(LOOKUP_ORDER)
        or contract.get("minimum_applied_source_support")
        != MINIMUM_APPLIED_SOURCE_SUPPORT
        or contract.get("support_unit") != "dense_observed_development_rows"
        or not isinstance(groups, list)
        or len(groups) != DEVELOPMENT_GROUP_COUNT
        or groups != sorted(groups)
        or len(set(groups)) != DEVELOPMENT_GROUP_COUNT
        or canonical_sha256(groups)
        != contract.get("development_groups_canonical_sha256")
        or not _is_sha256(contract.get("materialization_groups_sha256"))
        or not isinstance(contract.get("development_rows"), int)
        or int(contract["development_rows"]) < DEVELOPMENT_GROUP_COUNT
        or not isinstance(contract.get("dense_observed_rows"), int)
        or int(contract["dense_observed_rows"])
        < MINIMUM_APPLIED_SOURCE_SUPPORT
        or contract.get("source_table_generated_by")
        != DURATION_HIERARCHY_PROTOCOL_V2
    ):
        raise RuntimeError("final duration hierarchy protocol changed")
    v2_view = _final_as_v2(contract)
    validate_duration_hierarchy_contract(v2_view)
    if v2_view["contract_sha256"] != contract.get(
        "source_generator_contract_sha256"
    ):
        raise RuntimeError("final duration source-generator hash mismatch")
    return dict(contract)


def validate_empirical_registry_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(contract)
    recorded = unsigned.pop("registry_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("empirical body/policy registry signature mismatch")
    bodies = contract.get("observed_bodies")
    policies = contract.get("observed_policies")
    cells = contract.get("observed_policy_body_cells")
    if (
        contract.get("format") != EMPIRICAL_REGISTRY_FORMAT
        or contract.get("status")
        != "authenticated_D250_observed_registry_one_cell_only"
        or contract.get("logical_groups") != DEVELOPMENT_GROUP_COUNT
        or contract.get("one_cell_only") is not True
        or contract.get("cross_body_validated") is not False
        or contract.get("cross_policy_validated") is not False
        or not isinstance(bodies, list)
        or len(bodies) != 1
        or not isinstance(policies, list)
        or len(policies) != 1
        or not isinstance(cells, list)
        or len(cells) != 1
    ):
        raise RuntimeError("empirical body/policy registry protocol changed")
    body, policy, cell = bodies[0], policies[0], cells[0]
    group_shas = {
        item.get("logical_groups_canonical_sha256")
        for item in (body, policy, cell)
        if isinstance(item, Mapping)
    }
    if (
        not isinstance(body, Mapping)
        or not isinstance(policy, Mapping)
        or not isinstance(cell, Mapping)
        or not isinstance(body.get("body"), str)
        or not body["body"]
        or not isinstance(body.get("body_id"), int)
        or body["body_id"] < 0
        or not isinstance(policy.get("policy"), str)
        or not policy["policy"]
        or not isinstance(policy.get("policy_id"), int)
        or policy["policy_id"] < 0
        or cell.get("body") != body["body"]
        or cell.get("body_id") != body["body_id"]
        or cell.get("policy") != policy["policy"]
        or cell.get("policy_id") != policy["policy_id"]
        or any(
            item.get("logical_group_count") != DEVELOPMENT_GROUP_COUNT
            or not _is_sha256(item.get("logical_groups_canonical_sha256"))
            for item in (body, policy, cell)
        )
        or len(group_shas) != 1
        or not _is_sha256(contract.get("development_groups_sha256"))
    ):
        raise RuntimeError("empirical body/policy registry entries are invalid")
    return dict(contract)


def validate_duration_activation(
    activation: Mapping[str, Any], *, verify_local_code: bool = True
) -> dict[str, Any]:
    unsigned = dict(activation)
    recorded = unsigned.pop("activation_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("duration activation signature mismatch")
    permissions = activation.get("permissions")
    evidence = activation.get("evidence")
    hierarchy = activation.get("final_hierarchy_contract")
    registry = activation.get("empirical_registry_contract")
    scope = activation.get("empirical_evidence_scope")
    coverage = activation.get("development_coverage")
    if (
        activation.get("format") != ACTIVATION_FORMAT
        or "actor_or_policy_specific" in activation
        or activation.get("status")
        != "activated_duration_prediction_only_development"
        or activation.get("evidence_scope") != "adaptive_development_only"
        or activation.get("duration_residual_multiplier") != RESIDUAL_MULTIPLIER
        or activation.get("formula")
        != "baseline+0.375*(frozen_duration_log_mean-baseline)"
        or activation.get("interface_actor_policy_agnostic") is not True
        or activation.get("transfer_claim_authorized") is not False
        or activation.get("fresh50_inputs_accepted") is not False
        or activation.get("fresh50_labels_read") is not False
        or activation.get("fresh50_confirmation_authorized") is not False
        or activation.get("selector_authorized") is not False
        or activation.get("prospective_claim_allowed") is not False
        or permissions
        != {
            "duration_prediction_adapter": True,
            "actor_control": False,
            "policy_modification": False,
            "reward_or_value": False,
            "candidate_ranking": False,
            "selector": False,
        }
        or not isinstance(evidence, Mapping)
        or not isinstance(coverage, Mapping)
        or not isinstance(hierarchy, Mapping)
        or not isinstance(registry, Mapping)
        or not isinstance(scope, Mapping)
        or coverage.get("logical_groups") != DEVELOPMENT_GROUP_COUNT
        or coverage.get("five_holdouts_cover_each_group_exactly_once") is not True
        or coverage.get("development_groups_sha256")
        != hierarchy.get("materialization_groups_sha256")
        or coverage.get("development_groups_sha256")
        != registry.get("development_groups_sha256")
        or coverage.get("dense_observed_rows")
        != hierarchy.get("dense_observed_rows")
        or coverage.get("materialized_rows") != hierarchy.get("development_rows")
        or hierarchy.get("contract_sha256")
        != activation.get("final_hierarchy_contract_sha256")
        or registry.get("registry_sha256")
        != activation.get("empirical_registry_contract_sha256")
    ):
        raise RuntimeError("duration activation permission/protocol changed")
    validate_final_duration_hierarchy(hierarchy)
    validate_empirical_registry_contract(registry)
    body = registry["observed_bodies"][0]
    policy = registry["observed_policies"][0]
    if scope != {
        "policy": policy["policy"],
        "policy_id": policy["policy_id"],
        "body": body["body"],
        "body_id": body["body_id"],
        "one_cell_only": True,
        "cross_body_validated": False,
        "cross_policy_validated": False,
    }:
        raise RuntimeError("duration activation empirical evidence scope changed")
    required_evidence = (
        "factual_checkpoint_sha256",
        "factual_state_sha256",
        "event_spec_sha256",
        "materialization_sha256",
        "materialization_file_sha256",
        "r5_result_sha256",
        "r5_result_file_sha256",
        "r5_rows_file_sha256",
    )
    if any(
        not isinstance(evidence.get(key), str) or len(evidence[key]) != 64
        for key in required_evidence
    ):
        raise RuntimeError("duration activation evidence hashes are incomplete")
    source_paths = activation.get("source_paths")
    if not isinstance(source_paths, Mapping) or set(source_paths) != {
        "materialization_manifest",
        "r5_result_json",
        "r5_rows_npz",
    }:
        raise RuntimeError("duration activation source paths are incomplete")
    for value in source_paths.values():
        if not isinstance(value, str) or not Path(value).is_absolute() or any(
            token in part.lower()
            for part in Path(value).parts
            for token in ("fresh", "confirmation")
        ):
            raise RuntimeError("duration activation source path crossed Fresh boundary")
    code = activation.get("implementation_files")
    if not isinstance(code, Mapping):
        raise RuntimeError("duration activation implementation hashes are missing")
    required_code = {
        "openvla_etsf_duration_hierarchy.py",
        "openvla_etsf_duration_hierarchy_adapter.py",
        "freeze_openvla_etsf_duration_hierarchy_activation.py",
        "evaluate_openvla_etsf_duration_hierarchy_oof.py",
        "openvla_etsf_v8_structured_adapters.py",
        "train_openvla_etsf_v8_structured_adapters.py",
    }
    if set(code) != required_code or any(
        not isinstance(value, str) or len(value) != 64 for value in code.values()
    ):
        raise RuntimeError("duration activation implementation hash set changed")
    if verify_local_code:
        scripts_root = Path(__file__).resolve().parent
        for filename in (
            "openvla_etsf_duration_hierarchy.py",
            "openvla_etsf_duration_hierarchy_adapter.py",
        ):
            if sha256_path(scripts_root / filename) != code.get(filename):
                raise RuntimeError(f"duration adapter code hash mismatch: {filename}")
    return dict(activation)


def load_duration_activation(path: Path) -> dict[str, Any]:
    path = _reject_fresh_path(path, role="duration activation")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError("duration activation must contain a JSON object")
    return validate_duration_activation(value)


def predict_duration_candidates(
    activation: Mapping[str, Any],
    *,
    body_registry_contract: Mapping[str, Any],
    current_event_id: Any,
    body_id: Any,
    frozen_duration_log_mean: Any,
) -> dict[str, np.ndarray]:
    """Predict candidate durations without ranking or changing any policy."""

    activation = validate_duration_activation(activation)
    body_registry_contract = validate_empirical_registry_contract(
        body_registry_contract
    )
    if body_registry_contract.get("registry_sha256") != activation.get(
        "empirical_registry_contract_sha256"
    ):
        raise RuntimeError("caller body registry does not match activation evidence")
    frozen = _vector(
        frozen_duration_log_mean, name="frozen_duration_log_mean"
    ).astype(np.float64)
    if not np.isfinite(frozen).all():
        raise ValueError("frozen_duration_log_mean must be finite")
    hierarchy = activation["final_hierarchy_contract"]
    v2_view = _final_as_v2(hierarchy)
    applied = apply_duration_hierarchy(
        v2_view,
        current_event_id=current_event_id,
        body_id=body_id,
        expected_training_logical_groups_sha256=hierarchy[
            "development_groups_canonical_sha256"
        ],
    )
    baseline = applied["baseline_log1p_duration"]
    if len(baseline) != len(frozen):
        raise ValueError("candidate duration inputs must be aligned")
    predicted = baseline + RESIDUAL_MULTIPLIER * (frozen - baseline)
    if not np.isfinite(predicted).all():
        raise RuntimeError("duration adapter produced a non-finite result")
    body_values = _vector(body_id, name="body_id", length=len(frozen)).astype(
        np.int64
    )
    observed_body_ids = {
        int(item["body_id"])
        for item in body_registry_contract["observed_bodies"]
    }
    out_of_scope = np.asarray(
        [int(value) not in observed_body_ids for value in body_values], dtype=bool
    )
    return {
        "duration_log_location": predicted,
        "baseline_log_location": baseline,
        "source_kind": applied["source_kind"],
        "source_key": applied["source_key"],
        "source_support": applied["source_support"],
        "source_logical_group_support": applied[
            "source_logical_group_support"
        ],
        "source_training_logical_groups_sha256": applied[
            "source_training_logical_groups_sha256"
        ],
        "out_of_empirical_body_scope": out_of_scope,
        "transfer_claim_authorized": np.zeros(len(frozen), dtype=bool),
        "cross_body_validated": np.zeros(len(frozen), dtype=bool),
        "cross_policy_validated": np.zeros(len(frozen), dtype=bool),
    }


__all__ = [
    "ACTIVATION_FORMAT",
    "DEVELOPMENT_GROUP_COUNT",
    "EMPIRICAL_REGISTRY_FORMAT",
    "FINAL_HIERARCHY_FORMAT",
    "RESIDUAL_MULTIPLIER",
    "fit_final_duration_hierarchy",
    "load_duration_activation",
    "predict_duration_candidates",
    "sha256_path",
    "validate_duration_activation",
    "validate_empirical_registry_contract",
    "validate_final_duration_hierarchy",
]
