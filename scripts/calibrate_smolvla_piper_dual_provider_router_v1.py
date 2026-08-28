#!/usr/bin/env python3
"""In-memory nested OOF calibration for two five-member head providers.

The core API accepts already materialized NumPy arrays.  It performs no file,
HDF, simulator, policy, checkpoint, training, signing, or deployment I/O.  No
external gate booleans or evidence SHAs are accepted: support, calibration,
baseline skill, uncertainty/AURC, paired gain, and harmful-rate evidence are
all recomputed from the supplied arrays.

``source_contract_rank_score`` is deliberately outside this router.  The
returned contract always keeps rank routing unauthorized and requires the
actor baseline until a separate whole-provider OOF protocol is passed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import calibrate_smolvla_piper_adapter_ensemble as legacy


FORMAT = "etsf_smolvla_piper_dual_provider_router_v1"
STATUS = "complete_in_memory_nested_oof_no_rank_route_no_promotion"
PROVIDERS = ("body_agnostic_adapter", "body_conditioned_adapter")
REFERENCE_PROVIDER = PROVIDERS[0]
CANDIDATE_PROVIDER = PROVIDERS[1]
HEADS = (
    "post_event",
    "next_event",
    "duration",
    "success",
    "recovery",
    "object_effect",
)
EVENT_VOCAB = ("e0", "e12", "e3", "e4", "eK")
MEMBER_COUNT = 5
FOLD_COUNT = 5
BOOTSTRAP_SEED = 20260829
DEFAULT_BOOTSTRAP_SAMPLES = 5000
MINIMUM_SUPPORT = {
    "post_event": 10,
    "next_event": 10,
    "duration": 10,
    "success": 50,
    "recovery": 10,
    "object_effect": 50,
}
MAXIMUM_HARMFUL_RATE_UCB = 0.10
MAXIMUM_ABSOLUTE_COVERAGE_ERROR = 0.10

LABEL_KEYS = frozenset(
    {
        "sample_id",
        "group_id",
        "group_row_ordinal",
        "semantic_reset_cluster_id",
        "body_id",
        "actor_contract_id",
        "current_event",
        "post_event",
        "post_event_observed",
        "next_event",
        "next_event_observed",
        "success",
        "success_observed",
        "regress",
        "recovery",
        "recovery_observed",
        "duration",
        "duration_applicable",
        "duration_observed",
        "object_target",
        "object_observed",
    }
)
PROVIDER_KEYS = frozenset(
    {
        "provider_id",
        "provider_artifact_sha256",
        "provider_manifest",
        "member_count",
        "sample_id",
        "group_id",
        "group_row_ordinal",
        "applicable_masks",
        "post_event_logits",
        "next_event_logits",
        "success_logit",
        "recovery_logit",
        "duration_log_mean",
        "duration_log_scale",
        "object_mean",
        "object_log_scale",
    }
)
PREDICTION_ARRAY_KEYS = (
    "post_event_logits", "next_event_logits", "success_logit", "recovery_logit",
    "duration_log_mean", "duration_log_scale", "object_mean", "object_log_scale",
)


class DualProviderRouterError(RuntimeError):
    """An array, split, calibration, or nested-OOF invariant failed."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_provider_manifest(
    value: Any, provider_id: str, labels: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    fields = {
        "provider_id", "provider_format", "shared_core_lineage_sha256",
        "provider_artifact_sha256",
        "prediction_tensor_sha256",
        "training_execution_group_ids", "training_semantic_reset_cluster_ids",
        "members",
    }
    if not isinstance(value, Mapping) or set(value) != fields | {"manifest_sha256"}:
        raise DualProviderRouterError("provider manifest fields changed")
    unsigned = {key: child for key, child in value.items() if key != "manifest_sha256"}
    if value["manifest_sha256"] != canonical_sha256(unsigned):
        raise DualProviderRouterError("provider manifest canonical SHA mismatch")
    if (
        value["provider_id"] != provider_id
        or value["provider_format"]
        != "etsf_smolvla_piper_frozen_five_member_provider_manifest_v1"
        or not _is_sha256(value["shared_core_lineage_sha256"])
        or not _is_sha256(value["provider_artifact_sha256"])
        or not _is_sha256(value["prediction_tensor_sha256"])
    ):
        raise DualProviderRouterError("provider manifest identity changed")
    training_groups = value["training_execution_group_ids"]
    training_clusters = value["training_semantic_reset_cluster_ids"]
    for rows, role in ((training_groups, "training groups"), (training_clusters, "training clusters")):
        if not isinstance(rows, list) or not rows or rows != sorted(set(rows)) or any(not isinstance(row, str) or not row for row in rows):
            raise DualProviderRouterError(f"provider manifest {role} changed")
    if set(training_groups) & set(labels["group_id"].tolist()):
        raise DualProviderRouterError("provider training/development execution groups overlap")
    if set(training_clusters) & set(labels["semantic_reset_cluster_id"].tolist()):
        raise DualProviderRouterError("provider training/development semantic clusters overlap")
    members = value["members"]
    if not isinstance(members, list) or len(members) != MEMBER_COUNT:
        raise DualProviderRouterError("provider manifest must bind five members")
    checkpoints = set()
    seeds = set()
    normalized_members = []
    for index, row in enumerate(members):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"member_index", "seed", "checkpoint_sha256"}
            or row["member_index"] != index
            or type(row["seed"]) is not int
            or row["seed"] < 0
            or row["seed"] in seeds
            or not _is_sha256(row["checkpoint_sha256"])
            or row["checkpoint_sha256"] in checkpoints
        ):
            raise DualProviderRouterError("provider member manifest changed")
        checkpoints.add(row["checkpoint_sha256"])
        seeds.add(row["seed"])
        normalized_members.append(dict(row))
    return {**unsigned, "members": normalized_members, "manifest_sha256": value["manifest_sha256"]}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def build_provider_manifest(
    *,
    provider_id: str,
    provider_artifact_sha256: str,
    shared_core_lineage_sha256: str,
    prediction_tensor_sha256: str,
    training_execution_group_ids: Sequence[str],
    training_semantic_reset_cluster_ids: Sequence[str],
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact content-addressed five-member input manifest."""

    if provider_id not in PROVIDERS:
        raise DualProviderRouterError("manifest provider id is not canonical")
    if (
        not _is_sha256(provider_artifact_sha256)
        or not _is_sha256(shared_core_lineage_sha256)
        or not _is_sha256(prediction_tensor_sha256)
    ):
        raise DualProviderRouterError("manifest artifact/lineage SHA changed")
    training_groups = list(training_execution_group_ids)
    training_clusters = list(training_semantic_reset_cluster_ids)
    if (
        not training_groups or training_groups != sorted(set(training_groups))
        or not training_clusters or training_clusters != sorted(set(training_clusters))
        or any(not isinstance(value, str) or not value for value in training_groups + training_clusters)
    ):
        raise DualProviderRouterError("manifest training identities must be sorted unique strings")
    member_rows = [dict(row) for row in members]
    if len(member_rows) != MEMBER_COUNT:
        raise DualProviderRouterError("manifest must contain five members")
    seen_seeds: set[int] = set()
    seen_checkpoints: set[str] = set()
    for index, row in enumerate(member_rows):
        if (
            set(row) != {"member_index", "seed", "checkpoint_sha256"}
            or row["member_index"] != index
            or type(row["seed"]) is not int or row["seed"] < 0
            or row["seed"] in seen_seeds
            or not _is_sha256(row["checkpoint_sha256"])
            or row["checkpoint_sha256"] in seen_checkpoints
        ):
            raise DualProviderRouterError("manifest member inventory changed")
        seen_seeds.add(row["seed"])
        seen_checkpoints.add(row["checkpoint_sha256"])

    base = {
        "provider_id": provider_id,
        "provider_format": "etsf_smolvla_piper_frozen_five_member_provider_manifest_v1",
        "provider_artifact_sha256": provider_artifact_sha256,
        "shared_core_lineage_sha256": shared_core_lineage_sha256,
        "prediction_tensor_sha256": prediction_tensor_sha256,
        "training_execution_group_ids": training_groups,
        "training_semantic_reset_cluster_ids": training_clusters,
        "members": member_rows,
    }
    return {**base, "manifest_sha256": canonical_sha256(base)}


def prediction_tensor_set_sha256(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping) or not set(PREDICTION_ARRAY_KEYS) <= set(value):
        raise DualProviderRouterError("prediction tensor set is incomplete")
    return canonical_sha256(
        {
            key: _ndarray_sha256(np.asarray(value[key], dtype=np.float64))
            for key in PREDICTION_ARRAY_KEYS
        }
    )


def _finite(array: Any, role: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = np.asarray(array)
    if shape is not None and value.shape != shape:
        raise DualProviderRouterError(f"{role} shape changed")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise DualProviderRouterError(f"{role} must be finite numeric")
    return value.astype(np.float64, copy=False)


def _prediction_float(
    array: Any, role: str, shape: tuple[int, ...]
) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
        raise DualProviderRouterError(f"{role} must be an exact floating prediction tensor")
    if not np.isfinite(value).all():
        raise DualProviderRouterError(f"{role} must be finite")
    return value.astype(np.float64, copy=False)


def _bool(array: Any, role: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape or value.dtype != np.bool_:
        raise DualProviderRouterError(f"{role} must be exact bool {shape}")
    return value


def _int(array: Any, role: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape or not np.issubdtype(value.dtype, np.integer):
        raise DualProviderRouterError(f"{role} must be integer {shape}")
    return value.astype(np.int64, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def _group_row_weights(
    groups: np.ndarray,
    mask: np.ndarray,
    semantic_clusters: np.ndarray | None = None,
) -> np.ndarray:
    weights = np.zeros(len(groups), dtype=np.float64)
    if semantic_clusters is None:
        semantic_clusters = groups
    clusters = sorted(set(semantic_clusters[mask].tolist()))
    if not clusters:
        return weights
    for cluster in clusters:
        cluster_mask = mask & (semantic_clusters == cluster)
        executions = sorted(set(groups[cluster_mask].tolist()))
        for execution in executions:
            selected = cluster_mask & (groups == execution)
            weights[selected] = 1.0 / (
                len(clusters) * len(executions) * int(selected.sum())
            )
    return weights


def _group_means(
    values: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    execution_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if bool((mask & ~np.isfinite(values)).any()):
        raise DualProviderRouterError(
            "masked evidence contains non-finite values; row deletion is forbidden"
        )
    valid = mask
    names = np.asarray(sorted(set(groups[valid].tolist())), dtype=str)
    means = []
    for name in names:
        selected = valid & (groups == name)
        if execution_groups is None:
            means.append(float(values[selected].mean()))
        else:
            execution_names = sorted(set(execution_groups[selected].tolist()))
            means.append(
                float(
                    np.mean(
                        [values[selected & (execution_groups == group)].mean() for group in execution_names]
                    )
                )
            )
    means = np.asarray(means, dtype=np.float64)
    return names, means


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or values.shape != weights.shape or weights.sum() <= 0:
        raise DualProviderRouterError("weighted median requires non-empty positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, 0.5 * weights.sum(), side="left")]])


def _seed_offset(role: str) -> int:
    return int.from_bytes(hashlib.sha256(role.encode("utf-8")).digest()[:4], "big")


def _bootstrap_interval(
    values: np.ndarray, *, samples: int, role: str
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        return {"point": None, "lcb95": None, "ucb95": None, "groups": len(array)}
    # A fixed seed makes paired statistics over the same sorted group universe
    # use identical resampling draws. ``role`` labels evidence; it never changes
    # the draws.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "point": float(array.mean()),
        "lcb95": float(np.quantile(means, 0.025)),
        "ucb95": float(np.quantile(means, 0.975)),
        "groups": int(len(array)),
        "shared_draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
    }


def _gain_evidence(
    model_loss: np.ndarray,
    reference_loss: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    execution_groups: np.ndarray | None = None,
    *,
    samples: int,
    role: str,
) -> dict[str, Any]:
    model_names, model = _group_means(model_loss, groups, mask, execution_groups)
    reference_names, reference = _group_means(reference_loss, groups, mask, execution_groups)
    if not np.array_equal(model_names, reference_names):
        raise DualProviderRouterError("paired group loss membership changed")
    result = _bootstrap_interval(reference - model, samples=samples, role=role)
    return {
        **result,
        "orientation": "positive_is_first_model_improvement",
        "equal_group_weighting": True,
        "group_ids_sha256": canonical_sha256(model_names.tolist()),
    }


def _harm_evidence(
    candidate_proper: np.ndarray,
    candidate_decision: np.ndarray,
    reference_proper: np.ndarray,
    reference_decision: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    execution_groups: np.ndarray | None = None,
    *,
    samples: int,
    role: str,
) -> dict[str, Any]:
    results = {}
    shared_sha = None
    for metric, candidate_values, reference_values in (
        ("proper_loss", candidate_proper, reference_proper),
        ("decision_loss", candidate_decision, reference_decision),
    ):
        names, candidate = _group_means(candidate_values, groups, mask, execution_groups)
        other_names, reference = _group_means(reference_values, groups, mask, execution_groups)
        if not np.array_equal(names, other_names):
            raise DualProviderRouterError("harmful-rate group membership changed")
        if len(names) == 0:
            results[metric] = {
                "point": None,
                "ucb95": None,
                "descriptive_bootstrap_ucb95": None,
                "groups": 0,
                "harmful_definition": f"candidate_semantic_cluster_{metric}_gt_reference",
                "gate_interval": "one_sided_95pct_wilson_semantic_cluster_proportion",
                "shared_draws_sha256": None,
            }
            continue
        result = _bootstrap_interval(
            (candidate > reference + 1e-12).astype(np.float64),
            samples=samples,
            role=f"{role}:{metric}",
        )
        successes = int(np.sum(candidate > reference + 1e-12))
        total = len(candidate)
        z = 1.6448536269514722
        proportion = successes / total
        denominator = 1.0 + z * z / total
        center = proportion + z * z / (2.0 * total)
        radius = z * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        wilson_ucb = min(1.0, (center + radius) / denominator)
        if shared_sha is not None and result.get("shared_draws_sha256") != shared_sha:
            raise DualProviderRouterError("paired harm metrics did not share bootstrap draws")
        shared_sha = result.get("shared_draws_sha256")
        results[metric] = {
            "point": result["point"],
            "ucb95": float(wilson_ucb),
            "descriptive_bootstrap_ucb95": result["ucb95"],
            "groups": result["groups"],
            "harmful_definition": f"candidate_semantic_cluster_{metric}_gt_reference",
            "gate_interval": "one_sided_95pct_wilson_semantic_cluster_proportion",
            "shared_draws_sha256": shared_sha,
        }
    return {
        **results,
        "both_harmful_rate_ucb_passed": bool(
            all(
                row["ucb95"] is not None
                and row["ucb95"] <= MAXIMUM_HARMFUL_RATE_UCB
                for row in results.values()
            )
        ),
        "shared_draws_sha256": shared_sha,
    }


def _aurc_statistics(uncertainty: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    order = np.argsort(uncertainty, kind="stable")
    ordered_uncertainty = uncertainty[order]
    ordered_error = error[order].astype(np.float64, copy=True)
    # Replace every exact uncertainty tie block by its random-order expected
    # error. This prevents source row order from manufacturing AURC skill.
    start = 0
    while start < len(ordered_error):
        stop = start + 1
        while stop < len(ordered_error) and ordered_uncertainty[stop] == ordered_uncertainty[start]:
            stop += 1
        ordered_error[start:stop] = float(ordered_error[start:stop].mean())
        start = stop
    risk = np.cumsum(ordered_error) / np.arange(1, len(error) + 1)
    gain = float(error.mean() - risk.mean())
    cut = max(1, len(error) // 4)
    low = float(ordered_error[:cut].mean())
    high = float(ordered_error[-cut:].mean())
    return gain, high - low


def _uncertainty_evidence(
    uncertainty: np.ndarray,
    decision_loss: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    execution_groups: np.ndarray | None = None,
    *,
    samples: int,
    role: str,
) -> dict[str, Any]:
    names_u, group_u = _group_means(uncertainty, groups, mask, execution_groups)
    names_e, group_e = _group_means(decision_loss, groups, mask, execution_groups)
    if not np.array_equal(names_u, names_e) or len(names_u) < 2:
        return {"passed": False, "groups": int(len(names_u)), "status": "insufficient"}
    point_gain, point_gap = _aurc_statistics(group_u, group_e)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(names_u), size=(samples, len(names_u)))
    sampled = np.asarray(
        [_aurc_statistics(group_u[index], group_e[index]) for index in draws],
        dtype=np.float64,
    )
    gain_lcb = float(np.quantile(sampled[:, 0], 0.025))
    gap_lcb = float(np.quantile(sampled[:, 1], 0.025))
    passed = bool(gain_lcb > 0.0 and gap_lcb > 0.0)
    return {
        "status": "computed_from_provider_uncertainty_and_group_decision_loss",
        "passed": passed,
        "groups": int(len(names_u)),
        "aurc_gain_over_random": point_gain,
        "aurc_gain_lcb95": gain_lcb,
        "high_minus_low_uncertainty_quartile_error": point_gap,
        "quartile_error_gap_lcb95": gap_lcb,
        "group_ids_sha256": canonical_sha256(names_u.tolist()),
        "shared_draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
    }


def _derived_masks(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "post_event": labels["post_event_observed"],
        "next_event": labels["next_event_observed"],
        "duration": labels["duration_applicable"],
        "success": labels["success_observed"],
        "recovery": labels["recovery_observed"] & labels["regress"],
        "object_effect": labels["object_observed"],
    }


def _validate_labels(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping) or set(raw) != LABEL_KEYS:
        raise DualProviderRouterError("label array inventory changed")
    sample = np.asarray(raw["sample_id"]).astype(str)
    group = np.asarray(raw["group_id"]).astype(str)
    semantic = np.asarray(raw["semantic_reset_cluster_id"]).astype(str)
    body = np.asarray(raw["body_id"]).astype(str)
    actor = np.asarray(raw["actor_contract_id"]).astype(str)
    n = len(sample)
    ordinal = _int(raw["group_row_ordinal"], "group row ordinal", (n,))
    if (
        n == 0
        or sample.shape != (n,)
        or group.shape != (n,)
        or semantic.shape != (n,)
        or body.shape != (n,)
        or actor.shape != (n,)
        or len(set(sample.tolist())) != n
        or any(
            not value
            for value in sample.tolist() + group.tolist() + semantic.tolist()
            + body.tolist() + actor.tolist()
        )
    ):
        raise DualProviderRouterError("sample/group identity changed")
    for name in sorted(set(group.tolist())):
        selected_group = group == name
        values = ordinal[selected_group]
        if not np.array_equal(values, np.arange(len(values))):
            raise DualProviderRouterError("group row order must be exact and contiguous")
        if (
            len(set(semantic[selected_group].tolist())) != 1
            or len(set(body[selected_group].tolist())) != 1
            or len(set(actor[selected_group].tolist())) != 1
        ):
            raise DualProviderRouterError(
                "one execution group must map to one semantic/body/actor identity"
            )
    current = _int(raw["current_event"], "current event", (n,))
    post = _int(raw["post_event"], "post event", (n,))
    nxt = _int(raw["next_event"], "next event", (n,))
    for name, array in (("current", current), ("post", post), ("next", nxt)):
        if bool(((array < 0) | (array >= len(EVENT_VOCAB))).any()):
            raise DualProviderRouterError(f"{name} event id changed")
    binary = {}
    for name in ("success", "recovery"):
        value = _int(raw[name], name, (n,))
        if not np.isin(value, [0, 1]).all():
            raise DualProviderRouterError(f"{name} must be binary")
        binary[name] = value
    booleans = {
        name: _bool(raw[name], name, (n,))
        for name in (
            "post_event_observed",
            "next_event_observed",
            "success_observed",
            "regress",
            "recovery_observed",
            "duration_applicable",
            "duration_observed",
            "object_observed",
        )
    }
    if not np.array_equal(booleans["next_event_observed"], booleans["duration_observed"]):
        raise DualProviderRouterError("next-event mask must equal observed duration")
    if bool((booleans["next_event_observed"] & (nxt == 0)).any()):
        raise DualProviderRouterError("observed next-event cannot be e0")
    if bool((booleans["duration_observed"] & ~booleans["duration_applicable"]).any()):
        raise DualProviderRouterError("observed duration is outside duration applicability")
    if bool((booleans["recovery_observed"] & ~booleans["regress"]).any()):
        raise DualProviderRouterError("recovery is observed outside operational regress")
    if bool((binary["recovery"].astype(bool) & ~booleans["recovery_observed"]).any()):
        raise DualProviderRouterError("unobserved recovery cannot be positive")
    duration = _finite(raw["duration"], "duration", (n,))
    if bool((duration < 0).any()):
        raise DualProviderRouterError("duration must be nonnegative")
    target = _finite(raw["object_target"], "object target")
    if target.ndim != 2 or target.shape[0] != n or target.shape[1] < 1:
        raise DualProviderRouterError("object target shape changed")
    return {
        "sample_id": sample,
        "group_id": group,
        "group_row_ordinal": ordinal,
        "semantic_reset_cluster_id": semantic,
        "body_id": body,
        "actor_contract_id": actor,
        "current_event": current,
        "post_event": post,
        "next_event": nxt,
        "success": binary["success"],
        "recovery": binary["recovery"],
        "duration": duration,
        "object_target": target,
        **booleans,
    }


def _validate_provider(
    raw: Mapping[str, Any], provider_id: str, labels: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != PROVIDER_KEYS:
        raise DualProviderRouterError(
            "provider prediction inventory changed; rank fields are forbidden"
        )
    if raw["provider_id"] != provider_id or raw["member_count"] != MEMBER_COUNT:
        raise DualProviderRouterError("provider identity/member count changed")
    artifact_sha = raw["provider_artifact_sha256"]
    if not _is_sha256(artifact_sha):
        raise DualProviderRouterError("provider artifact SHA must be lowercase SHA-256")
    manifest = _validate_provider_manifest(raw["provider_manifest"], provider_id, labels)
    if manifest["provider_artifact_sha256"] != artifact_sha:
        raise DualProviderRouterError("provider artifact differs from signed manifest")
    n = len(labels["sample_id"])
    for name in ("sample_id", "group_id", "group_row_ordinal"):
        value = np.asarray(raw[name])
        expected = labels[name]
        if name != "group_row_ordinal":
            value = value.astype(str)
        if not np.array_equal(value, expected):
            raise DualProviderRouterError(
                "providers and labels do not share sample/group/order"
            )
    masks = raw["applicable_masks"]
    expected_masks = _derived_masks(labels)
    if not isinstance(masks, Mapping) or set(masks) != set(HEADS):
        raise DualProviderRouterError("provider applicability inventory changed")
    normalized_masks = {}
    for head in HEADS:
        mask = _bool(masks[head], f"{provider_id}/{head} mask", (n,))
        if not np.array_equal(mask, expected_masks[head]):
            raise DualProviderRouterError("provider/label applicability mask mismatch")
        normalized_masks[head] = mask
    event_shape = (MEMBER_COUNT, n, len(EVENT_VOCAB))
    vector_shape = (MEMBER_COUNT, n)
    object_shape = (MEMBER_COUNT, n, labels["object_target"].shape[1])
    normalized = {
        "provider_id": provider_id,
        "provider_artifact_sha256": artifact_sha,
        "provider_manifest": manifest,
        "applicable_masks": normalized_masks,
        "post_event_logits": _prediction_float(raw["post_event_logits"], "post logits", event_shape),
        "next_event_logits": _prediction_float(raw["next_event_logits"], "next logits", event_shape),
        "success_logit": _prediction_float(raw["success_logit"], "success logit", vector_shape),
        "recovery_logit": _prediction_float(raw["recovery_logit"], "recovery logit", vector_shape),
        "duration_log_mean": _prediction_float(raw["duration_log_mean"], "duration mean", vector_shape),
        "duration_log_scale": _prediction_float(raw["duration_log_scale"], "duration scale", vector_shape),
        "object_mean": _prediction_float(raw["object_mean"], "object mean", object_shape),
        "object_log_scale": _prediction_float(raw["object_log_scale"], "object scale", object_shape),
    }
    if prediction_tensor_set_sha256(normalized) != manifest["prediction_tensor_sha256"]:
        raise DualProviderRouterError("provider manifest prediction tensor SHA mismatch")
    return normalized


def build_five_fold_group_plan(
    group_id: Sequence[str],
    semantic_reset_cluster_id: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    groups = np.asarray(group_id).astype(str)
    unique = sorted(set(groups.tolist()))
    semantic = (
        groups.copy()
        if semantic_reset_cluster_id is None
        else np.asarray(semantic_reset_cluster_id).astype(str)
    )
    if semantic.shape != groups.shape:
        raise DualProviderRouterError("semantic reset cluster shape changed")
    group_to_semantic = {}
    for group in unique:
        values = set(semantic[groups == group].tolist())
        if len(values) != 1:
            raise DualProviderRouterError("one execution group spans reset clusters")
        group_to_semantic[group] = next(iter(values))
    clusters = sorted(set(group_to_semantic.values()))
    if len(clusters) < FOLD_COUNT:
        raise DualProviderRouterError("at least five semantic reset clusters are required")
    cluster_owner = {name: index % FOLD_COUNT for index, name in enumerate(clusters)}
    result = []
    for fold in range(FOLD_COUNT):
        heldout = [
            group for group in unique
            if cluster_owner[group_to_semantic[group]] == fold
        ]
        training = sorted(set(unique) - set(heldout))
        result.append(
            {
                "fold_index": fold,
                "training_group_ids": training,
                "heldout_group_ids": heldout,
            }
        )
    return result


def _validate_folds(
    value: Sequence[Mapping[str, Any]], groups: np.ndarray, semantic: np.ndarray
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != FOLD_COUNT:
        raise DualProviderRouterError("exactly five group folds are required")
    all_groups = set(groups.tolist())
    owner: dict[str, int] = {}
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {
            "fold_index", "training_group_ids", "heldout_group_ids"
        }:
            raise DualProviderRouterError("fold fields changed")
        train, heldout = row["training_group_ids"], row["heldout_group_ids"]
        if (
            type(row["fold_index"]) is not int
            or row["fold_index"] != index
            or not isinstance(train, list)
            or not isinstance(heldout, list)
            or train != sorted(train)
            or heldout != sorted(heldout)
            or not heldout
            or len(set(train)) != len(train)
            or len(set(heldout)) != len(heldout)
            or set(train) & set(heldout)
            or set(train) | set(heldout) != all_groups
            or set(train) != all_groups - set(heldout)
        ):
            raise DualProviderRouterError("fold train/heldout membership changed")
        for group in heldout:
            if group in owner:
                raise DualProviderRouterError("heldout group is not exact-once")
            owner[group] = index
        result.append({"fold_index": index, "training_group_ids": train, "heldout_group_ids": heldout})
    if set(owner) != all_groups:
        raise DualProviderRouterError("heldout groups do not have exact-once coverage")
    semantic_owners: dict[str, set[int]] = defaultdict(set)
    for group, cluster in zip(groups.tolist(), semantic.tolist()):
        semantic_owners[cluster].add(owner[group])
    if any(len(indices) != 1 for indices in semantic_owners.values()):
        raise DualProviderRouterError("one semantic reset cluster crosses folds")
    expected = build_five_fold_group_plan(groups, semantic)
    if result != expected:
        raise DualProviderRouterError("fold plan is not the deterministic canonical plan")
    return result


def _support(
    head: str, labels: Mapping[str, np.ndarray], selected: np.ndarray
) -> dict[str, Any]:
    groups = labels["group_id"]
    clusters = labels["semantic_reset_cluster_id"]
    contexts = np.char.add(
        np.char.add(labels["body_id"].astype(str), "::"),
        labels["actor_contract_id"].astype(str),
    )
    mask = _derived_masks(labels)[head] & selected
    minimum = MINIMUM_SUPPORT[head]
    if head in {"post_event", "next_event"}:
        target = labels[head]
        categories = range(len(EVENT_VOCAB)) if head == "post_event" else range(1, len(EVENT_VOCAB))
        category_masks = {EVENT_VOCAB[index]: mask & (target == index) for index in categories}
    elif head in {"success", "recovery"}:
        target = labels[head]
        category_masks = {
            "positive": mask & (target == 1),
            "negative": mask & (target == 0),
        }
    elif head == "duration":
        observed = labels["duration_observed"]
        category_masks = {"observed": mask & observed, "censored": mask & ~observed}
    else:
        norm = np.linalg.norm(labels["object_target"], axis=1)
        category_masks = {
            "nonzero": mask & (norm > 1e-8),
            "near_zero": mask & (norm <= 1e-8),
        }
    def counts_for(scope: np.ndarray) -> dict[str, dict[str, int]]:
        return {
            category: {
                "independent_execution_groups": len(
                    set(groups[category_mask & scope].tolist())
                ),
                "independent_semantic_reset_clusters": len(
                    set(clusters[category_mask & scope].tolist())
                ),
            }
            for category, category_mask in category_masks.items()
        }
    global_counts = counts_for(np.ones(len(groups), dtype=bool))
    by_context = {
        context: counts_for(contexts == context)
        for context in sorted(set(contexts[selected].tolist()))
    }
    all_rows = list(global_counts.values()) + [
        counts for row in by_context.values() for counts in row.values()
    ]
    return {
        "minimum_per_category": minimum,
        "global": global_counts,
        "by_body_actor_context": by_context,
        "passed": all(
            counts["independent_execution_groups"] >= minimum
            and counts["independent_semantic_reset_clusters"] >= minimum
            for counts in all_rows
        ),
    }


def _fit_parameter(
    head: str,
    provider: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    training: np.ndarray,
) -> dict[str, Any]:
    groups = labels["group_id"]
    mask = _derived_masks(labels)[head] & training
    clusters = labels["semantic_reset_cluster_id"]
    weights = _group_row_weights(groups, mask, clusters)
    value: float | None
    if head in {"post_event", "next_event"}:
        required = range(len(EVENT_VOCAB)) if head == "post_event" else range(1, len(EVENT_VOCAB))
        if (
            len(set(clusters[mask].tolist())) < 2
            or any(not bool((mask & (labels[head] == category)).any()) for category in required)
        ):
            value = None
        else:
            grid = np.exp(np.linspace(-3.0, 3.0, 121))
            losses = []
            selected_logits = provider[f"{head}_logits"][:, mask]
            selected_target = labels[head][mask]
            selected_weights = weights[mask]
            for temperature in grid:
                probability = _softmax(
                    selected_logits / float(temperature)
                ).mean(axis=0)
                loss = -np.log(
                    np.clip(
                        probability[np.arange(len(selected_target)), selected_target],
                        1e-12, 1.0,
                    )
                )
                losses.append(float(np.sum(selected_weights * loss)))
            value = float(grid[int(np.argmin(losses))])
        return {"kind": "temperature", "value": 1.0 if value is None else float(value), "fit_passed": value is not None}
    if head in {"success", "recovery"}:
        if len(set(clusters[mask].tolist())) < 2 or len(set(labels[head][mask].tolist())) < 2:
            value = None
        else:
            grid = np.exp(np.linspace(-3.0, 3.0, 121))
            losses = []
            selected_logits = provider[f"{head}_logit"][:, mask]
            selected_target = labels[head][mask]
            selected_weights = weights[mask]
            for temperature in grid:
                probability = _sigmoid(
                    selected_logits / float(temperature)
                ).mean(axis=0)
                loss = -(
                    selected_target * np.log(np.clip(probability, 1e-12, 1.0))
                    + (1 - selected_target) * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
                )
                losses.append(float(np.sum(selected_weights * loss)))
            value = float(grid[int(np.argmin(losses))])
        return {"kind": "temperature", "value": 1.0 if value is None else float(value), "fit_passed": value is not None}
    if head == "duration":
        value = _fit_duration_scale(provider, labels, mask)
        return {"kind": "lognormal_scale_multiplier", "value": 1.0 if value is None else float(value), "fit_passed": value is not None}
    if len(set(clusters[mask].tolist())) < 2:
        value = None
    else:
        grid = np.exp(np.linspace(-2.0, 2.0, 81))
        selected_weights = weights[mask]
        losses = [
            float(
                np.sum(
                    selected_weights
                    * legacy._object_nll(
                        provider["object_mean"][:, mask],
                        provider["object_log_scale"][:, mask],
                        labels["object_target"][mask], float(scale),
                    )
                )
            )
            for scale in grid
        ]
        value = float(grid[int(np.argmin(losses))])
    robust = _object_uncertainty_scale(labels, mask)
    return {
        "kind": "gaussian_scale_multiplier",
        "value": 1.0 if value is None else float(value),
        "uncertainty_robust_scale": 1.0 if robust is None else float(robust),
        "fit_passed": value is not None and robust is not None,
    }


def _normal_survival(z: np.ndarray) -> np.ndarray:
    flat = np.fromiter(
        (0.5 * math.erfc(float(item) / math.sqrt(2.0)) for item in z.flat),
        dtype=np.float64,
        count=z.size,
    )
    return flat.reshape(z.shape)


def _duration_loss(
    mean: np.ndarray,
    log_scale: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    adjusted = np.clip(log_scale + math.log(multiplier), -8.0, 5.0)
    scale = np.exp(adjusted)
    log_target = np.log1p(target)
    z = (log_target[None, :] - mean) / scale
    log_pdf = (
        -0.5 * np.square(z)
        - adjusted
        - 0.5 * math.log(2.0 * math.pi)
        - log_target[None, :]
    )
    observed_nll = -(legacy._logsumexp(log_pdf, axis=0) - math.log(MEMBER_COUNT))
    survival = np.clip(_normal_survival(z).mean(axis=0), 1e-12, 1.0)
    censored_nll = -np.log(survival)
    return np.where(observed, observed_nll, censored_nll)


def _fit_duration_scale(
    provider: Mapping[str, Any], labels: Mapping[str, np.ndarray], mask: np.ndarray
) -> float | None:
    if len(set(labels["semantic_reset_cluster_id"][mask].tolist())) < 2:
        return None
    weights = _group_row_weights(
        labels["group_id"], mask, labels["semantic_reset_cluster_id"]
    )[mask]
    grid = np.exp(np.linspace(-2.0, 2.0, 81))
    losses = [
        float(
            np.sum(
                weights * _duration_loss(
                    provider["duration_log_mean"][:, mask],
                    provider["duration_log_scale"][:, mask],
                    labels["duration"][mask], labels["duration_observed"][mask],
                    float(scale),
                )
            )
        )
        for scale in grid
    ]
    return float(grid[int(np.argmin(losses))])


def _object_uncertainty_scale(
    labels: Mapping[str, np.ndarray], mask: np.ndarray
) -> float | None:
    if not bool(mask.any()):
        return None
    target = labels["object_target"][mask]
    weights = _group_row_weights(
        labels["group_id"], mask, labels["semantic_reset_cluster_id"]
    )[mask]
    center = np.asarray(
        [_weighted_median(target[:, axis], weights) for axis in range(target.shape[1])]
    )
    scale = _weighted_median(np.linalg.norm(target - center, axis=1), weights)
    return max(scale, 1e-8)


def _evaluate_provider(
    head: str,
    provider: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    parameter: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    n = len(labels["sample_id"])
    if head in {"post_event", "next_event"}:
        member = _softmax(provider[f"{head}_logits"] / float(parameter["value"]))
        probability = member.mean(axis=0)
        target = labels[head]
        proper = -np.log(np.clip(probability[np.arange(n), target], 1e-12, 1.0))
        decision = (probability.argmax(axis=1) != target).astype(np.float64)
        uncertainty = _entropy(probability) / math.log(probability.shape[1])
        one_hot = np.eye(probability.shape[1])[target]
        secondary = np.square(probability - one_hot).sum(axis=1)
        coverage = np.full(n, np.nan)
    elif head in {"success", "recovery"}:
        member = _sigmoid(provider[f"{head}_logit"] / float(parameter["value"]))
        probability = member.mean(axis=0)
        target = labels[head]
        proper = -(
            target * np.log(np.clip(probability, 1e-12, 1.0))
            + (1 - target) * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
        )
        decision = ((probability >= 0.5) != target).astype(np.float64)
        aleatoric = (member * (1.0 - member)).mean(axis=0)
        epistemic = member.var(axis=0)
        uncertainty = np.clip((aleatoric + epistemic) / 0.25, 0.0, 4.0)
        secondary = np.square(probability - target)
        coverage = np.full(n, np.nan)
    elif head == "duration":
        multiplier = float(parameter["value"])
        adjusted = provider["duration_log_scale"] + math.log(multiplier)
        scale = np.exp(np.clip(adjusted, -8.0, 5.0))
        median = legacy._mixture_lognormal_quantile(
            provider["duration_log_mean"], scale, 0.5
        )
        proper = _duration_loss(
            provider["duration_log_mean"], provider["duration_log_scale"],
            labels["duration"], labels["duration_observed"], multiplier,
        )
        observed_error = np.abs(np.log1p(median) - np.log1p(labels["duration"]))
        censored_error = np.maximum(labels["duration"] - median, 0.0) / np.maximum(
            1.0, labels["duration"]
        )
        decision = np.where(labels["duration_observed"], observed_error, censored_error)
        member_mean = np.exp(
            np.clip(provider["duration_log_mean"] + 0.5 * scale**2, -30, 30)
        ) - 1.0
        member_var = (np.exp(np.clip(scale**2, 0, 30)) - 1.0) * np.exp(
            np.clip(2.0 * provider["duration_log_mean"] + scale**2, -30, 30)
        )
        total = member_var.mean(axis=0) + member_mean.var(axis=0)
        relative = np.sqrt(np.maximum(total, 0.0)) / np.maximum(median, 1e-8)
        uncertainty = relative / (1.0 + relative)
        lower = legacy._mixture_lognormal_quantile(
            provider["duration_log_mean"], scale, 0.05
        )
        upper = legacy._mixture_lognormal_quantile(
            provider["duration_log_mean"], scale, 0.95
        )
        target = labels["duration"]
        secondary = (upper - lower) + 20.0 * (
            np.maximum(lower - target, 0.0) + np.maximum(target - upper, 0.0)
        )
        coverage = np.where(
            labels["duration_observed"],
            ((target >= lower) & (target <= upper)).astype(float),
            np.nan,
        )
    else:
        multiplier = float(parameter["value"])
        mean = provider["object_mean"].mean(axis=0)
        proper = legacy._object_nll(
            provider["object_mean"], provider["object_log_scale"],
            labels["object_target"], multiplier,
        )
        decision = np.linalg.norm(mean - labels["object_target"], axis=1)
        variance = np.exp(
            np.clip(2.0 * (provider["object_log_scale"] + math.log(multiplier)), -20, 20)
        ).mean(axis=0) + provider["object_mean"].var(axis=0)
        total_std = np.sqrt(np.maximum(variance.mean(axis=1), 0.0))
        robust = float(parameter["uncertainty_robust_scale"])
        uncertainty = total_std / (robust + total_std)
        lower = mean - 1.6448536269514722 * np.sqrt(np.maximum(variance, 0.0))
        upper = mean + 1.6448536269514722 * np.sqrt(np.maximum(variance, 0.0))
        target = labels["object_target"]
        secondary = (
            (upper - lower)
            + 20.0 * (np.maximum(lower - target, 0.0) + np.maximum(target - upper, 0.0))
        ).mean(axis=1)
        coverage = ((target >= lower) & (target <= upper)).mean(axis=1)
    return {
        "proper": proper,
        "decision": decision,
        "uncertainty": uncertainty,
        "calibration_secondary": secondary,
        "interval_coverage": coverage,
    }


def _fit_baseline(
    head: str, labels: Mapping[str, np.ndarray], training: np.ndarray
) -> dict[str, Any]:
    mask = _derived_masks(labels)[head] & training
    groups = labels["group_id"]
    weights = _group_row_weights(
        groups, mask, labels["semantic_reset_cluster_id"]
    )
    if head == "post_event":
        return {"kind": "persistence", "epsilon": 0.01}
    if head == "next_event":
        counts = np.asarray(
            [float(weights[(labels[head] == index) & mask].sum()) for index in range(len(EVENT_VOCAB))]
        )
        counts += 1e-3
        counts[0] = 1e-9
        counts /= counts.sum()
        return {"kind": "train_group_prior", "probability": counts.tolist()}
    if head in {"success", "recovery"}:
        prevalence = float(np.sum(weights * labels[head]))
        return {"kind": "train_group_prevalence", "probability": float(np.clip(prevalence, 1e-4, 1 - 1e-4))}
    if head == "duration":
        centers: dict[str, float] = {}
        scales: dict[str, float] = {}
        observed = mask & labels["duration_observed"]
        global_log = np.log1p(labels["duration"][observed])
        global_weights = _group_row_weights(
            groups, observed, labels["semantic_reset_cluster_id"]
        )[observed]
        global_center = _weighted_median(global_log, global_weights) if len(global_log) else 0.0
        global_scale = max(
            _weighted_median(np.abs(global_log - global_center), global_weights)
            if len(global_log) else 1.0,
            0.1,
        )
        for event in range(len(EVENT_VOCAB)):
            selected = observed & (labels["current_event"] == event)
            values = np.log1p(labels["duration"][selected])
            event_weights = _group_row_weights(
                groups, selected, labels["semantic_reset_cluster_id"]
            )[selected]
            center = _weighted_median(values, event_weights) if len(values) else global_center
            scale = max(
                _weighted_median(np.abs(values - center), event_weights)
                if len(values) else global_scale,
                0.1,
            )
            centers[str(event)], scales[str(event)] = center, scale
        event = labels["current_event"][mask]
        base_center = np.asarray([centers[str(int(value))] for value in event])
        log_target = np.log1p(labels["duration"][mask])
        selected_observed = labels["duration_observed"][mask]
        selected_weights = weights[mask]
        best = None
        for offset in np.linspace(0.0, 1.5, 16):
            for scale_value in np.linspace(0.1, 1.5, 15):
                center = base_center + float(offset)
                z = (log_target - center) / float(scale_value)
                proper = np.where(
                    selected_observed,
                    0.5 * np.square(z) + math.log(float(scale_value))
                    + 0.5 * math.log(2 * math.pi) + log_target,
                    -np.log(np.clip(_normal_survival(z), 1e-12, 1.0)),
                )
                score = float(np.sum(selected_weights * proper))
                candidate = (score, float(offset), float(scale_value))
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        return {
            "kind": "train_current_event_lognormal_censor_aware",
            "centers": centers,
            "scales": scales,
            "proper_center_offset": best[1],
            "proper_scale": best[2],
        }
    target = labels["object_target"][mask]
    if len(target) == 0:
        dimension = labels["object_target"].shape[1]
        return {
            "kind": "empty_support_safe_zero_gaussian_disabled_only",
            "proper_center": [0.0] * dimension,
            "proper_scale": [1.0] * dimension,
            "decision_center": [0.0] * dimension,
        }
    selected_weights = weights[mask]
    weighted_center = np.asarray(
        [_weighted_median(target[:, axis], selected_weights) for axis in range(target.shape[1])]
    )
    center_candidates = [np.zeros(target.shape[1]), weighted_center]
    best_proper = None
    best_decision = None
    for center in center_candidates:
        residual = target - center
        scale = np.maximum(
            np.asarray(
                [
                    _weighted_median(np.abs(residual[:, axis]), selected_weights)
                    for axis in range(target.shape[1])
                ]
            ),
            1e-3,
        )
        proper = (0.5 * np.square(residual / scale) + np.log(scale)).sum(axis=1)
        decision = np.linalg.norm(residual, axis=1)
        proper_candidate = (float(np.sum(selected_weights * proper)), center, scale)
        decision_candidate = (float(np.sum(selected_weights * decision)), center)
        if best_proper is None or proper_candidate[0] < best_proper[0]:
            best_proper = proper_candidate
        if best_decision is None or decision_candidate[0] < best_decision[0]:
            best_decision = decision_candidate
    assert best_proper is not None and best_decision is not None
    return {
        "kind": "train_zero_or_median_gaussian_separate_proper_decision",
        "proper_center": best_proper[1].tolist(),
        "proper_scale": best_proper[2].tolist(),
        "decision_center": best_decision[1].tolist(),
    }


def _evaluate_baseline(
    head: str, labels: Mapping[str, np.ndarray], baseline: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    n = len(labels["sample_id"])
    if head in {"post_event", "next_event"}:
        if head == "post_event":
            probability = np.full((n, len(EVENT_VOCAB)), 0.01 / (len(EVENT_VOCAB) - 1))
            probability[np.arange(n), labels["current_event"]] = 0.99
        else:
            probability = np.tile(np.asarray(baseline["probability"]), (n, 1))
        target = labels[head]
        proper = -np.log(np.clip(probability[np.arange(n), target], 1e-12, 1.0))
        decision = (probability.argmax(axis=1) != target).astype(np.float64)
    elif head in {"success", "recovery"}:
        probability = float(baseline["probability"])
        target = labels[head]
        proper = -(target * math.log(probability) + (1 - target) * math.log(1 - probability))
        decision = ((probability >= 0.5) != target).astype(np.float64)
    elif head == "duration":
        event = labels["current_event"]
        decision_center = np.asarray([baseline["centers"][str(int(value))] for value in event])
        center = decision_center + float(baseline["proper_center_offset"])
        scale = np.full(n, float(baseline["proper_scale"]))
        log_target = np.log1p(labels["duration"])
        z = (log_target - center) / scale
        observed = 0.5 * np.square(z) + np.log(scale) + 0.5 * math.log(2 * math.pi) + log_target
        censored = -np.log(np.clip(_normal_survival(z), 1e-12, 1.0))
        proper = np.where(labels["duration_observed"], observed, censored)
        median = np.maximum(np.exp(decision_center) - 1.0, 0.0)
        observed_error = np.abs(np.log1p(median) - log_target)
        censored_error = np.maximum(labels["duration"] - median, 0.0) / np.maximum(1.0, labels["duration"])
        decision = np.where(labels["duration_observed"], observed_error, censored_error)
    else:
        proper_center = np.asarray(baseline["proper_center"])
        scale = np.asarray(baseline["proper_scale"])
        residual = labels["object_target"] - proper_center
        proper = (0.5 * np.square(residual / scale) + np.log(scale) + 0.5 * math.log(2 * math.pi)).sum(axis=1)
        decision = np.linalg.norm(
            labels["object_target"] - np.asarray(baseline["decision_center"]), axis=1
        )
    return {"proper": np.asarray(proper), "decision": np.asarray(decision)}


def _empty_oof(n: int) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    return {
        provider: {
            head: {
                key: np.full(n, np.nan, dtype=np.float64)
                for key in (
                    "proper", "decision", "uncertainty",
                    "calibration_secondary", "interval_coverage",
                )
            }
            for head in HEADS
        }
        for provider in PROVIDERS
    }


def _empty_baseline_oof(n: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        head: {key: np.full(n, np.nan) for key in ("proper", "decision")}
        for head in HEADS
    }


def _crossfit_on_group_scope(
    decoded: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, np.ndarray],
    scope_group_ids: Sequence[str],
) -> dict[str, Any]:
    """Produce inner-OOF predictions; every fit sees only its inner-train groups."""

    scope_groups = sorted(set(str(value) for value in scope_group_ids))
    scope_rows = np.isin(labels["group_id"], scope_groups)
    plan = build_five_fold_group_plan(
        labels["group_id"][scope_rows],
        labels["semantic_reset_cluster_id"][scope_rows],
    )
    n = len(labels["sample_id"])
    oof = _empty_oof(n)
    uncalibrated_oof = _empty_oof(n)
    baseline_oof = _empty_baseline_oof(n)
    owner = np.full(n, -1, dtype=np.int64)
    support_by_head = {head: [] for head in HEADS}
    fit_by_provider = {
        provider: {head: [] for head in HEADS} for provider in PROVIDERS
    }
    receipts = []
    for fold in plan:
        training = np.isin(labels["group_id"], fold["training_group_ids"])
        heldout = np.isin(labels["group_id"], fold["heldout_group_ids"])
        parameters = {provider: {} for provider in PROVIDERS}
        support_rows = {}
        for head in HEADS:
            support = _support(head, labels, training)
            support_rows[head] = support
            support_by_head[head].append(bool(support["passed"]))
            baseline_parameter = _fit_baseline(head, labels, training)
            baseline_eval = _evaluate_baseline(head, labels, baseline_parameter)
            for key in ("proper", "decision"):
                baseline_oof[head][key][heldout] = baseline_eval[key][heldout]
            for provider in PROVIDERS:
                parameter = _fit_parameter(head, decoded[provider], labels, training)
                parameters[provider][head] = parameter
                fit_by_provider[provider][head].append(bool(parameter["fit_passed"]))
                evaluation = _evaluate_provider(
                    head, decoded[provider], labels, parameter
                )
                uncalibrated_parameter = dict(parameter)
                uncalibrated_parameter["value"] = 1.0
                raw_evaluation = _evaluate_provider(
                    head, decoded[provider], labels, uncalibrated_parameter
                )
                for key in (
                    "proper", "decision", "uncertainty",
                    "calibration_secondary", "interval_coverage",
                ):
                    oof[provider][head][key][heldout] = evaluation[key][heldout]
                    uncalibrated_oof[provider][head][key][heldout] = raw_evaluation[key][heldout]
        if bool((owner[heldout] != -1).any()):
            raise DualProviderRouterError("inner heldout sample predicted more than once")
        owner[heldout] = int(fold["fold_index"])
        receipts.append(
            {
                "inner_fold_index": int(fold["fold_index"]),
                "training_group_ids_sha256": canonical_sha256(fold["training_group_ids"]),
                "heldout_group_ids_sha256": canonical_sha256(fold["heldout_group_ids"]),
                "inner_heldout_labels_used_for_fit": False,
                "provider_parameters": parameters,
                "training_support_by_head": support_rows,
                "heldout_inference_calls_per_provider": {
                    provider: 1 for provider in PROVIDERS
                },
            }
        )
    scope = np.isin(labels["group_id"], scope_groups)
    if not np.all(owner[scope] >= 0) or bool((owner[~scope] >= 0).any()):
        raise DualProviderRouterError("inner OOF scope is not exact-once")
    return {
        "provider_oof": oof,
        "uncalibrated_provider_oof": uncalibrated_oof,
        "baseline_oof": baseline_oof,
        "support_by_head": support_by_head,
        "fit_by_provider": fit_by_provider,
        "inner_fold_receipts": receipts,
        "scope": scope,
    }


def _select_routes_from_oof(
    provider_oof: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    uncalibrated_provider_oof: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    baseline_oof: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    scope: np.ndarray,
    support_by_head: Mapping[str, Sequence[bool]],
    fit_by_provider: Mapping[str, Mapping[str, Sequence[bool]]],
    *,
    bootstrap_samples: int,
    role_prefix: str,
) -> dict[str, dict[str, Any]]:
    routes = {}
    masks = _derived_masks(labels)
    for head in HEADS:
        mask = masks[head] & scope
        provider_evidence = {}
        effective_oof = {}
        for provider in PROVIDERS:
            calibration_quality = _calibration_quality(
                head, provider_oof[provider][head],
                uncalibrated_provider_oof[provider][head], labels, mask,
                bootstrap_samples=bootstrap_samples,
                role=f"{role_prefix}:{head}:{provider}:calibration",
            )
            effective_oof[provider] = (
                provider_oof[provider][head]
                if calibration_quality["selected_mode"] == "fitted_parameter"
                else uncalibrated_provider_oof[provider][head]
            )
            proper_gain = _gain_evidence(
                effective_oof[provider]["proper"], baseline_oof[head]["proper"],
                labels["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
                execution_groups=labels["group_id"],
                role=f"{role_prefix}:{head}:{provider}:baseline:proper",
            )
            decision_gain = _gain_evidence(
                effective_oof[provider]["decision"], baseline_oof[head]["decision"],
                labels["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
                execution_groups=labels["group_id"],
                role=f"{role_prefix}:{head}:{provider}:baseline:decision",
            )
            uncertainty = _uncertainty_evidence(
                effective_oof[provider]["uncertainty"],
                effective_oof[provider]["decision"], labels["semantic_reset_cluster_id"], mask,
                samples=bootstrap_samples, role=f"{role_prefix}:{head}:{provider}",
                execution_groups=labels["group_id"],
            )
            row = {
                "all_crossfit_train_folds_support_passed": all(support_by_head[head]),
                "all_crossfit_train_calibration_fits_passed": all(
                    fit_by_provider[provider][head]
                ),
                "proper_loss_gain_vs_train_only_baseline": proper_gain,
                "decision_loss_gain_vs_train_only_baseline": decision_gain,
                "uncertainty_aurc_gate": uncertainty,
                "calibration_quality": calibration_quality,
            }
            row["gate_passed"] = bool(
                row["all_crossfit_train_folds_support_passed"]
                and row["all_crossfit_train_calibration_fits_passed"]
                and proper_gain["lcb95"] is not None
                and proper_gain["lcb95"] >= 0.0
                and decision_gain["lcb95"] is not None
                and decision_gain["lcb95"] >= 0.0
                and uncertainty["passed"]
                and calibration_quality["passed"]
            )
            provider_evidence[provider] = row
        paired_proper = _gain_evidence(
            effective_oof[CANDIDATE_PROVIDER]["proper"],
            effective_oof[REFERENCE_PROVIDER]["proper"], labels["semantic_reset_cluster_id"], mask,
            samples=bootstrap_samples, role=f"{role_prefix}:{head}:paired:proper",
            execution_groups=labels["group_id"],
        )
        paired_decision = _gain_evidence(
            effective_oof[CANDIDATE_PROVIDER]["decision"],
            effective_oof[REFERENCE_PROVIDER]["decision"], labels["semantic_reset_cluster_id"], mask,
            samples=bootstrap_samples, role=f"{role_prefix}:{head}:paired:decision",
            execution_groups=labels["group_id"],
        )
        harm = _harm_evidence(
            effective_oof[CANDIDATE_PROVIDER]["proper"],
            effective_oof[CANDIDATE_PROVIDER]["decision"],
            effective_oof[REFERENCE_PROVIDER]["proper"],
            effective_oof[REFERENCE_PROVIDER]["decision"],
            labels["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"{role_prefix}:{head}:harm",
            execution_groups=labels["group_id"],
        )
        candidate_selected = bool(
            provider_evidence[CANDIDATE_PROVIDER]["gate_passed"]
            and provider_evidence[REFERENCE_PROVIDER]["gate_passed"]
            and paired_proper["lcb95"] is not None and paired_proper["lcb95"] > 0.0
            and paired_decision["lcb95"] is not None and paired_decision["lcb95"] > 0.0
            and harm["both_harmful_rate_ucb_passed"]
        )
        if candidate_selected:
            selected_provider, status = CANDIDATE_PROVIDER, "selected_body_conditioned_adapter"
        elif provider_evidence[REFERENCE_PROVIDER]["gate_passed"]:
            selected_provider, status = REFERENCE_PROVIDER, "fallback_body_agnostic_adapter"
        else:
            selected_provider, status = None, "head_disabled_actor_baseline"
        routes[head] = {
            "status": status,
            "selected_provider": selected_provider,
            "provider_evidence": provider_evidence,
            "paired_provider_proper_loss_gain": paired_proper,
            "paired_provider_decision_loss_gain": paired_decision,
            "paired_provider_harmful_rate": harm,
        }
    return routes


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.kind in "OUS":
        payload = canonical_bytes(
            {"dtype": "unicode", "shape": list(array.shape), "values": array.astype(str).tolist()}
        )
    else:
        contiguous = np.ascontiguousarray(array)
        header = canonical_bytes({"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)})
        payload = header + b"\x00" + contiguous.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _prediction_content_sha(provider: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            key: _ndarray_sha256(value)
            for key, value in provider.items()
            if isinstance(value, np.ndarray)
        }
    )


def _oof_content_sha(value: Mapping[str, Any]) -> str:
    rows = {}
    for outer, heads in value.items():
        rows[outer] = {
            head: {metric: _ndarray_sha256(array) for metric, array in metrics.items()}
            for head, metrics in heads.items()
        }
    return canonical_sha256(rows)


def _calibration_quality(
    head: str,
    calibrated: Mapping[str, np.ndarray],
    uncalibrated: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    bootstrap_samples: int,
    role: str,
    allow_identity_fallback: bool = True,
) -> dict[str, Any]:
    clusters = labels["semantic_reset_cluster_id"]
    executions = labels["group_id"]
    proper = _gain_evidence(
        calibrated["proper"], uncalibrated["proper"], clusters, mask,
        execution_groups=executions, samples=bootstrap_samples, role=f"{role}:proper",
    )
    secondary_mask = mask.copy()
    if head == "duration":
        secondary_mask &= labels["duration_observed"]
    secondary = _gain_evidence(
        calibrated["calibration_secondary"],
        uncalibrated["calibration_secondary"], clusters, secondary_mask,
        execution_groups=executions, samples=bootstrap_samples,
        role=f"{role}:secondary",
    )
    result = {
        "proper_loss_noninferiority": proper,
        "secondary_score_noninferiority": secondary,
        "secondary_score": (
            "multiclass_brier" if head in {"post_event", "next_event"}
            else "binary_brier" if head in {"success", "recovery"}
            else "lognormal_mixture_q05_q95_interval_score"
            if head == "duration"
            else "moment_matched_gaussian_q05_q95_interval_score"
        ),
    }
    coverage_passed = True
    identity_coverage_passed = True
    if head in {"duration", "object_effect"}:
        names, values = _group_means(
            calibrated["interval_coverage"], clusters, secondary_mask, executions
        )
        interval = _bootstrap_interval(
            values, samples=bootstrap_samples, role=f"{role}:coverage"
        )
        coverage_error_ucb = (
            None if interval["lcb95"] is None or interval["ucb95"] is None
            else max(abs(interval["lcb95"] - 0.90), abs(interval["ucb95"] - 0.90))
        )
        coverage_passed = bool(
            interval["lcb95"] is not None
            and coverage_error_ucb is not None
            and coverage_error_ucb <= MAXIMUM_ABSOLUTE_COVERAGE_ERROR + 1e-12
        )
        result["nominal_90pct_interval_coverage"] = {
            **interval,
            "nominal": 0.90,
            "absolute_coverage_error_ucb95": coverage_error_ucb,
            "maximum_absolute_coverage_error": MAXIMUM_ABSOLUTE_COVERAGE_ERROR,
            "coverage_gate_passed": coverage_passed,
            "semantic_cluster_ids_sha256": canonical_sha256(names.tolist()),
        }
        raw_names, raw_values = _group_means(
            uncalibrated["interval_coverage"], clusters, secondary_mask, executions
        )
        raw_interval = _bootstrap_interval(
            raw_values, samples=bootstrap_samples, role=f"{role}:identity_coverage"
        )
        raw_coverage_error_ucb = (
            None if raw_interval["lcb95"] is None or raw_interval["ucb95"] is None
            else max(
                abs(raw_interval["lcb95"] - 0.90),
                abs(raw_interval["ucb95"] - 0.90),
            )
        )
        identity_coverage_passed = bool(
            raw_interval["lcb95"] is not None
            and raw_coverage_error_ucb is not None
            and raw_coverage_error_ucb <= MAXIMUM_ABSOLUTE_COVERAGE_ERROR + 1e-12
        )
        result["identity_nominal_90pct_interval_coverage"] = {
            **raw_interval,
            "nominal": 0.90,
            "absolute_coverage_error_ucb95": raw_coverage_error_ucb,
            "maximum_absolute_coverage_error": MAXIMUM_ABSOLUTE_COVERAGE_ERROR,
            "coverage_gate_passed": identity_coverage_passed,
            "semantic_cluster_ids_sha256": canonical_sha256(raw_names.tolist()),
        }
    fitted_passed = bool(
        proper["lcb95"] is not None and proper["lcb95"] >= -1e-12
        and secondary["lcb95"] is not None and secondary["lcb95"] >= -1e-12
        and coverage_passed
    )
    identity_passed = bool(identity_coverage_passed)
    result["fitted_parameter_passed"] = fitted_passed
    result["identity_fallback_passed"] = identity_passed
    result["selected_mode"] = (
        "fitted_parameter" if fitted_passed else "identity_parameter_fallback"
        if identity_passed else "disabled"
    )
    result["identity_fallback_selection_allowed_at_this_layer"] = allow_identity_fallback
    if not allow_identity_fallback:
        result["selected_mode"] = "locked_routed_parameter"
    result["passed"] = fitted_passed or (allow_identity_fallback and identity_passed)
    if not allow_identity_fallback:
        result["locked_route_quality_passed"] = result["passed"]
    return result


def calibrate_dual_provider_router(
    providers: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Run strict outer/inner nested OOF routing and full-development refits."""

    if set(providers) != set(PROVIDERS):
        raise DualProviderRouterError("exact canonical adapter providers required")
    if type(bootstrap_samples) is not int or bootstrap_samples < 200:
        raise DualProviderRouterError("bootstrap sample count is too small")
    y = _validate_labels(labels)
    decoded = {
        name: _validate_provider(providers[name], name, y) for name in PROVIDERS
    }
    reference_manifest = decoded[REFERENCE_PROVIDER]["provider_manifest"]
    candidate_manifest = decoded[CANDIDATE_PROVIDER]["provider_manifest"]
    for field in (
        "shared_core_lineage_sha256",
        "training_execution_group_ids",
        "training_semantic_reset_cluster_ids",
    ):
        if reference_manifest[field] != candidate_manifest[field]:
            raise DualProviderRouterError("provider manifests do not share frozen lineage/training identities")
    if [
        (row["member_index"], row["seed"]) for row in reference_manifest["members"]
    ] != [
        (row["member_index"], row["seed"]) for row in candidate_manifest["members"]
    ]:
        raise DualProviderRouterError("provider member index/seed pairing changed")
    if decoded[REFERENCE_PROVIDER]["provider_artifact_sha256"] == decoded[CANDIDATE_PROVIDER]["provider_artifact_sha256"]:
        raise DualProviderRouterError("two provider artifacts must be distinct")
    split = _validate_folds(
        folds, y["group_id"], y["semantic_reset_cluster_id"]
    )
    n = len(y["sample_id"])
    masks = _derived_masks(y)
    stitched = _empty_oof(n)
    fitted_stitched = _empty_oof(n)
    uncalibrated_stitched = _empty_oof(n)
    baseline_stitched = _empty_baseline_oof(n)
    routed_stitched = {
        head: {
            key: np.full(n, np.nan)
            for key in (
                "proper", "decision", "uncertainty",
                "calibration_secondary", "interval_coverage",
            )
        }
        for head in HEADS
    }
    routed_uncalibrated_stitched = {
        head: {
            key: np.full(n, np.nan)
            for key in (
                "proper", "decision", "uncertainty",
                "calibration_secondary", "interval_coverage",
            )
        }
        for head in HEADS
    }
    owner = np.full(n, -1, dtype=np.int64)
    fold_receipts = []
    per_fold_support: dict[str, list[bool]] = {head: [] for head in HEADS}
    fit_by_provider: dict[str, dict[str, list[bool]]] = {
        provider: {head: [] for head in HEADS} for provider in PROVIDERS
    }
    for fold in split:
        training = np.isin(y["group_id"], fold["training_group_ids"])
        heldout = np.isin(y["group_id"], fold["heldout_group_ids"])
        if bool((owner[heldout] != -1).any()):
            raise DualProviderRouterError("heldout sample predicted more than once")
        inner = _crossfit_on_group_scope(decoded, y, fold["training_group_ids"])
        inner_routes = _select_routes_from_oof(
            inner["provider_oof"], inner["uncalibrated_provider_oof"],
            inner["baseline_oof"], y, inner["scope"],
            inner["support_by_head"], inner["fit_by_provider"],
            bootstrap_samples=bootstrap_samples, role_prefix=f"outer{fold['fold_index']}:inner",
        )
        parameters: dict[str, dict[str, Any]] = {provider: {} for provider in PROVIDERS}
        fitted_parameters: dict[str, dict[str, Any]] = {provider: {} for provider in PROVIDERS}
        support_rows: dict[str, Any] = {}
        for head in HEADS:
            support = _support(head, y, training)
            support_rows[head] = support
            per_fold_support[head].append(bool(support["passed"]))
            baseline_parameter = _fit_baseline(head, y, training)
            baseline_eval = _evaluate_baseline(head, y, baseline_parameter)
            provider_eval = {}
            for provider in PROVIDERS:
                parameter = _fit_parameter(head, decoded[provider], y, training)
                fitted_parameters[provider][head] = dict(parameter)
                fit_by_provider[provider][head].append(bool(parameter["fit_passed"]))
                calibration_mode = inner_routes[head]["provider_evidence"][provider]["calibration_quality"]["selected_mode"]
                fitted_eval = _evaluate_provider(head, decoded[provider], y, parameter)
                effective_parameter = dict(parameter)
                if calibration_mode == "identity_parameter_fallback":
                    effective_parameter["value"] = 1.0
                    effective_parameter["identity_parameter_sentinel"] = (
                        "exact_identity_value_one_selected_by_inner_oof"
                    )
                else:
                    effective_parameter["identity_parameter_sentinel"] = None
                effective_parameter["crossfit_selected_calibration_mode"] = calibration_mode
                parameters[provider][head] = effective_parameter
                provider_eval[provider] = _evaluate_provider(
                    head, decoded[provider], y, effective_parameter
                )
                if calibration_mode == "disabled":
                    provider_eval[provider] = {
                        key: np.full(n, np.nan) for key in provider_eval[provider]
                    }
                raw_parameter = dict(parameter)
                raw_parameter["value"] = 1.0
                raw_eval = _evaluate_provider(head, decoded[provider], y, raw_parameter)
                for key in (
                    "proper", "decision", "uncertainty",
                    "calibration_secondary", "interval_coverage",
                ):
                    stitched[provider][head][key][heldout] = provider_eval[provider][key][heldout]
                    fitted_stitched[provider][head][key][heldout] = fitted_eval[key][heldout]
                    uncalibrated_stitched[provider][head][key][heldout] = raw_eval[key][heldout]
            baseline_stitched[head]["proper"][heldout] = baseline_eval["proper"][heldout]
            baseline_stitched[head]["decision"][heldout] = baseline_eval["decision"][heldout]
            selected_provider = inner_routes[head]["selected_provider"]
            source = baseline_eval if selected_provider is None else provider_eval[selected_provider]
            raw_source = baseline_eval if selected_provider is None else _evaluate_provider(
                head,
                decoded[selected_provider],
                y,
                {**parameters[selected_provider][head], "value": 1.0},
            )
            for key in ("proper", "decision"):
                routed_stitched[head][key][heldout] = source[key][heldout]
                routed_uncalibrated_stitched[head][key][heldout] = raw_source[key][heldout]
            routed_stitched[head]["uncertainty"][heldout] = (
                np.nan if selected_provider is None else source["uncertainty"][heldout]
            )
            routed_uncalibrated_stitched[head]["uncertainty"][heldout] = (
                np.nan if selected_provider is None else raw_source["uncertainty"][heldout]
            )
            if selected_provider is not None:
                for key in ("calibration_secondary", "interval_coverage"):
                    routed_stitched[head][key][heldout] = source[key][heldout]
                    routed_uncalibrated_stitched[head][key][heldout] = raw_source[key][heldout]
        owner[heldout] = int(fold["fold_index"])
        fold_receipts.append(
            {
                "fold_index": int(fold["fold_index"]),
                "training_group_ids_sha256": canonical_sha256(fold["training_group_ids"]),
                "heldout_group_ids_sha256": canonical_sha256(fold["heldout_group_ids"]),
                "outer_heldout_labels_used_for_calibration_or_provider_selection": False,
                "provider_parameters": parameters,
                "outer_train_fitted_parameters": fitted_parameters,
                "outer_train_effective_parameters": parameters,
                "outer_train_fitted_parameters_sha256": canonical_sha256(fitted_parameters),
                "outer_train_effective_parameters_sha256": canonical_sha256(parameters),
                "inner_oof_selected_route_by_head": {
                    head: inner_routes[head]["selected_provider"] for head in HEADS
                },
                "inner_oof_route_evidence": inner_routes,
                "inner_fold_receipts": inner["inner_fold_receipts"],
                "training_support_by_head": support_rows,
                "heldout_inference_calls_per_provider": {
                    REFERENCE_PROVIDER: 1,
                    CANDIDATE_PROVIDER: 1,
                },
                "heldout_scored_once": True,
            }
        )
        fold_receipts[-1]["route_decision_sha256"] = canonical_sha256(
            {
                "selected": fold_receipts[-1]["inner_oof_selected_route_by_head"],
                "evidence": inner_routes,
                "provider_parameters": parameters,
                "fitted_parameters": fitted_parameters,
            }
        )
    if not np.array_equal(owner >= 0, np.ones(n, dtype=bool)):
        raise DualProviderRouterError("OOF sample coverage is incomplete")
    full_parameters = {
        provider: {
            head: _fit_parameter(head, decoded[provider], y, np.ones(n, dtype=bool))
            for head in HEADS
        }
        for provider in PROVIDERS
    }
    parameter_sha_by_provider = {
        provider: {
            head: canonical_sha256(
                {
                    "provider_id": provider,
                    "provider_artifact_sha256": decoded[provider]["provider_artifact_sha256"],
                    "provider_manifest_sha256": decoded[provider]["provider_manifest"]["manifest_sha256"],
                    "head": head,
                    "parameter": full_parameters[provider][head],
                }
            )
            for head in HEADS
        }
        for provider in PROVIDERS
    }
    routes = _select_routes_from_oof(
        fitted_stitched, uncalibrated_stitched, baseline_stitched, y, np.ones(n, dtype=bool),
        per_fold_support, fit_by_provider, bootstrap_samples=bootstrap_samples,
        role_prefix="full_development_outer_oof",
    )
    for provider in PROVIDERS:
        for head in HEADS:
            mode = routes[head]["provider_evidence"][provider]["calibration_quality"]["selected_mode"]
            if mode == "identity_parameter_fallback":
                full_parameters[provider][head]["value"] = 1.0
                full_parameters[provider][head]["identity_parameter_sentinel"] = (
                    "exact_identity_value_one_selected_by_full_development_crossfit"
                )
            else:
                full_parameters[provider][head]["identity_parameter_sentinel"] = None
            full_parameters[provider][head]["crossfit_selected_calibration_mode"] = mode
            parameter_sha_by_provider[provider][head] = canonical_sha256(
                {
                    "provider_id": provider,
                    "provider_artifact_sha256": decoded[provider]["provider_artifact_sha256"],
                    "provider_manifest_sha256": decoded[provider]["provider_manifest"]["manifest_sha256"],
                    "head": head,
                    "parameter": full_parameters[provider][head],
                }
            )
    for head in HEADS:
        selected = routes[head]["selected_provider"]
        parameter = None if selected is None else full_parameters[selected][head]
        routes[head]["deployment_parameter"] = parameter
        routes[head]["deployment_parameter_sha256"] = (
            None if selected is None else parameter_sha_by_provider[selected][head]
        )
    routed_oof_evidence = {}
    for head in HEADS:
        mask = masks[head]
        outer_route_available = all(
            fold["inner_oof_selected_route_by_head"][head] is not None
            for fold in fold_receipts
        )
        support_passed = all(per_fold_support[head])
        selected_fit_passed = outer_route_available and all(
            fold["provider_parameters"][fold["inner_oof_selected_route_by_head"][head]][head]["fit_passed"]
            for fold in fold_receipts
        )
        if not (outer_route_available and support_passed and selected_fit_passed):
            routed_oof_evidence[head] = {
                "status": "disabled_before_metric_evaluation",
                "outer_exact_once_passed": bool(np.all(owner >= 0)),
                "every_outer_fold_selected_provider": outer_route_available,
                "per_outer_fold_support_passed": support_passed,
                "per_outer_fold_selected_provider_fit_passed": selected_fit_passed,
                "proper_loss_gain_vs_train_only_baseline": None,
                "decision_loss_gain_vs_train_only_baseline": None,
                "uncertainty_aurc_gate": None,
                "paired_route_gain_vs_body_agnostic_reference": None,
                "harmful_rate_vs_body_agnostic_reference": None,
                "selection_aware_calibration_quality": None,
                "gate_passed": False,
            }
            continue
        proper = _gain_evidence(
            routed_stitched[head]["proper"], baseline_stitched[head]["proper"],
            y["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"outer_routed:{head}:proper",
            execution_groups=y["group_id"],
        )
        routed_calibration_quality = _calibration_quality(
            head, routed_stitched[head], routed_uncalibrated_stitched[head],
            y, mask, bootstrap_samples=bootstrap_samples,
            role=f"outer_routed:{head}:calibration",
            allow_identity_fallback=False,
        )
        decision = _gain_evidence(
            routed_stitched[head]["decision"], baseline_stitched[head]["decision"],
            y["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"outer_routed:{head}:decision",
            execution_groups=y["group_id"],
        )
        uncertainty = _uncertainty_evidence(
            routed_stitched[head]["uncertainty"], routed_stitched[head]["decision"],
            y["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"outer_routed:{head}:uncertainty",
            execution_groups=y["group_id"],
        )
        routed_vs_reference_proper = _gain_evidence(
            routed_stitched[head]["proper"],
            stitched[REFERENCE_PROVIDER][head]["proper"],
            y["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"outer_routed:{head}:vs_reference:proper",
            execution_groups=y["group_id"],
        )
        routed_vs_reference_decision = _gain_evidence(
            routed_stitched[head]["decision"],
            stitched[REFERENCE_PROVIDER][head]["decision"],
            y["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"outer_routed:{head}:vs_reference:decision",
            execution_groups=y["group_id"],
        )
        harm = _harm_evidence(
            routed_stitched[head]["proper"], routed_stitched[head]["decision"],
            stitched[REFERENCE_PROVIDER][head]["proper"],
            stitched[REFERENCE_PROVIDER][head]["decision"],
            y["semantic_reset_cluster_id"], mask, samples=bootstrap_samples,
            role=f"outer_routed:{head}:harm",
            execution_groups=y["group_id"],
        )
        routed_oof_evidence[head] = {
            "outer_exact_once_passed": bool(np.all(owner >= 0)),
            "per_outer_fold_support_passed": support_passed,
            "per_outer_fold_selected_provider_fit_passed": selected_fit_passed,
            "proper_loss_gain_vs_train_only_baseline": proper,
            "decision_loss_gain_vs_train_only_baseline": decision,
            "uncertainty_aurc_gate": uncertainty,
            "paired_route_gain_vs_body_agnostic_reference": {
                "proper_loss": routed_vs_reference_proper,
                "decision_loss": routed_vs_reference_decision,
            },
            "harmful_rate_vs_body_agnostic_reference": harm,
            "selection_aware_calibration_quality": routed_calibration_quality,
            "gate_passed": bool(
                np.all(owner >= 0)
                and support_passed
                and selected_fit_passed
                and proper["lcb95"] is not None and proper["lcb95"] >= 0.0
                and decision["lcb95"] is not None and decision["lcb95"] >= 0.0
                and uncertainty["passed"]
                and routed_calibration_quality["passed"]
                and (
                    routes[head]["selected_provider"] == REFERENCE_PROVIDER
                    or (
                        routed_vs_reference_proper["lcb95"] is not None
                        and routed_vs_reference_proper["lcb95"] > 0.0
                        and routed_vs_reference_decision["lcb95"] is not None
                        and routed_vs_reference_decision["lcb95"] > 0.0
                        and harm["both_harmful_rate_ucb_passed"]
                    )
                )
            ),
        }
    label_content = canonical_sha256(
        {key: _ndarray_sha256(value) for key, value in y.items()}
    )
    sample_order_sha = canonical_sha256(y["sample_id"].tolist())
    mask_set_sha = canonical_sha256(
        {head: masks[head].tolist() for head in HEADS}
    )
    semantic_identity_contract_sha = canonical_sha256(
        {
            "execution_group_id": y["group_id"].tolist(),
            "semantic_reset_cluster_id": y["semantic_reset_cluster_id"].tolist(),
            "body_id": y["body_id"].tolist(),
            "actor_contract_id": y["actor_contract_id"].tolist(),
        }
    )
    raw_prediction_content = {
        provider: _prediction_content_sha(decoded[provider]) for provider in PROVIDERS
    }
    provider_oof_content = _oof_content_sha(stitched)
    provider_fitted_oof_content = _oof_content_sha(fitted_stitched)
    provider_raw_oof_content = _oof_content_sha(uncalibrated_stitched)
    baseline_oof_content = canonical_sha256(
        {
            head: {metric: _ndarray_sha256(array) for metric, array in metrics.items()}
            for head, metrics in baseline_stitched.items()
        }
    )
    routed_oof_content = canonical_sha256(
        {
            head: {metric: _ndarray_sha256(array) for metric, array in metrics.items()}
            for head, metrics in routed_stitched.items()
        }
    )
    calibration_bundle_sha_by_provider = {
        provider: canonical_sha256(
            {
                "provider_id": provider,
                "provider_artifact_sha256": decoded[provider]["provider_artifact_sha256"],
                "provider_manifest_sha256": decoded[provider]["provider_manifest"]["manifest_sha256"],
                "raw_prediction_tensor_sha256": raw_prediction_content[provider],
                "effective_outer_oof_sha256": _oof_content_sha({provider: stitched[provider]}),
                "fitted_outer_oof_sha256": _oof_content_sha({provider: fitted_stitched[provider]}),
                "uncalibrated_outer_oof_sha256": _oof_content_sha(
                    {provider: uncalibrated_stitched[provider]}
                ),
                "train_only_baseline_outer_oof_sha256": baseline_oof_content,
                "head_calibration_parameter_sha256": parameter_sha_by_provider[provider],
                "sample_identity_order_sha256": sample_order_sha,
                "applicability_mask_set_sha256": mask_set_sha,
                "label_content_sha256": label_content,
                "semantic_identity_contract_sha256": semantic_identity_contract_sha,
            }
        )
        for provider in PROVIDERS
    }
    provider_descriptors = [
        {
            "provider_id": provider,
            "provider_artifact_sha256": decoded[provider]["provider_artifact_sha256"],
            "calibration_bundle_sha256": calibration_bundle_sha_by_provider[provider],
            "head_calibration_parameter_sha256": parameter_sha_by_provider[provider],
        }
        for provider in sorted(PROVIDERS)
    ]
    base = {
        "format": FORMAT,
        "status": STATUS,
        "providers": list(PROVIDERS),
        "member_count_per_provider": MEMBER_COUNT,
        "head_names": list(HEADS),
        "sample_count": n,
        "logical_group_count": len(set(y["group_id"].tolist())),
        "sample_identity_set_sha256": sample_order_sha,
        "group_order_sha256": canonical_sha256(y["group_id"].tolist()),
        "applicability_mask_set_sha256": mask_set_sha,
        "fold_count": FOLD_COUNT,
        "folds": fold_receipts,
        "every_sample_heldout_exactly_once": True,
        "bootstrap": {
            "unit": "equal_semantic_reset_cluster",
            "within_cluster_execution_groups_equal_weight": True,
            "seed": BOOTSTRAP_SEED,
            "samples": bootstrap_samples,
            "within_group_rows_iid": False,
        },
        "head_routes": routes,
        "nested_outer_routed_oof_evidence": routed_oof_evidence,
        "full_data_provider_specific_refit_parameters": full_parameters,
        "provider_descriptors_for_prediction_router": provider_descriptors,
        "content_bindings": {
            "labels_sample_order_masks_ndarray_sha256": label_content,
            "semantic_identity_contract_sha256": semantic_identity_contract_sha,
            "raw_prediction_tensor_sha256_by_provider": raw_prediction_content,
            "provider_artifact_sha256_by_provider": {
                provider: decoded[provider]["provider_artifact_sha256"]
                for provider in PROVIDERS
            },
            "provider_manifest_sha256_by_provider": {
                provider: decoded[provider]["provider_manifest"]["manifest_sha256"]
                for provider in PROVIDERS
            },
            "provider_effective_outer_oof_full_array_set_sha256": provider_oof_content,
            "provider_fitted_outer_oof_full_array_set_sha256": provider_fitted_oof_content,
            "provider_uncalibrated_outer_oof_full_array_set_sha256": provider_raw_oof_content,
            "train_only_baseline_outer_oof_full_array_set_sha256": baseline_oof_content,
            "routed_calibrated_outer_oof_full_array_set_sha256": routed_oof_content,
            "routed_uncalibrated_outer_oof_full_array_set_sha256": canonical_sha256(
                {
                    head: {
                        metric: _ndarray_sha256(array)
                        for metric, array in metrics.items()
                    }
                    for head, metrics in routed_uncalibrated_stitched.items()
                }
            ),
        },
        "rank_route": {
            "source_contract_rank_score_consumed": False,
            "source_contract_rank_score_is_a_six_head_route": False,
            "external_rank_gate_bool_or_sha_accepted": False,
            "independent_whole_provider_oof_passed": False,
            "rank_route_authorized": False,
            "required_action_route": "actor_baseline",
        },
        "capability": {
            "external_gate_bool_or_evidence_sha_accepted": False,
            "provider_manifest_prediction_tensor_binding_verified": True,
            "checkpoint_to_tensor_forward_attestation_verified": False,
            "prediction_arrays_remain_a_trusted_input_boundary": True,
            "filesystem_hdf_checkpoint_or_label_files_opened": 0,
            "training_collection_or_deployment_performed": False,
            "promotion_authorized": False,
            "performance_claim_authorized": False,
        },
    }
    evidence_sha = canonical_sha256(base)
    all_enabled = all(
        routes[head]["selected_provider"] is not None
        and routed_oof_evidence[head]["gate_passed"]
        for head in HEADS
    )
    head_decisions = {}
    for head in HEADS:
        route = routes[head]
        selected = route["selected_provider"]
        if selected is None:
            continue
        proper_lcb = route["paired_provider_proper_loss_gain"]["lcb95"]
        decision_lcb = route["paired_provider_decision_loss_gain"]["lcb95"]
        paired_lcb = min(float(proper_lcb), float(decision_lcb))
        head_decisions[head] = {
            "head": head,
            "support_gate_passed": bool(
                all(route["provider_evidence"][provider]["all_crossfit_train_folds_support_passed"] for provider in PROVIDERS)
            ),
            "reference_variant_gate_passed": bool(
                route["provider_evidence"][REFERENCE_PROVIDER]["gate_passed"]
            ),
            "candidate_variant_gate_passed": bool(
                route["provider_evidence"][CANDIDATE_PROVIDER]["gate_passed"]
            ),
            "selection_status": (
                "selected_body_conditioned_candidate"
                if selected == CANDIDATE_PROVIDER
                else "fallback_body_agnostic_reference"
            ),
            "selected_variant_id": selected,
            "reason": (
                "nested_selection_algorithm_passed_and_full_development_"
                "fixed_provider_oof_chose_final_route"
            ),
            "paired_gain_lcb95": paired_lcb,
            "harmful_rate_ucb95": max(
                float(route["paired_provider_harmful_rate"]["proper_loss"]["ucb95"]),
                float(route["paired_provider_harmful_rate"]["decision_loss"]["ucb95"]),
            ),
            "selected_deployment_calibration_parameter_sha256": route["deployment_parameter_sha256"],
            "actor_baseline_fallback_required": False,
        }
    receipt_core = {
        "format": "etsf_smolvla_piper_dual_provider_raw_recomputed_route_receipt_v1",
        "status": "verified_raw_recomputed_development_route_no_promotion_authority",
        "plan_sha256": canonical_sha256(split),
        "evidence_sha256": evidence_sha,
        "variant_scope": "full_body_adapter_ablation",
        "sample_identity_order_sha256": base["sample_identity_set_sha256"],
        "applicability_mask_set_sha256": base["applicability_mask_set_sha256"],
        "head_decisions": head_decisions,
        "all_six_heads_enabled": all_enabled,
        "system_fallback": "actor_baseline_required_rank_route_not_authorized",
        "rank_route_contract": {
            "source_contract_rank_score_is_a_six_head_route": False,
            "prior_rank_contract_sha_reusable": False,
            "independent_whole_provider_oof_passed": False,
            "rank_route_authorized": False,
            "rank_action_selection_must_fallback_to_actor_baseline": True,
        },
        "capability": {
            "raw_metric_truth_recomputed": True,
            "outer_oof_predictions_content_bound": True,
            "per_fold_support_recomputed": True,
            "provider_specific_calibration_content_bound": True,
            "selected_variant_ids_are_provisional_non_executable": False,
            "runtime_route_exported_for_development_only": True,
            "production_route_exported": False,
            "signature_or_issuer_verification": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "performance_claim_authorized": False,
        },
    }
    receipt = (
        {**receipt_core, "receipt_sha256": canonical_sha256(receipt_core)}
        if all_enabled else None
    )
    return {
        **base,
        "router_sha256": evidence_sha,
        "route_receipt": receipt,
        "route_receipt_export_status": (
            "executable_development_only" if all_enabled
            else "not_exported_one_or_more_heads_disabled"
        ),
    }


__all__ = [
    "BOOTSTRAP_SEED",
    "CANDIDATE_PROVIDER",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DualProviderRouterError",
    "EVENT_VOCAB",
    "FOLD_COUNT",
    "FORMAT",
    "HEADS",
    "MEMBER_COUNT",
    "PROVIDERS",
    "REFERENCE_PROVIDER",
    "build_five_fold_group_plan",
    "build_provider_manifest",
    "calibrate_dual_provider_router",
    "canonical_bytes",
    "canonical_sha256",
]
