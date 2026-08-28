#!/usr/bin/env python3
"""Fail-closed protocol for ETSF policy/embodiment transfer experiments.

This module does not load RoboTwin data and never runs an actor.  It freezes
logical group identities, audits that the shared event core did not change
outside one explicitly reserved policy/body embedding row, and evaluates a
summary produced by a separately executed target-domain experiment.

The protocol intentionally separates policy transfer from embodiment transfer.
Changing both in one experiment is rejected because policy and body shift would
otherwise be confounded.  The existing OpenVLA ``fresh_confirmation`` registry
is also rejected: transfer confirmation must use its own preregistered target
domain groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import numpy as np


PROTOCOL_FORMAT = "etsf_policy_body_transfer_protocol_v1"
AUDIT_FORMAT = "etsf_transfer_weight_audit_v1"
RESULT_FORMAT = "etsf_transfer_result_summary_v1"
DECISION_FORMAT = "etsf_transfer_acceptance_decision_v1"

ADAPTATION_SIZES = (0, 5, 10, 20, 50)
PRIMARY_ADAPTATION_SIZE = 20
MIN_VALIDATION_PER_TASK = 20
MIN_CONFIRMATION_PER_TASK = 50
FORBIDDEN_DATA_TOKENS = (
    "fresh50",
    "fresh_confirmation",
    "fresh-confirmation",
)

GROUP_FIELDS = {
    "task",
    "policy",
    "embodiment",
    "requested_seed",
    "resolved_seed",
    "artifact_sha256",
    "registry",
}
SPLIT_REGISTRY = {
    "source_training": "source_core_training",
    "adaptation": "transfer_adaptation",
    "validation": "transfer_validation",
    "confirmation": "transfer_confirmation",
}
REQUIRED_BASELINES = {
    "actor",
    "zero_shot_core",
    "target_from_scratch_matched",
    "full_finetune_upper",
    "no_factorization",
    "actor_hidden_observer",
    "privileged_pose_upper_bound",
}
METRIC_FIELDS = {
    "observer_event_macro_f1",
    "observer_event_frequency_macro_f1",
    "observer_predicate_macro_f1",
    "observer_predicate_constant_macro_f1",
    "observer_coverage",
    "next_event_macro_f1",
    "current_event_macro_f1",
    "event_frequency_macro_f1",
    "success_pr_auc",
    "success_brier",
    "constant_success_brier",
    "success_ece",
    "duration_mae",
    "event_body_median_duration_mae",
    "object_delta_mae",
    "zero_delta_mae",
    "pair_accuracy",
    "uncertainty_aurc",
    "random_aurc",
}


def paired_bootstrap_ci(
    *,
    episodes: int,
    helpful: int,
    harmful: int,
    seed: int = 20260827,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Frozen episode-paired percentile bootstrap for binary success deltas."""

    if episodes < 1 or min(helpful, harmful) < 0 or helpful + harmful > episodes:
        raise ValueError("invalid paired helpful/harmful counts")
    if resamples != 10_000 or seed != 20260827:
        raise ValueError("formal paired bootstrap seed/resample count is frozen")
    differences = np.concatenate(
        [
            np.ones(helpful, dtype=np.float64),
            -np.ones(harmful, dtype=np.float64),
            np.zeros(episodes - helpful - harmful, dtype=np.float64),
        ]
    )
    generator = np.random.default_rng(seed)
    means = differences[
        generator.integers(0, episodes, size=(resamples, episodes))
    ].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def exact_mcnemar_p(*, helpful: int, harmful: int) -> float:
    """Two-sided exact McNemar/sign-test p-value for discordant episodes."""

    if min(helpful, harmful) < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = helpful + harmful
    if discordant == 0:
        return 1.0
    tail = min(helpful, harmful)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _assert_exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    fields = set(value)
    if fields != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - fields)}, "
            f"extra={sorted(fields - expected)}"
        )


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_strings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _walk_strings(nested)


def _reject_forbidden_data_references(value: Any) -> None:
    for text in _walk_strings(value):
        lowered = text.lower()
        token = next((item for item in FORBIDDEN_DATA_TOKENS if item in lowered), None)
        if token is not None:
            raise ValueError(
                f"transfer protocol must not reuse the OpenVLA sealed registry: {token}"
            )


def _as_nonempty_names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence")
    names = tuple(str(item) for item in value)
    if not names or any(not item for item in names) or len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique non-empty names")
    return names


def _load_checkpoint(path: str | Path) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    numpy_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain one model state under 'model'")
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint model state must contain tensors only")
    return state, dict(payload)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _immutable_state_sha256(
    state: Mapping[str, torch.Tensor], allowed_rows: Mapping[str, int]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        if name not in allowed_rows:
            digest.update(_tensor_bytes(tensor))
            continue
        row = allowed_rows[name]
        if tensor.ndim < 1 or row < 0 or row >= tensor.shape[0]:
            raise ValueError(f"allowed row {row} is invalid for {name}")
        if row:
            digest.update(_tensor_bytes(tensor[:row]))
        if row + 1 < tensor.shape[0]:
            digest.update(_tensor_bytes(tensor[row + 1 :]))
    return digest.hexdigest()


def _checkpoint_core_metadata(path: str | Path) -> dict[str, Any]:
    state, payload = _load_checkpoint(path)
    config = payload.get("config")
    contract = payload.get("contract")
    if not isinstance(config, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("checkpoint must freeze config and contract mappings")
    body_to_id = contract.get("body_to_id")
    policy_to_id = contract.get("policy_to_id")
    if not isinstance(body_to_id, Mapping) or not isinstance(policy_to_id, Mapping):
        raise ValueError("checkpoint contract lacks body_to_id/policy_to_id")
    expansion = payload.get("transfer_source_core_expansion")
    if expansion is not None:
        proof = payload.get("reserved_source_retraining")
        if not isinstance(expansion, Mapping) or expansion.get("format") != (
            "etsf_transfer_source_core_expansion_v1"
        ):
            raise ValueError("prepared target vocabulary lineage is invalid")
        if not isinstance(proof, Mapping) or proof.get("format") != (
            "etsf_reserved_source_core_retraining_v1"
        ):
            raise ValueError(
                "a post-training vocabulary expansion cannot be frozen directly; "
                "source-only retraining with the reserved row is required"
            )
        if (
            proof.get("status") != "complete_source_only"
            or int(proof.get("source_training_steps", 0)) <= 0
            or int(proof.get("source_training_groups", 0)) <= 0
            or proof.get("target_data_read") is not False
            or proof.get("target_labels_read") is not False
            or proof.get("reserved_row_used_in_source_batches") is not False
            or proof.get("shared_core_retrained") is not True
        ):
            raise ValueError("reserved-row source retraining proof is incomplete")
        for path_key, sha_key in (
            ("source_manifest_path", "source_manifest_sha256"),
            ("source_split_path", "source_split_sha256"),
        ):
            source_path = Path(str(proof.get(path_key, ""))).expanduser()
            prepared_path = Path(str(expansion.get(path_key, ""))).expanduser()
            if (
                not source_path.is_file()
                or file_sha256(source_path) != proof.get(sha_key)
                or source_path.resolve() != prepared_path.resolve()
                or proof.get(sha_key) != expansion.get(sha_key)
            ):
                raise ValueError("reserved-row source retraining provenance changed")
    return {
        "path": str(Path(path).expanduser().resolve()),
        "file_sha256": file_sha256(path),
        "state_dict_sha256": state_dict_sha256(state),
        "config_sha256": json_sha256(config),
        "num_bodies": int(config["num_bodies"]),
        "num_policies": int(config["num_policies"]),
        "body_to_id": {str(key): int(value) for key, value in body_to_id.items()},
        "policy_to_id": {str(key): int(value) for key, value in policy_to_id.items()},
    }


def _validate_group(group: Any, split: str) -> tuple[Any, ...]:
    if not isinstance(group, Mapping):
        raise ValueError(f"{split} group must be a mapping")
    _assert_exact_fields(group, GROUP_FIELDS, f"{split} group")
    for name in ("task", "policy", "embodiment"):
        if not isinstance(group[name], str) or not group[name]:
            raise ValueError(f"{split}.{name} must be a non-empty string")
    for name in ("requested_seed", "resolved_seed"):
        if not isinstance(group[name], int) or isinstance(group[name], bool):
            raise ValueError(f"{split}.{name} must be an integer")
    if not _valid_sha256(group["artifact_sha256"]):
        raise ValueError(f"{split} group artifact_sha256 is invalid")
    if group["registry"] != SPLIT_REGISTRY[split]:
        raise ValueError(f"{split} group uses the wrong registry")
    return (
        group["task"],
        group["policy"],
        group["embodiment"],
        group["requested_seed"],
        group["resolved_seed"],
    )


def _expected_adapter_kinds(protocol: Mapping[str, Any]) -> set[str]:
    axis = protocol["axis"]
    contracts = protocol["contracts"]
    expected: set[str] = set()
    if contracts["state"]["mode"] == "learned_state_adapter":
        expected.add("state_adapter")
    if axis == "policy":
        if contracts["action_effect"]["mode"] != "identity_content_addressed":
            expected.add("policy_action_adapter")
    else:
        if contracts["action_effect"]["mode"] != "identity_content_addressed":
            expected.add("body_action_adapter")
        expected.add("clock_adapter")
    if contracts["predicate"]["mode"] != "identity_content_addressed":
        expected.add("predicate_adapter")
    if contracts["observer"]["mode"] in {
        "actor_hidden_observer",
        "rgb_observer",
    }:
        expected.add("state_observer")
    return expected


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate a fully frozen policy/body transfer protocol."""

    _reject_forbidden_data_references(protocol)
    expected_top = {
        "format",
        "study_id",
        "axis",
        "source_domain",
        "target_domain",
        "core",
        "contracts",
        "adaptation",
        "splits",
        "split_contract",
        "baselines",
        "acceptance",
    }
    _assert_exact_fields(protocol, expected_top, "protocol")
    if protocol["format"] != PROTOCOL_FORMAT:
        raise ValueError("unsupported transfer protocol format")
    if not isinstance(protocol["study_id"], str) or not protocol["study_id"]:
        raise ValueError("study_id must be non-empty")
    axis = protocol["axis"]
    if axis not in ("policy", "embodiment"):
        raise ValueError("axis must be policy or embodiment")

    source = protocol["source_domain"]
    target = protocol["target_domain"]
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("source_domain and target_domain must be mappings")
    _assert_exact_fields(source, {"policies", "embodiments", "tasks"}, "source_domain")
    _assert_exact_fields(target, {"policy", "embodiment", "tasks"}, "target_domain")
    source_policies = _as_nonempty_names(source["policies"], "source policies")
    source_bodies = _as_nonempty_names(source["embodiments"], "source embodiments")
    source_tasks = set(_as_nonempty_names(source["tasks"], "source tasks"))
    target_tasks = set(_as_nonempty_names(target["tasks"], "target tasks"))
    target_policy = str(target["policy"])
    target_body = str(target["embodiment"])
    if not target_policy or not target_body or source_tasks != target_tasks:
        raise ValueError("source/target must use the same non-empty task set")
    if axis == "policy":
        if target_policy in source_policies:
            raise ValueError("target policy must be held out from shared-core training")
        if target_body not in source_bodies:
            raise ValueError("policy transfer must hold embodiment fixed")
    else:
        if target_body in source_bodies:
            raise ValueError("target embodiment must be held out from shared-core training")
        if target_policy not in source_policies:
            raise ValueError("embodiment transfer must hold policy fixed")

    core = protocol["core"]
    if not isinstance(core, Mapping):
        raise ValueError("core must be a mapping")
    _assert_exact_fields(
        core,
        {
            "path",
            "file_sha256",
            "state_dict_sha256",
            "config_sha256",
            "num_bodies",
            "num_policies",
            "body_to_id",
            "policy_to_id",
            "target_embedding",
        },
        "core",
    )
    for name in ("file_sha256", "state_dict_sha256", "config_sha256"):
        if not _valid_sha256(core[name]):
            raise ValueError(f"core.{name} is invalid")
    if min(int(core["num_bodies"]), int(core["num_policies"])) < 1:
        raise ValueError("core vocabulary sizes must be positive")
    for mapping_name, upper_name in (
        ("body_to_id", "num_bodies"),
        ("policy_to_id", "num_policies"),
    ):
        mapping = core[mapping_name]
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"core.{mapping_name} must be non-empty")
        ids = [int(value) for value in mapping.values()]
        if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= int(core[upper_name]):
            raise ValueError(f"core.{mapping_name} ids are invalid or duplicated")

    embedding = core["target_embedding"]
    if not isinstance(embedding, Mapping):
        raise ValueError("target_embedding must be a mapping")
    _assert_exact_fields(
        embedding,
        {"parameter", "row", "reservation_name", "status"},
        "target_embedding",
    )
    expected_parameter = (
        "action_encoder.policy_embedding.weight"
        if axis == "policy"
        else "action_encoder.body_embedding.weight"
    )
    expected_registry = "policy_to_id" if axis == "policy" else "body_to_id"
    expected_upper = "num_policies" if axis == "policy" else "num_bodies"
    if embedding["status"] != "preallocated":
        raise ValueError(
            "current core requires a preallocated target embedding row; expanding "
            "the vocabulary after freezing is not an auditable direct transfer"
        )
    if embedding["parameter"] != expected_parameter:
        raise ValueError("target embedding parameter does not match transfer axis")
    row = int(embedding["row"])
    reservation = str(embedding["reservation_name"])
    if (
        not reservation.startswith("__reserved__")
        or core[expected_registry].get(reservation) != row
        or row < 0
        or row >= int(core[expected_upper])
    ):
        raise ValueError("target embedding row is not content-frozen and preallocated")

    contracts = protocol["contracts"]
    if not isinstance(contracts, Mapping):
        raise ValueError("contracts must be a mapping")
    _assert_exact_fields(
        contracts,
        {"state", "action_effect", "predicate", "clock", "observer"},
        "contracts",
    )
    allowed_contract_modes = {
        "state": {"identity_content_addressed", "learned_state_adapter"},
        "action_effect": {
            "identity_content_addressed",
            "policy_action_adapter",
            "body_action_adapter",
        },
        "predicate": {"identity_content_addressed", "predicate_adapter"},
        "clock": {"identity_fixed_body", "clock_adapter"},
        "observer": {
            "actor_hidden_observer",
            "rgb_observer",
            "privileged_simulator_pose_upper_bound",
        },
    }
    for name, entry in contracts.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"contracts.{name} must be a mapping")
        _assert_exact_fields(entry, {"source_id", "target_id", "mode"}, f"contracts.{name}")
        source_id, target_id, mode = entry["source_id"], entry["target_id"], entry["mode"]
        if not source_id or not target_id or mode not in allowed_contract_modes[name]:
            raise ValueError(f"contracts.{name} has an invalid id or mode")
        if name != "observer":
            same = source_id == target_id
            identity_mode = mode in {"identity_content_addressed", "identity_fixed_body"}
            if identity_mode != same:
                raise ValueError(f"contracts.{name} identity mode must exactly match content ids")
    if contracts["observer"]["mode"] == "privileged_simulator_pose_upper_bound":
        raise ValueError(
            "the primary transfer path must use an RGB or actor-hidden observer; "
            "simulator pose is an upper-bound baseline only"
        )
    if axis == "policy":
        if contracts["action_effect"]["mode"] not in (
            "identity_content_addressed",
            "policy_action_adapter",
        ):
            raise ValueError("policy transfer requires a policy action adapter")
        if contracts["clock"]["mode"] != "identity_fixed_body":
            raise ValueError("policy transfer must keep the body clock fixed")
    else:
        if contracts["action_effect"]["mode"] not in (
            "identity_content_addressed",
            "body_action_adapter",
        ):
            raise ValueError("embodiment transfer requires a body action adapter")
        if contracts["clock"]["mode"] != "clock_adapter":
            raise ValueError("embodiment transfer requires target clock calibration")

    adaptation = protocol["adaptation"]
    if not isinstance(adaptation, Mapping):
        raise ValueError("adaptation must be a mapping")
    _assert_exact_fields(
        adaptation,
        {
            "sizes_per_task",
            "primary_size_per_task",
            "shared_core_update",
            "target_credit_assignment",
            "allowed_external_adapters",
            "allowed_core_row_updates",
        },
        "adaptation",
    )
    if tuple(adaptation["sizes_per_task"]) != ADAPTATION_SIZES:
        raise ValueError("adaptation sizes must be frozen to 0/5/10/20/50")
    if int(adaptation["primary_size_per_task"]) != PRIMARY_ADAPTATION_SIZE:
        raise ValueError("primary adaptation size must be 20 per task")
    if adaptation["shared_core_update"] != "forbidden":
        raise ValueError("shared event core updates are forbidden")
    if adaptation["target_credit_assignment"] != "adapter_supervision_only_no_td":
        raise ValueError("target-domain TD/shared-core credit assignment is forbidden")
    adapters = set(_as_nonempty_names(
        adaptation["allowed_external_adapters"], "allowed external adapters"
    )) if adaptation["allowed_external_adapters"] else set()
    if adapters != _expected_adapter_kinds(protocol):
        raise ValueError("allowed external adapters do not match content-addressed contracts")
    row_updates = adaptation["allowed_core_row_updates"]
    if not isinstance(row_updates, Sequence) or isinstance(row_updates, (str, bytes)):
        raise ValueError("allowed_core_row_updates must be a sequence")
    if len(row_updates) != 1 or not isinstance(row_updates[0], Mapping):
        raise ValueError("exactly one preallocated target embedding row may update")
    _assert_exact_fields(row_updates[0], {"parameter", "row"}, "core row update")
    if row_updates[0]["parameter"] != expected_parameter or int(row_updates[0]["row"]) != row:
        raise ValueError("allowed row update does not match the reserved target row")

    split_contract = protocol["split_contract"]
    if not isinstance(split_contract, Mapping):
        raise ValueError("split_contract must be a mapping")
    expected_split_contract = {
        "unit": "task_policy_embodiment_requested_resolved_seed",
        "adaptation_order": "ascending_preregistered_index_fixed_prefix",
        "minimum_adaptation_per_task": max(ADAPTATION_SIZES),
        "minimum_validation_per_task": MIN_VALIDATION_PER_TASK,
        "minimum_confirmation_per_task": MIN_CONFIRMATION_PER_TASK,
        "confirmation_access": "once_after_validation_freeze",
    }
    if dict(split_contract) != expected_split_contract:
        raise ValueError("split_contract is not the frozen strict transfer contract")

    splits = protocol["splits"]
    if not isinstance(splits, Mapping):
        raise ValueError("splits must be a mapping")
    _assert_exact_fields(splits, set(SPLIT_REGISTRY), "splits")
    all_keys: dict[tuple[Any, ...], str] = {}
    target_requested: dict[tuple[str, int], str] = {}
    target_resolved: dict[tuple[str, int], str] = {}
    for split in SPLIT_REGISTRY:
        groups = splits[split]
        if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
            raise ValueError(f"split {split} must be a non-empty sequence")
        per_task: Counter[str] = Counter()
        for group in groups:
            key = _validate_group(group, split)
            if key in all_keys:
                raise ValueError(f"logical group overlaps {all_keys[key]} and {split}")
            all_keys[key] = split
            per_task[str(group["task"])] += 1
            if split == "source_training":
                if group["task"] not in source_tasks:
                    raise ValueError("source training group is outside the source tasks")
                if group["policy"] not in source_policies or group["embodiment"] not in source_bodies:
                    raise ValueError("source training group is outside the source domain")
            else:
                if (
                    group["task"] not in target_tasks
                    or group["policy"] != target_policy
                    or group["embodiment"] != target_body
                ):
                    raise ValueError(f"{split} group is outside the held-out target domain")
                requested_key = (str(group["task"]), int(group["requested_seed"]))
                resolved_key = (str(group["task"]), int(group["resolved_seed"]))
                if requested_key in target_requested or resolved_key in target_resolved:
                    raise ValueError("requested/resolved target seeds overlap across transfer splits")
                target_requested[requested_key] = split
                target_resolved[resolved_key] = split
        if split != "source_training" and set(per_task) != target_tasks:
            raise ValueError(f"{split} does not cover every target task")
        minimum = {
            "source_training": 1,
            "adaptation": max(ADAPTATION_SIZES),
            "validation": MIN_VALIDATION_PER_TASK,
            "confirmation": MIN_CONFIRMATION_PER_TASK,
        }[split]
        if any(per_task[task] < minimum for task in (source_tasks if split == "source_training" else target_tasks)):
            raise ValueError(f"{split} has fewer than {minimum} groups per task")

    baselines = protocol["baselines"]
    if set(_as_nonempty_names(baselines, "baselines")) != REQUIRED_BASELINES:
        raise ValueError("strict transfer baselines are incomplete")
    acceptance = protocol["acceptance"]
    expected_acceptance = {
        "prediction": (
            "all_structured_heads_beat_frozen_baselines_at_primary_n"
        ),
        "uncertainty": "aurc_below_random_at_primary_n",
        "success": "paired_delta_positive_and_ci95_low_nonnegative",
        "harmful_rate_max": 0.10,
        "minimum_changed": 10,
        "minimum_coverage": 0.10,
        "sample_efficiency": "beats_target_from_scratch_matched_at_primary_n",
    }
    if dict(acceptance) != expected_acceptance:
        raise ValueError("acceptance gate differs from the strict frozen contract")


def freeze_protocol(
    draft: Mapping[str, Any], checkpoint_path: str | Path
) -> dict[str, Any]:
    """Bind a draft protocol to one immutable shared-core checkpoint."""

    frozen = json.loads(json.dumps(draft))
    core = frozen.get("core")
    if not isinstance(core, Mapping) or set(core) != {"target_embedding"}:
        raise ValueError("draft core must contain only target_embedding")
    target_embedding = dict(core["target_embedding"])
    frozen["core"] = {
        **_checkpoint_core_metadata(checkpoint_path),
        "target_embedding": target_embedding,
    }
    validate_protocol(frozen)
    return frozen


def audit_transfer_weights(
    protocol: Mapping[str, Any], before_path: str | Path, after_path: str | Path
) -> dict[str, Any]:
    """Prove that only the reserved policy/body embedding row changed."""

    validate_protocol(protocol)
    if file_sha256(before_path) != protocol["core"]["file_sha256"]:
        raise ValueError("before checkpoint is not the protocol-frozen shared core")
    before_state, before_payload = _load_checkpoint(before_path)
    after_state, after_payload = _load_checkpoint(after_path)
    if set(before_state) != set(after_state):
        raise ValueError("checkpoint state keys changed during adaptation")
    if json_sha256(before_payload.get("config")) != protocol["core"]["config_sha256"]:
        raise ValueError("before checkpoint config differs from the protocol")
    if before_payload.get("config") != after_payload.get("config"):
        raise ValueError("model config changed during adaptation")
    allowed_rows = {
        str(entry["parameter"]): int(entry["row"])
        for entry in protocol["adaptation"]["allowed_core_row_updates"]
    }
    changed_rows: list[dict[str, Any]] = []
    for name in sorted(before_state):
        before = before_state[name]
        after = after_state[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"tensor shape/dtype changed: {name}")
        if name not in allowed_rows:
            if not torch.equal(before, after):
                raise ValueError(f"shared core parameter changed: {name}")
            continue
        row = allowed_rows[name]
        if before.ndim < 1 or row < 0 or row >= before.shape[0]:
            raise ValueError(f"invalid allowed row for {name}")
        keep = torch.ones(before.shape[0], dtype=torch.bool)
        keep[row] = False
        if not torch.equal(before[keep], after[keep]):
            raise ValueError(f"non-target rows changed in {name}")
        changed_rows.append(
            {
                "parameter": name,
                "row": row,
                "changed": not torch.equal(before[row], after[row]),
            }
        )
    immutable_before = _immutable_state_sha256(before_state, allowed_rows)
    immutable_after = _immutable_state_sha256(after_state, allowed_rows)
    if immutable_before != immutable_after:
        raise AssertionError("immutable shared-core digest changed")
    if not changed_rows or not all(entry["changed"] for entry in changed_rows):
        raise ValueError("reserved target embedding row was not adapted")
    return {
        "format": AUDIT_FORMAT,
        "study_id": protocol["study_id"],
        "protocol_sha256": json_sha256(protocol),
        "before_file_sha256": file_sha256(before_path),
        "after_file_sha256": file_sha256(after_path),
        "before_state_dict_sha256": state_dict_sha256(before_state),
        "after_state_dict_sha256": state_dict_sha256(after_state),
        "immutable_shared_core_sha256": immutable_before,
        "allowed_row_changes": changed_rows,
        "authorized": True,
    }


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_metrics(metrics: Any, name: str) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{name} metrics must be a mapping")
    _assert_exact_fields(metrics, METRIC_FIELDS, f"{name} metrics")
    parsed = {field: _finite_number(metrics[field], f"{name}.{field}") for field in METRIC_FIELDS}
    bounded = {
        "observer_event_macro_f1",
        "observer_event_frequency_macro_f1",
        "observer_predicate_macro_f1",
        "observer_predicate_constant_macro_f1",
        "observer_coverage",
        "next_event_macro_f1",
        "current_event_macro_f1",
        "event_frequency_macro_f1",
        "success_pr_auc",
        "success_brier",
        "constant_success_brier",
        "success_ece",
        "pair_accuracy",
    }
    if any(parsed[field] < 0 or parsed[field] > 1 for field in bounded):
        raise ValueError(f"{name} contains a probability metric outside [0,1]")
    nonnegative = METRIC_FIELDS - bounded
    if any(parsed[field] < 0 for field in nonnegative):
        raise ValueError(f"{name} contains a negative error/risk metric")
    return parsed


def evaluate_transfer_results(
    protocol: Mapping[str, Any], audit: Mapping[str, Any], results: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen prediction, transfer, and paired-success acceptance gate."""

    validate_protocol(protocol)
    _reject_forbidden_data_references(results)
    if audit.get("format") != AUDIT_FORMAT or audit.get("authorized") is not True:
        raise ValueError("a successful transfer weight audit is required")
    protocol_hash = json_sha256(protocol)
    if audit.get("protocol_sha256") != protocol_hash:
        raise ValueError("weight audit belongs to another protocol")
    expected_fields = {
        "format",
        "study_id",
        "protocol_sha256",
        "weight_audit_sha256",
        "adapters",
        "adaptation_curve",
        "baselines",
        "confirmation",
        "deployment_contract",
    }
    _assert_exact_fields(results, expected_fields, "results")
    if results["format"] != RESULT_FORMAT or results["study_id"] != protocol["study_id"]:
        raise ValueError("result format/study id mismatch")
    if results["protocol_sha256"] != protocol_hash:
        raise ValueError("results belong to another protocol")
    if results["weight_audit_sha256"] != json_sha256(audit):
        raise ValueError("results do not bind the supplied weight audit")

    adapters = results["adapters"]
    if not isinstance(adapters, Sequence) or isinstance(adapters, (str, bytes)):
        raise ValueError("adapters must be a sequence")
    adapter_kinds: set[str] = set()
    for entry in adapters:
        if not isinstance(entry, Mapping):
            raise ValueError("adapter record must be a mapping")
        _assert_exact_fields(entry, {"kind", "artifact_sha256", "trainable_parameters"}, "adapter")
        if not _valid_sha256(entry["artifact_sha256"]):
            raise ValueError("adapter artifact SHA256 is invalid")
        if int(entry["trainable_parameters"]) < 0:
            raise ValueError("adapter parameter count must be non-negative")
        adapter_kinds.add(str(entry["kind"]))
    if adapter_kinds != set(protocol["adaptation"]["allowed_external_adapters"]):
        raise ValueError("result adapter kinds differ from the frozen protocol")

    curve = results["adaptation_curve"]
    if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes)):
        raise ValueError("adaptation_curve must be a sequence")
    curve_by_n: dict[int, tuple[Mapping[str, Any], dict[str, float]]] = {}
    for row in curve:
        if not isinstance(row, Mapping):
            raise ValueError("adaptation row must be a mapping")
        _assert_exact_fields(row, {"n_per_task", "split", "group_count", "trainable_parameters", "metrics"}, "adaptation row")
        n = int(row["n_per_task"])
        if n in curve_by_n:
            raise ValueError("duplicate adaptation size")
        if row["split"] != "validation" or int(row["group_count"]) < len(protocol["target_domain"]["tasks"]) * MIN_VALIDATION_PER_TASK:
            raise ValueError("adaptation curve must use the complete transfer validation split")
        if int(row["trainable_parameters"]) < 0:
            raise ValueError("trainable parameter count must be non-negative")
        curve_by_n[n] = (row, _validate_metrics(row["metrics"], f"adaptation n={n}"))
    if set(curve_by_n) != set(ADAPTATION_SIZES):
        raise ValueError("adaptation curve must report every frozen N")

    baselines = results["baselines"]
    if not isinstance(baselines, Sequence) or isinstance(baselines, (str, bytes)):
        raise ValueError("baselines must be a sequence")
    baseline_by_name: dict[str, Mapping[str, Any]] = {}
    for row in baselines:
        if not isinstance(row, Mapping):
            raise ValueError("baseline row must be a mapping")
        _assert_exact_fields(row, {"name", "n_per_task", "trainable_parameters", "success_rate"}, "baseline row")
        name = str(row["name"])
        if name in baseline_by_name:
            raise ValueError("duplicate baseline")
        if int(row["n_per_task"]) not in (0, PRIMARY_ADAPTATION_SIZE):
            raise ValueError("baseline uses an unsupported adaptation size")
        if int(row["trainable_parameters"]) < 0:
            raise ValueError("baseline parameter count must be non-negative")
        rate = _finite_number(row["success_rate"], f"baseline {name} success_rate")
        if rate < 0 or rate > 1:
            raise ValueError("baseline success rate is outside [0,1]")
        baseline_by_name[name] = row
    if set(baseline_by_name) != REQUIRED_BASELINES:
        raise ValueError("result baselines differ from the frozen protocol")
    if int(baseline_by_name["target_from_scratch_matched"]["n_per_task"]) != PRIMARY_ADAPTATION_SIZE:
        raise ValueError("target-from-scratch baseline must use the primary N")

    confirmation = results["confirmation"]
    if not isinstance(confirmation, Mapping):
        raise ValueError("confirmation must be a mapping")
    confirmation_fields = {
        "split",
        "group_count",
        "episodes",
        "baseline_successes",
        "plugin_successes",
        "changed",
        "helpful",
        "harmful",
        "proposal_count",
        "coverage",
        "paired_delta",
        "paired_delta_ci95_low",
        "paired_delta_ci95_high",
        "exact_mcnemar_p",
    }
    _assert_exact_fields(confirmation, confirmation_fields, "confirmation")
    task_count = len(protocol["target_domain"]["tasks"])
    minimum_groups = task_count * MIN_CONFIRMATION_PER_TASK
    group_count = int(confirmation["group_count"])
    episodes = int(confirmation["episodes"])
    if confirmation["split"] != "confirmation" or group_count < minimum_groups or episodes != group_count:
        raise ValueError("confirmation does not cover the frozen held-out groups once")
    integer_fields = (
        "baseline_successes",
        "plugin_successes",
        "changed",
        "helpful",
        "harmful",
        "proposal_count",
    )
    counts = {name: int(confirmation[name]) for name in integer_fields}
    if any(count < 0 for count in counts.values()) or any(
        counts[name] > episodes for name in ("baseline_successes", "plugin_successes", "changed")
    ):
        raise ValueError("confirmation counts are invalid")
    if counts["helpful"] + counts["harmful"] > counts["changed"]:
        raise ValueError("helpful/harmful counts exceed changed episodes")
    if counts["proposal_count"] < counts["changed"]:
        raise ValueError("proposal count cannot be below executed changes")
    coverage = _finite_number(confirmation["coverage"], "coverage")
    paired_delta = _finite_number(confirmation["paired_delta"], "paired_delta")
    ci_low = _finite_number(confirmation["paired_delta_ci95_low"], "paired delta CI low")
    ci_high = _finite_number(confirmation["paired_delta_ci95_high"], "paired delta CI high")
    mcnemar = _finite_number(confirmation["exact_mcnemar_p"], "McNemar p")
    expected_coverage = counts["changed"] / episodes
    expected_delta = (counts["plugin_successes"] - counts["baseline_successes"]) / episodes
    discordant_delta = (counts["helpful"] - counts["harmful"]) / episodes
    if not math.isclose(coverage, expected_coverage, abs_tol=1e-12):
        raise ValueError("reported coverage does not match changed/episodes")
    if not math.isclose(paired_delta, expected_delta, abs_tol=1e-12):
        raise ValueError("reported paired delta does not match success counts")
    if not math.isclose(paired_delta, discordant_delta, abs_tol=1e-12):
        raise ValueError("paired delta does not match helpful/harmful episodes")
    expected_ci_low, expected_ci_high = paired_bootstrap_ci(
        episodes=episodes,
        helpful=counts["helpful"],
        harmful=counts["harmful"],
    )
    if not (
        math.isclose(ci_low, expected_ci_low, abs_tol=1e-12)
        and math.isclose(ci_high, expected_ci_high, abs_tol=1e-12)
    ):
        raise ValueError("paired CI does not match the frozen episode bootstrap")
    expected_mcnemar = exact_mcnemar_p(
        helpful=counts["helpful"], harmful=counts["harmful"]
    )
    if not math.isclose(mcnemar, expected_mcnemar, abs_tol=1e-12):
        raise ValueError("McNemar p does not match discordant episode counts")
    if not (0 <= coverage <= 1 and -1 <= ci_low <= paired_delta <= ci_high <= 1 and 0 <= mcnemar <= 1):
        raise ValueError("confirmation interval/probability values are invalid")

    deployment = results["deployment_contract"]
    if not isinstance(deployment, Mapping):
        raise ValueError("deployment_contract must be a mapping")
    _assert_exact_fields(
        deployment,
        {
            "target_policy",
            "target_embodiment",
            "guard_frozen_on",
            "confirmation_access_count",
            "shared_core_immutable_sha256",
            "action_ranking_authorized",
            "observer_mode",
            "observer_artifact_sha256",
            "privileged_inputs_used",
        },
        "deployment_contract",
    )
    if (
        deployment["target_policy"] != protocol["target_domain"]["policy"]
        or deployment["target_embodiment"] != protocol["target_domain"]["embodiment"]
        or deployment["guard_frozen_on"] != "validation"
        or int(deployment["confirmation_access_count"]) != 1
        or deployment["shared_core_immutable_sha256"] != audit["immutable_shared_core_sha256"]
        or deployment["observer_mode"] != protocol["contracts"]["observer"]["mode"]
        or not _valid_sha256(deployment["observer_artifact_sha256"])
        or deployment["privileged_inputs_used"] is not False
    ):
        raise ValueError("deployment contract does not match protocol/audit")

    primary_row, metrics = curve_by_n[PRIMARY_ADAPTATION_SIZE]
    reasons: list[str] = []
    if metrics["observer_event_macro_f1"] <= metrics["observer_event_frequency_macro_f1"]:
        reasons.append("observer_event_not_above_frequency_baseline")
    if metrics["observer_predicate_macro_f1"] <= metrics["observer_predicate_constant_macro_f1"]:
        reasons.append("observer_predicate_not_above_constant_baseline")
    if metrics["observer_coverage"] < 0.90:
        reasons.append("observer_coverage_below_0.90")
    if metrics["next_event_macro_f1"] <= max(
        metrics["current_event_macro_f1"], metrics["event_frequency_macro_f1"]
    ):
        reasons.append("next_event_not_above_frozen_baselines")
    if metrics["success_brier"] >= metrics["constant_success_brier"]:
        reasons.append("success_brier_not_improved")
    if metrics["duration_mae"] >= metrics["event_body_median_duration_mae"]:
        reasons.append("duration_mae_not_improved")
    if metrics["object_delta_mae"] >= metrics["zero_delta_mae"]:
        reasons.append("object_delta_mae_not_improved")
    if metrics["pair_accuracy"] <= 0.5:
        reasons.append("pair_accuracy_not_above_chance")
    if metrics["uncertainty_aurc"] >= metrics["random_aurc"]:
        reasons.append("uncertainty_ranking_not_useful")
    if metrics["success_ece"] > 0.10:
        reasons.append("success_calibration_above_0.10_ece")
    if counts["changed"] < int(protocol["acceptance"]["minimum_changed"]):
        reasons.append("too_few_executed_changes")
    if coverage < float(protocol["acceptance"]["minimum_coverage"]):
        reasons.append("confirmation_coverage_below_gate")
    harmful_rate = counts["harmful"] / max(counts["changed"], 1)
    if harmful_rate > float(protocol["acceptance"]["harmful_rate_max"]):
        reasons.append("harmful_rate_above_gate")
    if paired_delta <= 0 or ci_low < 0:
        reasons.append("paired_success_improvement_not_confirmed")
    plugin_success_rate = counts["plugin_successes"] / episodes
    target_scratch = float(baseline_by_name["target_from_scratch_matched"]["success_rate"])
    no_factorization = float(baseline_by_name["no_factorization"]["success_rate"])
    if plugin_success_rate <= target_scratch:
        reasons.append("not_better_than_target_from_scratch_matched")
    if plugin_success_rate < no_factorization:
        reasons.append("worse_than_no_factorization")
    if int(primary_row["trainable_parameters"]) >= int(
        baseline_by_name["target_from_scratch_matched"]["trainable_parameters"]
    ):
        reasons.append("no_target_parameter_efficiency")
    authorized = not reasons
    if bool(deployment["action_ranking_authorized"]) != authorized:
        raise ValueError("deployment authorization flag does not equal computed gates")
    return {
        "format": DECISION_FORMAT,
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "weight_audit_sha256": json_sha256(audit),
        "result_sha256": json_sha256(results),
        "axis": protocol["axis"],
        "primary_n_per_task": PRIMARY_ADAPTATION_SIZE,
        "shared_core_immutable": True,
        "prediction_gate_passed": not any(
            reason
            in {
                "next_event_not_above_frozen_baselines",
                "observer_event_not_above_frequency_baseline",
                "observer_predicate_not_above_constant_baseline",
                "observer_coverage_below_0.90",
                "success_brier_not_improved",
                "duration_mae_not_improved",
                "object_delta_mae_not_improved",
                "pair_accuracy_not_above_chance",
                "uncertainty_ranking_not_useful",
                "success_calibration_above_0.10_ece",
            }
            for reason in reasons
        ),
        "paired_success_delta": paired_delta,
        "paired_success_ci95": [ci_low, ci_high],
        "harmful_rate_among_changes": harmful_rate,
        "action_ranking_authorized": authorized,
        "reasons": reasons,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--checkpoint", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", type=Path, required=True)
    audit = subparsers.add_parser("audit-weights")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--before", type=Path, required=True)
    audit.add_argument("--after", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--audit", type=Path, required=True)
    evaluate.add_argument("--results", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "freeze":
        value = freeze_protocol(_read_json(args.draft), args.checkpoint)
        _write_json(args.output, value)
        print(json.dumps({"status": "frozen", "sha256": json_sha256(value)}))
    elif args.command == "validate":
        value = _read_json(args.protocol)
        validate_protocol(value)
        print(json.dumps({"status": "valid", "sha256": json_sha256(value)}))
    elif args.command == "audit-weights":
        protocol = _read_json(args.protocol)
        value = audit_transfer_weights(protocol, args.before, args.after)
        _write_json(args.output, value)
        print(json.dumps({"status": "authorized", "sha256": json_sha256(value)}))
    else:
        protocol = _read_json(args.protocol)
        audit_value = _read_json(args.audit)
        value = evaluate_transfer_results(protocol, audit_value, _read_json(args.results))
        _write_json(args.output, value)
        print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
