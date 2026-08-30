#!/usr/bin/env python3
"""Frozen RoboTwin2 actor execution protocols shared by future consumers.

This module is deliberately independent of collectors, trainers, runners, and
remote experiment results.  It defines the two legal actor execution
protocols that the execute-5/execute-50 deployment study may select.  A
consumer can bind either a stride or the corresponding actor-report method,
validate a serialized protocol fail-closed, and materialize the exact planned
prefix mask used for critic supervision.

The protocol identity covers both sampling allocation and temporal semantics.
In particular, changing the stride changes the primary query grid and the
number of roots collected per condition/query, but not the five-body total of
8,000 four-candidate branches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


FORMAT = "etsf_robotwin2_actor_execution_protocol_v1"
FILE_BINDING_FORMAT = "etsf_robotwin2_actor_execution_protocol_file_binding_v1"
TASK = "move_can_pot"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")

MAX_STEPS = 200
FPS = 15
NATIVE_CHUNK_STEPS = 50
CANDIDATE_COUNT = 4
EXPECTED_TOTAL_BRANCHES = 8_000
SUPPLEMENT_HORIZONS = (10, 25, 50, 100, 200)
LEGAL_STRIDES = (5, 50)

METHOD_EXECUTE5 = "actor_candidate0_execute5_replan"
METHOD_EXECUTE50 = "actor_candidate0_execute50_native"
ACTOR_REPORT_METHOD_BY_STRIDE = MappingProxyType(
    {5: METHOD_EXECUTE5, 50: METHOD_EXECUTE50}
)
STRIDE_BY_ACTOR_REPORT_METHOD = MappingProxyType(
    {method: stride for stride, method in ACTOR_REPORT_METHOD_BY_STRIDE.items()}
)

PROTOCOL_ID_BY_STRIDE = MappingProxyType(
    {5: "execute5_replan", 50: "execute50_native"}
)

# These literal digests are filled from the canonical unsigned protocol
# documents.  Keeping them literal makes a protocol edit an explicit versioned
# contract change instead of silently changing an identity at import time.
_PINNED_LOGICAL_SHA256_BY_STRIDE = MappingProxyType(
    {
        5: "6a3193c79a6f5f88738559711916b68fcfd795f1944bf5fe0cbae31736e98ce8",
        50: "4d9401a2777b9f42b6036cf6ba18729b4fa30c1d88d8f8e11174c47bc9cc65d4",
    }
)


class ActorExecutionProtocolError(ValueError):
    """A protocol, method mapping, schedule entry, or action plan is invalid."""


def sha256_file(path: Path) -> str:
    """Hash one real, non-symbolic protocol file."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ActorExecutionProtocolError("execution protocol file may not be symbolic")
    resolved = expanded.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ActorExecutionProtocolError("execution protocol file must be a real file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used for logical identities."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ActorExecutionProtocolError(
            "value is not finite canonical-JSON data"
        ) from error
    return payload.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash canonical finite JSON with SHA-256."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_plain_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActorExecutionProtocolError(f"{label} must be an integer")
    return value


def _legal_stride(value: Any) -> int:
    stride = _require_plain_int(value, label="stride")
    if stride not in LEGAL_STRIDES:
        raise ActorExecutionProtocolError(
            f"stride must be one of {list(LEGAL_STRIDES)}"
        )
    return stride


def _action_plan_fields(
    stride: int,
    *,
    remaining_action_budget: int,
    horizon: int,
) -> dict[str, Any]:
    planned_steps = min(stride, remaining_action_budget, horizon)
    return {
        "remaining_action_budget": remaining_action_budget,
        "horizon": horizon,
        "planned_steps": planned_steps,
        "planned_dt_seconds": planned_steps / FPS,
        "action_mask": [index < planned_steps for index in range(NATIVE_CHUNK_STEPS)],
    }


def _unsigned_protocol(stride: int) -> dict[str, Any]:
    stride = _legal_stride(stride)
    if MAX_STEPS % stride:
        raise ActorExecutionProtocolError("stride must exactly partition max_steps")
    query_indices = list(range(MAX_STEPS // stride))
    remaining_budgets = [MAX_STEPS - query * stride for query in query_indices]

    allocation_denominator = (
        len(BODIES) * len(CONDITIONS) * len(query_indices) * CANDIDATE_COUNT
    )
    if EXPECTED_TOTAL_BRANCHES % allocation_denominator:
        raise ActorExecutionProtocolError(
            "8,000 branches cannot be allocated evenly over the protocol grid"
        )
    target_per_condition_query = EXPECTED_TOTAL_BRANCHES // allocation_denominator
    groups_per_body = (
        len(CONDITIONS) * len(query_indices) * target_per_condition_query
    )
    branches_per_body = groups_per_body * CANDIDATE_COUNT
    total_branches = branches_per_body * len(BODIES)
    if total_branches != EXPECTED_TOTAL_BRANCHES:
        raise ActorExecutionProtocolError("internal branch allocation changed")

    primary_schedule = []
    for query_index, remaining in zip(
        query_indices, remaining_budgets, strict=True
    ):
        primary_schedule.append(
            {
                "query_index": query_index,
                **_action_plan_fields(
                    stride,
                    remaining_action_budget=remaining,
                    horizon=remaining,
                ),
            }
        )

    supplement_schedule = [
        _action_plan_fields(
            stride,
            remaining_action_budget=horizon,
            horizon=horizon,
        )
        for horizon in SUPPLEMENT_HORIZONS
    ]

    return {
        "format": FORMAT,
        "task": TASK,
        "protocol_id": PROTOCOL_ID_BY_STRIDE[stride],
        "actor_report_method": ACTOR_REPORT_METHOD_BY_STRIDE[stride],
        "stride": stride,
        "max_steps": MAX_STEPS,
        "fps": FPS,
        "native_chunk_steps": NATIVE_CHUNK_STEPS,
        "candidate_count": CANDIDATE_COUNT,
        "bodies": list(BODIES),
        "conditions": list(CONDITIONS),
        "query_indices": query_indices,
        "target_per_condition_query": target_per_condition_query,
        "primary_remaining_action_budgets": remaining_budgets,
        "supplement_horizons": list(SUPPLEMENT_HORIZONS),
        "planning_contract": {
            "planned_steps": "min(stride,remaining_action_budget,horizon)",
            "planned_dt_seconds": "planned_steps/fps",
            "action_mask": (
                "length_native_chunk_steps_boolean_prefix_true_for_planned_steps"
            ),
            "mask_uses_planned_not_observed_executed_steps": True,
        },
        "primary_query_schedule": primary_schedule,
        "supplement_root_schedule": supplement_schedule,
        "branch_accounting": {
            "body_count": len(BODIES),
            "condition_count": len(CONDITIONS),
            "query_count": len(query_indices),
            "target_per_condition_query": target_per_condition_query,
            "candidate_count": CANDIDATE_COUNT,
            "groups_per_body": groups_per_body,
            "branches_per_body": branches_per_body,
            "five_body_total_branches": total_branches,
        },
    }


def execution_protocol(stride: int) -> dict[str, Any]:
    """Return a fresh canonical protocol document for a legal stride."""

    stride = _legal_stride(stride)
    unsigned = _unsigned_protocol(stride)
    actual = canonical_sha256(unsigned)
    pinned = _PINNED_LOGICAL_SHA256_BY_STRIDE[stride]
    if actual != pinned:
        raise ActorExecutionProtocolError(
            f"internal protocol {stride} digest differs from its frozen identity"
        )
    return {**unsigned, "logical_sha256": pinned}


def execution_protocol_for_actor_report_method(method: str) -> dict[str, Any]:
    """Resolve an exact execute-5/execute-50 report method to its protocol."""

    if not isinstance(method, str) or method not in STRIDE_BY_ACTOR_REPORT_METHOD:
        raise ActorExecutionProtocolError("unknown actor report method")
    return execution_protocol(STRIDE_BY_ACTOR_REPORT_METHOD[method])


def validate_execution_protocol(
    value: Any,
    *,
    expected_stride: int | None = None,
    expected_actor_report_method: str | None = None,
) -> dict[str, Any]:
    """Validate a serialized protocol exactly and return an isolated copy.

    Validation is intentionally fail-closed: missing or extra fields, reordered
    schedules, numeric type drift, a recomputed-but-unapproved digest, and any
    nested semantic change are rejected.
    """

    if not isinstance(value, Mapping):
        raise ActorExecutionProtocolError("execution protocol must be a mapping")
    observed = copy.deepcopy(dict(value))
    if set(observed) != set(_unsigned_protocol(5)) | {"logical_sha256"}:
        raise ActorExecutionProtocolError("execution protocol fields changed")

    stride = _legal_stride(observed.get("stride"))
    expected = execution_protocol(stride)
    recorded_sha = observed.get("logical_sha256")
    unsigned = dict(observed)
    unsigned.pop("logical_sha256", None)
    if not isinstance(recorded_sha, str) or canonical_sha256(unsigned) != recorded_sha:
        raise ActorExecutionProtocolError("execution protocol logical SHA is invalid")
    # Canonical bytes, rather than Python equality, distinguish 15 from 15.0
    # and otherwise enforce exact JSON types throughout the nested document.
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ActorExecutionProtocolError(
            "execution protocol is not a frozen legal value"
        )

    if expected_stride is not None and stride != _legal_stride(expected_stride):
        raise ActorExecutionProtocolError("execution protocol stride is not expected")
    if expected_actor_report_method is not None:
        if (
            not isinstance(expected_actor_report_method, str)
            or expected_actor_report_method
            not in STRIDE_BY_ACTOR_REPORT_METHOD
        ):
            raise ActorExecutionProtocolError(
                "expected actor report method is not legal"
            )
        if observed["actor_report_method"] != expected_actor_report_method:
            raise ActorExecutionProtocolError(
                "execution protocol actor report method is not expected"
            )
    return copy.deepcopy(expected)


def load_execution_protocol_file(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_stride: int | None = None,
    expected_actor_report_method: str | None = None,
) -> dict[str, Any]:
    """Load an explicitly byte-bound frozen execution protocol.

    Consumers must bind both identities: ``expected_file_sha256`` protects the
    deployed bytes and the protocol's pinned ``logical_sha256`` protects its
    canonical meaning.  A path alone is deliberately insufficient.
    """

    if (
        not isinstance(expected_file_sha256, str)
        or len(expected_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_file_sha256)
    ):
        raise ActorExecutionProtocolError(
            "expected execution protocol file SHA-256 must be lowercase hexadecimal"
        )
    observed_file_sha256 = sha256_file(path)
    if observed_file_sha256 != expected_file_sha256:
        raise ActorExecutionProtocolError("execution protocol file SHA-256 mismatch")
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ActorExecutionProtocolError(
            "execution protocol file is not valid UTF-8 JSON"
        ) from error
    return validate_execution_protocol(
        value,
        expected_stride=expected_stride,
        expected_actor_report_method=expected_actor_report_method,
    )


def _resolved_path_root(path_root: Path) -> Path:
    expanded = path_root.expanduser()
    if expanded.is_symlink():
        raise ActorExecutionProtocolError("path_root may not be symbolic")
    resolved = expanded.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ActorExecutionProtocolError("path_root must be a real directory")
    return resolved


def execution_protocol_file_binding(
    path: Path,
    expected_file_sha256: str,
    *,
    path_root: Path,
    expected_stride: int | None = None,
    expected_actor_report_method: str | None = None,
) -> dict[str, Any]:
    """Bind one frozen protocol file relative to one explicit artifact root.

    The absolute root is intentionally part of the contract.  All downstream
    artifacts can therefore resolve the same relative path without guessing
    whether it is relative to the artifact file, run directory, or HOME.
    """

    root = _resolved_path_root(path_root)
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ActorExecutionProtocolError("execution protocol file may not be symbolic")
    resolved = expanded.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ActorExecutionProtocolError(
            "execution protocol file must be contained by path_root"
        ) from error
    if not relative.parts:
        raise ActorExecutionProtocolError("execution protocol relative path is empty")
    protocol = load_execution_protocol_file(
        resolved,
        expected_file_sha256,
        expected_stride=expected_stride,
        expected_actor_report_method=expected_actor_report_method,
    )
    return {
        "format": FILE_BINDING_FORMAT,
        "path_root": str(root),
        "path": relative.as_posix(),
        "file_sha256": expected_file_sha256,
        "protocol_logical_sha256": protocol["logical_sha256"],
        "protocol": protocol,
    }


def resolve_execution_protocol_file_binding_path(
    value: Mapping[str, Any],
) -> Path:
    """Resolve only the contained path encoded by a protocol file binding."""

    if not isinstance(value, Mapping):
        raise ActorExecutionProtocolError("execution protocol binding must be a mapping")
    root_raw = value.get("path_root")
    relative_raw = value.get("path")
    if not isinstance(root_raw, str) or not root_raw:
        raise ActorExecutionProtocolError("execution protocol binding path_root is invalid")
    if not isinstance(relative_raw, str) or not relative_raw:
        raise ActorExecutionProtocolError("execution protocol binding path is invalid")
    root = _resolved_path_root(Path(root_raw))
    if root_raw != str(root):
        raise ActorExecutionProtocolError(
            "execution protocol binding path_root must be canonical absolute"
        )
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ActorExecutionProtocolError(
            "execution protocol binding path must be a contained relative path"
        )
    resolved = (root / relative).resolve()
    try:
        canonical_relative = resolved.relative_to(root)
    except ValueError as error:
        raise ActorExecutionProtocolError(
            "execution protocol binding path escapes path_root"
        ) from error
    if relative_raw != canonical_relative.as_posix():
        raise ActorExecutionProtocolError(
            "execution protocol binding path must be canonical relative"
        )
    return resolved


def validate_execution_protocol_file_binding(
    value: Any,
    *,
    expected_path_root: Path | None = None,
    expected_stride: int | None = None,
    expected_actor_report_method: str | None = None,
) -> dict[str, Any]:
    """Validate both identities and the live bytes of a protocol binding."""

    if not isinstance(value, Mapping):
        raise ActorExecutionProtocolError("execution protocol binding must be a mapping")
    observed = copy.deepcopy(dict(value))
    required = {
        "format",
        "path_root",
        "path",
        "file_sha256",
        "protocol_logical_sha256",
        "protocol",
    }
    if set(observed) != required or observed.get("format") != FILE_BINDING_FORMAT:
        raise ActorExecutionProtocolError("execution protocol binding fields changed")
    resolved = resolve_execution_protocol_file_binding_path(observed)
    root = _resolved_path_root(Path(str(observed["path_root"])))
    if expected_path_root is not None:
        expected_root = _resolved_path_root(expected_path_root)
        if root != expected_root:
            raise ActorExecutionProtocolError(
                "execution protocol binding path_root is not expected"
            )
    file_sha = observed.get("file_sha256")
    if not isinstance(file_sha, str):
        raise ActorExecutionProtocolError("execution protocol binding file SHA is invalid")
    live = load_execution_protocol_file(
        resolved,
        file_sha,
        expected_stride=expected_stride,
        expected_actor_report_method=expected_actor_report_method,
    )
    recorded = validate_execution_protocol(
        observed.get("protocol"),
        expected_stride=expected_stride,
        expected_actor_report_method=expected_actor_report_method,
    )
    if (
        observed.get("protocol_logical_sha256") != live["logical_sha256"]
        or canonical_json_bytes(recorded) != canonical_json_bytes(live)
    ):
        raise ActorExecutionProtocolError(
            "execution protocol binding document or logical SHA changed"
        )
    return copy.deepcopy(observed)


def write_execution_protocol_file(path: Path, protocol: Mapping[str, Any]) -> str:
    """Create one canonical protocol file once, or verify its exact bytes."""

    validated = validate_execution_protocol(protocol)
    payload = json.dumps(
        validated,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ActorExecutionProtocolError("execution protocol output may not be symbolic")
    resolved = expanded.resolve()
    if resolved.exists():
        if not resolved.is_file() or resolved.is_symlink() or resolved.read_bytes() != payload:
            raise ActorExecutionProtocolError(
                "existing execution protocol file differs from the frozen value"
            )
        return sha256_file(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".create", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, resolved)
        except FileExistsError:
            if resolved.is_symlink() or not resolved.is_file() or resolved.read_bytes() != payload:
                raise ActorExecutionProtocolError(
                    "racing execution protocol file differs from the frozen value"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(resolved)


def action_plan(
    protocol: Mapping[str, Any],
    *,
    remaining_action_budget: int,
    horizon: int,
) -> dict[str, Any]:
    """Materialize ``min(stride, remaining, horizon)`` and its 50-token mask."""

    validated = validate_execution_protocol(protocol)
    remaining = _require_plain_int(
        remaining_action_budget, label="remaining_action_budget"
    )
    requested_horizon = _require_plain_int(horizon, label="horizon")
    if not 1 <= remaining <= MAX_STEPS:
        raise ActorExecutionProtocolError(
            "remaining_action_budget must be in [1,max_steps]"
        )
    if not 1 <= requested_horizon <= MAX_STEPS:
        raise ActorExecutionProtocolError("horizon must be in [1,max_steps]")
    return {
        "execution_protocol_logical_sha256": validated["logical_sha256"],
        "stride": validated["stride"],
        **_action_plan_fields(
            validated["stride"],
            remaining_action_budget=remaining,
            horizon=requested_horizon,
        ),
    }


def primary_action_plan(
    protocol: Mapping[str, Any], query_index: int
) -> dict[str, Any]:
    """Return the frozen primary plan at one legal logical actor query."""

    validated = validate_execution_protocol(protocol)
    query = _require_plain_int(query_index, label="query_index")
    if query not in validated["query_indices"]:
        raise ActorExecutionProtocolError("query_index is outside the primary grid")
    remaining = MAX_STEPS - query * validated["stride"]
    return {
        "query_index": query,
        **action_plan(
            validated,
            remaining_action_budget=remaining,
            horizon=remaining,
        ),
    }


def supplement_action_plan(
    protocol: Mapping[str, Any],
    horizon: int,
    *,
    remaining_action_budget: int | None = None,
) -> dict[str, Any]:
    """Return a plan for one of the five frozen supplement horizons."""

    requested_horizon = _require_plain_int(horizon, label="horizon")
    if requested_horizon not in SUPPLEMENT_HORIZONS:
        raise ActorExecutionProtocolError("horizon is not a frozen supplement horizon")
    remaining = (
        requested_horizon
        if remaining_action_budget is None
        else remaining_action_budget
    )
    return action_plan(
        protocol,
        remaining_action_budget=remaining,
        horizon=requested_horizon,
    )


EXECUTE5_LOGICAL_SHA256 = _PINNED_LOGICAL_SHA256_BY_STRIDE[5]
EXECUTE50_LOGICAL_SHA256 = _PINNED_LOGICAL_SHA256_BY_STRIDE[50]


__all__ = [
    "ACTOR_REPORT_METHOD_BY_STRIDE",
    "ActorExecutionProtocolError",
    "BODIES",
    "CANDIDATE_COUNT",
    "CONDITIONS",
    "EXECUTE5_LOGICAL_SHA256",
    "EXECUTE50_LOGICAL_SHA256",
    "EXPECTED_TOTAL_BRANCHES",
    "FILE_BINDING_FORMAT",
    "FORMAT",
    "FPS",
    "LEGAL_STRIDES",
    "MAX_STEPS",
    "METHOD_EXECUTE5",
    "METHOD_EXECUTE50",
    "NATIVE_CHUNK_STEPS",
    "STRIDE_BY_ACTOR_REPORT_METHOD",
    "SUPPLEMENT_HORIZONS",
    "action_plan",
    "canonical_json_bytes",
    "canonical_sha256",
    "execution_protocol",
    "execution_protocol_file_binding",
    "execution_protocol_for_actor_report_method",
    "load_execution_protocol_file",
    "primary_action_plan",
    "resolve_execution_protocol_file_binding_path",
    "sha256_file",
    "supplement_action_plan",
    "validate_execution_protocol",
    "validate_execution_protocol_file_binding",
    "write_execution_protocol_file",
]
