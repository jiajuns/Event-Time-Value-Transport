#!/usr/bin/env python3
"""Freeze a source63 causal-observer materialization request.

This program is deliberately an identity/provenance-only pre-stage.  It
partitions the already frozen source63 development split and computes opaque
whole-file SHA256 values for the selected development HDF files.  It never
imports an HDF library and never opens, stats, resolves, or hashes a group in
the original ``test`` split.

The primary output has exactly the field set accepted by
``materialize_smolvla_piper_causal_event_observer_dataset_v1.freeze_request``.
Selection and the zero-contact test exclusion are recorded in a separate,
content-addressed ``<output>.audit.json`` file so the materializer contract is
not weakened with extension fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence


REQUEST_FORMAT = (
    "etsf_smolvla_piper_causal_event_observer_materialization_request_v1"
)
REQUEST_STATUS = "frozen_before_hdf_access"
AUDIT_FORMAT = "etsf_smolvla_causal_observer_source63_request_freeze_audit_v1"
AUDIT_STATUS = "complete_request_frozen_before_any_hdf_open_test_excluded"
SPLIT_FORMAT = "etsf_smolvla_schema5_native_source_split_v1"
SPLIT_STATUS = "frozen_development_split"
SCHEMA_VERSION = 5
EXPECTED_SOURCE_GROUPS = 63
EXPECTED_SPLIT_COUNTS = {"train": 44, "validation": 14, "test": 5}
EXPECTED_HIDDEN_DIM = 960
EXPECTED_ACTION_DIM = 14
EXPECTED_EVENTS = ("e0", "e12", "e3", "e4", "eK")
DEFAULT_CALIBRATION_COUNT = 10
DEFAULT_SOURCE_NAME = "smolvla_source63"
DEFAULT_ACTOR_NAME = "smolvla_aloha_agilex"
DEFAULT_POLICY_FAMILY = "smolvla"
SPLIT_ALGORITHM = "ordered_original_train_tail_fixed_count_v1"
SHA_CHARS = frozenset("0123456789abcdef")
PROTECTED_PATH_MARKERS = ("fresh", "confirmation", "evaluation", "formal")
REQUEST_FIELDS = {
    "format",
    "status",
    "event_spec",
    "actors",
    "sources",
    "splits",
    "split_unit",
    "split_leakage_allowed",
    "privileged_label_source_available_to_model_inputs",
    "future_query_features_available_to_model_inputs",
    "request_sha256",
}


class Source63ObserverRequestError(RuntimeError):
    """The frozen source, split, path, or content-address contract failed."""


@dataclass(frozen=True)
class GroupIdentity:
    index: int
    requested_seed: int
    resolved_seed: int
    relative_path: str


@dataclass(frozen=True)
class FrozenMembership:
    observer_train: tuple[int, ...]
    observer_calibration: tuple[int, ...]
    observer_validation: tuple[int, ...]
    excluded_test: tuple[int, ...]
    calibration_count: int


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_opaque_selected_group(path: Path) -> str:
    """Hash opaque selected bytes without importing or invoking HDF readers."""

    return file_sha256(path)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _reject_protected_path(path: PurePath, role: str) -> None:
    for component in path.parts:
        lowered = component.casefold()
        if any(marker in lowered for marker in PROTECTED_PATH_MARKERS):
            raise Source63ObserverRequestError(
                f"{role} identifies protected fresh/formal/evaluation data"
            )


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            if current.is_symlink():
                raise Source63ObserverRequestError(
                    f"{role} path contains a symbolic link"
                )
        except OSError as error:
            raise Source63ObserverRequestError(f"{role} path is unavailable") from error


def _resolve_existing_file(path: Path, role: str) -> Path:
    _reject_protected_path(path, role)
    _reject_symlink_components(path, role)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise Source63ObserverRequestError(f"{role} is unavailable") from error
    _reject_protected_path(resolved, role)
    if resolved.is_symlink() or not resolved.is_file():
        raise Source63ObserverRequestError(f"{role} is not a regular file")
    return resolved


def _resolve_existing_directory(path: Path, role: str) -> Path:
    _reject_protected_path(path, role)
    _reject_symlink_components(path, role)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise Source63ObserverRequestError(f"{role} is unavailable") from error
    _reject_protected_path(resolved, role)
    if resolved.is_symlink() or not resolved.is_dir():
        raise Source63ObserverRequestError(f"{role} is not a materialized directory")
    return resolved


def _resolve_new_output(path: Path, role: str) -> Path:
    _reject_protected_path(path, role)
    if path.exists() or path.is_symlink():
        raise Source63ObserverRequestError(f"{role} already exists")
    parent = _resolve_existing_directory(path.absolute().parent, f"{role} parent")
    resolved = parent / path.name
    _reject_protected_path(resolved, role)
    if resolved.exists() or resolved.is_symlink():
        raise Source63ObserverRequestError(f"{role} already exists")
    return resolved


def _render_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_bound_json(path: Path, expected_sha256: str, role: str) -> tuple[Path, dict[str, Any]]:
    if not _is_sha(expected_sha256):
        raise Source63ObserverRequestError(f"{role} expected SHA256 is invalid")
    resolved = _resolve_existing_file(path, role)
    if file_sha256(resolved) != expected_sha256:
        raise Source63ObserverRequestError(f"{role} file SHA256 differs")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Source63ObserverRequestError(f"{role} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise Source63ObserverRequestError(f"{role} must be a JSON object")
    return resolved, value


def _validate_split(value: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    if (
        value.get("format") != SPLIT_FORMAT
        or value.get("status") != SPLIT_STATUS
        or value.get("split_unit") != "requested_seed_logical_group"
        or value.get("fresh_inputs_allowed") is not False
        or value.get("fresh_trajectory_or_label_opened") is not False
    ):
        raise Source63ObserverRequestError("frozen source63 split contract differs")
    result: dict[str, tuple[int, ...]] = {}
    for name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        raw = value.get(name)
        if (
            not isinstance(raw, list)
            or len(raw) != expected_count
            or any(not _strict_int(seed) for seed in raw)
            or len(set(raw)) != len(raw)
        ):
            raise Source63ObserverRequestError(
                f"frozen source63 {name} support/count differs"
            )
        result[name] = tuple(int(seed) for seed in raw)
    all_ids = [seed for name in ("train", "validation", "test") for seed in result[name]]
    if len(all_ids) != EXPECTED_SOURCE_GROUPS or len(set(all_ids)) != len(all_ids):
        raise Source63ObserverRequestError("frozen source63 split overlaps or is incomplete")
    for key in ("task", "body", "policy"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise Source63ObserverRequestError(f"frozen source63 split {key} is invalid")
    return result


def freeze_membership(
    split: Mapping[str, Any], *, calibration_count: int
) -> FrozenMembership:
    groups = _validate_split(split)
    if (
        not _strict_int(calibration_count, minimum=1)
        or calibration_count >= len(groups["train"])
    ):
        raise Source63ObserverRequestError(
            "calibration count leaves insufficient independent train/calibration support"
        )
    boundary = len(groups["train"]) - calibration_count
    membership = FrozenMembership(
        observer_train=groups["train"][:boundary],
        observer_calibration=groups["train"][boundary:],
        observer_validation=groups["validation"],
        excluded_test=groups["test"],
        calibration_count=calibration_count,
    )
    selected = [
        *membership.observer_train,
        *membership.observer_calibration,
        *membership.observer_validation,
    ]
    if (
        not membership.observer_train
        or not membership.observer_calibration
        or not membership.observer_validation
        or set(selected) & set(membership.excluded_test)
        or len(selected) != len(set(selected))
    ):
        raise Source63ObserverRequestError("observer split group support is insufficient")
    return membership


def _validate_group_path_text(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise Source63ObserverRequestError(f"schema5 group {index} path is invalid")
    pure = PurePath(value)
    _reject_protected_path(pure, f"schema5 group {index}")
    if (
        pure.is_absolute()
        or pure.name != value
        or pure.suffix.casefold() not in (".hdf5", ".h5")
    ):
        raise Source63ObserverRequestError(
            f"schema5 group {index} path must be one relative HDF filename"
        )
    return value


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    split_groups: Mapping[str, Sequence[int]],
    event_spec_sha256: str,
) -> tuple[dict[int, GroupIdentity], str]:
    expected_order = [
        seed
        for name in ("train", "validation", "test")
        for seed in split_groups[name]
    ]
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("task") != split.get("task")
        or manifest.get("body") != split.get("body")
        or manifest.get("policy") != split.get("policy")
        or manifest.get("requested_seeds") != expected_order
        or manifest.get("event_spec_sha256") != event_spec_sha256
        or manifest.get("hidden_dim") != EXPECTED_HIDDEN_DIM
        or manifest.get("action_dim") != EXPECTED_ACTION_DIM
        or manifest.get("event_vocab") != list(EXPECTED_EVENTS)
        or manifest.get("completed") != EXPECTED_SOURCE_GROUPS
    ):
        raise Source63ObserverRequestError(
            "schema5 manifest/split/event/hash contract differs from frozen source63"
        )
    state_contract = manifest.get("shared_state_contract")
    state_source = (
        state_contract.get("calibration_id")
        if isinstance(state_contract, Mapping)
        else None
    )
    if not _is_sha(state_source):
        raise Source63ObserverRequestError(
            "schema5 shared_state_contract.calibration_id is not a SHA256"
        )
    rows = manifest.get("groups")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_GROUPS:
        raise Source63ObserverRequestError("schema5 manifest group support differs")
    resolved_summary = manifest.get("resolved_seeds")
    if not isinstance(resolved_summary, list) or len(resolved_summary) != len(rows):
        raise Source63ObserverRequestError("schema5 resolved-seed summary is invalid")
    result: dict[int, GroupIdentity] = {}
    resolved_ids: list[int] = []
    paths: set[str] = set()
    for index, (raw, expected_seed) in enumerate(zip(rows, expected_order)):
        if not isinstance(raw, Mapping):
            raise Source63ObserverRequestError(f"schema5 group {index} is invalid")
        seed = raw.get("seed")
        resolved_seed = raw.get("resolved_seed")
        if (
            raw.get("index") != index
            or seed != expected_seed
            or not _strict_int(resolved_seed)
            or raw.get("status") not in ("collected", "existing")
        ):
            raise Source63ObserverRequestError(
                f"schema5 group {index} seed/identity differs from frozen split"
            )
        relative_path = _validate_group_path_text(raw.get("path"), index)
        if relative_path in paths:
            raise Source63ObserverRequestError("schema5 group paths are duplicated")
        paths.add(relative_path)
        resolved_ids.append(int(resolved_seed))
        result[int(seed)] = GroupIdentity(
            index=index,
            requested_seed=int(seed),
            resolved_seed=int(resolved_seed),
            relative_path=relative_path,
        )
    if (
        len(result) != EXPECTED_SOURCE_GROUPS
        or len(set(resolved_ids)) != EXPECTED_SOURCE_GROUPS
        or resolved_summary != resolved_ids
    ):
        raise Source63ObserverRequestError(
            "schema5 requested/resolved seed support or summary differs"
        )
    return result, str(state_source)


def _selected_group_file(group_root: Path, group: GroupIdentity) -> Path:
    # Called only after the immutable membership has excluded all original test
    # IDs.  No caller may use this helper for an excluded test identity.
    candidate = group_root / group.relative_path
    _reject_protected_path(candidate, "selected development HDF")
    _reject_symlink_components(candidate, "selected development HDF")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise Source63ObserverRequestError(
            "selected development HDF is unavailable"
        ) from error
    if group_root not in resolved.parents:
        raise Source63ObserverRequestError("selected development HDF escapes group root")
    if resolved.is_symlink() or not resolved.is_file():
        raise Source63ObserverRequestError(
            "selected development HDF is not a regular file"
        )
    return resolved


def schema5_logical_group_id(
    *, source_name: str, task: str, body: str, policy: str, requested_seed: int
) -> str:
    return (
        f"{source_name}/schema5/{task}/{body}/{policy}/"
        f"requested_seed/{requested_seed}"
    )


def _group_references(
    seeds: Sequence[int],
    *,
    groups: Mapping[int, GroupIdentity],
    hashes: Mapping[int, str],
    source_name: str,
    task: str,
    body: str,
    policy: str,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for seed in seeds:
        if seed not in groups or not _is_sha(hashes.get(seed)):
            raise Source63ObserverRequestError("selected group hash/support is incomplete")
        result.append(
            {
                "source_name": source_name,
                "logical_group_id": schema5_logical_group_id(
                    source_name=source_name,
                    task=task,
                    body=body,
                    policy=policy,
                    requested_seed=seed,
                ),
                "source_file_sha256": hashes[seed],
            }
        )
    return result


def _validate_name(value: str, role: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or "/" in value:
        raise Source63ObserverRequestError(f"{role} is invalid")
    return value


def audit_output_path(output: Path) -> Path:
    return Path(f"{output}.audit.json")


def freeze_source63_request(
    *,
    schema5_manifest: Path,
    schema5_manifest_sha256: str,
    frozen_split: Path,
    frozen_split_sha256: str,
    event_spec: Path,
    event_spec_sha256: str,
    output: Path,
    group_root: Path | None = None,
    calibration_count: int = DEFAULT_CALIBRATION_COUNT,
    source_name: str = DEFAULT_SOURCE_NAME,
    actor_name: str = DEFAULT_ACTOR_NAME,
    policy_family: str = DEFAULT_POLICY_FAMILY,
) -> dict[str, Any]:
    """Freeze, content-address, and publish one exact materializer request."""

    source_name = _validate_name(source_name, "source name")
    actor_name = _validate_name(actor_name, "actor name")
    policy_family = _validate_name(policy_family, "policy family")
    manifest_path, manifest = _read_bound_json(
        schema5_manifest, schema5_manifest_sha256, "schema5 source manifest"
    )
    split_path, split = _read_bound_json(
        frozen_split, frozen_split_sha256, "frozen source63 split"
    )
    event_path, event_value = _read_bound_json(
        event_spec, event_spec_sha256, "event specification"
    )
    if not event_value:
        raise Source63ObserverRequestError("event specification is empty")

    # This immutable identity-only boundary is established before resolving,
    # stating, opening, or hashing any group file.
    split_groups = _validate_split(split)
    membership = freeze_membership(split, calibration_count=calibration_count)
    groups, state_source_sha256 = _validate_manifest(
        manifest,
        split=split,
        split_groups=split_groups,
        event_spec_sha256=event_spec_sha256,
    )
    selected_ids = (
        *membership.observer_train,
        *membership.observer_calibration,
        *membership.observer_validation,
    )
    if set(selected_ids) & set(membership.excluded_test):
        raise Source63ObserverRequestError("original test group escaped exclusion gate")

    root_candidate = group_root if group_root is not None else manifest_path.parent / "groups"
    groups_path = _resolve_existing_directory(root_candidate, "schema5 selected group root")
    output_path = _resolve_new_output(output, "materializer request output")
    audit_path = _resolve_new_output(
        audit_output_path(output_path), "request freeze audit output"
    )

    selected_hashes: dict[int, str] = {}
    for seed in selected_ids:
        selected_path = _selected_group_file(groups_path, groups[seed])
        selected_hashes[seed] = hash_opaque_selected_group(selected_path)
    if set(selected_hashes) != set(selected_ids):
        raise Source63ObserverRequestError("selected development hash set is incomplete")

    task = str(split["task"])
    body = str(split["body"])
    policy = str(split["policy"])
    source_record = {
        "source_name": source_name,
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": schema5_manifest_sha256,
        "manifest_logical_sha256": canonical_sha256(manifest),
        "group_root": str(groups_path),
        "actor_name": actor_name,
    }
    actor_record = {
        "actor_name": actor_name,
        "policy_family": policy_family,
        "body": body,
        "policy": policy,
        "state_feature_source_sha256": state_source_sha256,
    }
    logical_request: dict[str, Any] = {
        "format": REQUEST_FORMAT,
        "status": REQUEST_STATUS,
        "event_spec": {
            "path": str(event_path),
            "file_sha256": event_spec_sha256,
        },
        "actors": [actor_record],
        "sources": [source_record],
        "splits": {
            "train": _group_references(
                membership.observer_train,
                groups=groups,
                hashes=selected_hashes,
                source_name=source_name,
                task=task,
                body=body,
                policy=policy,
            ),
            "calibration": _group_references(
                membership.observer_calibration,
                groups=groups,
                hashes=selected_hashes,
                source_name=source_name,
                task=task,
                body=body,
                policy=policy,
            ),
            "validation": _group_references(
                membership.observer_validation,
                groups=groups,
                hashes=selected_hashes,
                source_name=source_name,
                task=task,
                body=body,
                policy=policy,
            ),
        },
        "split_unit": "logical_reset_group",
        "split_leakage_allowed": False,
        "privileged_label_source_available_to_model_inputs": False,
        "future_query_features_available_to_model_inputs": False,
    }
    request = {**logical_request, "request_sha256": canonical_sha256(logical_request)}
    if set(request) != REQUEST_FIELDS:
        raise AssertionError("internal materializer request field drift")
    request_bytes = _render_json(request)

    audit_without_sha: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "status": AUDIT_STATUS,
        "request": {
            "path": str(output_path),
            "file_sha256": bytes_sha256(request_bytes),
            "request_sha256": request["request_sha256"],
        },
        "inputs": {
            "schema5_manifest": {
                "path": str(manifest_path),
                "file_sha256": schema5_manifest_sha256,
                "logical_sha256": source_record["manifest_logical_sha256"],
            },
            "frozen_split": {
                "path": str(split_path),
                "file_sha256": frozen_split_sha256,
            },
            "event_spec": {
                "path": str(event_path),
                "file_sha256": event_spec_sha256,
            },
            "actor_state_feature_source_sha256": state_source_sha256,
        },
        "split_freeze": {
            "algorithm": SPLIT_ALGORITHM,
            "calibration_count": membership.calibration_count,
            "original_train_requested_seed_ids": list(split_groups["train"]),
            "observer_train_requested_seed_ids": list(membership.observer_train),
            "observer_calibration_requested_seed_ids": list(
                membership.observer_calibration
            ),
            "observer_validation_requested_seed_ids": list(
                membership.observer_validation
            ),
            "excluded_original_test_requested_seed_ids": list(
                membership.excluded_test
            ),
            "membership_frozen_before_selected_file_hashing": True,
            "random_split_seed_used": False,
        },
        "data_access_audit": {
            "selected_development_group_count": len(selected_ids),
            "selected_development_hdf_file_sha256_computed": len(selected_hashes),
            "selected_development_hdf_parsed": False,
            "hdf5_library_imported": False,
            "hdf5_container_opened_count": 0,
            "original_test_groups_excluded_from_all_request_splits": True,
            "original_test_group_paths_resolved": False,
            "original_test_group_files_statted": False,
            "original_test_group_files_opened": 0,
            "original_test_group_files_hashed": 0,
            "original_test_trajectory_or_label_datasets_opened": 0,
            "test_groups_excluded_note": (
                "Original frozen source63 test groups are audit-listed by requested "
                "seed only; their paths and files were never resolved, statted, "
                "opened, parsed, or hashed."
            ),
        },
    }
    audit = {**audit_without_sha, "audit_sha256": canonical_sha256(audit_without_sha)}
    audit_bytes = _render_json(audit)

    _write_bytes_exclusive(output_path, request_bytes)
    try:
        _write_bytes_exclusive(audit_path, audit_bytes)
    except BaseException:
        # The request is unusable without its explicit exclusion audit.  This
        # only removes the file created by this invocation.
        output_path.unlink(missing_ok=True)
        raise
    return {
        "format": AUDIT_FORMAT,
        "status": AUDIT_STATUS,
        "request_path": str(output_path),
        "request_file_sha256": bytes_sha256(request_bytes),
        "request_sha256": request["request_sha256"],
        "audit_path": str(audit_path),
        "audit_file_sha256": bytes_sha256(audit_bytes),
        "audit_sha256": audit["audit_sha256"],
        "split_counts": {
            "train": len(membership.observer_train),
            "calibration": len(membership.observer_calibration),
            "validation": len(membership.observer_validation),
            "excluded_test": len(membership.excluded_test),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema5-manifest", type=Path, required=True)
    parser.add_argument("--schema5-manifest-sha256", required=True)
    parser.add_argument("--frozen-split", type=Path, required=True)
    parser.add_argument("--frozen-split-sha256", required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-root", type=Path)
    parser.add_argument("--calibration-count", type=int, default=DEFAULT_CALIBRATION_COUNT)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--actor-name", default=DEFAULT_ACTOR_NAME)
    parser.add_argument("--policy-family", default=DEFAULT_POLICY_FAMILY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = freeze_source63_request(
        schema5_manifest=args.schema5_manifest,
        schema5_manifest_sha256=args.schema5_manifest_sha256,
        frozen_split=args.frozen_split,
        frozen_split_sha256=args.frozen_split_sha256,
        event_spec=args.event_spec,
        event_spec_sha256=args.event_spec_sha256,
        output=args.output,
        group_root=args.group_root,
        calibration_count=args.calibration_count,
        source_name=args.source_name,
        actor_name=args.actor_name,
        policy_family=args.policy_family,
    )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
