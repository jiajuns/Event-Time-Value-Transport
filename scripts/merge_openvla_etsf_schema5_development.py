#!/usr/bin/env python3
"""Fail-closed hard-link merge of schema-v5 development collections.

The formal merge is fixed to legacy official development100 plus the frozen
development-expansion150 collection.  HDF5 files are opened read-only for
identity attrs and hashed byte-for-byte; no label dataset is opened.  A new
output root is atomically published only after all 250 hard links and their
provenance have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py

from robotwin_development_seed_contract import (
    REGISTRY as DEVELOPMENT_REGISTRY,
    official_seeds,
    validate_development_manifest,
)


FORMAT = "etsf_openvla_schema5_development_merge_v1"
MERGED_REGISTRY = "merged_official100_plus_explicit_development150"
LANGUAGE_CONTRACT = (
    "same_instruction_for_initial_query_and_all_candidate_branches"
)
SOURCE_SPECS = (
    {
        "role": "official_development100",
        "groups": 100,
        # The first collection predates the explicit registry field.  The
        # separate official-seed membership proof below is authoritative.
        "registries": (None, "", "official_150"),
        "candidate_count": 4,
        "blends": [0.25, 0.5, 0.75],
    },
    {
        "role": "explicit_development_expansion150",
        "groups": 150,
        "registries": (DEVELOPMENT_REGISTRY,),
        "candidate_count": 5,
        "blends": [0.25, 0.5, 0.75, 1.0],
    },
)
COMMON_CONTRACT_KEYS = (
    "task",
    "body",
    "model_path",
    "unnorm_key",
    "temperature",
    "top_k",
    "preserve_grippers",
    "intervention",
    "language_contract",
    "event_vocab",
    "event_spec_sha256",
    "hidden_dim",
    "hidden_anchor",
    "action_dim",
    "action_chunk",
    "max_steps",
    "trajectory_contract",
    "continuation_query_contract",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _resolve_inside(root: Path, recorded: str) -> Path:
    value = Path(recorded).expanduser()
    candidates = (
        ((value,) if value.is_absolute() else ())
        + (root / "groups" / value, root / value)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"source group path escapes collection root: {recorded}"
            ) from error
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"source group is unavailable: {recorded}")


def _resolve_recorded_artifact(recorded: str, anchor: Path) -> Path:
    value = Path(recorded).expanduser()
    if value.is_file():
        return value.resolve()
    portable = anchor / value.name
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(value)


def _validate_collection(
    root: Path,
    *,
    role: str,
    expected_count: int,
    allowed_registries: Sequence[str | None],
    expected_candidate_count: int,
    expected_blends: Sequence[float],
) -> tuple[Mapping[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    rows = manifest.get("groups")
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("schema_version", -1)) != 5
        or manifest.get("seed_registry") not in allowed_registries
        or not isinstance(rows, list)
        or len(rows) != expected_count
        or int(manifest.get("completed", -1)) != expected_count
    ):
        raise RuntimeError(f"{role} collection completion/registry contract mismatch")
    candidate_count = int(manifest.get("candidate_count", -1))
    blends = [float(value) for value in manifest.get("blends", [])]
    if candidate_count != expected_candidate_count or blends != list(expected_blends):
        raise RuntimeError(f"{role} candidate intervention contract mismatch")
    if manifest.get("fresh_seed_manifest") not in (None, "") or manifest.get(
        "fresh_seed_manifest_sha256"
    ) not in (None, ""):
        raise RuntimeError(f"{role} collection is fresh-confirmation data")
    requested_mirror = [int(value) for value in manifest.get("requested_seeds", [])]
    resolved_mirror = [int(value) for value in manifest.get("resolved_seeds", [])]
    if len(requested_mirror) != expected_count or len(resolved_mirror) != expected_count:
        raise RuntimeError(f"{role} seed mirrors are incomplete")

    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not row.get("path"):
            raise RuntimeError(f"{role} group row is invalid")
        requested = int(row.get("requested_seed", row.get("seed", -1)))
        resolved = int(row.get("resolved_seed", -1))
        if min(requested, resolved) < 0:
            raise RuntimeError(f"{role} group seed identity is invalid")
        path = _resolve_inside(root, str(row["path"]))
        with h5py.File(path, "r") as handle:
            schema = int(handle.attrs.get("schema_version", -1))
            hdf_requested = int(
                handle.attrs.get("requested_seed", handle.attrs.get("seed", -1))
            )
            hdf_resolved = int(handle.attrs.get("resolved_seed", -1))
            task = str(handle.attrs.get("task", manifest.get("task", "")))
            body = str(handle.attrs.get("body", manifest.get("body", "")))
            candidate_count = int(handle.attrs.get("candidate_count", -1))
            language = str(handle.attrs.get("language_contract", ""))
            language_consistent = bool(
                handle.attrs.get("branch_instruction_consistent", False)
            )
        if (
            schema != 5
            or hdf_requested != requested
            or hdf_resolved != resolved
            or task != str(manifest.get("task", ""))
            or body != str(manifest.get("body", ""))
            or candidate_count != int(manifest.get("candidate_count", -1))
            or language != LANGUAGE_CONTRACT
            or not language_consistent
        ):
            raise RuntimeError(f"{role} HDF5 identity/contract mismatch: {path}")
        digest = sha256(path)
        if row.get("sha256") not in (None, "", digest):
            raise RuntimeError(f"{role} recorded HDF5 SHA256 mismatch: {path}")
        stat = path.stat()
        normalized.append(
            {
                "source_index": index,
                "requested_seed": requested,
                "resolved_seed": resolved,
                "task": task,
                "body": body,
                "logical_key": f"{task}|{body}|{resolved}",
                "source_path": str(path),
                "sha256": digest,
                "source_device": int(stat.st_dev),
                "source_inode": int(stat.st_ino),
            }
        )
    requested = [row["requested_seed"] for row in normalized]
    resolved = [row["resolved_seed"] for row in normalized]
    logical = [row["logical_key"] for row in normalized]
    if (
        requested != requested_mirror
        or resolved != resolved_mirror
        or len(set(requested)) != expected_count
        or len(set(resolved)) != expected_count
        or len(set(logical)) != expected_count
    ):
        raise RuntimeError(f"{role} seed/logical mirrors changed")
    audit = {
        "role": role,
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "seed_registry": manifest.get("seed_registry"),
        "seed_registry_audit": (
            "legacy_missing_registry_official_membership_checked_below"
            if manifest.get("seed_registry") in (None, "")
            else str(manifest.get("seed_registry"))
        ),
        "groups": expected_count,
        "candidate_contract": {
            "candidate_count": candidate_count,
            "baseline_candidate_name": "deterministic",
            "blends": blends,
            "temperature": float(manifest["temperature"]),
            "top_k": int(manifest["top_k"]),
        },
    }
    return manifest, normalized, audit


def _development_expansion_contract(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    recorded = str(manifest.get("development_seed_manifest", ""))
    digest = str(manifest.get("development_seed_manifest_sha256", ""))
    path = _resolve_recorded_artifact(recorded, root)
    if not digest or sha256(path) != digest:
        raise RuntimeError("development expansion seed manifest SHA256 mismatch")
    validated = validate_development_manifest(
        path, task=str(manifest.get("task", ""))
    )
    if (
        validated["requested_seeds"]
        != [int(value) for value in manifest.get("requested_seeds", [])]
        or validated["resolved_seeds"]
        != [int(value) for value in manifest.get("resolved_seeds", [])]
    ):
        raise RuntimeError("development expansion collection differs from seed manifest")
    return validated


def _common_contract(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {key: manifests[0].get(key) for key in COMMON_CONTRACT_KEYS}
    missing = [key for key, value in result.items() if value is None]
    if missing:
        raise RuntimeError(f"source collection lacks common contract fields: {missing}")
    for manifest in manifests[1:]:
        changed = [key for key in COMMON_CONTRACT_KEYS if manifest.get(key) != result[key]]
        if changed:
            raise RuntimeError(f"source collection contracts differ: {changed}")
    if result["language_contract"] != LANGUAGE_CONTRACT:
        raise RuntimeError("source language contract is unsupported")
    return result


def merge_development_roots(old100: Path, new150: Path, output: Path) -> dict[str, Any]:
    old100 = old100.expanduser().resolve()
    new150 = new150.expanduser().resolve()
    output = output.expanduser().absolute()
    if old100 == new150:
        raise RuntimeError("development merge inputs must be distinct")
    if output.exists():
        raise FileExistsError(f"merge output already exists; refusing overwrite: {output}")
    for source in (old100, new150):
        try:
            output.resolve().relative_to(source)
        except ValueError:
            pass
        else:
            raise RuntimeError("merge output may not be inside a source collection")

    old_manifest, old_groups, old_audit = _validate_collection(
        old100,
        role=str(SOURCE_SPECS[0]["role"]),
        expected_count=int(SOURCE_SPECS[0]["groups"]),
        allowed_registries=SOURCE_SPECS[0]["registries"],
        expected_candidate_count=int(SOURCE_SPECS[0]["candidate_count"]),
        expected_blends=SOURCE_SPECS[0]["blends"],
    )
    new_manifest, new_groups, new_audit = _validate_collection(
        new150,
        role=str(SOURCE_SPECS[1]["role"]),
        expected_count=int(SOURCE_SPECS[1]["groups"]),
        allowed_registries=SOURCE_SPECS[1]["registries"],
        expected_candidate_count=int(SOURCE_SPECS[1]["candidate_count"]),
        expected_blends=SOURCE_SPECS[1]["blends"],
    )
    common = _common_contract((old_manifest, new_manifest))
    expansion = _development_expansion_contract(new150, new_manifest)
    official = _json(Path(expansion["official_seed_registry"]["path"]))
    official_values = set(official_seeds(official, str(common["task"])))
    if not {group["requested_seed"] for group in old_groups} <= official_values:
        raise RuntimeError("legacy development100 is not a subset of official150")

    old_identity = {
        value for row in old_groups for value in (row["requested_seed"], row["resolved_seed"])
    }
    new_identity = {
        value for row in new_groups for value in (row["requested_seed"], row["resolved_seed"])
    }
    if old_identity & new_identity:
        raise RuntimeError(
            f"development sources overlap requested/resolved scenes: {sorted(old_identity & new_identity)}"
        )
    all_groups = old_groups + new_groups
    if len({row["logical_key"] for row in all_groups}) != 250:
        raise RuntimeError("development sources overlap logical scenes")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent)
    )
    try:
        group_root = temporary / "groups"
        group_root.mkdir()
        merged_rows = []
        for merged_index, row in enumerate(all_groups):
            source_role = (
                SOURCE_SPECS[0]["role"]
                if merged_index < 100
                else SOURCE_SPECS[1]["role"]
            )
            filename = f"group_{merged_index:03d}_seed_{row['requested_seed']}.hdf5"
            destination = group_root / filename
            try:
                os.link(row["source_path"], destination)
            except OSError as error:
                raise RuntimeError(
                    "hard-link merge failed; output must share a filesystem with both sources"
                ) from error
            destination_stat = destination.stat()
            if (
                destination_stat.st_dev != row["source_device"]
                or destination_stat.st_ino != row["source_inode"]
                or sha256(destination) != row["sha256"]
            ):
                raise RuntimeError("merged hard-link identity/SHA256 verification failed")
            merged_rows.append(
                {
                    "index": merged_index,
                    "path": filename,
                    "status": "hardlinked_verified",
                    "source_role": source_role,
                    "source_index": row["source_index"],
                    "seed": row["requested_seed"],
                    "requested_seed": row["requested_seed"],
                    "resolved_seed": row["resolved_seed"],
                    "logical_key": row["logical_key"],
                    "sha256": row["sha256"],
                    "source_path": row["source_path"],
                    "source_device": row["source_device"],
                    "source_inode": row["source_inode"],
                }
            )
        sources = [
            old_audit,
            {
                **new_audit,
                "development_seed_manifest": {
                    "path": expansion["path"],
                    "sha256": expansion["sha256"],
                    "official_seed_registry": expansion["official_seed_registry"],
                    "fresh_seed_exclusion_manifest": expansion["fresh_seed_manifest"],
                    "label_access_contract": expansion["label_access_contract"],
                },
            },
        ]
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "schema_version": 5,
            "status": "complete",
            **common,
            "completed": 250,
            "expected_groups": 250,
            "requested_seeds": [row["requested_seed"] for row in all_groups],
            "resolved_seeds": [row["resolved_seed"] for row in all_groups],
            "seed_registry": MERGED_REGISTRY,
            "seed_registry_contract": {
                "official_development_groups": 100,
                "explicit_development_expansion_groups": 150,
                "fresh_confirmation_eligible": False,
                "fresh_confirmation_access": "forbidden_during_merge_and_oof_development",
                "official150_membership_verified_from_expansion_contract": True,
            },
            "candidate_contract": {
                "variable_candidate_count_supported": True,
                "candidate_count_histogram": {"4": 100, "5": 150},
                "baseline_candidate_name": "deterministic",
                "baseline_index": 0,
                "source_schedules_preserved_without_padding": True,
            },
            "fresh_seed_manifest": None,
            "fresh_seed_manifest_sha256": None,
            "development_seed_manifest": expansion["path"],
            "development_seed_manifest_sha256": expansion["sha256"],
            "source_collections": sources,
            "storage_contract": {
                "mode": "hardlink_no_copy",
                "source_hdf5_open_mode": "read_only_identity_attrs_no_label_datasets",
                "source_sha256_verified": True,
                "destination_inode_matches_source": True,
            },
            "groups": merged_rows,
            "fresh_confirmation_labels_read": False,
        }
        manifest["merge_payload_sha256"] = canonical_sha256(manifest)
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-development100", type=Path, required=True)
    parser.add_argument("--new-development150", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = merge_development_roots(
        args.old_development100, args.new_development150, args.output
    )
    print(
        "SCHEMA5_DEVELOPMENT_MERGED="
        + json.dumps(
            {
                "output": str(args.output.expanduser().absolute()),
                "groups": manifest["completed"],
                "manifest_sha256": sha256(
                    args.output.expanduser().absolute() / "manifest.json"
                ),
                "merge_payload_sha256": manifest["merge_payload_sha256"],
                "fresh_confirmation_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
