#!/usr/bin/env python3
"""Bind five completed scripted-root collectors for strict LOBO training.

The per-body raw manifests are already canonical supplement manifests.  This
materializer validates the body-local ordered reserve resolution without
opening transition NPZ payloads, requires the complete 5 x 2 x 2 x 5 selected
design, rejects primary-reset seed overlap, and writes the single signed
binding consumed by the trainer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import collect_robotwin2_scripted_expert_root_actor_branches_v1 as collector
import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer


FORMAT = "etsf_robotwin2_scripted_expert_root_supplement_binding_materializer_v1"


class SupplementBindingError(RuntimeError):
    """A raw supplement or its primary/actor authority is not exact."""


def _read_signed(path: Path, label: str) -> tuple[dict[str, Any], str]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SupplementBindingError(f"{label} may not be a symbolic link")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise SupplementBindingError(f"{label} must be a real JSON file")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupplementBindingError(f"{label} must be a JSON object")
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    if logical != trainer.canonical_sha256(unsigned):
        raise SupplementBindingError(f"{label} logical SHA-256 mismatch")
    return value, trainer.sha256_file(resolved)


def parse_body_manifest_bindings(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        body, separator, path = str(raw).partition("=")
        if not separator or body not in trainer.BODIES or not path or body in result:
            raise SupplementBindingError(
                "body manifests must be five unique BODY=/path/manifest.json entries"
            )
        result[body] = Path(os.path.abspath(Path(path).expanduser()))
    if set(result) != set(trainer.BODIES):
        raise SupplementBindingError("all five body manifests are required")
    return result


def _contained_relative(parent: Path, child: Path, label: str) -> str:
    resolved_parent = parent.expanduser().resolve()
    resolved_child = child.expanduser().resolve()
    if resolved_child.is_symlink():
        raise SupplementBindingError(f"{label} may not be a symbolic link")
    try:
        relative = resolved_child.relative_to(resolved_parent)
    except ValueError as error:
        raise SupplementBindingError(
            f"{label} must be inside the supplement binding directory"
        ) from error
    if not relative.parts:
        raise SupplementBindingError(f"{label} path is empty")
    return relative.as_posix()


def _primary_seed_pairs(primary: Mapping[str, Any], primary_dir: Path) -> dict[str, set[tuple[str, int]]]:
    bindings = primary.get("body_manifests")
    if not isinstance(bindings, Mapping) or set(bindings) != set(trainer.BODIES):
        raise SupplementBindingError("primary binding lacks five body manifests")
    result: dict[str, set[tuple[str, int]]] = {}
    for body in trainer.BODIES:
        item = bindings[body]
        if not isinstance(item, Mapping):
            raise SupplementBindingError(f"primary manifest binding is invalid for {body}")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise SupplementBindingError("primary manifest path is not contained")
        manifest_path = (primary_dir / relative).resolve()
        manifest, observed_sha = _read_signed(
            manifest_path, f"{body} primary body manifest"
        )
        if observed_sha != item.get("sha256"):
            raise SupplementBindingError(f"{body} primary manifest SHA-256 mismatch")
        groups = manifest.get("groups")
        if not isinstance(groups, list):
            raise SupplementBindingError(f"{body} primary groups are missing")
        pairs: set[tuple[str, int]] = set()
        for group in groups:
            if not isinstance(group, Mapping):
                raise SupplementBindingError(f"{body} primary group is invalid")
            condition = group.get("condition")
            seed = group.get("requested_seed")
            if (
                condition not in trainer.CONDITIONS
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                raise SupplementBindingError(f"{body} primary group identity is invalid")
            pairs.add((str(condition), int(seed)))
        result[body] = pairs
    return result


def _validate_complete_design(
    value: Mapping[str, Any], *, body: str
) -> set[tuple[str, int]]:
    try:
        return collector.validate_completed_design_metadata(value, body=body)
    except collector.ScriptedRootCollectionError as error:
        raise SupplementBindingError(
            f"{body} supplement is not the complete ordered-reserve 20-decision design: {error}"
        ) from error


def build_binding(
    *,
    primary_binding_path: Path,
    actor_authority_path: Path,
    body_manifest_paths: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    """Validate five raw manifests and return the signed trainer binding."""

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    primary, primary_sha = _read_signed(primary_binding_path, "primary binding")
    if (
        primary.get("format") != trainer.BINDING_FORMAT
        or primary.get("dataset_repo") != trainer.DATASET_REPO
        or primary.get("dataset_revision") != trainer.DATASET_REVISION
        or primary.get("task") != trainer.TASK
        or primary.get("instruction") != trainer.DEFAULT_INSTRUCTION
    ):
        raise SupplementBindingError("primary binding identity changed")
    try:
        primary_audit = trainer.load_binding(primary_binding_path, primary_sha)
    except trainer.FiveBodyContractError as error:
        raise SupplementBindingError(
            f"primary binding failed the trainer contract: {error}"
        ) from error
    actor_authority, actor_authority_sha = _read_signed(
        actor_authority_path, "actor authority"
    )
    declared_actor = primary.get("actor_authority")
    if (
        actor_authority.get("format") != trainer.ACTOR_FORMAT
        or actor_authority.get("task") != trainer.TASK
        or not isinstance(declared_actor, Mapping)
        or declared_actor.get("sha256") != actor_authority_sha
    ):
        raise SupplementBindingError(
            "actor authority is not the exact primary binding authority"
        )
    primary_pairs = _primary_seed_pairs(primary, primary_binding_path.resolve().parent)

    body_bindings: dict[str, dict[str, Any]] = {}
    event_implementations: set[str] = set()
    rejected_attempt_count = 0
    selected_seed_count = 0
    for body in trainer.BODIES:
        manifest_path = body_manifest_paths.get(body)
        if manifest_path is None:
            raise SupplementBindingError(f"missing {body} raw supplement manifest")
        manifest, manifest_sha = _read_signed(
            manifest_path, f"{body} raw supplement manifest"
        )
        if manifest.get("actor_authority_sha256") != actor_authority_sha:
            raise SupplementBindingError(
                f"{body} raw manifest does not bind the primary actor authority"
            )
        supplement_pairs = _validate_complete_design(manifest, body=body)
        try:
            validated = trainer.validate_supplement_body_manifest(
                manifest,
                expected_body=body,
                manifest_dir=manifest_path.resolve().parent,
                expected_actor_checkpoint_sha256=str(
                    primary_audit["actor"]["checkpoint_sha256_by_body"][body]
                ),
            )
        except trainer.FiveBodyContractError as error:
            raise SupplementBindingError(
                f"{body} raw manifest is not trainer-compatible: {error}"
            ) from error
        overlap = primary_pairs[body].intersection(supplement_pairs)
        if overlap:
            raise SupplementBindingError(
                f"{body} supplement seeds overlap primary groups: {sorted(overlap)}"
            )
        event_implementations.add(
            str(validated["event_derivation_implementation_sha256"])
        )
        rejected_attempt_count += sum(
            1
            for attempt in manifest["attempts"]
            if attempt.get("status") == "rejected_before_actor_outcomes"
        )
        selected_seed_count += len(manifest["selected_seed_by_slot"])
        body_bindings[body] = {
            "path": _contained_relative(
                output.parent, manifest_path, f"{body} supplement manifest"
            ),
            "sha256": manifest_sha,
            "group_count": len(validated["groups"]),
            "selected_seed_by_slot_sha256": trainer.canonical_sha256(
                manifest["selected_seed_by_slot"]
            ),
            "reserve_roster_sha256": trainer.canonical_sha256(
                manifest["reserve_roster"]
            ),
        }
    if len(event_implementations) != 1:
        raise SupplementBindingError(
            "five supplement bodies do not share one event implementation"
        )

    binding: dict[str, Any] = {
        "format": trainer.SUPPLEMENT_BINDING_FORMAT,
        "dataset_repo": trainer.DATASET_REPO,
        "dataset_revision": trainer.DATASET_REVISION,
        "task": trainer.TASK,
        "instruction": trainer.DEFAULT_INSTRUCTION,
        "event_spec_sha256": trainer.EVENT_SPEC_SHA256,
        "candidate_noise_contract": trainer.CANDIDATE_NOISE_CONTRACT,
        "terminal_supervision_contract": trainer.TERMINAL_SUPERVISION_CONTRACT,
        "event_age_contract": trainer.EVENT_AGE_CONTRACT,
        "terminal_horizon_contract": trainer.TERMINAL_HORIZON_CONTRACT,
        "branch_root_snapshot_contract": trainer.BRANCH_ROOT_SNAPSHOT_CONTRACT,
        "object_effect_schema": trainer.OBJECT_EFFECT_SCHEMA,
        "branch_diagnostic_contract": trainer.BRANCH_DIAGNOSTIC_CONTRACT,
        "primary_binding_file_sha256": primary_sha,
        "actor_authority_sha256": actor_authority_sha,
        "proper_loss_weight": trainer.SUPPLEMENT_PROPER_LOSS_WEIGHT,
        "usage_contract": trainer.SUPPLEMENT_USAGE_CONTRACT,
        "expert_root_provenance_contract": trainer.EXPERT_ROOT_PROVENANCE_CONTRACT,
        "body_manifests": body_bindings,
        "materializer_provenance": {
            "format": FORMAT,
            "payload_npz_files_opened": 0,
            "complete_decisions": collector.EXPECTED_FIVE_BODY_DECISIONS,
            "complete_branches": collector.EXPECTED_FIVE_BODY_BRANCHES,
            "seed_overlap_with_primary": 0,
            "selected_seed_count": selected_seed_count,
            "rejected_attempt_count": rejected_attempt_count,
            "selection_occurs_before_actor_candidate_outcomes": True,
            "heldout_payload_npz_files_opened": 0,
        },
    }
    binding["logical_sha256"] = trainer.canonical_sha256(binding)
    return binding


def write_binding_create_once(path: Path, binding: Mapping[str, Any]) -> bool:
    """Create a binding once; an existing path is accepted only byte-identically.

    Returns ``True`` when this call created the output and ``False`` for an
    already-identical binding.  Neither the final path nor a stale partial is
    ever overwritten.
    """

    expanded = Path(os.path.abspath(path.expanduser()))
    if expanded.is_symlink():
        raise SupplementBindingError(
            "supplement binding output may not be a symbolic link"
        )
    output = expanded.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(binding), indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    if output.exists():
        if not output.is_file() or output.is_symlink() or output.read_bytes() != payload:
            raise SupplementBindingError(
                "existing supplement binding differs; refusing to overwrite"
            )
        return False

    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        if not partial.is_file() or partial.is_symlink() or partial.read_bytes() != payload:
            raise SupplementBindingError(
                "stale supplement binding partial differs; refusing to overwrite"
            )
    else:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    try:
        os.link(partial, output)
    except FileExistsError:
        if not output.is_file() or output.is_symlink() or output.read_bytes() != payload:
            raise SupplementBindingError(
                "supplement binding appeared concurrently with different content"
            )
        created = False
    else:
        created = True
    partial.unlink()
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-binding", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument(
        "--body-manifest",
        action="append",
        required=True,
        help="repeat exactly five times as BODY=/contained/path/manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    body_manifests = parse_body_manifest_bindings(args.body_manifest)
    binding = build_binding(
        primary_binding_path=args.primary_binding,
        actor_authority_path=args.actor_authority,
        body_manifest_paths=body_manifests,
        output_path=args.output,
    )
    output = args.output.expanduser().resolve()
    created = write_binding_create_once(output, binding)
    print(
        "SUPPLEMENT_BINDING="
        + json.dumps(
            {
                "path": str(output),
                "sha256": trainer.sha256_file(output),
                "created": created,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
