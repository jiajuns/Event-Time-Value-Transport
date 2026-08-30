#!/usr/bin/env python3
"""Train a group-split, multi-embodiment canonical event world model.

The module joins two existing ETSF lines without pretending their checkpoints
are shape-compatible:

* Stage1 supplies embodiment-neutral object geometry and, for Aloha/ARX, 14-D
  expert actions.
* OpenVLA schema-5 branch groups supply action-conditioned Piper transitions.

All train/validation/test assignment is computed from the label-free tuple
``(body, policy, task, seed)`` before an episode/group HDF5 is opened.  Test
files are never opened by this program.  Paths containing ``fresh`` or
``confirmation`` are rejected, including through symlinks.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


CANONICAL_EVENTS = ("e0", "e12", "e3", "e4", "eK")
EVENT_TO_ID = {name: index for index, name in enumerate(CANONICAL_EVENTS)}
TASKS = (
    "adjust_bottle",
    "handover_block",
    "move_can_pot",
    "place_container_plate",
    "beat_block_hammer",
    "lift_pot",
)
TASK_TO_ID = {name: index for index, name in enumerate(TASKS)}
ACTION_SCHEMAS = {"aloha": 0, "arx": 1, "openvla": 2}
ACTION_SCHEMA_NAMES = {value: key for key, value in ACTION_SCHEMAS.items()}
# This table is deliberately finite.  Adding a new spelling is a protocol
# change that must be reviewed instead of silently creating a new body row.
CANONICAL_BODY_ALIASES = {
    "aloha-agilex": "aloha-agilex",
    "ARX-X5": "ARX-X5",
    "piper": "piper",
    "piper_piper_0.6": "piper",
    "ur5-wsg": "ur5-wsg",
}
FORBIDDEN_PATH_TOKENS = ("fresh", "confirmation")
STATE_DIM = 27
ACTION_DIM = 14
OBJECT_DELTA_DIM = 6
SEMANTIC_DIM = 96
FORMAT = "etsf_multibody_canonical_event_world_model_v1"


def canonical_event_name(value: str) -> str:
    """Map raw Stage1 event names to the frozen five-event vocabulary."""

    normalized = str(value).strip()
    if normalized in {"e1", "e2", "e12"}:
        return "e12"
    if normalized not in EVENT_TO_ID:
        raise ValueError(f"unknown event {value!r}")
    return normalized


def canonical_event_id(value: str) -> int:
    return EVENT_TO_ID[canonical_event_name(value)]


def canonical_body_name(value: str) -> str:
    """Return an audited canonical body id or fail closed."""

    raw = str(value).strip()
    try:
        return CANONICAL_BODY_ALIASES[raw]
    except KeyError as error:
        raise ValueError(f"unknown body identity {value!r}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def reject_forbidden_path(path: Path, name: str) -> Path:
    """Resolve and reject any active input/output path in a held-out namespace."""

    resolved = path.expanduser().resolve()
    lowered = str(resolved).lower()
    token = next((item for item in FORBIDDEN_PATH_TOKENS if item in lowered), None)
    if token is not None:
        raise ValueError(f"{name} contains forbidden path token {token!r}")
    return resolved


@dataclasses.dataclass(frozen=True)
class InputBinding:
    stage1_root: Path
    stage1_source_manifest: Path
    stage1_source_manifest_sha256: str
    stage1_target_manifest: Path
    stage1_target_manifest_sha256: str
    event_spec: Path
    event_spec_sha256: str
    openvla_schema5_manifest: Path
    openvla_schema5_manifest_sha256: str


@dataclasses.dataclass(frozen=True)
class GroupDescriptor:
    source: str
    body: str
    policy: str
    task: str
    seed: int
    path: Path
    raw_body: str | None = None
    auxiliary_path: Path | None = None
    episode_index: int | None = None

    @property
    def logical_group(self) -> str:
        # Source is deliberately absent: the protocol estimand is exactly the
        # body/policy/task/seed identity requested in the preregistration.
        return f"{self.body}|{self.policy}|{self.task}|{self.seed}"

    @property
    def observed_raw_body(self) -> str:
        return self.body if self.raw_body is None else self.raw_body

    @property
    def stratum(self) -> str:
        return f"{self.body}|{self.policy}|{self.task}"


def verify_input_bindings(binding: InputBinding) -> dict[str, Any]:
    """Verify immutable inputs without opening any rollout/group HDF5."""

    paths = {
        "stage1_root": reject_forbidden_path(binding.stage1_root, "stage1 root"),
        "stage1_source_manifest": reject_forbidden_path(
            binding.stage1_source_manifest, "stage1 source manifest"
        ),
        "stage1_target_manifest": reject_forbidden_path(
            binding.stage1_target_manifest, "stage1 target manifest"
        ),
        "event_spec": reject_forbidden_path(binding.event_spec, "event spec"),
        "openvla_schema5_manifest": reject_forbidden_path(
            binding.openvla_schema5_manifest, "OpenVLA schema5 manifest"
        ),
    }
    if not paths["stage1_root"].is_dir():
        raise FileNotFoundError(paths["stage1_root"])
    expected = {
        "stage1_source_manifest": binding.stage1_source_manifest_sha256,
        "stage1_target_manifest": binding.stage1_target_manifest_sha256,
        "event_spec": binding.event_spec_sha256,
        "openvla_schema5_manifest": binding.openvla_schema5_manifest_sha256,
    }
    actual: dict[str, str] = {}
    for name, digest in expected.items():
        if not _is_digest(digest):
            raise ValueError(f"{name} expected SHA-256 is malformed")
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256_file(path)
        if actual[name] != digest:
            raise ValueError(f"{name} SHA-256 mismatch")

    event_spec = json.loads(paths["event_spec"].read_text(encoding="utf-8"))
    if not set(TASKS).issubset(event_spec.get("calibration", {})):
        raise ValueError("event spec does not calibrate all six canonical tasks")
    source_manifest = json.loads(
        paths["stage1_source_manifest"].read_text(encoding="utf-8")
    )
    if not isinstance(source_manifest.get("entries"), list):
        raise ValueError("Stage1 source manifest lacks entries")
    schema5 = json.loads(
        paths["openvla_schema5_manifest"].read_text(encoding="utf-8")
    )
    if int(schema5.get("schema_version", -1)) != 5:
        raise ValueError("OpenVLA manifest is not schema 5")
    if schema5.get("status") != "complete" or not schema5.get("groups"):
        raise ValueError("OpenVLA schema5 collection is incomplete")
    if schema5.get("event_spec_sha256") != binding.event_spec_sha256:
        raise ValueError("OpenVLA manifest binds a different event spec")
    group_root = paths["openvla_schema5_manifest"].parent / "groups"
    group_paths: set[Path] = set()
    for item in schema5["groups"]:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("schema5 group path must be relative and contained")
        path = reject_forbidden_path(group_root / relative, "schema5 group")
        try:
            path.relative_to(group_root.resolve())
        except ValueError as error:
            raise ValueError("schema5 group escapes its group root") from error
        if path in group_paths:
            raise ValueError("duplicate schema5 group path")
        if not path.is_file():
            raise FileNotFoundError(path)
        group_paths.add(path)
    return {
        "format": FORMAT,
        "input_sha256": actual,
        "event_spec_sha256": binding.event_spec_sha256,
        "schema5_groups": len(group_paths),
        "schema5_group_hdf5_opened": 0,
        "sealed_test_group_hdf5_opened": 0,
        "dormant_manifest_references_dereferenced": False,
    }


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def scan_stage1_groups(binding: InputBinding) -> list[GroupDescriptor]:
    """Scan label-free Stage1 identities; episode payloads stay unopened."""

    root = reject_forbidden_path(binding.stage1_root, "stage1 root")
    source = json.loads(
        binding.stage1_source_manifest.read_text(encoding="utf-8")
    )
    rows: list[GroupDescriptor] = []
    for entry in source["entries"]:
        task = str(entry["task"])
        raw_body = str(entry["embodiment"])
        body = canonical_body_name(raw_body)
        if task not in TASK_TO_ID or body not in {"aloha-agilex", "ARX-X5"}:
            raise ValueError(f"unsupported Stage1 source identity {task}/{raw_body}")
        source_root = reject_forbidden_path(Path(str(entry["path"])), "Stage1 source entry")
        seed_path = source_root / "seed.txt"
        if not seed_path.is_file():
            raise FileNotFoundError(seed_path)
        seeds = [int(value) for value in seed_path.read_text().split()]
        episode_count = int(entry.get("episodes", len(seeds)))
        if episode_count != len(seeds):
            raise ValueError("Stage1 source episode/seed count mismatch")
        for index, seed in enumerate(seeds):
            action_path = source_root / "data" / f"episode{index}.hdf5"
            pose_path = root / "source_object_poses" / task / body / f"episode_{index:06d}.npz"
            if not action_path.is_file() or not pose_path.is_file():
                raise FileNotFoundError(f"missing Stage1 source pair {action_path} / {pose_path}")
            rows.append(
                GroupDescriptor(
                    source="stage1_source",
                    body=body,
                    policy="robotwin_expert",
                    task=task,
                    seed=seed,
                    path=pose_path,
                    raw_body=raw_body,
                    auxiliary_path=action_path,
                    episode_index=index,
                )
            )

    # Only identity/path columns are retained.  Outcome and trajectory columns
    # in this development manifest never participate in split assignment.
    with binding.stage1_target_manifest.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            if int(item.get("valid_rollout", "0")) != 1:
                continue
            task = str(item["task"])
            raw_body = str(item["embodiment"])
            body = canonical_body_name(raw_body)
            path = reject_forbidden_path(Path(item["path"]), "Stage1 target episode")
            if task not in TASK_TO_ID or body not in {"piper", "ur5-wsg"}:
                raise ValueError(f"unsupported Stage1 target identity {task}/{raw_body}")
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                GroupDescriptor(
                    source="stage1_target",
                    body=body,
                    policy="robotwin_scripted",
                    task=task,
                    seed=int(item["seed"]),
                    path=path,
                    raw_body=raw_body,
                    episode_index=int(item["rollout_index"]),
                )
            )
    return _validate_descriptors(rows)


def scan_schema5_groups(binding: InputBinding) -> list[GroupDescriptor]:
    """Read only group identity/path fields from the schema5 manifest."""

    path = reject_forbidden_path(
        binding.openvla_schema5_manifest, "OpenVLA schema5 manifest"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    task = str(manifest["task"])
    raw_body = str(manifest["body"])
    body = canonical_body_name(raw_body)
    policy = str(manifest.get("model_path", "openvla"))
    group_root = path.parent / "groups"
    rows = []
    for item in manifest["groups"]:
        seed = int(item.get("resolved_seed", item.get("seed")))
        group_path = reject_forbidden_path(
            group_root / str(item["path"]), "OpenVLA schema5 group"
        )
        rows.append(
            GroupDescriptor(
                source="openvla_schema5",
                body=body,
                policy=policy,
                task=task,
                seed=seed,
                path=group_path,
                raw_body=raw_body,
                episode_index=int(item.get("index", len(rows))),
            )
        )
    return _validate_descriptors(rows)


def _validate_descriptors(rows: Sequence[GroupDescriptor]) -> list[GroupDescriptor]:
    if not rows:
        raise ValueError("no label-free groups were discovered")
    seen: set[str] = set()
    for row in rows:
        expected_body = canonical_body_name(row.observed_raw_body)
        if row.body != expected_body:
            raise ValueError(
                f"descriptor body alias mismatch {row.observed_raw_body!r} -> "
                f"{row.body!r}, expected {expected_body!r}"
            )
        if row.logical_group in seen:
            raise ValueError(f"duplicate logical group {row.logical_group}")
        seen.add(row.logical_group)
    return list(rows)


def body_alias_receipt(
    descriptors: Sequence[GroupDescriptor],
) -> dict[str, Any]:
    """Freeze every raw body spelling observed by the current input set."""

    observed: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for row in descriptors:
        raw = row.observed_raw_body
        canonical = canonical_body_name(raw)
        if row.body != canonical:
            raise ValueError("descriptor contains a noncanonical body id")
        previous = observed.setdefault(raw, canonical)
        if previous != canonical:
            raise RuntimeError("one raw body spelling maps to multiple canonical ids")
        counts[raw] += 1
    payload = {
        "format": "etsf_canonical_body_alias_v1",
        "unknown_body_policy": "fail_closed",
        "raw_to_canonical": dict(sorted(observed.items())),
        "raw_group_counts": dict(sorted(counts.items())),
        "canonical_body_ids": sorted(set(observed.values())),
    }
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def strict_group_split(
    descriptors: Sequence[GroupDescriptor],
    *,
    split_seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.10,
) -> dict[str, list[GroupDescriptor]]:
    """Deterministically split within each body/policy/task stratum, label-free."""

    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation/test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation plus test fraction must be below one")
    unique = _validate_descriptors(descriptors)
    strata: dict[str, list[GroupDescriptor]] = defaultdict(list)
    for row in unique:
        strata[row.stratum].append(row)
    result: dict[str, list[GroupDescriptor]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for stratum, rows in sorted(strata.items()):
        if len(rows) < 3:
            raise ValueError(f"stratum {stratum!r} has fewer than three groups")
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{split_seed}|{row.logical_group}".encode("utf-8")
            ).hexdigest(),
        )
        test_count = max(1, int(round(len(ordered) * test_fraction)))
        validation_count = max(1, int(round(len(ordered) * validation_fraction)))
        if test_count + validation_count >= len(ordered):
            validation_count = 1
            test_count = 1
        result["test"].extend(ordered[:test_count])
        result["validation"].extend(
            ordered[test_count : test_count + validation_count]
        )
        result["train"].extend(ordered[test_count + validation_count :])
    memberships = {
        name: {row.logical_group for row in values} for name, values in result.items()
    }
    if any(
        memberships[left] & memberships[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("logical group leakage across splits")
    if set.union(*memberships.values()) != {row.logical_group for row in unique}:
        raise RuntimeError("split omitted one or more logical groups")
    return result


def split_receipt(splits: Mapping[str, Sequence[GroupDescriptor]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split_unit": "body_policy_task_seed_logical_group",
        "labels_used_for_split": False,
        "sealed_test_group_hdf5_opened": 0,
    }
    for name in ("train", "validation", "test"):
        identities = sorted(row.logical_group for row in splits[name])
        result[f"{name}_groups"] = len(identities)
        result[f"{name}_identity_sha256"] = canonical_json_sha256(identities)
    return result


def logical_group_bootstrap_weights(
    groups: Sequence[str], *, members: int = 5, seed: int
) -> np.ndarray:
    """Return Poisson bootstrap weights, constant within each logical group."""

    if members != 5:
        raise ValueError("formal epistemic ensemble requires exactly five members")
    unique = sorted(set(groups))
    if not unique:
        raise ValueError("cannot bootstrap an empty group list")
    generator = np.random.default_rng(seed)
    group_weights = generator.poisson(1.0, size=(members, len(unique))).astype(np.float32)
    for member in range(members):
        if not np.any(group_weights[member]):
            group_weights[member, member % len(unique)] = 1.0
    lookup = {group: index for index, group in enumerate(unique)}
    return np.stack(
        [[group_weights[member, lookup[group]] for group in groups] for member in range(members)]
    ).astype(np.float32)


def fit_train_action_normalization(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_schema_ids: Sequence[int] = tuple(sorted(ACTION_SCHEMA_NAMES)),
) -> dict[str, Any]:
    """Fit valid-action statistics from train rows only, independently by schema."""

    accumulators = {
        int(schema): {
            "sum": np.zeros(ACTION_DIM, dtype=np.float64),
            "sum_square": np.zeros(ACTION_DIM, dtype=np.float64),
            "valid_steps": 0,
            "rows": 0,
            "logical_groups": set(),
        }
        for schema in required_schema_ids
    }
    unavailable_rows = 0
    for row in rows:
        available = bool(row["action_available"])
        if not available:
            unavailable_rows += 1
            continue
        schema = int(row["action_schema_id"])
        if schema not in accumulators:
            raise ValueError(f"train row uses unsupported action schema id {schema}")
        actions = np.asarray(row["actions"], dtype=np.float64)
        mask = np.asarray(row["action_mask"], dtype=bool)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError("train action rows must be [H,14]")
        if mask.shape != actions.shape[:1] or not mask.any():
            raise ValueError("available train action row has an invalid mask")
        valid = actions[mask]
        if not np.isfinite(valid).all():
            raise ValueError("train actions contain non-finite values")
        accumulator = accumulators[schema]
        accumulator["sum"] += valid.sum(axis=0)
        accumulator["sum_square"] += np.square(valid).sum(axis=0)
        accumulator["valid_steps"] += len(valid)
        accumulator["rows"] += 1
        accumulator["logical_groups"].add(str(row["logical_group"]))

    means = np.zeros((len(ACTION_SCHEMA_NAMES), ACTION_DIM), dtype=np.float32)
    stds = np.ones_like(means)
    schemas: dict[str, Any] = {}
    for schema in required_schema_ids:
        accumulator = accumulators[int(schema)]
        count = int(accumulator["valid_steps"])
        if count <= 0:
            raise ValueError(
                f"train split has no valid actions for schema "
                f"{ACTION_SCHEMA_NAMES[int(schema)]!r}"
            )
        mean64 = accumulator["sum"] / count
        variance64 = np.maximum(
            accumulator["sum_square"] / count - np.square(mean64), 0.0
        )
        std64 = np.maximum(np.sqrt(variance64), 1e-4)
        means[int(schema)] = mean64.astype(np.float32)
        stds[int(schema)] = std64.astype(np.float32)
        schemas[ACTION_SCHEMA_NAMES[int(schema)]] = {
            "schema_id": int(schema),
            "train_rows": int(accumulator["rows"]),
            "train_logical_groups": len(accumulator["logical_groups"]),
            "valid_action_steps": count,
            "mean": means[int(schema)].tolist(),
            "std": stds[int(schema)].tolist(),
        }
    receipt = {
        "format": "etsf_train_only_action_normalization_v1",
        "source_split": "train_only",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "unavailable_train_rows_excluded": unavailable_rows,
        "schemas": schemas,
    }
    receipt["sha256"] = canonical_json_sha256(receipt)
    return receipt


def action_normalization_arrays(
    receipt: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and restore normalization arrays from a signed receipt."""

    unsigned = dict(receipt)
    digest = unsigned.pop("sha256", None)
    if digest != canonical_json_sha256(unsigned):
        raise ValueError("action normalization receipt SHA-256 mismatch")
    if receipt.get("source_split") != "train_only":
        raise ValueError("action normalization is not train-only")
    means = np.zeros((len(ACTION_SCHEMA_NAMES), ACTION_DIM), dtype=np.float32)
    stds = np.ones_like(means)
    schemas = receipt.get("schemas")
    if not isinstance(schemas, Mapping):
        raise ValueError("action normalization receipt lacks schemas")
    for schema_id, name in sorted(ACTION_SCHEMA_NAMES.items()):
        item = schemas.get(name)
        if not isinstance(item, Mapping) or int(item.get("schema_id", -1)) != schema_id:
            raise ValueError(f"action normalization lacks schema {name!r}")
        means[schema_id] = np.asarray(item["mean"], dtype=np.float32)
        stds[schema_id] = np.asarray(item["std"], dtype=np.float32)
        if means[schema_id].shape != (ACTION_DIM,) or stds[schema_id].shape != (
            ACTION_DIM,
        ):
            raise ValueError("action normalization vector has the wrong shape")
        if not np.isfinite(means[schema_id]).all() or not np.isfinite(
            stds[schema_id]
        ).all():
            raise ValueError("action normalization contains non-finite values")
        if np.any(stds[schema_id] < 1e-4):
            raise ValueError("action normalization std is below the floor")
    return means, stds


class CanonicalSemanticEncoder(nn.Module):
    """Stage3-style 27-D canonical state sequence to a 96-D semantic state."""

    def __init__(self, semantic_dim: int = SEMANTIC_DIM) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.input_map = nn.Sequential(
            nn.Linear(STATE_DIM, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )
        self.cell = nn.GRUCell(semantic_dim, semantic_dim)

    def forward(
        self, state: torch.Tensor, state_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if state.ndim == 2:
            state = state[:, None]
        if state.ndim != 3 or state.shape[-1] != STATE_DIM:
            raise ValueError(f"canonical state must be [B,T,{STATE_DIM}]")
        if state_mask is None:
            state_mask = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        if state_mask.shape != state.shape[:2]:
            raise ValueError("state mask shape mismatch")
        hidden = state.new_zeros(state.shape[0], self.semantic_dim)
        for step in range(state.shape[1]):
            proposal = self.cell(self.input_map(state[:, step]), hidden)
            hidden = torch.where(state_mask[:, step, None], proposal, hidden)
        return hidden


class PerSchemaActionEncoder(nn.Module):
    """Independent temporal stems prevent cross-robot joint-index aliasing."""

    def __init__(self, schema_count: int, semantic_dim: int = SEMANTIC_DIM) -> None:
        super().__init__()
        self.schema_count = schema_count
        self.register_buffer(
            "action_mean", torch.zeros(schema_count, ACTION_DIM)
        )
        self.register_buffer(
            "action_std", torch.ones(schema_count, ACTION_DIM)
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ACTION_DIM, semantic_dim),
                    nn.GELU(),
                )
                for _ in range(schema_count)
            ]
        )
        self.cells = nn.ModuleList(
            [nn.GRUCell(semantic_dim, semantic_dim) for _ in range(schema_count)]
        )
        self.outputs = nn.ModuleList(
            [nn.LayerNorm(semantic_dim) for _ in range(schema_count)]
        )

    @torch.no_grad()
    def set_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        expected = (self.schema_count, ACTION_DIM)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(f"action normalization must have shape {expected}")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("action normalization contains non-finite values")
        if bool((std < 1e-4).any()):
            raise ValueError("action normalization std is below the floor")
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std))

    def forward(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        action_available: torch.Tensor,
        action_schema_id: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"actions must be [B,H,{ACTION_DIM}]")
        if action_mask.shape != actions.shape[:2]:
            raise ValueError("action mask shape mismatch")
        if action_available.shape != actions.shape[:1]:
            raise ValueError("action availability shape mismatch")
        if action_schema_id.shape != actions.shape[:1]:
            raise ValueError("action schema shape mismatch")
        if bool((action_available & (action_schema_id < 0)).any()):
            raise ValueError("available action lacks a schema id")
        if bool((action_available & (action_schema_id >= self.schema_count)).any()):
            raise ValueError("action schema id is out of range")
        if bool(((~action_available) & (action_schema_id != -1)).any()):
            raise ValueError("missing action must use schema id -1")
        if bool((action_available & ~action_mask.any(dim=1)).any()):
            raise ValueError("available action must contain a nonempty valid prefix")
        effect = actions.new_zeros(actions.shape[0], self.outputs[0].normalized_shape[0])
        for schema in range(self.schema_count):
            selected = action_available & (action_schema_id == schema)
            if not bool(selected.any()):
                continue
            subset = actions[selected]
            subset_mask = action_mask[selected]
            subset = (
                subset - self.action_mean[schema][None, None]
            ) / self.action_std[schema][None, None]
            normalization_clip = getattr(self, "normalization_clip", None)
            if normalization_clip is not None:
                clip = float(normalization_clip)
                if not math.isfinite(clip) or clip <= 0.0:
                    raise ValueError("action normalization clip must be finite/positive")
                subset = subset.clamp(min=-clip, max=clip)
            hidden = subset.new_zeros(subset.shape[0], effect.shape[-1])
            for step in range(subset.shape[1]):
                proposal = self.cells[schema](self.projections[schema](subset[:, step]), hidden)
                hidden = torch.where(subset_mask[:, step, None], proposal, hidden)
            effect[selected] = self.outputs[schema](hidden)
        # Missing-action Piper/UR5 rows remain bit-exact zero and cannot update
        # any action stem through semantic/clock/success losses.
        return effect


class IsolatedClock(nn.Module):
    def __init__(self, semantic_dim: int, clock_dim: int, body_count: int) -> None:
        super().__init__()
        self.body_beta = nn.Embedding(body_count, 1)
        self.base_tau = nn.Linear(semantic_dim, clock_dim)
        self.beta_shape = nn.Linear(semantic_dim, clock_dim)
        self.candidate = nn.Linear(semantic_dim, clock_dim)

    def forward(
        self, semantic: torch.Tensor, dt: torch.Tensor, body_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Stop-gradient preserves the canonical semantic geometry from clock
        # supervision, matching the Stage3 separation contract.
        semantic = semantic.detach()
        beta = self.body_beta(body_id).squeeze(-1)
        base = math.log(10.0) + 1.5 * torch.tanh(self.base_tau(semantic))
        shape = torch.tanh(self.beta_shape(semantic))
        shape = shape - shape.mean(-1, keepdim=True)
        shape = shape / torch.sqrt(shape.square().mean(-1, keepdim=True) + 1e-6)
        log_tau = torch.clamp(base + 0.5 * beta[:, None] * shape, -3.0, 7.0)
        decay = torch.exp(-dt[:, None].clamp_min(0.0) / torch.exp(log_tau))
        hidden = (1.0 - decay) * torch.tanh(self.candidate(semantic))
        return hidden, log_tau


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    body_count: int
    action_schema_count: int = 3
    semantic_dim: int = SEMANTIC_DIM
    clock_dim: int = 64
    object_delta_dim: int = OBJECT_DELTA_DIM
    dropout: float = 0.1


class MultibodyCanonicalEventWorldModel(nn.Module):
    """Plug-in action-effect model shared across embodiment/policy identities."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.semantic_dim != SEMANTIC_DIM:
            raise ValueError("canonical semantic contract is fixed at 96 dimensions")
        if config.object_delta_dim != OBJECT_DELTA_DIM:
            raise ValueError("canonical object/relative-goal delta is fixed at 6D")
        self.config = config
        self.semantic = CanonicalSemanticEncoder(config.semantic_dim)
        self.action = PerSchemaActionEncoder(
            config.action_schema_count, config.semantic_dim
        )
        self.transition = nn.Sequential(
            nn.Linear(2 * config.semantic_dim, config.semantic_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.semantic_dim, config.semantic_dim),
            nn.LayerNorm(config.semantic_dim),
        )
        self.post_event = nn.Linear(config.semantic_dim, len(CANONICAL_EVENTS))
        self.next_event = nn.Linear(config.semantic_dim, len(CANONICAL_EVENTS))
        self.success = nn.Linear(config.semantic_dim, 1)
        self.recovery = nn.Linear(config.semantic_dim, 1)
        self.object_mean = nn.Linear(config.semantic_dim, config.object_delta_dim)
        self.object_scale = nn.Linear(config.semantic_dim, config.object_delta_dim)
        self.clock = IsolatedClock(
            config.semantic_dim, config.clock_dim, config.body_count
        )
        self.duration_mean = nn.Linear(config.clock_dim, len(CANONICAL_EVENTS))
        self.duration_scale = nn.Linear(config.clock_dim, len(CANONICAL_EVENTS))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        semantic = self.semantic(batch["state"], batch.get("state_mask"))
        action_effect = self.action(
            batch["actions"],
            batch["action_mask"].bool(),
            batch["action_available"].bool(),
            batch["action_schema_id"].long(),
        )
        residual = self.transition(torch.cat([semantic, action_effect], dim=-1))
        transitioned = semantic + residual
        clock_hidden, log_tau = self.clock(
            transitioned, batch["dt"], batch["body_id"].long()
        )
        duration_log_mean = self.duration_mean(clock_hidden)
        duration_log_scale = torch.clamp(self.duration_scale(clock_hidden), -5.0, 2.0)
        current = batch["current_event_id"].long()[:, None]
        return {
            "semantic": semantic,
            "action_effect": action_effect,
            "transitioned": transitioned,
            # Expose the isolated physical-time representation so downstream
            # decision heads can consume time without reimplementing the
            # clock.  The core proper duration heads continue to own the
            # gradients into this representation.
            "clock_hidden": clock_hidden,
            "post_event_logits": self.post_event(transitioned),
            "next_event_logits": self.next_event(transitioned),
            "success_logit": self.success(transitioned).squeeze(-1),
            "recovery_logit": self.recovery(transitioned).squeeze(-1),
            "object_delta_mean": self.object_mean(transitioned),
            "object_delta_log_scale": torch.clamp(
                self.object_scale(transitioned), -5.0, 2.0
            ),
            "duration_log_mean": duration_log_mean,
            "duration_log_scale": duration_log_scale,
            "duration_selected_log_mean": duration_log_mean.gather(1, current).squeeze(1),
            "duration_selected_log_scale": duration_log_scale.gather(1, current).squeeze(1),
            "clock_log_tau": log_tau,
        }


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=loss.device, dtype=loss.dtype)
    valid = weight > 0
    if not bool(valid.any()):
        return loss.reshape(-1)[0] * 0.0
    return (loss[valid] * weight[valid]).sum() / weight[valid].sum()


def censored_lognormal_loss(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    duration: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    target = torch.log1p(duration.clamp_min(0.0))
    scale = torch.exp(log_scale).clamp_min(1e-4)
    z = (target - mean) / scale
    observed_nll = 0.5 * z.square() + log_scale + 0.5 * math.log(2.0 * math.pi)
    # Stable Gaussian log-survival.  A fixed probability clamp makes the
    # censored objective exactly flat for moderately large z and discards the
    # gradient from long right-censored event durations.
    censored_nll = -torch.special.log_ndtr(-z)
    return torch.where(observed.bool(), observed_nll, censored_nll)


DEFAULT_LOSS_WEIGHTS = {
    "post_event": 1.0,
    "next_event": 0.5,
    "duration": 0.5,
    "success": 1.0,
    "recovery": 0.5,
    "object": 0.5,
}

VALIDATION_SELECTION_RULE = {
    "format": "etsf_multibody_validation_selection_v1",
    "split": "validation_only",
    "direction": "minimize",
    "primary": "composite_relative_to_train_only_baselines",
    "components": [
        "post_event_macro_error_ratio",
        "next_event_macro_error_ratio",
        "observed_duration_mae_ratio",
        "success_brier_ratio",
        "object_rmse_ratio",
    ],
    "aggregation": "arithmetic_mean_of_available_finite_components",
    "tie_breakers": [
        "higher_next_event_macro_f1",
        "lower_success_brier",
        "lower_observed_duration_nll",
        "lower_object_nll",
        "earlier_step",
    ],
    "test_metrics_used": False,
}


def compute_multitask_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    sample_weight: torch.Tensor | None = None,
    loss_weights: Mapping[str, float] = DEFAULT_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = output["success_logit"].shape[0]
    if sample_weight is None:
        sample_weight = output["success_logit"].new_ones(batch_size)
    if sample_weight.shape != (batch_size,):
        raise ValueError("sample weight must be [B]")
    action_available = batch["action_available"].to(sample_weight)
    post = _weighted_mean(
        F.cross_entropy(
            output["post_event_logits"], batch["post_event_id"].long(), reduction="none"
        ),
        sample_weight * batch["post_event_mask"].to(sample_weight),
    )
    next_event = _weighted_mean(
        F.cross_entropy(
            output["next_event_logits"], batch["next_event_id"].long(), reduction="none"
        ),
        sample_weight * batch["next_event_mask"].to(sample_weight),
    )
    duration_rows = censored_lognormal_loss(
        output["duration_selected_log_mean"],
        output["duration_selected_log_scale"],
        batch["duration"],
        batch["duration_observed"],
    )
    duration = _weighted_mean(
        duration_rows, sample_weight * batch["duration_mask"].to(sample_weight)
    )
    success = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output["success_logit"], batch["success"].to(output["success_logit"]), reduction="none"
        ),
        sample_weight * batch["success_mask"].to(sample_weight),
    )
    # Recovery is defined only after a predicate/event regression and only on
    # rows with an observed action.  Missing-action target bodies never provide
    # a gradient to this action-effect head.
    recovery_weight = (
        sample_weight * action_available * batch["recovery_mask"].to(sample_weight)
    )
    recovery = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output["recovery_logit"],
            batch["recovery"].to(output["recovery_logit"]),
            reduction="none",
        ),
        recovery_weight,
    )
    scale = torch.exp(output["object_delta_log_scale"]).clamp_min(1e-4)
    object_nll = (
        0.5
        * ((batch["object_delta"] - output["object_delta_mean"]) / scale).square()
        + output["object_delta_log_scale"]
        + 0.5 * math.log(2.0 * math.pi)
    ).mean(-1)
    object_weight = (
        sample_weight * action_available * batch["object_delta_mask"].to(sample_weight)
    )
    object_loss = _weighted_mean(object_nll, object_weight)
    pieces = {
        "post_event": post,
        "next_event": next_event,
        "duration": duration,
        "success": success,
        "recovery": recovery,
        "object": object_loss,
    }
    total = sum(float(loss_weights[name]) * value for name, value in pieces.items())
    pieces["total"] = total
    pieces["recovery_supervised_rows"] = (recovery_weight > 0).sum().to(total)
    pieces["object_supervised_rows"] = (object_weight > 0).sum().to(total)
    return total, pieces


def _macro_f1(
    labels: np.ndarray, predictions: np.ndarray, classes: int
) -> float | None:
    if len(labels) == 0:
        return None
    values = []
    for class_id in range(classes):
        true_positive = int(((labels == class_id) & (predictions == class_id)).sum())
        false_positive = int(((labels != class_id) & (predictions == class_id)).sum())
        false_negative = int(((labels == class_id) & (predictions != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            values.append(2.0 * true_positive / denominator)
    return float(np.mean(values)) if values else None


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, str]:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if not len(positive) or not len(negative):
        return None, "unavailable_single_class"
    comparison = positive[:, None] - negative[None, :]
    auc = ((comparison > 0).sum() + 0.5 * (comparison == 0).sum()) / comparison.size
    return float(auc), "available"


def _event_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    if len(labels) == 0:
        return {"accuracy": None, "macro_f1": None, "support": 0, "class_counts": [0] * 5}
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": _macro_f1(labels, predictions, len(CANONICAL_EVENTS)),
        "support": int(len(labels)),
        "class_counts": np.bincount(labels, minlength=len(CANONICAL_EVENTS)).tolist(),
    }


def fit_train_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit every baseline from train rows only and freeze a signed receipt."""

    post_labels = np.asarray(
        [int(row["post_event_id"]) for row in rows if bool(row["post_event_mask"])],
        dtype=np.int64,
    )
    next_labels = np.asarray(
        [int(row["next_event_id"]) for row in rows if bool(row["next_event_mask"])],
        dtype=np.int64,
    )
    if not len(post_labels) or not len(next_labels):
        raise ValueError("train split cannot fit majority event baselines")
    duration_by_body_event: dict[str, list[float]] = defaultdict(list)
    duration_by_event: dict[str, list[float]] = defaultdict(list)
    all_duration = []
    successes = []
    object_rows = []
    for row in rows:
        if bool(row["duration_mask"]) and bool(row["duration_observed"]):
            duration = float(row["duration"])
            event = int(row["current_event_id"])
            duration_by_body_event[f"{row['body']}|{event}"].append(duration)
            duration_by_event[str(event)].append(duration)
            all_duration.append(duration)
        if bool(row["success_mask"]):
            successes.append(float(row["success"]))
        if bool(row["action_available"]) and bool(row["object_delta_mask"]):
            object_rows.append(np.asarray(row["object_delta"], dtype=np.float64))
    if not all_duration or not successes or not object_rows:
        raise ValueError("train split lacks duration/success/object baseline support")
    object_values = np.stack(object_rows)
    object_scale = np.maximum(np.sqrt(np.mean(np.square(object_values), axis=0)), 1e-4)
    payload = {
        "format": "etsf_train_only_validation_baselines_v1",
        "source_split": "train_only",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "majority_post_event": int(np.bincount(post_labels, minlength=5).argmax()),
        "majority_next_event": int(np.bincount(next_labels, minlength=5).argmax()),
        "duration_median_by_body_event": {
            key: float(np.median(value)) for key, value in sorted(duration_by_body_event.items())
        },
        "duration_median_by_event": {
            key: float(np.median(value)) for key, value in sorted(duration_by_event.items())
        },
        "duration_global_median": float(np.median(all_duration)),
        "empirical_success": float(np.mean(successes)),
        "zero_object_delta": [0.0] * OBJECT_DELTA_DIM,
        "zero_object_scale": object_scale.astype(np.float32).tolist(),
        "support": {
            "post_event_rows": int(len(post_labels)),
            "next_event_rows": int(len(next_labels)),
            "observed_duration_rows": len(all_duration),
            "success_rows": len(successes),
            "object_rows": len(object_rows),
        },
    }
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def evaluate_train_only_baselines(
    baseline: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    post_labels = np.asarray(
        [int(row["post_event_id"]) for row in rows if bool(row["post_event_mask"])],
        dtype=np.int64,
    )
    next_labels = np.asarray(
        [int(row["next_event_id"]) for row in rows if bool(row["next_event_mask"])],
        dtype=np.int64,
    )
    post_prediction = np.full_like(post_labels, int(baseline["majority_post_event"]))
    next_prediction = np.full_like(next_labels, int(baseline["majority_next_event"]))
    duration_labels = []
    duration_predictions = []
    success_labels = []
    object_labels = []
    for row in rows:
        if bool(row["duration_mask"]) and bool(row["duration_observed"]):
            event = int(row["current_event_id"])
            prediction = baseline["duration_median_by_body_event"].get(
                f"{row['body']}|{event}",
                baseline["duration_median_by_event"].get(
                    str(event), baseline["duration_global_median"]
                ),
            )
            duration_labels.append(float(row["duration"]))
            duration_predictions.append(float(prediction))
        if bool(row["success_mask"]):
            success_labels.append(float(row["success"]))
        if bool(row["action_available"]) and bool(row["object_delta_mask"]):
            object_labels.append(np.asarray(row["object_delta"], dtype=np.float64))
    duration_labels_array = np.asarray(duration_labels)
    duration_predictions_array = np.asarray(duration_predictions)
    success_array = np.asarray(success_labels)
    success_scores = np.full_like(success_array, float(baseline["empirical_success"]))
    success_auc, success_auc_status = _binary_auc(success_array, success_scores)
    objects = np.stack(object_labels)
    object_scale = np.asarray(baseline["zero_object_scale"], dtype=np.float64)
    object_nll = (
        0.5 * np.square(objects / object_scale[None])
        + np.log(object_scale[None])
        + 0.5 * math.log(2.0 * math.pi)
    )
    return {
        "source": "train_only_baselines_evaluated_on_validation",
        "post_event": _event_metrics(post_labels, post_prediction),
        "next_event": _event_metrics(next_labels, next_prediction),
        "observed_duration_mae": float(
            np.mean(np.abs(duration_predictions_array - duration_labels_array))
        ),
        "observed_duration_support": int(len(duration_labels_array)),
        "success_brier": float(np.mean(np.square(success_scores - success_array))),
        "success_auroc": success_auc,
        "success_auroc_status": success_auc_status,
        "success_support": {
            "rows": int(len(success_array)),
            "positive": int((success_array > 0.5).sum()),
            "negative": int((success_array <= 0.5).sum()),
        },
        "object_rmse": float(np.sqrt(np.mean(np.square(objects)))),
        "object_nll": float(np.mean(object_nll)),
        "object_support": int(len(objects)),
    }


@torch.no_grad()
def evaluate_validation_model(
    model: MultibodyCanonicalEventWorldModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for raw in loader:
        batch = _move_batch(raw, device)
        output = model(batch)
        for key, tensor in {
            "post_label": batch["post_event_id"],
            "post_mask": batch["post_event_mask"],
            "post_prediction": output["post_event_logits"].argmax(-1),
            "next_label": batch["next_event_id"],
            "next_mask": batch["next_event_mask"],
            "next_prediction": output["next_event_logits"].argmax(-1),
            "duration": batch["duration"],
            "duration_observed": batch["duration_observed"],
            "duration_mask": batch["duration_mask"],
            "duration_mean": output["duration_selected_log_mean"],
            "duration_log_scale": output["duration_selected_log_scale"],
            "success": batch["success"],
            "success_mask": batch["success_mask"],
            "success_probability": torch.sigmoid(output["success_logit"]),
            "recovery": batch["recovery"],
            "recovery_mask": batch["recovery_mask"] * batch["action_available"],
            "object": batch["object_delta"],
            "object_mask": batch["object_delta_mask"] * batch["action_available"],
            "object_mean": output["object_delta_mean"],
            "object_log_scale": output["object_delta_log_scale"],
        }.items():
            collected[key].append(tensor.detach().cpu().numpy())
    values = {key: np.concatenate(parts) for key, parts in collected.items()}
    post_mask = values["post_mask"] > 0.5
    next_mask = values["next_mask"] > 0.5
    duration_observed = (values["duration_observed"] > 0.5) & (
        values["duration_mask"] > 0.5
    )
    duration_prediction = np.expm1(values["duration_mean"]).clip(min=0.0)
    duration_nll = censored_lognormal_loss(
        torch.from_numpy(values["duration_mean"]),
        torch.from_numpy(values["duration_log_scale"]),
        torch.from_numpy(values["duration"]),
        torch.ones(len(values["duration"])),
    ).numpy()
    success_mask = values["success_mask"] > 0.5
    success_labels = values["success"][success_mask]
    success_scores = values["success_probability"][success_mask]
    success_auc, success_auc_status = _binary_auc(success_labels, success_scores)
    object_mask = values["object_mask"] > 0.5
    object_error = values["object"][object_mask] - values["object_mean"][object_mask]
    object_scale = np.exp(values["object_log_scale"][object_mask]).clip(min=1e-4)
    object_nll = (
        0.5 * np.square(object_error / object_scale)
        + np.log(object_scale)
        + 0.5 * math.log(2.0 * math.pi)
    )
    recovery_mask = values["recovery_mask"] > 0.5
    return {
        "split": "validation_only",
        "post_event": _event_metrics(
            values["post_label"][post_mask].astype(np.int64),
            values["post_prediction"][post_mask].astype(np.int64),
        ),
        "next_event": _event_metrics(
            values["next_label"][next_mask].astype(np.int64),
            values["next_prediction"][next_mask].astype(np.int64),
        ),
        "observed_duration_mae": (
            float(
                np.mean(
                    np.abs(
                        duration_prediction[duration_observed]
                        - values["duration"][duration_observed]
                    )
                )
            )
            if duration_observed.any()
            else None
        ),
        "observed_duration_nll": (
            float(np.mean(duration_nll[duration_observed]))
            if duration_observed.any()
            else None
        ),
        "duration_support": {
            "observed": int(duration_observed.sum()),
            "censored": int(
                ((values["duration_mask"] > 0.5) & ~duration_observed).sum()
            ),
        },
        "success_brier": (
            float(np.mean(np.square(success_scores - success_labels)))
            if len(success_labels)
            else None
        ),
        "success_auroc": success_auc,
        "success_auroc_status": success_auc_status,
        "success_support": {
            "rows": int(len(success_labels)),
            "positive": int((success_labels > 0.5).sum()),
            "negative": int((success_labels <= 0.5).sum()),
        },
        "object_rmse": (
            float(np.sqrt(np.mean(np.square(object_error))))
            if len(object_error)
            else None
        ),
        "object_nll": float(np.mean(object_nll)) if len(object_nll) else None,
        "object_support": int(object_mask.sum()),
        "recovery_support": {
            "rows": int(recovery_mask.sum()),
            "positive": int((values["recovery"][recovery_mask] > 0.5).sum()),
            "negative": int((values["recovery"][recovery_mask] <= 0.5).sum()),
        },
    }


def validation_selection_score(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[float, dict[str, float]]:
    def ratio(value: Any, reference: Any) -> float | None:
        if value is None or reference is None:
            return None
        value_float = float(value)
        reference_float = float(reference)
        if not math.isfinite(value_float) or not math.isfinite(reference_float):
            return None
        return value_float / max(reference_float, 1e-6)

    candidates = {
        "post_event_macro_error_ratio": ratio(
            None
            if metrics["post_event"]["macro_f1"] is None
            else 1.0 - float(metrics["post_event"]["macro_f1"]),
            None
            if baseline["post_event"]["macro_f1"] is None
            else 1.0 - float(baseline["post_event"]["macro_f1"]),
        ),
        "next_event_macro_error_ratio": ratio(
            None
            if metrics["next_event"]["macro_f1"] is None
            else 1.0 - float(metrics["next_event"]["macro_f1"]),
            None
            if baseline["next_event"]["macro_f1"] is None
            else 1.0 - float(baseline["next_event"]["macro_f1"]),
        ),
        "observed_duration_mae_ratio": ratio(
            metrics["observed_duration_mae"], baseline["observed_duration_mae"]
        ),
        "success_brier_ratio": ratio(
            metrics["success_brier"], baseline["success_brier"]
        ),
        "object_rmse_ratio": ratio(metrics["object_rmse"], baseline["object_rmse"]),
    }
    components = {
        name: float(value)
        for name, value in candidates.items()
        if value is not None and math.isfinite(float(value))
    }
    if not components:
        raise RuntimeError("validation selection has no finite component")
    return float(np.mean(list(components.values()))), components


def validation_selection_key(
    metrics: Mapping[str, Any], score: float, step: int
) -> tuple[float, float, float, float, float, int]:
    """Materialize the preregistered primary and tie-breaker ordering."""

    next_f1 = metrics["next_event"]["macro_f1"]
    return (
        float(score),
        -float(next_f1) if next_f1 is not None else math.inf,
        float(metrics["success_brier"])
        if metrics["success_brier"] is not None
        else math.inf,
        float(metrics["observed_duration_nll"])
        if metrics["observed_duration_nll"] is not None
        else math.inf,
        float(metrics["object_nll"])
        if metrics["object_nll"] is not None
        else math.inf,
        int(step),
    )


@torch.no_grad()
def ensemble_predict(
    models: Sequence[MultibodyCanonicalEventWorldModel],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if len(models) != 5:
        raise ValueError("epistemic prediction requires exactly five members")
    outputs = [model(batch) for model in models]
    event_prob = torch.stack(
        [torch.softmax(item["post_event_logits"], -1) for item in outputs]
    )
    success_prob = torch.stack([torch.sigmoid(item["success_logit"]) for item in outputs])
    object_mean = torch.stack([item["object_delta_mean"] for item in outputs])
    duration_mean = torch.stack(
        [torch.expm1(item["duration_selected_log_mean"]).clamp_min(0.0) for item in outputs]
    )
    components = torch.stack(
        [
            event_prob.var(0, correction=0).mean(-1),
            success_prob.var(0, correction=0),
            object_mean.var(0, correction=0).mean(-1),
            duration_mean.var(0, correction=0),
        ],
        dim=-1,
    )
    return {
        "post_event_probability": event_prob.mean(0),
        "success_probability": success_prob.mean(0),
        "object_delta_mean": object_mean.mean(0),
        "duration_mean": duration_mean.mean(0),
        "epistemic_components": components,
        "epistemic_uncertainty": components.mean(-1),
    }


def _goal_vector(
    poses: np.ndarray,
    names: Sequence[str],
    step: int,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    moving_index = list(names).index(str(calibration["moving"]))
    moving = poses[step, moving_index, :3]
    anchor_name = str(calibration.get("anchor", ""))
    if anchor_name:
        anchor_index = list(names).index(anchor_name)
        goal = poses[step, anchor_index, :3] + np.asarray(
            calibration.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32
        )
    else:
        centers = np.asarray(calibration["centers"], dtype=np.float32)
        goal = centers[np.linalg.norm(centers - moving[None], axis=1).argmin()]
    return moving.astype(np.float32), (goal - moving).astype(np.float32)


def derive_predicates_and_events(
    poses: np.ndarray,
    names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    moving_name = str(calibration["moving"])
    if moving_name not in names:
        raise ValueError(f"moving object {moving_name!r} absent")
    moving_index = list(names).index(moving_name)
    position = poses[:, moving_index, :3]
    displacement = np.linalg.norm(position - position[0], axis=1)
    lifted = position[:, 2] >= position[0, 2] + float(calibration["delta_z"])
    near = np.asarray(
        [
            np.linalg.norm(_goal_vector(poses, names, step, calibration)[1])
            <= float(calibration["tau_d"])
            for step in range(len(poses))
        ],
        dtype=bool,
    )
    motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    instant_stationary = near & (motion <= float(calibration["tau_motion"]))
    width = int(calibration["stationary_steps"])
    stationary = np.zeros(len(poses), dtype=bool)
    for step in range(width - 1, len(poses)):
        stationary[step] = bool(instant_stationary[step - width + 1 : step + 1].all())
    succeeded = np.zeros(len(poses), dtype=bool)
    if success:
        succeeded[-1] = True
    moved = displacement >= float(calibration["delta_move"])
    predicates = np.stack([moved, lifted, near, stationary, succeeded], axis=-1)
    events = np.full(len(poses), EVENT_TO_ID["e0"], dtype=np.int64)
    events[moved | lifted] = EVENT_TO_ID["e12"]
    events[near] = EVENT_TO_ID["e3"]
    events[stationary] = EVENT_TO_ID["e4"]
    events[succeeded] = EVENT_TO_ID["eK"]
    return predicates.astype(np.float32), events


def canonical_state_vector(
    poses: np.ndarray,
    names: Sequence[str],
    step: int,
    task: str,
    calibration: Mapping[str, Any],
    predicates: np.ndarray,
    event_id: int,
) -> np.ndarray:
    moving_name = str(calibration["moving"])
    moving_index = list(names).index(moving_name)
    moving, relative_goal = _goal_vector(poses, names, step, calibration)
    initial = poses[0, moving_index, :3]
    displacement = moving - initial
    norm = float(np.linalg.norm(displacement))
    direction = displacement / max(norm, 1e-6)
    quaternion = poses[step, moving_index, 3:7].astype(np.float32)
    progress = np.asarray(
        [min(norm / max(float(calibration["delta_move"]), 1e-6), 4.0)],
        dtype=np.float32,
    )
    task_one_hot = np.zeros(len(TASKS), dtype=np.float32)
    task_one_hot[TASK_TO_ID[task]] = 1.0
    event_one_hot = np.zeros(len(CANONICAL_EVENTS), dtype=np.float32)
    event_one_hot[int(event_id)] = 1.0
    # Geometry14 + task6 + event5 + reversible predicates2 = 27.
    state = np.concatenate(
        [
            displacement.astype(np.float32),
            relative_goal,
            quaternion,
            progress,
            direction.astype(np.float32),
            task_one_hot,
            event_one_hot,
            predicates[step, [1, 2]].astype(np.float32),
        ]
    )
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise RuntimeError("canonical state construction violated the 27-D contract")
    return state


def _next_event_target(events: np.ndarray, query: int) -> tuple[int, float, bool]:
    current = int(events[query])
    future = np.flatnonzero(events[query + 1 :] != current)
    if len(future):
        boundary = query + 1 + int(future[0])
        return int(events[boundary]), float(boundary - query), True
    return current, float(max(len(events) - 1 - query, 0)), False


def _transition_row(
    *,
    descriptor: GroupDescriptor,
    poses: np.ndarray,
    names: Sequence[str],
    success: bool,
    calibration: Mapping[str, Any],
    predicates: np.ndarray,
    events: np.ndarray,
    query: int,
    end: int,
    actions: np.ndarray | None,
    action_mask: np.ndarray | None,
    action_schema: int,
) -> dict[str, Any]:
    current = int(events[query])
    next_event, duration, observed = _next_event_target(events, query)
    moving_start, relative_start = _goal_vector(poses, names, query, calibration)
    moving_end, relative_end = _goal_vector(poses, names, end, calibration)
    padded_actions = np.zeros((25, ACTION_DIM), dtype=np.float32)
    padded_mask = np.zeros(25, dtype=bool)
    available = actions is not None
    if available:
        assert action_mask is not None
        length = min(len(actions), 25)
        padded_actions[:length] = actions[:length]
        padded_mask[:length] = action_mask[:length]
    regressed = int(events[end]) < current
    recovered = bool(regressed and np.any(events[end + 1 :] >= current))
    return {
        "state": canonical_state_vector(
            poses, names, query, descriptor.task, calibration, predicates, current
        ),
        "actions": padded_actions,
        "action_mask": padded_mask,
        "action_available": np.float32(available),
        "action_schema_id": np.int64(action_schema if available else -1),
        "current_event_id": np.int64(current),
        "post_event_id": np.int64(events[end]),
        "post_event_mask": np.float32(1.0),
        "next_event_id": np.int64(next_event),
        "next_event_mask": np.float32(observed),
        "duration": np.float32(duration),
        "duration_observed": np.float32(observed),
        "duration_mask": np.float32(1.0),
        "success": np.float32(success),
        "success_mask": np.float32(1.0),
        "recovery": np.float32(recovered),
        "recovery_mask": np.float32(regressed),
        "object_delta": np.concatenate(
            [moving_end - moving_start, relative_end - relative_start]
        ).astype(np.float32),
        "object_delta_mask": np.float32(available),
        "dt": np.float32(max(end - query, 1)),
        "logical_group": descriptor.logical_group,
        "body": descriptor.body,
        "policy": descriptor.policy,
        "task": descriptor.task,
    }


def load_stage1_rows(
    descriptor: GroupDescriptor, calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if descriptor.source == "stage1_source":
        with np.load(descriptor.path, allow_pickle=False) as payload:
            poses = payload["poses"].astype(np.float32)
            names = [_decode(value) for value in payload["object_names"]]
            success = bool(payload["success"])
        assert descriptor.auxiliary_path is not None
        with h5py.File(descriptor.auxiliary_path, "r") as handle:
            full_actions = handle["joint_action/vector"][:].astype(np.float32)
        if full_actions.shape[1] != ACTION_DIM:
            raise ValueError("Stage1 source action is not 14-D")
        action_schema = (
            ACTION_SCHEMAS["aloha"]
            if descriptor.body == "aloha-agilex"
            else ACTION_SCHEMAS["arx"]
        )
    elif descriptor.source == "stage1_target":
        with h5py.File(descriptor.path, "r") as handle:
            poses = handle["object_poses"][:].astype(np.float32)
            names = [_decode(value) for value in handle["object_names"][:]]
            success = bool(handle.attrs["success"])
        full_actions = None
        action_schema = -1
    else:
        raise ValueError("descriptor is not Stage1")
    predicates, events = derive_predicates_and_events(
        poses, names, success, calibration
    )
    rows = []
    for query in range(0, max(len(poses) - 1, 1), 25):
        end = min(query + 25, len(poses) - 1)
        if end <= query:
            continue
        if full_actions is None:
            actions = None
            mask = None
        else:
            # Pose ``end`` is the result after exactly ``end-query`` actions;
            # never include the final unpaired action in a partial chunk.
            actions = full_actions[query : min(end, len(full_actions))]
            mask = np.ones(len(actions), dtype=bool)
        rows.append(
            _transition_row(
                descriptor=descriptor,
                poses=poses,
                names=names,
                success=success,
                calibration=calibration,
                predicates=predicates,
                events=events,
                query=query,
                end=end,
                actions=actions,
                action_mask=mask,
                action_schema=action_schema,
            )
        )
    return rows


def load_schema5_rows(
    descriptor: GroupDescriptor, calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(descriptor.path, "r") as handle:
        names = [_decode(value) for value in handle["object_names"][:]]
        success_values = handle["success"][:].astype(bool)
        branches = handle["branches"]
        if len(branches) != len(success_values):
            raise ValueError("schema5 branch/success count mismatch")
        for candidate, branch_name in enumerate(sorted(branches)):
            branch = branches[branch_name]
            poses = branch["object_poses"][:].astype(np.float32)
            success = bool(success_values[candidate])
            predicates, events = derive_predicates_and_events(
                poses, names, success, calibration
            )
            query_steps = branch["query_steps"][:].astype(np.int64)
            post_steps = branch["query_post_steps"][:].astype(np.int64)
            actions = branch["query_actions"][:].astype(np.float32)
            masks = branch["query_action_mask"][:].astype(bool)
            if not (
                len(query_steps) == len(post_steps) == len(actions) == len(masks)
            ):
                raise ValueError("schema5 continuation arrays are misaligned")
            for index, query in enumerate(query_steps):
                end = int(post_steps[index])
                if not 0 <= int(query) < end < len(poses):
                    raise ValueError("schema5 query/post step is out of range")
                rows.append(
                    _transition_row(
                        descriptor=descriptor,
                        poses=poses,
                        names=names,
                        success=success,
                        calibration=calibration,
                        predicates=predicates,
                        events=events,
                        query=int(query),
                        end=end,
                        actions=actions[index],
                        action_mask=masks[index],
                        action_schema=ACTION_SCHEMAS["openvla"],
                    )
                )
    return rows


def load_rows(
    descriptors: Sequence[GroupDescriptor], event_spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Open only explicitly selected train/validation descriptors."""

    rows: list[dict[str, Any]] = []
    calibration = event_spec["calibration"]
    for descriptor in descriptors:
        if descriptor.source == "openvla_schema5":
            rows.extend(load_schema5_rows(descriptor, calibration[descriptor.task]))
        else:
            rows.extend(load_stage1_rows(descriptor, calibration[descriptor.task]))
    if not rows:
        raise ValueError("selected split produced no transitions")
    return rows


class TransitionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        body_to_id: Mapping[str, int],
    ) -> None:
        self.rows = list(rows)
        self.body_to_id = dict(body_to_id)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        result = {
            key: torch.as_tensor(value)
            for key, value in row.items()
            if key not in {"logical_group", "body", "policy", "task"}
        }
        result["body_id"] = torch.tensor(self.body_to_id[str(row["body"])])
        result["logical_group"] = str(row["logical_group"])
        return result


def collate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [key for key in rows[0] if key != "logical_group"]
    result: dict[str, Any] = {
        key: torch.stack([row[key] for row in rows]) for key in keys
    }
    result["logical_group"] = [str(row["logical_group"]) for row in rows]
    return result


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def synthetic_batch(batch_size: int = 10) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260828)
    available = torch.tensor(
        [True, True, True, False, False] * ((batch_size + 4) // 5)
    )[:batch_size]
    schemas = torch.tensor([0, 1, 2, -1, -1] * ((batch_size + 4) // 5))[:batch_size]
    action_mask = torch.arange(25)[None] < torch.randint(
        5, 26, (batch_size, 1), generator=generator
    )
    action_mask &= available[:, None]
    current = torch.randint(0, 4, (batch_size,), generator=generator)
    return {
        "state": torch.randn(batch_size, STATE_DIM, generator=generator),
        "actions": torch.randn(batch_size, 25, ACTION_DIM, generator=generator),
        "action_mask": action_mask,
        "action_available": available.float(),
        "action_schema_id": schemas,
        "body_id": torch.arange(batch_size) % 4,
        "current_event_id": current,
        "post_event_id": torch.randint(0, 5, (batch_size,), generator=generator),
        "post_event_mask": torch.ones(batch_size),
        "next_event_id": torch.randint(0, 5, (batch_size,), generator=generator),
        "next_event_mask": torch.randint(0, 2, (batch_size,), generator=generator).float(),
        "duration": torch.randint(1, 100, (batch_size,), generator=generator).float(),
        "duration_observed": torch.randint(0, 2, (batch_size,), generator=generator).float(),
        "duration_mask": torch.ones(batch_size),
        "success": torch.randint(0, 2, (batch_size,), generator=generator).float(),
        "success_mask": torch.ones(batch_size),
        "recovery": torch.randint(0, 2, (batch_size,), generator=generator).float(),
        "recovery_mask": available.float(),
        "object_delta": torch.randn(batch_size, OBJECT_DELTA_DIM, generator=generator),
        "object_delta_mask": available.float(),
        "dt": torch.full((batch_size,), 25.0),
    }


def run_synthetic_smoke() -> dict[str, Any]:
    torch.manual_seed(20260828)
    batch = synthetic_batch()
    models = [
        MultibodyCanonicalEventWorldModel(ModelConfig(body_count=4, dropout=0.0))
        for _ in range(5)
    ]
    losses = []
    for model in models:
        output = model(batch)
        loss, pieces = compute_multitask_loss(output, batch)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic loss is non-finite")
        losses.append(float(loss.detach()))
        if int(pieces["object_supervised_rows"].item()) != 6:
            raise RuntimeError("missing-action object mask failed")
    prediction = ensemble_predict([model.eval() for model in models], batch)
    if prediction["epistemic_uncertainty"].shape != (10,):
        raise RuntimeError("ensemble epistemic shape mismatch")
    return {
        "status": "synthetic_smoke_passed",
        "members": 5,
        "losses": losses,
        "state_dim": STATE_DIM,
        "semantic_dim": SEMANTIC_DIM,
        "events": list(CANONICAL_EVENTS),
        "missing_action_effect_is_zero": bool(
            torch.equal(
                models[0](batch)["action_effect"][~batch["action_available"].bool()],
                torch.zeros(4, SEMANTIC_DIM),
            )
        ),
    }


def _binding_from_args(args: argparse.Namespace) -> InputBinding:
    return InputBinding(
        stage1_root=args.stage1_root,
        stage1_source_manifest=args.stage1_source_manifest,
        stage1_source_manifest_sha256=args.stage1_source_manifest_sha256,
        stage1_target_manifest=args.stage1_target_manifest,
        stage1_target_manifest_sha256=args.stage1_target_manifest_sha256,
        event_spec=args.event_spec,
        event_spec_sha256=args.event_spec_sha256,
        openvla_schema5_manifest=args.openvla_schema5_manifest,
        openvla_schema5_manifest_sha256=args.openvla_schema5_manifest_sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train", "synthetic-smoke"), required=True)
    parser.add_argument("--stage1-root", type=Path)
    parser.add_argument("--stage1-source-manifest", type=Path)
    parser.add_argument("--stage1-source-manifest-sha256")
    parser.add_argument("--stage1-target-manifest", type=Path)
    parser.add_argument("--stage1-target-manifest-sha256")
    parser.add_argument("--event-spec", type=Path)
    parser.add_argument("--event-spec-sha256")
    parser.add_argument("--openvla-schema5-manifest", type=Path)
    parser.add_argument("--openvla-schema5-manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--ensemble-seeds", nargs=5, type=int, default=[20260828, 20260829, 20260830, 20260831, 20260832])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _require_binding_args(args: argparse.Namespace) -> None:
    names = (
        "stage1_root",
        "stage1_source_manifest",
        "stage1_source_manifest_sha256",
        "stage1_target_manifest",
        "stage1_target_manifest_sha256",
        "event_spec",
        "event_spec_sha256",
        "openvla_schema5_manifest",
        "openvla_schema5_manifest_sha256",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"mode {args.mode} requires binding arguments: {missing}")


def run_preflight(args: argparse.Namespace) -> tuple[InputBinding, dict[str, Any], dict[str, list[GroupDescriptor]]]:
    _require_binding_args(args)
    binding = _binding_from_args(args)
    audit = verify_input_bindings(binding)
    descriptors = scan_stage1_groups(binding) + scan_schema5_groups(binding)
    splits = strict_group_split(descriptors, split_seed=args.split_seed)
    audit.update(split_receipt(splits))
    audit["body_alias"] = body_alias_receipt(descriptors)
    audit["total_groups"] = len(descriptors)
    return binding, audit, splits


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output is None:
        raise ValueError("train mode requires --output")
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("steps and eval-every must be positive")
    output = reject_forbidden_path(args.output, "output")
    if output.exists():
        raise FileExistsError("training output must be a new immutable path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    binding, audit, splits = run_preflight(args)
    event_spec = json.loads(binding.event_spec.read_text(encoding="utf-8"))
    # Crucial protocol boundary: test descriptors are never passed to a loader.
    train_rows = load_rows(splits["train"], event_spec)
    validation_rows = load_rows(splits["validation"], event_spec)
    action_normalization = fit_train_action_normalization(train_rows)
    action_mean, action_std = action_normalization_arrays(action_normalization)
    train_baselines = fit_train_baselines(train_rows)
    validation_baselines = evaluate_train_only_baselines(
        train_baselines, validation_rows
    )
    bodies = sorted({row["body"] for row in train_rows + validation_rows})
    body_to_id = {name: index for index, name in enumerate(bodies)}
    train_dataset = TransitionDataset(train_rows, body_to_id)
    validation_dataset = TransitionDataset(validation_rows, body_to_id)
    device = torch.device(args.device)
    output.mkdir(parents=True)
    audit.update(
        {
            "train_transitions": len(train_rows),
            "validation_transitions": len(validation_rows),
            "test_transition_count": "unknown_not_loaded",
            "test_group_hdf5_opened": 0,
            "body_to_id": body_to_id,
            "action_schema_to_id": ACTION_SCHEMAS,
            "action_normalization": action_normalization,
            "train_only_baselines": train_baselines,
            "validation_baseline_metrics": validation_baselines,
            "validation_selection_rule": {
                **VALIDATION_SELECTION_RULE,
                "sha256": canonical_json_sha256(VALIDATION_SELECTION_RULE),
            },
        }
    )
    atomic_json(output / "protocol_receipt.json", audit)
    group_order = [str(row["logical_group"]) for row in train_rows]
    bootstrap = logical_group_bootstrap_weights(
        group_order, members=5, seed=args.split_seed
    )
    bootstrap_by_group = {
        group: bootstrap[:, index].tolist() for index, group in enumerate(group_order)
    }
    summaries = []
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_rows,
    )
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = MultibodyCanonicalEventWorldModel(
            ModelConfig(body_count=len(body_to_id))
        ).to(device)
        model.action.set_normalization(
            torch.as_tensor(action_mean, device=device),
            torch.as_tensor(action_std, device=device),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_rows,
        )
        iterator = iter(loader)
        model.train()
        last_loss = math.inf
        best_key: tuple[float, float, float, float, float, int] | None = None
        best_step = 0
        best_score = math.inf
        best_metrics: dict[str, Any] | None = None
        checkpoint = output / f"member_{member:02d}_seed_{seed}_best.pt"
        for step in range(1, args.steps + 1):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = _move_batch(raw, device)
            weights = torch.tensor(
                [bootstrap_by_group[group][member] for group in raw["logical_group"]],
                device=device,
            )
            prediction = model(batch)
            loss, _ = compute_multitask_loss(
                prediction, batch, sample_weight=weights
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite member {member} loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            last_loss = float(loss.detach())
            if step % args.eval_every != 0 and step != args.steps:
                continue
            # Validation is opened, but never used to tune split membership,
            # normalization, or baselines.
            metrics = evaluate_validation_model(model, validation_loader, device)
            score, components = validation_selection_score(
                metrics, validation_baselines
            )
            metrics["selection_score"] = score
            metrics["selection_components"] = components
            selection_key = validation_selection_key(metrics, score, step)
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_step = step
                best_score = score
                best_metrics = metrics
                torch.save(
                    {
                        "format": FORMAT,
                        "model": model.state_dict(),
                        "config": dataclasses.asdict(model.config),
                        "contract": audit,
                        "action_normalization": action_normalization,
                        "train_only_baselines": train_baselines,
                        "validation_baseline_metrics": validation_baselines,
                        "validation_selection_rule": audit[
                            "validation_selection_rule"
                        ],
                        "member": member,
                        "seed": seed,
                        "step": step,
                        "train_loss": last_loss,
                        "validation": metrics,
                        "selection_score": score,
                    },
                    checkpoint,
                )
            model.train()
        if best_metrics is None or not checkpoint.is_file():
            raise RuntimeError(f"member {member} produced no best validation checkpoint")
        summaries.append(
            {
                "member": member,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "train_loss": last_loss,
                "best_step": best_step,
                "best_validation_selection_score": best_score,
                "best_validation": best_metrics,
            }
        )
    summary = {
        "format": FORMAT,
        "status": "training_complete",
        "members": summaries,
        "protocol": audit,
        "sealed_test_evaluated": False,
        "test_group_hdf5_opened": 0,
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic-smoke":
        print("SYNTHETIC_SMOKE=" + json.dumps(run_synthetic_smoke(), sort_keys=True))
        return
    if args.mode == "preflight":
        _, audit, _ = run_preflight(args)
        print("PREFLIGHT=" + json.dumps(audit, sort_keys=True))
        return
    print("TRAINING=" + json.dumps(train(args), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ACTION_SCHEMAS",
    "CANONICAL_BODY_ALIASES",
    "CANONICAL_EVENTS",
    "CanonicalSemanticEncoder",
    "GroupDescriptor",
    "InputBinding",
    "ModelConfig",
    "MultibodyCanonicalEventWorldModel",
    "PerSchemaActionEncoder",
    "TransitionDataset",
    "action_normalization_arrays",
    "body_alias_receipt",
    "canonical_body_name",
    "canonical_event_id",
    "canonical_event_name",
    "canonical_state_vector",
    "censored_lognormal_loss",
    "collate_rows",
    "compute_multitask_loss",
    "derive_predicates_and_events",
    "ensemble_predict",
    "evaluate_train_only_baselines",
    "evaluate_validation_model",
    "fit_train_baselines",
    "fit_train_action_normalization",
    "logical_group_bootstrap_weights",
    "reject_forbidden_path",
    "run_synthetic_smoke",
    "scan_schema5_groups",
    "split_receipt",
    "strict_group_split",
    "validation_selection_key",
    "validation_selection_score",
    "verify_input_bindings",
]
