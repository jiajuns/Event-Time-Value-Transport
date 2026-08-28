#!/usr/bin/env python3
"""Train lightweight scalar-progress controls for the ETSF event model.

This is intentionally *not* a reproduction of VLAC or ProgressVLA.  It is a
same-data ablation with either a direct ``state + action -> progress`` path or
an ``action -> future latent -> progress`` path.  Both variants consume only
schema-v4/v5 counterfactual train and validation roots.  A sealed test root is
not accepted by the CLI and is never opened.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


SUPPORTED_SCHEMAS = (4, 5)
ACTION_DIM = 14
HIDDEN_DIM = 4096
EVENT_NAMES = ("e0", "e12", "e3", "e4", "eK")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def dynamic_phase_progress(
    poses: np.ndarray,
    object_names: Sequence[str],
    calibration: Mapping[str, Any],
    successful_terminal: bool,
) -> float:
    """Collapse reversible task predicates at one state to a scalar in [0, 1].

    The prefix is required because ``moved`` and ``stationary`` depend on
    history.  Lift/near-goal/stationary remain reversible: the returned scalar
    may decrease on a later query.  Success is used only when this prefix ends
    at a genuinely successful terminal state.
    """

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-1] < 3 or len(poses) == 0:
        raise ValueError("poses must have shape [T,O,>=3]")
    names = list(object_names)
    moving_name = str(calibration["moving"])
    if moving_name not in names:
        raise ValueError(f"moving object {moving_name!r} absent from {names}")
    position = poses[:, names.index(moving_name), :3]
    motion = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    cumulative_motion = np.cumsum(motion)

    anchor_name = calibration.get("anchor")
    if anchor_name:
        anchor_name = str(anchor_name)
        if anchor_name not in names:
            raise ValueError(f"anchor object {anchor_name!r} absent from {names}")
        anchor = poses[:, names.index(anchor_name), :3]
        offset = np.asarray(calibration.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
        if offset.shape != (3,):
            raise ValueError("calibration offset must contain xyz")
        goal_distance = np.linalg.norm(position - anchor - offset, axis=1)
    else:
        centers = np.asarray(calibration["centers"], dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("calibration centers must have shape [N,3]")
        goal_distance = np.linalg.norm(position[:, None] - centers[None], axis=2).min(1)

    moved = cumulative_motion >= float(calibration["delta_move"])
    lifted = position[:, 2] >= position[0, 2] + float(calibration["delta_z"])
    near_goal = goal_distance <= float(calibration["tau_d"])
    instant_stationary = near_goal & (motion <= float(calibration["tau_motion"]))
    width = int(calibration["stationary_steps"])
    if width <= 0:
        raise ValueError("stationary_steps must be positive")
    stationary = bool(
        len(poses) >= width and instant_stationary[len(poses) - width :].all()
    )
    phase = 0
    if bool(moved[-1] or lifted[-1]):
        phase = 1
    if bool(near_goal[-1]):
        phase = 2
    if stationary:
        phase = 3
    if successful_terminal:
        phase = 4
    return float(phase / (len(EVENT_NAMES) - 1))


@dataclass
class CandidateExample:
    logical_key: str
    schema_version: int
    candidate_id: int
    query_index: int
    hidden: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    post_hidden: np.ndarray
    progress: float
    success: float


@dataclass
class LoadedRoot:
    root: str
    manifest_sha256: str
    task: str
    body: str
    policy: str
    examples: list[CandidateExample]
    declared_split: str | None = None
    split_manifest_sha256: str | None = None

    @property
    def logical_keys(self) -> list[str]:
        return sorted({example.logical_key for example in self.examples})


def _group_paths(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    items = manifest.get("groups", [])
    paths: list[Path] = []
    for item in items:
        relative = item.get("path") if isinstance(item, Mapping) else None
        if relative:
            path = Path(str(relative))
            paths.append(path if path.is_absolute() else root / "groups" / path)
    if not paths:
        paths = sorted((root / "groups").glob("*.hdf5"))
    if not paths:
        raise RuntimeError(f"no group HDF5 files in {root}")
    return paths


def load_counterfactual_root(
    root: Path,
    calibrations: Mapping[str, Mapping[str, Any]],
) -> LoadedRoot:
    """Load one explicitly assigned train or validation root.

    Schema v4 contributes the common first-query transition.  Schema v5 also
    contributes every audited continuation query, while candidate ranking is
    still evaluated only at ``query_index == 0``.
    """

    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"collection is not complete: {root}")
    manifest_schema = int(manifest.get("schema_version", -1))
    if manifest_schema not in SUPPORTED_SCHEMAS:
        raise RuntimeError(
            f"progress baseline requires schema {SUPPORTED_SCHEMAS}, got {manifest_schema}"
        )
    task_default = str(manifest.get("task", "unknown"))
    body_default = str(manifest.get("body", "unknown"))
    policy_default = str(manifest.get("policy", manifest.get("model_path", "openvla")))
    view_expected: dict[str, str] | None = None
    declared_split: str | None = None
    view_split_manifest_sha256: str | None = None
    if manifest.get("format") == "etsf_progress_split_view_v1":
        split_name = str(manifest.get("split", ""))
        if split_name not in {"train", "validation"}:
            raise RuntimeError("progress split view may contain only train or validation")
        declared_split = split_name
        split_path = Path(str(manifest.get("split_manifest", ""))).expanduser()
        if not split_path.is_absolute() or not split_path.is_file():
            raise RuntimeError("progress split view requires an absolute frozen split manifest")
        view_split_manifest_sha256 = str(manifest.get("split_manifest_sha256", ""))
        if sha256(split_path) != view_split_manifest_sha256:
            raise RuntimeError("progress split view split-manifest SHA256 mismatch")
        rows = manifest.get("groups")
        logical_keys = manifest.get("logical_keys")
        if not isinstance(rows, list) or not isinstance(logical_keys, list) or not rows:
            raise RuntimeError("progress split view lacks group rows/logical keys")
        view_expected = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("progress split view group row must be a mapping")
            forbidden = {"success", "steps", "labels", "outcomes"} & set(row)
            if forbidden:
                raise RuntimeError(
                    f"progress split view embeds forbidden labels: {sorted(forbidden)}"
                )
            path = Path(str(row.get("path", ""))).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise RuntimeError("progress split view paths must be existing absolute HDF5 paths")
            resolved = str(path.resolve())
            if resolved in view_expected:
                raise RuntimeError("progress split view contains a duplicate HDF5 path")
            expected_sha = str(row.get("sha256", ""))
            if len(expected_sha) != 64 or sha256(path) != expected_sha:
                raise RuntimeError(f"progress split view HDF5 SHA256 mismatch: {path}")
            view_expected[resolved] = str(row.get("logical_key", ""))
        if sorted(view_expected.values()) != sorted(str(key) for key in logical_keys):
            raise RuntimeError("progress split view logical-key mirror mismatch")
    examples: list[CandidateExample] = []
    seen_keys: set[str] = set()
    for path in _group_paths(root, manifest):
        path = path.resolve()
        if view_expected is not None and str(path) not in view_expected:
            raise RuntimeError(f"unregistered path in progress split view: {path}")
        with h5py.File(path, "r") as handle:
            schema = int(handle.attrs.get("schema_version", manifest_schema))
            if schema not in SUPPORTED_SCHEMAS:
                raise RuntimeError(f"unsupported schema {schema} in {path}")
            required = {
                "candidate_actions",
                "first_chunk_action_mask",
                "post_chunk_hidden",
                "post_chunk_step",
                "success",
                "steps",
                "object_names",
                "branches",
            }
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise RuntimeError(f"missing {missing} in {path}")
            actions = handle["candidate_actions"][:].astype(np.float32)
            mask = handle["first_chunk_action_mask"][:].astype(bool)
            post_hidden = handle["post_chunk_hidden"][:].astype(np.float32)
            post_steps = handle["post_chunk_step"][:].astype(np.int64)
            terminal_steps = handle["steps"][:].astype(np.int64)
            success = handle["success"][:].astype(np.float32)
            if "pre_hidden" in handle:
                hidden = handle["pre_hidden"][:].astype(np.float32)
            else:
                initial = handle["initial_hidden"][:].astype(np.float32)
                hidden = np.repeat(initial[None], len(actions), axis=0)
            count = len(actions)
            expected = {
                "actions": (count, actions.shape[1], ACTION_DIM),
                "mask": (count, actions.shape[1]),
                "hidden": (count, HIDDEN_DIM),
                "post_hidden": (count, HIDDEN_DIM),
                "post_steps": (count,),
                "terminal_steps": (count,),
                "success": (count,),
            }
            actual = {
                "actions": actions.shape,
                "mask": mask.shape,
                "hidden": hidden.shape,
                "post_hidden": post_hidden.shape,
                "post_steps": post_steps.shape,
                "terminal_steps": terminal_steps.shape,
                "success": success.shape,
            }
            for name, shape in expected.items():
                if actual[name] != shape:
                    raise RuntimeError(f"invalid {name} shape {actual[name]} in {path}")
            for name, value in {
                "actions": actions,
                "hidden": hidden,
                "post_hidden": post_hidden,
                "success": success,
            }.items():
                if not np.isfinite(value).all():
                    raise RuntimeError(f"non-finite {name} in {path}")
            lengths = mask.sum(1)
            if (lengths <= 0).any() or not np.array_equal(lengths, post_steps):
                raise RuntimeError(f"action-mask/post-step mismatch in {path}")
            task = str(handle.attrs.get("task", task_default))
            body = str(handle.attrs.get("body", body_default))
            policy = str(handle.attrs.get("policy", policy_default))
            if task not in calibrations:
                raise RuntimeError(f"event spec has no calibration for task {task!r}")
            resolved_seed = int(handle.attrs.get("resolved_seed", handle.attrs.get("seed", -1)))
            if resolved_seed < 0:
                raise RuntimeError(f"missing seed in {path}")
            logical_key = f"{task}|{body}|{resolved_seed}"
            if (
                view_expected is not None
                and view_expected[str(path)] != logical_key
            ):
                raise RuntimeError(
                    f"progress split view logical key differs from HDF5 attrs: {path}"
                )
            if logical_key in seen_keys:
                raise RuntimeError(f"duplicate logical group {logical_key}")
            seen_keys.add(logical_key)
            object_names = decode_strings(handle["object_names"][:])
            branches = handle["branches"]
            if len(branches) != count:
                raise RuntimeError(f"candidate/trajectory count mismatch in {path}")
            for candidate_id in range(count):
                branch_name = f"candidate_{candidate_id:03d}"
                if branch_name not in branches or "object_poses" not in branches[branch_name]:
                    raise RuntimeError(f"missing {branch_name}/object_poses in {path}")
                branch = branches[branch_name]
                poses = branch["object_poses"][:].astype(np.float32)
                if poses.shape[0] != int(terminal_steps[candidate_id]) + 1:
                    raise RuntimeError(f"trajectory length mismatch in {path}:{branch_name}")
                query_fields = {
                    "query_steps",
                    "query_post_steps",
                    "query_hidden",
                    "query_post_hidden",
                    "query_actions",
                    "query_action_mask",
                }
                if schema == 5:
                    query_missing = sorted(query_fields - set(branch.keys()))
                    if query_missing:
                        raise RuntimeError(
                            f"schema-v5 branch lacks {query_missing} in {path}:{branch_name}"
                        )
                    query_steps = branch["query_steps"][:].astype(np.int64)
                    query_post_steps = branch["query_post_steps"][:].astype(np.int64)
                    query_hidden = branch["query_hidden"][:].astype(np.float32)
                    query_post_hidden = branch["query_post_hidden"][:].astype(np.float32)
                    query_actions = branch["query_actions"][:].astype(np.float32)
                    query_mask = branch["query_action_mask"][:].astype(bool)
                else:
                    query_steps = np.asarray([0], dtype=np.int64)
                    query_post_steps = np.asarray([post_steps[candidate_id]], dtype=np.int64)
                    query_hidden = hidden[candidate_id : candidate_id + 1]
                    query_post_hidden = post_hidden[candidate_id : candidate_id + 1]
                    query_actions = actions[candidate_id : candidate_id + 1]
                    query_mask = mask[candidate_id : candidate_id + 1]
                query_count = len(query_steps)
                query_expected = {
                    "query_post_steps": (query_count,),
                    "query_hidden": (query_count, HIDDEN_DIM),
                    "query_post_hidden": (query_count, HIDDEN_DIM),
                    "query_actions": (query_count, actions.shape[1], ACTION_DIM),
                    "query_action_mask": (query_count, actions.shape[1]),
                }
                query_actual = {
                    "query_post_steps": query_post_steps.shape,
                    "query_hidden": query_hidden.shape,
                    "query_post_hidden": query_post_hidden.shape,
                    "query_actions": query_actions.shape,
                    "query_action_mask": query_mask.shape,
                }
                for field, shape in query_expected.items():
                    if query_actual[field] != shape:
                        raise RuntimeError(
                            f"invalid {field} shape {query_actual[field]} in {path}:{branch_name}"
                        )
                if query_count == 0 or int(query_steps[0]) != 0:
                    raise RuntimeError(f"query sequence must include step zero in {path}:{branch_name}")
                if (
                    (np.diff(query_steps) <= 0).any()
                    or not np.array_equal(query_steps[1:], query_post_steps[:-1])
                    or (
                        schema == 5
                        and int(query_post_steps[-1]) != int(terminal_steps[candidate_id])
                    )
                    or (
                        query_count > 1
                        and not np.array_equal(query_hidden[1:], query_post_hidden[:-1])
                    )
                ):
                    raise RuntimeError(f"query chain is not contiguous in {path}:{branch_name}")
                query_lengths = query_mask.sum(1)
                expected_query_masks = (
                    np.arange(actions.shape[1])[None] < query_lengths[:, None]
                )
                if (
                    (query_lengths <= 0).any()
                    or not np.array_equal(query_steps + query_lengths, query_post_steps)
                    or (query_post_steps > terminal_steps[candidate_id]).any()
                    or not np.array_equal(query_mask, expected_query_masks)
                ):
                    raise RuntimeError(f"invalid query boundaries in {path}:{branch_name}")
                if not (
                    np.array_equal(query_hidden[0], hidden[candidate_id])
                    and np.array_equal(query_post_hidden[0], post_hidden[candidate_id])
                    and np.array_equal(query_actions[0], actions[candidate_id])
                    and np.array_equal(query_mask[0], mask[candidate_id])
                ):
                    raise RuntimeError(f"first-query/top-level mismatch in {path}:{branch_name}")
                for field, value in {
                    "query_hidden": query_hidden,
                    "query_post_hidden": query_post_hidden,
                    "query_actions": query_actions,
                }.items():
                    if not np.isfinite(value).all():
                        raise RuntimeError(f"non-finite {field} in {path}:{branch_name}")
                for query_index, post_step_value in enumerate(query_post_steps):
                    post_step = int(post_step_value)
                    if not 0 < post_step < len(poses):
                        raise RuntimeError(
                            f"post step is outside trajectory in {path}:{branch_name}"
                        )
                    successful_terminal = bool(
                        success[candidate_id] > 0.5
                        and post_step == int(terminal_steps[candidate_id])
                    )
                    progress = dynamic_phase_progress(
                        poses[: post_step + 1],
                        object_names,
                        calibrations[task],
                        successful_terminal,
                    )
                    examples.append(
                        CandidateExample(
                            logical_key=logical_key,
                            schema_version=schema,
                            candidate_id=candidate_id,
                            query_index=query_index,
                            hidden=query_hidden[query_index],
                            actions=query_actions[query_index],
                            action_mask=query_mask[query_index],
                            post_hidden=query_post_hidden[query_index],
                            progress=progress,
                            success=float(success[candidate_id]),
                        )
                    )
    if not examples:
        raise RuntimeError(f"no candidate examples in {root}")
    if view_expected is not None and sorted(seen_keys) != sorted(view_expected.values()):
        raise RuntimeError("progress split view did not load its exact logical-key set")
    tasks = {example.logical_key.split("|", 1)[0] for example in examples}
    if len(tasks) != 1:
        task_default = "mixed"
    return LoadedRoot(
        root=str(root),
        manifest_sha256=sha256(manifest_path),
        task=task_default,
        body=body_default,
        policy=policy_default,
        examples=examples,
        declared_split=declared_split,
        split_manifest_sha256=view_split_manifest_sha256,
    )


def audit_split(
    train: LoadedRoot,
    validation: LoadedRoot,
    split_manifest: Path | None = None,
) -> dict[str, Any]:
    overlap = set(train.logical_keys) & set(validation.logical_keys)
    if overlap:
        raise RuntimeError(f"train/validation logical-group leakage: {sorted(overlap)}")
    audit: dict[str, Any] = {
        "train_logical_keys": train.logical_keys,
        "validation_logical_keys": validation.logical_keys,
        "sealed_test_policy": "not accepted_by_cli_not_loaded_not_evaluated",
    }
    if split_manifest is not None:
        split_path = split_manifest.resolve()
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split_digest = sha256(split_path)
        for label, loaded in (("train", train), ("validation", validation)):
            if loaded.declared_split not in (None, label):
                raise RuntimeError(f"{label} data was built from a {loaded.declared_split} view")
            if (
                loaded.split_manifest_sha256 is not None
                and loaded.split_manifest_sha256 != split_digest
            ):
                raise RuntimeError(f"{label} view is bound to a different split manifest")
        # Intentionally access only development split keys.  In particular,
        # never resolve or open anything referenced by an optional test key.
        expected_train = sorted(str(item) for item in split["train"])
        expected_validation = sorted(str(item) for item in split["validation"])
        if expected_train != train.logical_keys:
            raise RuntimeError("train root does not match the frozen split manifest")
        if expected_validation != validation.logical_keys:
            raise RuntimeError("validation root does not match the frozen split manifest")
        audit["split_manifest"] = str(split_path)
        audit["split_manifest_sha256"] = split_digest
    return audit


class ProgressDataset(Dataset):
    def __init__(self, examples: Sequence[CandidateExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "hidden": torch.from_numpy(example.hidden),
            "actions": torch.from_numpy(example.actions),
            "action_mask": torch.from_numpy(example.action_mask),
            "post_hidden": torch.from_numpy(example.post_hidden),
            "progress": torch.tensor(example.progress, dtype=torch.float32),
            "success": torch.tensor(example.success, dtype=torch.float32),
            "logical_key": example.logical_key,
            "candidate_id": example.candidate_id,
            "query_index": example.query_index,
        }


class FixedHiddenProjector(nn.Module):
    """Deterministic JL projection; it has no trainable VLA-sized backbone."""

    def __init__(self, hidden_dim: int, latent_dim: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(hidden_dim, latent_dim, generator=generator)
        projection.mul_(1.0 / math.sqrt(hidden_dim))
        self.register_buffer("projection", projection)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        return F.layer_norm(normalized @ self.projection, (self.projection.shape[1],))


@dataclass(frozen=True)
class ProgressModelConfig:
    variant: str = "direct"
    hidden_dim: int = HIDDEN_DIM
    action_dim: int = ACTION_DIM
    latent_dim: int = 64
    action_hidden_dim: int = 48
    projection_seed: int = 20260827


class ScalarProgressBaseline(nn.Module):
    def __init__(
        self,
        config: ProgressModelConfig,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if config.variant not in {"direct", "latent_future"}:
            raise ValueError("variant must be direct or latent_future")
        self.config = config
        self.projector = FixedHiddenProjector(
            config.hidden_dim, config.latent_dim, config.projection_seed
        )
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float().clamp_min(1e-4))
        self.action_gru = nn.GRU(
            config.action_dim, config.action_hidden_dim, batch_first=True
        )
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )
        transition_dim = config.latent_dim * 3
        if config.variant == "direct":
            self.progress_head = nn.Sequential(
                nn.Linear(transition_dim, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, 1),
            )
        else:
            self.future_head = nn.Sequential(
                nn.Linear(transition_dim, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
            )
            self.progress_head = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim // 2),
                nn.GELU(),
                nn.Linear(config.latent_dim // 2, 1),
            )

    def encode_action(self, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = (actions - self.action_mean) / self.action_std
        lengths = mask.long().sum(1)
        if (lengths <= 0).any():
            raise ValueError("every action block must contain a valid step")
        packed = pack_padded_sequence(
            normalized,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final = self.action_gru(packed)
        return self.action_projection(final[-1])

    def forward(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        post_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.projector(hidden)
        action = self.encode_action(actions, action_mask)
        transition = torch.cat([state, action, state * action], dim=-1)
        output: dict[str, torch.Tensor] = {}
        if self.config.variant == "direct":
            progress_logit = self.progress_head(transition).squeeze(-1)
        else:
            future = self.future_head(transition)
            output["future_latent"] = future
            progress_logit = self.progress_head(future).squeeze(-1)
            if post_hidden is not None:
                with torch.no_grad():
                    output["target_future_latent"] = self.projector(post_hidden)
        output["progress_logit"] = progress_logit
        output["progress"] = torch.sigmoid(progress_logit)
        return output


def action_statistics(examples: Sequence[CandidateExample]) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for example in examples:
        values.append(example.actions[example.action_mask])
    actions = torch.from_numpy(np.concatenate(values).astype(np.float32))
    return actions.mean(0), actions.std(0, unbiased=False).clamp_min(1e-3)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def progress_loss(
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    latent_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    scalar = F.smooth_l1_loss(output["progress"], target)
    loss = scalar
    latent = scalar.new_zeros(())
    if "future_latent" in output:
        predicted = output["future_latent"]
        expected = output["target_future_latent"]
        latent = (1.0 - F.cosine_similarity(predicted, expected, dim=-1)).mean()
        latent = latent + 0.1 * F.smooth_l1_loss(predicted, expected)
        loss = loss + latent_weight * latent
    return loss, {"progress": float(scalar.detach()), "latent": float(latent.detach())}


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels > 0.5).sum())
    negative_count = int((labels < 0.5).sum())
    if not positive_count or not negative_count:
        return None
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    rank_sum = float(ranks[labels > 0.5].sum())
    statistic = rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(statistic / (positive_count * negative_count))


@torch.no_grad()
def evaluate(
    model: ScalarProgressBaseline,
    examples: Sequence[CandidateExample],
    batch_size: int,
    device: torch.device,
    bootstrap_seed: int,
    *,
    include_policy_diagnostics: bool = True,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(ProgressDataset(examples), batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    successes: list[np.ndarray] = []
    keys: list[str] = []
    candidate_ids: list[int] = []
    query_indices: list[int] = []
    latent_cosines: list[np.ndarray] = []
    for raw in loader:
        batch = move_batch(raw, device)
        output = model(
            batch["hidden"], batch["actions"], batch["action_mask"], batch["post_hidden"]
        )
        predictions.append(output["progress"].cpu().numpy())
        targets.append(batch["progress"].cpu().numpy())
        successes.append(batch["success"].cpu().numpy())
        keys.extend(raw["logical_key"])
        candidate_ids.extend(int(item) for item in raw["candidate_id"])
        query_indices.extend(int(item) for item in raw["query_index"])
        if "future_latent" in output:
            latent_cosines.append(
                F.cosine_similarity(
                    output["future_latent"], output["target_future_latent"], dim=-1
                ).cpu().numpy()
            )
    predicted = np.concatenate(predictions)
    target = np.concatenate(targets)
    success = np.concatenate(successes)
    candidate_ids_array = np.asarray(candidate_ids)
    query_indices_array = np.asarray(query_indices)
    keys_array = np.asarray(keys)
    metrics: dict[str, Any] = {
        "examples": int(len(examples)),
        "first_query_examples": int(np.sum(query_indices_array == 0)),
        "continuation_query_examples": int(np.sum(query_indices_array > 0)),
        "groups": int(len(set(keys))),
        "progress_mae": float(np.mean(np.abs(predicted - target))),
        "progress_rmse": float(np.sqrt(np.mean(np.square(predicted - target)))),
        "progress_prediction_mean": float(predicted.mean()),
        "progress_target_mean": float(target.mean()),
        "policy_diagnostics_included": include_policy_diagnostics,
    }
    if include_policy_diagnostics:
        selected: list[float] = []
        baseline: list[float] = []
        oracle: list[float] = []
        selected_ids: list[int] = []
        pair_correct: list[float] = []
        ndcg: list[float] = []
        for key in sorted(set(keys)):
            # Later v5 continuation queries are dense regression supervision,
            # not aligned alternatives. Candidate ranking uses only query 0.
            indices = np.flatnonzero(
                (keys_array == key) & (query_indices_array == 0)
            )
            base_indices = indices[candidate_ids_array[indices] == 0]
            if len(base_indices) != 1:
                raise RuntimeError(f"logical group {key} lacks a unique candidate 0")
            picked = indices[int(np.argmax(predicted[indices]))]
            base = int(base_indices[0])
            selected.append(float(success[picked]))
            baseline.append(float(success[base]))
            oracle.append(float(success[indices].max()))
            selected_ids.append(int(candidate_ids_array[picked]))
            positive = predicted[indices][success[indices] > 0.5]
            negative = predicted[indices][success[indices] < 0.5]
            if len(positive) and len(negative):
                pair_difference = positive[:, None] - negative[None]
                pair_correct.extend(
                    (pair_difference > 0).astype(np.float32).ravel().tolist()
                )
            order = indices[np.argsort(-predicted[indices])]
            gains = success[order]
            discounts = 1.0 / np.log2(np.arange(len(order)) + 2.0)
            dcg = float(np.sum(gains * discounts))
            ideal = float(np.sum(np.sort(success[indices])[::-1] * discounts))
            if ideal > 0:
                ndcg.append(dcg / ideal)
        selected_array = np.asarray(selected)
        baseline_array = np.asarray(baseline)
        difference = selected_array - baseline_array
        rng = np.random.default_rng(bootstrap_seed)
        bootstrap = difference[
            rng.integers(0, len(difference), size=(5000, len(difference)))
        ].mean(1)
        first_query = query_indices_array == 0
        metrics.update(
            {
                # Continuation queries are temporally related supervision from
                # one branch, not additional aligned candidate alternatives.
                "candidate_success_auc": _binary_auc(
                    success[first_query], predicted[first_query]
                ),
                "candidate_success_auc_scope": "first_query_candidates_only",
                "within_group_success_pair_accuracy": (
                    float(np.mean(pair_correct)) if pair_correct else None
                ),
                "candidate_success_ndcg": float(np.mean(ndcg)) if ndcg else None,
                "baseline_success_rate": float(baseline_array.mean()),
                "selected_success_rate": float(selected_array.mean()),
                "oracle_success_rate": float(np.mean(oracle)),
                "paired_success_difference": float(difference.mean()),
                "paired_difference_ci95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "changed_groups": int(np.sum(np.asarray(selected_ids) != 0)),
                "improved_groups": int(np.sum(difference > 0)),
                "harmed_groups": int(np.sum(difference < 0)),
                "selected_candidate_ids": selected_ids,
            }
        )
    if latent_cosines:
        metrics["future_latent_cosine"] = float(np.concatenate(latent_cosines).mean())
    return metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_baseline(
    config: ProgressModelConfig,
    train_examples: Sequence[CandidateExample],
    validation_examples: Sequence[CandidateExample],
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    latent_weight: float,
    seed: int,
    evaluation_interval: int,
) -> tuple[ScalarProgressBaseline, dict[str, Any]]:
    set_seed(seed)
    action_mean, action_std = action_statistics(train_examples)
    model = ScalarProgressBaseline(config, action_mean, action_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ProgressDataset(train_examples),
        batch_size=min(batch_size, len(train_examples)),
        shuffle=True,
        generator=generator,
    )
    iterator = iter(loader)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: float | None = None
    best_step: int | None = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batch = move_batch(raw, device)
        model.train()
        output = model(
            batch["hidden"], batch["actions"], batch["action_mask"], batch["post_hidden"]
        )
        loss, parts = progress_loss(output, batch["progress"], latent_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % evaluation_interval == 0 or step == steps:
            validation = evaluate(
                model, validation_examples, batch_size, device, seed + step
            )
            # Checkpointing uses scalar progress accuracy only.  Terminal
            # success is reported as a downstream diagnostic, never optimized
            # or used to choose a checkpoint.
            score = -float(validation["progress_mae"])
            row = {
                "step": step,
                "train_batch_loss": float(loss.detach()),
                "train_batch_parts": parts,
                "validation": validation,
            }
            history.append(row)
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = validation
    if best_state is None or best_metrics is None or best_step is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_validation": best_metrics,
        "selection": {
            "data": "validation_only",
            "metric": "progress_mae",
            "mode": "min",
            "best_step": best_step,
            "best_value": float(best_metrics["progress_mae"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["direct", "latent_future"], default="direct")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--action-hidden-dim", type=int, default=48)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.evaluation_interval <= 0:
        parser.error("steps, batch-size and evaluation-interval must be positive")
    if args.latent_dim < 4 or args.action_hidden_dim < 4:
        parser.error("latent dimensions must be at least 4")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    train_roots = {str(path.resolve()) for path in args.train_data}
    validation_roots = {str(path.resolve()) for path in args.validation_data}
    if train_roots & validation_roots:
        parser.error("the same data root cannot be both train and validation")

    event_spec = json.loads(args.event_spec.read_text(encoding="utf-8"))
    calibrations = event_spec.get("calibration")
    if not isinstance(calibrations, Mapping):
        raise RuntimeError("event spec has no calibration mapping")
    train_parts = [load_counterfactual_root(path, calibrations) for path in args.train_data]
    validation_parts = [
        load_counterfactual_root(path, calibrations) for path in args.validation_data
    ]

    def merge(parts: Sequence[LoadedRoot], label: str) -> LoadedRoot:
        examples = [example for part in parts for example in part.examples]
        keys = [key for part in parts for key in part.logical_keys]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate logical groups across {label} roots")
        declared = {part.declared_split for part in parts if part.declared_split is not None}
        split_digests = {
            part.split_manifest_sha256
            for part in parts
            if part.split_manifest_sha256 is not None
        }
        if declared and declared != {label}:
            raise RuntimeError(f"{label} inputs contain a differently declared split view")
        if len(split_digests) > 1:
            raise RuntimeError(f"{label} views bind different split manifests")
        return LoadedRoot(
            root=";".join(part.root for part in parts),
            manifest_sha256=";".join(part.manifest_sha256 for part in parts),
            task="mixed" if len({part.task for part in parts}) > 1 else parts[0].task,
            body="mixed" if len({part.body for part in parts}) > 1 else parts[0].body,
            policy="mixed" if len({part.policy for part in parts}) > 1 else parts[0].policy,
            examples=examples,
            declared_split=next(iter(declared)) if declared else None,
            split_manifest_sha256=(
                next(iter(split_digests)) if split_digests else None
            ),
        )

    train = merge(train_parts, "train")
    validation = merge(validation_parts, "validation")
    split_audit = audit_split(train, validation, args.split_manifest)
    device = torch.device(args.device)
    config = ProgressModelConfig(
        variant=args.variant,
        latent_dim=args.latent_dim,
        action_hidden_dim=args.action_hidden_dim,
        projection_seed=args.seed,
    )
    model, result = train_baseline(
        config=config,
        train_examples=train.examples,
        validation_examples=validation.examples,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_weight=args.latent_weight,
        seed=args.seed,
        evaluation_interval=args.evaluation_interval,
    )
    train_metrics = evaluate(
        model,
        train.examples,
        args.batch_size,
        device,
        args.seed + 100000,
        include_policy_diagnostics=False,
    )
    validation_metrics = evaluate(
        model, validation.examples, args.batch_size, device, args.seed + 200000
    )
    args.output.mkdir(parents=True, exist_ok=True)
    def root_contract(part: LoadedRoot) -> dict[str, Any]:
        return {
            "root": part.root,
            "manifest_sha256": part.manifest_sha256,
            "task": part.task,
            "body": part.body,
            "policy": part.policy,
            "examples": len(part.examples),
            "logical_groups": len(part.logical_keys),
        }

    contract = {
        "method": (
            "state_action_to_scalar_progress"
            if args.variant == "direct"
            else "action_conditioned_future_hidden_projection_to_scalar_progress"
        ),
        "scope": "lightweight_same_data_ablation_not_full_VLAC_or_ProgressVLA_reproduction",
        "target": "dynamic_reversible_event_phase_collapsed_to_[0,1]_at_post_chunk",
        "success_supervision": "terminal_eK_progress_target_only",
        "success_loss": False,
        "checkpoint_selection": "validation_progress_mae_only",
        "model_config": dataclasses.asdict(config),
        "optimization": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "latent_weight": args.latent_weight,
            "evaluation_interval": args.evaluation_interval,
        },
        "training_seed": args.seed,
        "candidate_policy_diagnostics": "validation_only",
        "schema_versions": list(SUPPORTED_SCHEMAS),
        "schema_v5_policy": "all_contiguous_queries_train_progress_first_query_only_for_candidate_ranking",
        "train_roots": [root_contract(part) for part in train_parts],
        "validation_roots": [root_contract(part) for part in validation_parts],
        "event_spec": str(args.event_spec.resolve()),
        "event_spec_sha256": sha256(args.event_spec),
        "split_audit": split_audit,
    }
    checkpoint = args.output / f"openvla_etsf_progress_{args.variant}.pt"
    atomic_torch_save(
        checkpoint,
        {
            "format": "etsf_scalar_progress_baseline_v1",
            "model": model.state_dict(),
            "config": dataclasses.asdict(config),
            "contract": contract,
        },
    )
    summary = {
        "format": "etsf_scalar_progress_baseline_v1",
        "status": "training_complete",
        "variant": args.variant,
        "training_seed": args.seed,
        "config": dataclasses.asdict(config),
        "train": train_metrics,
        "validation": validation_metrics,
        "training": result,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "contract": contract,
        "limitations": [
            "This control has no large-scale heterogeneous video pretraining.",
            "It does not reproduce VLAC actor/critic generation or RL fine-tuning.",
            "It does not reproduce ProgressVLA diffusion guidance or latent-action expert.",
            "Offline candidate selection metrics do not establish closed-loop success improvement.",
            "Sealed test data was not loaded or evaluated.",
        ],
    }
    atomic_json(args.output / f"progress_{args.variant}_summary.json", summary)
    print(
        "PROGRESS_BASELINE_COMPLETE="
        + json.dumps(
            {
                "variant": args.variant,
                "validation": validation_metrics,
                "checkpoint": str(checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
