#!/usr/bin/env python3
"""Safety-gated three-seed progress-baseline suite for the formal v5 root.

The source root contains the counterfactual trainer's deterministic 70/15/15
split.  This launcher writes manifest-only train/validation views containing
absolute HDF5 paths; the internal sealed 15 logical groups are recorded only
as keys and are never passed to the progress trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from launch_openvla_etsf_counterfactual_v5 import (
    DEFAULT_CODE_ROOT,
    DEFAULT_DATA,
    DEFAULT_EVENT_SPEC,
    DEFAULT_FACTUAL_ROOT,
    DEFAULT_OUTPUT as DEFAULT_COUNTERFACTUAL_OUTPUT,
    DEFAULT_PYTHON,
    EXPECTED_GROUPS,
    SEEDS,
    atomic_json,
    audit_collector,
    audit_factual_summaries,
    crosscheck_training_seeds,
    sha256,
    validate_complete_output as validate_counterfactual_output,
    wait_for_collector,
    wait_for_factual_members,
    wait_for_gpu_idle,
)
from train_openvla_etsf_counterfactual import (
    SPLIT_SEED,
    GroupDescriptor,
    make_group_splits,
    scan_group_descriptors,
)


FORMAT = "etsf_progress_v5_launch_audit_v1"
VIEW_FORMAT = "etsf_progress_split_view_v1"
SUITE_FORMAT = "etsf_progress_baseline_suite_v1"
VARIANTS = ("direct", "latent_future")
DEFAULT_OUTPUT = Path("/home/user/etsf_openvla_progress_v5_move_can_pot_20260827")


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def counterfactual_ready(root: Path) -> tuple[bool, str]:
    audit_path = root / "launch_audit.json"
    if not audit_path.is_file():
        return False, "counterfactual launch audit is absent"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"counterfactual launch audit is not readable yet: {error}"
    status = str(audit.get("status", ""))
    if status.startswith("failed"):
        raise RuntimeError(f"counterfactual prerequisite failed: {status}")
    if status != "launcher_complete":
        return False, f"counterfactual launcher status={status!r}"
    required = (
        root / "ensemble_manifest.json",
        root / "split_manifest.json",
        root / "counterfactual_ensemble.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"completed counterfactual prerequisite lacks artifacts: {missing}"
        )
    return True, "complete"


def wait_for_counterfactual(
    root: Path, timeout_seconds: float, poll_seconds: float
) -> None:
    started = time.monotonic()
    while True:
        ready, reason = counterfactual_ready(root)
        if ready:
            return
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"counterfactual ensemble was not ready within {timeout_seconds}s: {reason}"
            )
        delay = min(poll_seconds, timeout_seconds - elapsed, 60.0)
        print(
            "WAITING_FOR_COUNTERFACTUAL="
            + json.dumps(
                {"root": str(root), "reason": reason, "elapsed_seconds": elapsed},
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(delay, 0.01))


def validate_counterfactual_split(
    root: Path,
    splits: Mapping[str, Sequence[str]],
    descriptors: Sequence[GroupDescriptor],
    event_spec_sha256: str,
) -> dict[str, Any]:
    split_path = root / "split_manifest.json"
    ensemble_path = root / "ensemble_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    contract = ensemble.get("contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("counterfactual ensemble lacks a contract")
    for name in ("train", "validation", "test"):
        expected = sorted(str(key) for key in splits[name])
        if sorted(str(key) for key in split.get(name, [])) != expected:
            raise RuntimeError(
                f"counterfactual split_manifest differs from deterministic {name} split"
            )
        contract_name = "sealed_test_groups" if name == "test" else f"{name}_groups"
        if sorted(str(key) for key in contract.get(contract_name, [])) != expected:
            raise RuntimeError(
                f"counterfactual ensemble contract differs from deterministic {name} split"
            )
    if str(contract.get("event_spec_sha256", "")) != event_spec_sha256:
        raise RuntimeError("counterfactual ensemble event-spec SHA256 mismatch")
    if int(contract.get("schema_counts", {}).get("5", 0)) != EXPECTED_GROUPS:
        raise RuntimeError("counterfactual ensemble does not cover exactly 100 schema-v5 groups")
    group_rows = contract.get("group_files")
    if not isinstance(group_rows, list) or len(group_rows) != len(descriptors):
        raise RuntimeError("counterfactual group-file provenance is incomplete")
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in group_rows:
        if not isinstance(row, Mapping) or not row.get("logical_key"):
            raise RuntimeError("invalid counterfactual group-file provenance row")
        key = str(row["logical_key"])
        if key in by_key:
            raise RuntimeError("duplicate counterfactual group-file logical key")
        if {"success", "steps", "labels", "outcomes"} & set(row):
            raise RuntimeError("counterfactual group-file provenance embeds labels")
        by_key[key] = row
    descriptor_keys = {descriptor.logical_key for descriptor in descriptors}
    if set(by_key) != descriptor_keys:
        raise RuntimeError("counterfactual group-file provenance differs from source root")
    for descriptor in descriptors:
        row = by_key[descriptor.logical_key]
        if (
            int(row.get("schema_version", -1)) != 5
            or Path(str(row.get("path", ""))).resolve()
            != Path(descriptor.path).resolve()
            or not row.get("sha256")
        ):
            raise RuntimeError(
                f"counterfactual group-file identity mismatch: {descriptor.logical_key}"
            )
    return {
        "ensemble_manifest": str(ensemble_path.resolve()),
        "ensemble_manifest_sha256": sha256(ensemble_path),
        "counterfactual_split_manifest": str(split_path.resolve()),
        "counterfactual_split_manifest_sha256": sha256(split_path),
        "group_file_contract": by_key,
    }


def build_split_views(
    *,
    data_root: Path,
    output: Path,
    collector: Mapping[str, Any],
    descriptors: Sequence[GroupDescriptor],
    counterfactual_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if len(descriptors) != EXPECTED_GROUPS:
        raise RuntimeError("formal progress split requires exactly 100 logical groups")
    if any(descriptor.schema_version != 5 for descriptor in descriptors):
        raise RuntimeError("formal progress split refuses non-schema-v5 groups")
    splits = make_group_splits(descriptors)
    if tuple(len(splits[name]) for name in ("train", "validation", "test")) != (
        70,
        15,
        15,
    ):
        raise RuntimeError("deterministic progress split is not 70/15/15")
    descriptor_map = {descriptor.logical_key: descriptor for descriptor in descriptors}
    provenance = counterfactual_audit["group_file_contract"]
    split_path = (output / "split_views" / "split_manifest.json").resolve()
    split_payload = {
        "format": "etsf_progress_deterministic_split_v1",
        "source_data_root": str(data_root.resolve()),
        "source_manifest": str(Path(collector["manifest"]).resolve()),
        "source_manifest_sha256": str(collector["manifest_sha256"]),
        "algorithm": "train_openvla_etsf_counterfactual.make_group_splits",
        "split_seed": SPLIT_SEED,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "train": list(splits["train"]),
        "validation": list(splits["validation"]),
        "test": list(splits["test"]),
        "sealed_test_group_count": len(splits["test"]),
        "sealed_test_policy": (
            "logical_keys_only_no_hdf5_path_sha_or_label_dataset_passed_to_progress"
        ),
        "counterfactual_split_manifest": counterfactual_audit[
            "counterfactual_split_manifest"
        ],
        "counterfactual_split_manifest_sha256": counterfactual_audit[
            "counterfactual_split_manifest_sha256"
        ],
    }
    split_digest = payload_sha256(split_payload)
    views: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        rows = []
        split_descriptors = [descriptor_map[key] for key in splits[split_name]]
        tasks = {descriptor.task for descriptor in split_descriptors}
        bodies = {descriptor.body for descriptor in split_descriptors}
        policies = {descriptor.policy for descriptor in split_descriptors}
        if len(tasks) != 1 or len(bodies) != 1 or len(policies) != 1:
            raise RuntimeError("formal progress views require one task/body/policy contract")
        for descriptor in split_descriptors:
            path = Path(descriptor.path).resolve()
            digest = sha256(path)
            recorded_digest = str(provenance[descriptor.logical_key].get("sha256", ""))
            if digest != recorded_digest:
                raise RuntimeError(
                    f"train/validation HDF5 changed after counterfactual training: {path}"
                )
            rows.append(
                {
                    "logical_key": descriptor.logical_key,
                    "schema_version": 5,
                    "resolved_seed": descriptor.seed,
                    "path": str(path),
                    "sha256": digest,
                }
            )
        view_path = (output / "split_views" / split_name / "manifest.json").resolve()
        payload = {
            "format": VIEW_FORMAT,
            "status": "complete",
            "schema_version": 5,
            "split": split_name,
            "task": next(iter(tasks)),
            "body": next(iter(bodies)),
            "policy": next(iter(policies)),
            "source_data_root": str(data_root.resolve()),
            "source_manifest_sha256": str(collector["manifest_sha256"]),
            "split_manifest": str(split_path),
            "split_manifest_sha256": split_digest,
            "logical_keys": list(splits[split_name]),
            "groups": rows,
            "labels_embedded_in_view": False,
            "hdf5_paths": "absolute",
            "sealed_test_groups_in_view": 0,
        }
        views[split_name] = {
            "path": str(view_path),
            "payload": payload,
            "sha256": payload_sha256(payload),
        }
    return {
        "split_manifest": {
            "path": str(split_path),
            "payload": split_payload,
            "sha256": split_digest,
        },
        "views": views,
        "counts": {name: len(splits[name]) for name in splits},
        "sealed_hdf5_paths_recorded_in_views": False,
        "sealed_hdf5_sha256_computed_by_view_builder": False,
    }


def build_commands(args: argparse.Namespace, views: Mapping[str, Any]) -> list[dict[str, Any]]:
    trainer = args.trainer.resolve()
    # Keep the virtual-environment launcher path intact.  Resolving this
    # symlink executes the base interpreter without the venv site-packages.
    python_bin = args.python_bin.expanduser().absolute()
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    commands = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_output = (args.output / variant / f"seed_{seed}").resolve()
            command = [
                str(python_bin),
                str(trainer),
                "--train-data",
                str(Path(views["views"]["train"]["path"]).parent),
                "--validation-data",
                str(Path(views["views"]["validation"]["path"]).parent),
                "--event-spec",
                str(args.event_spec.resolve()),
                "--split-manifest",
                str(views["split_manifest"]["path"]),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--output",
                str(run_output),
                "--device",
                "cuda",
                "--steps",
                "2000",
                "--latent-dim",
                "64",
                "--action-hidden-dim",
                "48",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.0003",
                "--latent-weight",
                "0.5",
                "--evaluation-interval",
                "50",
            ]
            commands.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "output": str(run_output),
                    "argv": command,
                    "argv_sha256": hashlib.sha256(
                        json.dumps(command, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                }
            )
    return commands


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    event_spec = args.event_spec.resolve()
    if not event_spec.is_file():
        raise FileNotFoundError(event_spec)
    event_digest = sha256(event_spec)
    data_root = args.data.resolve()
    wait_for_collector(data_root, args.wait_timeout_seconds, args.poll_seconds)
    collector = audit_collector(data_root, event_digest)
    factual_root = args.factual_root.resolve()
    wait_for_factual_members(
        factual_root, args.wait_timeout_seconds, args.poll_seconds
    )
    factual_rows, selected = audit_factual_summaries(factual_root, event_digest)
    selected_payload = torch.load(
        Path(str(selected["checkpoint"])), map_location="cpu", weights_only=False
    )
    factual_contract = selected_payload.get("contract")
    if not isinstance(factual_contract, Mapping):
        raise RuntimeError("selected factual checkpoint lacks a contract")
    collector["seed_registry_audit"] = crosscheck_training_seeds(
        collector, factual_contract
    )
    counterfactual_root = args.counterfactual_root.resolve()
    wait_for_counterfactual(
        counterfactual_root, args.wait_timeout_seconds, args.poll_seconds
    )
    counterfactual_complete = validate_counterfactual_output(
        counterfactual_root,
        selected["checkpoint_sha256"],
        collector["manifest_sha256"],
    )
    if counterfactual_complete is None:
        raise RuntimeError("counterfactual prerequisite is not a complete formal output")
    descriptors = scan_group_descriptors([data_root])
    computed_splits = make_group_splits(descriptors)
    counterfactual_audit = validate_counterfactual_split(
        counterfactual_root, computed_splits, descriptors, event_digest
    )
    views = build_split_views(
        data_root=data_root,
        output=args.output.resolve(),
        collector=collector,
        descriptors=descriptors,
        counterfactual_audit=counterfactual_audit,
    )
    commands = build_commands(args, views)
    return {
        "format": FORMAT,
        "status": "preflight_complete",
        "data": collector,
        "event_spec": {"path": str(event_spec), "sha256": event_digest},
        "factual_candidates": factual_rows,
        "selected_factual": selected,
        "counterfactual": {
            **counterfactual_complete,
            **{
                key: value
                for key, value in counterfactual_audit.items()
                if key != "group_file_contract"
            },
        },
        "split_views": views,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "run_count": len(commands),
        "commands": commands,
        "execution": "sequential_direct_three_seeds_then_latent_future_three_seeds",
        "runtime": {"device": "cuda", "expected_gpu": "4090"},
        "output": str(args.output.resolve()),
        "sealed_internal_15": {
            "labels_read": False,
            "hdf5_paths_passed_to_progress": False,
            "actor_oracle_evaluation": False,
        },
    }


def write_split_views(view_audit: Mapping[str, Any]) -> None:
    split = view_audit["split_manifest"]
    split_path = Path(str(split["path"]))
    atomic_json(split_path, split["payload"])
    if sha256(split_path) != str(split["sha256"]):
        raise RuntimeError("written progress split manifest SHA256 mismatch")
    for split_name in ("train", "validation"):
        view = view_audit["views"][split_name]
        path = Path(str(view["path"]))
        atomic_json(path, view["payload"])
        if sha256(path) != str(view["sha256"]):
            raise RuntimeError(f"written {split_name} view SHA256 mismatch")


def command_option(command: Mapping[str, Any], name: str) -> str:
    argv = command.get("argv")
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
        raise RuntimeError("progress command argv is invalid")
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError(f"progress command must contain one {name}")
    return str(argv[positions[0] + 1])


def summarize_completed_runs(
    output: Path,
    commands: Sequence[Mapping[str, Any]],
    split_manifest_sha256: str,
) -> dict[str, Any]:
    split_paths = {
        Path(command_option(command, "--split-manifest")).resolve()
        for command in commands
    }
    if len(split_paths) != 1:
        raise RuntimeError("progress commands do not share one frozen split manifest")
    split_path = next(iter(split_paths))
    if not split_path.is_file() or sha256(split_path) != split_manifest_sha256:
        raise RuntimeError("progress split manifest SHA256 changed or is missing")
    variants: dict[str, Any] = {}
    forbidden_train = {
        "baseline_success_rate",
        "selected_success_rate",
        "oracle_success_rate",
        "candidate_success_auc",
        "candidate_success_auc_scope",
        "within_group_success_pair_accuracy",
        "candidate_success_ndcg",
        "paired_success_difference",
        "paired_difference_ci95",
        "changed_groups",
        "improved_groups",
        "harmed_groups",
        "selected_candidate_ids",
    }
    for variant in VARIANTS:
        members = []
        for command in commands:
            if command["variant"] != variant:
                continue
            seed = int(command["seed"])
            run_output = Path(str(command["output"]))
            summary_path = run_output / f"progress_{variant}_summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"progress member summary is missing: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            contract = summary.get("contract")
            expected_event_spec = Path(command_option(command, "--event-spec")).resolve()
            expected_split = Path(command_option(command, "--split-manifest")).resolve()
            expected_train = Path(command_option(command, "--train-data")).resolve()
            expected_validation = Path(
                command_option(command, "--validation-data")
            ).resolve()
            expected_config = {
                "variant": variant,
                "hidden_dim": 4096,
                "action_dim": 14,
                "latent_dim": int(command_option(command, "--latent-dim")),
                "action_hidden_dim": int(
                    command_option(command, "--action-hidden-dim")
                ),
                "projection_seed": seed,
            }
            expected_optimization = {
                "steps": int(command_option(command, "--steps")),
                "batch_size": int(command_option(command, "--batch-size")),
                "learning_rate": float(
                    command_option(command, "--learning-rate")
                ),
                "latent_weight": float(command_option(command, "--latent-weight")),
                "evaluation_interval": int(
                    command_option(command, "--evaluation-interval")
                ),
            }
            if (
                summary.get("format") != "etsf_scalar_progress_baseline_v1"
                or summary.get("status") != "training_complete"
                or summary.get("variant") != variant
                or int(summary.get("training_seed", -1)) != seed
                or not isinstance(contract, Mapping)
                or int(contract.get("training_seed", -1)) != seed
                or contract.get("candidate_policy_diagnostics") != "validation_only"
                or contract.get("success_supervision")
                != "terminal_eK_progress_target_only"
                or contract.get("success_loss") is not False
                or contract.get("checkpoint_selection")
                != "validation_progress_mae_only"
                or contract.get("model_config") != expected_config
                or contract.get("optimization") != expected_optimization
                or Path(str(contract.get("event_spec", ""))).resolve()
                != expected_event_spec
                or str(contract.get("event_spec_sha256", ""))
                != sha256(expected_event_spec)
                or Path(str(contract.get("split_audit", {}).get("split_manifest", ""))).resolve()
                != expected_split
                or str(contract.get("split_audit", {}).get("split_manifest_sha256", ""))
                != split_manifest_sha256
            ):
                raise RuntimeError(f"progress member contract mismatch: {summary_path}")
            for root_name, expected_root in (
                ("train_roots", expected_train),
                ("validation_roots", expected_validation),
            ):
                roots = contract.get(root_name)
                if not isinstance(roots, list) or len(roots) != 1:
                    raise RuntimeError(f"progress member {root_name} contract is invalid")
                root_contract = roots[0]
                if (
                    not isinstance(root_contract, Mapping)
                    or Path(str(root_contract.get("root", ""))).resolve()
                    != expected_root
                    or str(root_contract.get("manifest_sha256", ""))
                    != sha256(expected_root / "manifest.json")
                ):
                    raise RuntimeError(
                        f"progress member {root_name} provenance mismatch"
                    )
            train_metrics = summary.get("train")
            validation = summary.get("validation")
            if not isinstance(train_metrics, Mapping) or not isinstance(validation, Mapping):
                raise RuntimeError(f"progress member metrics are missing: {summary_path}")
            if forbidden_train & set(train_metrics) or train_metrics.get(
                "policy_diagnostics_included"
            ) is not False:
                raise RuntimeError("actor/oracle diagnostics leaked into train reporting")
            if (
                validation.get("policy_diagnostics_included") is not True
                or "baseline_success_rate" not in validation
                or "oracle_success_rate" not in validation
                or validation.get("candidate_success_auc_scope")
                != "first_query_candidates_only"
            ):
                raise RuntimeError("validation lacks frozen actor/oracle diagnostics")
            checkpoint = Path(str(summary.get("checkpoint", "")))
            if not checkpoint.is_file():
                portable = run_output / checkpoint.name
                checkpoint = portable if portable.is_file() else checkpoint
            digest = sha256(checkpoint)
            if digest != str(summary.get("checkpoint_sha256", "")):
                raise RuntimeError(f"progress member checkpoint SHA256 mismatch: {checkpoint}")
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if (
                not isinstance(checkpoint_payload, Mapping)
                or checkpoint_payload.get("format")
                != "etsf_scalar_progress_baseline_v1"
                or checkpoint_payload.get("config") != expected_config
                or summary.get("config") != expected_config
                or checkpoint_payload.get("contract") != contract
                or not isinstance(checkpoint_payload.get("model"), Mapping)
            ):
                raise RuntimeError(
                    f"progress checkpoint/summary mirror mismatch: {checkpoint}"
                )
            training = summary.get("training")
            selection = training.get("selection") if isinstance(training, Mapping) else None
            best_validation = (
                training.get("best_validation")
                if isinstance(training, Mapping)
                else None
            )
            history = training.get("history") if isinstance(training, Mapping) else None
            best_step = int(selection.get("best_step", -1)) if isinstance(selection, Mapping) else -1
            history_rows = (
                [row for row in history if isinstance(row, Mapping)]
                if isinstance(history, list)
                else []
            )
            selected_history = [
                row for row in history_rows if int(row.get("step", -1)) == best_step
            ]
            history_mae = [
                float(row["validation"]["progress_mae"])
                for row in history_rows
                if isinstance(row.get("validation"), Mapping)
                and row["validation"].get("progress_mae") is not None
            ]
            if (
                not isinstance(selection, Mapping)
                or selection.get("data") != "validation_only"
                or selection.get("metric") != "progress_mae"
                or selection.get("mode") != "min"
                or not isinstance(best_validation, Mapping)
                or len(selected_history) != 1
                or len(history_mae) != len(history_rows)
                or not history_mae
                or not math.isclose(
                    float(selection.get("best_value", math.inf)),
                    float(best_validation.get("progress_mae", -math.inf)),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(selection.get("best_value", math.inf)),
                    min(history_mae),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(validation.get("progress_mae", math.inf)),
                    float(best_validation.get("progress_mae", -math.inf)),
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                )
            ):
                raise RuntimeError(
                    f"progress validation-only checkpoint selection is invalid: {summary_path}"
                )
            members.append(
                {
                    "seed": seed,
                    "summary": str(summary_path.resolve()),
                    "summary_sha256": sha256(summary_path),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": digest,
                    "validation": dict(validation),
                }
            )
        if len(members) != len(SEEDS):
            raise RuntimeError(f"variant {variant} does not contain three seeds")
        aggregate = {}
        for key in (
            "progress_mae",
            "progress_rmse",
            "baseline_success_rate",
            "selected_success_rate",
            "oracle_success_rate",
            "paired_success_difference",
            "candidate_success_auc",
            "within_group_success_pair_accuracy",
            "future_latent_cosine",
        ):
            values = [member["validation"].get(key) for member in members]
            numeric = [float(value) for value in values if value is not None]
            if numeric:
                mean = sum(numeric) / len(numeric)
                variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
                aggregate[key] = {
                    "members": len(numeric),
                    "mean": mean,
                    "population_std": variance**0.5,
                }
        variants[variant] = {
            "seed_count": len(members),
            "members": members,
            "validation_aggregate": aggregate,
        }
    return {
        "format": SUITE_FORMAT,
        "status": "complete",
        "variants": variants,
        "seed_count_per_variant": len(SEEDS),
        "actor_oracle_scope": "validation_only",
        "sealed_internal_15_evaluated": False,
        "split_manifest_sha256": split_manifest_sha256,
        "command_contracts": [
            {
                "variant": str(command["variant"]),
                "seed": int(command["seed"]),
                "argv_sha256": str(command["argv_sha256"]),
            }
            for command in commands
        ],
    }


def validate_complete_output(
    output: Path,
    commands: Sequence[Mapping[str, Any]],
    split_manifest_sha256: str,
) -> dict[str, Any] | None:
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        return None
    audit_path = output / "launch_audit.json"
    suite_path = output / "progress_baseline_suite_summary.json"
    if not audit_path.is_file() or not suite_path.is_file():
        raise RuntimeError(
            "progress output is partial/conflicting and safe resume is unavailable"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    expected_command_contracts = [
        {
            "variant": str(command["variant"]),
            "seed": int(command["seed"]),
            "argv_sha256": str(command["argv_sha256"]),
        }
        for command in commands
    ]
    audit_commands = audit.get("commands")
    audit_command_contracts = (
        [
            {
                "variant": str(command.get("variant", "")),
                "seed": int(command.get("seed", -1)),
                "argv_sha256": str(command.get("argv_sha256", "")),
            }
            for command in audit_commands
            if isinstance(command, Mapping)
        ]
        if isinstance(audit_commands, list)
        else []
    )
    audit_split_views = audit.get("split_views")
    audit_split_record = (
        audit_split_views.get("split_manifest")
        if isinstance(audit_split_views, Mapping)
        else None
    )
    if (
        audit.get("format") != FORMAT
        or audit.get("status") != "launcher_complete"
        or audit_command_contracts != expected_command_contracts
        or not isinstance(audit_split_record, Mapping)
        or str(audit_split_record.get("sha256", "")) != split_manifest_sha256
        or audit.get("sealed_internal_15")
        != {
            "labels_read": False,
            "hdf5_paths_passed_to_progress": False,
            "actor_oracle_evaluation": False,
        }
    ):
        raise RuntimeError("progress launch audit is incomplete; refusing resume")
    audit_views = audit_split_views.get("views")
    if not isinstance(audit_views, Mapping):
        raise RuntimeError("progress launch audit lacks frozen split views")
    for split_name, option_name in (
        ("train", "--train-data"),
        ("validation", "--validation-data"),
    ):
        expected_roots = {
            Path(command_option(command, option_name)).resolve()
            for command in commands
        }
        record = audit_views.get(split_name)
        if len(expected_roots) != 1 or not isinstance(record, Mapping):
            raise RuntimeError("progress launch audit split-view contract is invalid")
        manifest_path = Path(str(record.get("path", ""))).resolve()
        if (
            manifest_path.parent != next(iter(expected_roots))
            or not manifest_path.is_file()
            or sha256(manifest_path) != str(record.get("sha256", ""))
        ):
            raise RuntimeError(
                f"progress {split_name} split-view manifest SHA256 mismatch"
            )
    if (
        suite.get("format") != SUITE_FORMAT
        or suite.get("status") != "complete"
        or suite.get("split_manifest_sha256") != split_manifest_sha256
    ):
        raise RuntimeError("existing progress suite has a different frozen contract")
    reconstructed = summarize_completed_runs(output, commands, split_manifest_sha256)
    if suite != reconstructed:
        raise RuntimeError("existing progress suite differs from member summaries/checkpoints")
    return {
        "status": "already_complete_skip",
        "suite": str(suite_path.resolve()),
        "suite_sha256": sha256(suite_path),
        "runs": len(commands),
    }


def parse_args() -> argparse.Namespace:
    local_trainer = Path(__file__).resolve().parent / "train_openvla_etsf_progress_baseline.py"
    default_python = DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable)
    remote_trainer = DEFAULT_CODE_ROOT / "scripts/train_openvla_etsf_progress_baseline.py"
    default_trainer = remote_trainer if remote_trainer.is_file() else local_trainer
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--factual-root", type=Path, default=DEFAULT_FACTUAL_ROOT)
    parser.add_argument(
        "--counterfactual-root", type=Path, default=DEFAULT_COUNTERFACTUAL_OUTPUT
    )
    parser.add_argument("--event-spec", type=Path, default=DEFAULT_EVENT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trainer", type=Path, default=default_trainer)
    parser.add_argument("--python-bin", type=Path, default=default_python)
    parser.add_argument("--wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.wait_timeout_seconds < 0
        or args.gpu_wait_timeout_seconds < 0
        or not 0 < args.poll_seconds <= 60
        or not 0 < args.gpu_poll_seconds <= 60
    ):
        raise ValueError(
            "wait timeouts must be non-negative and poll intervals in (0,60]"
        )
    audit = preflight(args)
    complete = validate_complete_output(
        args.output.resolve(),
        audit["commands"],
        audit["split_views"]["split_manifest"]["sha256"],
    )
    if complete is not None:
        print("PROGRESS_BASELINE_SUITE_SKIP=" + json.dumps(complete, sort_keys=True))
        return
    if args.dry_run:
        print("PROGRESS_BASELINE_V5_DRY_RUN=" + json.dumps(audit, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("formal progress launch requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090" not in gpu_name:
        raise RuntimeError(f"formal progress launch requires RTX 4090, found {gpu_name!r}")
    # Avoid creating a non-resumable output merely because the preceding
    # counterfactual process has not released its CUDA context yet.
    initial_idle = wait_for_gpu_idle(
        args.gpu_wait_timeout_seconds,
        args.gpu_poll_seconds,
        gpu_index=0,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "launch.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("progress launch lock exists; concurrent/resume launch refused") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload_sha256(audit) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        write_split_views(audit["split_views"])
        audit["status"] = "launching_nonresumable"
        audit["runtime"]["gpu_name"] = gpu_name
        audit["runtime"]["gpu_idle_checks"] = [
            {**initial_idle, "variant": VARIANTS[0], "seed": SEEDS[0]}
        ]
        atomic_json(output / "launch_audit.json", audit)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "8",
            }
        )
        for index, run in enumerate(audit["commands"]):
            if index:
                idle = wait_for_gpu_idle(
                    args.gpu_wait_timeout_seconds,
                    args.gpu_poll_seconds,
                    gpu_index=0,
                )
                idle.update({"variant": run["variant"], "seed": run["seed"]})
                audit["runtime"]["gpu_idle_checks"].append(idle)
            atomic_json(output / "launch_audit.json", audit)
            subprocess.run(run["argv"], check=True, env=environment)
        suite = summarize_completed_runs(
            output,
            audit["commands"],
            audit["split_views"]["split_manifest"]["sha256"],
        )
        atomic_json(output / "progress_baseline_suite_summary.json", suite)
        audit["status"] = "launcher_complete"
        atomic_json(output / "launch_audit.json", audit)
        complete = validate_complete_output(
            output,
            audit["commands"],
            audit["split_views"]["split_manifest"]["sha256"],
        )
        assert complete is not None
        print("PROGRESS_BASELINE_SUITE_COMPLETE=" + json.dumps(complete, sort_keys=True))
    except BaseException:
        audit["status"] = "failed_nonresumable_manual_new_output_required"
        atomic_json(output / "launch_audit.json", audit)
        raise


if __name__ == "__main__":
    main()
