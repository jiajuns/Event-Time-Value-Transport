#!/usr/bin/env python3
"""Preregister and evaluate the single-look prospective v7 development set."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from openvla_etsf_v7_development_confirmation import (
    DEPLOYMENT_CANDIDATE_NAMES, EXPECTED_GROUPS, RESULT_FORMAT, TASK,
    canonical_sha256, directory_sha256, evaluate_fixed_policy,
    make_preregistration, sha256, validate_preregistration, validate_seed_manifest,
)
from openvla_etsf_event_world_model import ActionConditionedEventWorldModel
from openvla_etsf_structured_event_time_utility import (
    structured_event_time_utility_numpy,
)
from train_openvla_etsf_counterfactual import (
    atomic_json, atomic_torch_save, canonical_policy_mapping, collate_groups,
    forward_model, load_descriptor_groups, load_pretrained, move_batch,
    scan_group_descriptors,
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise RuntimeError(f"expected JSON object: {path}")
    return value


def _refuse(path: Path) -> None:
    if path.exists(): raise FileExistsError(f"v7 refuses overwrite: {path}")


def _task_calibration(event_spec: Path) -> tuple[Mapping[str, Any], str]:
    value = _json(event_spec)
    calibrations = value.get("calibration")
    if not isinstance(calibrations, Mapping) or not isinstance(calibrations.get(TASK), Mapping):
        raise RuntimeError("v7 event spec lacks explicit move_can_pot calibration")
    calibration = dict(calibrations[TASK])
    return calibrations, canonical_sha256(calibration)


def _implementation_files() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        here / "openvla_etsf_v7_development_confirmation.py",
        here / "preregister_robotwin_v7_development_confirmation.py",
        here / "launch_openvla_etsf_v7_development_confirmation.py",
        here / "collect_openvla_etsf_event_branches.py",
        here / "openvla_etsf_event_world_model.py",
        here / "openvla_etsf_structured_event_time_utility.py",
        here / "train_openvla_etsf_counterfactual.py",
    ]


def preregister(args: argparse.Namespace) -> None:
    _refuse(args.output)
    seed_path, pretrained_path, event_path, actor_path = (
        args.seed_manifest.resolve(), args.pretrained.resolve(), args.event_spec.resolve(),
        args.actor_model_path.resolve(),
    )
    for path in (seed_path, pretrained_path, event_path, *_implementation_files()):
        if not path.is_file(): raise FileNotFoundError(path)
    seeds = _json(seed_path); validate_seed_manifest(seeds, verify_files=True)
    checkpoint, config = load_pretrained(pretrained_path)
    if not config.structured_events or config.num_events != 5:
        raise RuntimeError("v7 requires the frozen five-event factual world model")
    if config.action_rank_residual or config.action_rank_success_only:
        raise RuntimeError("v7 forbids action-rank heads in its factual checkpoint")
    checkpoint_contract = checkpoint.get("contract")
    if not isinstance(checkpoint_contract, Mapping) or checkpoint_contract.get(
        "event_spec_sha256"
    ) != sha256(event_path):
        raise RuntimeError("v7 factual/event-spec contract mismatch")
    _, calibration_sha = _task_calibration(event_path)
    source = {
        "seed_manifest": str(seed_path), "seed_manifest_file_sha256": sha256(seed_path),
        "pretrained": str(pretrained_path), "pretrained_sha256": sha256(pretrained_path),
        "event_spec": str(event_path), "event_spec_sha256": sha256(event_path),
        "actor_model": {"path": str(actor_path), "tree_sha256": directory_sha256(actor_path)},
        "implementation_files": [
            {"path": str(path.resolve()), "sha256": sha256(path.resolve())}
            for path in _implementation_files()
        ],
        "labels_collected_before_preregistration": False,
    }
    value = make_preregistration(seed_manifest=seeds, source_contract=source,
                                 task_calibration_sha256=calibration_sha)
    validate_preregistration(value)
    args.output.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.output, value)


def _validate_frozen_sources(prereg: Mapping[str, Any], args: argparse.Namespace):
    validate_preregistration(prereg)
    source = prereg.get("source_contract")
    if not isinstance(source, Mapping): raise RuntimeError("v7 source contract missing")
    fixed = {
        args.seed_manifest.resolve(): source.get("seed_manifest_file_sha256"),
        args.pretrained.resolve(): source.get("pretrained_sha256"),
        args.event_spec.resolve(): source.get("event_spec_sha256"),
    }
    for path, digest in fixed.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"v7 frozen source changed: {path}")
    for row in source.get("implementation_files", []):
        path = Path(str(row.get("path", ""))).resolve()
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"v7 implementation changed after preregistration: {path}")
    actor = source.get("actor_model")
    if not isinstance(actor, Mapping) or Path(str(actor.get("path", ""))).resolve() != args.actor_model_path.resolve():
        raise RuntimeError("v7 actor model path changed")
    if directory_sha256(args.actor_model_path.resolve()) != actor.get("tree_sha256"):
        raise RuntimeError("v7 actor model content changed")
    seeds = _json(args.seed_manifest.resolve()); validate_seed_manifest(seeds, verify_files=True)
    if seeds["seed_manifest_payload_sha256"] != prereg["seed_manifest_payload_sha256"]:
        raise RuntimeError("v7 seed manifest differs from preregistration")
    calibrations, calibration_sha = _task_calibration(args.event_spec.resolve())
    if calibration_sha != prereg["event_value_registry"]["task_calibration_sha256"]:
        raise RuntimeError("v7 task calibration changed")
    return seeds, calibrations


def _collection(args: argparse.Namespace, seeds: Mapping[str, Any]):
    root = args.data.resolve(); manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    if "fresh" in root.name.lower() or any(
        manifest.get(key) not in (None, "") for key in (
            "fresh_seed_manifest", "fresh_seed_manifest_sha256")
    ):
        raise RuntimeError("fresh confirmation data are forbidden in v7")
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("schema_version", -1)) != 5
        or int(manifest.get("completed", -1)) != EXPECTED_GROUPS
        or int(manifest.get("candidate_count", -1)) != 4
        or manifest.get("candidate_names") not in (None, list(DEPLOYMENT_CANDIDATE_NAMES))
        or manifest.get("seed_registry") != "explicit_v7_prospective_development"
        or manifest.get("v7_seed_manifest_sha256") != sha256(args.seed_manifest.resolve())
        or manifest.get("v7_preregistration_sha256") != args.preregistration_sha256
        or Path(str(manifest.get("model_path", ""))).resolve() != args.actor_model_path.resolve()
        or manifest.get("requested_seeds") != seeds["requested_seeds"]
        or manifest.get("resolved_seeds") != seeds["resolved_seeds"]
    ):
        raise RuntimeError("v7 collection provenance/completion contract mismatch")
    descriptors = scan_group_descriptors([root])
    if len(descriptors) != EXPECTED_GROUPS or any(d.schema_version != 5 for d in descriptors):
        raise RuntimeError("v7 collection lacks 250 schema-v5 groups")
    return root, manifest_path, descriptors


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    _refuse(args.output)
    prereg = _json(args.preregistration.resolve())
    args.preregistration_sha256 = prereg["preregistration_sha256"]
    seeds, calibrations = _validate_frozen_sources(prereg, args)
    root, collection_manifest, descriptors = _collection(args, seeds)
    checkpoint, config = load_pretrained(args.pretrained.resolve())
    contract = checkpoint["contract"]
    objects = contract.get("object_names"); bodies = contract.get("body_to_id")
    policies = canonical_policy_mapping(contract.get("policy_to_id"))
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not isinstance(bodies, Mapping):
        raise RuntimeError("v7 factual context incomplete")
    norm = checkpoint.get("normalization")
    if not isinstance(norm, Mapping): raise RuntimeError("v7 factual normalization missing")
    mean = np.asarray(norm["object_delta_mean"], dtype=np.float32)
    std = np.asarray(norm["object_delta_std"], dtype=np.float32)
    groups = load_descriptor_groups(
        descriptors, config, list(map(str, objects)), bodies, policies,
        calibrations=calibrations, expected_event_spec_sha256=sha256(args.event_spec.resolve())
    )
    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(checkpoint["model"], strict=True); model.float().cpu().eval()
    rows = []
    for group in groups:
        if tuple(group.candidate_names) != DEPLOYMENT_CANDIDATE_NAMES:
            raise RuntimeError("v7 group candidate names/order changed")
        batch = move_batch(collate_groups([group], object_mean=mean, object_std=std), torch.device("cpu"))
        output = forward_model(model, batch)
        decomposition = structured_event_time_utility_numpy(
            output["next_reached_event_logits"][:4].float().cpu().numpy(),
            output["next_event_logits"][:4].float().cpu().numpy(),
            output["duration_selected_log_mean"][:4].float().cpu().numpy(),
            event_values=prereg["event_value_registry"]["values"],
        )
        # Formula and checkpoint are frozen; labels are consulted only for the
        # single registered paired statistic below, never for score selection.
        rows.append({
            "logical_key": group.logical_key,
            "utility": np.asarray(decomposition["utility"]),
            "destination_expected_progress": np.asarray(
                decomposition["destination_expected_progress"]
            ),
            "immediate_next_event_expected_progress": np.asarray(
                decomposition["immediate_next_event_expected_progress"]
            ),
            "duration_selected_log_mean": np.asarray(
                decomposition["duration_selected_log_mean"]
            ),
            "destination_z": np.asarray(decomposition["destination_z"]),
            "immediate_next_event_z": np.asarray(
                decomposition["immediate_next_event_z"]
            ),
            "duration_z": np.asarray(decomposition["duration_z"]),
            "success": np.asarray(group.success[:4], dtype=np.float32),
        })
    metrics = evaluate_fixed_policy(rows)
    raw_path = args.output.with_name("v7_fixed_predictions.pt")
    _refuse(raw_path)
    atomic_torch_save(raw_path, {"format": RESULT_FORMAT, "rows": rows,
        "preregistration_sha256": prereg["preregistration_sha256"]})
    result = {
        "format": RESULT_FORMAT, "status": "complete_development_only",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "collection_manifest": str(collection_manifest),
        "collection_manifest_sha256": sha256(collection_manifest),
        "predictions": str(raw_path.resolve()), "predictions_sha256": sha256(raw_path),
        "metrics": metrics,
        "authorization": {
            "fresh50_confirmation_authorized": bool(metrics["development_gate_pass"]),
            "authorization_basis": "single_preregistered_v7_gate",
            "automatic_fresh_launch": False,
        },
        "fresh_confirmation_labels_read": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.output, result)
    authorization_path = args.output.with_name("v7_fresh50_authorization.json")
    _refuse(authorization_path)
    token = {
        "format": "etsf_v7_signed_fresh50_authorization_v1",
        "status": "authorized" if metrics["development_gate_pass"] else "denied",
        "fresh50_confirmation_authorized": bool(metrics["development_gate_pass"]),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "result": str(args.output.resolve()), "result_file_sha256": sha256(args.output),
        "result_payload_sha256": result["result_sha256"],
        "collection_manifest_sha256": sha256(collection_manifest),
        "predictions_sha256": sha256(raw_path),
        "gate": {key: metrics[key] for key in (
            "development_gate_pass", "changed_groups", "helpful_changes", "harmful_changes",
            "harmful_rate_over_all_changes", "unconditional_mean_success_delta",
            "unconditional_bootstrap_95_ci", "exact_two_sided_sign_test_p")},
        "v7_fresh_labels_read": False, "automatic_fresh_launch": False,
    }
    token["authorization_sha256"] = canonical_sha256(token)
    atomic_json(authorization_path, token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed-manifest", type=Path, required=True)
    common.add_argument("--pretrained", type=Path, required=True)
    common.add_argument("--event-spec", type=Path, required=True)
    common.add_argument("--actor-model-path", type=Path, required=True)
    pre = sub.add_parser("preregister", parents=[common]); pre.add_argument("--output", type=Path, required=True)
    ev = sub.add_parser("evaluate", parents=[common]); ev.add_argument("--preregistration", type=Path, required=True)
    ev.add_argument("--data", type=Path, required=True); ev.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "preregister": preregister(args)
    elif args.stage == "evaluate": evaluate(args)
    else: raise AssertionError(args.stage)


if __name__ == "__main__": main()
