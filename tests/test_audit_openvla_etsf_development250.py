from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_openvla_etsf_development250 import (  # noqa: E402
    EXPECTED_NAMES,
    audit_development250,
    sha256,
)
from preregister_robotwin_development_expansion_seeds import build_manifest  # noqa: E402
from robotwin_development_seed_contract import (  # noqa: E402
    CANDIDATE_FORMAT,
    CANDIDATE_STATUS,
    PURPOSE,
    REGISTRY,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _group(path: Path, requested: int, resolved: int, names: tuple[str, ...], positive: bool) -> None:
    count = len(names)
    steps = np.ones(count, dtype=np.int32)
    success = np.zeros(count, dtype=bool)
    if positive:
        success[-1] = True
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 5,
                "seed": requested,
                "requested_seed": requested,
                "resolved_seed": resolved,
                "candidate_count": count,
                "language_contract": "same_instruction_for_initial_query_and_all_candidate_branches",
                "branch_instruction_consistent": True,
                "intervention": "candidate_first_chunk_then_deterministic_actor",
                "post_query_action_contract": "executed_as_next_query_when_nonterminal",
            }
        )
        handle.create_dataset("candidate_names", data=[name.encode() for name in names])
        handle.create_dataset("success", data=success)
        handle.create_dataset("steps", data=steps)
        handle.create_dataset("duration_observed", data=np.ones(count, dtype=bool))
        handle.create_dataset("next_event_duration_observed", data=np.ones(count, dtype=bool))
        handle.create_dataset("pre_event_id", data=np.zeros(count, dtype=np.int64))
        handle.create_dataset("post_event_id", data=np.ones(count, dtype=np.int64))
        handle.create_dataset("next_event_id", data=np.full(count, 2, dtype=np.int64))
        handle.create_dataset("first_chunk_executed_length", data=np.ones(count, dtype=np.int32))
        handle.create_dataset("first_chunk_action_mask", shape=(count, 25), dtype=bool)
        handle.create_dataset("candidate_actions", shape=(count, 25, 14), dtype=np.float32)
        branches = handle.create_group("branches")
        for index in range(count):
            branch = branches.create_group(f"candidate_{index:03d}")
            branch.create_dataset("query_steps", data=np.asarray([0], dtype=np.int32))
            branch.create_dataset("query_post_steps", data=np.asarray([1], dtype=np.int32))
            branch.create_dataset("query_hidden", shape=(1, 4096), dtype=np.float16)
            branch.create_dataset("query_post_hidden", shape=(1, 4096), dtype=np.float16)
            branch.create_dataset("query_actions", shape=(1, 25, 14), dtype=np.float32)
            branch.create_dataset("query_action_mask", shape=(1, 25), dtype=bool)
            branch.create_dataset("object_poses", shape=(2, 1, 7), dtype=np.float32)
            branch.create_dataset("proprio", shape=(2, 14), dtype=np.float32)
            events = [b"e0", b"eK"] if success[index] else [b"e0"]
            branch.create_dataset("event_names", data=events)


def _collection(
    root: Path,
    seeds: list[int],
    names: tuple[str, ...],
    event_digest: str,
    *,
    development_manifest: Path | None,
) -> None:
    groups = root / "groups"
    groups.mkdir(parents=True)
    rows = []
    for index, seed in enumerate(seeds):
        path = groups / f"group_{index:03d}.hdf5"
        _group(path, seed, seed, names, positive=index % 10 == 0)
        rows.append(
            {
                "index": index,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": seed,
                "path": path.name,
                "status": "collected",
            }
        )
    development = development_manifest is not None
    manifest = {
        "status": "complete",
        "schema_version": 5,
        "completed": len(seeds),
        "candidate_count": len(names),
        "requested_seeds": seeds,
        "resolved_seeds": seeds,
        "event_spec_sha256": event_digest,
        "seed_registry": REGISTRY if development else None,
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
        "development_seed_manifest": (
            str(development_manifest.resolve()) if development else None
        ),
        "development_seed_manifest_sha256": (
            sha256(development_manifest) if development else None
        ),
        "blends": [0.25, 0.5, 0.75, 1.0] if development else [0.25, 0.5, 0.75],
        "preserve_grippers": True,
        "groups": rows,
    }
    _json(root / "manifest.json", manifest)


def _inputs(tmp_path: Path) -> argparse.Namespace:
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {})
    digest = sha256(event_spec)
    candidate_path = tmp_path / "candidate.json"
    official_path = tmp_path / "official.json"
    fresh_path = tmp_path / "fresh.json"
    candidate = {
        "format": CANDIDATE_FORMAT,
        "status": CANDIDATE_STATUS,
        "task": "move_can_pot",
        "purpose": PURPOSE,
        "candidate_seed_range": {"start": 100100276, "count": 200, "step": 1},
        "selection_rule": "first150",
        "freeze_rule": "frozen",
    }
    _json(candidate_path, candidate)
    _json(official_path, {"move_can_pot": {"success_seeds": list(range(150))}})
    _json(
        fresh_path,
        {
            "status": "fresh_confirmation_preregistered_resolved",
            "task": "move_can_pot",
            "requested_seeds": list(range(100100196, 100100246)),
            "resolved_seeds": list(range(200100196, 200100246)),
        },
    )
    development_seeds = list(range(100100276, 100100426))
    selected = [
        {"seed": seed, "requested_seed": seed, "resolved_seed": seed}
        for seed in development_seeds
    ]
    audit = [
        {"requested_seed": seed, "resolved_seed": seed, "decision": "selected"}
        for seed in development_seeds
    ]
    development_seed_manifest = tmp_path / "development_seeds.json"
    _json(
        development_seed_manifest,
        build_manifest(
            task="move_can_pot",
            selected=selected,
            audit=audit,
            candidate_path=candidate_path,
            official_path=official_path,
            fresh_path=fresh_path,
            candidate=candidate,
        ),
    )
    old = tmp_path / "old100"
    development = tmp_path / "development150"
    _collection(
        old,
        list(range(5000, 5100)),
        EXPECTED_NAMES["old100"],
        digest,
        development_manifest=None,
    )
    _collection(
        development,
        development_seeds,
        EXPECTED_NAMES["development150"],
        digest,
        development_manifest=development_seed_manifest,
    )
    return argparse.Namespace(
        old_data=old,
        development_data=development,
        development_seed_manifest=development_seed_manifest,
        fresh_seed_manifest=fresh_path,
        event_spec=event_spec,
        task="move_can_pot",
    )


def test_training_ready_audit_reports_250_groups_1150_branches_and_density(
    tmp_path: Path,
) -> None:
    result = audit_development250(_inputs(tmp_path))
    assert result["status"] == "training_ready"
    assert result["combined"]["groups"] == 250
    assert result["combined"]["candidate_branches"] == 1150
    assert result["development150"]["candidate_count"] == 5
    assert result["development150"]["seed_registry_audit"] == REGISTRY
    assert result["overlap_audit"] == {
        "old100_vs_development150_resolved": [],
        "development150_vs_fresh50_resolved": [],
        "development150_resolved_vs_fresh50_requested": [],
        "development150_requested_vs_fresh50_resolved": [],
    }
    assert result["combined"]["label_density"]["success_positives"] == 25
    assert result["combined"]["label_density"]["query_transitions"] == 1150


def test_new_registry_fresh_binding_and_resolved_overlap_fail_closed(
    tmp_path: Path,
) -> None:
    args = _inputs(tmp_path)
    development_manifest_path = args.development_data / "manifest.json"
    manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    manifest["fresh_seed_manifest"] = str(args.fresh_seed_manifest)
    _json(development_manifest_path, manifest)
    with pytest.raises(RuntimeError, match="registry/fresh/candidate"):
        audit_development250(args)

    args = _inputs(tmp_path / "overlap")
    old_manifest_path = args.old_data / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    new_seed = json.loads(
        (args.development_data / "manifest.json").read_text(encoding="utf-8")
    )["resolved_seeds"][0]
    old_manifest["resolved_seeds"][0] = new_seed
    old_manifest["groups"][0]["resolved_seed"] = new_seed
    _json(old_manifest_path, old_manifest)
    first_group = args.old_data / "groups" / "group_000.hdf5"
    with h5py.File(first_group, "r+") as handle:
        handle.attrs["resolved_seed"] = new_seed
    with pytest.raises(RuntimeError, match="seed leakage"):
        audit_development250(args)
