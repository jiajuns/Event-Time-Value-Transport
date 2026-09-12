#!/usr/bin/env python3
"""Hard data gate before universal-event pretraining."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from download_official import BODIES, UNAVAILABLE_PAIRS
from universal_event.schema import (
    CANONICAL_ACTION_DIM, EDGE_FEATURE_DIM, EVENT_NAMES, MAX_NODES,
    NODE_FEATURE_DIM, RELATION_NAMES, TASK_SPECS, schema_sha256,
)


FORMAT = "etsf_robotwin2_universal_event_dataset_audit_v1"
EPISODE_FORMAT = "etsf_robotwin2_universal_event_episode_v1"


def audit_file(path: Path) -> dict:
    with h5py.File(path) as handle:
        if str(handle.attrs.get("format", "")) != EPISODE_FORMAT:
            raise ValueError("wrong episode format")
        if str(handle.attrs.get("schema_sha256", "")) != schema_sha256():
            raise ValueError("schema mismatch")
        task = str(handle.attrs["task"]); body = str(handle.attrs["body"])
        condition = str(handle.attrs["condition"]); seed = int(handle.attrs["seed"])
        expected_tail = {
            "node_features": (MAX_NODES, NODE_FEATURE_DIM),
            "node_types": (MAX_NODES,), "node_mask": (MAX_NODES,),
            "edge_features": (MAX_NODES, MAX_NODES, EDGE_FEATURE_DIM),
            "relations": (len(RELATION_NAMES), MAX_NODES, MAX_NODES),
            "relation_mask": (len(RELATION_NAMES), MAX_NODES, MAX_NODES),
            "events": (len(EVENT_NAMES),), "ee_state16": (16,), "success": (),
            "direct_contact2": (2,),
        }
        length = int(handle["node_features"].shape[0])
        if length < 2:
            raise ValueError("short episode")
        for key, tail in expected_tail.items():
            if handle[key].shape != (length, *tail):
                raise ValueError(f"bad shape for {key}: {handle[key].shape}")
        if handle["actions14"].shape != (length - 1, CANONICAL_ACTION_DIM):
            raise ValueError("bad actions14 shape")
        if handle["sim_times"].shape != (length,):
            raise ValueError("bad sim_times shape")
        if handle["goal"].shape != (len(RELATION_NAMES), MAX_NODES, MAX_NODES):
            raise ValueError("bad goal shape")
        if handle["goal_mask"].shape != handle["goal"].shape:
            raise ValueError("bad goal_mask shape")
        if any(key.lower() in {"state25", "joint_state", "qpos"} for key in handle.keys()):
            raise ValueError("embodiment-specific state leaked into shared dataset")
        times = np.asarray(handle["sim_times"])
        if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
            raise ValueError("non-monotonic physical time")
        for key in ("node_features", "edge_features", "relations", "events", "actions14"):
            if not np.isfinite(handle[key][...]).all():
                raise ValueError(f"non-finite values in {key}")
        relation_positive = np.asarray(handle["relations"]).astype(bool).sum((0, 2, 3))
        relation_known = np.asarray(handle["relation_mask"]).astype(bool).sum((0, 2, 3))
        event_positive = np.asarray(handle["events"]).astype(bool).sum(0)
        return {
            "task": task, "body": body, "condition": condition, "seed": seed,
            "frames": length, "success": bool(handle.attrs.get("native_success", False)),
            "planner_error": bool(str(handle.attrs.get("planner_error", ""))),
            "relation_positive": relation_positive.tolist(),
            "relation_known": relation_known.tolist(),
            "event_positive": event_positive.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-clean-per-pair", type=int, default=40)
    parser.add_argument("--require-full-matrix", action="store_true")
    args = parser.parse_args()
    files = sorted(args.data.resolve().rglob("*.hdf5"))
    rows, invalid = [], []
    for path in files:
        try:
            rows.append(audit_file(path))
        except Exception as exc:
            invalid.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    clean_pairs = Counter((row["task"], row["body"]) for row in rows if row["condition"] == "clean")
    expected_pairs = [(task, body) for task in TASK_SPECS for body in BODIES
                      if (task, body) not in UNAVAILABLE_PAIRS]
    insufficient = [
        {"task": task, "body": body, "count": clean_pairs[(task, body)]}
        for task, body in expected_pairs
        if clean_pairs[(task, body)] < args.minimum_clean_per_pair
    ]
    relation_positive = np.asarray([row["relation_positive"] for row in rows], dtype=np.int64).sum(0) if rows else np.zeros(len(RELATION_NAMES), np.int64)
    relation_known = np.asarray([row["relation_known"] for row in rows], dtype=np.int64).sum(0) if rows else np.zeros(len(RELATION_NAMES), np.int64)
    event_positive = np.asarray([row["event_positive"] for row in rows], dtype=np.int64).sum(0) if rows else np.zeros(len(EVENT_NAMES), np.int64)
    per_task = defaultdict(lambda: {"episodes": 0, "success": 0, "bodies": set()})
    for row in rows:
        item = per_task[row["task"]]; item["episodes"] += 1
        item["success"] += int(row["success"]); item["bodies"].add(row["body"])
    receipt = {
        "format": FORMAT, "schema_sha256": schema_sha256(), "root": str(args.data.resolve()),
        "file_count": len(files), "valid_episodes": len(rows), "invalid": invalid,
        "expected_clean_pairs": len(expected_pairs),
        "minimum_clean_per_pair": args.minimum_clean_per_pair,
        "insufficient_clean_pairs": insufficient,
        "per_task": {task: {**value, "bodies": sorted(value["bodies"])}
                     for task, value in sorted(per_task.items())},
        "relation_positive": dict(zip(RELATION_NAMES, relation_positive.tolist())),
        "relation_known": dict(zip(RELATION_NAMES, relation_known.tolist())),
        "event_positive": dict(zip(EVENT_NAMES, event_positive.tolist())),
        "has_success": any(row["success"] for row in rows),
        "has_failure": any(not row["success"] for row in rows),
        "has_drop": bool(event_positive[EVENT_NAMES.index("drop")]),
        "has_regrasp": bool(event_positive[EVENT_NAMES.index("regrasp")]),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    hard_errors = bool(invalid) or not rows
    if args.require_full_matrix:
        hard_errors = hard_errors or bool(insufficient)
    if hard_errors:
        raise RuntimeError(
            f"dataset audit failed: {len(invalid)} invalid, {len(insufficient)} insufficient pairs"
        )
    print(json.dumps({"receipt": str(args.output.resolve()), "valid": len(rows),
                      "insufficient_pairs": len(insufficient),
                      "has_drop": receipt["has_drop"], "has_regrasp": receipt["has_regrasp"]}))


if __name__ == "__main__":
    main()
