#!/usr/bin/env python3
"""Convert existing move_can_pot success/failure simulator rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import collect_robotwin as collector
from universal_event.schema import MAX_NODES, NODE_FEATURE_DIM, NODE_TYPES, RELATION_NAMES, TASK_SPECS, schema_sha256


def convert(source: Path, output: Path, sample_hz: float) -> dict:
    spec = TASK_SPECS["move_can_pot"]
    written, skipped, errors = 0, 0, []
    for path in sorted(source.rglob("*.hdf5")):
        try:
            with h5py.File(path) as handle:
                times = np.asarray(handle["sim_times"], dtype=np.float64)
                if len(times) < 2:
                    raise ValueError("short trajectory")
                sample_times = np.arange(times[0], times[-1] + 1e-9, 1.0 / sample_hz)
                indices = np.unique(np.searchsorted(times, sample_times, side="left").clip(0, len(times)-1))
                object_poses = np.asarray(handle["object_poses"])[indices]
                names = [value.decode() if isinstance(value, bytes) else str(value)
                         for value in np.asarray(handle["object_names"])]
                ee = np.asarray(handle["ee_poses"])[indices]
                grippers = np.asarray(handle["grippers"])[indices]
                old_events = np.asarray(handle["event_ids"])[indices]
                body = str(handle.attrs["body"])
                seed = int(handle.attrs["seed"])
                success_final = bool(handle.attrs.get("success", False))
                failure_class = str(handle.attrs.get("failure_class", "unknown"))
            canonical_body = {"ARX-X5": "arx-x5", "ur5-wsg": "ur5"}.get(body, body)
            mode = path.parent.name
            target = output / "move_can_pot" / canonical_body / f"exp1_{mode}" / f"episode_seed_{seed}.hdf5"
            if target.exists():
                skipped += 1; continue
            order = [names.index("can"), names.index("pot")]
            object_poses = object_poses[:, order]
            length = len(indices)
            features = np.zeros((length, MAX_NODES, NODE_FEATURE_DIM), np.float32)
            features[:, :2, :7] = object_poses
            features[:, 2, :7] = ee[:, 0]
            features[:, 3, :7] = ee[:, 1]
            features[:, :4, 23] = 1
            features[:, 0, 13] = 1; features[:, 0, 21] = 1; features[:, 1, 22] = 1
            features[:, 2, 13] = 1; features[:, 3, 13] = 1
            velocity = np.gradient(features[:, :4, :3], times[indices], axis=0)
            features[:, :4, 7:10] = velocity
            features[:, :4, 18] = np.linalg.norm(velocity, axis=-1)
            features[:, :4, 17] = np.linalg.norm(features[:, :4, :3] - features[0:1, :4, :3], axis=-1)
            features[:, :4, 20] = features[:, :4, 2] - features[0:1, :4, 2]
            features[:, 2, 16] = grippers[:, 0]; features[:, 3, 16] = grippers[:, 1]
            roster = [
                {"name": "moving", "type": "object", "movable": True, "articulation": False},
                {"name": "target", "type": "support", "movable": False, "articulation": False},
                {"name": "left_gripper", "type": "left_gripper", "movable": True, "articulation": False},
                {"name": "right_gripper", "type": "right_gripper", "movable": True, "articulation": False},
            ]
            mask = np.zeros((length, MAX_NODES), bool); mask[:, :4] = True
            closed = grippers < np.median(grippers, axis=0, keepdims=True)
            success = np.zeros(length, bool)
            if success_final: success[-1] = True
            success = np.maximum.accumulate(success)
            relations, relation_mask = collector.derive_relations(features, closed, success, roster, spec)
            # Existing analytic event labels are more faithful than center-distance
            # for move_can_pot's signed side-of-pot success region.
            r_near = RELATION_NAMES.index("near")
            relations[:, r_near, 0, 1] = old_events >= 2
            goal, goal_mask = collector._goal_arrays(spec, roster)
            ee16 = np.concatenate((ee[:, 0], grippers[:, :1], ee[:, 1], grippers[:, 1:2]), -1).astype(np.float32)
            payload = {
                "node_features": features,
                "node_types": np.broadcast_to(np.asarray([
                    NODE_TYPES.index("object"), NODE_TYPES.index("support"),
                    NODE_TYPES.index("left_gripper"), NODE_TYPES.index("right_gripper"), 0, 0, 0, 0
                ]), (length, MAX_NODES)).copy(),
                "node_mask": mask,
                "edge_features": collector.edge_features(features, mask[0]),
                "relations": relations,
                "relation_mask": relation_mask,
                "events": collector.derive_events(relations, relation_mask, features),
                "goal": goal, "goal_mask": goal_mask, "ee_state16": ee16,
                "actions14": collector.canonical_actions(ee16),
                "sim_times": times[indices], "success": success,
                "planner_error": "" if success_final else failure_class,
                "roster": roster,
            }
            metadata = {
                "format": collector.FORMAT, "schema_sha256": schema_sha256(),
                "task": "move_can_pot", "body": canonical_body,
                "condition": f"exp1_{mode}", "seed": seed,
                "instruction": str(path.name if False else spec.instruction),
                "sample_hz": sample_hz, "native_success": success_final,
                "failure_class": failure_class, "source_path": str(path),
            }
            collector.write_episode(target, payload, metadata); written += 1
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    result = {"format": "etsf_robotwin2_exp1_to_universal_event_v1", "written": written,
              "skipped": skipped, "errors": errors, "schema_sha256": schema_sha256()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "exp1_conversion_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    args = parser.parse_args()
    print(json.dumps(convert(args.source.resolve(), args.output.resolve(), args.sample_hz), indent=2))


if __name__ == "__main__":
    main()
