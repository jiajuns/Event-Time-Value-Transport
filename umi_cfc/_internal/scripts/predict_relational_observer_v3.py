"""Frozen, causal relational observer predictions; renderer-compatible event view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.factorized_events import HEADS
from event_rl.event_stability import CausalEventStabilizer
from event_rl.relational_observer import (
    FORMAT, RELATIONS, GOALS, GOAL_CLASSES, RelationalWindows, load_relational_data,
    load_observer, confirm_relations, publish_goal,
)
from event_rl.semantic_cues import sha256
from train_relational_observer_v3 import forward_all
from train_factorized_event_observer_v2 import cue_signature


def graph_signature(contract):
    return {key: contract.get(key) for key in (
        "format", "roles", "max_nodes", "roi_size", "node_features", "edge_features", "tracking",
        "max_missing_observations", "iou_threshold", "bbox_only_no_physical_contact_claim")}


def predict(data, graph, checkpoint, output, *, device="cpu", batch_size=16, confidence=.6, goal="place_on"):
    output = Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite relational predictions")
    if goal not in GOALS or not 0 <= confidence <= 1 or batch_size <= 0:
        raise ValueError("invalid goal/confidence/batch size")
    arrays, contract = load_relational_data(data, graph)  # No relation annotation path accepted.
    model, saved = load_observer(checkpoint, device)
    if cue_signature(contract) != cue_signature(saved["input_contract"]):
        raise ValueError("detector/image contract differs from training")
    if graph_signature(contract["relational_graph"]) != graph_signature(saved["input_contract"]["relational_graph"]):
        raise ValueError("graph feature/tracking contract differs from training")
    dataset = RelationalWindows(arrays, saved["config"]["window"])
    predicted, indices = forward_all(model, dataset, torch.device(device), batch_size)
    result = {key: arrays[key][indices] for key in ("attempt_uid", "query_id", "elapsed_s", "split", "group_uid")}
    result["history_features"] = predicted["history_features"]
    for key in ("event_state", "event_posterior", "event_progress", "event_boundary_probability",
                "event_uncertainty", "event_value"):
        result[key] = predicted[key]
    result["relation_stable_ids"] = np.full((len(indices), 3), 2, np.int64)
    for name in RELATIONS:
        result[f"relation_{name}_probabilities"] = predicted[f"relation_{name}"]
    for name in ("phase", "transition"):
        p = predicted[f"event_{name}"]
        result[f"{name}_probabilities"] = p
        result[f"{name}_id"] = np.where(p.max(-1) >= confidence, p.argmax(-1), HEADS[name].index("unknown"))
    result["holding_probabilities"] = predicted["relation_held_by_actor"]
    result["placement_probabilities"] = predicted["goal_probabilities"][:, list(GOALS).index("place_on")]
    for name in ("holding", "placement"):
        p = result[f"{name}_probabilities"]
        result[f"{name}_id"] = np.where(p.max(-1) >= confidence, p.argmax(-1), 2)
    for name in ("phase", "holding", "placement"):
        result[f"{name}_stable_id"] = np.full(len(indices), HEADS[name].index("unknown"), np.int64)
    for uid in np.unique(result["attempt_uid"]):
        chosen = np.flatnonzero(result["attempt_uid"] == uid)
        chosen = chosen[np.argsort(result["elapsed_s"][chosen])]
        result["relation_stable_ids"][chosen] = confirm_relations(
            result["elapsed_s"][chosen], {name: predicted[f"relation_{name}"][chosen] for name in RELATIONS}, confidence=confidence)
        stabilizer = CausalEventStabilizer(confidence=confidence)
        for i in chosen:
            current = stabilizer.update(float(result["elapsed_s"][i]), {name: result[f"{name}_probabilities"][i] for name in HEADS})
            result["phase_stable_id"][i] = HEADS["phase"].index(current["phase"]["state"])
    result["goal_probabilities"] = predicted["goal_probabilities"]
    result["goal_state_ids"] = np.stack([
        publish_goal(result["relation_stable_ids"], predicted["goal_probabilities"][:, j], spec, confidence=confidence)
        for j, spec in enumerate(GOALS.values())], -1)
    result["holding_stable_id"] = result["relation_stable_ids"][:, 0]
    result["placement_stable_id"] = result["goal_state_ids"][:, list(GOALS).index("place_on")]
    result["placement_raw_id"] = result["placement_id"].copy()
    result["placement_id"] = result["placement_stable_id"].copy()
    # No retained success beliefs: missing support remains unknown in the published view.
    for name in ("phase", "holding", "placement"):
        result[f"{name}_belief_id"] = result[f"{name}_stable_id"].copy()
    for key, value in predicted.items():
        if key.startswith("forecast_"):
            result[key + "_probabilities"] = value
    relation_counts = saved["relation_training_counts"]
    receipt = dict(format=FORMAT + "_predictions", checkpoint_sha256=sha256(Path(checkpoint)),
                   data_sha256=sha256(Path(data)), graph_sha256=sha256(Path(graph)),
                   relation_schema={name: list(classes) for name, classes in RELATIONS.items()},
                   goal_schema={name: list(spec) for name, spec in GOALS.items()}, goal_classes=list(GOAL_CLASSES),
                   requested_goal=goal, confidence_threshold=confidence, threshold_calibrated=False,
                   ground_truth_used_as_input=False, relation_annotations_used_as_input=False,
                   raw_joints_used=False, future_frames_used_as_inputs=False,
                   forecast_horizons_s=[],  # Legacy renderer must not mislabel relation forecasts as goal probability.
                   relation_forecast_horizons_s=saved["model_spec"]["forecast_horizons_s"],
                   relation_training_counts=relation_counts,
                   observer_training_counts=saved.get("event_training_counts", {}),
                   stable_confirmation=dict(frames=3, includes_current=True),
                   placement_semantics="place_on_goal_requires_unheld_supported_inside_and_confident_goal_head",
                   goal_score_is_calibrated_value=False, graph_signature=graph_signature(contract["relational_graph"]),
                   event_observer_outputs=("event_state", "event_posterior", "event_progress",
                                           "event_boundary_probability", "event_uncertainty", "event_value"),
                   event_value_is_rl_trained=False,
                   fitting_group_uids=saved["fitting_group_uids"], selection_group_uids=saved["selection_group_uids"],
                   deployment_authorized=False, cross_task_validated=False, cross_embodiment_validated=False)
    result["contract_json"] = np.asarray(json.dumps(receipt, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "graph", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=.6)
    parser.add_argument("--goal", choices=list(GOALS), default="place_on")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(predict(args.data, args.graph, args.checkpoint, args.output, device=args.device,
                              batch_size=args.batch_size, confidence=args.confidence, goal=args.goal), ensure_ascii=False, indent=2))
