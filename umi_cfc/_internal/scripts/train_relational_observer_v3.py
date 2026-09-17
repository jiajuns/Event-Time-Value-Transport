"""Train the YOLO-assisted GNN relation observer on train/validation only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.factorized_events import HEADS
from event_rl.relational_observer import (
    FORMAT, RELATIONS, GOALS, GOAL_CLASSES, RelationalObserver, RelationalWindows,
    load_relational_data, load_observer, model_inputs, confirm_relations, publish_goal,
)
from event_rl.semantic_cues import sha256


@dataclass(frozen=True)
class Config:
    steps: int = 4000
    batch_size: int = 16
    window: int = 12
    hidden: int = 96
    graph_hidden: int = 48
    graph_layers: int = 2
    learning_rate: float = 1e-4
    eval_every: int = 200
    seed: int = 20260906
    forecast_horizons_s: tuple = (.3, .6)
    use_graph: bool = True


def metrics(probabilities, truth, uids, classes):
    valid = truth >= 0
    prediction = probabilities.argmax(-1)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (truth[valid], prediction[valid]), 1)
    tp = np.diag(confusion)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros(len(classes), float), where=confusion.sum(0) > 0)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(len(classes), float), where=confusion.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes), float), where=precision + recall > 0)
    ce = -np.log(np.maximum(probabilities[valid, truth[valid]], 1e-12))
    loss = float(np.mean([ce[uids[valid] == uid].mean() for uid in np.unique(uids[valid])])) if valid.any() else None
    return dict(classes=list(classes), labeled_frames=int(valid.sum()), confusion=confusion.tolist(),
                support=confusion.sum(1).tolist(), precision=precision.tolist(), recall=recall.tolist(),
                accuracy=float(tp.sum() / valid.sum()) if valid.any() else None,
                macro_f1_present_classes=float(f1[confusion.sum(1) > 0].mean()) if valid.any() else None,
                episode_balanced_ce=loss)


def forward_all(model, dataset, device, batch_size):
    collected, indices = {}, []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            prediction = model(**model_inputs(batch, device), return_features=True)
            fields = {f"relation_{name}": value.softmax(-1) for name, value in prediction["relations"].items()}
            fields.update({f"event_{name}": value.softmax(-1) for name, value in prediction["events"].items()})
            fields.update({f"forecast_{name}": value.softmax(-1) for name, value in prediction["forecast"].items()})
            fields["goal_probabilities"] = prediction["goals"].softmax(-1)
            fields["history_features"] = prediction["history_features"]
            # Persist event-sidecar outputs for audit and future πRL joins. The
            # current UMI labels supervise only auxiliary phase/transition
            # heads; no unreviewed labels are fabricated for these outputs.
            fields["event_state"] = prediction["event_state"]
            fields["event_posterior"] = prediction["event_posterior"].softmax(-1)
            fields["event_progress"] = prediction["event_progress"]
            fields["event_boundary_probability"] = prediction["event_boundary_logit"].sigmoid()
            fields["event_uncertainty"] = prediction["event_uncertainty"]
            fields["event_value"] = prediction["event_value"]
            for name, value in fields.items():
                collected.setdefault(name, []).append(value.cpu().numpy())
            indices.extend(batch["index"].tolist())
    if not indices:
        raise ValueError("empty evaluation")
    return {name: np.concatenate(parts) for name, parts in collected.items()}, np.asarray(indices, np.int64)


def evaluate_fields(fields, arrays, indices):
    uids = arrays["attempt_uid"][indices]
    report = dict(relations={}, events={}, goals={}, published_goals={})
    for j, (name, classes) in enumerate(RELATIONS.items()):
        report["relations"][name] = metrics(fields[f"relation_{name}"], arrays["relation_labels"][indices, j], uids, classes)
    for name in ("phase", "transition"):
        report["events"][name] = metrics(fields[f"event_{name}"], arrays["labels"][indices, list(HEADS).index(name)], uids, HEADS[name])
    stable = np.full((len(indices), 3), 2, np.int64)
    for uid in np.unique(uids):
        chosen = np.flatnonzero(uids == uid)
        chosen = chosen[np.argsort(arrays["elapsed_s"][indices[chosen]])]
        stable[chosen] = confirm_relations(arrays["elapsed_s"][indices[chosen]],
                                          {name: fields[f"relation_{name}"][chosen] for name in RELATIONS})
    for j, (name, goal) in enumerate(GOALS.items()):
        truth = arrays["goal_labels"][indices, j]
        report["goals"][name] = metrics(fields["goal_probabilities"][:, j], truth, uids, GOAL_CLASSES)
        published = publish_goal(stable, fields["goal_probabilities"][:, j], goal)
        result = metrics(np.eye(3)[published], truth, uids, GOAL_CLASSES)
        result.pop("episode_balanced_ce")
        result["false_success_count"] = int(((published == 1) & (truth == 0)).sum())
        result["known_negative_count"] = int((truth == 0).sum())
        result["false_success_rate"] = float(((published == 1) & (truth == 0)).sum() / (truth == 0).sum()) if (truth == 0).any() else None
        result["unknown_fraction"] = float((published == 2).mean())
        report["published_goals"][name] = result
    losses = []
    for group, weight in (("relations", .5), ("events", .25), ("goals", .25)):
        values = [m["episode_balanced_ce"] for m in report[group].values() if m["episode_balanced_ce"] is not None]
        if values:
            losses.append(weight * float(np.mean(values)))
    report["selection_loss"] = sum(losses) if losses else None
    return report


def future_targets(arrays, horizons):
    target = np.full((len(arrays["labels"]), len(horizons), 3), -1, np.int64)
    for uid in np.unique(arrays["attempt_uid"]):
        selected = np.flatnonzero(arrays["attempt_uid"] == uid)
        selected = selected[np.argsort(arrays["elapsed_s"][selected])]
        times = arrays["elapsed_s"][selected]
        for k, horizon in enumerate(horizons):
            for pos, index in enumerate(selected):
                dest = np.searchsorted(times, times[pos] + horizon - 1e-6)
                if dest < len(times) and abs(times[dest] - times[pos] - horizon) <= .051 and np.all(np.diff(times[pos:dest + 1]) <= .151):
                    target[index, k] = arrays["relation_labels"][selected[dest]]
    return target


def train(data, graph, annotations, output, config=Config(), device="cpu"):
    if Path(output).exists():
        raise ValueError("refusing to overwrite relational training output")
    if min(config.steps, config.batch_size, config.window, config.hidden, config.eval_every) <= 0 or not 0 < config.learning_rate < 1:
        raise ValueError("invalid training config")
    arrays, contract = load_relational_data(data, graph, annotations)
    if set(arrays["split"].tolist()) != {"train", "validation"}:
        raise ValueError("v3 training only accepts train/validation; test/inference are forbidden")
    training = np.flatnonzero(arrays["split"] == "train")
    validation_indices = np.flatnonzero(arrays["split"] == "validation")
    if set(arrays["group_uid"][training]) & set(arrays["group_uid"][validation_indices]):
        raise ValueError("training and validation source groups overlap")
    relation_counts = {name: np.bincount(arrays["relation_labels"][training, j][arrays["relation_labels"][training, j] >= 0], minlength=3)
                       for j, name in enumerate(RELATIONS)}
    if any((count == 0).any() for count in relation_counts.values()):
        raise ValueError("all current relation classes including unknown need actual training supervision")
    for j in range(3):
        if not np.any(arrays["relation_labels"][validation_indices, j] >= 0):
            raise ValueError("validation relation supervision missing")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    target_device = torch.device(device)
    model = RelationalObserver(arrays["node_features"].shape[-1], arrays["edge_features"].shape[-1],
                               config.hidden, config.graph_hidden, config.graph_layers,
                               config.forecast_horizons_s, config.use_graph).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    dataset = RelationalWindows(arrays, config.window, training)
    validation = RelationalWindows(arrays, config.window, validation_indices)
    episode_pools = [np.flatnonzero(arrays["attempt_uid"][training] == uid) for uid in np.unique(arrays["attempt_uid"][training])]
    class_pools = [np.flatnonzero(arrays["relation_labels"][training, j] == label) for j in range(3) for label in range(3)]
    rare = arrays["labels"][training, list(HEADS).index("transition")]
    rare_pools = [np.flatnonzero(rare == HEADS["transition"].index(name)) for name in ("drop", "regrasp", "release")]
    rare_pools = [pool for pool in rare_pools if len(pool)]
    future = future_targets(arrays, config.forecast_horizons_s)
    weights = {}
    for name, count in relation_counts.items():
        weights[name] = torch.as_tensor(np.minimum(np.sqrt(count.sum() / np.maximum(count, 1)), 4.), dtype=torch.float32, device=target_device)
    event_weights, event_counts = {}, {}
    for name in ("phase", "transition"):
        y = arrays["labels"][training, list(HEADS).index(name)]
        count = np.bincount(y[y >= 0], minlength=len(HEADS[name]))
        event_counts[name] = count.tolist()
        event_weights[name] = torch.as_tensor(np.minimum(np.sqrt(count.sum() / np.maximum(count, 1)), 5.), dtype=torch.float32, device=target_device)
    rng = np.random.default_rng(config.seed)
    code_hashes = {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                   for path in (Path(__file__).resolve(),
                                Path(__file__).resolve().parents[1] / "event_rl/relational_observer.py",
                                Path(__file__).resolve().parents[1] / "event_rl/relational_graph.py")}
    output = Path(output); output.mkdir(parents=True)
    best, best_step, history = float("inf"), 0, []

    def ce(logits, labels, weight=None):
        valid = labels >= 0
        return F.cross_entropy(logits[valid], labels[valid], weight=weight) if bool(valid.any()) else logits.sum() * 0.

    for step in range(1, config.steps + 1):
        model.train()
        selected = []
        for _ in range(config.batch_size):
            chance = rng.random()
            pools = class_pools if chance < .2 else rare_pools if chance < .3 and rare_pools else episode_pools
            pool = pools[int(rng.integers(len(pools)))]
            selected.append(int(rng.choice(pool)))
        batch = torch.utils.data.default_collate([dataset[i] for i in selected])
        result = model(**model_inputs(batch, target_device))
        relation_y = batch["relation_labels"].to(target_device)
        relation_loss = torch.stack([ce(result["relations"][name], relation_y[:, j], weights[name]) for j, name in enumerate(RELATIONS)]).mean()
        event_loss = torch.stack([ce(result["events"][name], batch["labels"][:, list(HEADS).index(name)].to(target_device), event_weights[name]) for name in ("phase", "transition")]).mean()
        goal_loss = ce(result["goals"].flatten(0, 1), batch["goal_labels"].to(target_device).flatten())
        forecast_losses = []
        for k, _ in enumerate(config.forecast_horizons_s):
            for j, name in enumerate(RELATIONS):
                y = torch.as_tensor(future[batch["index"].numpy(), k, j], device=target_device)
                forecast_losses.append(ce(result["forecast"][f"h{k}_{name}"], y, weights[name]))
        forecast_loss = torch.stack(forecast_losses).mean() if forecast_losses else relation_loss * 0.
        # Soft consistency regularization complements, but does not replace, visual labels.
        goal_bounds = []
        for goal in GOALS.values():
            goal_bounds.append(torch.stack([result["relations"][name].softmax(-1)[:, goal[j]]
                                             for j, name in enumerate(RELATIONS) if goal[j] >= 0], -1).min(-1).values)
        bound = torch.stack(goal_bounds, -1)
        consistency = F.relu(result["goals"].softmax(-1)[..., 1] - bound).mean()
        loss = .5 * relation_loss + .25 * event_loss + .25 * goal_loss + .2 * forecast_loss + .05 * consistency
        if not torch.isfinite(loss):
            raise ValueError("nonfinite relational training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            fields, indices = forward_all(model, validation, target_device, config.batch_size)
            report = evaluate_fields(fields, arrays, indices)
            score = report["selection_loss"]
            history.append(dict(step=step, train_loss=float(loss.detach()), relation_loss=float(relation_loss.detach()),
                                event_loss=float(event_loss.detach()), goal_loss=float(goal_loss.detach()),
                                forecast_loss=float(forecast_loss.detach()), gradient_norm=float(gradient), validation=report))
            (output / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
            print(json.dumps(dict(step=step, loss=float(loss.detach()), validation_loss=score,
                                  place_on=report["published_goals"]["place_on"])), flush=True)
            if score < best:
                best, best_step = score, step
                saved = dict(format=FORMAT, model_spec=model.model_spec, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                             config=asdict(config), selected_step=step, labels_are_model_inputs=False,
                             relation_schema={k: list(v) for k, v in RELATIONS.items()}, goals={k: list(v) for k, v in GOALS.items()},
                             input_contract=contract, training_data_sha256=sha256(Path(data)),
                             graph_sha256=sha256(Path(graph)), annotation_sha256=sha256(Path(annotations)),
                             relation_training_counts={k: v.tolist() for k, v in relation_counts.items()},
                             event_training_counts=event_counts, code_sha256=code_hashes,
                             fitting_group_uids=sorted(set(arrays["group_uid"][training].tolist())),
                             selection_group_uids=sorted(set(arrays["group_uid"][validation_indices].tolist())))
                torch.save(saved, output / "selected_observer.pt")
    selected_model, saved = load_observer(output / "selected_observer.pt", device)
    fields, indices = forward_all(selected_model, validation, target_device, config.batch_size)
    report = evaluate_fields(fields, arrays, indices)
    updates = {prefix: sum(float((value.detach().cpu() - initial[name]).square().sum())
                           for name, value in selected_model.named_parameters() if name.startswith(prefix)) ** .5
               for prefix in ("history_fusion.", "messages.", "updates.", "relation_heads.", "goal_head.")}
    required = ("history_fusion.", "relation_heads.", "goal_head.") + (("messages.", "updates.") if config.use_graph else ())
    if any(updates[name] <= 0 for name in required):
        raise ValueError("a required network component was not updated")
    with (output / "validation_predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **fields, indices=indices, relation_labels=arrays["relation_labels"][indices],
                            goal_labels=arrays["goal_labels"][indices], attempt_uid=arrays["attempt_uid"][indices])
    receipt = dict(format=FORMAT, status="trained_not_deployment_validated", config=asdict(config), selected_step=best_step,
                   selection="validation_only_0.5_relation_0.25_event_0.25_goal_episode_balanced_CE",
                   selected_validation_loss=best, validation=report, checkpoint_sha256=sha256(output / "selected_observer.pt"),
                   parameter_count=sum(p.numel() for p in selected_model.parameters()), component_update_norm=updates,
                   train_videos=len(np.unique(arrays["attempt_uid"][training])), validation_videos=len(np.unique(arrays["attempt_uid"][validation_indices])),
                   relation_training_counts={k: v.tolist() for k, v in relation_counts.items()},
                   code_sha256=code_hashes, source_data_sha256=sha256(Path(data)),
                   graph_sha256=sha256(Path(graph)), relation_annotations_sha256=sha256(Path(annotations)),
                   confidence_threshold=.6, threshold_calibrated=False, test_used=False, labels_used_as_input=False,
                   raw_joints_used=False, future_images_used=False, actor_modified=False, deployment_authorized=False,
                   cross_task_validated=False, cross_embodiment_validated=False,
                   limitations=["AI relation annotations are not a human gold standard.",
                                "Typed roles and shared GNN parameters, not a full heterogeneous Transformer.",
                                "RGB and bounding-box graph; no depth/contact sensor or segmentation mask input.",
                                "Region membership is not general 3D containment.",
                                "Task goals are predicate compositions, not unrestricted natural language.",
                                "Event-State/Event-Value heads require reviewed event labels and online SMDP returns before use in RL.",
                                "Goal satisfaction/forecast are not calibrated RL value or chunk advantage."])
    (output / "training_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "graph", "annotations", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = Config(steps=args.steps, batch_size=args.batch_size, eval_every=args.eval_every,
                    window=args.window, learning_rate=args.learning_rate, use_graph=not args.no_graph)
    print(json.dumps(train(args.data, args.graph, args.annotations, args.output, config, args.device), ensure_ascii=False, indent=2))
