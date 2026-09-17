"""Train v4 causal imagined beliefs and directly observed relation evidence."""
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
from event_rl.evidence_observer import (
    FORMAT, RELATIONS, GOALS, EvidenceObserver, EvidenceWindows,
    load_evidence_data, load_observer, model_inputs, publish_evidence,
)
from event_rl.semantic_cues import sha256
from event_rl.relational_observer import GOAL_CLASSES
from train_relational_observer_v3 import metrics, future_targets



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
    use_evidence: bool = True


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
            fields["observability_probabilities"] = prediction["observability"].sigmoid()
            for kind in ("current_relations", "prior_relations"):
                fields.update({f"{kind}_{name}": value.softmax(-1) for name, value in prediction[kind].items()})
            fields["history_features"] = prediction["history_features"]
            for name, value in fields.items():
                collected.setdefault(name, []).append(value.cpu().numpy())
            indices.extend(batch["index"].tolist())
    if not indices:
        raise ValueError("empty evaluation")
    return {name: np.concatenate(parts) for name, parts in collected.items()}, np.asarray(indices, np.int64)



def published_fields(fields, arrays, indices, use_evidence=True):
    n = len(indices)
    stable = np.full((n, 3), 2, np.int64)
    inferred = np.full((n, len(GOALS)), 2, np.int64)
    confirmed = inferred.copy()
    uids = arrays["attempt_uid"][indices]
    for uid in np.unique(uids):
        chosen = np.flatnonzero(uids == uid)
        chosen = chosen[np.argsort(arrays["elapsed_s"][indices[chosen]])]
        stable[chosen], inferred[chosen], confirmed[chosen] = publish_evidence(
            arrays["elapsed_s"][indices[chosen]],
            {name: fields[f"relation_{name}"][chosen] for name in RELATIONS},
            {name: fields[f"current_relations_{name}"][chosen] for name in RELATIONS},
            fields["observability_probabilities"][chosen],
            fields["goal_probabilities"][chosen], use_evidence=use_evidence)
    return dict(relation_stable_ids=stable, goal_inferred_ids=inferred,
                goal_confirmed_ids=confirmed, goal_state_ids=confirmed)


def evaluate_fields(fields, arrays, indices, use_evidence=True):
    uids = arrays["attempt_uid"][indices]
    report = dict(relations={}, events={}, goals={}, published_goals={}, inferred_goals={}, observability={})
    for j, (name, classes) in enumerate(RELATIONS.items()):
        report["relations"][name] = metrics(fields[f"relation_{name}"], arrays["relation_labels"][indices, j], uids, classes)
        observable = fields["observability_probabilities"][:, j]
        report["observability"][name] = metrics(np.stack((1-observable, observable), -1),
            arrays["observability_labels"][indices, j], uids, ("unobservable", "observable"))
    for name in ("phase", "transition"):
        report["events"][name] = metrics(fields[f"event_{name}"], arrays["labels"][indices, list(HEADS).index(name)], uids, HEADS[name])
    publication = published_fields(fields, arrays, indices, use_evidence)
    for j, name in enumerate(GOALS):
        truth = arrays["goal_labels"][indices, j]
        report["goals"][name] = metrics(fields["goal_probabilities"][:, j], truth, uids, GOAL_CLASSES)
        for field, group in (("goal_confirmed_ids", "published_goals"), ("goal_inferred_ids", "inferred_goals")):
            published = publication[field][:, j]
            result = metrics(np.eye(3)[published], truth, uids, GOAL_CLASSES)
            result.pop("episode_balanced_ce")
            result["false_success_count"] = int(((published == 1) & (truth == 0)).sum())
            result["known_negative_count"] = int((truth == 0).sum())
            result["unknown_fraction"] = float((published == 2).mean())
            result["success_on_unknown_reference"] = int(((published == 1) & (truth == 2)).sum())
            report[group][name] = result
    losses = []
    for group, weight in (("relations", .5), ("events", .25), ("goals", .25)):
        values = [m["episode_balanced_ce"] for m in report[group].values() if m["episode_balanced_ce"] is not None]
        if values:
            losses.append(weight * float(np.mean(values)))
    report["selection_loss"] = sum(losses) if losses else None
    return report


def train(data, graph, visual, annotations, evidence, output, config=Config(), device="cpu"):
    if Path(output).exists():
        raise ValueError("refusing to overwrite relational training output")
    if min(config.steps, config.batch_size, config.window, config.hidden, config.eval_every) <= 0 or not 0 < config.learning_rate < 1:
        raise ValueError("invalid training config")
    arrays, contract = load_evidence_data(data, graph, visual, annotations, evidence)
    if set(arrays["split"].tolist()) != {"train", "validation"}:
        raise ValueError("v4 training only accepts train/validation; test/inference are forbidden")
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
    model = EvidenceObserver(arrays["node_features"].shape[-1], arrays["edge_features"].shape[-1],
                               config.hidden, config.graph_hidden, config.graph_layers,
                               config.forecast_horizons_s, config.use_graph, config.use_evidence).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    dataset = EvidenceWindows(arrays, config.window, training)
    validation = EvidenceWindows(arrays, config.window, validation_indices)
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
    evidence_pool = np.flatnonzero((arrays["observability_labels"][training] >= 0).any(1))
    if not len(evidence_pool):
        raise ValueError("v4 requires actual reviewed evidence annotations")
    for j in range(3):
        if not {0, 1}.issubset(set(arrays["observability_labels"][training, j])):
            raise ValueError("both observable and unobservable training examples are required")
    if not (arrays["segmentation_labels"][training] >= 0).any():
        raise ValueError("surface auxiliary branch requires actual reviewed training polygons")
    polygon_pool = np.flatnonzero((arrays["segmentation_labels"][training] >= 0).any(axis=(1, 2, 3)))
    rng = np.random.default_rng(config.seed)
    code_hashes = {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                   for path in (Path(__file__).resolve(),
                                Path(__file__).resolve().parents[1] / "event_rl/evidence_observer.py",
                                Path(__file__).resolve().parents[1] / "event_rl/relational_visual.py",
                                Path(__file__).resolve().parents[1] / "event_rl/relational_graph.py")}
    output = Path(output); output.mkdir(parents=True)
    best, best_step, history = float("inf"), 0, []

    def ce(logits, labels, weight=None):
        valid = labels >= 0
        return F.cross_entropy(logits[valid], labels[valid], weight=weight) if bool(valid.any()) else logits.sum() * 0.

    for step in range(1, config.steps + 1):
        model.train()
        selected = []
        for sample_index in range(config.batch_size):
            chance = rng.random()
            pools = class_pools if chance < .2 else rare_pools if chance < .3 and rare_pools else episode_pools
            pool = pools[int(rng.integers(len(pools)))]
            if chance >= .8:
                pool = evidence_pool
            if sample_index == 0:
                pool = polygon_pool
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
        evidence_y = batch["observability_labels"].to(target_device)
        valid_evidence = evidence_y >= 0
        evidence_loss = F.binary_cross_entropy_with_logits(result["observability"][valid_evidence],
            evidence_y[valid_evidence].float()) if valid_evidence.any() else result["observability"].sum() * 0.
        # Direct-frame heads are supervised on reviewed current evidence, with
        # weak coarse known-state supervision elsewhere, not future hindsight.
        current_losses = []
        for j, name in enumerate(RELATIONS):
            cy = batch["current_relation_labels"][:, j].to(target_device)
            coarse = torch.where(relation_y[:, j] < 2, relation_y[:, j], -1)
            current_losses.append(ce(result["current_relations"][name], cy) +
                                  .1 * ce(result["current_relations"][name], torch.where(cy >= 0, -1, coarse)))
        current_loss = torch.stack(current_losses).mean()
        sy = batch["segmentation_labels"].to(target_device)
        sv = sy >= 0
        segmentation_loss = F.binary_cross_entropy_with_logits(result["segmentation"][sv], sy[sv].float()) if sv.any() else result["segmentation"].sum() * 0.
        # Train imagination without leaking the hidden current frame, boxes, or
        # motion. Reference targets are known observations, never invented labels.
        imagination_loss = relation_loss * 0.
        if step % 2 == 0:
            inputs = model_inputs(batch, target_device)
            visible = inputs["mask"].clone()
            for row in range(len(visible)):
                valid = torch.nonzero(visible[row], as_tuple=True)[0]
                hide = min(int(rng.integers(1, 5)), max(0, len(valid) - 1))
                if hide:
                    visible[row, valid[-hide:]] = False
            imagined = model(**inputs, observation_mask=visible)
            imagination_loss = torch.stack([ce(imagined["relations"][name],
                torch.where(relation_y[:, j] < 2, relation_y[:, j], -1)) for j, name in enumerate(RELATIONS)]).mean()
        # Soft consistency regularization complements, but does not replace, visual labels.
        goal_bounds = []
        for goal in GOALS.values():
            goal_bounds.append(torch.stack([result["relations"][name].softmax(-1)[:, goal[j]]
                                             for j, name in enumerate(RELATIONS) if goal[j] >= 0], -1).min(-1).values)
        bound = torch.stack(goal_bounds, -1)
        consistency = F.relu(result["goals"].softmax(-1)[..., 1] - bound).mean()
        loss = (.5 * relation_loss + .25 * event_loss + .25 * goal_loss + .2 * forecast_loss + .05 * consistency
                + .2 * current_loss + .1 * segmentation_loss + .3 * imagination_loss
                + (.2 * evidence_loss if config.use_evidence else 0.))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite relational training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            fields, indices = forward_all(model, validation, target_device, config.batch_size)
            report = evaluate_fields(fields, arrays, indices, config.use_evidence)
            score = report["selection_loss"]
            history.append(dict(step=step, train_loss=float(loss.detach()), relation_loss=float(relation_loss.detach()),
                                event_loss=float(event_loss.detach()), goal_loss=float(goal_loss.detach()),
                                forecast_loss=float(forecast_loss.detach()), evidence_loss=float(evidence_loss.detach()),
                                current_loss=float(current_loss.detach()), segmentation_loss=float(segmentation_loss.detach()),
                                imagination_loss=float(imagination_loss.detach()), gradient_norm=float(gradient), validation=report))
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
                             visual_sha256=sha256(Path(visual)), evidence_sha256=sha256(Path(evidence)),
                             relation_training_counts={k: v.tolist() for k, v in relation_counts.items()},
                             event_training_counts=event_counts, code_sha256=code_hashes,
                             fitting_group_uids=sorted(set(arrays["group_uid"][training].tolist())),
                             selection_group_uids=sorted(set(arrays["group_uid"][validation_indices].tolist())))
                torch.save(saved, output / "selected_observer.pt")
    selected_model, saved = load_observer(output / "selected_observer.pt", device)
    fields, indices = forward_all(selected_model, validation, target_device, config.batch_size)
    report = evaluate_fields(fields, arrays, indices, config.use_evidence)
    updates = {prefix: sum(float((value.detach().cpu() - initial[name]).square().sum())
                           for name, value in selected_model.named_parameters() if name.startswith(prefix)) ** .5
               for prefix in ("messages.", "updates.", "relation_heads.", "goal_head.",
                              "pair_encoder.", "surface_encoder.", "surface_head.", "current_heads.", "evidence_head.", "prior_transition.")}
    required = ("relation_heads.", "goal_head.", "pair_encoder.", "surface_encoder.", "surface_head.", "current_heads.", "prior_transition.") + (("messages.", "updates.") if config.use_graph else ()) + (("evidence_head.",) if config.use_evidence else ())
    if any(updates[name] <= 0 for name in required):
        raise ValueError("a required network component was not updated")
    with (output / "validation_predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **fields, **published_fields(fields, arrays, indices, config.use_evidence),
                            query_id=arrays["query_id"][indices], elapsed_s=arrays["elapsed_s"][indices], split=arrays["split"][indices], indices=indices, relation_labels=arrays["relation_labels"][indices],
                            goal_labels=arrays["goal_labels"][indices], attempt_uid=arrays["attempt_uid"][indices])
    receipt = dict(format=FORMAT, status="trained_not_deployment_validated", config=asdict(config), selected_step=best_step,
                   selection="validation_only_0.5_relation_0.25_event_0.25_goal_episode_balanced_CE",
                   selected_validation_loss=best, validation=report, checkpoint_sha256=sha256(output / "selected_observer.pt"),
                   parameter_count=sum(p.numel() for p in selected_model.parameters()), component_update_norm=updates,
                   train_videos=len(np.unique(arrays["attempt_uid"][training])), validation_videos=len(np.unique(arrays["attempt_uid"][validation_indices])),
                   relation_training_counts={k: v.tolist() for k, v in relation_counts.items()},
                   code_sha256=code_hashes, source_data_sha256=sha256(Path(data)),
                   graph_sha256=sha256(Path(graph)), relation_annotations_sha256=sha256(Path(annotations)),
                   visual_sha256=sha256(Path(visual)), evidence_sha256=sha256(Path(evidence)),
                   imagination="causal trailing-1-to-4-observation masking; only past visual motion available",
                   evidence_training_counts={name: np.bincount(arrays["observability_labels"][training,j][arrays["observability_labels"][training,j]>=0],minlength=2).tolist() for j,name in enumerate(RELATIONS)},
                   confidence_threshold=.6, threshold_calibrated=False, test_used=False, labels_used_as_input=False,
                   raw_joints_used=False, future_images_used=False, actor_modified=False, deployment_authorized=False,
                   cross_task_validated=False, cross_embodiment_validated=False,
                   limitations=["AI relation annotations are not a human gold standard.",
                                "Typed roles and shared GNN parameters, not a full heterogeneous Transformer.",
                                "Auxiliary coarse AI polygon supervision; no depth/contact sensor. Predicted masks do not prove support.",
                                "Region membership is not general 3D containment.",
                                "Task goals are predicate compositions, not unrestricted natural language.",
                                "Goal satisfaction/forecast are not calibrated RL value or chunk advantage."])
    (output / "training_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "graph", "visual", "annotations", "evidence", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-evidence", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = Config(steps=args.steps, batch_size=args.batch_size, eval_every=args.eval_every,
                    window=args.window, learning_rate=args.learning_rate, use_graph=not args.no_graph, use_evidence=not args.no_evidence)
    print(json.dumps(train(args.data, args.graph, args.visual, args.annotations, args.evidence, args.output, config, args.device), ensure_ascii=False, indent=2))
