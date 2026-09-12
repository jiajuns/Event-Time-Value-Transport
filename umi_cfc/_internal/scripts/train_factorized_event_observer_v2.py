#!/usr/bin/env python3
"""Train RGB+YOLO CfC event recognition; no actor loading or AWR reward rewrite."""
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
from event_rl.factorized_events import CHECKPOINT_FORMAT, CORE_HEADS, HEADS, SCHEMA
from event_rl.factorized_observer import (
    CausalWindows, FactorizedEventObserver, classification_loss, load_dataset, load_observer, model_inputs,
)
from event_rl.semantic_cues import sha256
from event_rl.event_stability import future_label_targets


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 16
    window: int = 12
    hidden: int = 96
    learning_rate: float = 3e-4
    eval_every: int = 100
    seed: int = 20260906
    use_cues: bool = True
    forecast_horizons_s: tuple = ()
    forecast_coefficient: float = .2
    rare_event_fraction: float = 0.0


RARE_TRANSITIONS = ("drop", "regrasp", "release")


def rare_event_episode_indices(arrays, supervised):
    """Return dataset-position anchors grouped by rare event and train episode."""
    supervised = np.asarray(supervised, dtype=np.int64)
    train_positions = np.flatnonzero(arrays["split"][supervised].astype(str) == "train")
    transition = list(HEADS).index("transition")
    result = {}
    for event in RARE_TRANSITIONS:
        event_id = HEADS["transition"].index(event)
        positions = train_positions[arrays["labels"][supervised[train_positions], transition] == event_id]
        episodes = [positions[arrays["attempt_uid"][supervised[positions]] == uid]
                    for uid in np.unique(arrays["attempt_uid"][supervised[positions]])]
        if episodes:
            result[event] = episodes
    return result


def sample_training_indices(generator, episode_indices, rare_event_episodes, batch_size, rare_event_fraction):
    """Sample dataset positions, preserving the legacy RNG path when fraction is zero."""
    selected = []
    rare_events = tuple(rare_event_episodes)
    for _ in range(batch_size):
        use_rare = (rare_event_fraction > 0 and rare_events and
                    generator.random() < rare_event_fraction)
        pools = rare_event_episodes[rare_events[int(generator.integers(len(rare_events)))]] if use_rare else episode_indices
        episode = pools[int(generator.integers(len(pools)))]
        selected.append(int(generator.choice(episode)))
    return selected


def cue_signature(contract):
    cues = contract["semantic_cues"]
    detector = cues["detector"]
    return dict(cameras=contract["cameras"], image_size=contract["image_size"],
                features=cues["feature_names"], confidence=cues["confidence_threshold"],
                detector={key: detector.get(key) for key in ("backend", "version", "weights_sha256", "prompts", "imgsz", "conf")})


def evaluate(model, dataset, device, batch_size, active_heads, *, with_features=False):
    collected = {name: [] for name in active_heads}
    horizons=model.model_spec.get('forecast_horizons_s', [])
    forecast={f'h{k}_{name}':[] for k,_ in enumerate(horizons) for name in active_heads}
    features = []
    indices = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**model_inputs(batch, device), return_features=with_features)
            if with_features:
                features.append(output["history_features"].cpu().numpy())
            indices.extend(batch["index"].tolist())
            for name in active_heads:
                collected[name].append(output[name].cpu())
            for key in forecast:
                forecast[key].append(output['forecast'][key].cpu())
    if not indices:
        raise ValueError("empty evaluation split")
    logits = {name: torch.cat(parts) for name, parts in collected.items()}
    metrics, losses = {}, []
    indices = np.asarray(indices, dtype=np.int64)
    for name in active_heads:
        truth = dataset.arrays["labels"][indices, list(HEADS).index(name)]
        valid = truth >= 0
        predicted = logits[name].argmax(-1).numpy()
        confusion = np.zeros((len(HEADS[name]), len(HEADS[name])), dtype=np.int64)
        np.add.at(confusion, (truth[valid], predicted[valid]), 1)
        support = confusion.sum(1)
        tp = np.diag(confusion)
        precision = np.divide(tp, confusion.sum(0), out=np.zeros(len(tp), dtype=float), where=confusion.sum(0) > 0)
        recall = np.divide(tp, support, out=np.zeros(len(tp), dtype=float), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(tp), dtype=float), where=precision + recall > 0)
        loss = None
        if valid.any():
            ce = F.cross_entropy(logits[name][valid], torch.as_tensor(truth[valid]), reduction="none").numpy()
            uids = dataset.arrays["attempt_uid"][indices][valid]
            loss = float(np.mean([ce[uids == uid].mean() for uid in np.unique(uids)]))
            losses.append(loss)
        metrics[name] = dict(labeled_frames=int(valid.sum()), episode_balanced_ce=loss,
                             accuracy=float(tp.sum() / valid.sum()) if valid.any() else None,
                             macro_f1_present_classes=float(f1[support > 0].mean()) if valid.any() else None,
                             classes=list(HEADS[name]), confusion=confusion.tolist(),
                             precision=precision.tolist(), recall=recall.tolist(), support=support.tolist())
        if name == "holding":
            held, unheld = HEADS[name].index("held"), HEADS[name].index("unheld")
            negatives = truth == unheld
            metrics[name]["unheld_predicted_held_rate"] = float((predicted[negatives] == held).mean()) if negatives.any() else None
    metrics["selection_loss"] = float(np.mean(losses)) if losses else None
    if horizons:
        targets=future_label_targets(dataset.arrays,horizons)[indices]
        metrics['forecast_supervision_only']={}
        for k,horizon in enumerate(horizons):
            for column,name in enumerate(HEADS):
                if name not in active_heads:
                    continue
                truth=targets[:,k,column]
                valid=truth>=0
                prediction=torch.cat(forecast[f'h{k}_{name}']).argmax(-1).numpy()
                metrics['forecast_supervision_only'][f'{horizon}s_{name}']=dict(
                    labeled_frames=int(valid.sum()),accuracy=float((prediction[valid]==truth[valid]).mean()) if valid.any() else None)
    result = (logits, indices, metrics)
    return (*result, np.concatenate(features)) if with_features else result


def train(data: Path, output: Path, config: TrainConfig, device="cpu"):
    if (min(config.steps, config.batch_size, config.window, config.hidden, config.eval_every) <= 0 or
            not np.isfinite(config.learning_rate) or config.learning_rate <= 0):
        raise ValueError("invalid training configuration")
    if not np.isfinite(config.rare_event_fraction) or not 0 <= config.rare_event_fraction <= 1:
        raise ValueError("rare_event_fraction must be between 0 and 1")
    if output.exists():
        raise ValueError("refusing to overwrite training output")
    arrays, contract = load_dataset(data)
    if not np.isfinite(config.forecast_coefficient) or config.forecast_coefficient<0:
        raise ValueError('invalid forecast loss coefficient')
    future_targets=future_label_targets(arrays,config.forecast_horizons_s)
    split = arrays["split"].astype(str)
    if not {"train", "validation"}.issubset(set(split)) or not set(split) <= {"train", "validation", "test"}:
        raise ValueError("training requires train/validation episode splits; test is optional")
    training = np.flatnonzero(split == "train")
    counts = {name: np.bincount(arrays["labels"][training, column][arrays["labels"][training, column] >= 0],
                                minlength=len(classes)) for column, (name, classes) in enumerate(HEADS.items())}
    active = [name for name in HEADS if counts[name].sum() > 0]
    if not set(CORE_HEADS).issubset(active):
        raise ValueError("train split needs reviewed phase/holding/placement labels")
    for name in active:
        if not np.any(arrays["labels"][split == "validation", list(HEADS).index(name)] >= 0):
            raise ValueError(f"validation needs reviewed {name} labels for model selection")
    columns = [list(HEADS).index(name) for name in active]
    supervised = training[np.any(arrays["labels"][training][:, columns] >= 0, axis=1)]
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose a compatible environment or --device cpu")
    model = FactorizedEventObserver(len(contract["cameras"]), arrays["semantic_features"].shape[1],
                                    config.hidden, config.use_cues, config.forecast_horizons_s)
    model.fit_normalization(arrays["semantic_features"][training])
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    weights = {name: torch.as_tensor(np.minimum(np.sqrt(counts[name].sum() /
                                                       np.maximum(counts[name], 1)), 5.),
                                    dtype=torch.float32, device=target_device) for name in active}
    dataset = CausalWindows(arrays, config.window, supervised)
    # Equal episode probability, then equal anchor probability within episode.
    episode_indices = [np.flatnonzero(arrays["attempt_uid"][supervised] == uid)
                       for uid in np.unique(arrays["attempt_uid"][supervised])]
    rare_event_episodes = rare_event_episode_indices(arrays, supervised)
    validation = CausalWindows(arrays, config.window, np.flatnonzero(split == "validation"))
    generator = np.random.default_rng(config.seed)
    output.mkdir(parents=True)
    initial_cfc = {name: value.detach().cpu().clone() for name, value in model.cfc.named_parameters()}
    best, best_step, history = float("inf"), 0, []
    for step in range(1, config.steps + 1):
        model.train()
        selected = sample_training_indices(generator, episode_indices, rare_event_episodes,
                                           config.batch_size, config.rare_event_fraction)
        batch = torch.utils.data.default_collate([dataset[index] for index in selected])
        prediction = model(**model_inputs(batch, target_device))
        loss = classification_loss(prediction, batch["labels"].to(target_device), active, weights)
        current_loss=loss
        future_losses=[]
        for k,_ in enumerate(config.forecast_horizons_s):
            target=torch.as_tensor(future_targets[batch['index'].numpy(),k],device=target_device)
            if bool((target[:,columns]>=0).any()):
                future_losses.append(classification_loss({name:prediction['forecast'][f'h{k}_{name}'] for name in active},
                                                          target,active,weights))
        forecast_loss=torch.stack(future_losses).mean() if future_losses else loss.new_zeros(())
        loss=loss+config.forecast_coefficient*forecast_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            _, _, metrics = evaluate(model, validation, target_device, config.batch_size, active)
            score = metrics["selection_loss"]
            history.append(dict(step=step, train_loss=float(loss.detach()), current_loss=float(current_loss.detach()),
                                forecast_loss=float(forecast_loss.detach()), grad_norm=float(grad_norm), validation=metrics))
            (output / 'training_history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2))
            print(json.dumps(dict(step=step, train_loss=float(loss.detach()), validation_loss=score)), flush=True)
            if score < best:
                best, best_step = score, step
                saved = dict(format=CHECKPOINT_FORMAT, schema=SCHEMA, model_spec=model.model_spec,
                             model={name: value.detach().cpu() for name, value in model.state_dict().items()},
                             config=asdict(config), active_heads=active, selected_step=step,
                             input_contract=contract, cue_signature=cue_signature(contract),
                             training_data_sha256=sha256(data), labels_are_model_inputs=False)
                saved["fitting_group_uids"] = sorted(set(arrays["group_uid"][split == "train"].tolist()))
                saved["selection_group_uids"] = sorted(set(arrays["group_uid"][split == "validation"].tolist()))
                saved["training_counts"] = {name: value.tolist() for name, value in counts.items()}
                torch.save(saved, output / "selected_observer.pt")
    selected_model, saved = load_observer(output / "selected_observer.pt", device)
    test_metrics = None
    if np.any(split == "test"):
        test = CausalWindows(arrays, config.window, np.flatnonzero(split == "test"))
        _, _, test_metrics = evaluate(selected_model, test, target_device, config.batch_size, active)
    update = sum(float((value.detach().cpu() - initial_cfc[name]).square().sum())
                 for name, value in selected_model.cfc.named_parameters()) ** .5
    receipt = dict(format=CHECKPOINT_FORMAT, status="trained_not_deployment_validated", config=asdict(config),
                   active_heads=active, inactive_heads=[name for name in HEADS if name not in active],
                   training_counts={name: value.tolist() for name, value in counts.items()},
                   missing_train_classes={name: [HEADS[name][i] for i, count in enumerate(counts[name]) if count == 0] for name in active},
                   selected_step=best_step, selected_validation_loss=best,
                   selection="validation_only_episode_balanced_cross_entropy",
                   test_evaluated_after_selection=test_metrics is not None, test=test_metrics, cfc_update_norm=update,
                   source_data_sha256=sha256(data), checkpoint_sha256=sha256(output / "selected_observer.pt"),
                   labels_are_model_inputs=False, raw_joint_state_used=False,
                   actor_modified=False, cross_embodiment_validated=False, deployment_authorized=False,
                   future_frames_used_as_inputs=False, forecast_is_current_fact=False,
                   limitations=["RGB encoder is trained from scratch; no real-video accuracy is implied by a smoke run.",
                                "Unknown is a supervised class; low-confidence abstention is a separate uncalibrated policy.",
                                "No value/AWR reward conversion or robot action adapter is implemented in this observer."])
    (output / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-cues", action="store_true", help="controlled RGB-only ablation on the same dataset")
    parser.add_argument('--forecast-horizons-s',type=float,nargs='*',default=[])
    parser.add_argument('--forecast-coefficient',type=float,default=.2)
    parser.add_argument('--rare-event-fraction', type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    config = TrainConfig(args.steps, args.batch_size, args.window, args.hidden, args.learning_rate,
                         args.eval_every, args.seed, not args.no_cues, tuple(args.forecast_horizons_s),
                         args.forecast_coefficient, args.rare_event_fraction)
    print(json.dumps(train(args.data, args.output, config, args.device), ensure_ascii=False, indent=2))
