#!/usr/bin/env python3
"""Train one fixed TABLE-V method/seed for exactly 3000 optimizer steps.

The script runs two source-body LOBO folds for leakage-free success
calibration, then a formal 540-source-trajectory fit.  The formal fit always
runs exactly 3000 optimizer steps and its published checkpoint is step 3000.
All formal step snapshots are retained by default.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment2_fixed_core_v2 import (  # noqa: E402
    FORMAT,
    METHODS,
    STAGES,
    BalancedExampleSampler,
    BoundaryDataset,
    MatchedRankSampler,
    aggregate_fold_metrics,
    build_boundary_dataset,
    collect_rollouts,
    compute_training_loss,
    evaluate_model,
    file_sha256,
    fit_success_calibration,
    forward_indices,
    json_dump,
    logical_data_manifest_sha256,
    model_for_method,
    parameter_report,
    save_checkpoint,
    split_adapt_test_rollouts,
    state_dict_cpu,
    update_ema,
    validate_frozen_protocol,
)


FROZEN_DATA_MANIFEST_SHA256 = "a71666e8297c1b3919c2ec76de3e6b4e2a022adba0889d861f7435895b89d971"
FROZEN_STATS_SHA256 = "f84434f276e8a4f118e76b35c061e72af04a662a58df613503ebfb04041f5d14"
SOURCE_BODIES = ("aloha-agilex", "arx-x5")
TARGET_BODIES = ("piper", "ur5")
FORMAL_STEPS = 3000
PRETRAIN_STEPS = 2000


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def optimizer_for(method: str, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    if STAGES[method] != "ABC":
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    time_parameters = []
    shared_parameters = []
    for name, parameter in model.named_parameters():
        if any(token in name for token in ("time_", "duration_head", "elapsed_head")):
            time_parameters.append(parameter)
        else:
            shared_parameters.append(parameter)
    # Keep the trajectory/success path conservative while the clock branch
    # learns its residual correction from real 0--8.5 s event gaps.
    return torch.optim.AdamW(
        [
            {"params": shared_parameters, "lr": base_lr * 0.25},
            {"params": time_parameters, "lr": base_lr * 0.5},
        ],
        weight_decay=1e-4,
    )


def train_once(
    *,
    method: str,
    seed: int,
    dataset: BoundaryDataset,
    train_indices: np.ndarray,
    device: torch.device,
    steps: int,
    pretrain_steps: int,
    batch_size: int,
    base_lr: float,
    output_dir: Path,
    snapshot_every: int,
    save_every_step: bool,
) -> tuple[Path, list[dict], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = model_for_method(method).to(device)
    target_model = copy.deepcopy(model).to(device).eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    optimizer = optimizer_for(method, model, base_lr)
    tensor_data = dataset.to_tensors(device)
    rank_sampler = None
    example_sampler = BalancedExampleSampler(dataset, train_indices)
    if STAGES[method] != "A":
        rank_sampler = MatchedRankSampler(dataset, train_indices)
    teacher = None
    trace: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = output_dir / "snapshots"
    if save_every_step or snapshot_every > 0:
        snapshots.mkdir(parents=True, exist_ok=True)
    # Step zero is retained for exact optimizer-horizon auditing.
    save_checkpoint(
        snapshots / "step_000000.pt",
        model,
        method,
        seed,
        0,
        extra={"phase": "initial", "train_examples": int(len(train_indices))},
        half=save_every_step,
    )
    started = time.perf_counter()
    last_components: dict[str, float] = {}
    for step in range(1, steps + 1):
        if step == pretrain_steps + 1 and STAGES[method] != "A":
            teacher = copy.deepcopy(model).to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        batch_np = example_sampler.sample(batch_size, rng)
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        model.train()
        loss, components = compute_training_loss(
            method,
            model,
            target_model,
            tensor_data,
            batch,
            rank_sampler,
            rng,
            ranking_enabled=STAGES[method] != "A" and step > pretrain_steps,
            teacher=teacher,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        update_ema(target_model, model)
        last_components = components
        row = {
            "step": step,
            "phase": "rank_finetune" if step > pretrain_steps and STAGES[method] != "A" else "event_pretrain",
            "loss": float(loss.detach()),
            "cfc_required": STAGES[method] == "ABC",
        }
        row.update({f"loss_{name}": value for name, value in components.items()})
        trace.append(row)
        should_save = save_every_step or step == steps or (
            snapshot_every > 0 and step % snapshot_every == 0
        )
        if should_save:
            save_checkpoint(
                snapshots / f"step_{step:06d}.pt",
                model,
                method,
                seed,
                step,
                extra={"phase": row["phase"], "loss": row["loss"]},
                half=save_every_step and step != steps,
            )
        if step % 100 == 0 or step == 1:
            print(
                f"[TRAIN] method={method} seed={seed} step={step}/{steps} "
                f"phase={row['phase']} loss={row['loss']:.6f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    formal_checkpoint = output_dir / "step_3000.pt"
    if steps != FORMAL_STEPS:
        formal_checkpoint = output_dir / f"step_{steps}.pt"
    save_checkpoint(
        formal_checkpoint,
        model,
        method,
        seed,
        steps,
        extra={
            "phase": "formal_step_3000" if steps == FORMAL_STEPS else "nonformal",
            "wall_seconds": elapsed,
            "last_components": last_components,
        },
        half=False,
    )
    write_csv(output_dir / "training_trace.csv", trace)
    return formal_checkpoint, trace, {
        "wall_seconds": elapsed,
        "last_components": last_components,
        "parameters": parameter_report(model),
    }


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path)
    return loaded["mean"], loaded["std"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=FORMAL_STEPS)
    parser.add_argument("--pretrain-steps", type=int, default=PRETRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lobo-snapshot-every", type=int, default=100)
    parser.add_argument("--formal-snapshot-every", type=int, default=1)
    parser.add_argument("--expected-data-sha256", default=FROZEN_DATA_MANIFEST_SHA256)
    parser.add_argument("--expected-stats-sha256", default=FROZEN_STATS_SHA256)
    parser.add_argument("--allow-nonformal-steps", action="store_true")
    args = parser.parse_args()

    if args.steps != FORMAL_STEPS and not args.allow_nonformal_steps:
        raise SystemExit(f"formal protocol requires exactly {FORMAL_STEPS} steps")
    if not (0 <= args.pretrain_steps <= args.steps):
        raise SystemExit("pretrain steps must be inside the training horizon")
    data_sha = logical_data_manifest_sha256(args.data_root)
    stats_sha = file_sha256(args.stats)
    if data_sha != args.expected_data_sha256:
        raise SystemExit(f"DATA HASH MISMATCH: {data_sha} != {args.expected_data_sha256}")
    if stats_sha != args.expected_stats_sha256:
        raise SystemExit(f"STATS HASH MISMATCH: {stats_sha} != {args.expected_stats_sha256}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rollouts = collect_rollouts(args.data_root)
    protocol = validate_frozen_protocol(rollouts)
    mean, std = load_stats(args.stats)
    dataset = build_boundary_dataset(rollouts, mean, std)
    source_rollout_ids = [
        i for i, rollout in enumerate(rollouts) if rollout.body in SOURCE_BODIES
    ]
    target_rollout_ids = [
        i for i, rollout in enumerate(rollouts) if rollout.body in TARGET_BODIES
    ]
    source_indices = dataset.indices_for_rollouts(source_rollout_ids)
    if set(dataset.arrays["rollout_id"][source_indices]) & set(target_rollout_ids):
        raise AssertionError("target leakage into source training indices")
    run_root = args.output / args.method / f"seed{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[DATA] {protocol} boundary_examples={dataset.size} "
        f"source_examples={len(source_indices)} data_sha={data_sha}",
        flush=True,
    )

    fold_rows: list[dict] = []
    oof_predictions: list[dict] = []
    fold_records: list[dict] = []
    for fold_index, holdout_body in enumerate(SOURCE_BODIES):
        train_body = SOURCE_BODIES[1 - fold_index]
        train_rollout_ids = [
            i for i, rollout in enumerate(rollouts) if rollout.body == train_body
        ]
        adapt_ids, test_ids = split_adapt_test_rollouts(rollouts, [holdout_body])
        train_indices = dataset.indices_for_rollouts(train_rollout_ids)
        fold_dir = run_root / "lobo" / f"holdout_{holdout_body}"
        checkpoint, _, training_info = train_once(
            method=args.method,
            seed=args.seed,
            dataset=dataset,
            train_indices=train_indices,
            device=device,
            steps=args.steps,
            pretrain_steps=args.pretrain_steps,
            batch_size=args.batch_size,
            base_lr=args.lr,
            output_dir=fold_dir,
            snapshot_every=args.lobo_snapshot_every,
            save_every_step=False,
        )
        from experiment2_fixed_core_v2 import load_checkpoint_model

        model, _ = load_checkpoint_model(checkpoint, device)
        metrics = evaluate_model(
            args.method,
            model,
            dataset,
            dataset.to_tensors(device),
            test_ids,
            adapt_rollout_ids=adapt_ids,
            calibration=None,
            return_predictions=True,
        )
        predictions = metrics.pop("predictions")
        oof_predictions.extend(predictions)
        row = {
            "fold": holdout_body,
            "train_body": train_body,
            "step": args.steps,
            **{name: metrics[name] for name in (
                "bellman_mse_target", "ranking_auc", "sign_consistency",
                "success_rate_mae", "success_brier"
            )},
        }
        fold_rows.append(row)
        fold_records.append(
            {
                "holdout_body": holdout_body,
                "train_body": train_body,
                "checkpoint": str(checkpoint),
                "adapt_rollouts": len(adapt_ids),
                "test_rollouts": len(test_ids),
                "metrics": metrics,
                "training": training_info,
            }
        )
        print(f"[LOBO] {json.dumps(row, ensure_ascii=False)}", flush=True)

    calibration = fit_success_calibration(args.method, oof_predictions)
    aggregate = aggregate_fold_metrics(fold_rows)
    # The formal comparison is frozen at step 3000; LOBO is calibration and
    # audit only, never an early-stopping path.
    selection_audit = {
        "formal_step": args.steps,
        "early_stopping": False,
        "reason": "all horizontal and ablation models are fixed to 3000 optimizer steps",
        "source_lobo_step3000": aggregate[args.steps],
    }

    formal_dir = run_root / "formal"
    formal_checkpoint, _, formal_training = train_once(
        method=args.method,
        seed=args.seed,
        dataset=dataset,
        train_indices=source_indices,
        device=device,
        steps=args.steps,
        pretrain_steps=args.pretrain_steps,
        batch_size=args.batch_size,
        base_lr=args.lr,
        output_dir=formal_dir,
        snapshot_every=args.formal_snapshot_every,
        save_every_step=args.formal_snapshot_every == 1,
    )
    info = {
        "format": FORMAT,
        "method": args.method,
        "stage": STAGES[args.method],
        "seed": args.seed,
        "formal_steps": args.steps,
        "pretrain_steps": args.pretrain_steps,
        "rank_finetune_steps": max(args.steps - args.pretrain_steps, 0)
        if STAGES[args.method] != "A" else 0,
        "formal_checkpoint": str(formal_checkpoint),
        "calibration": calibration,
        "selection": selection_audit,
        "folds": fold_records,
        "formal_training": formal_training,
        "data": {
            **protocol,
            "manifest_sha256": data_sha,
            "stats_sha256": stats_sha,
            "source_bodies": list(SOURCE_BODIES),
            "target_bodies": list(TARGET_BODIES),
            "target_used_for_training_or_selection": False,
        },
        "snapshots": {
            "formal_every_step": args.formal_snapshot_every == 1,
            "formal_count_expected": args.steps + 1 if args.formal_snapshot_every == 1 else None,
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "pid": os.getpid(),
        },
    }
    json_dump(run_root / "run_info.json", info)
    write_csv(run_root / "source_lobo_step3000.csv", fold_rows)
    print(
        f"[DONE] method={args.method} seed={args.seed} formal={formal_checkpoint} "
        f"calibration={calibration}",
        flush=True,
    )


if __name__ == "__main__":
    main()
