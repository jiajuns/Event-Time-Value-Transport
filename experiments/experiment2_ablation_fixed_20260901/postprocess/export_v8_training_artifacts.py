#!/usr/bin/env python3
"""Export complete V8 training traces, parameter updates, tables, and figures."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SEEDS = tuple(range(20260901, 20260906))
RUNS = (
    ("A", "e2_base_gru_fixed", "results/train_v5/e2_base_gru_fixed"),
    ("A+B", "e2_gru_event_fixed", "results/train_v5/e2_gru_event_fixed"),
    ("A+B+C", "e2_cfc_event_fixed", "results/train_v8/e2_cfc_event_fixed"),
)
PHASE_BOUNDARY = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_loss_tables(exp: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    formal = []
    lobo = []
    for stage, method, relative_root in RUNS:
        root = exp / relative_root
        for seed in SEEDS:
            seed_root = root / f"seed{seed}"
            trace = pd.read_csv(seed_root / "formal/training_trace.csv")
            if len(trace) != 3000 or trace["step"].tolist() != list(range(1, 3001)):
                raise RuntimeError(f"invalid formal trace: {seed_root}")
            trace.insert(0, "seed", seed)
            trace.insert(0, "method", method)
            trace.insert(0, "stage", stage)
            trace.insert(3, "scope", "formal")
            formal.append(trace)
            for lobo_trace in sorted((seed_root / "lobo").glob("holdout_*/training_trace.csv")):
                fold = pd.read_csv(lobo_trace)
                if len(fold) != 3000:
                    raise RuntimeError(f"invalid LOBO trace: {lobo_trace}")
                fold.insert(0, "holdout_body", lobo_trace.parent.name.removeprefix("holdout_"))
                fold.insert(0, "seed", seed)
                fold.insert(0, "method", method)
                fold.insert(0, "stage", stage)
                fold.insert(4, "scope", "source_lobo")
                lobo.append(fold)
    return pd.concat(formal, ignore_index=True), pd.concat(lobo, ignore_index=True)


def loss_long(frame: pd.DataFrame) -> pd.DataFrame:
    identity = [name for name in ("stage", "method", "seed", "scope", "holdout_body", "step", "phase") if name in frame]
    loss_columns = [name for name in frame if name == "loss" or name.startswith("loss_")]
    return frame.melt(
        id_vars=identity,
        value_vars=loss_columns,
        var_name="loss_name",
        value_name="value",
    ).dropna(subset=["value"])


def summarize_losses(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in long.groupby(["stage", "method", "seed", "loss_name"], sort=True):
        ordered = group.sort_values("step")
        first = ordered[ordered["step"] <= 100]["value"]
        last = ordered[ordered["step"] > 2900]["value"]
        rows.append({
            "stage": keys[0],
            "method": keys[1],
            "seed": keys[2],
            "loss_name": keys[3],
            "first100_mean": first.mean(),
            "last100_mean": last.mean(),
            "last_over_first": last.mean() / max(first.mean(), 1e-12),
            "minimum": ordered["value"].min(),
            "maximum": ordered["value"].max(),
            "final_step_value": ordered.iloc[-1]["value"],
        })
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["stage", "method", "loss_name"], sort=True).agg(
        seeds=("seed", "count"),
        first100_mean=("first100_mean", "mean"),
        first100_std=("first100_mean", "std"),
        last100_mean=("last100_mean", "mean"),
        last100_std=("last100_mean", "std"),
        last_over_first_mean=("last_over_first", "mean"),
        minimum_mean=("minimum", "mean"),
        maximum_mean=("maximum", "mean"),
        final_step_mean=("final_step_value", "mean"),
        final_step_std=("final_step_value", "std"),
    ).reset_index()
    return per_seed, summary


def parameter_names(exp: Path, method: str) -> set[str]:
    sys.path.insert(0, str(exp / "scripts"))
    from experiment2_fixed_core_v2 import model_for_method  # noqa: PLC0415

    return {name for name, _ in model_for_method(method).named_parameters()}


def tensor_group(name: str) -> str:
    return name.split(".", 1)[0]


def state_parameters(payload: dict, allowed: set[str]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().float().cpu()
        for name, value in payload["model"].items()
        if name in allowed
    }


def parameter_update_tables(exp: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectory_rows = []
    tensor_rows = []
    group_final_rows = []
    for stage, method, relative_root in RUNS:
        allowed = parameter_names(exp, method)
        for seed in SEEDS:
            formal = exp / relative_root / f"seed{seed}/formal"
            initial_payload = torch.load(
                formal / "snapshots/step_000000.pt", map_location="cpu", weights_only=False
            )
            initial = state_parameters(initial_payload, allowed)
            previous = initial
            group_names = sorted({tensor_group(name) for name in initial})
            for step in range(0, 3001):
                path = formal / "snapshots" / f"step_{step:06d}.pt"
                payload = initial_payload if step == 0 else torch.load(
                    path, map_location="cpu", weights_only=False
                )
                current = initial if step == 0 else state_parameters(payload, allowed)
                for group in group_names:
                    names = [name for name in current if tensor_group(name) == group]
                    step_sq = sum(float((current[name] - previous[name]).square().sum()) for name in names)
                    cumulative_sq = sum(float((current[name] - initial[name]).square().sum()) for name in names)
                    norm_sq = sum(float(current[name].square().sum()) for name in names)
                    trajectory_rows.append({
                        "stage": stage,
                        "method": method,
                        "seed": seed,
                        "step": step,
                        "parameter_group": group,
                        "step_delta_l2": math.sqrt(step_sq),
                        "cumulative_delta_l2": math.sqrt(cumulative_sq),
                        "parameter_l2": math.sqrt(norm_sq),
                    })
                previous = current
            final = current
            for name in sorted(final):
                before = initial[name]
                after = final[name]
                initial_norm = float(before.norm())
                delta = float((after - before).norm())
                tensor_rows.append({
                    "stage": stage,
                    "method": method,
                    "seed": seed,
                    "parameter_group": tensor_group(name),
                    "tensor_name": name,
                    "numel": before.numel(),
                    "initial_l2": initial_norm,
                    "final_l2": float(after.norm()),
                    "delta_l2": delta,
                    "relative_delta_l2": delta / max(initial_norm, 1e-12),
                })
            for group in group_names:
                names = [name for name in final if tensor_group(name) == group]
                initial_sq = sum(float(initial[name].square().sum()) for name in names)
                final_sq = sum(float(final[name].square().sum()) for name in names)
                delta_sq = sum(float((final[name] - initial[name]).square().sum()) for name in names)
                group_final_rows.append({
                    "stage": stage,
                    "method": method,
                    "seed": seed,
                    "parameter_group": group,
                    "parameter_tensors": len(names),
                    "numel": sum(initial[name].numel() for name in names),
                    "initial_l2": math.sqrt(initial_sq),
                    "final_l2": math.sqrt(final_sq),
                    "delta_l2": math.sqrt(delta_sq),
                    "relative_delta_l2": math.sqrt(delta_sq) / max(math.sqrt(initial_sq), 1e-12),
                })
    return pd.DataFrame(trajectory_rows), pd.DataFrame(tensor_rows), pd.DataFrame(group_final_rows)


def plot_loss_panels(long: pd.DataFrame, figures: Path) -> None:
    for stage, method, _ in RUNS:
        subset = long[long["method"] == method]
        names = sorted(subset["loss_name"].unique(), key=lambda value: (value != "loss", value))
        columns = 3
        rows = math.ceil(len(names) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows), squeeze=False)
        for axis, loss_name in zip(axes.flat, names):
            loss = subset[subset["loss_name"] == loss_name]
            pivot = loss.pivot(index="step", columns="seed", values="value").sort_index()
            smooth = pivot.rolling(50, min_periods=1).mean()
            mean = smooth.mean(axis=1)
            std = smooth.std(axis=1).fillna(0.0)
            for seed in smooth:
                axis.plot(smooth.index, smooth[seed], alpha=0.25, linewidth=0.7)
            axis.plot(mean.index, mean, color="black", linewidth=1.4, label="5-seed mean")
            axis.fill_between(mean.index, mean - std, mean + std, color="black", alpha=0.12)
            axis.axvline(PHASE_BOUNDARY, color="tab:red", linestyle="--", linewidth=0.8)
            axis.set_title(loss_name)
            axis.set_xlabel("optimizer step")
            axis.grid(alpha=0.2)
        for axis in axes.flat[len(names):]:
            axis.axis("off")
        fig.suptitle(f"{stage} formal losses (50-step rolling mean; all seeds)", fontsize=14)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{stage.replace('+', 'plus')}_formal_losses.{suffix}", dpi=220)
        plt.close(fig)


def plot_total_loss_comparison(long: pd.DataFrame, figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    total = long[long["loss_name"] == "loss"]
    for stage, method, _ in RUNS:
        pivot = total[total["method"] == method].pivot(
            index="step", columns="seed", values="value"
        ).sort_index().rolling(50, min_periods=1).mean()
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1).fillna(0.0)
        axis.plot(mean.index, mean, linewidth=1.8, label=stage)
        axis.fill_between(mean.index, mean - std, mean + std, alpha=0.12)
    axis.axvline(PHASE_BOUNDARY, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("total loss")
    axis.set_title("Formal total training loss (5-seed mean ± std)")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"formal_total_loss_comparison.{suffix}", dpi=220)
    plt.close(fig)


def plot_parameter_updates(trajectory: pd.DataFrame, group_final: pd.DataFrame, figures: Path) -> None:
    for stage, method, _ in RUNS:
        subset = trajectory[trajectory["method"] == method]
        fig, axis = plt.subplots(figsize=(11, 6))
        for group, values in subset.groupby("parameter_group"):
            pivot = values.pivot(index="step", columns="seed", values="cumulative_delta_l2")
            axis.plot(pivot.index, pivot.mean(axis=1), linewidth=1.2, label=group)
        axis.axvline(PHASE_BOUNDARY, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("L2(parameter_step - parameter_0)")
        axis.set_title(f"{stage} cumulative parameter updates")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{stage.replace('+', 'plus')}_parameter_updates.{suffix}", dpi=220)
        plt.close(fig)

    summary = group_final.groupby(["stage", "parameter_group"], sort=True)["relative_delta_l2"].mean().reset_index()
    stages = [item[0] for item in RUNS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), squeeze=False)
    for axis, stage in zip(axes.flat, stages):
        values = summary[summary["stage"] == stage].sort_values("relative_delta_l2")
        axis.barh(values["parameter_group"], values["relative_delta_l2"])
        axis.set_title(stage)
        axis.set_xlabel("mean relative L2 update")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Initial-to-final parameter update by group")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"parameter_group_final_updates.{suffix}", dpi=220)
    plt.close(fig)


def plot_metrics(exp: Path, figures: Path) -> None:
    raw = pd.read_csv(exp / "results/final_v8/ablation_raw_metrics.csv")
    metrics = ["bellman_mse_target", "ranking_auc", "sign_consistency", "success_rate_mae"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), squeeze=False)
    order = [item[1] for item in RUNS]
    labels = [item[0] for item in RUNS]
    for axis, metric in zip(axes.flat, metrics):
        means = []
        stds = []
        for method in order:
            values = pd.to_numeric(raw.loc[raw["method"] == method, metric], errors="coerce").dropna()
            means.append(values.mean() if len(values) else np.nan)
            stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axis.bar(labels, means, yerr=stds, capsize=4)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("V8 ablation metrics (5-seed mean ± std)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"ablation_metrics.{suffix}", dpi=220)
    plt.close(fig)


def inventory(exp: Path) -> pd.DataFrame:
    rows = []
    for stage, method, relative_root in RUNS:
        for seed in SEEDS:
            formal = exp / relative_root / f"seed{seed}/formal"
            snapshots = sorted((formal / "snapshots").glob("step_*.pt"))
            final = formal / "step_3000.pt"
            rows.append({
                "stage": stage,
                "method": method,
                "seed": seed,
                "formal_trace_rows": len(pd.read_csv(formal / "training_trace.csv")),
                "formal_snapshot_count": len(snapshots),
                "formal_checkpoint": str(final),
                "formal_checkpoint_bytes": final.stat().st_size,
                "formal_checkpoint_sha256": sha256(final),
                "lobo_trace_count": len(list((formal.parent / "lobo").glob("holdout_*/training_trace.csv"))),
            })
    return pd.DataFrame(rows)


def write_readme(output: Path) -> None:
    text = """# V8 plot-ready training package

This directory is post-processing output.  It does not modify training code or checkpoints.

## Tables

- `tables/formal_training_trace_wide.csv`: every formal loss value, 15 runs × 3000 steps.
- `tables/formal_training_loss_long.csv`: plot-ready long form of all formal losses.
- `tables/lobo_training_trace_wide.csv`: every source-LOBO training trace.
- `tables/lobo_training_loss_long.csv`: plot-ready long form of source-LOBO losses.
- `tables/loss_summary_per_seed.csv`: first/last windows, extrema, and final value per seed/loss.
- `tables/loss_summary_5seed.csv`: 5-seed aggregate loss summary.
- `tables/parameter_update_by_step_long.csv`: every formal step's parameter-group update norm.
- `tables/parameter_tensor_initial_final.csv`: initial/final/delta norm for every parameter tensor.
- `tables/parameter_group_initial_final.csv`: initial/final/delta norm by parameter group.
- `tables/ablation_raw_metrics.csv` and `tables/ablation_summary.csv`: frozen final metrics.
- `tables/horizontal_summary.csv`: frozen horizontal comparison.
- `tables/checkpoint_inventory.csv`: counts, sizes, locations, and SHA-256 for all 15 final runs.

## Figures

PNG and vector PDF versions are provided for all loss panels, cumulative parameter updates,
final parameter-group changes, total-loss comparison, and final ablation metrics.  Loss plots
show every seed plus 5-seed mean/std with a marker at the step-2000 phase transition.

`MANIFEST_SHA256.txt` hashes every generated file except the manifest itself.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_manifest(output: Path) -> None:
    manifest = output / "MANIFEST_SHA256.txt"
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path != manifest)
    with manifest.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256(path)}  {path.relative_to(output)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment_root.resolve()
    output = args.output.resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    formal, lobo = load_loss_tables(exp)
    formal_long = loss_long(formal)
    lobo_long = loss_long(lobo)
    per_seed, summary = summarize_losses(formal_long)
    save_frame(formal, tables / "formal_training_trace_wide.csv")
    save_frame(formal_long, tables / "formal_training_loss_long.csv")
    save_frame(lobo, tables / "lobo_training_trace_wide.csv")
    save_frame(lobo_long, tables / "lobo_training_loss_long.csv")
    save_frame(per_seed, tables / "loss_summary_per_seed.csv")
    save_frame(summary, tables / "loss_summary_5seed.csv")

    trajectory, tensor_final, group_final = parameter_update_tables(exp)
    save_frame(trajectory, tables / "parameter_update_by_step_long.csv")
    save_frame(tensor_final, tables / "parameter_tensor_initial_final.csv")
    save_frame(group_final, tables / "parameter_group_initial_final.csv")
    save_frame(inventory(exp), tables / "checkpoint_inventory.csv")

    copies = {
        "ablation_raw_metrics.csv": exp / "results/final_v8/ablation_raw_metrics.csv",
        "ablation_summary.csv": exp / "results/final_v8/ablation_summary.csv",
        "horizontal_summary.csv": exp / "results/final_v8/horizontal_with_same_abc_summary.csv",
    }
    for name, source in copies.items():
        save_frame(pd.read_csv(source), tables / name)

    plot_loss_panels(formal_long, figures)
    plot_total_loss_comparison(formal_long, figures)
    plot_parameter_updates(trajectory, group_final, figures)
    plot_metrics(exp, figures)
    write_readme(output)
    write_manifest(output)
    print(f"PLOT_READY_V8_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
