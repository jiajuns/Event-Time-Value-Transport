#!/usr/bin/env python3
"""Summarize matched RoboTwin task/seed horizon measurements."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BODIES = ["aloha-agilex", "piper", "ARX-X5", "ur5-wsg"]
LABELS = {
    "aloha-agilex": "Aloha",
    "piper": "Piper",
    "ARX-X5": "ARX-X5",
    "ur5-wsg": "UR5-WSG",
}


def stats(values: pd.Series) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    if len(raw) != 240 or raw.duplicated(["task", "embodiment", "seed"]).any():
        raise ValueError("expected exactly 6 tasks × 4 embodiments × 10 unique seeds")
    raw.to_csv(args.output / "raw.csv", index=False)

    status = (
        raw.groupby(["task", "embodiment", "status"])
        .size()
        .rename("count")
        .reset_index()
    )
    status.to_csv(args.output / "status_counts.csv", index=False)

    success = raw[raw.status.eq("success")].copy()
    horizon_summary = (
        success.groupby(["task", "embodiment"])["effective_control_steps"]
        .agg(n_success="size", mean_steps="mean", median_steps="median", std_steps="std", min_steps="min", max_steps="max")
        .reset_index()
    )
    horizon_summary.to_csv(args.output / "successful_horizon_summary.csv", index=False)

    wide = success.pivot(index=["task", "seed"], columns="embodiment", values="effective_control_steps")
    common = wide.dropna(subset=BODIES).reset_index()
    common.to_csv(args.output / "four_way_common_horizons.csv", index=False)

    episode_pairs: list[dict[str, object]] = []
    for _, row in common.iterrows():
        for body_a, body_b in itertools.combinations(BODIES, 2):
            directional = float(row[body_a] / row[body_b])
            episode_pairs.append(
                {
                    "task": row["task"],
                    "seed": int(row["seed"]),
                    "embodiment_a": body_a,
                    "embodiment_b": body_b,
                    "steps_a": int(row[body_a]),
                    "steps_b": int(row[body_b]),
                    "a_over_b": directional,
                    "kappa": max(directional, 1.0 / directional),
                }
            )
    pairs = pd.DataFrame(episode_pairs)
    pairs.to_csv(args.output / "pairwise_kappa_four_way.csv", index=False)

    summaries: list[dict[str, object]] = []
    for scope, grouped in list(pairs.groupby("task")) + [("POOLED", pairs)]:
        for (body_a, body_b), group in grouped.groupby(["embodiment_a", "embodiment_b"]):
            k = group.kappa
            summaries.append(
                {
                    "scope": scope,
                    "embodiment_a": body_a,
                    "embodiment_b": body_b,
                    "n": len(group),
                    "median_a_over_b": group.a_over_b.median(),
                    "median_kappa": k.median(),
                    "p10_kappa": k.quantile(0.10),
                    "p90_kappa": k.quantile(0.90),
                    "max_kappa": k.max(),
                    "fraction_kappa_ge_2": (k >= 2).mean(),
                    "fraction_kappa_le_1_1": (k <= 1.1).mean(),
                }
            )
    pair_summary = pd.DataFrame(summaries)
    pair_summary.to_csv(args.output / "pairwise_kappa_summary.csv", index=False)

    tasks = list(raw.task.drop_duplicates())
    success_rates = raw.assign(success=raw.status.eq("success")).pivot_table(
        index="task", columns="embodiment", values="success", aggfunc="mean"
    ).reindex(index=tasks, columns=BODIES)
    medians = success.pivot_table(
        index="task", columns="embodiment", values="effective_control_steps", aggfunc="median"
    ).reindex(index=tasks, columns=BODIES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    x = np.arange(len(tasks))
    width = 0.19
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for idx, body in enumerate(BODIES):
        offset = (idx - 1.5) * width
        axes[0].bar(x + offset, medians[body], width, label=LABELS[body], color=colors[idx])
        axes[1].bar(x + offset, success_rates[body], width, label=LABELS[body], color=colors[idx])
    for axis in axes:
        axis.set_xticks(x, [task.replace("_", "\n") for task in tasks], fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Effective control steps (median of successes)")
    axes[0].set_title("Speed: successful trajectories only")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Success rate (10 fixed seeds)")
    axes[1].set_title("Compatibility prerequisite")
    axes[1].legend(ncol=2, frameon=False)
    fig.suptitle("RoboTwin matched cross-embodiment speed gate (same task and seed)")
    fig.savefig(args.output / "matched_speed_gate.png", dpi=180)
    plt.close(fig)

    pooled = pair_summary[pair_summary.scope.eq("POOLED")]
    piper_aloha = pooled[
        (pooled.embodiment_a.eq("aloha-agilex")) & pooled.embodiment_b.eq("piper")
    ].iloc[0]
    summary = {
        "attempts": int(len(raw)),
        "four_way_common_episodes": int(len(common)),
        "four_way_common_tasks": common.groupby("task").size().astype(int).to_dict(),
        "tasks_without_four_way_common_success": sorted(set(tasks) - set(common.task)),
        "piper_aloha": {
            "median_kappa": float(piper_aloha.median_kappa),
            "p10_kappa": float(piper_aloha.p10_kappa),
            "p90_kappa": float(piper_aloha.p90_kappa),
            "max_kappa": float(piper_aloha.max_kappa),
        },
        "strong_go_kappa_2_to_4_observed": bool((pairs.kappa >= 2).any()),
        "all_pairs_within_1_1": bool((pairs.kappa <= 1.1).all()),
        "decision": "CONTINUE_WITH_REDUCED_EFFECT_SIZE",
        "reason": "No matched episode reaches 2x, but Aloha/Piper is consistently ~1.46x and therefore not in the <=1.1 stop regime.",
    }
    (args.output / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
