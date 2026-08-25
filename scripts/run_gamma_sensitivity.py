"""Pre-authorized gamma sensitivity check for the discounted TD critic."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_stage0_experiment as stage0


def main() -> None:
    feature_root = Path("/home/user/etsf_stage0/stage0/features")
    output_root = Path("/home/user/etsf_stage0/stage0/results")
    manifest = json.loads((feature_root / "manifest.json").read_text())
    with np.load(output_root / "feature_normalization.npz") as normalization:
        mean = normalization["mean"]
        std = normalization["std"]
    train_episodes, eval_episodes = stage0.load_episodes(
        feature_root, manifest, mean, std
    )
    task_count = len(manifest["tasks"])
    visual_dim = train_episodes[0].features.shape[1]
    input_dim = visual_dim + task_count + 1
    device = torch.device("cuda:0")
    datasets = {
        speed: stage0.flatten_resampled(train_episodes, speed, "frame")
        for speed in stage0.TRAIN_ORDER
    }

    rows: list[dict[str, float | int]] = []
    for gamma in [0.95, 0.995]:
        stage0.GAMMA = gamma
        predictions: dict[tuple[float, int], list[np.ndarray]] = {}
        baseline_states: dict[int, dict[str, torch.Tensor]] = {}
        for speed in stage0.TRAIN_ORDER:
            for seed in stage0.SEEDS:
                head, loss = stage0.train_head(
                    datasets[speed],
                    "discounted_td",
                    seed,
                    seed + int(speed * 1000),
                    task_count,
                    800,
                    512,
                    device,
                )
                prediction = stage0.predict_episodes(
                    head,
                    eval_episodes,
                    "discounted_td",
                    task_count,
                    device,
                )
                predictions[(speed, seed)] = prediction
                if speed == 1.0:
                    baseline_states[seed] = copy.deepcopy(head.state_dict())
                row = {
                    "gamma": gamma,
                    "speed": speed,
                    "seed": seed,
                    "final_train_loss": loss,
                }
                row.update(stage0.rank_metrics(prediction))
                rows.append(row)
                del head

        for row in rows:
            if row["gamma"] != gamma:
                continue
            absolute, signed = stage0.paired_metrics(
                predictions[(row["speed"], row["seed"])],
                predictions[(1.0, row["seed"])],
            )
            row["paired_abs_diff"] = absolute
            row["paired_signed_diff"] = signed

    sensitivity = pd.DataFrame(rows)
    base = pd.read_csv(output_root / "stage0_raw_metrics.csv")
    base = base[
        (base["method"] == "frame") & (base["model"] == "discounted_td")
    ][
        [
            "gamma",
            "speed",
            "seed",
            "final_train_loss",
            "voc_spearman",
            "kendall_tau",
            "v_s0_mean",
            "dynamic_range_mean",
            "advantage_sign_consistency",
            "paired_abs_diff",
            "paired_signed_diff",
        ]
    ]
    sensitivity = pd.concat([sensitivity, base], ignore_index=True).sort_values(
        ["gamma", "speed", "seed"]
    )
    sensitivity.to_csv(output_root / "stage0_gamma_sensitivity.csv", index=False)

    summary = (
        sensitivity.groupby(["gamma", "speed"])[
            ["voc_spearman", "paired_abs_diff"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "gamma",
        "speed",
        "voc_mean",
        "voc_std",
        "paired_abs_diff_mean",
        "paired_abs_diff_std",
    ]
    summary.to_csv(
        output_root / "stage0_gamma_sensitivity_summary.csv", index=False
    )
    for gamma, part in summary.groupby("gamma"):
        print(
            "GAMMA_RESULT="
            + json.dumps(
                {
                    "gamma": gamma,
                    "voc_span": float(part.voc_mean.max() - part.voc_mean.min()),
                    "voc_min": float(part.voc_mean.min()),
                    "max_paired_abs_diff": float(part.paired_abs_diff_mean.max()),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
