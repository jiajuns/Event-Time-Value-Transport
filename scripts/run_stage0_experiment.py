"""Stage-0 synthetic temporal reparameterization study.

Pre-registered qualitative expectations from ETSF_agent_runbook.md (written
before looking at results):

    model                    VOC vs c       paired |V_c - V_1| vs c
    normalized_progress      flat           flat / near zero
    discounted_td            flat           increasing
    remaining_steps          flat           approximately linear increasing
    frame_index              about 1, flat  linear increasing
    random_head              about 0         not applicable

Variant A separately trains V_c and evaluates every head on the exact same
base-speed evaluation frames. Variant B trains at c=1 and evaluates Bellman
residuals after temporal reparameterization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from torch import nn


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
TRAIN_ORDER = [1.0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0]
METHODS = ["frame", "interpolation"]
MODELS = [
    "normalized_progress",
    "discounted_td",
    "remaining_steps",
    "frame_index",
    "random_head",
]
MODEL_LABELS = {
    "normalized_progress": "1 归一化进度",
    "discounted_td": "2 折扣 TD critic",
    "remaining_steps": "3 剩余步数回归",
    "frame_index": "4 帧序号回归",
    "random_head": "5 随机头",
}
SEEDS = [17, 29, 43]
GAMMA = 0.99
TARGET_STEP_SCALE = 400.0
POSITION_SCALE = 400.0
HIDDEN_DIM = 256


@dataclass
class Episode:
    task_id: int
    task_name: str
    episode_id: int
    features: np.ndarray


@dataclass
class FlatDataset:
    features: torch.Tensor
    task_ids: torch.Tensor
    frame_indices: torch.Tensor
    remaining_steps: torch.Tensor
    progress: torch.Tensor
    segments: list[tuple[int, int]]

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


class ValueHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def feature_mean_std(
    feature_root: Path, manifest: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, int]:
    total = 0
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    for task, task_info in manifest["tasks"].items():
        for episode_id in task_info["train_episode_ids"]:
            path = feature_root / task / f"episode_{episode_id:06d}.npz"
            with np.load(path) as data:
                features = data["features"].astype(np.float64)
            if sum_x is None:
                sum_x = np.zeros(features.shape[1], dtype=np.float64)
                sum_x2 = np.zeros(features.shape[1], dtype=np.float64)
            sum_x += features.sum(axis=0)
            sum_x2 += np.square(features).sum(axis=0)
            total += features.shape[0]
    assert sum_x is not None and sum_x2 is not None
    mean = sum_x / total
    variance = np.maximum(sum_x2 / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), total


def load_episodes(
    feature_root: Path,
    manifest: dict[str, object],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[list[Episode], list[Episode]]:
    train: list[Episode] = []
    evaluation: list[Episode] = []
    for task_id, (task, task_info) in enumerate(manifest["tasks"].items()):
        train_ids = set(task_info["train_episode_ids"])
        eval_ids = set(task_info["eval_episode_ids"])
        for episode_id in sorted(train_ids | eval_ids):
            path = feature_root / task / f"episode_{episode_id:06d}.npz"
            with np.load(path) as data:
                raw = data["features"].astype(np.float32)
            standardized = ((raw - mean) / std).astype(np.float16)
            episode = Episode(task_id, task, episode_id, standardized)
            (train if episode_id in train_ids else evaluation).append(episode)
    return train, evaluation


def time_positions(length: int, speed: float) -> np.ndarray:
    if length < 2:
        return np.array([0.0], dtype=np.float32)
    positions = np.arange(0.0, length - 1 + 1e-7, speed, dtype=np.float32)
    if positions[-1] < length - 1 - 1e-6:
        positions = np.append(positions, np.float32(length - 1))
    else:
        positions[-1] = np.float32(length - 1)
    return positions


def resample_features(
    features: np.ndarray, speed: float, method: str
) -> np.ndarray:
    positions = time_positions(features.shape[0], speed)
    if method == "frame":
        indices = np.floor(positions + 1e-7).astype(np.int64)
        return features[indices]
    if method == "interpolation":
        left = np.floor(positions).astype(np.int64)
        right = np.minimum(left + 1, features.shape[0] - 1)
        alpha = (positions - left).astype(np.float32)[:, None]
        output = (
            features[left].astype(np.float32) * (1.0 - alpha)
            + features[right].astype(np.float32) * alpha
        )
        return output.astype(np.float16)
    raise ValueError(method)


def flatten_resampled(
    episodes: Iterable[Episode], speed: float, method: str
) -> FlatDataset:
    all_features: list[np.ndarray] = []
    all_tasks: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    all_remaining: list[np.ndarray] = []
    all_progress: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    offset = 0
    for episode in episodes:
        features = resample_features(episode.features, speed, method)
        length = features.shape[0]
        indices = np.arange(length, dtype=np.float32)
        remaining = (length - 1 - indices).astype(np.float32)
        progress = indices / max(length - 1, 1)
        all_features.append(features)
        all_tasks.append(np.full(length, episode.task_id, dtype=np.int64))
        all_indices.append(indices)
        all_remaining.append(remaining)
        all_progress.append(progress.astype(np.float32))
        segments.append((offset, offset + length))
        offset += length
    return FlatDataset(
        features=torch.from_numpy(np.concatenate(all_features, axis=0)),
        task_ids=torch.from_numpy(np.concatenate(all_tasks, axis=0)),
        frame_indices=torch.from_numpy(np.concatenate(all_indices, axis=0)),
        remaining_steps=torch.from_numpy(np.concatenate(all_remaining, axis=0)),
        progress=torch.from_numpy(np.concatenate(all_progress, axis=0)),
        segments=segments,
    )


def make_input(
    features: torch.Tensor,
    task_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    model_name: str,
    task_count: int,
) -> torch.Tensor:
    batch = features.shape[0]
    visual_dim = features.shape[1]
    x = torch.zeros(
        (batch, visual_dim + task_count + 1),
        device=features.device,
        dtype=torch.float32,
    )
    if model_name != "frame_index":
        x[:, :visual_dim] = features.float()
    x[torch.arange(batch, device=x.device), visual_dim + task_ids.long()] = 1.0
    if model_name == "frame_index":
        x[:, -1] = frame_indices.float() / POSITION_SCALE
    return x


def prediction_target(dataset: FlatDataset, model_name: str) -> torch.Tensor:
    if model_name in {"normalized_progress", "frame_index"}:
        return dataset.progress
    if model_name == "discounted_td":
        # For deterministic successful trajectories with a sparse terminal reward,
        # gamma**remaining is the exact fixed point of the one-step TD equation.
        return torch.pow(torch.tensor(GAMMA), dataset.remaining_steps)
    if model_name == "remaining_steps":
        return -dataset.remaining_steps / TARGET_STEP_SCALE
    raise ValueError(model_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_head(
    dataset: FlatDataset,
    model_name: str,
    seed: int,
    sampling_seed: int,
    task_count: int,
    train_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[ValueHead, float]:
    seed_everything(seed)
    input_dim = dataset.features.shape[1] + task_count + 1
    head = ValueHead(input_dim).to(device)
    if model_name == "random_head":
        return head.eval(), math.nan

    features = dataset.features.to(device=device, non_blocking=True)
    task_ids = dataset.task_ids.to(device=device, non_blocking=True)
    frame_indices = dataset.frame_indices.to(device=device, non_blocking=True)
    targets = prediction_target(dataset, model_name).to(
        device=device, non_blocking=True
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling_seed)
    final_loss = math.nan
    head.train()
    for _ in range(train_steps):
        selection = torch.randint(
            dataset.size,
            (batch_size,),
            generator=generator,
            device=device,
        )
        x = make_input(
            features[selection],
            task_ids[selection],
            frame_indices[selection],
            model_name,
            task_count,
        )
        prediction = head(x)
        loss = torch.mean(torch.square(prediction - targets[selection]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    del features, task_ids, frame_indices, targets, optimizer
    return head.eval(), final_loss


def decode_prediction(prediction: np.ndarray, model_name: str) -> np.ndarray:
    if model_name == "remaining_steps":
        return prediction * TARGET_STEP_SCALE
    return prediction


def predict_episodes(
    head: ValueHead,
    episodes: list[Episode],
    model_name: str,
    task_count: int,
    device: torch.device,
    speed: float = 1.0,
    method: str = "frame",
    batch_size: int = 4096,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for episode in episodes:
            features_np = resample_features(episode.features, speed, method)
            frame_indices_np = np.arange(features_np.shape[0], dtype=np.float32)
            pieces: list[np.ndarray] = []
            for start in range(0, features_np.shape[0], batch_size):
                end = min(start + batch_size, features_np.shape[0])
                features = torch.from_numpy(features_np[start:end]).to(device)
                tasks = torch.full(
                    (end - start,), episode.task_id, device=device, dtype=torch.long
                )
                indices = torch.from_numpy(frame_indices_np[start:end]).to(device)
                x = make_input(features, tasks, indices, model_name, task_count)
                pieces.append(head(x).float().cpu().numpy())
            prediction = np.concatenate(pieces)
            outputs.append(decode_prediction(prediction, model_name))
    return outputs


def rank_metrics(predictions: list[np.ndarray]) -> dict[str, float]:
    spearman_values = []
    kendall_values = []
    starts = []
    ranges = []
    advantage_sign = []
    for values in predictions:
        indices = np.arange(values.shape[0])
        rho = spearmanr(indices, values).statistic
        tau = kendalltau(indices, values).statistic
        spearman_values.append(float(rho) if np.isfinite(rho) else 0.0)
        kendall_values.append(float(tau) if np.isfinite(tau) else 0.0)
        starts.append(float(values[0]))
        ranges.append(float(values.max() - values.min()))
        advantage_sign.append(float(np.mean(np.diff(values) > 0)))
    return {
        "voc_spearman": float(np.mean(spearman_values)),
        "kendall_tau": float(np.mean(kendall_values)),
        "v_s0_mean": float(np.mean(starts)),
        "dynamic_range_mean": float(np.mean(ranges)),
        "monotonic_increase_rate": float(np.mean(advantage_sign)),
    }


def paired_metrics(
    predictions: list[np.ndarray], baseline: list[np.ndarray]
) -> tuple[float, float]:
    differences = np.concatenate(
        [current - reference for current, reference in zip(predictions, baseline)]
    )
    return float(np.mean(np.abs(differences))), float(np.mean(differences))


def paired_advantage_sign_consistency(
    predictions: list[np.ndarray], baseline: list[np.ndarray]
) -> float:
    """Compare temporal-advantage signs on identical base transitions.

    Here the trajectory-local temporal advantage is V(s_{t+1}) - V(s_t).
    Unlike the monotonic-increase rate, this metric asks whether V_c and V_1
    would assign the same sign to every paired evaluation transition.
    """
    agreements = []
    for current, reference in zip(predictions, baseline):
        current_sign = np.sign(np.diff(current))
        reference_sign = np.sign(np.diff(reference))
        agreements.append(current_sign == reference_sign)
    return float(np.mean(np.concatenate(agreements)))


def bellman_error(predictions: list[np.ndarray]) -> float:
    residuals: list[np.ndarray] = []
    for values in predictions:
        if values.shape[0] > 1:
            residuals.append(np.abs(GAMMA * values[1:] - values[:-1]))
        residuals.append(np.array([abs(1.0 - float(values[-1]))]))
    return float(np.mean(np.concatenate(residuals)))


def dataframe_with_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "voc_spearman",
        "kendall_tau",
        "paired_abs_diff",
        "paired_signed_diff",
        "v_s0_mean",
        "dynamic_range_mean",
        "monotonic_increase_rate",
        "advantage_sign_consistency",
        "paired_abs_diff_over_baseline_range",
        "variant_b_td_error",
        "final_train_loss",
    ]
    grouped = raw.groupby(["method", "model", "model_label", "speed"], dropna=False)
    rows = []
    for keys, group in grouped:
        row = dict(zip(["method", "model", "model_label", "speed"], keys))
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "model", "speed"])


def plot_main(aggregate: pd.DataFrame, method: str, output_path: Path) -> None:
    frame = aggregate[aggregate["method"] == method]
    fig, left_axis = plt.subplots(figsize=(11.5, 7.0), dpi=180)
    right_axis = left_axis.twinx()
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    for model_name, color in zip(MODELS, colors):
        part = frame[frame["model"] == model_name].sort_values("speed")
        left_axis.errorbar(
            part["speed"],
            part["voc_spearman_mean"],
            yerr=part["voc_spearman_std"],
            color=color,
            marker="o",
            linewidth=1.8,
            capsize=2,
            label=f"{MODEL_LABELS[model_name]} — VOC",
        )
        if model_name != "random_head":
            right_axis.errorbar(
                part["speed"],
                part["paired_abs_diff_mean"],
                yerr=part["paired_abs_diff_std"],
                color=color,
                marker="s",
                linestyle="--",
                linewidth=1.5,
                alpha=0.82,
                capsize=2,
                label=f"{MODEL_LABELS[model_name]} — 配对差",
            )
    left_axis.set_xscale("log", base=2)
    left_axis.set_xticks(SPEEDS, [str(speed) for speed in SPEEDS])
    left_axis.set_xlabel("时间速度 c（对数刻度）")
    left_axis.set_ylabel("VOC / Spearman ρ")
    right_axis.set_yscale("symlog", linthresh=0.01)
    right_axis.set_ylim(0.0, 400.0)
    right_axis.set_ylabel("同一评测帧上的平均 |V_c(x) − V_1(x)|（对称对数）")
    left_axis.set_title("指标看不见的失真")
    left_axis.grid(True, which="both", alpha=0.25)
    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    fig.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8,
        ncol=3,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    started = time.time()
    feature_root = Path("/home/user/etsf_stage0/stage0/features")
    output_root = Path("/home/user/etsf_stage0/stage0/results")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((feature_root / "manifest.json").read_text())
    mean, std, normalization_samples = feature_mean_std(feature_root, manifest)
    np.savez(output_root / "feature_normalization.npz", mean=mean, std=std)
    train_episodes, eval_episodes = load_episodes(
        feature_root, manifest, mean, std
    )
    task_count = len(manifest["tasks"])
    visual_dim = train_episodes[0].features.shape[1]
    input_dim = visual_dim + task_count + 1
    parameter_count = sum(p.numel() for p in ValueHead(input_dim).parameters())
    device = torch.device("cuda:0")

    rows: list[dict[str, object]] = []
    prediction_store: dict[
        tuple[str, str, float, int], list[np.ndarray]
    ] = {}
    baseline_states: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}

    for method_index, method in enumerate(METHODS):
        datasets = {
            speed: flatten_resampled(train_episodes, speed, method)
            for speed in TRAIN_ORDER
        }
        for speed in TRAIN_ORDER:
            dataset = datasets[speed]
            for model_name in MODELS:
                for seed in SEEDS:
                    sampling_seed = seed + method_index * 100_000 + int(speed * 1000)
                    head, loss = train_head(
                        dataset,
                        model_name,
                        seed,
                        sampling_seed,
                        task_count,
                        args.train_steps,
                        args.batch_size,
                        device,
                    )
                    predictions = predict_episodes(
                        head,
                        eval_episodes,
                        model_name,
                        task_count,
                        device,
                    )
                    key = (method, model_name, speed, seed)
                    prediction_store[key] = predictions
                    if speed == 1.0:
                        baseline_states[(method, model_name, seed)] = {
                            name: value.detach().cpu().clone()
                            for name, value in head.state_dict().items()
                        }
                    row: dict[str, object] = {
                        "method": method,
                        "variant": "A_train_time_distortion",
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "speed": speed,
                        "seed": seed,
                        "gamma": GAMMA,
                        "train_steps": 0 if model_name == "random_head" else args.train_steps,
                        "batch_size": args.batch_size,
                        "training_samples": (
                            0
                            if model_name == "random_head"
                            else args.train_steps * args.batch_size
                        ),
                        "available_resampled_states": dataset.size,
                        "head_parameters": parameter_count,
                        "final_train_loss": loss,
                    }
                    row.update(rank_metrics(predictions))
                    rows.append(row)
                    print(
                        "TRAIN_RESULT="
                        + json.dumps(
                            {
                                "method": method,
                                "model": model_name,
                                "speed": speed,
                                "seed": seed,
                                "loss": loss,
                                "voc": row["voc_spearman"],
                                "available_states": dataset.size,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    del head
            torch.cuda.empty_cache()
        del datasets

    for row in rows:
        key = (row["method"], row["model"], row["speed"], row["seed"])
        baseline_key = (row["method"], row["model"], 1.0, row["seed"])
        if row["model"] == "random_head":
            row["paired_abs_diff"] = math.nan
            row["paired_signed_diff"] = math.nan
            row["advantage_sign_consistency"] = math.nan
            row["paired_abs_diff_over_baseline_range"] = math.nan
        else:
            absolute, signed = paired_metrics(
                prediction_store[key], prediction_store[baseline_key]
            )
            row["paired_abs_diff"] = absolute
            row["paired_signed_diff"] = signed
            row["advantage_sign_consistency"] = paired_advantage_sign_consistency(
                prediction_store[key], prediction_store[baseline_key]
            )

    baseline_ranges = {
        (row["method"], row["model"], row["seed"]): row["dynamic_range_mean"]
        for row in rows
        if row["speed"] == 1.0 and row["model"] != "random_head"
    }
    for row in rows:
        if row["model"] != "random_head":
            row["paired_abs_diff_over_baseline_range"] = row[
                "paired_abs_diff"
            ] / baseline_ranges[(row["method"], row["model"], row["seed"])]

    # Variant B: c=1 heads are fixed; only the evaluation successor relation is
    # reparameterized. The value at a shared sampled frame is therefore unchanged
    # for image-based heads, while the one-step Bellman residual can change.
    variant_b: dict[tuple[str, str, float, int], float] = {}
    for method in METHODS:
        for model_name in MODELS:
            for seed in SEEDS:
                seed_everything(seed)
                head = ValueHead(input_dim).to(device)
                head.load_state_dict(baseline_states[(method, model_name, seed)])
                head.eval()
                for speed in SPEEDS:
                    predictions = predict_episodes(
                        head,
                        eval_episodes,
                        model_name,
                        task_count,
                        device,
                        speed=speed,
                        method=method,
                    )
                    variant_b[(method, model_name, speed, seed)] = bellman_error(
                        predictions
                    )
                del head

    for row in rows:
        row["variant_b_td_error"] = variant_b[
            (row["method"], row["model"], row["speed"], row["seed"])
        ]

    raw = pd.DataFrame(rows).sort_values(["method", "model", "speed", "seed"])
    raw_path = output_root / "stage0_raw_metrics.csv"
    raw.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)
    aggregate = dataframe_with_aggregates(raw)
    aggregate_path = output_root / "stage0_aggregate_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)
    plot_main(aggregate, "frame", output_root / "stage0_main_plot.png")
    plot_main(
        aggregate,
        "interpolation",
        output_root / "stage0_interpolation_check.png",
    )

    run_config = {
        "speeds": SPEEDS,
        "methods": METHODS,
        "models": MODELS,
        "seeds": SEEDS,
        "gamma": GAMMA,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "training_samples_per_trained_head": args.train_steps * args.batch_size,
        "head": f"MLP {input_dim}->{HIDDEN_DIM}->{HIDDEN_DIM}->1, GELU",
        "head_parameters": parameter_count,
        "visual_dim": visual_dim,
        "task_count": task_count,
        "normalization_train_frames": normalization_samples,
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "elapsed_seconds": time.time() - started,
        "variant_a_fixed_eval_frames": True,
        "model4_operationalization": (
            "The identical-capacity MLP receives task one-hot and raw frame index/400; "
            "all visual dimensions are zero. It regresses j/(T_c-1) in the c-speed "
            "training sequence and is evaluated at the fixed base sequence index."
        ),
        "discounted_td_operationalization": (
            "gamma**remaining, the exact fixed point of the one-step Bellman equation "
            "for deterministic successful trajectories with sparse terminal reward 1"
        ),
    }
    (output_root / "stage0_run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print("STAGE0_COMPLETE=" + json.dumps(run_config, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
