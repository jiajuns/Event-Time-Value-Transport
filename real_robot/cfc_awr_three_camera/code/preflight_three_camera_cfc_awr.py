#!/usr/bin/env python3
"""Fail-closed preflight for three-camera SmolVLA + original-CfC AWR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CAMERAS = {
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_frames(root: Path, columns: list[str]) -> pd.DataFrame:
    paths = sorted((root / "data").rglob("*.parquet"))
    require(bool(paths), f"no parquet data in {root}")
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths]).sort_values("index").reset_index(drop=True)


def task_text(root: Path) -> str:
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    require(len(tasks) == 1, f"expected one task in {root}")
    if "task" in tasks.columns:
        return str(tasks.iloc[0]["task"])
    return str(tasks.index[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--actor-data", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--expected-observer-sha", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(args.source.parent.name == "050000", "source is not checkpoint 050000")
    source_sha = sha256(args.source / "model.safetensors")
    require(source_sha == args.expected_source_sha, "three-camera source model SHA256 mismatch")
    require(sha256(args.observer) == args.expected_observer_sha, "original CfC observer SHA256 mismatch")

    cfg = json.loads((args.source / "config.json").read_text())
    require(cfg["type"] == "smolvla", "source is not SmolVLA")
    require(cfg["chunk_size"] == cfg["n_action_steps"] == 50, "chunk contract changed")
    require(cfg["output_features"]["action"]["shape"] == [25], "action dimension is not 25")
    inputs = cfg["input_features"]
    visual = {key for key, value in inputs.items() if value["type"] == "VISUAL"}
    require(visual == CAMERAS, f"source is not the exact three-camera model: {sorted(visual)}")
    require(inputs["observation.state"]["shape"] == [25], "state dimension is not 25")

    info = json.loads((args.actor_data / "meta/info.json").read_text())
    require(info["total_episodes"] == args.expected_episodes, "actor episode count mismatch")
    require(info["total_frames"] == args.expected_frames, "actor frame count mismatch")
    dataset_visual = {key for key in info["features"] if key.startswith("observation.images.") and key.endswith("_rgb")}
    require(dataset_visual == CAMERAS, f"dataset is not exact three-camera data: {sorted(dataset_visual)}")
    require(task_text(args.actor_data) == args.task, "actor task prompt mismatch")

    columns = ["index", "episode_index", "frame_index", "observation.state", "action"]
    actor = read_frames(args.actor_data, columns)
    reference = read_frames(args.reference_data, columns)
    require(len(actor) == len(reference) == args.expected_frames, "actor/reference frame count mismatch")
    for key in ("index", "episode_index", "frame_index"):
        require(np.array_equal(actor[key], reference[key]), f"three-camera/reference identity mismatch: {key}")
    for key in ("observation.state", "action"):
        require(np.array_equal(np.stack(actor[key]), np.stack(reference[key])), f"three-camera/reference values mismatch: {key}")

    sidecar_path = args.scores / "event_weights.parquet"
    sidecar = pd.read_parquet(sidecar_path)
    require(len(sidecar) == args.expected_frames, "CFC-AWR sidecar frame count mismatch")
    for key in ("index", "episode_index", "frame_index"):
        require(np.array_equal(actor[key], sidecar[key]), f"sidecar identity mismatch: {key}")
    weights = sidecar["event_weight"].to_numpy()
    require(np.isfinite(weights).all(), "non-finite AWR weight")
    require(weights.min() >= 0.9 - 1e-6 and weights.max() <= 1.1 + 1e-6, "AWR weight outside [0.9, 1.1]")
    require(weights.std() > 1e-7, "AWR weights are uniform")
    for episode, part in sidecar.groupby("episode_index"):
        endpoint = np.minimum(part["index"].to_numpy() + 50, int(part["index"].max()))
        require(np.array_equal(part["endpoint_index"], endpoint), f"chunk endpoint crosses episode {episode}")

    score_receipt = json.loads((args.scores / "receipt.json").read_text())
    require(score_receipt["sidecar_sha256"] == sha256(sidecar_path), "sidecar receipt mismatch")
    require(score_receipt.get("observer_sha256") == args.expected_observer_sha, "score did not use requested original CfC")

    result = {
        "status": "passed",
        "method": "original_cfc_awr_three_camera",
        "task_text": args.task,
        "episodes": args.expected_episodes,
        "frames": args.expected_frames,
        "source_model_sha256": source_sha,
        "observer_sha256": args.expected_observer_sha,
        "sidecar_sha256": sha256(sidecar_path),
        "cameras": sorted(CAMERAS),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "fresh_optimizer": True,
        "additional_steps": 10000,
        "deployment_requires_critic": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
