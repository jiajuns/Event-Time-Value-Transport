#!/usr/bin/env python3
"""Extract causal frozen RGB-CfC features from the aligned rl_fruit head view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

TASK = "put the fruit to the plate"
EXPECTED_OBSERVER_SHA = "0b9638af3dd39a39d2f6fe61c6f671c08176e4316f424c1233d4a59f86bb8b89"


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("data", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite extracted CfC features")

    from event_rl.factorized_observer import load_observer, CausalWindows, model_inputs
    from event_rl.factorized_events import HEADS
    from event_rl.semantic_cues import sha256

    if sha256(args.checkpoint) != EXPECTED_OBSERVER_SHA:
        raise ValueError("wrong original frozen CfC checkpoint")
    model, saved = load_observer(args.checkpoint, args.device)
    if saved["model_spec"]["use_cues"] or saved["config"]["window"] != 24:
        raise ValueError("expected RGB-only 24-step CfC observer")
    manifest_path = args.data / "event_dataset_contract.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["format"] != "s1_event_finetune_dataset_v1" or manifest["role"] != "observer" or
            manifest["task_text"] != TASK or manifest["total_episodes"] != 52 or manifest["total_frames"] != 9649):
        raise ValueError("wrong rl_fruit observer dataset contract")
    tasks = pd.read_parquet(args.data / "meta/tasks.parquet")
    text = tasks.task.iloc[0] if "task" in tasks.columns else tasks.index[0]
    if len(tasks) != 1 or int(tasks.task_index.iloc[0]) != 0 or text != TASK:
        raise ValueError("task text mismatch")
    frames = pd.concat([pd.read_parquet(p) for p in sorted((args.data / "data").rglob("*.parquet"))])
    frames = frames.sort_values("index").reset_index(drop=True)
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((args.data / "meta/episodes").rglob("*.parquet"))])
    episodes = episodes.sort_values("episode_index")
    times = pd.read_parquet(args.data / "capture_timestamps.parquet")
    if len(frames) != 9649 or len(episodes) != 52 or not np.array_equal(frames["index"], np.arange(9649)):
        raise ValueError("wrong rl_fruit data size")
    for key in ("index", "episode_index", "frame_index"):
        if not np.array_equal(frames[key], times[key]):
            raise ValueError(f"capture timestamp identity mismatch: {key}")
    states = np.stack(frames["observation.state"]).astype(np.float32)
    if states.shape != (9649, 25) or not np.isfinite(states).all():
        raise ValueError("invalid state25")
    source_hashes = {str(p.relative_to(args.data)): sha256(p)
                     for p in sorted((args.data / "data").rglob("*.parquet"))}
    for path in (args.data / "capture_timestamps.parquet", args.data / "conversion_mapping.json",
                 args.data / "meta/info.json", args.data / "meta/tasks.parquet", manifest_path):
        source_hashes[str(path.relative_to(args.data))] = sha256(path)
    camera = "observation.images.base_0_rgb"
    all_features = []
    for _, episode_row in episodes.iterrows():
        eid, length = int(episode_row.episode_index), int(episode_row.length)
        begin, end = int(episode_row.dataset_from_index), int(episode_row.dataset_to_index)
        part = frames.iloc[begin:end]
        if length != end - begin or not np.all(part.episode_index == eid) or not np.array_equal(part.frame_index, np.arange(length)):
            raise ValueError("episode boundary mismatch")
        prefix = "videos/" + camera
        video = args.data / prefix / f"chunk-{int(episode_row[prefix + '/chunk_index']):03d}" / f"file-{int(episode_row[prefix + '/file_index']):03d}.mp4"
        source_hashes[str(video.relative_to(args.data))] = sha256(video)
        capture = cv2.VideoCapture(str(video))
        offset = float(episode_row[prefix + "/from_timestamp"]) * 30
        if not capture.isOpened() or abs(offset - round(offset)) > 1e-3 or abs(capture.get(cv2.CAP_PROP_FPS) - 30) > .001:
            raise ValueError("invalid observer video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(offset))
        images, anchors = [], []
        for frame_index in range(length):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"video decode failed episode={eid} frame={frame_index}")
            if frame_index % 3 == 0:
                rgb = cv2.cvtColor(cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                images.append(np.moveaxis(rgb, -1, 0)[None])
                anchors.append(begin + frame_index)
        capture.release()
        count = len(anchors)
        arrays = dict(images_uint8=np.stack(images), semantic_features=np.zeros((count, 34), np.float32),
                      labels=np.full((count, 4), -1, np.int64), attempt_uid=np.full(count, str(eid)),
                      query_id=np.arange(count), elapsed_s=times.iloc[anchors].elapsed_s.to_numpy())
        pieces = []
        with torch.inference_mode():
            for batch in DataLoader(CausalWindows(arrays, 24), batch_size=16):
                result = model(**model_inputs(batch, args.device), return_features=True)
                pieces.append(torch.cat([result["history_features"],
                                         *[result[key].softmax(-1) for key in HEADS]], -1).cpu().numpy())
        sampled = np.concatenate(pieces)
        all_features.append(sampled[np.arange(length) // 3])
        print(json.dumps(dict(stage="extract_original_cfc", episode=eid, frames=length)), flush=True)
    features = np.concatenate(all_features).astype(np.float32)
    if features.shape != (9649, 111) or not np.isfinite(features).all():
        raise ValueError(f"invalid extracted feature shape {features.shape}")
    contract = dict(format="rl_fruit_original_cfc_features_v1", checkpoint_sha256=sha256(args.checkpoint),
                    source_hashes=source_hashes, video_stride=3, window=24,
                    feature_names="96 CfC hidden plus 15 event-head probabilities",
                    observer_camera=camera, future_inputs=False, state_in_shared_encoder=False,
                    raw_action_inputs=False, sample_hold="floor(frame/3), within episode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, states=states,
                        index=frames["index"].to_numpy(), episode=frames.episode_index.to_numpy(),
                        frame=frames.frame_index.to_numpy(), time=times.elapsed_s.to_numpy(),
                        contract_json=json.dumps(contract))
    print(json.dumps(dict(stage="extract_original_cfc_complete", path=str(args.output),
                          sha256=sha256(args.output), shape=list(features.shape))), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
