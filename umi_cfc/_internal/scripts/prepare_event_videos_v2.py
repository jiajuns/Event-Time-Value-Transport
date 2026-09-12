#!/usr/bin/env python3
"""Synchronized CFR videos -> sampled RGB images + blank interval annotation CSV.

No event labels are inferred. Explicit capture FPS is required; variable-rate
or unsynchronized videos must first be converted with truthful timestamps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.factorized_events import read_observations, write_annotation_template
from event_rl.semantic_cues import feature_names, sha256


def prepare(source: Path, output: Path, stride: int = 3) -> dict:
    spec = json.loads(source.read_text(encoding="utf-8"))
    cameras = spec["cameras"]
    feature_names(cameras)
    if spec.get("synchronized_cfr") is not True or stride <= 0:
        raise ValueError("explicit synchronized_cfr=true and positive stride required")
    if output.exists():
        raise ValueError("refusing to overwrite output directory")
    if not spec["episodes"]:
        raise ValueError("no episodes")
    identities, group_splits, source_groups, prepared = set(), {}, {}, []
    for episode in spec["episodes"]:
        uid, group, split = episode["attempt_uid"], episode["group_uid"], episode["split"]
        fps = float(episode["fps"])
        if (not isinstance(uid, str) or not uid or uid in identities or not isinstance(group, str) or
                not group or split not in ("train", "validation", "test", "inference") or
                not np.isfinite(fps) or fps <= 0):
            raise ValueError("invalid episode/FPS/split")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("source group crosses splits")
        identities.add(uid)
        group_splits[group] = split
        paths, hashes = [], []
        for camera in cameras:
            path = (source.parent / episode["videos"][camera]).resolve()
            if not path.is_file():
                raise ValueError(f"missing video: {path}")
            digest = sha256(path)
            if digest in source_groups and source_groups[digest] != group:
                raise ValueError("same video assigned to different source groups")
            source_groups[digest] = group
            paths.append(path)
            hashes.append(digest)
        prepared.append((episode, paths, hashes))
    output.mkdir(parents=True)
    observations, receipts = [], []
    for number, (episode, paths, hashes) in enumerate(prepared):
        captures = [cv2.VideoCapture(str(path)) for path in paths]
        try:
            fps = float(episode["fps"])
            if any(not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps, rtol=1e-3)
                   for cap in captures):
                raise ValueError("video decoder FPS does not match declared synchronized capture FPS")
            counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
            if min(counts) <= 0 or len(set(counts)) != 1:
                raise ValueError("synchronized videos need equal positive frame counts")
            directory = output / "frames" / f"episode_{number:06d}"
            directory.mkdir(parents=True)
            for frame in range(counts[0]):
                images = [cap.read() for cap in captures]
                if any(not ok for ok, _ in images):
                    raise ValueError("video decoding failed; partial output is not a dataset")
                if frame % stride:
                    continue
                timestamp, paths_by_camera = frame / fps, {}
                for camera, (_, image) in zip(cameras, images):
                    # Camera names are metadata, not filesystem path components.
                    destination = directory / f"frame_{frame:08d}_view_{cameras.index(camera):02d}.png"
                    if not cv2.imwrite(str(destination), image):
                        raise ValueError("failed to write decoded frame")
                    paths_by_camera[camera] = dict(path=str(destination.relative_to(output)), elapsed_s=timestamp)
                observations.append(dict(attempt_uid=episode["attempt_uid"], group_uid=episode["group_uid"],
                                         split=episode["split"], query_id=frame, elapsed_s=timestamp,
                                         images=paths_by_camera))
            receipts.append(dict(attempt_uid=episode["attempt_uid"], videos_sha256=hashes, fps=fps, frames=counts[0]))
        finally:
            for cap in captures:
                cap.release()
    manifest = output / "observations.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in observations), encoding="utf-8")
    read_observations(manifest)
    write_annotation_template(output / "annotations.csv", observations)
    receipt = dict(format="eksf_event_video_preparation_v2", source_sha256=sha256(source),
                   cameras=cameras, stride=stride, frames=len(observations), episodes=receipts,
                   annotation_status="blank_not_reviewed", event_labels_generated=False)
    (output / "preparation_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(prepare(args.videos, args.output, args.stride), ensure_ascii=False, indent=2))
