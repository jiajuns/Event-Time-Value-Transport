#!/usr/bin/env python3
"""Build causal pair-image and relative-geometry cache for relational v4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.relational_visual import (CausalPairVisualState, PAIR_FEATURE_NAMES, PAIR_NAMES,
                                        VISUAL_VERSION, crop_pair_images)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    videos = value.get("source_videos") if isinstance(value, dict) else value if isinstance(value, list) else None
    if not isinstance(videos, list):
        raise ValueError("provenance must contain source_videos")
    return value, videos


def build(data: Path, graph: Path, provenance: Path, raw: Path, output: Path,
          pair_size=96, margin=.15, max_history_s=.3):
    if output.exists():
        raise ValueError("refusing to overwrite visual cache")
    if pair_size <= 0 or not np.isfinite([margin, max_history_s]).all() or margin < 0 or max_history_s < 0:
        raise ValueError("invalid visual cache configuration")
    with np.load(data, allow_pickle=False) as source:
        arrays = {key: source[key] for key in ("attempt_uid", "query_id", "elapsed_s", "split")}
    if "test" in set(arrays["split"].astype(str)):
        raise ValueError("visual cache builder refuses test observations")
    with np.load(graph, allow_pickle=False) as cached:
        graph_contract = json.loads(str(cached["contract_json"].item()))
        graph_arrays = {key: cached[key] for key in ("attempt_uid", "query_id", "elapsed_s",
                        "node_features", "node_mask", "node_boxes", "bindings")}
    for key in ("attempt_uid", "query_id", "elapsed_s"):
        if not np.array_equal(arrays[key], graph_arrays[key]):
            raise ValueError(f"graph/data {key} mismatch")
    if graph_contract.get("graph_version") != "umi_relational_graph_v3":
        raise ValueError("unsupported graph version")
    provenance_value, videos = _provenance(provenance)
    video_by_uid = {str(video["attempt_uid"]): video for video in videos}
    if len(video_by_uid) != len(videos) or set(arrays["attempt_uid"].astype(str)) != set(video_by_uid):
        raise ValueError("data/provenance attempt identities differ")
    uid_values = arrays["attempt_uid"].astype(str)
    for uid, video in video_by_uid.items():
        if video.get("split") == "test" or set(arrays["split"][uid_values == uid].astype(str)) != {str(video.get("split"))}:
            raise ValueError("invalid provenance split")
        path = raw / video["file"]
        if not path.is_file() or sha256(path) != video["sha256"]:
            raise ValueError(f"raw video hash mismatch: {uid}")

    pair_images, pair_features, output_indices = [], [], []
    for uid in dict.fromkeys(uid_values):
        indices = np.flatnonzero(uid_values == uid)
        indices = indices[np.argsort(arrays["query_id"][indices])]
        queries, times = arrays["query_id"][indices], arrays["elapsed_s"][indices]
        if np.any(queries < 0) or np.any(np.diff(queries) <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError("invalid episode chronology")
        if not np.allclose(times, queries / 30.0, atol=1e-6):
            raise ValueError("query/time does not match 30 FPS source")
        capture = cv2.VideoCapture(str(raw / video_by_uid[uid]["file"]))
        if not capture.isOpened():
            raise ValueError(f"cannot open raw video: {uid}")
        state = CausalPairVisualState(max_history_s=max_history_s)
        decoded_query = -1
        for index in indices:
            query = int(arrays["query_id"][index])
            while decoded_query < query:
                ok, frame = capture.read(); decoded_query += 1
                if not ok:
                    break
            if not ok:
                raise ValueError(f"cannot decode query frame: {uid}/{query}")
            features, crops = state.update(arrays["elapsed_s"][index], graph_arrays["node_features"][index],
                                           graph_arrays["node_mask"][index], graph_arrays["node_boxes"][index],
                                           graph_arrays["bindings"][index])
            pair_features.append(features)
            pair_images.append(crop_pair_images(frame, crops, pair_size, margin))
            output_indices.append(int(index))
        capture.release()
    order = np.argsort(np.asarray(output_indices))
    if not np.array_equal(np.asarray(output_indices)[order], np.arange(len(uid_values))):
        raise ValueError("visual construction did not cover source rows exactly once")
    contract = dict(format=VISUAL_VERSION, visual_version=VISUAL_VERSION,
                    pair_names=list(PAIR_NAMES), pair_feature_names=list(PAIR_FEATURE_NAMES),
                    pair_feature_dim=len(PAIR_FEATURE_NAMES), pair_size=pair_size, crop_margin=margin,
                    max_history_s=max_history_s, causal=True, future_interpolation=False,
                    fallback="full current frame when either role is absent or ambiguous; fallback is not an unknown relation",
                    relative_delta="change in role-relative screen center; common camera translation cancels algebraically",
                    screen_motion_is_physical_motion=False, labels_used_as_inputs=False,
                    actions_used=False, uid_used_as_network_input=False,
                    source_data_sha256=sha256(data), source_graph_sha256=sha256(graph),
                    provenance_sha256=sha256(provenance), raw_root=str(raw.resolve()),
                    provenance_format=provenance_value.get("format") if isinstance(provenance_value, dict) else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, attempt_uid=arrays["attempt_uid"], query_id=arrays["query_id"],
                            elapsed_s=arrays["elapsed_s"], pair_images_uint8=np.stack(pair_images)[order],
                            pair_features=np.stack(pair_features)[order],
                            contract_json=np.asarray(json.dumps(contract, ensure_ascii=False)))
    return {"observations": len(uid_values), "episodes": len(video_by_uid), **contract}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=.15)
    parser.add_argument("--max-history-s", type=float, default=.3)
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.graph, args.provenance, args.raw, args.output,
                           args.pair_size, args.margin, args.max_history_s), ensure_ascii=False, indent=2))
