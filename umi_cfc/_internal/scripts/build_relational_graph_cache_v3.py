#!/usr/bin/env python3
"""Build a causal YOLO relational-graph cache from origin videos only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.relational_graph import (CausalGraphTracker, EDGE_FEATURES, GRAPH_VERSION,
                                       NODE_FEATURES, ROLES)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_provenance(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    videos = value.get("source_videos") if isinstance(value, dict) else value if isinstance(value, list) else None
    if not isinstance(videos, list):
        raise ValueError("provenance must contain source_videos")
    return value, videos


def _detections(path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(row["attempt_uid"]), int(row["query_id"]))
            if key in rows:
                raise ValueError("duplicate detection identity")
            rows[key] = row
    return rows


def _crop(frame, box, size):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, int(np.floor(x1 * width))), max(0, int(np.floor(y1 * height)))
    right, bottom = min(width, int(np.ceil(x2 * width))), min(height, int(np.ceil(y2 * height)))
    if right <= left or bottom <= top:
        return np.zeros((3, size, size), np.uint8)
    roi = cv2.resize(frame[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA)
    return np.moveaxis(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), -1, 0)


def build(data: Path, detections: Path, provenance: Path, raw: Path, output: Path,
          max_nodes=8, roi_size=64):
    if output.exists():
        raise ValueError("refusing to overwrite graph cache")
    with np.load(data, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("attempt_uid", "query_id", "elapsed_s", "split")}
    if "test" in set(arrays["split"].astype(str)):
        raise ValueError("graph cache builder refuses datasets containing test observations")
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("data identity arrays differ in length")
    provenance_value, videos = _load_provenance(provenance)
    video_by_uid = {str(v["attempt_uid"]): v for v in videos}
    if len(video_by_uid) != len(videos) or set(arrays["attempt_uid"].astype(str)) != set(video_by_uid):
        raise ValueError("data/provenance attempt identities differ")
    for uid, video in video_by_uid.items():
        if video.get("split") == "test":
            raise ValueError("provenance contains test video")
        uid_splits = set(arrays["split"][arrays["attempt_uid"].astype(str) == uid].astype(str))
        if uid_splits != {str(video.get("split"))}:
            raise ValueError("data/provenance split mismatch")
        path = raw / video["file"]
        if not path.is_file() or sha256(path) != video["sha256"]:
            raise ValueError(f"raw video hash mismatch: {uid}")
    detected = _detections(detections)
    expected = set(zip(arrays["attempt_uid"].astype(str), arrays["query_id"].astype(int)))
    if set(detected) != expected:
        raise ValueError("detections do not exactly cover data identities")

    outputs = {"node_features": [], "node_mask": [], "node_boxes": [], "node_track_ids": [],
               "edge_features": [], "bindings": [], "node_images_uint8": [], "graph_truncated": []}
    output_indices = []
    for uid in dict.fromkeys(arrays["attempt_uid"].astype(str)):
        indices = np.flatnonzero(arrays["attempt_uid"].astype(str) == uid)
        indices = indices[np.argsort(arrays["query_id"][indices])]
        queries = arrays["query_id"][indices]
        times = arrays["elapsed_s"][indices]
        if np.any(queries < 0) or np.any(np.diff(queries) <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError("invalid episode chronology")
        if not np.allclose(times, queries / 30.0, atol=1e-6):
            raise ValueError("query/time does not match 30 FPS source")
        capture = cv2.VideoCapture(str(raw / video_by_uid[uid]["file"]))
        if not capture.isOpened():
            raise ValueError(f"cannot open raw video: {uid}")
        tracker = CausalGraphTracker(max_nodes=max_nodes)
        decoded_query = -1
        prior_time = None
        for index in indices:
            query = int(arrays["query_id"][index]); row = detected[(uid, query)]
            if abs(float(row["elapsed_s"]) - float(arrays["elapsed_s"][index])) > 1e-6:
                raise ValueError("detection/data timestamp mismatch")
            # Query IDs increase within an episode: decode once, strictly forward.
            while decoded_query < query:
                ok, frame = capture.read()
                decoded_query += 1
                if not ok:
                    break
            if not ok:
                raise ValueError(f"cannot decode query frame: {uid}/{query}")
            camera_detections = row.get("detections", {}).get("umi")
            if not isinstance(camera_detections, list):
                raise ValueError("missing umi detections")
            if prior_time is not None and float(arrays["elapsed_s"][index]) - prior_time > .151:
                tracker = CausalGraphTracker(max_nodes=max_nodes)
            graph = tracker.update(camera_detections)
            prior_time = float(arrays["elapsed_s"][index])
            features, mask, boxes, track_ids, edges, bindings = graph
            images = np.zeros((max_nodes, 3, roi_size, roi_size), np.uint8)
            for slot in np.flatnonzero(mask):
                images[slot] = _crop(frame, boxes[slot], roi_size)
            for key, value in zip(outputs, (*graph, images, np.asarray(tracker.last_truncated))):
                outputs[key].append(value)
            output_indices.append(int(index))
        capture.release()
    order = np.argsort(np.asarray(output_indices))
    if not np.array_equal(np.asarray(output_indices)[order], np.arange(len(arrays["query_id"]))):
        raise ValueError("graph construction did not cover every source row exactly once")
    contract = dict(format=GRAPH_VERSION, graph_version=GRAPH_VERSION, roles=list(ROLES),
                    max_nodes=max_nodes, roi_size=roi_size, node_features=list(NODE_FEATURES),
                    edge_features=list(EDGE_FEATURES), tracking="causal_same_role_greedy_iou",
                    max_missing_observations=2, iou_threshold=.2,
                    bindings="unique currently-visible role node, else -1",
                    graph_truncated="true when valid detections exceeded max_nodes",
                    ambiguity="true per kept node when multiple valid detections share its role",
                    track_ids_semantic=False, labels_used_as_features=False,
                    folder_categories_used_as_features=False, future_interpolation=False,
                    causal=True, bbox_only_no_physical_contact_claim=True,
                    source_data_sha256=sha256(data), detections_sha256=sha256(detections),
                    provenance_sha256=sha256(provenance), raw_root=str(raw.resolve()),
                    provenance_format=provenance_value.get("format") if isinstance(provenance_value, dict) else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, attempt_uid=arrays["attempt_uid"], query_id=arrays["query_id"],
                            elapsed_s=arrays["elapsed_s"], contract_json=np.asarray(json.dumps(contract, ensure_ascii=False)),
                            **{key: np.stack(value)[order] for key, value in outputs.items()})
    return {"observations": len(arrays["query_id"]), "episodes": len(video_by_uid), **contract}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--roi-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.detections, args.provenance, args.raw, args.output,
                           args.max_nodes, args.roi_size), ensure_ascii=False, indent=2))
