#!/usr/bin/env python3
"""Extract offline YOLOE auxiliary cues from timestamped query images, not labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.semantic_cues import (
    FORMAT, YOLOECueDetector, feature_names, frame_cues, sha256, validate_prompts,
)


def extract(manifest: Path, output: Path, detector, cameras: list[str], *, confidence: float = .25,
            detections_output: Path | None = None):
    """Manifest JSONL paths are relative to its directory; missing images fail.

    Each row supplies current-query images with per-camera elapsed_s in the
    same attempt clock. Future/stale frames are rejected (1 ms tolerance).
    This relies on truthful capture metadata; it cannot verify scene content.
    """
    names = feature_names(cameras)
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    if detections_output is not None and detections_output.exists():
        raise ValueError('refusing to overwrite detection records')
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty image manifest")
    seen, prepared = set(), []
    for row in rows:
        uid, query, timestamp = row["attempt_uid"], row["query_id"], float(row["elapsed_s"])
        if (not isinstance(uid, str) or not uid or type(query) is not int or query < 0 or
                not np.isfinite(timestamp) or timestamp < 0):
            raise ValueError("invalid query identity/timestamp")
        key = (uid, query)
        if key in seen:
            raise ValueError("duplicate query identity")
        seen.add(key)
        images = []
        for camera in cameras:
            image = row["images"][camera]
            frame_time = float(image["elapsed_s"])
            if not np.isfinite(frame_time) or abs(frame_time - timestamp) > 1e-3:
                raise ValueError("images must be current query observations, not future/stale frames")
            path = (manifest.parent / image["path"]).resolve()
            if not path.is_file():
                raise ValueError(f"missing image: {path}")
            images.append(path)
        prepared.append((uid, query, timestamp, images))
    features, image_hashes = [], []
    detection_rows=[]
    for uid, query, timestamp, images in prepared:
        detections=[detector(path) for path in images]
        features.append(np.concatenate([frame_cues(items, confidence=confidence) for items in detections]))
        image_hashes.append([sha256(path) for path in images])
        if detections_output is not None:
            detection_rows.append(dict(attempt_uid=uid,query_id=query,elapsed_s=timestamp,
                                       detections=dict(zip(cameras,detections))))
        if len(features)%200==0:
            print(json.dumps(dict(extracted=len(features),total=len(prepared))),flush=True)
    contract = dict(format=FORMAT, cameras=cameras, feature_names=names,
                    detector=detector.provenance, confidence_threshold=confidence,
                    source_manifest_sha256=sha256(manifest),
                    event_labels_generated=False, raw_joint_state_used=False,
                    temporal_tracking=False, geometry="normalized_2d_boxes_per_camera",
                    missing_or_multiple_candidates="unknown_geometry_not_failure",
                    observation_time="current_query_only_1ms_capture_tolerance")
    if detections_output is not None:
        detections_output.parent.mkdir(parents=True,exist_ok=True)
        with detections_output.open('x') as stream:
            for row in detection_rows:
                stream.write(json.dumps(row)+'\n')
        contract['detections_sha256']=sha256(detections_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, semantic_features=np.stack(features),
                            attempt_uid=np.asarray([x[0] for x in prepared]),
                            query_id=np.asarray([x[1] for x in prepared], dtype=np.int64),
                            elapsed_s=np.asarray([x[2] for x in prepared], dtype=np.float64),
                            image_sha256=np.asarray(image_hashes),
                            contract_json=np.asarray(json.dumps(contract, ensure_ascii=False)))
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSON object/target/gripper prompt lists")
    parser.add_argument("--cameras", nargs="+", default=["head"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument('--detections-output',type=Path,help='optional exact boxes for visual audit')
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    validate_prompts(prompts)
    detector = YOLOECueDetector(args.weights, prompts, device=args.device,
                                imgsz=args.imgsz, confidence=args.confidence)
    contract = extract(args.manifest, args.output, detector, args.cameras, confidence=args.confidence,
                       detections_output=args.detections_output)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
