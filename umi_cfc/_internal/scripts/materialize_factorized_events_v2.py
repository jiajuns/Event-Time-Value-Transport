#!/usr/bin/env python3
"""Join reviewed video intervals and YOLO cues by identity and image hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.factorized_events import DATA_FORMAT, HEADS, SCHEMA, interval_labels, read_observations
from event_rl.semantic_cues import append_semantic_cues, sha256


def materialize(manifest: Path, sidecar: Path, output: Path, annotations: Path | None = None,
                image_size: int = 96, annotation_provenance: Path | None = None) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite dataset")
    if not 32 <= image_size <= 256:
        raise ValueError("image_size must be in [32,256]")
    rows = read_observations(manifest)
    labels = interval_labels(annotations, rows)
    provenance = None
    if annotation_provenance is not None:
        provenance = json.loads(annotation_provenance.read_text(encoding="utf-8"))
        if annotations is None or provenance.get("annotations_sha256") != sha256(annotations):
            raise ValueError("annotation provenance does not match annotation CSV")
    arrays = dict(attempt_uid=np.asarray([row["attempt_uid"] for row in rows]),
                  query_id=np.asarray([row["query_id"] for row in rows], dtype=np.int64),
                  elapsed_s=np.asarray([row["elapsed_s"] for row in rows], dtype=np.float64),
                  vision_features=np.zeros((len(rows), 1), dtype=np.float32))
    joined, cue_contract = append_semantic_cues(arrays, sidecar)
    if cue_contract["source_manifest_sha256"] != sha256(manifest):
        raise ValueError("YOLO sidecar was not extracted from this exact observation manifest")
    with np.load(sidecar, allow_pickle=False) as archive:
        hashes = archive["image_sha256"]
        keys = list(zip(archive["attempt_uid"].astype(str).tolist(), archive["query_id"].tolist()))
    cameras = cue_contract["cameras"]
    if hashes.shape != (len(rows), len(cameras)):
        raise ValueError("image hash shape mismatch")
    lookup = {key: index for index, key in enumerate(keys)}
    images, hash_splits = [], {}
    for row in rows:
        views = []
        for view, camera in enumerate(cameras):
            item = row["images"][camera]
            if not np.isfinite(item["elapsed_s"]) or abs(item["elapsed_s"] - row["elapsed_s"]) > .001:
                raise ValueError("image timestamp is not current observation")
            path = (manifest.parent / item["path"]).resolve()
            digest = sha256(path)
            if digest != hashes[lookup[(row["attempt_uid"], row["query_id"])]][view]:
                raise ValueError("image changed after YOLO extraction")
            if digest in hash_splits and hash_splits[digest] != row["split"]:
                raise ValueError("identical image leaked across splits")
            hash_splits[digest] = row["split"]
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot decode image: {path}")
            rgb = cv2.cvtColor(cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            views.append(np.moveaxis(rgb, -1, 0))
        images.append(np.stack(views))
    cue_contract = {key: value for key, value in cue_contract.items()
                    if key not in ("base_vision_dim", "fusion")}
    contract = dict(format=DATA_FORMAT, schema=SCHEMA, cameras=cameras, image_size=image_size,
                    semantic_cues=cue_contract, manifest_sha256=sha256(manifest),
                    annotations_sha256=sha256(annotations) if annotations else None,
                    labels_are_model_inputs=False, raw_joint_state_used=False,
                    observation_sampling="manifest_only_independent_of_labels")
    if provenance is not None:
        contract["annotation_provenance"] = provenance
        contract["annotation_provenance_sha256"] = sha256(annotation_provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, images_uint8=np.stack(images),
                            semantic_features=joined["vision_features"][:, 1:], labels=labels,
                            attempt_uid=arrays["attempt_uid"], query_id=arrays["query_id"],
                            elapsed_s=arrays["elapsed_s"], group_uid=np.asarray([row["group_uid"] for row in rows]),
                            split=np.asarray([row["split"] for row in rows]),
                            contract_json=np.asarray(json.dumps(contract, ensure_ascii=False)))
    return dict(contract, observations=len(rows),
                labeled_frames={head: int((labels[:, column] >= 0).sum()) for column, head in enumerate(HEADS)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-features", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, help="omit for unlabeled inference data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--annotation-provenance", type=Path, help="optional hash-bound annotation origin/quality receipt")
    args = parser.parse_args()
    print(json.dumps(materialize(args.manifest, args.semantic_features, args.output,
                                 args.annotations, args.image_size, args.annotation_provenance), ensure_ascii=False, indent=2))
